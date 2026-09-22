"""Rate limiting and log redaction.

Neither is clever. Both are the kind of thing that is missing until the day it
matters, and the tests here are mostly about the failure modes: a concurrency
slot leaked on an error path locks a caller out silently, and a redactor that
misses the field carrying the question has done nothing at all.
"""

from __future__ import annotations

import time

from query_intake.ratelimit import InMemoryRateLimiter, NoRateLimit, RateLimits
from query_intake.redact import (
    digest, question_for_log, redact_trace, summarise_for_log,
)


# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------


def limiter(**kwargs) -> InMemoryRateLimiter:
    defaults = dict(requests_per_window=3, window_s=60.0, max_concurrent=2)
    defaults.update(kwargs)
    return InMemoryRateLimiter(RateLimits(**defaults))


def test_requests_are_allowed_up_to_the_window_limit():
    rl = limiter()
    for _ in range(3):
        decision = rl.check("alice")
        assert decision.allowed
        rl.release("alice")
    assert rl.check("alice").allowed is False


def test_the_refusal_says_how_long_to_wait():
    rl = limiter(requests_per_window=1)
    rl.check("alice")
    rl.release("alice")
    decision = rl.check("alice")
    assert decision.allowed is False
    assert 0 < decision.retry_after_s <= 60.0


def test_callers_are_counted_separately():
    """Identity is the whole basis of the limit; sharing counters between
    callers would make it a global throttle instead."""
    rl = limiter(requests_per_window=1)
    assert rl.check("alice").allowed
    assert rl.check("bob").allowed


def test_concurrency_is_limited_independently_of_rate():
    """Ten requests an hour is a reasonable rate. Ten simultaneous multi-minute
    searches is not, and a per-window count does not catch it."""
    rl = limiter(requests_per_window=100, max_concurrent=2)
    rl.check("alice")
    rl.check("alice")
    blocked = rl.check("alice")
    assert blocked.allowed is False
    assert "already running" in blocked.reason

    rl.release("alice")
    assert rl.check("alice").allowed


def test_a_released_slot_comes_back():
    rl = limiter(requests_per_window=100, max_concurrent=1)
    rl.check("alice")
    rl.release("alice")
    assert rl.check("alice").allowed


def test_releasing_more_than_taken_does_not_go_negative():
    """A double release on an error path must not hand out a free slot."""
    rl = limiter(max_concurrent=1)
    rl.check("alice")
    rl.release("alice")
    rl.release("alice")
    rl.release("alice")
    assert rl.check("alice").allowed
    assert rl.check("alice").allowed is False


def test_the_window_slides():
    rl = limiter(requests_per_window=1, window_s=0.05)
    rl.check("alice")
    rl.release("alice")
    assert rl.check("alice").allowed is False
    time.sleep(0.06)
    assert rl.check("alice").allowed


def test_the_null_limiter_allows_everything():
    rl = NoRateLimit()
    for _ in range(50):
        assert rl.check("alice").allowed


# ---------------------------------------------------------------------------
# Redaction
# ---------------------------------------------------------------------------


def test_a_digest_is_stable_within_a_process_and_hides_the_text():
    question = "which drugs treat dermatitis herpetiformis?"
    assert digest(question) == digest(question)
    assert question not in digest(question)
    assert digest(question) != digest(question + "?")


def test_raw_logging_is_opt_in():
    question = "which drugs treat eczema?"
    assert question_for_log(question) != question
    assert question_for_log(question, raw=True) == question


def test_a_trace_is_redacted_in_every_place_the_question_appears():
    trace = {
        "question": "which drugs treat eczema?",
        "state": {"question": "which drugs treat eczema?", "iterations": 2},
        "planner_exchanges": [
            {"kind": "plan", "ok": True,
             "request": {"question": "which drugs treat eczema?"},
             "raw_response": "a long model reply quoting the question"},
        ],
        "decision_attempts": [
            {"iteration": 1, "accepted": True, "raw_response": "{...}"},
        ],
    }
    redacted = redact_trace(trace)

    blob = str(redacted)
    assert "which drugs treat eczema" not in blob
    assert redacted["question"].startswith("q:")
    assert redacted["state"]["question"].startswith("q:")
    assert redacted["state"]["iterations"] == 2       # non-question fields kept


def test_planner_prompts_are_dropped_rather_than_rewritten():
    """They quote the question inside a much longer prompt; rewriting would mean
    scanning prose for it, and this needs to be predictable."""
    trace = {"planner_exchanges": [
        {"kind": "relax_plan", "ok": True, "note": "revised",
         "request": {"question": "secret question"},
         "raw_response": "secret question appears here too"},
    ]}
    redacted = redact_trace(trace)
    exchange = redacted["planner_exchanges"][0]
    assert exchange == {"kind": "relax_plan", "ok": True, "note": "revised"}
    assert "planner_exchanges_note" in redacted


def test_decision_raw_responses_are_dropped():
    trace = {"decision_attempts": [
        {"iteration": 1, "problems": ["x"], "raw_response": "model text"},
    ]}
    attempt = redact_trace(trace)["decision_attempts"][0]
    assert "raw_response" not in attempt
    assert attempt["problems"] == ["x"]


def test_raw_mode_leaves_a_trace_untouched():
    trace = {"question": "which drugs treat eczema?"}
    assert redact_trace(trace, raw=True) == trace


def test_a_log_line_carries_what_an_operator_needs_and_not_the_question():
    record = summarise_for_log(
        request_id="req-1", caller_id="alice",
        question="which drugs treat eczema?",
        event="completed", iterations=2,
    )
    assert record["request_id"] == "req-1"
    assert record["caller_id"] == "alice"
    assert record["event"] == "completed"
    assert record["iterations"] == 2
    assert record["question_chars"] == len("which drugs treat eczema?")
    assert "eczema" not in record["question"]
