"""The model chooses, and the fence holds.

The point of these tests is that nothing the model does can put the loop
outside the policy layer's rules — not malformed JSON, not an invented move,
not a legal-looking move with illegal parameters, not being unreachable. In
every one of those cases the loop still gets a decision it can act on.
"""

from __future__ import annotations

import json

from loop_controller.contracts import (
    ABANDON, ACCEPT, Budget, LoopState, OUTCOME_NO_DATA, RELAX_PLAN,
    REPAIR_PLAN, REPORT_ABSENCE,
)
from loop_controller.decide import LLMDecisionMaker, PolicyDecisionMaker, build_user_message
from loop_controller.diagnose import diagnose
from loop_controller.policy import legal_actions

from helpers import (
    StubLLM, make_candidate, make_path, make_plan, make_resolution, make_result,
    relaxation_axes,
)


def empty_diag(plan=None):
    plan = plan or make_plan()
    result = make_result(
        outcome=OUTCOME_NO_DATA, verdict="no_answer", plan=plan,
        paths={"P1": make_path(
            verdict="no_answer", relaxations=relaxation_axes(), num_candidates=0,
        )},
    )
    return diagnose(result, None, plan)


def setup(diag=None):
    diag = diag or empty_diag()
    st = LoopState(question="which drugs treat it?", budget=Budget())
    return diag, st, legal_actions(diag, st)


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


def test_a_valid_decision_is_taken_as_given():
    diag, st, legal = setup()
    llm = StubLLM([json.dumps({
        "action": "relax_plan",
        "rationale": "the query completed and found nothing; the qualifier is "
                     "the tightest constraint",
        "relaxation_key": "P1:qualifiers",
        "expected_change": "edges without the qualifier should match",
    })])
    decision = LLMDecisionMaker(llm).decide(diag, st, legal)

    assert decision.action == RELAX_PLAN
    assert decision.source == "llm"
    assert decision.relaxation_key == "P1:qualifiers"
    assert decision.overridden_from is None


def test_the_model_may_disagree_with_the_deterministic_ordering():
    """Adaptivity is the reason the model is here; it is allowed to use it."""
    diag, st, legal = setup()
    llm = StubLLM([json.dumps({
        "action": "report_absence",
        "rationale": "the graph plainly does not model this relationship; "
                     "loosening the predicate would find unrelated edges",
    })])
    decision = LLMDecisionMaker(llm).decide(diag, st, legal)
    assert decision.action == REPORT_ABSENCE
    assert decision.source == "llm"


def test_fenced_json_is_parsed():
    diag, st, legal = setup()
    llm = StubLLM([
        '```json\n{"action": "report_absence", "rationale": "nothing there"}\n```'
    ])
    assert LLMDecisionMaker(llm).decide(diag, st, legal).action == REPORT_ABSENCE


# ---------------------------------------------------------------------------
# The fence
# ---------------------------------------------------------------------------


def test_an_invented_move_is_rejected_then_overridden():
    diag, st, legal = setup()
    llm = StubLLM([
        json.dumps({"action": "ask_the_user", "rationale": "I need more detail"}),
        json.dumps({"action": "ask_the_user", "rationale": "still need detail"}),
    ])
    decision = LLMDecisionMaker(llm).decide(diag, st, legal)

    assert decision.source == "policy"
    assert decision.overridden_from == "ask_the_user"
    assert decision.action in legal.allowed


def test_a_rejected_decision_gets_one_corrected_attempt():
    diag, st, legal = setup()
    llm = StubLLM([
        json.dumps({"action": "accept", "rationale": "good enough"}),
        json.dumps({
            "action": "relax_plan", "rationale": "corrected",
            "relaxation_key": "P1:predicate",
        }),
    ])
    maker = LLMDecisionMaker(llm)
    decision = maker.decide(diag, st, legal)

    assert decision.action == RELAX_PLAN
    assert decision.source == "llm"
    assert len(llm.calls) == 2
    assert "YOUR PREVIOUS DECISION WAS REJECTED" in llm.calls[1]["user"]
    assert "no results to accept" in llm.calls[1]["user"]


def test_relaxing_a_locked_anchor_is_overridden():
    """The model is shown the option is unavailable and asks for it anyway."""
    diag, st, legal = setup()
    payload = json.dumps({
        "action": "relax_plan",
        "rationale": "widening the disease category should find more",
        "relaxation_key": "P1:category:disease",
    })
    llm = StubLLM([payload, payload])
    decision = LLMDecisionMaker(llm).decide(diag, st, legal)

    assert decision.source == "policy"
    assert decision.overridden_from == "relax_plan"
    assert "pinned anchor" in (decision.override_reason or "")


def test_a_tightening_retry_is_overridden():
    result = make_result(
        outcome="truncated_or_timed_out", verdict="inconclusive",
        paths={"P1": make_path(direct_outcome="timeout", num_candidates=0)},
    )
    diag = diagnose(result, None, make_plan())
    st = LoopState(question="q", budget=Budget())
    st.execution_overrides = {"timeout": 300}
    legal = legal_actions(diag, st)

    payload = json.dumps({
        "action": "retry_execution", "rationale": "be quicker this time",
        "execution_overrides": {"timeout": 30},
    })
    decision = LLMDecisionMaker(StubLLM([payload, payload])).decide(diag, st, legal)
    assert decision.source == "policy"
    assert "do not loosen" in (decision.override_reason or "")


def test_unparseable_output_falls_back():
    diag, st, legal = setup()
    llm = StubLLM(["I think we should try again with a broader query.", "still prose"])
    decision = LLMDecisionMaker(llm).decide(diag, st, legal)
    assert decision.source == "policy"
    assert decision.action in legal.allowed


def test_an_unreachable_model_falls_back_immediately():
    """The loop degrades; it does not stall."""
    diag, st, legal = setup()
    llm = StubLLM([], raises=True)
    decision = LLMDecisionMaker(llm).decide(diag, st, legal)

    assert decision.source == "policy"
    assert "llm call failed" in (decision.override_reason or "")
    assert len(llm.calls) == 1  # no retry against a model that is not there


def test_no_call_is_made_when_only_one_move_exists():
    result = make_result(outcome="plan_refused", verdict="refused")
    diag = diagnose(result, None, make_plan())
    st = LoopState(question="q", budget=Budget())
    llm = StubLLM([])
    decision = LLMDecisionMaker(llm).decide(diag, st, legal_actions(diag, st))

    assert decision.action == ABANDON
    assert llm.calls == []


# ---------------------------------------------------------------------------
# The prompt
# ---------------------------------------------------------------------------


def test_the_prompt_shows_the_legal_moves_and_why_the_others_are_not():
    diag, st, legal = setup()
    message = build_user_message(diag, st, legal)

    assert "LEGAL MOVES" in message
    assert "MOVES THAT ARE NOT AVAILABLE, AND WHY" in message
    assert "there are no results to accept" in message

    options = message.split("RELAXATION OPTIONS")[1].split("\n\n")[0]
    assert "P1:qualifiers" in options
    assert "P1:category:candidate_drug" in options
    assert "P1:category:disease" not in options

    # Not offered, but explained: a model told only "choose from these" invents
    # a seventh option; one told why an option is missing argues with the
    # reason instead, which is a legible disagreement rather than a bad move.
    assert "pinned anchor" in message


def test_the_prompt_carries_the_grounded_menus():
    from helpers import make_resolution

    result = make_result(
        outcome="unresolved_grounding", replannable=True, verdict="unexecutable",
        resolution=make_resolution(resolved=False, considered=4),
    )
    diag = diagnose(result, None, make_plan())
    st = LoopState(question="q", budget=Budget())
    message = build_user_message(diag, st, legal_actions(diag, st))

    assert "RESOLUTION ALTERNATIVES" in message
    assert "MONDO:1" in message


def test_the_prompt_shows_what_was_already_tried():
    from loop_controller.contracts import Attempt, Decision

    diag, st, _ = setup()
    st.attempts.append(Attempt(
        index=1, plan_fingerprint="plan:abc", diagnosis=diag,
        decision=Decision(action=RELAX_PLAN, rationale="first try",
                          relaxation_key="P1:qualifiers"),
    ))
    st.spent_axes.add("P1:qualifiers")
    message = build_user_message(diag, st, legal_actions(diag, st))

    assert "WHAT HAS BEEN TRIED" in message
    assert "loosened P1:qualifiers" in message
    assert "axes already loosened" in message


# ---------------------------------------------------------------------------
# The deterministic maker
# ---------------------------------------------------------------------------


def test_the_policy_maker_needs_no_model():
    diag = diagnose(make_result(candidates=[make_candidate()]), None, make_plan())
    st = LoopState(question="q", budget=Budget())
    decision = PolicyDecisionMaker().decide(diag, st, legal_actions(diag, st))
    assert decision.action == ACCEPT
    assert decision.source == "policy"


def test_results_with_a_mismatched_anchor_are_accepted_not_repaired(requires_biolink):
    """The question 6 regression, at the layer where it was decided.

    `decide()` tries `repair_plan` before it ever looks at `accept`, which is
    right for a plan that could not run — one will not become runnable by
    being run again. It was wrong here: the plan ran, returned candidates, and
    the only complaint was that an anchor's category did not match what it
    resolved to. Repair-first spent the answer on another iteration and the
    run ended with nothing.

    The fix is in the diagnosis, not the ordering: a mismatch on a run that
    returned results no longer makes the plan repairable, so `repair_plan` is
    not legal and the ordering never gets the chance.
    """
    result = make_result(
        candidates=[make_candidate()],
        resolution=make_resolution(
            types=["biolink:PhenotypicFeature"], expected_category="Disease",
        ),
    )
    diag = diagnose(result, None, make_plan())
    st = LoopState(question="q", budget=Budget())
    legal = legal_actions(diag, st)

    assert REPAIR_PLAN not in legal
    assert ACCEPT in legal
    # The reader is still told. It is a caveat on the answer, not a reason to
    # discard it.
    assert any("does not match what it resolved to" in c for c in legal.cautions)

    decision = PolicyDecisionMaker().decide(diag, st, legal)
    assert decision.action == ACCEPT
