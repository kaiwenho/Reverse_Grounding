"""The command line front door.

This entry point exists because there wasn't one. `QueryService` is where a
question is supposed to arrive — rate limits, screening, topic scope, and the
fixed table of what a caller may be told all live there — but the only
runnable command started a layer below it, at `loop-controller`. Every run
this repo had ever done, benchmarks included, skipped intake.

So these tests are mostly about one thing: that going through the command
really does go through the service, and that the convenience flags cannot let
a caller do something an HTTP caller could not.

None of them reach a model or the graph. Every case here is decided by intake
before the loop is asked to do anything, which is itself worth knowing.
"""

from __future__ import annotations

import json

from query_intake import cli


CLINICAL = "What dose of dapsone should I take for my rash?"


def run(args, tmp_path):
    return cli.main([*args, "--out", str(tmp_path), "--quiet"])


def last_json(capsys):
    """The result document, which `main` prints as its final output."""
    out = capsys.readouterr().out
    start = out.index("{")
    return json.loads(out[start:])


# --- the command really is the service ------------------------------------

def test_a_clinical_question_is_refused_without_reaching_a_model(tmp_path, capsys):
    """The clinical screen is intake's, not the planner's. Running through
    `loop-controller` would never have consulted it."""
    code = run(["--question", CLINICAL], tmp_path)
    body = last_json(capsys)

    assert body["status"] == "refused"
    assert body["schema"].startswith("query-intake/")
    assert code == cli.EXIT_NOT_ANSWERED


def test_the_message_comes_from_the_fixed_table(tmp_path, capsys):
    """Never assembled from an internal reason or anything a model wrote."""
    run(["--question", CLINICAL], tmp_path)
    message = last_json(capsys)["message"]

    assert "clinician" in message.lower()
    # The caller is told what *is* in scope, rather than only being refused.
    assert "research question" in message.lower()


# --- --question is not a way round the door -------------------------------

def test_a_payload_with_an_unknown_field_is_rejected(tmp_path, capsys):
    path = tmp_path / "body.json"
    path.write_text(json.dumps({"question": "Which drugs treat psoriasis?",
                                "max_results": 5}))
    code = run(["--payload", str(path)], tmp_path)
    body = last_json(capsys)

    assert body["status"] == "invalid_request"
    assert "max_results" in body["message"]
    assert code == cli.EXIT_NOT_ANSWERED


def test_a_payload_cannot_claim_a_caller_id(tmp_path, capsys):
    """Identity is the server's. If it could arrive in the body a caller
    could spend someone else's quota, and the rate limiter would be
    decorative. `caller_id` is simply not a field, so it is rejected as
    unknown rather than quietly ignored."""
    path = tmp_path / "body.json"
    path.write_text(json.dumps({"question": "Which drugs treat psoriasis?",
                                "caller_id": "someone-else"}))
    run(["--payload", str(path)], tmp_path)
    body = last_json(capsys)

    assert body["status"] == "invalid_request"
    assert "caller_id" in body["message"]


def test_question_and_review_depth_go_through_the_same_validation(tmp_path, capsys):
    """`--question` builds a payload and hands it to the same door, so the
    convenience flag cannot accept a shape an HTTP caller could not send."""
    code = run(["--question", CLINICAL, "--review-depth", "deep"], tmp_path)
    body = last_json(capsys)

    assert body["review_depth"] == "deep"
    assert code == cli.EXIT_NOT_ANSWERED


def test_no_question_and_no_payload_is_an_error(tmp_path):
    assert run([], tmp_path) == cli.EXIT_NOT_ANSWERED


def test_an_unreadable_payload_does_not_traceback(tmp_path, capsys):
    assert run(["--payload", str(tmp_path / "missing.json")], tmp_path) == (
        cli.EXIT_NOT_ANSWERED
    )


# --- output ---------------------------------------------------------------

def test_the_result_is_also_written_when_asked(tmp_path, capsys):
    out = tmp_path / "result.json"
    run(["--question", CLINICAL, "--result", str(out)], tmp_path)

    written = json.loads(out.read_text())
    assert written == last_json(capsys)


def test_every_result_carries_a_request_id(tmp_path, capsys):
    """The handle on a run for anyone reading a log. It is generated per
    request and returned whatever the outcome, including a refusal."""
    run(["--question", CLINICAL], tmp_path)
    assert last_json(capsys)["request_id"]
