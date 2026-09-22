"""The loop, end to end, on recorded runs.

Every scenario here is a list of result documents replayed through a scripted
executor. That is enough to exercise the real diagnosis, the real policy layer,
the real decision maker, the real revision assembly and the real composer — the
only thing simulated is ARAX and the model.

The tests that matter most are the termination ones. A loop with an LLM
choosing its moves is only safe if it cannot repeat itself, cannot relax the
same constraint twice, cannot climb the execution ladder forever, and cannot
outlive its budget — and those four have to hold against a decision maker that
is actively trying to keep going.
"""

from __future__ import annotations

import json

from loop_controller.contracts import (
    ABANDON, Budget, Decision, LOOP_ABSENCE, LOOP_ANSWERED, LOOP_EXHAUSTED,
    LOOP_FAILED, LOOP_REFUSED, LoopState, OUTCOME_NO_DATA, RELAX_PLAN,
    REPAIR_PLAN, RETRY_EXECUTION,
)
from loop_controller.executor_cli import ExecutorUnavailable, ScriptedExecutor
from loop_controller.loop import ControllerConfig, LoopController
from loop_controller.planner_port import NoPlanner, ScriptedPlanner
from loop_controller.ports import PlanAttempt
from loop_controller.diagnose import diagnose
from loop_controller.revision import RevisionRequest, build_request

from helpers import (
    make_candidate, make_path, make_plan, make_resolution, make_result,
    plan_relaxation_axes,
)


def config(**kwargs) -> ControllerConfig:
    kwargs.setdefault("verbose", False)
    return ControllerConfig(**kwargs)


def results_run(n: int = 2):
    return make_result(candidates=[
        make_candidate(f"CHEBI:{i}", f"drug {i}", rank=i) for i in range(1, n + 1)
    ])


def timeout_run():
    return make_result(
        outcome="truncated_or_timed_out", verdict="inconclusive",
        detail="a pathfinder query did not finish",
        paths={"P1": make_path(
            verdict="inconclusive", direct_outcome="timeout",
            coverage_complete=False, num_candidates=0,
        )},
    )


def empty_run(plan=None):
    return make_result(
        outcome="no_graph_data", verdict="no_answer", plan=plan,
        paths={"P1": make_path(
            verdict="no_answer", relaxations=plan_relaxation_axes(),
            num_candidates=0,
        )},
        resolution=make_resolution(),
    )


def unresolved_run():
    return make_result(
        outcome="unresolved_grounding", replannable=True, verdict="unexecutable",
        detail="named entities did not resolve: ['disease']",
        resolution=make_resolution(resolved=False, considered=4),
        paths={},
    )


# ---------------------------------------------------------------------------
# The simple ending
# ---------------------------------------------------------------------------


def test_results_on_the_first_run_are_accepted():
    executor = ScriptedExecutor([results_run()])
    outcome = LoopController(executor, config=config()).run(plan=make_plan())

    assert outcome.status == LOOP_ANSWERED
    assert outcome.state.iteration == 1
    assert outcome.answer["grounding"]["ok"] is True
    assert len(executor.calls) == 1


# ---------------------------------------------------------------------------
# Retrying execution: same plan, no planner call
# ---------------------------------------------------------------------------


def test_a_timeout_is_retried_with_looser_settings_and_no_planner_call():
    executor = ScriptedExecutor([timeout_run(), results_run()])
    planner = ScriptedPlanner([])
    outcome = LoopController(executor, planner, config=config()).run(plan=make_plan())

    assert outcome.status == LOOP_ANSWERED
    assert outcome.state.iteration == 2
    assert planner.requests == []                      # the plan was never at fault
    assert executor.calls[0]["overrides"] == {}
    assert executor.calls[0]["plan"] == executor.calls[1]["plan"]
    assert executor.calls[1]["overrides"]["timeout"] == 120.0


def test_the_execution_ladder_terminates_rather_than_retrying_forever():
    """A decision maker that always wants to retry cannot make the loop hang."""

    class AlwaysRetry:
        attempts: list = []

        def decide(self, diag, state, legal):
            if RETRY_EXECUTION in legal:
                return Decision(
                    action=RETRY_EXECUTION, rationale="one more time",
                    execution_overrides=dict(legal.escalation or {}),
                    source="llm",
                )
            return Decision(action=ABANDON, rationale="nothing left", source="llm")

    executor = ScriptedExecutor([timeout_run()] * 12)
    outcome = LoopController(
        executor, decision_maker=AlwaysRetry(),
        config=config(budget=Budget(max_iterations=12)),
    ).run(plan=make_plan())

    assert outcome.status == LOOP_EXHAUSTED
    # Three ladder rungs, then the fourth iteration has nothing left to loosen.
    assert outcome.state.iteration == 4


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------


def test_an_empty_result_relaxes_one_axis_and_then_answers():
    revised = make_plan(predicate="biolink:affects", plan_id="P-2")
    executor = ScriptedExecutor([empty_run(), results_run()])
    planner = ScriptedPlanner([PlanAttempt(ok=True, plan=revised)])

    outcome = LoopController(executor, planner, config=config()).run(plan=make_plan())

    assert outcome.status == LOOP_ANSWERED
    assert outcome.state.iteration == 2
    assert outcome.state.spent_axes == {"P1:predicate"}

    request = planner.requests[0]
    assert isinstance(request, RevisionRequest)
    assert request.action == RELAX_PLAN
    assert request.relaxation.axis == "predicate"
    assert executor.calls[1]["plan"] == revised


def test_relaxation_is_monotone_across_iterations():
    """Each iteration spends a different axis; none is spent twice.

    The revisions are cumulative — each keeps the loosening from the one
    before and adds the axis it was just asked for — because that is what a
    planner told to change one thing and leave the rest alone produces, and
    because the loop now checks it. A second revision that re-loosened the
    predicate instead of the category would be rejected as over-broad.
    """
    plans = [
        make_plan(predicate="biolink:pred2", plan_id="P-2"),
        make_plan(predicate="biolink:pred2", answer_category="NamedThing",
                  plan_id="P-3"),
    ]
    executor = ScriptedExecutor([empty_run()] * 4)
    planner = ScriptedPlanner([PlanAttempt(ok=True, plan=p) for p in plans])

    outcome = LoopController(
        executor, planner, config=config(budget=Budget(max_iterations=4)),
    ).run(plan=make_plan())

    assert outcome.status in (LOOP_ABSENCE, LOOP_EXHAUSTED, LOOP_ANSWERED)
    spent = outcome.state.spent_axes

    # One axis per relaxing iteration, each one different, tightest first, and
    # never the pinned anchor.
    relaxing = [
        a for a in outcome.state.attempts
        if a.decision and a.decision.action == RELAX_PLAN
    ]
    assert len(spent) == len(relaxing)
    assert [a.decision.relaxation_key for a in relaxing] == sorted(
        spent, key=["P1:predicate", "P1:category:candidate_drug"].index,
    )
    assert "P1:category:disease" not in spent


def test_the_revision_request_carries_the_locked_constraints():
    plan = make_plan(evidence_policy={
        "origin": "user_requested", "application": "filter",
        "rationale": "the user asked for published evidence",
        "min_publications": 1,
    })
    executor = ScriptedExecutor([empty_run(plan=plan), results_run()])
    planner = ScriptedPlanner([
        PlanAttempt(ok=True, plan=make_plan(predicate="biolink:affects")),
    ])

    LoopController(executor, planner, config=config()).run(plan=plan)

    request = planner.requests[0]
    assert "evidence_policy:min_publications" in request.must_not_change
    assert "anchor:disease" in request.must_not_change
    assert request.forbidden_fingerprints

    from loop_controller.revision import build_revision_message
    message = build_revision_message(request)
    assert "never weakened" in message
    assert "Change nothing else" in message


def test_repairing_an_anchors_category_does_not_also_lock_it(requires_biolink):
    """The request must not tell the planner to fix a field and keep it.

    Question 6 of the final run went out with `repair_focus: ['anchor:
    target_tnf']` beside `must_not_change: ['anchor:target_tnf']` — the same
    anchor in both lists — because the lock was rendered as "it keeps its
    surface name *and its category*", and the category was the thing being
    repaired. The planner resolved the contradiction the only way open to it,
    by widening to `GeneOrGeneProduct`, and produced a hop ARAX cannot answer.

    The concept stays locked. The one field under repair does not.
    """
    from loop_controller.revision import build_request, build_revision_message

    plan = make_plan()
    result = make_result(
        outcome=OUTCOME_NO_DATA, verdict="no_answer", plan=plan,
        resolution=make_resolution(
            types=["biolink:PhenotypicFeature"], expected_category="Disease",
        ),
    )
    diag = diagnose(result, None, plan)
    assert diag.anchor_category_mismatch          # the precondition

    request = build_request(
        question="which drugs treat it?", prior_plan=plan,
        prior_fingerprint="plan:1", action=REPAIR_PLAN, diagnosis=diag,
        repair_focus=["anchor_category:disease"],
    )

    # The lock is still there — the concept may not be swapped.
    assert "anchor:disease" in request.must_not_change
    # But the request now says which part of it is open.
    assert request.category_open == ["disease"]

    message = build_revision_message(request)
    assert "biolink_category is the field being repaired" in message
    assert "keeps its surface name and its category" not in message


def test_an_anchor_with_no_mismatch_keeps_its_category_locked():
    """The narrowing applies only to the anchor actually being repaired."""
    from loop_controller.revision import build_request, build_revision_message

    plan = make_plan()
    result = make_result(
        outcome=OUTCOME_NO_DATA, verdict="no_answer", plan=plan,
        resolution=make_resolution(),
    )
    diag = diagnose(result, None, plan)

    request = build_request(
        question="which drugs treat it?", prior_plan=plan,
        prior_fingerprint="plan:1", action=REPAIR_PLAN, diagnosis=diag,
    )

    assert request.category_open == []
    assert "keeps its surface name and its category" in build_revision_message(request)


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------


def test_an_unresolved_anchor_is_repaired_with_the_resolver_menu():
    repaired = make_plan(anchor_name="dermatitis herpetiformis (Duhring disease)")
    executor = ScriptedExecutor([unresolved_run(), results_run()])
    planner = ScriptedPlanner([PlanAttempt(ok=True, plan=repaired)])

    outcome = LoopController(executor, planner, config=config()).run(plan=make_plan())

    assert outcome.status == LOOP_ANSWERED
    request = planner.requests[0]
    assert request.action == REPAIR_PLAN
    assert request.diagnosis.unresolved_entities == ["disease"]

    from loop_controller.revision import build_revision_message
    message = build_revision_message(request)
    assert "did not resolve" in message
    assert "MONDO:1" in message           # the alternatives the resolver saw


# ---------------------------------------------------------------------------
# No repetition
# ---------------------------------------------------------------------------


def test_a_revision_that_would_query_the_same_thing_is_not_executed():
    """The failure this guard exists to prevent: four iterations, one query."""
    original = make_plan()
    reworded = make_plan(restated="A different way of saying the same thing.")

    executor = ScriptedExecutor([empty_run(), results_run()])
    planner = ScriptedPlanner([
        PlanAttempt(ok=True, plan=reworded),   # same query, new prose
        PlanAttempt(ok=True, plan=reworded),   # and again
    ])

    outcome = LoopController(executor, planner, config=config()).run(plan=original)

    assert len(executor.calls) == 1            # the duplicate never ran
    assert outcome.status == LOOP_EXHAUSTED
    assert "differs in what it would query" in outcome.reason


def test_the_planner_is_told_specifically_why_the_duplicate_was_rejected():
    original = make_plan()
    reworded = make_plan(restated="Reworded, identical query.")
    revised = make_plan(predicate="biolink:affects", plan_id="P-3")

    executor = ScriptedExecutor([empty_run(), results_run()])
    planner = ScriptedPlanner([
        PlanAttempt(ok=True, plan=reworded),
        PlanAttempt(ok=True, plan=revised),
    ])

    outcome = LoopController(executor, planner, config=config()).run(plan=original)

    assert outcome.status == LOOP_ANSWERED
    second = planner.requests[1]
    assert second.previous_rejection is not None
    assert "same queries as one already executed" in second.previous_rejection

    from loop_controller.revision import build_revision_message
    assert "REJECTED WITHOUT BEING EXECUTED" in build_revision_message(second)


# ---------------------------------------------------------------------------
# Budgets
# ---------------------------------------------------------------------------


def test_the_iteration_budget_stops_the_loop():
    executor = ScriptedExecutor([empty_run()] * 10)
    planner = ScriptedPlanner([
        PlanAttempt(ok=True, plan=make_plan(predicate=f"biolink:p{i}", plan_id=f"P-{i}"))
        for i in range(2, 12)
    ])
    outcome = LoopController(
        executor, planner, config=config(budget=Budget(max_iterations=2)),
    ).run(plan=make_plan())

    assert outcome.state.iteration == 2
    assert len(executor.calls) == 2


def test_no_planner_call_is_wasted_on_the_final_iteration():
    """On the last permitted iteration a new plan could not be run, so it is
    not asked for."""
    executor = ScriptedExecutor([empty_run()])
    planner = ScriptedPlanner([
        PlanAttempt(ok=True, plan=make_plan(predicate="biolink:affects")),
    ])
    outcome = LoopController(
        executor, planner, config=config(budget=Budget(max_iterations=1)),
    ).run(plan=make_plan())

    assert planner.requests == []
    assert outcome.status == LOOP_ABSENCE


def test_the_planner_call_budget_is_enforced():
    executor = ScriptedExecutor([empty_run()] * 6)
    planner = ScriptedPlanner([
        PlanAttempt(ok=True, plan=make_plan(predicate=f"biolink:p{i}", plan_id=f"P-{i}"))
        for i in range(2, 8)
    ])
    outcome = LoopController(
        executor, planner,
        config=config(budget=Budget(max_iterations=6, max_planner_calls=2)),
    ).run(plan=make_plan())

    assert outcome.state.planner_calls <= 2
    assert outcome.status in (LOOP_ABSENCE, LOOP_EXHAUSTED)


def test_cost_is_accumulated_from_the_executor_ledger():
    executor = ScriptedExecutor([timeout_run(), results_run()])
    outcome = LoopController(executor, config=config()).run(plan=make_plan())
    assert outcome.state.arax_calls == 6      # 3 per run, from the ledger
    assert outcome.state.llm_calls == 10


# ---------------------------------------------------------------------------
# Endings that are not answers
# ---------------------------------------------------------------------------


def test_an_established_absence_is_reported_as_a_finding():
    executor = ScriptedExecutor([
        make_result(
            outcome="no_graph_data", verdict="no_answer",
            paths={"P1": make_path(num_candidates=0)},   # nothing left to relax
            resolution=make_resolution(),
        )
    ])
    outcome = LoopController(executor, config=config()).run(plan=make_plan())

    assert outcome.status == LOOP_ABSENCE
    assert outcome.answer["answer_kind"] == "absence"
    assert outcome.answer["grounding"]["ok"] is True


def test_a_planner_refusal_ends_the_loop_before_anything_runs():
    executor = ScriptedExecutor([results_run()])
    planner = ScriptedPlanner([PlanAttempt(
        ok=False, refused=True, refusal_reason="unsafe_or_clinical_advice",
        plan={"refusal": {
            "reason": "unsafe_or_clinical_advice",
            "message": "I cannot advise on your particular medication choice.",
        }},
    )])
    outcome = LoopController(executor, planner, config=config()).run(
        "what should I take for my rash?",
    )

    assert outcome.status == LOOP_REFUSED
    assert executor.calls == []
    text = json.dumps(outcome.answer)
    assert "not a source of clinical guidance" in text
    assert "your particular medication choice" not in text


def test_an_unavailable_executor_stops_rather_than_iterating():
    class Unavailable:
        def run(self, plan, *, overrides=None, tag=""):
            raise ExecutorUnavailable("the LLM is not reachable")

    outcome = LoopController(Unavailable(), config=config()).run(plan=make_plan())
    assert outcome.status == LOOP_FAILED
    assert "not reachable" in outcome.reason


def test_results_in_hand_survive_a_failed_revision():
    """A revision that could not be written does not un-retrieve the results."""
    result_with_warning = make_result(
        candidates=[make_candidate()],
        concept_warning="1 concept check(s) did not confirm the intended entity",
        outcome="unresolved_grounding", replannable=True,
    )
    executor = ScriptedExecutor([result_with_warning])
    planner = ScriptedPlanner([PlanAttempt(ok=False, errors=["it did not validate"])])

    outcome = LoopController(executor, planner, config=config()).run(plan=make_plan())

    assert outcome.status == LOOP_ANSWERED
    assert outcome.answer["candidates"]
    assert "did not validate" in outcome.reason
    assert any("different concept" in c for c in outcome.answer["caveats"])


# ---------------------------------------------------------------------------
# Running without a planner
# ---------------------------------------------------------------------------


def test_a_hand_written_plan_runs_without_a_planner():
    executor = ScriptedExecutor([empty_run()])
    outcome = LoopController(executor, NoPlanner(), config=config()).run(plan=make_plan())

    assert outcome.status == LOOP_ABSENCE
    legality = outcome.state.attempts[0]
    assert legality.decision.action == "report_absence"


def test_a_question_without_a_planner_is_an_honest_failure():
    outcome = LoopController(
        ScriptedExecutor([results_run()]), config=config(),
    ).run("which drugs treat it?")
    assert outcome.status == LOOP_FAILED
    assert "no planner is configured" in outcome.reason


# ---------------------------------------------------------------------------
# The trace
# ---------------------------------------------------------------------------


def test_the_trace_records_the_counterfactual(tmp_path):
    class AlwaysAbsence:
        attempts: list = []

        def decide(self, diag, state, legal):
            from loop_controller.contracts import REPORT_ABSENCE
            return Decision(
                action=REPORT_ABSENCE, rationale="the graph does not model this",
                source="llm",
            )

    executor = ScriptedExecutor([empty_run()])
    planner = ScriptedPlanner([])
    trace_path = tmp_path / "trace.json"
    LoopController(
        executor, planner, decision_maker=AlwaysAbsence(),
        config=config(trace_path=str(trace_path)),
    ).run(plan=make_plan())

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    counterfactual = trace["counterfactual"]["per_iteration"][0]
    assert counterfactual["chosen"] == "report_absence"
    assert counterfactual["policy_would_choose"] == "relax_plan"
    assert counterfactual["agreed"] is False
    assert trace["counterfactual"]["agreement_rate"] == 0.0


def test_the_trace_records_the_legal_moves_each_iteration(tmp_path):
    trace_path = tmp_path / "trace.json"
    LoopController(
        ScriptedExecutor([results_run()]),
        config=config(trace_path=str(trace_path)),
    ).run(plan=make_plan())

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    assert trace["legality"][0]["allowed"]
    assert "refusals" in trace["legality"][0]


def test_the_answer_is_written_when_a_path_is_given(tmp_path):
    answer_path = tmp_path / "answer.json"
    LoopController(
        ScriptedExecutor([results_run()]),
        config=config(answer_path=str(answer_path)),
    ).run(plan=make_plan())

    answer = json.loads(answer_path.read_text(encoding="utf-8"))
    assert answer["schema"].startswith("loop-controller-answer/")
    assert answer["grounding"]["ok"] is True


# ---------------------------------------------------------------------------
# The cheap probe
# ---------------------------------------------------------------------------


def invalid_plan_run():
    """What `--check` writes when a plan does not validate."""
    return make_result(
        outcome="invalid_plan", replannable=True, verdict="unexecutable",
        detail="the plan did not validate",
        paths={},
        plan_assessment={"blocked_paths": {}, "issues": [{
            "severity": "error", "code": "schema_invalid", "scope": "plan",
            "target": None,
            "message": "[semantic] paths/0/hops/0/predicate: "
                       "'biolink:nope' is not a valid active Biolink predicate",
        }]},
    )


def test_a_broken_plan_is_caught_without_running_a_search():
    """The whole point: seconds instead of minutes, and no graph query."""
    repaired = make_plan(predicate="biolink:affects")
    executor = ScriptedExecutor([results_run()], check_result=invalid_plan_run())
    planner = ScriptedPlanner([PlanAttempt(ok=True, plan=repaired)])

    outcome = LoopController(executor, planner, config=config()).run(plan=make_plan())

    assert outcome.status == LOOP_ANSWERED
    assert len(executor.checks) == 1          # probed
    assert len(executor.calls) == 1           # searched once, on the repair
    assert executor.calls[0]["plan"] == repaired
    assert outcome.state.attempts[0].decision.action == REPAIR_PLAN


def test_a_clean_probe_falls_through_to_the_real_search():
    """A probe that finds nothing means 'nothing obviously wrong', not 'this
    will work' — the backend supporting a hop shape says nothing about whether
    it holds data for these entities."""
    executor = ScriptedExecutor([results_run()])   # check_result=None -> clean
    outcome = LoopController(executor, config=config()).run(plan=make_plan())

    assert outcome.status == LOOP_ANSWERED
    assert len(executor.checks) == 1
    assert len(executor.calls) == 1


def test_the_probe_runs_only_on_the_first_iteration():
    """Later iterations follow a repair, which already produced a validated
    plan, or a retry, where the plan passed."""
    executor = ScriptedExecutor([empty_run(), results_run()])
    planner = ScriptedPlanner([
        PlanAttempt(ok=True, plan=make_plan(predicate="biolink:affects")),
    ])
    LoopController(executor, planner, config=config()).run(plan=make_plan())

    assert len(executor.checks) == 1
    assert len(executor.calls) == 2


def test_the_probe_is_skipped_on_a_retry():
    executor = ScriptedExecutor([timeout_run(), results_run()])
    LoopController(executor, config=config()).run(plan=make_plan())

    assert len(executor.checks) == 1          # first iteration only
    assert executor.calls[1]["overrides"]["timeout"] == 120.0


def test_a_probe_fault_that_is_not_repairable_does_not_stop_the_search():
    """The real run is the authority on whether the backend can be reached."""
    backend_down = make_result(
        outcome="backend_failure", verdict="error", detail="connection refused",
        paths={},
    )
    executor = ScriptedExecutor([results_run()], check_result=backend_down)
    outcome = LoopController(executor, config=config()).run(plan=make_plan())

    assert outcome.status == LOOP_ANSWERED
    assert len(executor.calls) == 1


def test_an_executor_without_a_probe_still_works():
    """`check` is optional on the port, so an adapter without it is fine."""

    class MinimalExecutor:
        def __init__(self):
            self.calls = []

        def run(self, plan, *, overrides=None, tag=""):
            self.calls.append(tag)
            from loop_controller.ports import ExecutionRun
            return ExecutionRun(result=results_run(), exit_code=0)

    executor = MinimalExecutor()
    outcome = LoopController(executor, config=config()).run(plan=make_plan())
    assert outcome.status == LOOP_ANSWERED
    assert len(executor.calls) == 1


# ---------------------------------------------------------------------------
# Model-call accounting
# ---------------------------------------------------------------------------


def test_llm_calls_is_the_sum_of_every_component():
    """It used to count only the executor, which made the ceiling a limit on
    the executor's spend rather than on the loop's."""
    state = LoopState(question="q", budget=Budget())
    state.executor_llm_calls = 5
    state.controller_llm_calls = 3
    state.planner_llm_calls = 2
    assert state.llm_calls == 10


def test_the_planners_calls_are_counted():
    executor = ScriptedExecutor([empty_run(), results_run()])
    planner = ScriptedPlanner([
        # attempts=2: the planner's first draft failed validation and it
        # repaired, which is two model calls, not one.
        PlanAttempt(ok=True, plan=make_plan(predicate="biolink:affects"),
                    attempts=2),
    ])
    outcome = LoopController(executor, planner, config=config()).run(
        plan=make_plan(),
    )
    # One revision at two attempts. The opening plan was supplied, not planned.
    assert outcome.state.planner_llm_calls == 2
    assert outcome.state.planner_calls == 1


def test_the_controllers_own_decision_calls_are_counted():
    from loop_controller.decide import LLMDecisionMaker

    class StubLLM:
        def __init__(self, replies):
            self.replies = list(replies)

        def complete(self, system, user):
            return self.replies.pop(0)

    # First reply is illegal and rejected, second is accepted: two calls.
    llm = StubLLM([
        json.dumps({"action": "relax_plan", "rationale": "x"}),      # no key
        json.dumps({"action": "accept", "rationale": "good enough"}),
    ])
    outcome = LoopController(
        ScriptedExecutor([results_run()]),
        decision_maker=LLMDecisionMaker(llm),
        config=config(),
    ).run(plan=make_plan())

    assert outcome.state.controller_llm_calls == 2
    assert outcome.state.llm_calls == 2 + outcome.state.executor_llm_calls


def test_the_deterministic_maker_costs_nothing():
    outcome = LoopController(
        ScriptedExecutor([results_run()]), config=config(),
    ).run(plan=make_plan())
    assert outcome.state.controller_llm_calls == 0


def test_the_budget_trips_on_the_total_not_the_executors_share():
    """A loop spending its ceiling on planner revisions must stop, even when
    the executor has barely run."""
    executor = ScriptedExecutor([empty_run()] * 5)
    planner = ScriptedPlanner([
        PlanAttempt(ok=True, plan=make_plan(predicate=f"biolink:p{i}"),
                    attempts=2)
        for i in range(2, 8)
    ])
    outcome = LoopController(
        executor, planner,
        config=config(budget=Budget(max_iterations=5, max_planner_calls=5,
                                    max_llm_calls=6)),
    ).run(plan=make_plan())

    assert outcome.state.llm_calls >= 6
    assert outcome.state.iteration < 5        # stopped before the iteration cap


def test_the_trace_breaks_the_cost_down_by_component(tmp_path):
    trace_path = tmp_path / "trace.json"
    LoopController(
        ScriptedExecutor([results_run()]),
        config=config(trace_path=str(trace_path)),
    ).run(plan=make_plan())

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    breakdown = trace["state"]["llm_calls_by_component"]
    assert set(breakdown) == {"executor", "controller", "planner"}
    assert sum(breakdown.values()) == trace["state"]["llm_calls"]


# ---------------------------------------------------------------------------
# One-axis relaxation, enforced
# ---------------------------------------------------------------------------
#
# The loop asks the planner to loosen one named constraint and change nothing
# else, so that results arriving afterwards can be attributed to that
# constraint. Until the structural diff existed, that was a sentence in a
# prompt with nothing behind it.


def test_an_over_broad_relaxation_is_rejected_once_and_corrected():
    """Told which field it also changed, the planner gets it right second time."""
    over_broad = make_plan(predicate="biolink:affects",
                           answer_category="NamedThing", plan_id="P-2")
    on_axis = make_plan(predicate="biolink:affects", plan_id="P-3")

    executor = ScriptedExecutor([empty_run(), results_run()])
    planner = ScriptedPlanner([
        PlanAttempt(ok=True, plan=over_broad),
        PlanAttempt(ok=True, plan=on_axis),
    ])

    outcome = LoopController(executor, planner, config=config()).run(plan=make_plan())

    assert outcome.status == LOOP_ANSWERED
    # The plan that ran is the corrected one; the over-broad one never reached
    # the executor.
    assert executor.calls[1]["plan"] == on_axis
    # Nothing to withdraw: the planner complied.
    assert outcome.state.over_relaxations == []

    # The second request named the offending field rather than complaining in
    # general terms.
    rejection = planner.requests[1].previous_rejection
    assert "entities[candidate_drug].biolink_category" in rejection


def test_a_planner_that_stays_over_broad_has_its_plan_used_and_recorded():
    """Discarding a valid, new plan would trade an answer for none. Keeping it
    silently would keep a claim that is no longer true. So it is used and the
    claim is withdrawn."""
    first = make_plan(predicate="biolink:affects",
                      answer_category="NamedThing", plan_id="P-2")
    second = make_plan(predicate="biolink:related_to",
                       answer_category="NamedThing", plan_id="P-3")

    executor = ScriptedExecutor([empty_run(), results_run()])
    planner = ScriptedPlanner([
        PlanAttempt(ok=True, plan=first),
        PlanAttempt(ok=True, plan=second),
    ])

    outcome = LoopController(executor, planner, config=config()).run(plan=make_plan())

    assert outcome.status == LOOP_ANSWERED
    assert executor.calls[1]["plan"] == second

    assert len(outcome.state.over_relaxations) == 1
    record = outcome.state.over_relaxations[0]
    assert record["axis"] == "P1:predicate"
    assert "entities[candidate_drug].biolink_category" in record["diff"]["off_axis"]

    # The reader is told, in the answer itself, that the attribution is gone.
    caveats = " ".join(outcome.answer["caveats"])
    assert "cannot be attributed to any single constraint" in caveats


def test_a_compliant_relaxation_adds_no_caveat():
    """The withdrawal appears only when it is owed."""
    executor = ScriptedExecutor([empty_run(), results_run()])
    planner = ScriptedPlanner([
        PlanAttempt(ok=True, plan=make_plan(predicate="biolink:affects")),
    ])

    outcome = LoopController(executor, planner, config=config()).run(plan=make_plan())

    caveats = " ".join(outcome.answer["caveats"])
    assert "cannot be attributed" not in caveats


def test_a_repair_is_not_held_to_the_one_axis_rule():
    """Repair fixes a broken plan and may touch whatever is broken; only
    relaxation carries the attribution claim."""
    repaired = make_plan(predicate="biolink:affects",
                         answer_category="NamedThing", anchor_name="coeliac disease")
    executor = ScriptedExecutor([unresolved_run(), results_run()])
    planner = ScriptedPlanner([PlanAttempt(ok=True, plan=repaired)])

    outcome = LoopController(executor, planner, config=config()).run(plan=make_plan())

    assert outcome.status == LOOP_ANSWERED
    assert outcome.state.over_relaxations == []


# ---------------------------------------------------------------------------
# A wrong anchor is not an absence
# ---------------------------------------------------------------------------


def test_a_mismatched_anchor_does_not_end_the_loop_in_an_absence(requires_biolink):
    """End to end: a completed empty run that would have been reported as a
    finding about the graph is not, because the query could not have matched."""
    mismatched = make_result(
        outcome="no_graph_data", verdict="no_answer",
        paths={"P1": make_path(verdict="no_answer", num_candidates=0)},
        resolution=make_resolution(types=["biolink:PhenotypicFeature"]),
    )
    executor = ScriptedExecutor([mismatched])
    outcome = LoopController(executor, config=config(
        budget=Budget(max_iterations=1),
    )).run(plan=make_plan())

    assert outcome.status != LOOP_ABSENCE
    assert outcome.answer["answer_kind"] == "inconclusive"

    # The reason is in the answer body, not only in the caveats, and it passes
    # the grounding gate rather than being withheld by it.
    assert outcome.answer["grounding"]["ok"] is True
    body = " ".join(s["text"] for s in outcome.answer["statements"])
    assert "not a finding about the data" in body

    caveats = " ".join(outcome.answer["caveats"])
    assert "not recorded under that category" in caveats


# ---------------------------------------------------------------------------
# The question that got smaller
# ---------------------------------------------------------------------------


def _q9_plan(with_disease_hop: bool = False):
    """The live Q9 shape: three concepts declared, one hop, disease unused."""
    plan = make_plan(
        question="Which drugs treat dermatitis herpetiformis by targeting CFTR?",
        predicate="biolink:affects",
    )
    plan["entities"] = [
        {"entity_ref": "cftr_gene", "name": "CFTR",
         "biolink_category": "Gene", "is_variable": False},
        {"entity_ref": "derm_herp", "name": "dermatitis herpetiformis",
         "biolink_category": "Disease", "is_variable": False},
        {"entity_ref": "candidate_drug", "name": "any drug",
         "biolink_category": "ChemicalEntity", "is_variable": True},
    ]
    plan["paths"] = [{
        "path_id": "P1", "return_entity_ref": "candidate_drug",
        "hops": [{"subject_ref": "candidate_drug",
                  "predicate": "biolink:affects", "object_ref": "cftr_gene"}],
    }]
    if with_disease_hop:
        plan["plan_id"] = "P-fixed"
        plan["paths"][0]["hops"].append({
            "subject_ref": "cftr_gene", "predicate": "biolink:contributes_to",
            "object_ref": "derm_herp",
        })
    return plan


def test_a_plan_that_dropped_a_concept_is_repaired_before_it_is_served():
    """End to end on the live failure. Twenty grounded results for a narrower
    question are not served while the plan can still be fixed."""
    executor = ScriptedExecutor([results_run(20), results_run(3)])
    planner = ScriptedPlanner([PlanAttempt(ok=True, plan=_q9_plan(True))])

    outcome = LoopController(executor, planner, config=config()).run(
        plan=_q9_plan(),
    )

    assert outcome.status == LOOP_ANSWERED
    assert [a.decision.action for a in outcome.state.attempts] == [
        "repair_plan", "accept",
    ]
    # The repaired plan is the one whose results were served.
    assert executor.calls[1]["plan"]["plan_id"] == "P-fixed"
    # And the finished answer carries no missing-concept caveat, because the
    # plan that produced it was not missing one.
    assert not any("does not account for" in c for c in outcome.answer["caveats"])


def test_a_dropped_concept_that_cannot_be_repaired_is_served_with_the_caveat():
    """Repair exhausted. Serving nothing would be worse; serving it silently
    would be worse still, because nothing else in the answer shows it."""
    executor = ScriptedExecutor([results_run(20)])
    planner = ScriptedPlanner([
        PlanAttempt(ok=False, errors=["the planner could not fix it"]),
    ])

    outcome = LoopController(executor, planner, config=config()).run(
        plan=_q9_plan(),
    )

    assert outcome.status == LOOP_ANSWERED
    assert outcome.answer["candidates"]

    caveats = " ".join(outcome.answer["caveats"])
    assert "does not account for dermatitis herpetiformis" in caveats
    assert "narrower question" in caveats

    # The results themselves are still perfectly grounded — which is exactly
    # why nothing but the caveat could have told the reader.
    assert outcome.answer["grounding"]["ok"] is True


def test_the_planner_is_told_which_entity_it_left_unused():
    executor = ScriptedExecutor([results_run(20), results_run(3)])
    planner = ScriptedPlanner([PlanAttempt(ok=True, plan=_q9_plan(True))])

    LoopController(executor, planner, config=config()).run(plan=_q9_plan())

    message = planner.requests[0].to_dict()
    orphans = message["diagnosis"]["orphan_entities"]
    assert [o["entity_ref"] for o in orphans] == ["derm_herp"]
    assert "unused_entity:derm_herp" in message["repair_focus"]


# ---------------------------------------------------------------------------
# A concept the graph does not know
# ---------------------------------------------------------------------------


def _unresolved_with_candidates():
    return make_result(
        outcome="unresolved_grounding", replannable=True, verdict="unexecutable",
        detail="named entities did not resolve: ['disease_pain']",
        paths={},
        resolution={"disease_pain": {
            "entity_ref": "disease_pain", "query": "neuropathic pain",
            "expected_category": "Disease", "is_variable": False,
            "resolved_curies": [], "confidence": "high",
            "reason": "LLM rejected all 20 candidates: all are subtypes.",
            "considered": [
                {"curie": "MONDO:0021667", "label": "neuralgia",
                 "types": ["biolink:Disease"], "rank": 1},
                {"curie": "UMLS:C1963916",
                 "label": "Diabetic peripheral neuropathic pain",
                 "types": ["biolink:Disease"], "rank": 2},
            ],
        }},
    )


def test_an_unresolved_anchor_ends_as_a_refusal_that_asks_a_question():
    """The live Q8 ending. It used to report `exhausted` and compose "no answer
    was established", while the twenty entries the resolver had retrieved sat
    unread in the result document."""
    executor = ScriptedExecutor([_unresolved_with_candidates()])
    planner = ScriptedPlanner([
        PlanAttempt(ok=False, refused=True, refusal_reason="needs_clarification"),
    ])

    outcome = LoopController(executor, planner, config=config()).run(
        plan=make_plan(),
    )

    assert outcome.status == LOOP_REFUSED
    assert "did not match any entry" in outcome.reason
    assert outcome.answer["answer_kind"] == "clarification_needed"

    text = json.dumps(outcome.answer)
    assert "neuralgia" in text and "MONDO:0021667" in text
    assert outcome.answer["grounding"]["ok"] is True
    # The resolver's own account of why it rejected them is model-written.
    assert "LLM rejected" not in text


def test_a_repair_that_succeeds_still_beats_asking_for_clarification():
    """The question is only asked when no repair could fix it. A planner that
    picks a candidate from the menu gets the ordinary answer."""
    executor = ScriptedExecutor([_unresolved_with_candidates(), results_run()])
    planner = ScriptedPlanner([
        PlanAttempt(ok=True, plan=make_plan(anchor_name="neuralgia")),
    ])

    outcome = LoopController(executor, planner, config=config()).run(
        plan=make_plan(),
    )

    assert outcome.status == LOOP_ANSWERED
    assert outcome.answer["answer_kind"] == "ranked_candidates"


def test_a_category_relaxation_names_the_field_that_must_move_with_it():
    """The planner was given contradictory orders and obeyed the wrong one.

    Widening a return entity's category makes the path's
    `expected_result_category` invalid unless it moves too. The relax message
    said "change nothing else", so four category relaxations out of five came
    back rejected by plan-core:

        expected category 'SmallMolecule' is incompatible with return entity
        'candidate_chemical' category 'ChemicalEntity'

    The fifth ignored the instruction, produced a valid plan, and was flagged
    over-broad for it. There was no answer that satisfied both.
    """
    plan = make_plan(answer_category="SmallMolecule")
    plan["paths"][0]["expected_result_category"] = "SmallMolecule"

    executor = ScriptedExecutor([
        make_result(
            outcome="no_graph_data", verdict="no_answer", plan=plan,
            paths={"P1": make_path(
                verdict="no_answer", num_candidates=0,
                relaxations=[{
                    "axis": "category",
                    "detail": "entity 'candidate_drug' is restricted to "
                              "SmallMolecule",
                    "entity_ref": "candidate_drug",
                    "current": "SmallMolecule",
                }],
            )},
            resolution=make_resolution(),
        ),
        results_run(),
    ])
    widened = make_plan(answer_category="ChemicalEntity", plan_id="P-2")
    widened["paths"][0]["expected_result_category"] = "ChemicalEntity"
    planner = ScriptedPlanner([PlanAttempt(ok=True, plan=widened)])

    outcome = LoopController(executor, planner, config=config()).run(plan=plan)

    request = planner.requests[0]
    assert request.must_also_change == ["paths[P1].expected_result_category"]

    message = json.dumps(request.to_dict())
    assert "paths[P1].expected_result_category" in message

    # And having been told, the plan that moves it is not over-broad.
    assert outcome.status == LOOP_ANSWERED
    assert outcome.state.over_relaxations == []


def test_the_permitted_fields_come_from_the_same_place_the_check_uses():
    """If the prompt and the check disagreed, the planner could obey one and
    fail the other — which is exactly what happened."""
    from loop_controller.fingerprint import dependent_patterns
    from loop_controller.contracts import RelaxationAxis

    plan = make_plan()
    axis = RelaxationAxis(path_id="P1", axis="category",
                          entity_ref="candidate_drug")
    executor = ScriptedExecutor([results_run()])
    planner = ScriptedPlanner([])
    request = build_request(
        question="q", prior_plan=plan, prior_fingerprint="plan:x",
        action=RELAX_PLAN, diagnosis=diagnose(results_run(), plan=plan),
        relaxation=axis,
    )

    assert request.must_also_change == dependent_patterns(axis, plan)


def test_an_unresolved_anchor_is_a_refusal_however_the_loop_stopped():
    """Two endings, one meaning.

    The failed-revision path knew that an unresolved anchor is a refusal; the
    `abandon` path did not. A live run composed a clarification listing twenty
    candidate entries and reported its status as `exhausted`, which described
    the budget rather than what happened.
    """
    unresolved = make_result(
        outcome="unresolved_grounding", replannable=True, verdict="unexecutable",
        detail="named entities did not resolve: ['disease_pain']",
        paths={},
        resolution={"disease_pain": {
            "entity_ref": "disease_pain", "query": "neuropathic pain",
            "expected_category": "Disease", "is_variable": False,
            "resolved_curies": [], "confidence": "high",
            "considered": [{"curie": "MONDO:0021667", "label": "neuralgia",
                            "types": ["biolink:Disease"], "rank": 1}],
        }},
    )

    # Ending via `abandon`: no planner, so repair is illegal from the start.
    executor = ScriptedExecutor([unresolved])
    abandoned = LoopController(executor, config=config(
        budget=Budget(max_iterations=1),
    )).run(plan=make_plan())

    assert abandoned.state.attempts[-1].decision.action == ABANDON
    assert abandoned.status == LOOP_REFUSED
    assert abandoned.answer["answer_kind"] == "clarification_needed"

    # Ending via a failed revision: same status, same answer.
    executor2 = ScriptedExecutor([unresolved])
    failed = LoopController(executor2, ScriptedPlanner([
        PlanAttempt(ok=False, errors=["could not fix it"]),
    ]), config=config()).run(plan=make_plan())

    assert failed.status == LOOP_REFUSED
    assert failed.answer["answer_kind"] == "clarification_needed"


def test_an_ordinary_dead_end_is_still_exhausted():
    """The change is narrow: only an unresolved anchor becomes a refusal."""
    executor = ScriptedExecutor([timeout_run()] * 5)
    outcome = LoopController(executor, config=config(
        budget=Budget(max_iterations=5),
    )).run(plan=make_plan())

    assert outcome.status == LOOP_EXHAUSTED
