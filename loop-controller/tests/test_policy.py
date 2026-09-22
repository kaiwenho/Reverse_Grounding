"""The fence.

These are the guarantees that hold no matter what chooses the next move, so
they are tested against the policy layer directly rather than through the loop.
If any of these fails, an LLM controller is unbounded.
"""

from __future__ import annotations

import pytest

from loop_controller.contracts import (
    ABANDON, ACCEPT, Budget, Decision, Diagnosis, LoopState,
    OUTCOME_FILTERED_OUT, RelaxationAxis,
    OUTCOME_JOIN_FAILURE, OUTCOME_NO_DATA, OUTCOME_REFUSED, OUTCOME_TRUNCATED,
    OUTCOME_UNRESOLVED, RELAX_PLAN, REPAIR_PLAN, REPORT_ABSENCE,
    RETRY_EXECUTION,
)
from loop_controller.diagnose import diagnose
from loop_controller.policy import (
    ESCALATION_LADDER, axis_is_locked, default_action, legal_actions, loosens,
    next_escalation, validate_decision,
)

from helpers import (
    make_candidate, make_path, make_plan, make_resolution, make_result,
    relaxation_axes,
)


def state(**kwargs) -> LoopState:
    budget = kwargs.pop("budget", Budget())
    st = LoopState(question="q", budget=budget)
    for key, value in kwargs.items():
        setattr(st, key, value)
    return st


def empty_run(plan=None):
    plan = plan or make_plan()
    result = make_result(
        outcome=OUTCOME_NO_DATA, verdict="no_answer", plan=plan,
        paths={"P1": make_path(
            verdict="no_answer", relaxations=relaxation_axes(), num_candidates=0,
        )},
    )
    return diagnose(result, None, plan)


# ---------------------------------------------------------------------------
# Accept
# ---------------------------------------------------------------------------


def test_accepting_requires_results():
    diag = diagnose(make_result(candidates=[make_candidate()]), None, make_plan())
    assert ACCEPT in legal_actions(diag, state())

    assert ACCEPT not in legal_actions(empty_run(), state())
    assert "no results" in legal_actions(empty_run(), state()).refusals[ACCEPT]


def test_accepting_with_a_concept_warning_is_legal_but_flagged():
    """Legal, because a caveated answer beats no answer when nothing better is
    available — and the caveat travels with it into the composed answer."""
    result = make_result(
        candidates=[make_candidate()],
        concept_warning="results may concern a different concept",
    )
    legal = legal_actions(diagnose(result, None, make_plan()), state())
    assert ACCEPT in legal
    assert any("concept check" in c for c in legal.cautions)


# ---------------------------------------------------------------------------
# Relaxation: monotone, and never on a locked constraint
# ---------------------------------------------------------------------------


def test_relaxation_is_offered_tightest_first():
    legal = legal_actions(empty_run(), state())
    assert RELAX_PLAN in legal
    assert legal.relaxation_options[0].axis == "qualifiers"


def test_a_spent_axis_is_never_offered_again():
    diag = empty_run()
    st = state(spent_axes={"P1:qualifiers"})
    legal = legal_actions(diag, st)
    assert "P1:qualifiers" not in [a.key for a in legal.relaxation_options]
    assert legal.relaxation_options[0].axis == "predicate"


def test_widening_a_pinned_anchor_is_refused():
    """The anchor is what the question is about. Widening it is how a loop
    quietly answers a different question and reports success."""
    diag = empty_run()
    axis = diag.axis_by_key("P1:category:disease")
    assert axis is not None
    assert "pinned anchor" in (axis_is_locked(axis, diag) or "")

    legal = legal_actions(diag, state())
    offered = [a.key for a in legal.relaxation_options]
    assert "P1:category:disease" not in offered
    assert "P1:category:candidate_drug" in offered  # the answer entity is fine


def test_a_user_requested_evidence_policy_is_never_relaxed():
    """The case that matters is the one where relaxing would work: the filter
    removed every candidate, and dropping it would produce results."""
    plan = make_plan(evidence_policy={
        "origin": "user_requested", "application": "filter",
        "rationale": "the user asked for published evidence",
        "min_publications": 2,
    })
    result = make_result(
        outcome=OUTCOME_FILTERED_OUT, verdict="no_answer", plan=plan,
        paths={"P1": make_path(num_candidates=0)},
        evidence={"filter_accounting": {"P1": {
            "candidates_kept": 0, "instances_dropped": 30,
            "instance_drop_reasons": {"min_publications": 30},
        }}},
    )
    diag = diagnose(result, None, plan)
    legal = legal_actions(diag, state())

    assert RELAX_PLAN not in legal
    assert ACCEPT not in legal
    assert any("never weakened" in c or "not be weakened" in c for c in legal.cautions)


def test_relaxation_is_refused_when_coverage_was_incomplete():
    """An empty result on an unfinished traversal is not evidence that the
    constraints were too tight."""
    result = make_result(
        outcome=OUTCOME_TRUNCATED, verdict="inconclusive",
        paths={"P1": make_path(
            coverage_complete=False, direct_outcome="timeout",
            relaxations=relaxation_axes(), num_candidates=0,
        )},
    )
    legal = legal_actions(diagnose(result, None, make_plan()), state())
    assert RELAX_PLAN not in legal
    assert "incomplete" in legal.refusals[RELAX_PLAN]


def test_relaxation_needs_a_planner():
    legal = legal_actions(empty_run(), state(), planner_available=False)
    assert RELAX_PLAN not in legal
    assert "planner" in legal.refusals[RELAX_PLAN]


# ---------------------------------------------------------------------------
# Repair
# ---------------------------------------------------------------------------


def test_repair_follows_the_executor_replannable_flag():
    result = make_result(
        outcome=OUTCOME_UNRESOLVED, replannable=True, verdict="unexecutable",
        resolution=make_resolution(resolved=False),
    )
    assert REPAIR_PLAN in legal_actions(diagnose(result, None, make_plan()), state())
    assert REPAIR_PLAN not in legal_actions(empty_run(), state())


def test_repair_stops_when_the_planner_budget_is_spent():
    result = make_result(
        outcome=OUTCOME_UNRESOLVED, replannable=True, verdict="unexecutable",
        resolution=make_resolution(resolved=False),
    )
    diag = diagnose(result, None, make_plan())
    st = state(budget=Budget(max_planner_calls=2), planner_calls=2)
    legal = legal_actions(diag, st)
    assert REPAIR_PLAN not in legal
    assert "planner call budget" in legal.refusals[REPAIR_PLAN]


# ---------------------------------------------------------------------------
# Execution escalation
# ---------------------------------------------------------------------------


def test_the_escalation_ladder_is_finite():
    applied: dict = {}
    rungs = 0
    while True:
        rung = next_escalation(applied)
        if rung is None:
            break
        applied.update(rung)
        rungs += 1
        assert rungs <= 10, "the ladder must terminate"
    assert rungs == len(ESCALATION_LADDER)


def test_retrying_stops_being_legal_at_the_top_of_the_ladder():
    result = make_result(
        outcome=OUTCOME_TRUNCATED, verdict="inconclusive",
        paths={"P1": make_path(
            coverage_complete=False, direct_outcome="timeout", num_candidates=0,
        )},
    )
    diag = diagnose(result, None, make_plan())

    assert RETRY_EXECUTION in legal_actions(diag, state())

    top = dict(ESCALATION_LADDER[-1])
    legal = legal_actions(diag, state(execution_overrides=top))
    assert RETRY_EXECUTION not in legal
    assert "ladder is spent" in legal.refusals[RETRY_EXECUTION]


def test_retrying_is_refused_when_the_run_was_not_at_fault():
    legal = legal_actions(empty_run(), state())
    assert RETRY_EXECUTION not in legal


@pytest.mark.parametrize("proposed,current,expected", [
    ({"timeout": 300}, {"timeout": 120}, True),
    ({"timeout": 60}, {"timeout": 120}, False),          # tightening
    ({"timeout": 120}, {"timeout": 120}, False),         # no change
    ({"skip_direct_over_hops": 1}, {"skip_direct_over_hops": 3}, True),
    ({"skip_direct_over_hops": 4}, {"skip_direct_over_hops": 3}, False),
    ({"timeout": 300}, {}, True),
    ({"nonsense": 1}, {}, False),
    ({}, {}, False),
])
def test_overrides_must_loosen(proposed, current, expected):
    assert loosens(proposed, current) is expected


# ---------------------------------------------------------------------------
# Absence
# ---------------------------------------------------------------------------


def test_absence_requires_a_completed_traversal():
    assert REPORT_ABSENCE in legal_actions(empty_run(), state())

    unfinished = make_result(
        outcome=OUTCOME_NO_DATA, verdict="no_answer",
        paths={"P1": make_path(coverage_complete=False, num_candidates=0)},
    )
    legal = legal_actions(diagnose(unfinished, None, make_plan()), state())
    assert REPORT_ABSENCE not in legal
    assert "incomplete" in legal.refusals[REPORT_ABSENCE]


def test_a_timeout_is_never_reported_as_an_absence():
    """The distinction the whole outcome taxonomy exists for."""
    result = make_result(
        outcome=OUTCOME_TRUNCATED, verdict="inconclusive",
        paths={"P1": make_path(direct_outcome="timeout", num_candidates=0)},
    )
    legal = legal_actions(diagnose(result, None, make_plan()), state())
    assert REPORT_ABSENCE not in legal
    assert "claim more than was established" in legal.refusals[REPORT_ABSENCE]


def test_reporting_absence_early_is_allowed_but_noted():
    legal = legal_actions(empty_run(), state())
    assert REPORT_ABSENCE in legal
    assert any("remain unspent" in c for c in legal.cautions)


def test_join_failure_is_an_absence_with_its_own_meaning():
    result = make_result(
        outcome=OUTCOME_JOIN_FAILURE, verdict="no_answer",
        paths={"P1": make_path(num_candidates=0)},
    )
    assert REPORT_ABSENCE in legal_actions(diagnose(result, None, make_plan()), state())


# ---------------------------------------------------------------------------
# Dead ends and budgets
# ---------------------------------------------------------------------------


def test_a_refusal_leaves_only_abandon():
    result = make_result(outcome=OUTCOME_REFUSED, verdict="refused")
    legal = legal_actions(diagnose(result, None, make_plan()), state())
    assert legal.allowed == [ABANDON]


def test_an_exhausted_budget_leaves_only_terminal_moves():
    diag = empty_run()
    st = state(budget=Budget(max_iterations=1))
    st.attempts = [object()]  # one iteration already run
    legal = legal_actions(diag, st)
    assert RELAX_PLAN not in legal
    assert REPAIR_PLAN not in legal
    assert RETRY_EXECUTION not in legal
    assert REPORT_ABSENCE in legal
    assert "budget exhausted" in legal.refusals[RELAX_PLAN]


def test_an_unknown_outcome_narrows_to_accept_or_stop():
    result = make_result(outcome="a_new_outcome", candidates=[make_candidate()])
    legal = legal_actions(diagnose(result, None, make_plan()), state())
    assert set(legal.allowed) == {ABANDON, ACCEPT}


# ---------------------------------------------------------------------------
# The deterministic default
# ---------------------------------------------------------------------------


def test_default_prefers_repair_over_everything():
    result = make_result(
        outcome=OUTCOME_UNRESOLVED, replannable=True, verdict="unexecutable",
        resolution=make_resolution(resolved=False),
    )
    diag = diagnose(result, None, make_plan())
    decision = default_action(diag, state(), legal_actions(diag, state()))
    assert decision.action == REPAIR_PLAN
    assert "unresolved:disease" in decision.repair_focus


def test_default_completes_an_unfinished_run_before_accepting_it():
    """Cached work is not repeated, so completing coverage usually costs one
    query rather than a whole run."""
    result = make_result(
        candidates=[make_candidate()],
        paths={"P1": make_path(coverage_complete=False, num_candidates=1)},
    )
    diag = diagnose(result, None, make_plan())
    decision = default_action(diag, state(), legal_actions(diag, state()))
    assert decision.action == RETRY_EXECUTION
    assert decision.execution_overrides["timeout"] == ESCALATION_LADDER[0]["timeout"]


def test_default_accepts_clean_results():
    diag = diagnose(make_result(candidates=[make_candidate()]), None, make_plan())
    assert default_action(diag, state(), legal_actions(diag, state())).action == ACCEPT


def test_default_relaxes_the_tightest_axis_on_an_empty_result():
    diag = empty_run()
    decision = default_action(diag, state(), legal_actions(diag, state()))
    assert decision.action == RELAX_PLAN
    assert decision.relaxation_key == "P1:qualifiers"


def test_default_falls_through_to_absence_then_abandon():
    result = make_result(
        outcome=OUTCOME_NO_DATA, verdict="no_answer",
        paths={"P1": make_path(num_candidates=0)},  # no relaxations offered
    )
    diag = diagnose(result, None, make_plan())
    assert default_action(diag, state(), legal_actions(diag, state())).action == REPORT_ABSENCE

    refused = diagnose(make_result(outcome=OUTCOME_REFUSED, verdict="refused"), None, make_plan())
    assert default_action(refused, state(), legal_actions(refused, state())).action == ABANDON


# ---------------------------------------------------------------------------
# Checking a proposed decision
# ---------------------------------------------------------------------------


def test_an_illegal_action_is_rejected_with_the_reason():
    diag = empty_run()
    legal = legal_actions(diag, state())
    problems = validate_decision(
        Decision(action=ACCEPT, rationale="looks fine"), diag, state(), legal,
    )
    assert any("not legal here" in p and "no results" in p for p in problems)


def test_an_invented_action_is_rejected():
    diag = empty_run()
    problems = validate_decision(
        Decision(action="ask_the_user", rationale="x"), diag, state(),
        legal_actions(diag, state()),
    )
    assert any("is not a move" in p for p in problems)


def test_relaxing_an_axis_that_was_not_suggested_is_rejected():
    diag = empty_run()
    problems = validate_decision(
        Decision(action=RELAX_PLAN, rationale="x", relaxation_key="P9:predicate"),
        diag, state(), legal_actions(diag, state()),
    )
    assert any("not an axis the executor suggested" in p for p in problems)


def test_relaxing_a_spent_axis_is_rejected():
    diag = empty_run()
    st = state(spent_axes={"P1:qualifiers"})
    problems = validate_decision(
        Decision(action=RELAX_PLAN, rationale="x", relaxation_key="P1:qualifiers"),
        diag, st, legal_actions(diag, st),
    )
    assert any("monotone" in p for p in problems)


def test_relaxing_the_anchor_is_rejected_even_when_named_directly():
    diag = empty_run()
    problems = validate_decision(
        Decision(action=RELAX_PLAN, rationale="x", relaxation_key="P1:category:disease"),
        diag, state(), legal_actions(diag, state()),
    )
    assert any("pinned anchor" in p for p in problems)


def test_a_retry_that_changes_nothing_is_rejected():
    result = make_result(
        outcome=OUTCOME_TRUNCATED, verdict="inconclusive",
        paths={"P1": make_path(direct_outcome="timeout", num_candidates=0)},
    )
    diag = diagnose(result, None, make_plan())
    st = state(execution_overrides={"timeout": 300})
    problems = validate_decision(
        Decision(action=RETRY_EXECUTION, rationale="x",
                 execution_overrides={"timeout": 300}),
        diag, st, legal_actions(diag, st),
    )
    assert any("do not loosen" in p for p in problems)


def test_a_missing_rationale_is_rejected():
    diag = diagnose(make_result(candidates=[make_candidate()]), None, make_plan())
    problems = validate_decision(
        Decision(action=ACCEPT, rationale="  "), diag, state(),
        legal_actions(diag, state()),
    )
    assert any("rationale is required" in p for p in problems)


# ---------------------------------------------------------------------------
# False absence from a bad anchor
# ---------------------------------------------------------------------------


def _mismatched_empty_run():
    """A completed, empty run whose anchor was pinned to the wrong category."""
    return make_result(
        outcome=OUTCOME_NO_DATA, verdict="no_answer",
        detail="the query returned no results",
        paths={"P1": make_path(verdict="no_answer", num_candidates=0)},
        resolution=make_resolution(types=["biolink:PhenotypicFeature"]),
    )


def test_absence_is_refused_when_an_anchor_does_not_match_its_category(requires_biolink):
    """The whole point. Without this the loop states a fact about the world on
    the strength of a query that could not have returned anything."""
    diag = diagnose(_mismatched_empty_run(), plan=make_plan())
    legal = legal_actions(diag, LoopState())

    assert REPORT_ABSENCE not in legal
    reason = legal.refusals[REPORT_ABSENCE]
    assert "anchor" in reason
    assert "not a finding" in reason


def test_the_same_run_with_a_matching_anchor_does_qualify_as_an_absence():
    """The guard is narrow: it fires on the mismatch, not on empty results."""
    result = make_result(
        outcome=OUTCOME_NO_DATA, verdict="no_answer",
        paths={"P1": make_path(verdict="no_answer", num_candidates=0)},
        resolution=make_resolution(),
    )
    legal = legal_actions(diagnose(result, plan=make_plan()), LoopState())

    assert REPORT_ABSENCE in legal


def test_a_mismatched_anchor_offers_repair_instead(requires_biolink):
    """Refusing the absence without offering a move would only turn a wrong
    answer into no answer. The fix is a better anchor, which is a plan change."""
    diag = diagnose(_mismatched_empty_run(), plan=make_plan())
    legal = legal_actions(diag, LoopState())

    assert REPAIR_PLAN in legal

    decision = default_action(diag, LoopState(), legal)
    assert decision.action == REPAIR_PLAN
    assert "anchor_category:disease" in decision.repair_focus


def test_a_filtered_absence_is_also_refused_on_a_mismatched_anchor(requires_biolink):
    """The narrower claim is still a claim. If the anchor was wrong, the
    candidates the filters removed may have been the wrong candidates."""
    result = make_result(
        outcome=OUTCOME_FILTERED_OUT, verdict="no_answer",
        resolution=make_resolution(types=["biolink:PhenotypicFeature"]),
    )
    legal = legal_actions(diagnose(result, plan=make_plan()), LoopState())

    assert REPORT_ABSENCE not in legal


def test_the_mismatch_is_reported_as_a_caution_to_the_decision_maker(requires_biolink):
    """A model told only that absence is unavailable invents a reason; one told
    which anchor and which category argues with the reason instead."""
    diag = diagnose(_mismatched_empty_run(), plan=make_plan())
    legal = legal_actions(diag, LoopState())

    assert any("PhenotypicFeature" in c for c in legal.cautions)


# ---------------------------------------------------------------------------
# Axis naming across the process boundary
# ---------------------------------------------------------------------------


def test_the_executors_entity_constraint_axis_is_recognised_as_locked():
    """The executor spells it `entity_constraint`; this package spells it
    `constraints`. Nothing reconciled the two, so a user-requested constraint
    arrived unlocked — and the only constraint the plan contract carries is
    one the user asked for.

    Never fired in any run, because no plan so far carried a constraint. It
    would have fired the first time somebody asked for approved drugs only.
    """
    raw = RelaxationAxis(
        path_id="P1", axis="entity_constraint", entity_ref="candidate_drug",
    )
    diag = Diagnosis(
        locked_constraints=["constraint:candidate_drug:approval_status"],
    )

    assert raw.axis == "constraints"
    assert axis_is_locked(raw, diag) is not None


def test_the_entity_constraint_axis_sorts_where_it_belongs():
    """Second tightest, not last. An unrecognised name sorts after every known
    axis, so the loop would have reached for a category widening before a
    constraint drop — the opposite of smallest-change-first."""
    constraint = RelaxationAxis(path_id="P1", axis="entity_constraint")
    category = RelaxationAxis(path_id="P1", axis="category")

    assert constraint.rank < category.rank


def test_a_locked_constraint_axis_is_never_offered_for_relaxation():
    """End to end through the policy layer, on a plan that carries one."""
    plan = make_plan()
    plan["entities"][1]["is_variable"] = False
    plan["entities"][1]["name"] = "approved drug"
    plan["entities"][1]["constraints"] = [
        {"field": "approval_status", "operator": "==", "value": "approved"},
    ]
    result = make_result(
        outcome=OUTCOME_NO_DATA, verdict="no_answer",
        paths={"P1": make_path(
            verdict="no_answer", num_candidates=0,
            relaxations=[{
                "axis": "entity_constraint",
                "detail": "entity 'candidate_drug' carries 1 constraint(s)",
                "entity_ref": "candidate_drug",
            }],
        )},
        resolution=make_resolution(),
    )
    legal = legal_actions(diagnose(result, plan=plan), LoopState())

    assert [a.key for a in legal.relaxation_options] == []
    assert RELAX_PLAN not in legal


# ---------------------------------------------------------------------------
# A plan that dropped part of the question
# ---------------------------------------------------------------------------


def _plan_with_an_orphan():
    plan = make_plan()
    plan["entities"].append({
        "entity_ref": "derm_herp", "name": "dermatitis herpetiformis",
        "biolink_category": "Disease", "is_variable": False,
    })
    return plan


def test_results_are_not_accepted_while_a_named_concept_went_unqueried():
    """Twenty real candidates, every claim groundable, and not the answer.

    The grounding gate passes them, the candidate table looks ordinary, and
    nothing downstream can tell a constraint went missing — so the fence has to
    be here.
    """
    result = make_result(candidates=[make_candidate()])
    diag = diagnose(result, plan=_plan_with_an_orphan())
    legal = legal_actions(diag, LoopState())

    assert ACCEPT not in legal
    assert "narrower question" in legal.refusals[ACCEPT]
    assert REPAIR_PLAN in legal

    decision = default_action(diag, LoopState(), legal)
    assert decision.action == REPAIR_PLAN
    assert "unused_entity:derm_herp" in decision.repair_focus


def test_the_same_results_are_accepted_once_repair_is_spent():
    """A partial answer beats no answer, as long as it says what it is missing.
    Refusing outright would turn a nearly-right plan into nothing at all."""
    result = make_result(candidates=[make_candidate()])
    diag = diagnose(result, plan=_plan_with_an_orphan())
    spent = LoopState(budget=Budget(max_planner_calls=0))
    legal = legal_actions(diag, spent)

    assert ACCEPT in legal
    assert REPAIR_PLAN not in legal
    assert any("no repair remains" in c for c in legal.cautions)


def test_an_orphan_makes_a_successful_run_repairable():
    """The executor cannot see this. It ran the plan it was given, the run
    succeeded, and results came back — all correct from where it stands."""
    diag = diagnose(make_result(candidates=[make_candidate()]),
                    plan=_plan_with_an_orphan())

    assert diag.verdict == "success"
    assert not diag.replannable
    assert diag.plan_repairable


def test_a_complete_plan_accepts_normally():
    """The guard is narrow: it fires on the unused entity, not on results."""
    diag = diagnose(make_result(candidates=[make_candidate()]), plan=make_plan())
    legal = legal_actions(diag, LoopState())

    assert ACCEPT in legal
    assert default_action(diag, LoopState(), legal).action == ACCEPT
