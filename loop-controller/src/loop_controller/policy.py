"""
policy.py — The fence. Which moves are legal, and what to do if nobody chooses.

An LLM picks the next move in this controller. That is a deliberate choice and
it buys real adaptivity: the model can read a thin-evidence distribution
alongside a concept warning and judge that repairing the anchor matters more
than accepting twenty-five candidates, which no table of outcome-to-action
rules would get right in every case.

What it must not buy is a loop that runs forever, spends without limit, answers
a question the user did not ask, or stalls when the model is unreachable. So
the model does not choose from all moves — it chooses from the moves this
module has already declared legal, and its choice is checked against the same
rules before it is acted on. The model's freedom is real and it is bounded, and
the bound is code rather than prompt text, because a prompt is a request and
this needs to be a guarantee.

Four invariants hold regardless of what anything chooses:

1. **No repetition.** A plan whose executable content matches one already run
   is not a new iteration. Enforced by fingerprint, in the loop.
2. **Monotone relaxation.** An axis is loosened at most once and never
   re-tightened. Since the axis set is finite and every relaxation spends one,
   relaxation terminates.
3. **Locked constraints stay locked.** A user-requested evidence policy and a
   pinned anchor are never relaxed, most of all when relaxing them would
   produce results — that is the case where the loop would otherwise answer a
   different question and report success.
4. **Bounded escalation.** Execution retries walk a finite ladder. When the
   ladder ends, retrying is no longer legal.

`default_action` is what happens when the model fails, times out, or proposes
something illegal twice. It is a plain priority ordering, and it is a complete
controller on its own: the loop is correct without an LLM and better with one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .contracts import (
    ABANDON, ACCEPT, ACTIONS, Decision, Diagnosis, EXECUTION_KNOBS, LoopState,
    OUTCOME_BACKEND, OUTCOME_FILTERED_OUT, OUTCOME_JOIN_FAILURE,
    OUTCOME_MISSING_INPUT, OUTCOME_NO_DATA, OUTCOME_REFUSED,
    OUTCOME_TRUNCATED, OUTCOME_UNSUPPORTED_VERSION, RELAX_PLAN, REPAIR_PLAN,
    REPORT_ABSENCE, RETRY_EXECUTION, RelaxationAxis,
)


#: Outcomes where an empty result is a statement about the graph rather than
#: about the run. Only these can be reported to a user as an absence.
ABSENCE_OUTCOMES = frozenset({OUTCOME_NO_DATA, OUTCOME_JOIN_FAILURE})

#: Outcomes where the run itself is what failed. Re-running may fix them;
#: replanning will not.
EXECUTION_FAULT_OUTCOMES = frozenset({OUTCOME_TRUNCATED, OUTCOME_BACKEND})

#: Outcomes no move can improve. The plan was declined, targets a contract this
#: executor does not implement, or depends on an input nobody supplied — none
#: of which the loop can supply on the user's behalf.
DEAD_END_OUTCOMES = frozenset({
    OUTCOME_REFUSED, OUTCOME_UNSUPPORTED_VERSION, OUTCOME_MISSING_INPUT,
})


# ---------------------------------------------------------------------------
# Execution escalation ladder
# ---------------------------------------------------------------------------

#: Successive loosenings for a run that did not finish. Finite by construction:
#: three rungs, each strictly looser than the last, and `next_escalation`
#: returns None past the end. That is what makes `retry_execution` terminate
#: without a separate retry counter — the ladder is the counter.
ESCALATION_LADDER: Tuple[Dict[str, Any], ...] = (
    {"timeout": 120.0},
    {"timeout": 300.0, "skip_direct_over_hops": 2},
    {"timeout": 600.0, "skip_direct_over_hops": 1},
)


def next_escalation(current: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """The next rung strictly looser than what is already applied.

    Compares on timeout because it is the rung's ordering key; the other knobs
    move with it. Returns None at the top of the ladder, which is what makes
    retrying stop being legal rather than stop being useful.
    """
    applied = float(current.get("timeout") or 0.0)
    for rung in ESCALATION_LADDER:
        if float(rung["timeout"]) > applied:
            return dict(rung)
    return None


def loosens(proposed: Dict[str, Any], current: Dict[str, Any]) -> bool:
    """True when every proposed knob is at least as loose, and one is looser.

    `skip_direct_over_hops` inverts: a smaller value decomposes sooner, which
    is the loosening. Anything that would tighten a knob is refused outright,
    because `retry_execution` is meant to be safe to take without re-asking
    whether the plan is still right, and a tightening retry could turn results
    into no results and be read as a fact about the graph.
    """
    if not proposed:
        return False
    looser_somewhere = False
    for knob, value in proposed.items():
        if knob not in EXECUTION_KNOBS:
            return False
        try:
            new = float(value)
        except (TypeError, ValueError):
            return False
        old_raw = current.get(knob)
        if old_raw is None:
            looser_somewhere = True
            continue
        old = float(old_raw)
        if knob == "skip_direct_over_hops":
            if new > old:
                return False
            if new < old:
                looser_somewhere = True
        else:
            if new < old:
                return False
            if new > old:
                looser_somewhere = True
    return looser_somewhere


# ---------------------------------------------------------------------------
# Legality
# ---------------------------------------------------------------------------


@dataclass
class LegalActions:
    """Which moves are available, and why the others are not.

    The refusals are kept and shown to the decision maker. A model told only
    "choose from accept or abandon" will invent a third option; a model told
    "relaxing is unavailable because the only remaining axis would widen the
    pinned anchor eczema" argues with the reason instead, which is both a
    better decision and a legible disagreement in the trace.
    """

    allowed: List[str] = field(default_factory=list)
    refusals: Dict[str, str] = field(default_factory=dict)
    relaxation_options: List[RelaxationAxis] = field(default_factory=list)
    escalation: Optional[Dict[str, Any]] = None
    cautions: List[str] = field(default_factory=list)

    def __contains__(self, action: str) -> bool:
        return action in self.allowed

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "refusals": self.refusals,
            "relaxation_options": [a.to_dict() for a in self.relaxation_options],
            "escalation": self.escalation,
            "cautions": self.cautions,
        }


def axis_is_locked(axis: RelaxationAxis, diag: Diagnosis) -> Optional[str]:
    """Why this axis may not be relaxed, or None.

    Category widening on a pinned anchor is refused because the anchor is what
    the question is about. The executor already resolves anchors under LLM
    review and checks afterwards that results concern the intended concept;
    widening the anchor's category re-opens exactly the door those two stages
    exist to close, and it does so at the moment the loop is least able to
    notice, since a broader anchor usually does return something.

    Entity constraints are refused because the plan contract carries only one
    of them — approval status — and it is there only when the user asked for
    approved drugs.
    """
    locked = set(diag.locked_constraints)

    if axis.axis == "category" and axis.entity_ref:
        if f"anchor:{axis.entity_ref}" in locked:
            return (
                f"'{axis.entity_ref}' is a pinned anchor; widening its category "
                f"would let the query drift to a different concept than the "
                f"question asked about"
            )

    if axis.axis == "constraints" and axis.entity_ref:
        matching = [c for c in locked if c.startswith(f"constraint:{axis.entity_ref}:")]
        if matching:
            return (
                f"'{axis.entity_ref}' carries a user-requested constraint "
                f"({', '.join(sorted(matching))}); the plan contract only holds "
                f"constraints the user asked for"
            )

    if axis.axis.startswith("evidence_policy"):
        return "the evidence policy records a user request and is never weakened"

    return None


def legal_actions(
    diag: Diagnosis,
    state: LoopState,
    *,
    planner_available: bool = True,
) -> LegalActions:
    """Compute the available moves for this diagnosis and state."""
    legal = LegalActions()
    exhausted = state.budget_exhausted()

    # A non-terminal move only makes sense if its result could be executed.
    # On the last permitted iteration it could not, so repairing, relaxing and
    # retrying are refused rather than taken and thrown away — which would
    # otherwise spend a planner call producing a plan the loop has no room to
    # run.
    no_room = exhausted or state.iterations_remaining() <= 1
    room_refusal = (
        f"budget exhausted: {exhausted}" if exhausted else
        "this is the last iteration the budget allows, so a new plan or a "
        "re-run could not be executed"
    )

    def allow(action: str) -> None:
        if action not in legal.allowed:
            legal.allowed.append(action)

    def refuse(action: str, reason: str) -> None:
        legal.refusals.setdefault(action, reason)

    # -- abandon is always available -------------------------------------
    allow(ABANDON)

    # -- dead ends --------------------------------------------------------
    if diag.outcome in DEAD_END_OUTCOMES:
        for action in ACTIONS:
            if action != ABANDON:
                refuse(action, f"outcome '{diag.outcome}' cannot be improved by any move")
        legal.cautions.append(f"{diag.outcome}: {diag.outcome_detail}")
        return legal

    if diag.unknown_outcome:
        legal.cautions.append(
            f"outcome '{diag.outcome}' is not known to this controller version"
        )
        if diag.num_results:
            allow(ACCEPT)
        for action in (REPAIR_PLAN, RELAX_PLAN, RETRY_EXECUTION, REPORT_ABSENCE):
            refuse(action, "unrecognised outcome; only accepting or stopping is safe")
        return legal

    # Whether a repair could still happen. Computed before the accept block
    # rather than in its own section further down, because accepting now
    # depends on it: an incomplete plan should be repaired while that is still
    # possible, and served with a caveat only when it is not.
    repair_possible = (
        not no_room
        and diag.plan_repairable
        and planner_available
        and not state.planner_budget_exhausted()
    )

    # -- accept -----------------------------------------------------------
    if diag.num_results > 0 and diag.orphan_entities and repair_possible:
        # The results are real and every claim about them will be grounded.
        # They are also answers to a smaller question than the one asked: a
        # concept the user named is sitting in the plan, queried by nothing.
        # Serving them is worse than spending another iteration, because
        # nothing downstream can tell the difference — the grounding gate
        # passes them, the candidate table looks ordinary, and the reader has
        # no way to know a constraint went missing.
        refuse(ACCEPT, _orphan_refusal(diag) + "; repairing the plan first")
        legal.cautions.append(_orphan_refusal(diag))
    elif diag.num_results > 0:
        allow(ACCEPT)
        if diag.orphan_entities:
            # Repair is spent. Serving a partial answer beats serving none, as
            # long as the answer says what it is missing.
            legal.cautions.append(
                _orphan_refusal(diag)
                + "; no repair remains, so these results answer the narrower "
                  "question and the answer must say so"
            )
        if diag.concept_warning:
            legal.cautions.append(
                "a concept check did not confirm the results concern the "
                "intended entity; accepting carries that caveat into the answer"
            )
        if diag.thin_annotation_paths:
            legal.cautions.append(
                f"paths {diag.thin_annotation_paths} have thin knowledge-level "
                f"annotation; the ranking's evidence terms rest on little metadata"
            )
        if diag.incomplete_coverage:
            legal.cautions.append(
                f"coverage incomplete on {diag.incomplete_coverage}; these "
                f"results are a partial view"
            )
    else:
        refuse(ACCEPT, "there are no results to accept")

    # -- everything past here needs an iteration left ---------------------
    if no_room:
        for action in (REPAIR_PLAN, RELAX_PLAN, RETRY_EXECUTION):
            refuse(action, room_refusal)
        if _absence_qualifies(diag):
            allow(REPORT_ABSENCE)
        else:
            refuse(REPORT_ABSENCE, _absence_refusal(diag))
        legal.cautions.append(room_refusal)
        return legal

    # -- repair -----------------------------------------------------------
    if not diag.plan_repairable:
        refuse(REPAIR_PLAN, f"the executor did not mark '{diag.outcome}' replannable")
    elif not planner_available:
        refuse(REPAIR_PLAN, "no planner is configured")
    elif state.planner_budget_exhausted():
        refuse(REPAIR_PLAN,
               f"planner call budget spent ({state.budget.max_planner_calls})")
    else:
        allow(REPAIR_PLAN)

    if diag.orphan_entities:
        legal.cautions.append(_orphan_refusal(diag))

    # -- relax ------------------------------------------------------------
    options: List[RelaxationAxis] = []
    blocked_reasons: List[str] = []
    for axis in diag.relaxation_axes:
        if axis.key in state.spent_axes:
            continue
        locked_reason = axis_is_locked(axis, diag)
        if locked_reason:
            blocked_reasons.append(f"{axis.key}: {locked_reason}")
            continue
        options.append(axis)
    legal.relaxation_options = options

    if diag.num_results:
        refuse(RELAX_PLAN, "relaxing is for an empty result, and there are results")
    elif not diag.relaxation_axes:
        refuse(RELAX_PLAN, "the executor suggested no relaxations for this run")
    elif diag.incomplete_coverage or diag.timed_out_paths:
        refuse(RELAX_PLAN,
               "coverage was incomplete, so an empty result is not evidence that "
               "the constraints were too tight")
    elif not options:
        refuse(RELAX_PLAN,
               "every suggested axis is already spent or locked"
               + (f" ({'; '.join(blocked_reasons)})" if blocked_reasons else ""))
    elif not planner_available:
        refuse(RELAX_PLAN, "no planner is configured to choose the replacement")
    elif state.planner_budget_exhausted():
        refuse(RELAX_PLAN,
               f"planner call budget spent ({state.budget.max_planner_calls})")
    else:
        allow(RELAX_PLAN)
        if blocked_reasons:
            legal.cautions.extend(blocked_reasons)

    if diag.filters_dropped_all:
        legal.cautions.append(
            "the evidence policy removed every candidate; it records a user "
            "request and will not be weakened, so this is a finding about the "
            "graph under the user's own filter"
        )

    if diag.anchor_category_mismatch:
        legal.cautions.append(_anchor_refusal(diag))

    # -- retry execution ---------------------------------------------------
    fault = (
        diag.outcome in EXECUTION_FAULT_OUTCOMES
        or bool(diag.timed_out_paths)
        or bool(diag.incomplete_coverage)
    )
    escalation = next_escalation(state.execution_overrides)
    if not fault:
        refuse(RETRY_EXECUTION,
               "nothing about this run suggests it was execution that failed")
    elif escalation is None:
        refuse(RETRY_EXECUTION,
               "the escalation ladder is spent; a longer timeout is no longer "
               "available")
    else:
        allow(RETRY_EXECUTION)
        legal.escalation = escalation

    # -- report absence ----------------------------------------------------
    if _absence_qualifies(diag):
        allow(REPORT_ABSENCE)
        if options:
            legal.cautions.append(
                f"{len(options)} relaxation axis/axes remain unspent; reporting "
                f"absence now states the graph has no data for the query "
                f"as written, which is narrower than having no data at all"
            )
    else:
        refuse(REPORT_ABSENCE, _absence_refusal(diag))

    return legal


def _absence_qualifies(diag: Diagnosis) -> bool:
    """Whether an empty result is defensible as a statement about the graph."""
    if diag.num_results:
        return False
    if diag.anchor_category_mismatch:
        # The query may have matched nothing because of what it pinned rather
        # than because of what the graph holds. See `_anchor_refusal`.
        return False
    if diag.outcome == OUTCOME_FILTERED_OUT:
        # Defensible, but as a narrower claim: nothing survived the user's own
        # filter. The composer says so in those words. Reached only when the
        # anchors are sound, since a mismatched anchor means the candidates
        # those filters removed may have been the wrong candidates.
        return True
    if diag.outcome not in ABSENCE_OUTCOMES:
        return False
    return not (diag.incomplete_coverage or diag.timed_out_paths)


def _absence_refusal(diag: Diagnosis) -> str:
    if diag.num_results:
        return "there are results, so there is no absence to report"
    if diag.anchor_category_mismatch:
        return _anchor_refusal(diag)
    if diag.outcome not in ABSENCE_OUTCOMES | {OUTCOME_FILTERED_OUT}:
        return (
            f"'{diag.outcome}' describes the run or the plan, not the graph; "
            f"reporting it as an absence would claim more than was established"
        )
    return (
        "coverage was incomplete or a query timed out, so nothing was "
        "established about what the graph contains"
    )


def _orphan_refusal(diag: Diagnosis) -> str:
    """Why results from a plan that dropped a concept are not the answer.

    Stated as what the plan did rather than as a score, because the reader
    needs to know which concept went missing in order to judge whether the
    remaining results are any use to them.
    """
    named = ", ".join(
        f"'{o.get('name') or o.get('entity_ref')}'"
        for o in diag.orphan_entities[:3]
    )
    return (
        f"the plan declares {named} and queries nothing with it, so any "
        f"results answer a narrower question than the one asked"
    )


def _anchor_refusal(diag: Diagnosis) -> str:
    """Why a mismatched anchor makes an absence claim unavailable.

    An absence says the graph holds nothing for this query. That is only worth
    saying if the query asked for what the question meant. A pinned anchor
    whose resolved identifier does not satisfy the category the plan gave it
    produces a node constraint that can match nothing whatever the graph
    contains — and because anchors are never widened, the loop has no move that
    would find out. Reporting an absence there states a fact about the world on
    the strength of a query that could not have returned anything.

    Repair remains available, and is the right move: the fix is a better anchor,
    which is a plan change rather than a relaxation.
    """
    parts = []
    for mismatch in diag.anchor_category_mismatch[:3]:
        parts.append(
            f"'{mismatch.get('entity_ref')}' was pinned as "
            f"{mismatch.get('expected_category')} but resolved to "
            f"{mismatch.get('curie')}, which the model types as "
            f"{', '.join(mismatch.get('reported_types') or [])}"
        )
    return (
        "an anchor's category does not match what it resolved to, so the query "
        "could return nothing regardless of what the graph holds ("
        + "; ".join(parts) + "); this is a plan problem, not a finding"
    )


# ---------------------------------------------------------------------------
# Deterministic default
# ---------------------------------------------------------------------------


def default_action(
    diag: Diagnosis, state: LoopState, legal: LegalActions,
) -> Decision:
    """The move to make when nothing else chooses one.

    A complete controller in its own right, and the reason the loop degrades
    rather than stalls when the model is unreachable. Priority order, with the
    reasoning for the two orderings that are not obvious:

    Repair before retry, because a plan that cannot be executed will not become
    executable by being executed again more patiently.

    Retry before accept, because coverage that is incomplete makes the results
    a partial view, and the executor's cache means completing it usually costs
    one query rather than a whole run.

    Accept before relax, because results in hand answer the question and a
    relaxed plan answers a slightly different one.
    """
    def pick(action: str, rationale: str, **kwargs: Any) -> Decision:
        return Decision(action=action, rationale=rationale, source="policy", **kwargs)

    if REPAIR_PLAN in legal:
        focus = _repair_focus(diag)
        return pick(
            REPAIR_PLAN,
            f"the executor marked this replannable ({diag.outcome}); "
            f"repairing {', '.join(focus) if focus else 'the plan'}",
            repair_focus=focus,
        )

    if RETRY_EXECUTION in legal and legal.escalation:
        return pick(
            RETRY_EXECUTION,
            "the run did not finish, so the plan has not been tested; "
            "re-running with a looser execution budget",
            execution_overrides=dict(legal.escalation),
        )

    if ACCEPT in legal and not diag.concept_warning:
        return pick(ACCEPT, f"{diag.num_results} candidate(s) with supporting evidence")

    if RELAX_PLAN in legal and legal.relaxation_options:
        axis = legal.relaxation_options[0]
        return pick(
            RELAX_PLAN,
            f"the query ran to completion and found nothing; loosening the "
            f"tightest remaining constraint ({axis.axis} on {axis.path_id})",
            relaxation_key=axis.key,
        )

    if ACCEPT in legal:
        return pick(
            ACCEPT,
            f"{diag.num_results} candidate(s); no better move remains, and the "
            f"concept-check caveat travels with the answer",
        )

    if REPORT_ABSENCE in legal:
        return pick(
            REPORT_ABSENCE,
            "the query ran to completion and the graph holds no data for it",
        )

    reason = state.budget_exhausted() or diag.outcome_detail or diag.outcome
    return pick(ABANDON, f"no move remains: {reason}")


def _repair_focus(diag: Diagnosis) -> List[str]:
    focus: List[str] = []
    for ref in diag.unresolved_entities:
        focus.append(f"unresolved:{ref}")
    for mismatch in diag.anchor_category_mismatch:
        focus.append(f"anchor_category:{mismatch.get('entity_ref')}")
    for orphan in diag.orphan_entities:
        focus.append(f"unused_entity:{orphan.get('entity_ref')}")
    for hop in diag.unsupported_hops:
        focus.append(f"unsupported:{hop.get('path_id')}")
    for err in diag.plan_validation_errors[:3]:
        focus.append(f"invalid:{err.split(':')[0]}")
    return focus


# ---------------------------------------------------------------------------
# Checking a proposed decision
# ---------------------------------------------------------------------------


def validate_decision(
    decision: Decision,
    diag: Diagnosis,
    state: LoopState,
    legal: LegalActions,
) -> List[str]:
    """Every reason this decision may not be acted on.

    Returned as a list rather than a first failure, because the list is sent
    back to the model as feedback and a model told one problem at a time
    fixes them one at a time.
    """
    problems: List[str] = []

    if decision.action not in ACTIONS:
        problems.append(
            f"'{decision.action}' is not a move; choose one of {list(ACTIONS)}"
        )
        return problems

    if decision.action not in legal.allowed:
        reason = legal.refusals.get(decision.action, "not available in this state")
        problems.append(f"'{decision.action}' is not legal here: {reason}")

    if not (decision.rationale or "").strip():
        problems.append("rationale is required; it is written to the audit trace")

    if decision.action == RELAX_PLAN:
        key = decision.relaxation_key
        if not key:
            problems.append(
                "relax_plan needs relaxation_key naming which axis to loosen"
            )
        else:
            axis = diag.axis_by_key(key)
            if axis is None:
                available = [a.key for a in legal.relaxation_options]
                problems.append(
                    f"'{key}' is not an axis the executor suggested; "
                    f"available: {available}"
                )
            elif key in state.spent_axes:
                problems.append(
                    f"'{key}' was already loosened on an earlier iteration; "
                    f"relaxation is monotone"
                )
            else:
                locked_reason = axis_is_locked(axis, diag)
                if locked_reason:
                    problems.append(f"'{key}' may not be relaxed: {locked_reason}")

    if decision.action == RETRY_EXECUTION:
        overrides = decision.execution_overrides or {}
        if not overrides:
            problems.append(
                "retry_execution needs execution_overrides; re-running a plan "
                "unchanged repeats the run rather than advancing it"
            )
        else:
            unknown = sorted(set(overrides) - set(EXECUTION_KNOBS))
            if unknown:
                problems.append(
                    f"unknown execution knob(s) {unknown}; "
                    f"available: {sorted(EXECUTION_KNOBS)}"
                )
            elif not loosens(overrides, state.execution_overrides):
                problems.append(
                    f"execution_overrides {overrides} do not loosen the current "
                    f"settings {state.execution_overrides or '(defaults)'}; a "
                    f"retry must change something in the direction of finishing"
                )

    if decision.action == REPAIR_PLAN and not diag.plan_repairable:
        problems.append(
            "repair_plan requires a replannable outcome; this one describes the "
            "backend or the data rather than the plan"
        )

    return problems
