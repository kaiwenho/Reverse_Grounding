"""
contracts.py — The types the loop is written in.

Three of these carry the design and are worth reading before the rest of the
package.

``Diagnosis`` is a reading of one execution, and only a reading. It restates
what the executor already reported, in the vocabulary the loop makes decisions
in. It contains no judgement about what to do next, because separating the
reading from the decision is what makes the decision reviewable: a bad
iteration is either a misread of the result or a bad call on a correct read,
and those have different fixes.

``Action`` is the closed set of moves. The loop can accept an answer, repair a
broken plan, relax an over-constrained one, re-run the same plan with different
execution settings, report a grounded absence, or stop. There is no seventh
move and no free-text move. A decision maker — including an LLM one — chooses
from this set or is overridden.

``Decision`` is one choice of action plus the parameters that action needs.
Every field is checked before it is acted on: an action must be legal for the
current diagnosis and state, a relaxation must name an axis the executor
actually suggested and that has not been used, an execution override must
loosen rather than tighten. A decision that fails those checks never reaches
the loop body.

``LoopState`` is what makes termination provable rather than hoped for. It
holds the plans already tried (by fingerprint), the relaxation axes already
spent, and the budget already consumed. The policy layer reads it to decide
what is still legal, so "the loop cannot repeat itself" is enforced by the
state rather than by the good behaviour of whatever is choosing.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple


CONTROLLER_SCHEMA_VERSION = "loop-controller/0.1.0"


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

#: The answer is good enough to serve. Only legal when there are results.
ACCEPT = "accept"

#: The plan was broken in a way a planner can fix: an invalid field, a name
#: that did not resolve, a hop the backend cannot answer. The executor marks
#: exactly these `replannable`, and two of them come with a grounded menu of
#: alternatives — the resolver's candidate list, or the meta knowledge graph's
#: supported relationships — so the repair is constrained rather than blind.
REPAIR_PLAN = "repair_plan"

#: The plan was valid, ran to completion, and found nothing. The constraints
#: may be tighter than the graph. One axis is loosened, and only one, so that
#: an eventual answer can be attributed to the constraint that was blocking it.
RELAX_PLAN = "relax_plan"

#: The plan is fine and the run was not. A timeout, an incomplete traversal, a
#: backend failure. Re-runs the same plan with different execution settings and
#: makes no planner call at all. The executor's cache makes this cheap: the
#: work that did finish is not repeated.
RETRY_EXECUTION = "retry_execution"

#: The graph has no data for this question, and the loop has established that
#: rather than merely failed to find any. A finding, and the most defensible
#: thing the system can say when it cannot answer.
REPORT_ABSENCE = "report_absence"

#: Stop without an answer. A refusal, a user-mandated filter that removed
#: everything, an exhausted budget, or a state with no legal move left.
ABANDON = "abandon"

ACTIONS: Tuple[str, ...] = (
    ACCEPT, REPAIR_PLAN, RELAX_PLAN, RETRY_EXECUTION, REPORT_ABSENCE, ABANDON,
)

#: Actions that end the loop.
TERMINAL_ACTIONS = frozenset({ACCEPT, REPORT_ABSENCE, ABANDON})

#: Actions that ask the planner for a new plan.
PLANNER_ACTIONS = frozenset({REPAIR_PLAN, RELAX_PLAN})


# ---------------------------------------------------------------------------
# Executor outcomes, mirrored from plan_executor.aggregate
# ---------------------------------------------------------------------------
# Mirrored rather than imported: the executor is reached as a subprocess, so
# importing its package here would create a dependency the integration does not
# otherwise need. `check_outcome_drift` in diagnose.py notices a value this
# package does not know rather than silently treating it as unrecognised.

OUTCOME_RESULTS = "results"
OUTCOME_REFUSED = "plan_refused"
OUTCOME_UNSUPPORTED_VERSION = "unsupported_plan_version"
OUTCOME_INVALID_PLAN = "invalid_plan"
OUTCOME_UNRESOLVED = "unresolved_grounding"
OUTCOME_MISSING_INPUT = "missing_external_input"
OUTCOME_UNSUPPORTED_CAPABILITY = "unsupported_backend_capability"
OUTCOME_NO_DATA = "no_graph_data"
OUTCOME_JOIN_FAILURE = "join_failure"
OUTCOME_FILTERED_OUT = "filters_removed_all_candidates"
OUTCOME_TRUNCATED = "truncated_or_timed_out"
OUTCOME_BACKEND = "backend_failure"

KNOWN_OUTCOMES = frozenset({
    OUTCOME_RESULTS, OUTCOME_REFUSED, OUTCOME_UNSUPPORTED_VERSION,
    OUTCOME_INVALID_PLAN, OUTCOME_UNRESOLVED, OUTCOME_MISSING_INPUT,
    OUTCOME_UNSUPPORTED_CAPABILITY, OUTCOME_NO_DATA, OUTCOME_JOIN_FAILURE,
    OUTCOME_FILTERED_OUT, OUTCOME_TRUNCATED, OUTCOME_BACKEND,
})

VERDICT_SUCCESS = "success"
VERDICT_NO_ANSWER = "no_answer"
VERDICT_INCONCLUSIVE = "inconclusive"
VERDICT_UNEXECUTABLE = "unexecutable"
VERDICT_REFUSED = "refused"
VERDICT_ERROR = "error"


# ---------------------------------------------------------------------------
# Relaxation
# ---------------------------------------------------------------------------

#: Relaxation axes the executor suggests, ordered tightest to loosest. The
#: order matters: the loop prefers the smallest change that could work, so an
#: answer found after relaxing can be attributed to a specific loosened
#: constraint rather than to a generally vaguer question.
AXIS_ORDER: Tuple[str, ...] = (
    "qualifiers",        # drop a qualifier constraint on a hop
    "constraints",       # drop an entity-level constraint
    "predicate_expansion",  # allow descendants instead of self_only
    "predicate",         # widen to a broader relation
    "category",          # widen an entity's Biolink category
)

#: Names the executor uses for an axis, mapped to the name this package does.
#:
#: The executor calls the entity-level one `entity_constraint`; everything
#: here calls it `constraints`. That divergence was silent and it was not
#: cosmetic: `axis_is_locked` matches on the name, so an axis arriving under
#: the other spelling was never recognised as locked, sorted last instead of
#: second, and had no scope patterns for the one-axis check. The loop would
#: have offered to drop a constraint the *user* asked for — "approved drugs
#: only" is the only kind the plan contract carries — which is the one thing
#: relaxation must never do.
#:
#: It had never fired because no plan in any run so far carried an entity
#: constraint. This is exactly the coupling the mirrored outcome vocabulary
#: warns about, in a place nothing was watching.
AXIS_ALIASES: Dict[str, str] = {
    "entity_constraint": "constraints",
    "entity_constraints": "constraints",
    "qualifier": "qualifiers",
}


def canonical_axis(name: str) -> str:
    """The name this package knows an axis by."""
    return AXIS_ALIASES.get(str(name), str(name))


@dataclass(frozen=True)
class RelaxationAxis:
    """One loosenable constraint on one path, as the executor described it.

    ``key`` identifies it across iterations. Relaxation is monotone — an axis
    is spent once and never re-tightened — and that guarantee is only as good
    as the identity used to track it, so the key includes the entity when the
    axis is entity-scoped. Two entities on the same path can each be widened;
    the same entity cannot be widened twice.

    The axis name is canonicalised on construction rather than wherever it is
    read. Three separate places match on it — the lock check, the ordering, and
    the one-axis scope patterns — and a normalisation applied at each of them
    is a normalisation that will be forgotten at the fourth.
    """

    path_id: str
    axis: str
    detail: str = ""
    current: Any = None
    entity_ref: Optional[str] = None

    def __post_init__(self) -> None:
        canonical = canonical_axis(self.axis)
        if canonical != self.axis:
            object.__setattr__(self, "axis", canonical)

    @property
    def key(self) -> str:
        if self.entity_ref:
            return f"{self.path_id}:{self.axis}:{self.entity_ref}"
        return f"{self.path_id}:{self.axis}"

    @property
    def rank(self) -> int:
        """Position in AXIS_ORDER; unknown axes sort last."""
        try:
            return AXIS_ORDER.index(self.axis)
        except ValueError:
            return len(AXIS_ORDER)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key, "path_id": self.path_id, "axis": self.axis,
            "detail": self.detail, "current": self.current,
            "entity_ref": self.entity_ref,
        }


# ---------------------------------------------------------------------------
# Diagnosis
# ---------------------------------------------------------------------------


@dataclass
class Diagnosis:
    """A reading of one execution. Facts only; no decision.

    Everything here is copied or counted from the executor's own output. The
    one thing that is not — ``locked_constraints`` — comes from the plan, and
    records which constraints the *user* asked for. The planner's design note
    is explicit that a user-requested evidence policy must not be weakened when
    nothing passes it, and the controller is the second place that rule has to
    hold, since the controller is what would otherwise relax it.
    """

    verdict: str = VERDICT_ERROR
    outcome: str = OUTCOME_BACKEND
    outcome_detail: str = ""
    replannable: bool = False
    num_results: int = 0

    # Repair material
    unresolved_entities: List[str] = field(default_factory=list)
    resolution_alternatives: Dict[str, List[Dict[str, Any]]] = field(default_factory=dict)
    unsupported_hops: List[Dict[str, Any]] = field(default_factory=list)
    plan_validation_errors: List[str] = field(default_factory=list)

    # Relaxation material
    relaxation_axes: List[RelaxationAxis] = field(default_factory=list)

    # Execution-retry material
    incomplete_coverage: List[str] = field(default_factory=list)
    timed_out_paths: List[str] = field(default_factory=list)

    # Quality signals, which is how an outcome of `results` can still fail to
    # be an answer.
    concept_warning: Optional[str] = None
    evidence_availability: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    thin_annotation_paths: List[str] = field(default_factory=list)
    filters_dropped_all: bool = False
    ungrounded_rerank_count: int = 0

    # Anchor risk. An anchor is a concept the question is *about*, and the loop
    # never widens one — so when a pinned anchor was resolved to an identifier
    # whose own Biolink types do not satisfy the category the plan pinned, the
    # query can match nothing for a reason that has nothing to do with the
    # graph. Relaxation cannot reach it, and an empty result then looks exactly
    # like a real absence. These two fields are what lets the policy layer tell
    # the difference.
    anchor_confidence: Dict[str, str] = field(default_factory=dict)
    anchor_category_mismatch: List[Dict[str, Any]] = field(default_factory=list)

    #: Why the anchor-category check could not run, or None when it did.
    #:
    #: An empty `anchor_category_mismatch` means one of two very different
    #: things — every anchor is sound, or nothing was checked — and without
    #: this field they are indistinguishable. The check depends on the Biolink
    #: model from plan-core, which is a soft dependency; when it is missing the
    #: loop is back to reporting absences it cannot justify, and that has to be
    #: visible rather than inferred from a suspiciously clean run.
    anchor_check_unavailable: Optional[str] = None

    # Fixed entities the plan declared and never queried. Read from the plan,
    # not the result — the result cannot show it, because a plan that dropped
    # part of the question executes perfectly and returns real, well-evidenced
    # answers to what is left.
    orphan_entities: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def anchor_uncertain(self) -> bool:
        """Whether any pinned anchor's identity is in doubt."""
        return bool(self.anchor_category_mismatch) or any(
            level == "low" for level in self.anchor_confidence.values()
        )

    @property
    def blocking_anchor_mismatch(self) -> bool:
        """Whether a mismatched anchor is a reason to go round again.

        Only when the run came back empty. The whole argument for treating a
        mismatch as a plan fault is that the node constraint may have matched
        nothing for a reason about the plan rather than the graph — and a run
        that returned results has already shown it matched something.

        This is not hypothetical. Question 6 of the final run resolved both
        anchors, ran, and returned eight candidates, and the loop repaired it
        anyway because the plan had typed TNF as a Protein and the resolver
        returned a Gene. The repair widened the category to
        ``GeneOrGeneProduct``, ARAX holds no ``Drug --[affects]-->
        GeneOrGeneProduct`` edges at all, and the run ended with nothing after
        410 seconds. The answer had been in hand at iteration 2.

        The mismatch is still worth recording and still reaches the reader as
        a caveat. It is just not worth an iteration once there are results.
        """
        return bool(self.anchor_category_mismatch) and not self.num_results

    @property
    def plan_repairable(self) -> bool:
        """Whether a planner could fix what went wrong.

        Wider than the executor's own ``replannable`` flag by two cases, and
        both are cases where only the controller can see the problem.

        An anchor whose resolved identifier does not satisfy the category the
        plan pinned on it, *on a run that returned nothing*. The executor does
        not classify that as a plan fault, and from where it stands it is not
        one — it resolved the name it was given, ran the query it was given,
        and the query returned nothing, which is an ordinary outcome. The
        empty-run condition is carried by `blocking_anchor_mismatch`; see it
        for what happens without it.

        A fixed entity the plan declared and never queried. The executor sees
        an entity it was not asked to use and correctly does nothing about it;
        the run succeeds and returns real results. Only something holding the
        plan against the question can tell that the question got smaller.
        """
        return (
            self.replannable
            or self.blocking_anchor_mismatch
            or bool(self.orphan_entities)
        )

    # Constraints the loop may not touch
    locked_constraints: List[str] = field(default_factory=list)

    # Cost so far, from the executor's ledger
    arax_calls: int = 0
    llm_calls: int = 0
    elapsed_s: float = 0.0

    actionable: List[str] = field(default_factory=list)
    unknown_outcome: bool = False

    def axis_by_key(self, key: str) -> Optional[RelaxationAxis]:
        for axis in self.relaxation_axes:
            if axis.key == key:
                return axis
        return None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "outcome": self.outcome,
            "outcome_detail": self.outcome_detail,
            "replannable": self.replannable,
            "num_results": self.num_results,
            "unresolved_entities": self.unresolved_entities,
            "resolution_alternatives": self.resolution_alternatives,
            "unsupported_hops": self.unsupported_hops,
            "plan_validation_errors": self.plan_validation_errors,
            "relaxation_axes": [a.to_dict() for a in self.relaxation_axes],
            "incomplete_coverage": self.incomplete_coverage,
            "timed_out_paths": self.timed_out_paths,
            "concept_warning": self.concept_warning,
            "evidence_availability": self.evidence_availability,
            "thin_annotation_paths": self.thin_annotation_paths,
            "filters_dropped_all": self.filters_dropped_all,
            "ungrounded_rerank_count": self.ungrounded_rerank_count,
            "anchor_confidence": self.anchor_confidence,
            "anchor_category_mismatch": self.anchor_category_mismatch,
            "anchor_check_unavailable": self.anchor_check_unavailable,
            "orphan_entities": self.orphan_entities,
            "locked_constraints": self.locked_constraints,
            "cost": {
                "arax_calls": self.arax_calls,
                "llm_calls": self.llm_calls,
                "elapsed_s": self.elapsed_s,
            },
            "actionable": self.actionable,
            "unknown_outcome": self.unknown_outcome,
        }


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------

#: Execution settings the loop may change between runs of the same plan. Each
#: maps to a run_plan.py option. The set is closed and every member loosens:
#: there is no override here that could make a run stricter, which is what lets
#: `retry_execution` be safe to take without re-deciding whether the plan is
#: still the right plan.
EXECUTION_KNOBS: Dict[str, str] = {
    "timeout": "--timeout",
    "skip_direct_over_hops": "--skip-direct-over-hops",
    "max_results": "--max-results",
    "batch_size": "--batch-size",
}


@dataclass
class Decision:
    """One move, with the parameters that move needs.

    ``source`` records who chose. ``overridden_from`` records what they chose
    when the policy layer refused it. Both are written to the trace: a loop
    where the model is routinely overridden is a loop whose prompt or whose
    legal-action reporting needs work, and that is only visible if the
    overrides are kept rather than quietly corrected.
    """

    action: str
    rationale: str = ""
    source: str = "policy"           # "llm" | "policy" | "forced"
    overridden_from: Optional[str] = None
    override_reason: Optional[str] = None

    # action == RELAX_PLAN
    relaxation_key: Optional[str] = None

    # action == REPAIR_PLAN
    repair_focus: List[str] = field(default_factory=list)

    # action == RETRY_EXECUTION
    execution_overrides: Dict[str, Any] = field(default_factory=dict)

    expected_change: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "action": self.action,
            "rationale": self.rationale,
            "source": self.source,
            "overridden_from": self.overridden_from,
            "override_reason": self.override_reason,
            "relaxation_key": self.relaxation_key,
            "repair_focus": self.repair_focus,
            "execution_overrides": self.execution_overrides,
            "expected_change": self.expected_change,
        }


# ---------------------------------------------------------------------------
# Budget and state
# ---------------------------------------------------------------------------


@dataclass
class Budget:
    """Limits the loop enforces on itself.

    Wall clock and iteration count bound the loop. The ARAX and LLM call
    ceilings bound its cost, and they are checkable because the executor
    reports both in its ledger — a budget the loop cannot measure is a
    comment, not a limit.
    """

    max_iterations: int = 4
    max_planner_calls: int = 3
    max_wall_clock_s: float = 1800.0
    max_arax_calls: int = 200
    max_llm_calls: int = 400

    def to_dict(self) -> Dict[str, Any]:
        return {
            "max_iterations": self.max_iterations,
            "max_planner_calls": self.max_planner_calls,
            "max_wall_clock_s": self.max_wall_clock_s,
            "max_arax_calls": self.max_arax_calls,
            "max_llm_calls": self.max_llm_calls,
        }


@dataclass
class Attempt:
    """One iteration: the plan that ran, what came back, and what was decided."""

    index: int
    plan_fingerprint: str
    plan_path: Optional[str] = None
    result_path: Optional[str] = None
    hints_path: Optional[str] = None
    execution_overrides: Dict[str, Any] = field(default_factory=dict)
    diagnosis: Optional[Diagnosis] = None
    decision: Optional[Decision] = None
    planner_note: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "plan_fingerprint": self.plan_fingerprint,
            "plan_path": self.plan_path,
            "result_path": self.result_path,
            "hints_path": self.hints_path,
            "execution_overrides": self.execution_overrides,
            "diagnosis": self.diagnosis.to_dict() if self.diagnosis else None,
            "decision": self.decision.to_dict() if self.decision else None,
            "planner_note": self.planner_note,
            "elapsed_s": round(self.elapsed_s, 2),
        }


@dataclass
class LoopState:
    """What has been spent and what has been tried.

    The two sets are the termination argument. ``tried_fingerprints`` makes
    re-running an equivalent plan illegal; ``spent_axes`` makes re-loosening a
    constraint illegal. Every iteration must therefore either add to one of
    them or change an execution knob, and all three are finite.
    """

    question: str = ""
    budget: Budget = field(default_factory=Budget)
    started_at: float = field(default_factory=time.time)

    attempts: List[Attempt] = field(default_factory=list)
    tried_fingerprints: Set[str] = field(default_factory=set)
    spent_axes: Set[str] = field(default_factory=set)
    execution_overrides: Dict[str, Any] = field(default_factory=dict)

    #: Relaxations the planner carried out by changing more than the one
    #: constraint it was asked to loosen, after being told once and asked
    #: again. The plan is still used — it is valid, and it differs from
    #: everything already tried — but the loop's usual claim about it is not
    #: available any more: if results arrive, they cannot be attributed to the
    #: single constraint that was named. Recorded here rather than swallowed,
    #: because that claim is the entire reason relaxation is one axis at a time,
    #: and an answer that quietly loses it reads exactly like one that kept it.
    over_relaxations: List[Dict[str, Any]] = field(default_factory=list)

    planner_calls: int = 0
    arax_calls: int = 0

    # Model calls, counted per component rather than in one bucket.
    #
    # Only the first of these used to be counted, which made `max_llm_calls` a
    # ceiling on the executor's spend rather than on the loop's. A run that
    # asked the decision maker twice per iteration and the planner twice per
    # revision was under-reporting by most of its model traffic, so the budget
    # said one thing and the bill said another.
    #
    # Kept separate rather than summed on the way in, because they answer
    # different questions: a high controller count means the decision prompt is
    # being rejected and retried, a high planner count means revisions are not
    # validating first time, and a high executor count is ordinary work.
    executor_llm_calls: int = 0
    controller_llm_calls: int = 0
    planner_llm_calls: int = 0

    @property
    def llm_calls(self) -> int:
        """Every model call this run is responsible for."""
        return (
            self.executor_llm_calls
            + self.controller_llm_calls
            + self.planner_llm_calls
        )

    @property
    def iteration(self) -> int:
        return len(self.attempts)

    @property
    def elapsed_s(self) -> float:
        return time.time() - self.started_at

    def iterations_remaining(self) -> int:
        return max(0, self.budget.max_iterations - self.iteration)

    def budget_exhausted(self) -> Optional[str]:
        """Which limit is spent, or None. Reported by name so the trace says
        which ceiling ended the loop rather than that some ceiling did."""
        b = self.budget
        if self.iteration >= b.max_iterations:
            return f"max_iterations ({b.max_iterations})"
        if self.elapsed_s >= b.max_wall_clock_s:
            return f"max_wall_clock_s ({b.max_wall_clock_s:.0f}s)"
        if self.arax_calls >= b.max_arax_calls:
            return f"max_arax_calls ({b.max_arax_calls})"
        if self.llm_calls >= b.max_llm_calls:
            return f"max_llm_calls ({b.max_llm_calls})"
        return None

    def planner_budget_exhausted(self) -> bool:
        return self.planner_calls >= self.budget.max_planner_calls

    def record(self, attempt: Attempt) -> None:
        attempt.elapsed_s = time.time() - attempt.started_at
        self.attempts.append(attempt)
        self.tried_fingerprints.add(attempt.plan_fingerprint)
        if attempt.diagnosis:
            self.arax_calls += attempt.diagnosis.arax_calls
            self.executor_llm_calls += attempt.diagnosis.llm_calls

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "budget": self.budget.to_dict(),
            "iterations": self.iteration,
            "elapsed_s": round(self.elapsed_s, 2),
            "planner_calls": self.planner_calls,
            "arax_calls": self.arax_calls,
            "llm_calls": self.llm_calls,
            "llm_calls_by_component": {
                "executor": self.executor_llm_calls,
                "controller": self.controller_llm_calls,
                "planner": self.planner_llm_calls,
            },
            "tried_fingerprints": sorted(self.tried_fingerprints),
            "spent_axes": sorted(self.spent_axes),
            "execution_overrides": self.execution_overrides,
            "over_relaxations": self.over_relaxations,
            "attempts": [a.to_dict() for a in self.attempts],
        }


# ---------------------------------------------------------------------------
# Loop outcome
# ---------------------------------------------------------------------------

LOOP_ANSWERED = "answered"
LOOP_ABSENCE = "absence"
LOOP_REFUSED = "refused"
LOOP_EXHAUSTED = "exhausted"
LOOP_FAILED = "failed"


@dataclass
class LoopOutcome:
    """How the loop ended, and with what."""

    status: str
    reason: str = ""
    final_result: Optional[Dict[str, Any]] = None
    final_plan: Optional[Dict[str, Any]] = None
    answer: Optional[Dict[str, Any]] = None
    state: Optional[LoopState] = None

    @property
    def ok(self) -> bool:
        return self.status in (LOOP_ANSWERED, LOOP_ABSENCE)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": CONTROLLER_SCHEMA_VERSION,
            "status": self.status,
            "reason": self.reason,
            "answer": self.answer,
            "state": self.state.to_dict() if self.state else None,
        }
