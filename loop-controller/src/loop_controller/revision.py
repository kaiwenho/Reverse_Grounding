"""
revision.py — Asking the planner for a different plan, and saying why.

The planner already repairs its own plans, but only against *validation*
errors: it knows how to fix a plan that does not conform to the contract. It
has no way to learn that a conforming plan named a hop the backend cannot
answer, resolved an anchor to the wrong disease, or ran to completion against
an empty region of the graph. That information exists only after execution, and
nothing carried it back.

A ``RevisionRequest`` is that carrier. It is deliberately more than "try
again": a resample of the same prompt reproduces the same plan often enough
that a loop built on it spends its budget confirming its first answer. What
makes a revision converge is that it arrives with the failure named in the
executor's own typed vocabulary, with the grounded alternatives attached — the
resolver's ranked candidates, the meta knowledge graph's supported triples —
and with the constraints that may not move stated explicitly rather than left
to be inferred.

The division of labour is the one the executor already assumes. It names the
axis that could be loosened and refuses to name a replacement, because choosing
one "needs the Biolink model and the question's intent, and both belong to the
planner". The controller inherits that refusal: it picks *which* constraint
gives way, and the planner picks what it becomes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .contracts import Diagnosis, RELAX_PLAN, REPAIR_PLAN, RelaxationAxis
from .fingerprint import dependent_patterns


REVISION_SCHEMA_VERSION = "loop-controller-revision/0.1.0"


@dataclass
class RevisionRequest:
    """Everything the planner needs to produce a better plan, and only that."""

    question: str
    prior_plan: Dict[str, Any]
    prior_fingerprint: str
    action: str                       # REPAIR_PLAN | RELAX_PLAN
    attempt_index: int = 1
    attempts_remaining: int = 1

    diagnosis: Optional[Diagnosis] = None
    relaxation: Optional[RelaxationAxis] = None
    repair_focus: List[str] = field(default_factory=list)

    #: Fields the plan contract forces to move with the relaxation. Widening
    #: a return entity's category makes the path's `expected_result_category`
    #: invalid unless it moves too, so telling the planner to "change nothing
    #: else" without naming these asks for a plan that cannot validate.
    #:
    #: It did exactly that. Measured against the live planner, four category
    #: relaxations out of five came back rejected with
    #: "expected category 'SmallMolecule' is incompatible with return entity
    #: 'candidate_chemical' category 'ChemicalEntity'" — the planner obeying an
    #: instruction that contradicted the contract.
    must_also_change: List[str] = field(default_factory=list)

    must_not_change: List[str] = field(default_factory=list)

    #: Anchors whose *concept* is locked but whose category is the thing being
    #: repaired. Without this the two halves of the message contradict each
    #: other: question 6 of the final run was sent
    #: ``repair_focus: ['anchor:target_tnf']`` beside
    #: ``must_not_change: ['anchor:target_tnf']`` — fix this, do not change
    #: this, one anchor, one message — because `_explain_locked` rendered the
    #: lock as "it keeps its surface name *and its category*". The planner
    #: resolved the contradiction the only way left to it, by widening the
    #: category to `GeneOrGeneProduct`, and produced a hop ARAX cannot answer.
    #: Listing the ref here narrows the lock to what is really locked.
    category_open: List[str] = field(default_factory=list)

    forbidden_fingerprints: List[str] = field(default_factory=list)
    available_inputs: List[Dict[str, Any]] = field(default_factory=list)

    #: Set when a previous revision was rejected without being executed —
    #: currently only for producing a plan that would query the same thing.
    #: Named specifically rather than as a generic retry, because "you changed
    #: the wording, not the query" is a correctable complaint and "try again"
    #: is not.
    previous_rejection: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        diag = self.diagnosis
        return {
            "schema": REVISION_SCHEMA_VERSION,
            "question": self.question,
            "prior_fingerprint": self.prior_fingerprint,
            "action": self.action,
            "attempt_index": self.attempt_index,
            "attempts_remaining": self.attempts_remaining,
            "diagnosis": {
                "verdict": diag.verdict,
                "outcome": diag.outcome,
                "detail": diag.outcome_detail,
                "num_results": diag.num_results,
                "unresolved_entities": diag.unresolved_entities,
                "resolution_alternatives": diag.resolution_alternatives,
                "unsupported_hops": diag.unsupported_hops,
                "plan_validation_errors": diag.plan_validation_errors,
                "concept_warning": diag.concept_warning,
                "anchor_category_mismatch": diag.anchor_category_mismatch,
                "orphan_entities": diag.orphan_entities,
            } if diag else None,
            "relaxation": self.relaxation.to_dict() if self.relaxation else None,
            "repair_focus": self.repair_focus,
            "must_also_change": self.must_also_change,
            "must_not_change": self.must_not_change,
            "category_open": self.category_open,
            "forbidden_fingerprints": self.forbidden_fingerprints,
            "available_inputs": self.available_inputs,
            "previous_rejection": self.previous_rejection,
        }


# ---------------------------------------------------------------------------
# Message assembly
# ---------------------------------------------------------------------------

_STRIP_FROM_PRIOR = ("plan_version", "biolink_version")


def build_revision_message(request: RevisionRequest) -> str:
    """Render the request as a user message for the planner's system prompt.

    Written in the planner's existing idiom, and for the same reason its own
    retry message is: the model is being asked to correct a specific plan
    rather than to answer the question again, so it is shown the plan, told
    exactly what went wrong, and told what to preserve. Version fields are
    stripped from the echoed plan because they belong to the orchestrator, and
    a model shown them copies them.
    """
    prior = {
        k: v for k, v in (request.prior_plan or {}).items()
        if k not in _STRIP_FROM_PRIOR
    }

    sections: List[str] = []

    sections.append(
        "A plan you produced for this question was executed against the "
        "knowledge graph and did not succeed. Revise it."
    )

    sections.append(
        "BIOMEDICAL QUESTION (copy only this text into plan.question):\n"
        f"{request.question}"
    )

    if request.available_inputs:
        sections.append(
            "AVAILABLE EXTERNAL INPUTS (authoritative manifest):\n"
            f"{json.dumps(request.available_inputs, indent=2)}\n"
            "Only input_ref values listed in this manifest are actually "
            "supplied. A dataset mentioned only in the question is unavailable."
        )

    sections.append(
        "THE PLAN THAT WAS EXECUTED, with runtime-owned version fields "
        f"removed:\n\n{json.dumps(prior, indent=2)}"
    )

    sections.append(_what_happened(request))

    if request.action == REPAIR_PLAN:
        sections.append(_repair_instruction(request))
    elif request.action == RELAX_PLAN:
        sections.append(_relax_instruction(request))
    else:  # pragma: no cover - guarded by the policy layer
        sections.append(
            "Produce a corrected plan that addresses the failure described above."
        )

    sections.append(_constraints(request))

    if request.previous_rejection:
        sections.append(
            "YOUR PREVIOUS REVISION WAS REJECTED WITHOUT BEING EXECUTED\n"
            f"  {request.previous_rejection}\n"
            "  Change what the plan would query, not how it is described."
        )

    sections.append(
        "Emit a corrected JSON plan and no commentary. Do not emit "
        "`plan_version` or `biolink_version`; the orchestrator supplies them. "
        "If the failure shows the question cannot be grounded in this graph, "
        "emit a refusal instead of a plan that cannot work."
    )

    return "\n\n".join(sections)


def _what_happened(request: RevisionRequest) -> str:
    diag = request.diagnosis
    if diag is None:  # pragma: no cover - the loop always supplies one
        return "WHAT HAPPENED\nThe execution failed."

    lines = [
        "WHAT HAPPENED",
        f"  outcome: {diag.outcome} — {diag.outcome_detail}",
        f"  candidates returned: {diag.num_results}",
    ]
    if diag.concept_warning:
        lines.append(f"  concept check: {diag.concept_warning}")
    return "\n".join(lines)


def _repair_instruction(request: RevisionRequest) -> str:
    diag = request.diagnosis
    lines = ["WHAT TO REPAIR"]

    if diag and diag.plan_validation_errors:
        lines.append("  The plan did not validate:")
        for err in diag.plan_validation_errors[:10]:
            lines.append(f"    - {err}")

    if diag and diag.unresolved_entities:
        lines.append(
            "  These entity names did not resolve to identifiers. The resolver "
            "searched and found nothing usable, so the surface name, the "
            "Biolink category, or both need to change:"
        )
        for ref in diag.unresolved_entities:
            options = (diag.resolution_alternatives or {}).get(ref) or []
            lines.append(f"    - {ref}")
            if options:
                lines.append(
                    "      the resolver considered, and you may name any of "
                    "these concepts by their preferred label:"
                )
                for opt in options[:8]:
                    types = ", ".join((opt.get("types") or [])[:3])
                    lines.append(
                        f"        {opt.get('label')} [{opt.get('curie')}]"
                        + (f" — {types}" if types else "")
                    )

    if diag and diag.orphan_entities:
        lines.append(
            "  The plan declares these entities and then never uses them. No "
            "hop, explanation query or binding references them, so the query "
            "that ran left out part of the question and any results answer "
            "something narrower than what was asked. Add the hop or "
            "explanation query that uses each one. If the relationship is not "
            "stated or implied by the question, use an open explanation_query "
            "between the endpoints rather than guessing a predicate:"
        )
        for orphan in diag.orphan_entities:
            lines.append(
                f"    - {orphan.get('entity_ref')} "
                f"({orphan.get('name') or 'unnamed'}, "
                f"{orphan.get('biolink_category') or 'no category'})"
            )

    if diag and diag.anchor_category_mismatch:
        lines.append(
            "  These anchors were pinned to a Biolink category that the "
            "identifier they resolved to does not satisfy. A node constraint "
            "like this matches nothing whatever the graph contains, so the "
            "empty result says nothing about the data. Either give the entity "
            "the category its identifier actually has, or name a different "
            "concept whose category is the one the question means:"
        )
        for mismatch in diag.anchor_category_mismatch:
            lines.append(
                f"    - {mismatch.get('entity_ref')}: pinned as "
                f"{mismatch.get('expected_category')}, resolved to "
                f"{mismatch.get('curie')}, typed as "
                f"{', '.join(mismatch.get('reported_types') or []) or 'nothing reported'}"
            )

    if diag and diag.unsupported_hops:
        lines.append(
            "  These hops name a triple the backend cannot answer. Replace the "
            "relationship or the categories with a shape it supports, keeping "
            "the biological meaning of the question:"
        )
        for hop in diag.unsupported_hops:
            lines.append(f"    - {hop.get('path_id')}: {hop.get('message')}")
            alternatives = hop.get("supported_alternatives") or []
            if alternatives:
                lines.append(
                    f"      relations it does support here: "
                    f"{', '.join(alternatives[:12])}"
                )

    if request.repair_focus:
        lines.append(f"  The controller asked you to focus on: {request.repair_focus}")

    return "\n".join(lines)


def _relax_instruction(request: RevisionRequest) -> str:
    """The one-axis instruction.

    Exactly one constraint is named, and the planner is asked to loosen that
    one and leave the rest alone. Loosening several at once would usually find
    something faster and would make the finding uninterpretable: nobody could
    say afterwards which constraint had been the obstacle, and the plan that
    produced the answer would no longer be a plan anyone chose.
    """
    axis = request.relaxation
    lines = ["WHAT TO RELAX"]

    if axis is None:  # pragma: no cover - the policy layer requires one
        lines.append("  The plan appears over-constrained. Loosen one constraint.")
        return "\n".join(lines)

    lines.append(
        "  The plan is valid and the query ran to completion, so this is not a "
        "plan error. The graph returned nothing for the query as written, "
        "which suggests one of its constraints is tighter than the data."
    )
    lines.append(
        f"  Loosen exactly this one constraint, on path {axis.path_id}:"
    )
    lines.append(f"    axis: {axis.axis}")
    if axis.entity_ref:
        lines.append(f"    entity: {axis.entity_ref}")
    if axis.current is not None:
        lines.append(f"    currently: {json.dumps(axis.current, default=str)}")
    lines.append(f"    the executor observed: {axis.detail}")

    guidance = _AXIS_GUIDANCE.get(axis.axis)
    if guidance:
        lines.append(f"  {guidance}")

    if request.must_also_change:
        lines.append(
            "  These fields must move with it, because the plan contract "
            "requires them to agree and a plan where they disagree does not "
            "validate. Changing them is part of this one relaxation, not an "
            "extra change:"
        )
        for field_path in request.must_also_change:
            lines.append(f"    {field_path}")

    lines.append(
        "  Change nothing else. Apart from the constraint named above and the "
        "fields listed as moving with it, every entity, hop, predicate, "
        "filter, ranking rule and identifier stays exactly as it is, so that "
        "if the revised plan returns results, this constraint is what was "
        "blocking them."
    )
    return "\n".join(lines)


#: What loosening means for each axis, in the planner's own terms. The
#: controller does not name the replacement — it says what kind of replacement
#: it is asking for, and leaves the Biolink choice to the component that holds
#: the model.
_AXIS_GUIDANCE: Dict[str, str] = {
    "predicate": (
        "Choose a broader active Biolink relation that still expresses what "
        "the question asks — typically an ancestor of the current predicate. "
        "Do not substitute a nearby predicate with a different meaning; if no "
        "broader relation is faithful, record a modelling gap and say so."
    ),
    "predicate_expansion": (
        "Allow descendants of the predicate instead of matching it alone. "
        "This widens the match without changing what the hop means."
    ),
    "qualifiers": (
        "Drop the qualifier constraint, or reduce it to the one qualifier the "
        "question actually requires. A qualified hop matches only edges "
        "carrying that qualifier, and many curated sources do not record them."
    ),
    "category": (
        "Widen this entity's Biolink category to a parent class that still "
        "covers what the question asks for. The answer entity is usually the "
        "safe one to widen; never widen an anchor the question is about."
    ),
    "constraints": (
        "Remove the entity-level constraint only if it was not something the "
        "user asked for. If it records a user request, do not remove it — "
        "report that the request cannot be satisfied instead."
    ),
}


def _constraints(request: RevisionRequest) -> str:
    """The non-negotiables, stated rather than assumed.

    The forbidden fingerprints are the interesting entry. A model asked to fix
    a plan will sometimes return the same plan with different prose, and the
    controller will reject it — but rejecting it after a full execution costs
    an iteration. Saying up front that a revision must differ in what it
    queries converts most of those into a usable plan on the first attempt.
    """
    lines = ["CONSTRAINTS ON THE REVISION"]

    lines.append(
        "  - The revision must differ from the plan above in what it would "
        "query: its entities, paths, hops, filters, ranking or aggregation. A "
        "reworded interpretation or a new confidence reason is not a "
        "revision, and will be rejected without being executed."
    )

    if request.must_not_change:
        lines.append(
            "  - These come from the user's own question and must survive "
            "unchanged:"
        )
        for item in request.must_not_change:
            lines.append(
                f"      {_explain_locked(item, request.category_open)}"
            )

    lines.append(
        "  - Preserve every semantic choice that was not identified as a "
        "problem, including entity and path identifiers, so the revision can "
        "be compared with the original."
    )

    if request.attempts_remaining <= 1:
        lines.append(
            "  - This is the last revision the budget allows. Prefer a plan "
            "that is likely to return something over one that is ideal."
        )

    return "\n".join(lines)


def _explain_locked(item: str, category_open: Sequence[str] = ()) -> str:
    if item.startswith("anchor:"):
        ref = item.split(":", 1)[1]
        if ref in set(category_open):
            # The category is what is being repaired, so it cannot also be
            # locked. Saying so plainly is the whole fix: the planner needs
            # to know the concept is fixed and the label is not.
            return (
                f"{item} — '{ref}' is a concept the question is about; it "
                f"keeps its surface name and its identifier. Its "
                f"biolink_category is the field being repaired, so that one "
                f"field may change and nothing else about this entity may"
            )
        return (
            f"{item} — '{ref}' is a concept the question is about; it keeps its "
            f"surface name and its category"
        )
    if item.startswith("evidence_policy"):
        return (
            f"{item} — a filter the user asked for; it is never weakened, not "
            f"even when it removes every candidate"
        )
    if item.startswith("constraint:"):
        return f"{item} — a user-requested entity constraint"
    return item


# ---------------------------------------------------------------------------
# Construction from loop state
# ---------------------------------------------------------------------------


def build_request(
    *,
    question: str,
    prior_plan: Dict[str, Any],
    prior_fingerprint: str,
    action: str,
    diagnosis: Diagnosis,
    relaxation: Optional[RelaxationAxis] = None,
    repair_focus: Optional[Sequence[str]] = None,
    forbidden_fingerprints: Optional[Sequence[str]] = None,
    available_inputs: Optional[Sequence[dict]] = None,
    attempt_index: int = 1,
    attempts_remaining: int = 1,
    previous_rejection: Optional[str] = None,
) -> RevisionRequest:
    return RevisionRequest(
        question=question,
        prior_plan=prior_plan,
        prior_fingerprint=prior_fingerprint,
        action=action,
        attempt_index=attempt_index,
        attempts_remaining=attempts_remaining,
        diagnosis=diagnosis,
        relaxation=relaxation,
        repair_focus=list(repair_focus or []),
        # Computed from the same function the structural diff uses, so what
        # the planner is told it may change and what the check allows it to
        # change cannot drift apart.
        must_also_change=(
            dependent_patterns(relaxation, prior_plan) if relaxation else []
        ),
        must_not_change=list(diagnosis.locked_constraints),
        # Read from the controller's own mismatch list, not from
        # `repair_focus`: the focus strings may have been written by the
        # decision-maker model, and what a lock means is not a model's to
        # decide.
        category_open=[
            str(m.get("entity_ref"))
            for m in (diagnosis.anchor_category_mismatch or [])
            if m.get("entity_ref")
        ],
        forbidden_fingerprints=list(forbidden_fingerprints or []),
        available_inputs=[dict(i) for i in (available_inputs or [])],
        previous_rejection=previous_rejection,
    )
