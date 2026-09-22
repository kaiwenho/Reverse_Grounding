"""
diagnose.py — Read one execution. Decide nothing.

The executor already classified what happened; this module restates that
classification in the vocabulary the loop decides in, and gathers the material
each possible move would need. It adds no analysis of its own, for the same
reason `aggregate.py` says it only combines and formats: a second opinion
computed here could disagree with the executor's, and then the loop would be
acting on a reading the result document does not support.

Two things here are more than transcription, and both are deliberate.

**Quality signals.** `outcome: results` is not the same as "this is an answer".
A run can return twenty-five ranked candidates while the concept check failed,
or while every supporting edge came from a path where almost nothing declares a
knowledge level. The executor reports both, prominently, but it reports them as
facts rather than as objections. The loop needs them as objections, because
accepting is a move and a move needs grounds. So they are collected here under
their own names and the policy layer decides what they license.

**Locked constraints.** These come from the plan, not the result. They record
which constraints the user asked for, and they exist because relaxation is the
one move that could silently answer a different question than the one asked.
"""

from __future__ import annotations

import re
from functools import lru_cache
from typing import Any, Dict, List, Optional

from .contracts import (
    Diagnosis, KNOWN_OUTCOMES, OUTCOME_BACKEND, OUTCOME_FILTERED_OUT,
    RelaxationAxis, VERDICT_ERROR,
)
from .fingerprint import locked_constraints, orphan_entities


#: Below this share of edges declaring a knowledge level, a knowledge-level
#: threshold filters on annotation completeness rather than on evidence
#: quality. The executor's own hint uses the same figure; it is repeated rather
#: than parsed out of the hint text.
THIN_ANNOTATION_PCT = 30.0


def diagnose(
    result: Dict[str, Any],
    hints: Optional[Dict[str, Any]] = None,
    plan: Optional[Any] = None,
) -> Diagnosis:
    """Build a Diagnosis from an executor result document.

    ``hints`` is the executor's own convenience view. It is used where it is
    present and re-derived from ``result`` where it is not, so a controller
    reading a result document straight off disk behaves identically to one
    reading the pair. The hints file is a shortcut, never the only source.
    """
    hints = hints or {}
    outcome_block = result.get("outcome") or {}
    outcome = outcome_block.get("outcome") or hints.get("outcome") or OUTCOME_BACKEND

    diag = Diagnosis(
        verdict=result.get("verdict") or hints.get("verdict") or VERDICT_ERROR,
        outcome=outcome,
        outcome_detail=outcome_block.get("detail", ""),
        replannable=bool(
            outcome_block.get("replannable", hints.get("replannable", False))
        ),
        num_results=len(result.get("results") or []),
        actionable=list(hints.get("actionable") or []),
        unknown_outcome=outcome not in KNOWN_OUTCOMES,
    )

    _read_repair_material(result, hints, diag)
    _read_relaxation_axes(result, hints, diag)
    _read_execution_material(result, hints, diag)
    _read_quality_signals(result, hints, diag)
    _read_anchor_risk(result, diag)
    _read_cost(result, diag)

    if plan is not None:
        diag.locked_constraints = locked_constraints(plan)
        diag.orphan_entities = orphan_entities(plan)

    if outcome == OUTCOME_FILTERED_OUT:
        diag.filters_dropped_all = True

    return diag


# ---------------------------------------------------------------------------
# Repair material: what a planner would need to fix the plan
# ---------------------------------------------------------------------------


def _read_repair_material(
    result: Dict[str, Any], hints: Dict[str, Any], diag: Diagnosis,
) -> None:
    """Collect the grounded menus.

    Two of the three repairable outcomes come with one. A name that did not
    resolve has the resolver's ranked candidate list — the executor kept what
    it considered, precisely so a wrong or failed choice can be revisited
    against the alternatives rather than guessed at again. A hop the backend
    cannot answer has the meta knowledge graph's message, which names the
    relationships ARAX does support for those categories.

    Passing those menus to the planner turns "try again" into "choose from
    these", which is the difference between a repair that converges and a
    retry that resamples.
    """
    resolution = result.get("resolution") or {}

    unresolved = list(hints.get("unresolved_entities") or [])
    if not unresolved:
        unresolved = [
            ref for ref, r in resolution.items()
            if not r.get("is_variable") and not r.get("resolved_curies")
        ]
    diag.unresolved_entities = unresolved

    for ref in unresolved:
        record = resolution.get(ref) or {}
        considered = record.get("considered") or []
        diag.resolution_alternatives[ref] = [
            {
                "curie": c.get("curie"),
                "label": c.get("label"),
                "types": c.get("types") or [],
            }
            for c in considered[:10]
        ]

    # A resolution that succeeded but landed on a low-confidence choice is
    # repair material too: the plan may need a more specific surface name.
    for ref, record in resolution.items():
        if record.get("is_variable") or ref in diag.resolution_alternatives:
            continue
        if str(record.get("confidence") or "").lower() == "low":
            diag.resolution_alternatives[ref] = [
                {
                    "curie": c.get("curie"),
                    "label": c.get("label"),
                    "types": c.get("types") or [],
                }
                for c in (record.get("considered") or [])[:10]
            ]

    assessment = result.get("plan_assessment") or {}
    for path_id, message in (assessment.get("blocked_paths") or {}).items():
        diag.unsupported_hops.append({
            "path_id": path_id,
            "message": message,
            "supported_alternatives": _parse_supported(message),
        })

    for issue in assessment.get("issues") or []:
        code = issue.get("code")
        if code in {"schema_invalid", "plan_core_missing", "validation_unavailable"}:
            diag.plan_validation_errors.append(
                f"{issue.get('target') or issue.get('scope') or 'plan'}: "
                f"{issue.get('message')}"
            )


_PREDICATE_RE = re.compile(r"biolink:[a-z_0-9]+")


def _parse_supported(message: str) -> List[str]:
    """Pull Biolink predicates out of a meta-KG rejection message.

    The message is prose written for a human reader, so this is a convenience
    extraction rather than a parse: it gives the planner a short list to look
    at first. The full message travels alongside it, and the planner is
    expected to read that — a predicate this misses is still available there,
    and a predicate it wrongly includes is still checked by plan-core before
    anything runs.
    """
    found = _PREDICATE_RE.findall(message or "")
    seen, out = set(), []
    for pred in found:
        if pred not in seen:
            seen.add(pred)
            out.append(pred)
    return out[:20]


# ---------------------------------------------------------------------------
# Relaxation material
# ---------------------------------------------------------------------------


def _read_relaxation_axes(
    result: Dict[str, Any], hints: Dict[str, Any], diag: Diagnosis,
) -> None:
    """Collect loosenable constraints, tightest first.

    The executor emits these only for a path that ran to completion and found
    nothing — an empty result on incomplete coverage is not evidence that the
    constraints were too tight, and suggesting relaxation there would trade a
    slow query for a vague one. That condition is the executor's to apply and
    it has applied it, so an axis appearing here already means the path was
    genuinely empty.
    """
    source = hints.get("suggested_relaxations")
    if source is None:
        source = {
            pid: path.get("suggested_relaxations") or []
            for pid, path in (result.get("paths") or {}).items()
            if path.get("suggested_relaxations")
        }

    axes: List[RelaxationAxis] = []
    for path_id, suggestions in (source or {}).items():
        for item in suggestions or []:
            if not isinstance(item, dict):
                continue
            axes.append(RelaxationAxis(
                path_id=path_id,
                axis=str(item.get("axis") or "unknown"),
                detail=str(item.get("detail") or ""),
                current=item.get("current"),
                entity_ref=item.get("entity_ref"),
            ))

    axes.sort(key=lambda a: (a.rank, a.path_id, a.entity_ref or ""))
    diag.relaxation_axes = axes


# ---------------------------------------------------------------------------
# Execution material
# ---------------------------------------------------------------------------


def _read_execution_material(
    result: Dict[str, Any], hints: Dict[str, Any], diag: Diagnosis,
) -> None:
    """Which paths did not finish, and which timed out.

    Kept apart from the empty-result paths because they license a different
    move. A path that timed out proves nothing about the graph, so re-running
    it with a longer timeout or forced decomposition is progress; relaxing its
    constraints would be guessing at a problem that has not been shown to
    exist.
    """
    paths = result.get("paths") or {}

    incomplete = list(hints.get("incomplete_coverage") or [])
    if not incomplete:
        incomplete = [
            pid for pid, path in paths.items()
            if not path.get("coverage_complete", True)
        ]
    diag.incomplete_coverage = incomplete

    timed_out: List[str] = []
    for pid, path in paths.items():
        markers = " ".join(str(x) for x in (
            path.get("direct_outcome"), path.get("failure_kind"),
            path.get("verdict"), path.get("error"),
        ) if x)
        if "timeout" in markers.lower() or "timed_out" in markers.lower():
            timed_out.append(pid)
    diag.timed_out_paths = timed_out


# ---------------------------------------------------------------------------
# Quality signals
# ---------------------------------------------------------------------------


def _read_quality_signals(
    result: Dict[str, Any], hints: Dict[str, Any], diag: Diagnosis,
) -> None:
    """Reasons a run with results might still not be an answer.

    Three, in descending severity.

    A failed concept check means the results may be about a related but
    different concept. The executor puts this at the top level of the result
    for exactly that reason, and it is the one signal that can make a
    well-ranked list of twenty-five candidates worthless.

    Thin knowledge-level annotation means the evidence metadata the ranking
    leans on is mostly absent. It does not invalidate the results; it bounds
    how much the ranking can be trusted, and it belongs in the answer's
    caveats whether or not it changes the decision.

    Ungrounded reranks are counted because the executor's rule is that a
    rerank reason must cite retrieved edges, and it falls back to the
    deterministic order when one does not. A high count means the model was
    reaching, which is worth knowing even though the executor already
    neutralised it.
    """
    diag.concept_warning = result.get("concept_warning")

    availability = hints.get("evidence_availability")
    if availability is None:
        distribution = (result.get("evidence") or {}).get("distribution_pre_filter") or {}
        availability = {
            pid: {
                "pct_knowledge_level_provided": d.get("pct_knowledge_level_provided"),
                "pct_with_publications": d.get("pct_with_publications"),
                "knowledge_level": d.get("knowledge_level"),
            }
            for pid, d in distribution.items()
        }
    diag.evidence_availability = availability or {}

    diag.thin_annotation_paths = [
        pid for pid, d in diag.evidence_availability.items()
        if (d.get("pct_knowledge_level_provided") or 0) < THIN_ANNOTATION_PCT
    ]

    ungrounded = 0
    for candidate in result.get("results") or []:
        if candidate.get("rerank_reason") and not candidate.get("rerank_grounded"):
            ungrounded += 1
    diag.ungrounded_rerank_count = ungrounded

    accounting = (result.get("evidence") or {}).get("filter_accounting") or {}
    if accounting and not diag.num_results:
        dropped_all = all(
            (acc.get("candidates_kept") or 0) == 0
            and (acc.get("instances_dropped") or 0) > 0
            for acc in accounting.values()
        )
        if dropped_all:
            diag.filters_dropped_all = True


# ---------------------------------------------------------------------------
# Anchor risk
# ---------------------------------------------------------------------------


def _read_anchor_risk(result: Dict[str, Any], diag: Diagnosis) -> None:
    """Whether the concepts the question is about were pinned confidently.

    This exists for one specific way the loop can be confidently wrong.

    An anchor is never widened — that rule is deliberate and it is what stops
    the loop drifting to a different question. But it has a cost. If the plan
    pinned an anchor to a category the resolved identifier does not actually
    satisfy — asking for a ``Disease`` and landing on something the resolver
    types only as a ``PhenotypicFeature`` — then the node constraint sent to
    ARAX can match nothing at all, for a reason that is about the plan rather
    than about the graph. Relaxation cannot touch it, because the axis is
    locked. And the empty result that comes back is indistinguishable, at the
    level the rest of the loop reads, from a real absence.

    So the mismatch is detected here, from data the executor already reports:
    the categories it considered for each candidate, against the category the
    plan asked for. The comparison goes through the Biolink model rather than
    string equality, because a category satisfies its ancestors — a ``Disease``
    answers a plan that asked for a ``BiologicalEntity`` — and an equality test
    would report most correct resolutions as mismatches.

    When the model cannot be loaded the check does not run, and says so in
    ``diag.anchor_check_unavailable``. A degraded check that guessed at
    ancestry would produce false mismatches, and a false mismatch suppresses a
    true absence — the failure this exists to prevent, in the other direction.

    Two limits on what a mismatch may then be used for, both learned the hard
    way on one live question.

    The comparison is against the categories the *resolver* accepted, not the
    single category the plan wrote down. Those differ whenever conflation is
    active, and a check that ignores the difference reports every gene/protein
    anchor in the system as broken.

    And a mismatch is only grounds for repair when the run came back empty.
    Every sentence above is about a query that matched nothing; a run holding
    results has disproved the premise, whatever the plan's category said. The
    mismatch is still recorded and still reaches the answer as a caveat — the
    plan's label really was wrong — but ``Diagnosis.plan_repairable`` stops
    counting it, so results in hand are not traded for another iteration.
    """
    satisfies, unavailable = biolink_check_status()
    diag.anchor_check_unavailable = unavailable

    for ref, record in (result.get("resolution") or {}).items():
        if not isinstance(record, dict) or record.get("is_variable"):
            continue
        curies = [str(c) for c in record.get("resolved_curies") or []]
        if not curies:
            # Already an unresolved-grounding repair; nothing to add.
            continue

        diag.anchor_confidence[ref] = str(
            record.get("confidence") or "unknown"
        ).lower()

        expected = record.get("expected_category")
        if not expected or satisfies is None:
            continue

        # What the resolver was allowed to accept, which is not always the
        # one category the plan named. Under gene/protein conflation it may
        # deliberately return a Gene for an entity the plan typed Protein —
        # the graph does not separate the two, and the identifier is right.
        # Recomputing that judgement here would need the conflation flags and
        # the group table, neither of which the controller has, so the
        # resolver states its own answer and this reads it. A result document
        # written before that field existed falls back to the plan's category,
        # which is the behaviour this check has always had.
        accepted = [
            str(c) for c in (record.get("accepted_categories") or []) if c
        ] or [str(expected)]

        types_by_curie = {
            str(c.get("curie")): [str(t) for t in c.get("types") or []]
            for c in record.get("considered") or []
            if c.get("curie")
        }
        for curie in curies:
            types = types_by_curie.get(curie)
            if not types:
                # The resolver reported no types for this identifier, which is
                # not evidence of a mismatch — only an absence of evidence.
                continue
            if any(satisfies(t, acc) for t in types for acc in accepted):
                continue
            diag.anchor_category_mismatch.append({
                "entity_ref": ref,
                "curie": curie,
                "expected_category": str(expected),
                "accepted_categories": accepted,
                "reported_types": types,
            })


@lru_cache(maxsize=1)
def biolink_check_status() -> tuple:
    """``(test, reason)`` — the Biolink ancestry test, or why there isn't one.

    plan-core is a soft dependency here for the same reason it is in the
    planner port: the controller reads result documents and must not fail to
    diagnose one because a sibling package moved.

    But soft has to mean *reported*, not *silent*. The first version of this
    returned a bare ``None`` on any exception, which made a missing dependency
    indistinguishable from a clean bill of health: every anchor came back
    unflagged, the loop happily reported absences it should have refused, and
    nothing anywhere said the check had not run. It cost nine confusing test
    failures to notice, and in a live run it would have cost nothing at all —
    which is worse, because nobody would have noticed.

    So the reason travels with the result, into the Diagnosis and out through
    the trace.

    Cached because loading the model parses a large pinned YAML and diagnosis
    runs once per iteration.
    """
    try:
        from plan_core import load_biolink_vocabulary
    except Exception as exc:
        return None, (
            f"plan-core is not importable ({type(exc).__name__}: {exc}); "
            f"the anchor-category check needs the Biolink model"
        )
    try:
        vocab = load_biolink_vocabulary()
    except Exception as exc:
        return None, (
            f"the Biolink model could not be loaded "
            f"({type(exc).__name__}: {exc})"
        )
    return vocab.category_satisfies, None


def _read_cost(result: Dict[str, Any], diag: Diagnosis) -> None:
    """Real spend, from the executor's ledger.

    A budget the loop cannot measure is a comment rather than a limit, so the
    counters come from what the run reported rather than from an estimate of
    what a run costs.
    """
    ledger = result.get("ledger") or {}
    arax = ledger.get("arax") or {}
    llm = ledger.get("llm") or {}
    diag.arax_calls = int(arax.get("live_calls") or 0)
    diag.llm_calls = int(llm.get("total_calls") or 0)
    diag.elapsed_s = float(result.get("elapsed_s") or 0.0)


# ---------------------------------------------------------------------------
# Drift
# ---------------------------------------------------------------------------


def check_outcome_drift(diag: Diagnosis) -> Optional[str]:
    """Warn when the executor reports an outcome this package does not know.

    The outcome vocabulary is mirrored here rather than imported, because the
    executor is a subprocess. That trade buys process isolation and costs a
    coupling the type system no longer checks, so the coupling is checked at
    runtime instead: an unrecognised outcome is treated conservatively by the
    policy layer and named loudly in the trace, rather than falling through a
    branch and looking like a backend failure.
    """
    if diag.unknown_outcome:
        return (
            f"executor reported outcome '{diag.outcome}', which this "
            f"controller version does not know; treating it as non-replannable "
            f"and stopping rather than guessing at the right move"
        )
    return None


def summary_line(diag: Diagnosis) -> str:
    """One line for the log."""
    bits = [f"{diag.verdict}/{diag.outcome}", f"{diag.num_results} result(s)"]
    if diag.replannable:
        bits.append("replannable")
    if diag.relaxation_axes:
        bits.append(f"{len(diag.relaxation_axes)} relaxation axis/axes")
    if diag.incomplete_coverage:
        bits.append(f"incomplete: {','.join(diag.incomplete_coverage)}")
    if diag.concept_warning:
        bits.append("CONCEPT WARNING")
    if diag.anchor_check_unavailable:
        bits.append("ANCHOR CHECK OFF")
    return " · ".join(bits)
