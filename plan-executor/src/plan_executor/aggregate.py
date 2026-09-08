"""
aggregate.py — Assemble the final result JSON.

This module only combines and formats. Everything it reports was computed
upstream: the executor produced per-path verdicts, postfilter produced drop
accounting and distributions, evidence produced literature verdicts, rank
produced the ordering. Recomputing any of it here would create a second
opinion that could disagree with the first.

The `aggregation` block
-----------------------
    combine        union | intersection | weighted_union | path_priority_union
    deduplicate_on resolved_id | name | entity_ref
    group_evidence_across_paths
    attach_explanations_to_candidates

`intersection` is the one that changes the answer rather than its order: it
keeps only candidates every path agreed on. That makes coverage decisive — if
one path was `inconclusive`, an empty intersection may mean the paths disagreed
or may mean one of them never finished, and those are different findings. The
result records which.

Two audiences
-------------
The `results` block answers the question. Everything else exists so a
controlling agent can decide what to do when the answer is unsatisfying:

  * per-path verdicts separate `no_answer` (the graph says no) from
    `inconclusive` (the run did not establish anything), and carry
    `suggested_relaxations` for the first case
  * the pre-filter distribution shows what evidence metadata was available, so
    a knowledge-level threshold can be chosen against real numbers rather than
    guessed at
  * filter accounting shows what each evidence rule cost, including how much
    of that cost was edges missing an attribute rather than failing one
  * resolution records what each name was resolved to and what it was chosen
    over, since a wrong anchor produces confident results about the wrong
    concept

Usage
-----
    result = build_result(
        plan_input=plan_input, resolutions=resolutions,
        executions=executions, ranked=ranked, ...
    )
    write_result(result, "runs/result.json")
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set


RESULT_SCHEMA_VERSION = "plan-executor-result/0.1.0"

VERDICT_SUCCESS = "success"
VERDICT_NO_ANSWER = "no_answer"
VERDICT_INCONCLUSIVE = "inconclusive"
VERDICT_UNEXECUTABLE = "unexecutable"
VERDICT_REFUSED = "refused"
VERDICT_ERROR = "error"

#: Plan-level outcomes the handoff asks the executor to distinguish. A caller
#: deciding whether to replan needs to know which of these happened: only a
#: repairable plan error justifies replanning, and zero results on their own
#: justify none of it.
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

#: Which outcomes describe something a planner could repair. Everything else
#: is a fact about the backend or the data, and replanning would produce the
#: same result more slowly.
REPLANNABLE_OUTCOMES = {
    OUTCOME_INVALID_PLAN, OUTCOME_UNRESOLVED, OUTCOME_UNSUPPORTED_CAPABILITY,
}

COMBINE_UNION = "union"
COMBINE_INTERSECTION = "intersection"
COMBINE_WEIGHTED_UNION = "weighted_union"
COMBINE_PATH_PRIORITY_UNION = "path_priority_union"


# ---------------------------------------------------------------------------
# Aggregation settings
# ---------------------------------------------------------------------------


@dataclass
class AggregationSettings:
    combine: str = COMBINE_UNION
    deduplicate_on: str = "resolved_id"
    group_evidence_across_paths: bool = True
    attach_explanations_to_candidates: bool = True

    @classmethod
    def from_plan(cls, plan_raw: Dict[str, Any]) -> "AggregationSettings":
        raw = plan_raw.get("aggregation") or {}
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(exclude_none=True)
        elif not isinstance(raw, dict):
            raw = {k: v for k, v in vars(raw).items() if v is not None}
        return cls(
            combine=raw.get("combine") or COMBINE_UNION,
            deduplicate_on=raw.get("deduplicate_on") or "resolved_id",
            group_evidence_across_paths=bool(
                raw.get("group_evidence_across_paths", True)
            ),
            attach_explanations_to_candidates=bool(
                raw.get("attach_explanations_to_candidates", True)
            ),
        )


# ---------------------------------------------------------------------------
# Combining
# ---------------------------------------------------------------------------


def apply_combine(
    ranked: Sequence[Any],
    executions: Dict[str, Any],
    settings: AggregationSettings,
) -> tuple:
    """Filter the ranked candidates by the plan's combine strategy.

    Returns (kept, notes). Only `intersection` removes anything — the union
    variants differ in how ranking already treated multi-path support, which
    `rank.py` handled through `num_supporting_paths` and the tie-breaker.
    """
    notes: List[str] = []

    if settings.combine != COMBINE_INTERSECTION:
        if settings.combine == COMBINE_PATH_PRIORITY_UNION:
            notes.append(
                "path_priority_union: candidates from all paths retained; ties "
                "resolved toward earlier-listed paths during ranking"
            )
        elif settings.combine == COMBINE_WEIGHTED_UNION:
            notes.append(
                "weighted_union: candidates from all paths retained, weighted "
                "by the plan's ranking criteria"
            )
        return list(ranked), notes

    # Intersection: only candidates every path that could produce them found.
    contributing = [
        pid for pid, ex in executions.items()
        if getattr(ex, "verdict", None) in (VERDICT_SUCCESS, VERDICT_NO_ANSWER)
    ]
    if not contributing:
        notes.append("intersection requested but no path produced a usable result")
        return [], notes

    kept = [c for c in ranked if set(contributing) <= set(c.path_ids)]
    dropped = len(ranked) - len(kept)
    notes.append(
        f"intersection over {len(contributing)} path(s): kept {len(kept)}, "
        f"dropped {dropped} found by only some paths"
    )

    # An empty intersection has two very different causes, and the incomplete
    # one must not be read as disagreement between paths.
    incomplete = [
        pid for pid, ex in executions.items()
        if not getattr(ex, "coverage_complete", True)
    ]
    if not kept and incomplete:
        notes.append(
            f"the empty intersection may be an artefact: {incomplete} did not "
            f"achieve complete coverage, so a shared candidate could have been "
            f"missed rather than absent"
        )
    return kept, notes


def deduplicate(candidates: Sequence[Any], on: str) -> tuple:
    """Collapse candidates that are the same thing under the chosen key.

    `resolved_id` is normally already unique because the executor keys
    candidates by CURIE, so this mainly matters for `name`, where two
    identifiers carrying the same label are merged.
    """
    if on == "resolved_id":
        return list(candidates), []

    seen: Dict[str, Any] = {}
    merged: List[str] = []
    for c in candidates:
        key = (c.label or c.curie).strip().lower() if on == "name" else c.curie
        if key in seen:
            keeper = seen[key]
            keeper.path_ids |= c.path_ids
            keeper.edge_ids |= c.edge_ids
            keeper.instances.extend(c.instances)
            keeper.notes.append(f"merged with {c.curie} on {on}")
            merged.append(f"{c.curie} -> {keeper.curie}")
        else:
            seen[key] = c
    return list(seen.values()), merged


# ---------------------------------------------------------------------------
# Roll-up verdict
# ---------------------------------------------------------------------------


def rollup_verdict(
    plan_input: Any,
    executions: Dict[str, Any],
    num_results: int,
) -> tuple:
    """Reduce per-path verdicts to one plan-level verdict.

    Returns (verdict, reasons). The distinction preserved throughout is
    between the graph having answered "nothing" and the run not having
    established anything: only the first is a finding the planner can act on,
    and reporting the second as the first would assert that no answer exists on
    the strength of a query that never completed.
    """
    reasons: List[str] = []

    if getattr(plan_input, "refused", False):
        refusal = getattr(plan_input, "refusal", None) or {}
        return VERDICT_REFUSED, [
            f"planner refused: {refusal.get('reason')} — {refusal.get('message', '')}"
        ]

    if not executions:
        return VERDICT_UNEXECUTABLE, [
            "no path was executed" + (
                f"; blocked: {list(getattr(plan_input, 'blocked_paths', {}))}"
                if getattr(plan_input, "blocked_paths", None) else ""
            )
        ]

    by_verdict: Dict[str, List[str]] = {}
    for pid, ex in executions.items():
        by_verdict.setdefault(getattr(ex, "verdict", VERDICT_ERROR), []).append(pid)

    for verdict, pids in sorted(by_verdict.items()):
        reasons.append(f"{verdict}: {pids}")

    if num_results:
        return VERDICT_SUCCESS, reasons

    if by_verdict.get(VERDICT_SUCCESS):
        # Paths found results but nothing survived filtering, which is a
        # statement about the evidence policy rather than about the graph.
        reasons.append(
            "paths returned results but none survived filtering or "
            "aggregation; the evidence policy or combine strategy is the "
            "binding constraint, not the knowledge graph"
        )
        return VERDICT_INCONCLUSIVE, reasons

    if by_verdict.get(VERDICT_INCONCLUSIVE) or by_verdict.get(VERDICT_ERROR):
        return VERDICT_INCONCLUSIVE, reasons

    if by_verdict.get(VERDICT_NO_ANSWER):
        complete = all(
            getattr(ex, "coverage_complete", True) for ex in executions.values()
        )
        if complete:
            reasons.append(
                "every path completed and found nothing; the knowledge graph "
                "does not support this question as posed"
            )
            return VERDICT_NO_ANSWER, reasons
        reasons.append(
            "paths found nothing, but coverage was incomplete, so absence is "
            "not established"
        )
        return VERDICT_INCONCLUSIVE, reasons

    if by_verdict.get(VERDICT_UNEXECUTABLE):
        return VERDICT_UNEXECUTABLE, reasons

    return VERDICT_INCONCLUSIVE, reasons


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


def typed_outcome(
    plan_input: Any,
    executions: Dict[str, Any],
    resolutions: Optional[Dict[str, Any]],
    filter_results: Optional[Dict[str, Any]],
    num_results: int,
) -> Dict[str, Any]:
    """Classify what happened, beyond whether there were results.

    Ordered from the earliest stage that could have stopped the run, so the
    first thing that actually went wrong is what gets reported rather than its
    downstream consequence.
    """
    issues = getattr(plan_input, "issues", None) or []
    codes = {getattr(i, "code", None) for i in issues if getattr(i, "blocking", False)}

    def out(kind: str, detail: str) -> Dict[str, Any]:
        return {
            "outcome": kind,
            "detail": detail,
            "replannable": kind in REPLANNABLE_OUTCOMES,
        }

    if getattr(plan_input, "refused", False):
        refusal = getattr(plan_input, "refusal", None) or {}
        return out(OUTCOME_REFUSED,
                   f"the planner declined: {refusal.get('reason')}")

    if "unsupported_plan_version" in codes:
        return out(OUTCOME_UNSUPPORTED_VERSION,
                   "plan targets a contract version this executor does not implement")

    if "missing_external_input" in codes:
        return out(OUTCOME_MISSING_INPUT,
                   "an entity is bound to an external input that was not supplied")

    if codes & {"schema_invalid", "plan_core_missing", "validation_unavailable"}:
        return out(OUTCOME_INVALID_PLAN, "the plan did not validate")

    if "hop_unsupported_by_arax" in codes:
        return out(OUTCOME_UNSUPPORTED_CAPABILITY,
                   "a hop names a triple the backend cannot answer")

    unresolved = [
        ref for ref, r in (resolutions or {}).items()
        if not getattr(r, "is_variable", False) and not getattr(r, "resolved", False)
    ]
    if unresolved and not executions:
        return out(OUTCOME_UNRESOLVED,
                   f"named entities did not resolve: {unresolved}")

    if num_results:
        return out(OUTCOME_RESULTS, f"{num_results} candidate(s) returned")

    # Nothing came back. Which of the remaining causes applies decides whether
    # anything is worth changing, so they are separated rather than collapsed
    # into an empty result.
    if any(getattr(ex, "join_failed", False) for ex in executions.values()):
        joined = {
            pid: ex.hop_counts() for pid, ex in executions.items()
            if getattr(ex, "join_failed", False)
        }
        return dict(out(OUTCOME_JOIN_FAILURE,
                        "every hop returned bindings, but they share no "
                        "intermediate node"),
                    hop_counts=joined)

    dropped_everything = any(
        getattr(fr, "kept_instances", None) == [] and getattr(fr, "dropped_instances", 0)
        for fr in (filter_results or {}).values()
    )
    if dropped_everything:
        return out(OUTCOME_FILTERED_OUT,
                   "paths returned candidates, but filters removed all of them")

    if any(getattr(ex, "failure_kind", None) in ("timeout", "truncation")
           for ex in executions.values()):
        return out(OUTCOME_TRUNCATED,
                   "a query timed out or was truncated, so absence is not established")

    if any(getattr(ex, "failure_kind", None) == "backend_failure"
           for ex in executions.values()):
        return out(OUTCOME_BACKEND, "the backend returned an error")

    return out(OUTCOME_NO_DATA,
               "the backend holds no data for this query as posed; this is a "
               "statement about this backend, version and data snapshot, not "
               "about the biology")


def candidate_output(
    candidate: Any,
    edges: Dict[str, Dict[str, Any]],
    literature: Optional[Dict[str, Any]] = None,
    explanations: Optional[Dict[str, Any]] = None,
    group_evidence: bool = True,
    attach_explanations: bool = True,
    max_edges: int = 20,
) -> Dict[str, Any]:
    """Render one candidate with the evidence that supports it."""
    from .postfilter import summarize_edge

    out = candidate.to_dict()

    if group_evidence:
        # Grouped by path so a reader can see the routes separately: two paths
        # reaching the same answer by different mechanisms is stronger than one
        # path reaching it twice, and a flat edge list hides the difference.
        by_path: Dict[str, List[Dict[str, Any]]] = {}
        for inst in candidate.instances:
            pid = getattr(inst, "path_id", None) or "unknown"
            entry = {
                "bindings": dict(getattr(inst, "bindings", {}) or {}),
                "edges": [],
            }
            for eid in getattr(inst, "edge_ids", []) or []:
                edge = edges.get(eid)
                if not edge:
                    continue
                item = summarize_edge(edge)
                item["edge_id"] = eid
                support = (literature or {}).get(eid)
                if support is not None:
                    item["literature"] = (
                        support if isinstance(support, dict) else support.to_dict()
                    )
                entry["edges"].append(item)
            by_path.setdefault(pid, []).append(entry)
        out["evidence_by_path"] = {
            pid: entries[:max_edges] for pid, entries in by_path.items()
        }
    else:
        evidence = []
        for eid in sorted(candidate.edge_ids)[:max_edges]:
            edge = edges.get(eid)
            if edge:
                item = summarize_edge(edge)
                item["edge_id"] = eid
                evidence.append(item)
        out["evidence"] = evidence

    if attach_explanations and explanations:
        paths = explanations.get(candidate.curie)
        if paths:
            out["explanation_paths"] = paths

    return out


def build_result(
    plan_input: Any,
    executions: Dict[str, Any],
    ranked: Sequence[Any],
    resolutions: Optional[Dict[str, Any]] = None,
    edges: Optional[Dict[str, Dict[str, Any]]] = None,
    filter_results: Optional[Dict[str, Any]] = None,
    distributions: Optional[Dict[str, Any]] = None,
    literature: Optional[Dict[str, Any]] = None,
    literature_summary: Optional[Dict[str, Any]] = None,
    explanations: Optional[Dict[str, Any]] = None,
    concept_checks: Optional[Sequence[Any]] = None,
    ledger: Optional[Dict[str, Any]] = None,
    started_at: Optional[str] = None,
    elapsed_s: float = 0.0,
) -> Dict[str, Any]:
    """Assemble the single output document."""
    raw = getattr(plan_input, "raw", {}) or {}
    settings = AggregationSettings.from_plan(raw)
    edges = edges or {}

    kept, combine_notes = apply_combine(ranked, executions, settings)
    kept, merged = deduplicate(kept, settings.deduplicate_on)
    if merged:
        combine_notes.append(
            f"deduplicated on {settings.deduplicate_on}: {merged[:5]}"
        )

    for i, cand in enumerate(kept, 1):
        cand.rank = i

    verdict, verdict_reasons = rollup_verdict(plan_input, executions, len(kept))
    outcome = typed_outcome(
        plan_input, executions, resolutions, filter_results, len(kept),
    )

    result: Dict[str, Any] = {
        "schema": RESULT_SCHEMA_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "elapsed_s": round(elapsed_s, 2),
        "plan": {
            "plan_id": raw.get("plan_id"),
            "plan_version": raw.get("plan_version"),
            "biolink_version": raw.get("biolink_version"),
            "question": raw.get("question"),
            "plan_mode": raw.get("plan_mode"),
            "interpretation": raw.get("interpretation"),
            "confidence": raw.get("confidence"),
            "gaps": raw.get("gaps") or [],
        },
        "verdict": verdict,
        "verdict_reasons": verdict_reasons,
        "outcome": outcome,
        "aggregation": {
            "combine": settings.combine,
            "deduplicate_on": settings.deduplicate_on,
            "notes": combine_notes,
        },
        "results": [
            candidate_output(
                c, edges, literature=literature, explanations=explanations,
                group_evidence=settings.group_evidence_across_paths,
                attach_explanations=settings.attach_explanations_to_candidates,
            )
            for c in kept
        ],
        "paths": {
            pid: ex.to_dict() if hasattr(ex, "to_dict") else dict(ex)
            for pid, ex in executions.items()
        },
    }

    if getattr(plan_input, "refused", False):
        result["refusal"] = plan_input.refusal

    blocked = getattr(plan_input, "blocked_paths", None)
    issues = getattr(plan_input, "issues", None)
    if blocked or issues:
        result["plan_assessment"] = {
            "blocked_paths": blocked or {},
            "issues": [i.to_dict() for i in (issues or [])],
        }

    if resolutions:
        result["resolution"] = {
            ref: (r.to_dict() if hasattr(r, "to_dict") else dict(r))
            for ref, r in resolutions.items()
        }

    if concept_checks:
        checks = [
            c.to_dict() if hasattr(c, "to_dict") else dict(c) for c in concept_checks
        ]
        result["concept_checks"] = checks
        failed = [c for c in checks if not c.get("ok")]
        if failed:
            # Surfaced at the top level because a failed concept check means
            # the results may be about a different concept than was asked
            # about, which invalidates them regardless of how well ranked they
            # are.
            result["concept_warning"] = (
                f"{len(failed)} concept check(s) did not confirm that results "
                f"concern the intended entity; results may be about a related "
                f"but different concept"
            )

    evidence_block: Dict[str, Any] = {}
    if distributions:
        evidence_block["distribution_pre_filter"] = distributions
    if filter_results:
        evidence_block["filter_accounting"] = {
            pid: (fr.to_dict() if hasattr(fr, "to_dict") else dict(fr))
            for pid, fr in filter_results.items()
        }
    if literature_summary or literature:
        # Summary and per-edge verdicts sit side by side under one key. The
        # per-edge detail was previously reachable only through each
        # candidate's evidence_by_path, which made the verification results
        # awkward to inspect as a whole.
        evidence_block["literature"] = {
            "summary": literature_summary or {},
            "by_edge": literature or {},
        }
    if evidence_block:
        result["evidence"] = evidence_block

    result["ledger"] = ledger or {}

    return result


# ---------------------------------------------------------------------------
# Ledger
# ---------------------------------------------------------------------------


def build_ledger(
    arax_client: Optional[Any] = None,
    cache: Optional[Any] = None,
    resolver: Optional[Any] = None,
    evidence_gatherer: Optional[Any] = None,
    llm_agent: Optional[Any] = None,
    executor_config: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Collect what the run cost and what decided its shape.

    Recorded so a run can be explained after the fact: which ARAX build
    answered, how much came from cache, how many LLM judgements were made and
    how many failed. A result whose LLM calls mostly failed looks the same as
    one where they succeeded, unless this is written down.
    """
    ledger: Dict[str, Any] = {}

    if arax_client is not None and hasattr(arax_client, "stats"):
        ledger["arax"] = arax_client.stats()
    if cache is not None and hasattr(cache, "stats"):
        stats = cache.stats()
        ledger["cache"] = {
            "path": stats.get("path"),
            "entries": stats.get("entries"),
            "stored_bytes": stats.get("stored_bytes"),
            "session": stats.get("session"),
        }
    if resolver is not None and hasattr(resolver, "stats"):
        ledger["resolver"] = resolver.stats()
    if evidence_gatherer is not None and hasattr(evidence_gatherer, "stats"):
        ledger["evidence"] = evidence_gatherer.stats()
    if llm_agent is not None and hasattr(llm_agent, "stats"):
        ledger["llm"] = llm_agent.stats()
    if executor_config:
        ledger["executor_config"] = executor_config
    if extra:
        ledger.update(extra)

    return ledger


# ---------------------------------------------------------------------------
# Controller hints
# ---------------------------------------------------------------------------


def literature_overview(result: Dict[str, Any]) -> Dict[str, Any]:
    """Aggregate the per-edge literature verdicts.

    Two numbers matter most: how the verdicts fell, and how often a claimed
    supporting quote could not be found in the abstract. A high unverified
    rate has two quite different causes — the model paraphrasing rather than
    quoting, or the citation genuinely not supporting the edge, which is
    common for machine-derived edges — so both are reported rather than
    collapsed.
    """
    from collections import Counter

    by_edge = ((result.get("evidence") or {}).get("literature") or {}).get("by_edge") or {}
    verdicts: Counter = Counter()
    quote_ok: Counter = Counter()
    statuses: Counter = Counter()
    unverified_supporting = 0

    for support in by_edge.values():
        statuses[support.get("status", "?")] += 1
        for v in support.get("verdicts") or []:
            verdicts[v.get("verdict", "?")] += 1
            quote_ok[bool(v.get("quote_verified"))] += 1
            if str(v.get("verdict", "")).startswith("supports") and not v.get("quote_verified"):
                unverified_supporting += 1

    total = sum(verdicts.values())
    return {
        "edges_checked": len(by_edge),
        "abstracts_judged": total,
        "verdicts": dict(verdicts.most_common()),
        "edge_status": dict(statuses.most_common()),
        "quote_verified": {"yes": quote_ok[True], "no": quote_ok[False]},
        "supporting_verdicts_discarded_for_unverifiable_quote": unverified_supporting,
        "pct_quote_verified": (
            round(100 * quote_ok[True] / total, 1) if total else None
        ),
    }


def controller_hints(result: Dict[str, Any]) -> Dict[str, Any]:
    """What a controlling agent would need to decide the next iteration.

    A convenience view over material already in the result — no new analysis.
    Its purpose is to put the actionable parts in one place, since a controller
    reading the full document would otherwise have to know where each signal
    lives.
    """
    outcome = result.get("outcome") or {}
    hints: Dict[str, Any] = {
        "verdict": result.get("verdict"),
        "outcome": outcome.get("outcome"),
        "replannable": outcome.get("replannable", False),
        "num_results": len(result.get("results") or []),
        "actionable": [],
    }
    if outcome and not outcome.get("replannable") and not result.get("results"):
        hints["actionable"].append(
            f"{outcome.get('outcome')}: {outcome.get('detail')} — replanning "
            f"would not change this"
        )

    relaxations: Dict[str, Any] = {}
    incomplete: List[str] = []
    for pid, path in (result.get("paths") or {}).items():
        if path.get("suggested_relaxations"):
            relaxations[pid] = path["suggested_relaxations"]
        if not path.get("coverage_complete", True):
            incomplete.append(pid)

    if relaxations:
        hints["suggested_relaxations"] = relaxations
        hints["actionable"].append(
            "one or more paths found nothing on complete coverage; the plan "
            "may be over-constrained (see suggested_relaxations)"
        )

    if incomplete:
        hints["incomplete_coverage"] = incomplete
        hints["actionable"].append(
            "coverage was incomplete on some paths; re-running may complete "
            "them, since cached work is not repeated"
        )

    distribution = (result.get("evidence") or {}).get("distribution_pre_filter")
    if distribution:
        hints["evidence_availability"] = {
            pid: {
                "pct_knowledge_level_provided": d.get("pct_knowledge_level_provided"),
                "pct_with_publications": d.get("pct_with_publications"),
                "knowledge_level": d.get("knowledge_level"),
            }
            for pid, d in distribution.items()
        }
        thin = [
            pid for pid, d in distribution.items()
            if (d.get("pct_knowledge_level_provided") or 0) < 30
        ]
        if thin:
            hints["actionable"].append(
                f"most edges on {thin} declare no knowledge_level; a "
                f"knowledge-level threshold would filter on annotation "
                f"completeness rather than evidence quality"
            )

    for pid, accounting in ((result.get("evidence") or {}).get("filter_accounting") or {}).items():
        for warning in accounting.get("warnings") or []:
            hints["actionable"].append(f"{pid}: {warning}")

    if result.get("concept_warning"):
        hints["actionable"].insert(0, result["concept_warning"])

    unresolved = [
        ref for ref, r in (result.get("resolution") or {}).items()
        if not r.get("is_variable") and not r.get("resolved_curies")
    ]
    if unresolved:
        hints["unresolved_entities"] = unresolved
        hints["actionable"].append(
            f"entities {unresolved} did not resolve; the plan may need a "
            f"different name, alias, or category"
        )

    return hints


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write_result(result: Dict[str, Any], path: str, indent: int = 2) -> str:
    parent = os.path.dirname(os.path.abspath(path))
    if parent:
        os.makedirs(parent, exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=indent, default=str)
    return path


def result_summary_lines(result: Dict[str, Any], limit: int = 10) -> List[str]:
    """Short console view. The JSON remains the deliverable."""
    lines = [
        f"verdict: {result.get('verdict')}  "
        f"results: {len(result.get('results') or [])}  "
        f"{result.get('elapsed_s')}s"
    ]
    for reason in result.get("verdict_reasons") or []:
        lines.append(f"  {reason}")

    for item in (result.get("results") or [])[:limit]:
        cite = item.get("best_citation") or {}
        suffix = f"  [{cite.get('pmid')}]" if cite.get("pmid") else ""

        # When an LLM reordered the list, the deterministic score no longer
        # explains the position. Showing it alone invites the reader to think
        # the ranking is inconsistent, so the score is labelled and the
        # movement and stated reason are shown alongside it.
        reason = item.get("rerank_reason")
        det = item.get("deterministic_rank")
        if reason and det and det != item.get("rank"):
            score_part = f"llm-ranked (was #{det}, score={item.get('score')})"
        elif reason:
            score_part = f"llm-confirmed (score={item.get('score')})"
        else:
            score_part = f"score={item.get('score')}"

        lines.append(
            f"  #{item.get('rank'):>2} {str(item.get('label'))[:40]:40s} "
            f"{score_part} paths={item.get('num_supporting_paths')}{suffix}"
        )
        if reason:
            lines.append(f"        {reason[:110]}")

    if result.get("concept_warning"):
        lines.append(f"  WARNING {result['concept_warning']}")
    return lines
