#!/usr/bin/env python3
"""
relax_probe.py — Measure the one-axis relaxation rule against the real planner,
with no knowledge graph involved.

    PYTHONPATH=../plan-core/src:src python tools/relax_probe.py

Four attempts to reach `relax_plan` through a live ARAX query produced zero
relaxations. That is not bad luck, it is the shape of the target: relaxation
needs a plan that validates, that the backend's meta knowledge graph supports,
whose entities resolve, that runs to completion, and that returns *nothing*.
"Supported by the backend" and "returns nothing" pull against each other, and
finding their intersection means guessing what ARAX holds.

But the graph was never the thing being measured. The open question is narrow:

    told to loosen exactly one named constraint and change nothing else,
    does gpt-oss:120b do that?

Answering it needs the real planner, the real revision prompt and the real
structural diff. It does not need a real executor — an empty result with
relaxation suggestions is a document, and `ScriptedExecutor` replays one. So
this runs the whole loop with the graph replaced by a fixture and the planner
left real, which costs one model call and no ARAX calls at all.

What this does NOT tell you: whether real ARAX results ever reach the
relaxation path, whether the executor's suggestions look like these in
practice, or anything about the graph. Those still need a live run. What it
does tell you is the thing four live runs failed to.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT.parent

for _src in (
    ROOT / "src",
    REPO / "plan-core" / "src",
    REPO / "planner" / "src",
    REPO / "plan-executor" / "src",
):
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from loop_controller.contracts import Budget, RELAX_PLAN        # noqa: E402
from loop_controller.executor_cli import ScriptedExecutor       # noqa: E402
from loop_controller.fingerprint import (                       # noqa: E402
    keyed_view, relaxation_diff,
)
from loop_controller.loop import ControllerConfig, LoopController  # noqa: E402


QUESTION = "Which small molecules increase the activity of TNF in psoriasis?"


def over_tight_plan() -> Dict[str, Any]:
    """A plausible, valid plan with three loosenable constraints on one hop.

    Deliberately ordinary. The point is to see what the planner does with a
    normal-looking plan and a specific instruction, not to trap it.
    """
    return {
        "plan_id": "RELAX-PROBE",
        "plan_version": "0.10.0",
        "biolink_version": "4.4.3",
        "question": QUESTION,
        "plan_mode": "discovery",
        "interpretation": {
            "archetypes": ["Q1_target_based"],
            "restated_question": "Find small molecules that increase TNF "
                                 "activity, for psoriasis.",
            "intent": "discovery",
        },
        "confidence": {"level": "high", "reasons": ["single archetype"]},
        "entities": [
            {"entity_ref": "candidate_chemical", "name": "any small molecule",
             "biolink_category": "SmallMolecule", "is_variable": True},
            {"entity_ref": "tnf", "name": "TNF", "biolink_category": "Gene",
             "is_variable": False, "taxa": ["NCBITaxon:9606"]},
        ],
        "paths": [{
            "path_id": "P1",
            "rationale": "Small molecules that increase TNF activity.",
            "hops": [{
                "subject_ref": "candidate_chemical",
                "predicate": "biolink:affects",
                "object_ref": "tnf",
                "negated": False,
                "predicate_expansion": "self_only",
                "qualifiers": {
                    "object_aspect_qualifier": "activity",
                    "object_direction_qualifier": "increased",
                },
            }],
            "return_entity_ref": "candidate_chemical",
            "expected_result_category": "SmallMolecule",
            "disabled": False,
        }],
        "ranking": {"candidate_ranking": {
            "explanation_influence": "none",
            "rationale": "Probe plan: evidence only.",
            "discovery": {
                "strategy": "evidence_weighted",
                "criteria": [{
                    "name": "knowledge_level", "direction": "desc",
                    "origin": "planner_recommended", "application": "rank",
                    "scope": "portable",
                    "preferred_values": ["knowledge_assertion", "observation"],
                    "rationale": "Prefer asserted evidence.",
                }],
                "top_k": 25,
            },
        }},
    }


def empty_result(plan: Dict[str, Any]) -> Dict[str, Any]:
    """A completed, empty run with the suggestions the executor would emit.

    Shaped from `plan_executor.executor.suggest_relaxations`: qualifiers when
    the hop carries them, predicate always, predicate_expansion when the hop
    says `self_only`, and a category entry per entity.
    """
    return {
        "schema": "plan-executor-result/0.1.0",
        "elapsed_s": 41.0,
        "plan": {k: plan.get(k) for k in (
            "plan_id", "plan_version", "biolink_version", "question",
            "plan_mode", "interpretation", "confidence",
        )},
        "verdict": "no_answer",
        "verdict_reasons": ["the query ran to completion and matched nothing"],
        "outcome": {
            "outcome": "no_graph_data",
            "detail": "the query ran to completion and matched nothing",
            "replannable": False,
        },
        "results": [],
        "paths": {"P1": {
            "path_id": "P1", "verdict": "no_answer", "mode": "discovery",
            "direct_outcome": "ok", "return_entity_ref": "candidate_chemical",
            "num_instances": 0, "num_candidates": 0,
            "coverage_complete": True, "coverage_notes": [], "warnings": [],
            "submitted_queries": [{
                "stage": "direct",
                "query_graph": {
                    "nodes": {"n0": {"ids": ["NCBIGene:7124"]},
                              "n1": {"categories": ["biolink:SmallMolecule"]}},
                    "edges": {"e0": {"subject": "n1", "object": "n0",
                                     "predicates": ["biolink:affects"]}},
                },
            }],
            "suggested_relaxations": [
                {"axis": "qualifiers",
                 "detail": "hop constrains 2 qualifier(s): "
                           "['object_aspect_qualifier', "
                           "'object_direction_qualifier']",
                 "current": {"object_aspect_qualifier": "activity",
                             "object_direction_qualifier": "increased"}},
                {"axis": "predicate",
                 "detail": "'biolink:affects' may be narrower than the "
                           "knowledge graph records",
                 "current": "biolink:affects"},
                {"axis": "predicate_expansion",
                 "detail": "hop requests self_only; allowing descendants "
                           "would widen the match",
                 "current": "self_only"},
                {"axis": "category",
                 "detail": "entity 'candidate_chemical' is restricted to "
                           "SmallMolecule",
                 "entity_ref": "candidate_chemical",
                 "current": "SmallMolecule"},
                {"axis": "category",
                 "detail": "entity 'tnf' is restricted to Gene",
                 "entity_ref": "tnf", "current": "Gene"},
            ],
            "elapsed_s": 41.0,
        }},
        "resolution": {
            "tnf": {
                "entity_ref": "tnf", "query": "TNF",
                "expected_category": "Gene", "is_variable": False,
                "resolved_curies": ["NCBIGene:7124"], "resolved_label": "TNF",
                "confidence": "high", "llm_used": True,
                "considered": [{"curie": "NCBIGene:7124", "label": "TNF",
                                "types": ["biolink:Gene"], "rank": 1}],
            },
            "candidate_chemical": {
                "entity_ref": "candidate_chemical", "is_variable": True,
                "resolved_curies": [], "considered": [],
            },
        },
        "ledger": {"arax": {"live_calls": 1}, "llm": {"total_calls": 2}},
    }



#: Tightest first, matching `contracts.AXIS_ORDER`. The policy always relaxes
#: the tightest axis the executor offered, so reaching a looser one means the
#: tighter ones must not be on the menu.
_AXIS_ORDER = ("qualifiers", "constraints", "predicate_expansion",
               "predicate", "category")


def _only_from(result: Dict[str, Any], axis: str) -> Dict[str, Any]:
    """Drop suggestions tighter than `axis`, so the policy picks `axis`.

    Not a trick: an executor emits a qualifiers suggestion only for a hop that
    carries qualifiers, and a predicate_expansion one only for `self_only`. A
    plan without those simply offers a shorter menu, and this produces the same
    menu without needing a different plan for each axis.

    Worth doing because the axes differ in difficulty, and the default is the
    easy one. Loosening qualifiers means deleting fields. Loosening a predicate
    or a category means choosing a Biolink replacement, and a model choosing a
    broader category may decide the predicate needs to change with it — the one
    legitimate-looking ripple the whole structural diff exists to catch.
    """
    keep = _AXIS_ORDER[_AXIS_ORDER.index(axis):]
    path = result["paths"]["P1"]
    path["suggested_relaxations"] = [
        s for s in path["suggested_relaxations"] if s["axis"] in keep
    ]
    return result


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--model", default="gpt-oss:120b")
    ap.add_argument("--ollama-host", default=None)
    ap.add_argument(
        "--axis", default="qualifiers",
        choices=["qualifiers", "predicate_expansion", "predicate", "category"],
        help="which axis to relax. The policy always takes the tightest one "
             "offered, so this drops the tighter suggestions to reach the one "
             "you want. It matters: loosening `qualifiers` means deleting "
             "fields and involves no judgement, while `predicate` and "
             "`category` require choosing a Biolink replacement — which is "
             "where a model is most likely to adjust something adjacent too.",
    )
    ap.add_argument("--rounds", type=int, default=1,
                    help="repeat the whole probe; the planner is sampled, so "
                         "one round is an anecdote and five is a rate")
    ap.add_argument("--out", default="runs/relax-probe")
    args = ap.parse_args(argv)

    try:
        from planner_agent import PlannerAgent
        from planner_agent.ollama_client import OllamaGPTOSSClient
        from loop_controller.planner_port import PlannerAgentPort
    except ModuleNotFoundError as exc:
        print(f"no planner available: {exc}", file=sys.stderr)
        return 1

    planner = PlannerAgentPort(PlannerAgent(
        llm=OllamaGPTOSSClient(model=args.model, host=args.ollama_host),
    ))

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows: List[Dict[str, Any]] = []

    for round_index in range(1, args.rounds + 1):
        plan = over_tight_plan()
        result = _only_from(empty_result(plan), args.axis)
        # Empty first, then results: the loop relaxes once and accepts, which
        # is the shortest path through the branch being measured.
        executor = ScriptedExecutor([result, _results_run(plan)])

        controller = LoopController(
            executor=executor, planner=planner,
            config=ControllerConfig(
                budget=Budget(max_iterations=3, max_planner_calls=3),
                runs_dir=str(out), trace_path=str(out / f"trace{round_index}.json"),
                verbose=True,
            ),
        )
        outcome = controller.run(QUESTION, plan=plan)
        rows.append(_read_round(round_index, outcome, plan, executor, controller))
        print(f"\n--- round {round_index}: {json.dumps(rows[-1], indent=2)}")

    (out / "relax_probe.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8",
    )
    print("\n" + _verdict(rows))
    print(f"\nwritten to {out}/relax_probe.json")
    return 0


def _results_run(plan: Dict[str, Any]) -> Dict[str, Any]:
    """Something to accept after the relaxation, so the loop terminates."""
    result = empty_result(plan)
    result["verdict"] = "success"
    result["outcome"] = {"outcome": "results", "detail": "", "replannable": False}
    result["results"] = [{
        "rank": 1, "deterministic_rank": 1, "curie": "CHEBI:1",
        "label": "a candidate", "categories": ["ChemicalEntity"], "score": 1.0,
        "supporting_paths": ["P1"], "num_supporting_paths": 1,
        "num_instances": 1, "mean_epc": 0.7,
        "evidence_by_path": {"P1": [{
            "bindings": {"candidate_chemical": "CHEBI:1", "tnf": "NCBIGene:7124"},
            "edges": [{
                "subject": "CHEBI:1", "predicate": "biolink:affects",
                "object": "NCBIGene:7124", "primary_source": "infores:ctd",
                "sources": ["infores:ctd"], "knowledge_level": "knowledge_assertion",
                "agent_type": "manual_agent", "publications": [],
                "num_publications": 0, "negated": False, "edge_id": "edge-1",
            }],
        }]},
    }]
    result["paths"]["P1"]["verdict"] = "success"
    result["paths"]["P1"]["num_candidates"] = 1
    result["paths"]["P1"]["suggested_relaxations"] = []
    return result


def _read_round(
    index: int, outcome: Any, plan: Dict[str, Any],
    executor: ScriptedExecutor, controller: Any,
) -> Dict[str, Any]:
    state = outcome.state
    relaxing = [
        a for a in (state.attempts if state else [])
        if a.decision and a.decision.action == RELAX_PLAN
    ]
    row: Dict[str, Any] = {
        "round": index,
        "status": outcome.status,
        "actions": [a.decision.action for a in (state.attempts if state else [])
                    if a.decision],
        "reached_relax_plan": bool(relaxing),
        "axis_requested": relaxing[0].decision.relaxation_key if relaxing else None,
        "planner_note": relaxing[0].planner_note if relaxing else None,
        "rejected_as_over_broad": sum(
            1 for n in controller.trace.notes
            if "rejected over-broad relaxation" in n
        ),
        # The rejected attempt is the informative one — it says which field the
        # planner reached for when it should not have. Keeping only the
        # accepted diff throws that away, which the first version did.
        "rejections": [
            n for n in controller.trace.notes
            if "rejected over-broad relaxation" in n
        ],
        "accepted_over_broad": len(state.over_relaxations) if state else 0,
    }

    # Why a round produced no diff at all. `relax_plan` was chosen and the
    # planner was asked, but nothing executable came back — the revision failed
    # validation, repeated a plan already tried, or was declined. That is a
    # fourth outcome and reporting it as "unaccounted" says nothing.
    note = (row["planner_note"] or "")
    if len(executor.calls) <= 1:
        row["revision_failed"] = (
            "invalid" if "did not validate" in note
            else "planner_refused" if "declined to revise" in note
            else "duplicate" if "differs in what it would query" in note
            else "budget" if "budget spent" in note
            else "no_second_execution"
        )

    # The plan that actually ran second, against the plan that ran first.
    if len(executor.calls) > 1 and relaxing:
        axis = relaxing[0].diagnosis.axis_by_key(relaxing[0].decision.relaxation_key)
        if axis is not None:
            diff = relaxation_diff(plan, executor.calls[1]["plan"], axis)
            row["diff"] = diff.to_dict()
    row["fields_changed"] = (
        sorted(set(keyed_view(plan)) ^ set(keyed_view(executor.calls[1]["plan"])))
        if len(executor.calls) > 1 else []
    )
    return row


def _verdict(rows: List[Dict[str, Any]]) -> str:
    reached = [r for r in rows if r["reached_relax_plan"]]
    if not reached:
        return (
            "NOT MEASURED: no round reached `relax_plan`. The fixture is "
            "supposed to guarantee it, so this means the empty result or its "
            "suggestions no longer match what the policy layer expects — read "
            "the trace's `legality` block for the refusal reason."
        )

    # Mutually exclusive, or five rounds report six outcomes. A round that
    # was rejected and then corrected ends with a clean diff, so testing
    # `ok` alone counts it twice.
    failed = [r for r in reached if r.get("revision_failed")]
    scored = [r for r in reached if not r.get("revision_failed")]
    stuck = [r for r in scored if r["accepted_over_broad"]]
    corrected = [r for r in scored
                 if r["rejected_as_over_broad"] and not r["accepted_over_broad"]]
    clean = [r for r in scored
             if r.get("diff", {}).get("ok") and not r["rejected_as_over_broad"]]

    lines = [
        f"MEASURED over {len(reached)} round(s) that reached `relax_plan`:",
        f"  loosened exactly the requested axis, first try : {len(clean)}",
        f"  came back over-broad and took the correction   : {len(corrected)}",
        f"  stayed over-broad, attribution withdrawn       : {len(stuck)}",
        f"  produced no usable revision at all             : {len(failed)}",
    ]
    if failed:
        from collections import Counter
        why = Counter(r["revision_failed"] for r in failed)
        lines.append(f"    why: {dict(why)}")
        lines.append(
            "    These say nothing about the one-axis rule — the rule never "
            "got a plan to check. A high count here is a revision-prompt "
            "problem: the planner was asked to loosen one constraint and could "
            "not produce a valid plan at all."
        )
        for r in failed[:3]:
            lines.append(f"    - round {r['round']}: {(r['planner_note'] or '')[:160]}")

    if len(clean) + len(corrected) + len(stuck) + len(failed) != len(reached):
        lines.append(
            f"  (unaccounted: "
            f"{len(reached) - len(clean) - len(corrected) - len(stuck) - len(failed)} "
            f"round(s) matched no category — read the rows)"
        )

    if corrected:
        lines.append("  what it reached for and was told to put back:")
        for r in corrected:
            for note in r.get("rejections", []):
                lines.append(f"    - {note[:200]}")

    if stuck:
        off = sorted({
            f for r in stuck for f in r.get("diff", {}).get("off_axis", [])
        })
        lines.append(f"  fields it would not stop changing: {off}")
        lines.append(
            "  That list is what the relaxation instruction has to name "
            "explicitly. A model that keeps rewriting a field is one that was "
            "never told the field was off limits."
        )
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
