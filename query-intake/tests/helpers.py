"""Fixtures: result documents and plans, shaped like the executor's real output.

Deliberately duplicated from `loop-controller/tests` rather than imported. The
two packages are separate, their test suites run independently, and a shared
fixture module between them would mean a change made for one package's tests
silently altering the other's. The shapes are small enough that the duplication
is cheaper than the coupling.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def make_plan(
    *,
    question: str = "Which drugs treat dermatitis herpetiformis?",
    anchor_name: str = "dermatitis herpetiformis",
    anchor_curies: Optional[List[str]] = None,
    predicate: str = "biolink:treats",
) -> Dict[str, Any]:
    disease: Dict[str, Any] = {
        "entity_ref": "disease",
        "name": anchor_name,
        "biolink_category": "Disease",
        "is_variable": False,
    }
    if anchor_curies:
        disease["input_binding"] = {"identifiers": list(anchor_curies)}

    return {
        "plan_id": "P-1",
        "plan_version": "0.10.0",
        "biolink_version": "4.4.3",
        "question": question,
        "plan_mode": "discovery",
        "interpretation": {
            "archetypes": ["Q11_indication_lookup"],
            "restated_question": "Find chemicals asserted to treat the disease.",
            "intent": "identify_candidates",
        },
        "confidence": {"level": "high", "reasons": ["single archetype"]},
        "entities": [
            disease,
            {
                "entity_ref": "candidate_drug",
                "biolink_category": "ChemicalEntity",
                "is_variable": True,
            },
        ],
        "paths": [
            {
                "path_id": "P1",
                "return_entity_ref": "candidate_drug",
                "hops": [
                    {
                        "subject_ref": "candidate_drug",
                        "predicate": predicate,
                        "object_ref": "disease",
                    }
                ],
            }
        ],
    }


def make_edge(edge_id: str = "edge-1", curie: str = "CHEBI:1") -> Dict[str, Any]:
    return {
        "subject": curie,
        "predicate": "biolink:treats",
        "object": "MONDO:1",
        "primary_source": "infores:drugcentral",
        "sources": ["infores:drugcentral", "infores:dogpark-tier0"],
        "knowledge_level": "knowledge_assertion",
        "agent_type": "manual_agent",
        "publications": [],
        "num_publications": 0,
        "negated": False,
        "edge_id": edge_id,
    }


def make_candidate(
    curie: str = "CHEBI:1",
    label: str = "dapsone",
    *,
    rank: int = 1,
    rerank_reason: Optional[str] = None,
) -> Dict[str, Any]:
    candidate: Dict[str, Any] = {
        "rank": rank,
        "deterministic_rank": rank,
        "curie": curie,
        "label": label,
        "categories": ["ChemicalEntity"],
        "score": 1.0,
        "supporting_paths": ["P1"],
        "num_supporting_paths": 1,
        "num_instances": 1,
        "evidence_by_path": {
            "P1": [
                {
                    "bindings": {"candidate_drug": curie, "disease": "MONDO:1"},
                    "edges": [make_edge(f"edge-{curie}", curie)],
                }
            ]
        },
    }
    if rerank_reason:
        candidate["rerank_reason"] = rerank_reason
        candidate["rerank_grounded"] = True
    return candidate


def make_result(
    *,
    outcome: str = "results",
    verdict: str = "success",
    detail: str = "",
    candidates: Optional[List[Dict[str, Any]]] = None,
    plan: Optional[Dict[str, Any]] = None,
    paths: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    plan = plan or make_plan()
    return {
        "schema": "plan-executor-result/0.1.0",
        "elapsed_s": 1.0,
        "plan": {
            "plan_id": plan.get("plan_id"),
            "plan_version": plan.get("plan_version"),
            "biolink_version": plan.get("biolink_version"),
            "question": plan.get("question"),
            "plan_mode": plan.get("plan_mode"),
            "interpretation": plan.get("interpretation"),
            "confidence": plan.get("confidence"),
            "gaps": [],
        },
        "verdict": verdict,
        "verdict_reasons": [detail] if detail else [],
        "outcome": {"outcome": outcome, "detail": detail, "replannable": False},
        "results": candidates or [],
        "paths": paths if paths is not None else {"P1": make_path()},
        "resolution": {
            "disease": {
                "entity_ref": "disease",
                "is_variable": False,
                "resolved_curies": ["MONDO:1"],
                "resolved_label": "dermatitis herpetiformis",
                "confidence": "high",
                "llm_used": True,
                "reason": "the model chose this among the resolver's candidates",
                "considered": [],
            },
        },
        "ledger": {"arax": {"live_calls": 2}, "llm": {"total_calls": 3}},
    }


def make_path(*, num_candidates: int = 1) -> Dict[str, Any]:
    return {
        "path_id": "P1",
        "verdict": "success" if num_candidates else "no_answer",
        "mode": "discovery",
        "direct_outcome": "ok",
        "return_entity_ref": "candidate_drug",
        "num_instances": num_candidates,
        "num_candidates": num_candidates,
        "coverage_complete": True,
        "coverage_notes": [],
        "suggested_relaxations": [],
        "warnings": [],
        "submitted_queries": [],
        "elapsed_s": 0.5,
    }


def results_run(n: int = 2) -> Dict[str, Any]:
    return make_result(candidates=[
        make_candidate(f"CHEBI:{i}", f"drug {i}", rank=i) for i in range(1, n + 1)
    ])


def absence_run() -> Dict[str, Any]:
    return make_result(
        outcome="no_graph_data", verdict="no_answer",
        paths={"P1": make_path(num_candidates=0)},
    )
