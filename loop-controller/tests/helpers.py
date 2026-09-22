"""Fixture builders: result documents and plans shaped like the real ones.

Everything here mirrors `plan-executor`'s output and `plan-core`'s plan
contract closely enough that the code under test cannot tell the difference.
The real example — `plan-executor/examples/gluten_output.json` — is used
directly where it is available (see `test_compose.py`), and these builders
cover the shapes that example does not contain: an empty result with relaxation
suggestions, a timed-out run, a policy that removed everything, a refusal.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional


def make_plan(
    *,
    question: str = "Which drugs treat dermatitis herpetiformis?",
    predicate: str = "biolink:treats",
    answer_category: str = "ChemicalEntity",
    anchor_name: str = "dermatitis herpetiformis",
    evidence_policy: Optional[Dict[str, Any]] = None,
    restated: str = "Find chemicals asserted to treat the disease.",
    plan_id: str = "P-1",
) -> Dict[str, Any]:
    plan: Dict[str, Any] = {
        "plan_id": plan_id,
        "plan_version": "0.10.0",
        "biolink_version": "4.2.0",
        "question": question,
        "plan_mode": "discovery",
        "interpretation": {
            "archetypes": ["Q11_indication_lookup"],
            "restated_question": restated,
            "intent": "identify_candidates",
        },
        "confidence": {"level": "high", "reasons": ["single archetype"]},
        "entities": [
            {
                "entity_ref": "disease",
                "name": anchor_name,
                "biolink_category": "Disease",
                "is_variable": False,
            },
            {
                "entity_ref": "candidate_drug",
                "biolink_category": answer_category,
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
    if evidence_policy:
        plan["evidence_policy"] = evidence_policy
    return plan


def make_edge(
    subject: str = "CHEBI:1",
    predicate: str = "biolink:treats",
    obj: str = "MONDO:1",
    *,
    edge_id: str = "edge-1",
    source: str = "infores:drugcentral",
    knowledge_level: str = "knowledge_assertion",
    publications: Optional[List[str]] = None,
    negated: bool = False,
    literature: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    edge = {
        "subject": subject,
        "predicate": predicate,
        "object": obj,
        "primary_source": source,
        "sources": [source, "infores:dogpark-tier0"],
        "knowledge_level": knowledge_level,
        "agent_type": "manual_agent",
        "publications": publications or [],
        "num_publications": len(publications or []),
        "negated": negated,
        "edge_id": edge_id,
    }
    if literature:
        edge["literature"] = literature
    return edge


def make_candidate(
    curie: str = "CHEBI:1",
    label: str = "dapsone",
    *,
    rank: int = 1,
    edges: Optional[List[Dict[str, Any]]] = None,
    rerank_reason: Optional[str] = None,
    rerank_grounded: bool = True,
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
        "mean_epc": 0.8,
        "evidence_by_path": {
            "P1": [
                {
                    "bindings": {"candidate_drug": curie, "disease": "MONDO:1"},
                    "edges": edges or [make_edge(subject=curie)],
                }
            ]
        },
    }
    if rerank_reason:
        candidate["rerank_reason"] = rerank_reason
        candidate["rerank_grounded"] = rerank_grounded
    return candidate


def make_result(
    *,
    outcome: str = "results",
    detail: str = "",
    replannable: bool = False,
    verdict: str = "success",
    candidates: Optional[List[Dict[str, Any]]] = None,
    paths: Optional[Dict[str, Any]] = None,
    resolution: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
    concept_warning: Optional[str] = None,
    evidence: Optional[Dict[str, Any]] = None,
    plan_assessment: Optional[Dict[str, Any]] = None,
    ledger: Optional[Dict[str, Any]] = None,
    refusal: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    plan = plan or make_plan()
    result: Dict[str, Any] = {
        "schema": "plan-executor-result/0.1.0",
        "elapsed_s": 12.0,
        # Exactly the keys `plan_executor.aggregate` copies into its result,
        # and no others. `evidence_policy` is conspicuously not among them: the
        # executor applies the policy and reports what it removed, but does not
        # echo the policy itself. This fixture used to include it anyway, which
        # made a `filtered_absence` answer look as though it could always name
        # the filters that emptied it. Against a real result it could not.
        "plan": {
            "plan_id": plan.get("plan_id"),
            "plan_version": plan.get("plan_version"),
            "biolink_version": plan.get("biolink_version"),
            "question": plan.get("question"),
            "plan_mode": plan.get("plan_mode"),
            "interpretation": plan.get("interpretation"),
            "confidence": plan.get("confidence"),
            "gaps": plan.get("gaps") or [],
        },
        "verdict": verdict,
        "verdict_reasons": [detail] if detail else [],
        "outcome": {
            "outcome": outcome,
            "detail": detail,
            "replannable": replannable,
        },
        "results": candidates or [],
        "paths": paths if paths is not None else {"P1": make_path()},
        "ledger": ledger or {
            "arax": {"live_calls": 3},
            "llm": {"total_calls": 5},
        },
    }
    if resolution is not None:
        result["resolution"] = resolution
    if concept_warning:
        result["concept_warning"] = concept_warning
    if evidence:
        result["evidence"] = evidence
    if plan_assessment:
        result["plan_assessment"] = plan_assessment
    if refusal:
        result["refusal"] = refusal
    return result


def submitted_query(
    anchor: str = "MONDO:1",
    predicate: str = "biolink:treats",
    category: str = "biolink:ChemicalEntity",
) -> Dict[str, Any]:
    """The TRAPI graph the executor actually sent.

    Real results always carry this and the fixtures originally did not, which
    is how an absence answer naming its own query's predicate got past the
    tests and failed the grounding gate on the first live run.
    """
    return {
        "stage": "direct",
        "query_graph": {
            "nodes": {
                "n0": {"ids": [anchor]},
                "n1": {"categories": [category]},
            },
            "edges": {
                "e0": {"subject": "n1", "object": "n0",
                       "predicates": [predicate]},
            },
        },
    }


def make_path(
    *,
    verdict: str = "success",
    coverage_complete: bool = True,
    relaxations: Optional[List[Dict[str, Any]]] = None,
    direct_outcome: str = "ok",
    num_candidates: int = 1,
    queries: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    return {
        "path_id": "P1",
        "verdict": verdict,
        "mode": "discovery",
        "direct_outcome": direct_outcome,
        "return_entity_ref": "candidate_drug",
        "num_instances": num_candidates,
        "num_candidates": num_candidates,
        "coverage_complete": coverage_complete,
        "coverage_notes": [],
        "suggested_relaxations": relaxations or [],
        "warnings": [],
        "submitted_queries": queries if queries is not None else [],
        "elapsed_s": 4.0,
    }


def relaxation_axes() -> List[Dict[str, Any]]:
    """The shape `executor.suggest_relaxations` produces, tightest first."""
    return [
        {
            "axis": "qualifiers",
            "detail": "hop constrains 1 qualifier(s): ['object_direction']",
            "current": {"object_direction": "increased"},
        },
        {
            "axis": "predicate",
            "detail": "'biolink:treats' may be narrower than the knowledge "
                      "graph records; a broader relation may match",
            "current": "biolink:treats",
        },
        {
            "axis": "category",
            "detail": "entity 'candidate_drug' is restricted to SmallMolecule",
            "entity_ref": "candidate_drug",
            "current": "SmallMolecule",
        },
        {
            "axis": "category",
            "detail": "entity 'disease' is restricted to Disease",
            "entity_ref": "disease",
            "current": "Disease",
        },
    ]


def plan_relaxation_axes() -> List[Dict[str, Any]]:
    """The axes an executor would suggest for `make_plan()` in particular.

    `relaxation_axes()` is the full menu, and it is the right fixture wherever
    the point is the ordering or the locking rules. It is the wrong one for a
    loop test, because the default plan has no qualifiers on its hop and no
    constraints on its entities — so an executor looking at that plan would
    never suggest dropping either.

    The difference used to be harmless. It stopped being harmless when the loop
    began checking that a revision changed the axis it was asked to change: an
    axis naming a field the plan does not have is one no revision can satisfy,
    so every planner reply reads as over-broad. That is a fixture which cannot
    be met, not a controller which is too strict.
    """
    return [a for a in relaxation_axes() if a["axis"] != "qualifiers"]


def make_resolution(
    *,
    resolved: bool = True,
    considered: int = 3,
    confidence: str = "high",
    reason: str = "",
    types: Optional[List[str]] = None,
    expected_category: str = "Disease",
    accepted_categories: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """The executor's record of what each named entity resolved to.

    ``types`` and ``expected_category`` are what the anchor-category check
    reads: the Biolink types the resolver reported for the identifier it chose,
    against the category the plan pinned. Setting them apart — say, a
    ``PhenotypicFeature`` returned for a slot the plan declared ``Disease`` —
    reproduces the case where a query cannot match anything for a reason that
    has nothing to do with the graph.

    ``accepted_categories`` is the resolver's own statement of everything it
    was willing to accept for this entity, which is wider than
    ``expected_category`` whenever conflation is active. Omitting it (None)
    leaves the key out of the record entirely, which is what a result document
    written before the field existed looks like. Passing ``[]`` is different
    and also worth testing: a record that has the key and nothing in it.
    """
    record: Dict[str, Any] = {
        "entity_ref": "disease",
        "query": "dermatitis herpetiformis",
        "expected_category": expected_category,
        "is_variable": False,
        "resolved_curies": ["MONDO:1"] if resolved else [],
        "resolved_label": "dermatitis herpetiformis" if resolved else None,
        "confidence": confidence,
        "llm_used": True,
        "reason": reason,
        "considered": [
            {
                "curie": f"MONDO:{i}",
                "label": f"candidate disease {i}",
                "types": list(types or ["biolink:Disease"]),
                "rank": i,
            }
            for i in range(1, considered + 1)
        ],
    }
    if accepted_categories is not None:
        record["accepted_categories"] = list(accepted_categories)

    return {
        "disease": record,
        "candidate_drug": {
            "entity_ref": "candidate_drug",
            "is_variable": True,
            "resolved_curies": [],
            "considered": [],
        },
    }


class StubLLM:
    """Returns canned strings in order, or raises."""

    def __init__(self, responses, raises: bool = False) -> None:
        self.responses = list(responses)
        self.raises = raises
        self.calls = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append({"system": system, "user": user})
        if self.raises:
            raise RuntimeError("model unreachable")
        if not self.responses:
            return "{}"
        return self.responses.pop(0)
