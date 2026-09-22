"""Path quality — a chain of true edges is not automatically a reason.

Every case here is taken from one live run: 25 candidates, 12,113 paths found,
125 kept, 1,092 seconds. Every edge in every one of them is real, so the
grounding gate passes all 125.
"""

from __future__ import annotations

from loop_controller.path_quality import (
    VERDICT_CONTRARY, VERDICT_NEGATED, VERDICT_NO_MECHANISM,
    collect_explanation_paths, judge_path, review_paths,
)


def path(*predicates, edges=None, description=""):
    return {
        "predicates": list(predicates),
        "length": len(predicates),
        "description": description,
        "edges": edges or [{"predicate": p, "negated": False} for p in predicates],
    }


def test_a_mechanistic_chain_is_servable():
    """Resveratrol affects PTGS2, which Sulfasalazine also affects, and
    Sulfasalazine is applied to treat the disease. Weak, but a reason."""
    assert judge_path(path(
        "biolink:affects", "biolink:affects", "biolink:applied_to_treat",
    )).servable


def test_a_chain_ending_in_a_contraindication_is_not_support():
    """41 of 125 kept paths ended this way. Read as a sentence it says another
    drug must not be given to these patients."""
    verdict = judge_path(path(
        "biolink:subclass_of", "biolink:subclass_of",
        "biolink:contraindicated_in",
    ))
    assert not verdict.servable
    assert verdict.verdict == VERDICT_CONTRARY
    assert verdict.predicate == "biolink:contraindicated_in"


def test_a_chain_ending_in_a_phenotype_is_not_support():
    verdict = judge_path(path(
        "biolink:in_clinical_trials_for", "biolink:has_phenotype",
        "biolink:has_phenotype",
    ))
    assert verdict.verdict == VERDICT_CONTRARY


def test_a_purely_taxonomic_chain_explains_nothing():
    """In a graph this dense almost any pair is reachable through subclass_of
    and correlated_with, which is why 12,113 paths existed to be narrowed."""
    verdict = judge_path(path(
        "biolink:subclass_of", "biolink:correlated_with", "biolink:related_to",
    ))
    assert verdict.verdict == VERDICT_NO_MECHANISM


def test_one_mechanistic_link_is_enough_to_keep_a_chain():
    """The rule is `all`, not `any`: a chain that does assert an effect
    somewhere is not dismissed for also containing a correlation."""
    assert judge_path(path(
        "biolink:correlated_with", "biolink:affects", "biolink:treats",
    )).servable


def test_a_negated_edge_anywhere_disqualifies_the_chain():
    verdict = judge_path(path(
        "biolink:affects", "biolink:treats",
        edges=[{"predicate": "biolink:affects", "negated": False},
               {"predicate": "biolink:treats", "negated": True}],
    ))
    assert verdict.verdict == VERDICT_NEGATED


def test_a_contrary_relation_partway_along_is_kept():
    """"Drug causes X, X is treated by Y, Y treats the disease" is a real if
    weak route. Only the relation that *reaches* the destination decides."""
    assert judge_path(path(
        "biolink:has_side_effect", "biolink:affects", "biolink:treats",
    )).servable


def test_the_review_counts_what_it_withheld_and_why():
    review = review_paths([
        path("biolink:affects", "biolink:treats"),
        path("biolink:affects", "biolink:contraindicated_in"),
        path("biolink:subclass_of", "biolink:related_to"),
    ])
    assert len(review.servable) == 1
    assert review.counts == {VERDICT_CONTRARY: 1, VERDICT_NO_MECHANISM: 1}


# ---------------------------------------------------------------------------
# Where the paths live
# ---------------------------------------------------------------------------


def test_paths_attached_to_candidates_are_found():
    """`attach_explanations_to_candidates: true` puts them on the candidates
    and leaves the endpoint map empty. Reading only the map produced an answer
    with no explanation section at all, after 1,092 seconds of pathfinding —
    and nothing reported the silence, because an empty map is also what a run
    with no explanations looks like."""
    result = {
        "evidence": {"explanations": {"paths_by_endpoint": {}}},
        "results": [{
            "curie": "CHEBI:45713", "label": "Resveratrol",
            "explanation_paths": [path("biolink:affects", "biolink:treats")],
        }],
    }
    found = collect_explanation_paths(result)
    assert sum(len(v) for v in found.values()) == 1


def test_paths_under_the_endpoint_map_are_still_found():
    result = {
        "evidence": {"explanations": {"paths_by_endpoint": {
            "MONDO:1": [path("biolink:affects", "biolink:treats")],
        }}},
        "results": [],
    }
    assert sum(len(v) for v in collect_explanation_paths(result).values()) == 1


def test_both_locations_at_once():
    result = {
        "evidence": {"explanations": {"paths_by_endpoint": {
            "MONDO:1": [path("biolink:affects", "biolink:treats")],
        }}},
        "results": [{
            "curie": "CHEBI:1",
            "explanation_paths": [path("biolink:affects", "biolink:treats")],
        }],
    }
    assert sum(len(v) for v in collect_explanation_paths(result).values()) == 2
