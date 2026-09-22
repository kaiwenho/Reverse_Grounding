"""Reading a result document.

Diagnosis is transcription plus two things it is allowed to add: quality
signals, and the locked constraints read off the plan. The tests below are
mostly about the transcription being faithful — including when the hints file
is absent, which is the case a controller reading a result straight off disk
hits.
"""

from __future__ import annotations

from loop_controller.contracts import (
    OUTCOME_FILTERED_OUT, OUTCOME_NO_DATA, OUTCOME_UNRESOLVED,
)
from loop_controller.diagnose import check_outcome_drift, diagnose, summary_line

from helpers import (
    make_candidate, make_path, make_plan, make_resolution, make_result,
    relaxation_axes,
)


def test_reads_the_executor_classification():
    result = make_result(
        candidates=[make_candidate()], detail="1 candidate(s) returned",
    )
    diag = diagnose(result, {}, make_plan())
    assert diag.outcome == "results"
    assert diag.verdict == "success"
    assert diag.num_results == 1
    assert diag.replannable is False


def test_works_without_a_hints_file():
    """The hints file is a shortcut, never the only source.

    Everything in it is re-derived from the result document, so a controller
    handed only `--out` behaves identically to one handed both.
    """
    result = make_result(
        outcome=OUTCOME_NO_DATA,
        verdict="no_answer",
        paths={"P1": make_path(
            verdict="no_answer", relaxations=relaxation_axes(), num_candidates=0,
        )},
        resolution=make_resolution(resolved=False),
    )
    with_hints = diagnose(result, {
        "suggested_relaxations": {"P1": relaxation_axes()},
        "unresolved_entities": ["disease"],
    }, make_plan())
    without_hints = diagnose(result, None, make_plan())

    assert [a.key for a in with_hints.relaxation_axes] == \
           [a.key for a in without_hints.relaxation_axes]
    assert with_hints.unresolved_entities == without_hints.unresolved_entities == ["disease"]


def test_relaxation_axes_come_back_tightest_first():
    result = make_result(
        outcome=OUTCOME_NO_DATA, verdict="no_answer",
        paths={"P1": make_path(relaxations=relaxation_axes(), num_candidates=0)},
    )
    diag = diagnose(result, None, make_plan())
    assert [a.axis for a in diag.relaxation_axes] == [
        "qualifiers", "predicate", "category", "category",
    ]


def test_entity_scoped_axes_get_distinct_keys():
    """Two entities on one path can each be widened; one cannot be widened
    twice. That distinction lives in the key."""
    result = make_result(
        outcome=OUTCOME_NO_DATA, verdict="no_answer",
        paths={"P1": make_path(relaxations=relaxation_axes(), num_candidates=0)},
    )
    diag = diagnose(result, None, make_plan())
    keys = [a.key for a in diag.relaxation_axes]
    assert "P1:category:candidate_drug" in keys
    assert "P1:category:disease" in keys
    assert len(set(keys)) == len(keys)


def test_unresolved_entities_carry_the_resolver_menu():
    result = make_result(
        outcome=OUTCOME_UNRESOLVED, replannable=True, verdict="unexecutable",
        resolution=make_resolution(resolved=False, considered=5),
    )
    diag = diagnose(result, None, make_plan())
    assert diag.unresolved_entities == ["disease"]
    assert len(diag.resolution_alternatives["disease"]) == 5
    assert diag.resolution_alternatives["disease"][0]["curie"] == "MONDO:1"


def test_a_low_confidence_resolution_is_repair_material_too():
    result = make_result(
        candidates=[make_candidate()],
        resolution=make_resolution(resolved=True, confidence="low"),
    )
    diag = diagnose(result, None, make_plan())
    assert "disease" in diag.resolution_alternatives


def test_unsupported_hops_carry_the_supported_alternatives():
    result = make_result(
        outcome="unsupported_backend_capability", replannable=True,
        verdict="unexecutable",
        plan_assessment={
            "blocked_paths": {
                "P1": "ARAX cannot answer ChemicalEntity -biolink:ameliorates-> "
                      "Disease; it supports biolink:treats, "
                      "biolink:treats_or_applied_or_studied_to_treat",
            },
            "issues": [],
        },
    )
    diag = diagnose(result, None, make_plan())
    hop = diag.unsupported_hops[0]
    assert hop["path_id"] == "P1"
    assert "biolink:treats" in hop["supported_alternatives"]
    assert "biolink:ameliorates" in hop["supported_alternatives"]


def test_concept_warning_is_carried_through():
    result = make_result(
        candidates=[make_candidate()],
        concept_warning="1 concept check(s) did not confirm the intended entity",
    )
    diag = diagnose(result, None, make_plan())
    assert diag.concept_warning
    assert "CONCEPT WARNING" in summary_line(diag)


def test_thin_annotation_is_flagged_against_the_real_distribution():
    result = make_result(
        candidates=[make_candidate()],
        evidence={"distribution_pre_filter": {
            "P1": {"pct_knowledge_level_provided": 4.0, "pct_with_publications": 0.0},
            "P2": {"pct_knowledge_level_provided": 88.0, "pct_with_publications": 60.0},
        }},
    )
    diag = diagnose(result, None, make_plan())
    assert diag.thin_annotation_paths == ["P1"]


def test_ungrounded_reranks_are_counted():
    result = make_result(candidates=[
        make_candidate("CHEBI:1", rank=1, rerank_reason="grounded", rerank_grounded=True),
        make_candidate("CHEBI:2", rank=2, rerank_reason="recalled", rerank_grounded=False),
    ])
    diag = diagnose(result, None, make_plan())
    assert diag.ungrounded_rerank_count == 1


def test_timeouts_are_separated_from_empty_results():
    """These license different moves, so they must not be collapsed."""
    result = make_result(
        outcome="truncated_or_timed_out", verdict="inconclusive",
        paths={"P1": make_path(
            verdict="inconclusive", direct_outcome="timeout",
            coverage_complete=False, num_candidates=0,
        )},
    )
    diag = diagnose(result, None, make_plan())
    assert diag.timed_out_paths == ["P1"]
    assert diag.incomplete_coverage == ["P1"]
    assert diag.relaxation_axes == []


def test_cost_comes_from_the_ledger():
    result = make_result(
        candidates=[make_candidate()],
        ledger={"arax": {"live_calls": 17}, "llm": {"total_calls": 42}},
    )
    diag = diagnose(result, None, make_plan())
    assert diag.arax_calls == 17
    assert diag.llm_calls == 42


def test_filters_that_removed_everything_are_recognised():
    result = make_result(
        outcome=OUTCOME_FILTERED_OUT, verdict="no_answer",
        evidence={"filter_accounting": {
            "P1": {
                "candidates_kept": 0, "instances_dropped": 12,
                "instance_drop_reasons": {"min_publications": 12},
            },
        }},
    )
    diag = diagnose(result, None, make_plan())
    assert diag.filters_dropped_all is True


def test_an_unknown_outcome_is_reported_rather_than_guessed_at():
    result = make_result(outcome="something_new_upstream")
    diag = diagnose(result, None, make_plan())
    assert diag.unknown_outcome is True
    assert "something_new_upstream" in (check_outcome_drift(diag) or "")


def test_locked_constraints_come_from_the_plan():
    plan = make_plan(evidence_policy={
        "origin": "user_requested", "application": "filter",
        "rationale": "the user asked for published evidence",
        "min_publications": 1,
    })
    diag = diagnose(make_result(plan=plan), None, plan)
    assert "evidence_policy:min_publications" in diag.locked_constraints
    assert "anchor:disease" in diag.locked_constraints


# ---------------------------------------------------------------------------
# Anchor risk
# ---------------------------------------------------------------------------
#
# The one case where an empty result can be a plan fault rather than a fact
# about the graph, and the loop has no move that would find out — anchors are
# never widened.


def test_a_confident_anchor_produces_no_mismatch():
    diag = diagnose(make_result(resolution=make_resolution()))

    assert diag.anchor_category_mismatch == []
    assert diag.anchor_confidence == {"disease": "high"}
    assert not diag.anchor_uncertain


def test_an_anchor_typed_outside_its_pinned_category_is_flagged(requires_biolink):
    """Pinned as a Disease, resolved to something the model calls a
    PhenotypicFeature. The node constraint sent to ARAX cannot match anything,
    whatever the graph holds."""
    diag = diagnose(make_result(resolution=make_resolution(
        types=["biolink:PhenotypicFeature"], expected_category="Disease",
    )))

    assert len(diag.anchor_category_mismatch) == 1
    mismatch = diag.anchor_category_mismatch[0]
    assert mismatch["entity_ref"] == "disease"
    assert mismatch["curie"] == "MONDO:1"
    assert mismatch["expected_category"] == "Disease"
    assert diag.anchor_uncertain


def test_an_anchor_is_not_flagged_for_satisfying_an_ancestor_category(requires_biolink):
    """The comparison goes through the Biolink hierarchy, not string equality.

    A plan that pinned a broad category is answered by any descendant of it,
    and an equality test would report most correct resolutions as mismatched —
    which would suppress every true absence.
    """
    diag = diagnose(make_result(resolution=make_resolution(
        types=["biolink:Disease"], expected_category="BiologicalEntity",
    )))

    assert diag.anchor_category_mismatch == []


def test_an_anchor_with_no_reported_types_is_not_flagged():
    """Absence of evidence. The resolver reported nothing to compare against,
    which is not the same as reporting a conflict."""
    diag = diagnose(make_result(resolution=make_resolution(types=[])))

    assert diag.anchor_category_mismatch == []


def test_a_low_confidence_anchor_is_recorded_without_being_a_mismatch():
    diag = diagnose(make_result(resolution=make_resolution(confidence="low")))

    assert diag.anchor_confidence == {"disease": "low"}
    assert diag.anchor_category_mismatch == []
    assert diag.anchor_uncertain


def test_an_anchor_mismatch_makes_the_plan_repairable(requires_biolink):
    """Wider than the executor's own flag, and deliberately so: only the
    controller holds both the plan's category and the resolver's types."""
    result = make_result(
        outcome=OUTCOME_NO_DATA, verdict="no_answer",
        resolution=make_resolution(types=["biolink:PhenotypicFeature"]),
    )
    diag = diagnose(result)

    assert not diag.replannable
    assert diag.plan_repairable


# --- what the resolver was allowed to accept ------------------------------
#
# The check compares the resolver's types against the plan's category, and for
# a long time that was the whole story. Then gene/protein conflation was
# switched on in the executor and the two components stopped agreeing about
# what a category means: the resolver returns a Gene for an entity the plan
# typed Protein, on purpose, because the graph does not separate them — and
# this check called it broken. So the resolver now states the set it accepted
# and the check reads it.


def test_a_sibling_category_the_resolver_accepted_is_not_a_mismatch(requires_biolink):
    """The live case. TNF, typed `Protein` by the planner, resolved to
    NCBIGene:7124 under gene/protein conflation. Correct, and the run returned
    eight candidates — but before this the loop flagged it and repaired it."""
    diag = diagnose(make_result(resolution=make_resolution(
        types=["biolink:Gene", "biolink:GeneOrGeneProduct"],
        expected_category="Protein",
        accepted_categories=["Gene", "GeneOrGeneProduct", "Protein"],
    )))

    assert diag.anchor_category_mismatch == []


def test_a_category_outside_the_accepted_set_is_still_a_mismatch(requires_biolink):
    """Widening the accepted set is not the same as switching the check off.
    A PhenotypicFeature does not become acceptable because Gene and Protein
    were conflated for some other entity."""
    diag = diagnose(make_result(resolution=make_resolution(
        types=["biolink:PhenotypicFeature"],
        expected_category="Protein",
        accepted_categories=["Gene", "GeneOrGeneProduct", "Protein"],
    )))

    assert len(diag.anchor_category_mismatch) == 1
    assert diag.anchor_category_mismatch[0]["accepted_categories"] == [
        "Gene", "GeneOrGeneProduct", "Protein",
    ]


def test_a_result_without_the_field_falls_back_to_the_plans_category(requires_biolink):
    """Result documents written before the executor recorded this exist on
    disk and in the fixtures. They have to keep diagnosing the way they did."""
    diag = diagnose(make_result(resolution=make_resolution(
        types=["biolink:PhenotypicFeature"], expected_category="Disease",
        accepted_categories=None,
    )))

    assert len(diag.anchor_category_mismatch) == 1
    assert diag.anchor_category_mismatch[0]["expected_category"] == "Disease"


def test_an_empty_accepted_set_falls_back_rather_than_accepting_everything(
    requires_biolink,
):
    """An empty list is not "anything goes". Read the other way it would
    silently disable the check for any entity the resolver skipped."""
    diag = diagnose(make_result(resolution=make_resolution(
        types=["biolink:PhenotypicFeature"], expected_category="Disease",
        accepted_categories=[],
    )))

    assert len(diag.anchor_category_mismatch) == 1


# --- a mismatch on a run that returned something ---------------------------


def test_a_mismatch_does_not_make_a_successful_run_repairable(requires_biolink):
    """Question 6 of the final run, and the reason `blocking_anchor_mismatch`
    exists.

    Every argument for treating a mismatch as a plan fault is about a query
    that matched nothing: the node constraint may have excluded everything for
    a reason about the plan rather than the graph. Eight candidates came back.
    The premise is disproved by the same document that carries the mismatch.

    What the loop did instead was repair, widen the category to
    `GeneOrGeneProduct`, discover ARAX holds no `Drug --[affects]-->
    GeneOrGeneProduct` edges, spend the planner budget and abandon — 410
    seconds to replace an answer with nothing.
    """
    result = make_result(
        candidates=[make_candidate()],
        resolution=make_resolution(
            types=["biolink:PhenotypicFeature"], expected_category="Disease",
        ),
    )
    diag = diagnose(result)

    assert diag.num_results == 1
    # Still recorded, and still reaches the reader: the plan's label was wrong.
    assert len(diag.anchor_category_mismatch) == 1
    assert diag.anchor_uncertain
    # Just not a reason to go round again.
    assert not diag.blocking_anchor_mismatch
    assert not diag.plan_repairable


def test_a_mismatch_on_an_empty_run_is_still_repairable(requires_biolink):
    """The case the check was built for, unchanged."""
    result = make_result(
        outcome=OUTCOME_NO_DATA, verdict="no_answer",
        resolution=make_resolution(types=["biolink:PhenotypicFeature"]),
    )
    diag = diagnose(result)

    assert diag.num_results == 0
    assert diag.blocking_anchor_mismatch
    assert diag.plan_repairable


def test_an_orphan_entity_still_makes_a_successful_run_repairable():
    """The neighbouring case, deliberately left alone. An unused entity means
    the question got smaller, which the results cannot show and no reader can
    detect — so it is worth an iteration even with results in hand. The policy
    layer refuses `accept` for it explicitly rather than by ordering."""
    plan = make_plan()
    plan["entities"].append({
        "entity_ref": "stranded", "name": "psoriasis",
        "biolink_category": "Disease", "is_variable": False,
    })
    diag = diagnose(
        make_result(candidates=[make_candidate()]), {}, plan,
    )

    assert diag.num_results == 1
    assert diag.orphan_entities
    assert diag.plan_repairable


def test_a_variable_entity_is_never_an_anchor():
    diag = diagnose(make_result(resolution=make_resolution()))

    assert "candidate_drug" not in diag.anchor_confidence


def test_an_unavailable_biolink_model_is_reported_rather_than_assumed_away():
    """An empty mismatch list means one of two very different things.

    Every anchor is sound, or nothing was checked. The first version of the
    soft import returned a bare None on any failure, so a run without plan-core
    looked exactly like a clean one: no mismatches, no warning, absences
    reported that the loop had no basis for. It cost nine cryptic test failures
    to notice, and in production it would have cost nothing — which is worse.
    """
    from loop_controller.diagnose import biolink_check_status

    test, reason = biolink_check_status()
    # Whichever way this environment is set up, the two must agree: a missing
    # test always carries a reason, and a present one never does.
    assert (test is None) == (reason is not None)

    diag = diagnose(make_result(resolution=make_resolution()))
    assert diag.anchor_check_unavailable == reason
    assert diag.to_dict()["anchor_check_unavailable"] == reason


def test_the_summary_line_says_when_the_anchor_check_did_not_run(monkeypatch):
    """Visible in the log, not only in the trace."""
    # `loop_controller.diagnose` is re-exported as the *function*, which
    # shadows the submodule of the same name on the package.
    import sys

    module = sys.modules["loop_controller.diagnose"]

    monkeypatch.setattr(
        module, "biolink_check_status",
        lambda: (None, "plan-core is not importable (test)"),
    )
    diag = diagnose(make_result(resolution=make_resolution()))

    assert diag.anchor_check_unavailable
    assert "ANCHOR CHECK OFF" in summary_line(diag)
