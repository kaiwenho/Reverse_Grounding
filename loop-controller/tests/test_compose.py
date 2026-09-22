"""Composition and the grounding gate.

These are the tests for the project's hard constraint: answer claims must be
supported by paths whose edges occur in the result graph, and no model-written
text is served. The gate is tested by trying to get things past it.

`test_the_real_example_composes_and_passes` runs against
`plan-executor/examples/gluten_output.json` — a genuine ARAX run with 25
candidates, 47 literature-checked edges, LLM rerank reasons on 20 of them, and
explanation paths. It is the only test here that can catch a mismatch between
what this package assumes a result document looks like and what one is.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from loop_controller.compose import (
    KIND_ABSENCE, KIND_CANDIDATES, KIND_CLARIFICATION, KIND_FILTERED_ABSENCE,
    KIND_INCONCLUSIVE, KIND_REFUSAL, ComposerSettings, compose, render_text,
)
from loop_controller.diagnose import diagnose
from loop_controller.grounding import (
    build_lexicon, check_statement, unquarantined_prose_fields,
)

from helpers import (
    make_candidate, make_edge, make_path, make_plan, make_resolution,
    make_result,
)


EXAMPLE = (
    Path(__file__).resolve().parents[2]
    / "plan-executor" / "examples" / "gluten_output.json"
)


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_a_statement_citing_an_edge_that_is_not_there_is_refused():
    result = make_result(candidates=[make_candidate()])
    lex = build_lexicon(result)
    violations = check_statement(
        {"id": "X", "text": "dapsone treats it.", "supported_by": ["edge-999"]},
        lex,
    )
    assert [v.kind for v in violations] == ["unknown_edge"]


def test_a_statement_naming_an_identifier_that_is_not_there_is_refused():
    result = make_result(candidates=[make_candidate()])
    lex = build_lexicon(result)
    violations = check_statement(
        {"id": "X", "text": "CHEBI:99999 also treats it.", "supported_by": []},
        lex,
    )
    assert [v.kind for v in violations] == ["ungrounded_identifier"]


def test_model_written_text_cannot_be_served_even_when_it_is_accurate():
    """`rerank_reason` is grounded, checked, and still not servable.

    It reads exactly like the one-line justification a candidate list wants,
    which is why it is the field a composer would reach for and why the gate
    has to catch it rather than trust that nobody will.
    """
    reason = (
        "The edge asserts that dapsone treats the condition with the highest "
        "evidence strength and a direct knowledge assertion from a curated source."
    )
    result = make_result(candidates=[
        make_candidate(rerank_reason=reason, rerank_grounded=True),
    ])
    lex = build_lexicon(result)
    assert reason in lex.quarantine

    violations = check_statement(
        {"id": "X", "text": f"1. dapsone — {reason}", "supported_by": []}, lex,
    )
    assert [v.kind for v in violations] == ["model_authored_text"]


def test_the_planners_restated_question_is_quarantined_too():
    """Written by the planner's model, and the most quotable line in the file."""
    restated = (
        "Identify chemical entities with an asserted therapeutic relationship "
        "to the specified dermatological condition."
    )
    plan = make_plan(restated=restated)
    lex = build_lexicon(make_result(plan=plan, candidates=[make_candidate()]))
    assert restated in lex.quarantine

    violations = check_statement(
        {"id": "X", "text": f"Interpreted as: {restated}", "supported_by": []}, lex,
    )
    assert [v.kind for v in violations] == ["model_authored_text"]


def test_an_unverified_quote_is_refused_and_a_verified_one_is_not():
    verified = "Dapsone produced complete remission in 12 of 14 patients."
    edge = make_edge(edge_id="e-lit", literature={
        "verdicts": [{
            "verdict": "supports",
            "quote": verified,
            "quote_verified": True,
            "pmid": "PMID:12345",
            "rationale": "the abstract reports a remission rate for this drug "
                         "in this condition, which supports the edge directly",
        }],
    })
    result = make_result(candidates=[make_candidate(edges=[edge])])
    lex = build_lexicon(result)

    assert check_statement(
        {"id": "A", "text": f'PMID:12345 states: "{verified}"',
         "supported_by": ["e-lit"]}, lex,
    ) == []

    invented = check_statement(
        {"id": "B", "text": 'PMID:12345 states: "Dapsone is first-line therapy."',
         "supported_by": ["e-lit"]}, lex,
    )
    assert [v.kind for v in invented] == ["unverified_quote"]


def test_the_gate_reports_prose_fields_it_does_not_classify():
    """The gate's own failure mode, surfaced rather than assumed away."""
    result = make_result(candidates=[make_candidate()])
    result["results"][0]["some_new_narrative_field"] = (
        "A long free-text field added by a future version of the executor that "
        "this controller has never heard of and would not know to quarantine."
    )
    assert any(
        "some_new_narrative_field" in f
        for f in unquarantined_prose_fields(result)
    )


# ---------------------------------------------------------------------------
# Composition
# ---------------------------------------------------------------------------


def test_candidates_compose_and_every_statement_is_grounded():
    result = make_result(candidates=[
        make_candidate("CHEBI:1", "dapsone", rank=1,
                       rerank_reason="a model wrote this and it must not appear "
                                     "anywhere in the composed answer at all"),
        make_candidate("CHEBI:2", "rituximab", rank=2),
    ])
    answer = compose(result, diagnose(result, None, make_plan()))

    assert answer["answer_kind"] == KIND_CANDIDATES
    assert answer["grounding"]["ok"] is True
    assert answer["grounding"]["failed"] == 0

    text = render_text(answer)
    assert "dapsone" in text and "CHEBI:1" in text
    assert "a model wrote this" not in text


def test_the_candidate_table_omits_model_written_fields():
    """The table is data, but a consumer renders it, so the same rule applies."""
    result = make_result(candidates=[
        make_candidate(rerank_reason="model prose that should never be exported"),
    ])
    answer = compose(result, diagnose(result, None, make_plan()))
    serialised = json.dumps(answer["candidates"])
    assert "rerank_reason" not in serialised
    assert "model prose" not in serialised


def test_identical_triples_from_many_sources_are_one_finding():
    edges = [
        make_edge(edge_id=f"e{i}", source=f"infores:src{i}")
        for i in range(1, 5)
    ]
    result = make_result(candidates=[make_candidate(edges=edges)])
    answer = compose(result, diagnose(result, None, make_plan()))
    candidate_statement = next(
        s for s in answer["statements"] if s["kind"] == "candidate"
    )
    assert candidate_statement["text"].count("—[biolink:treats]→") == 1
    assert len(candidate_statement["supported_by"]) == 4
    for i in range(1, 5):
        assert f"infores:src{i}" in candidate_statement["text"]


def test_a_negated_edge_is_called_out():
    result = make_result(candidates=[
        make_candidate(edges=[make_edge(edge_id="e1", negated=True)]),
    ])
    answer = compose(result, diagnose(result, None, make_plan()))
    assert any("does NOT hold" in s["text"] for s in answer["statements"])


# ---------------------------------------------------------------------------
# Absence, refusal, inconclusive
# ---------------------------------------------------------------------------


def test_an_absence_says_what_was_established():
    result = make_result(
        outcome="no_graph_data", verdict="no_answer",
        paths={"P1": make_path(num_candidates=0)},
        resolution=make_resolution(),
    )
    answer = compose(result, diagnose(result, None, make_plan()))

    assert answer["answer_kind"] == KIND_ABSENCE
    text = render_text(answer)
    assert "statement about the graph, not a failure" in text
    assert "MONDO:1" in text  # the anchor it was established for


def test_an_unfinished_run_is_not_dressed_up_as_an_absence():
    """The most useful-sounding available claim is the one the run cannot make."""
    result = make_result(
        outcome="truncated_or_timed_out", verdict="inconclusive",
        detail="a query did not finish",
        paths={"P1": make_path(direct_outcome="timeout", num_candidates=0)},
    )
    answer = compose(result, diagnose(result, None, make_plan()))

    assert answer["answer_kind"] == KIND_INCONCLUSIVE
    text = render_text(answer)
    assert "No answer was established" in text
    assert "not evidence that the graph lacks the data" in text


def test_a_filtered_absence_makes_the_narrower_claim():
    plan = make_plan(evidence_policy={
        "origin": "user_requested", "application": "filter",
        "rationale": "planner prose restating what the user asked for here",
        "min_publications": 2,
        "min_year": 2015,
    })
    result = make_result(
        outcome="filters_removed_all_candidates", verdict="no_answer", plan=plan,
        paths={"P1": make_path(num_candidates=0)},
        evidence={"filter_accounting": {"P1": {
            "candidates_kept": 0, "instances_dropped": 30,
            "instance_drop_reasons": {"min_publications": 22, "min_year": 8},
        }}},
    )
    answer = compose(result, diagnose(result, None, plan), plan=plan)

    assert answer["answer_kind"] == KIND_FILTERED_ABSENCE
    text = render_text(answer)
    assert "The graph is not empty here" in text
    assert "at least 2 publication(s) per edge" in text
    assert "publications from 2015 or later" in text
    assert "min_publications (22)" in text
    # The policy's own rationale is planner prose and is not reproduced.
    assert "planner prose restating" not in text


def test_a_filtered_absence_needs_the_plan_to_name_its_filters():
    """The executor's result does not echo the evidence policy.

    It applies the policy and reports what the policy removed, which is a
    reasonable place to stop — but it leaves the one answer kind whose entire
    content is *which* filters emptied it unable to name a single one. The
    composer is handed the executed plan for this.
    """
    plan = make_plan(evidence_policy={
        "origin": "user_requested", "application": "filter",
        "min_publications": 2,
    })
    result = make_result(
        outcome="filters_removed_all_candidates", verdict="no_answer", plan=plan,
        paths={"P1": make_path(num_candidates=0)},
        evidence={"filter_accounting": {"P1": {
            "candidates_kept": 0, "instances_dropped": 30,
            "instance_drop_reasons": {"min_publications": 30},
        }}},
    )
    assert "evidence_policy" not in result["plan"]

    without = compose(result, diagnose(result, None, plan))
    assert "at least 2 publication(s)" not in render_text(without)

    with_plan = compose(result, diagnose(result, None, plan), plan=plan)
    assert "at least 2 publication(s)" in render_text(with_plan)
    # Grafted for reading, never written back over the executor's own record.
    assert "evidence_policy" not in result["plan"]


def test_the_grafted_policy_does_not_overwrite_one_the_result_carries():
    """The result's plan block is the executor's record of what it ran.

    If a future executor version starts echoing the policy, that version is
    the authority — overwriting it with what the controller believes it sent
    would turn a record into an assumption.
    """
    plan = make_plan(evidence_policy={
        "origin": "user_requested", "application": "filter",
        "min_publications": 9,
    })
    result = make_result(
        outcome="filters_removed_all_candidates", verdict="no_answer", plan=plan,
        paths={"P1": make_path(num_candidates=0)},
        evidence={"filter_accounting": {"P1": {
            "candidates_kept": 0, "instances_dropped": 4,
            "instance_drop_reasons": {"min_publications": 4},
        }}},
    )
    result["plan"]["evidence_policy"] = {
        "origin": "user_requested", "application": "filter",
        "min_publications": 3,
    }

    text = render_text(compose(result, diagnose(result, None, plan), plan=plan))

    assert "at least 3 publication(s)" in text
    assert "at least 9 publication(s)" not in text


def test_a_filtered_absence_may_name_the_sources_its_filter_excluded():
    """Those sources appear in no returned edge — having removed every one.

    They are reachable only because the lexicon reads the same grafted policy,
    and only for a statement kind that describes the query rather than the
    graph.
    """
    plan = make_plan(evidence_policy={
        "origin": "user_requested", "application": "filter",
        "required_knowledge_sources": ["infores:ctd"],
    })
    result = make_result(
        outcome="filters_removed_all_candidates", verdict="no_answer", plan=plan,
        paths={"P1": make_path(num_candidates=0)},
        evidence={"filter_accounting": {"P1": {
            "candidates_kept": 0, "instances_dropped": 12,
            "instance_drop_reasons": {"required_knowledge_sources": 12},
        }}},
    )
    answer = compose(result, diagnose(result, None, plan), plan=plan)

    assert answer["grounding"]["ok"] is True
    assert "infores:ctd" in render_text(answer)


@pytest.mark.parametrize("reason,expected", [
    ("unsafe_or_clinical_advice", "not a source of clinical guidance"),
    ("needs_clarification", "underspecified"),
    ("out_of_scope", "outside what this system answers"),
])
def test_a_refusal_is_rendered_from_its_typed_reason(reason, expected):
    """Not from the planner's message, which is model-written prose."""
    message = (
        "I cannot recommend a specific medication for your particular case "
        "because that would require clinical judgement about you."
    )
    result = make_result(
        outcome="plan_refused", verdict="refused",
        refusal={"reason": reason, "message": message},
    )
    answer = compose(result, diagnose(result, None, make_plan()))

    assert answer["answer_kind"] == KIND_REFUSAL
    text = render_text(answer)
    assert expected in text
    assert message not in text


def test_an_unrecognised_refusal_reason_gets_a_generic_sentence():
    result = make_result(
        outcome="plan_refused", verdict="refused",
        refusal={"reason": "a_reason_from_a_later_contract", "message": "..."},
    )
    answer = compose(result, diagnose(result, None, make_plan()))
    assert "declined before any query was built" in render_text(answer)


# ---------------------------------------------------------------------------
# Caveats
# ---------------------------------------------------------------------------


def test_a_concept_warning_leads_the_caveats():
    result = make_result(
        candidates=[make_candidate()],
        concept_warning="1 concept check(s) did not confirm the intended entity",
    )
    answer = compose(result, diagnose(result, None, make_plan()))
    assert "may describe a related but different concept" in answer["caveats"][0]


def test_reranking_is_disclosed_without_reproducing_its_reasons():
    result = make_result(candidates=[
        make_candidate(rerank_reason="model prose about evidence strength here",
                       rerank_grounded=True),
    ])
    answer = compose(result, diagnose(result, None, make_plan()))
    caveats = " ".join(answer["caveats"])
    assert "adjusted by a model" in caveats
    assert "are not reproduced here" in caveats
    assert "model prose about" not in caveats


def test_provenance_names_the_model_assisted_steps():
    result = make_result(candidates=[make_candidate()], resolution=make_resolution())
    answer = compose(result, diagnose(result, None, make_plan()))
    provenance = answer["provenance"]
    assert provenance["entity_resolution"]["disease"]["chosen_by_model"] is True
    assert len(provenance["model_assisted_steps"]) == 3


# ---------------------------------------------------------------------------
# The real thing
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not EXAMPLE.exists(), reason="the example run is not present")
def test_the_real_example_composes_and_passes():
    result = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    diag = diagnose(result, None, None)
    answer = compose(result, diag, settings=ComposerSettings(top_k=25))

    assert answer["answer_kind"] == KIND_CANDIDATES
    assert answer["grounding"]["ok"] is True, answer["grounding"]["violations"]
    assert answer["grounding"]["checked"] >= 25
    assert answer["grounding"]["lexicon"]["quarantined_strings"] > 0

    text = render_text(answer)
    for candidate in result["results"]:
        reason = candidate.get("rerank_reason")
        if reason and len(reason) >= 40:
            assert reason not in text
    assert "Petrolatum" in text
    assert "DRUGBANK:DB11058" in text


@pytest.mark.skipif(not EXAMPLE.exists(), reason="the example run is not present")
def test_the_real_examples_explanation_paths_are_rebuilt_not_copied():
    result = json.loads(EXAMPLE.read_text(encoding="utf-8"))
    answer = compose(result, diagnose(result, None, None))
    explanations = [s for s in answer["statements"] if s["kind"] == "explanation"]

    assert explanations
    for statement in explanations:
        assert statement["supported_by"]
        assert "—[biolink:" in statement["text"]


# ---------------------------------------------------------------------------
# Query-describing statements
# ---------------------------------------------------------------------------
#
# A regression suite for a bug the fixtures hid. Real results always carry the
# TRAPI graph that was sent; these fixtures did not, so no test ever exercised
# the sentence an absence answer exists to say — and it failed the gate on the
# first live run.


def absence_with_a_real_query(**kwargs):
    from helpers import submitted_query
    return make_result(
        outcome="no_graph_data", verdict="no_answer",
        paths={"P1": make_path(num_candidates=0,
                               queries=[submitted_query(**kwargs)])},
        resolution=make_resolution(),
    )


def test_an_absence_may_name_the_query_it_ran():
    """The predicate an absence names came back on no edge — by definition,
    since nothing came back. Checking it against returned edges alone rejects
    the only sentence an absence has to offer."""
    result = absence_with_a_real_query()
    answer = compose(result, diagnose(result, None, make_plan()))

    assert answer["grounding"]["ok"] is True, answer["grounding"]["violations"]
    assert answer["grounding"]["failed"] == 0

    text = render_text(answer)
    assert "biolink:treats" in text
    assert "biolink:ChemicalEntity" in text
    assert "MONDO:1" in text


def test_the_query_description_follows_what_was_sent_not_what_was_planned():
    """When a multi-hop query is decomposed, the plan and the submitted query
    differ. The description must match the one that ran."""
    result = absence_with_a_real_query(predicate="biolink:affects")
    answer = compose(result, diagnose(result, None, make_plan()))  # plan says treats

    statement = next(
        s for s in answer["statements"] if s["kind"] == "query_description"
    )
    assert "biolink:affects" in statement["text"]
    assert "biolink:treats" not in statement["text"]


def test_a_claim_about_the_graph_may_not_borrow_query_terms():
    """The guarantee the fix must not weaken. `query_description` may name the
    predicate that was sent; `candidate` may not, because that is a claim about
    what the graph asserts."""
    lex = build_lexicon(absence_with_a_real_query())

    allowed = check_statement(
        {"id": "A", "kind": "query_description",
         "text": "the query was: any biolink:ChemicalEntity —[biolink:treats]→ MONDO:1",
         "supported_by": []}, lex,
    )
    assert allowed == []

    refused = check_statement(
        {"id": "B", "kind": "candidate",
         "text": "CHEBI:9 —[biolink:treats]→ MONDO:1", "supported_by": []}, lex,
    )
    assert {v.kind for v in refused} == {"ungrounded_identifier"}
    assert len(refused) == 2          # both the drug and the predicate


def test_an_unknown_kind_gets_the_strict_rule():
    """A statement kind nobody added to the allow-list is checked strictly.
    Failing closed is the right default for a gate."""
    lex = build_lexicon(absence_with_a_real_query())
    violations = check_statement(
        {"id": "C", "kind": "something_new",
         "text": "names biolink:treats", "supported_by": []}, lex,
    )
    assert [v.kind for v in violations] == ["ungrounded_identifier"]


def test_an_unreachable_explanation_endpoint_is_nameable():
    """The endpoint an explanation query failed to reach appears in no returned
    edge, and saying which endpoint was unreachable is the point."""
    result = make_result(
        candidates=[make_candidate()],
        evidence={"explanations": {
            "summaries": [{
                "query_id": "EQ1", "endpoint_b": "MONDO:0060057",
                "candidates_attempted": 3, "candidates_with_paths": 0,
                "paths_found": 0, "paths_kept": 0, "warnings": [],
            }],
            "paths_by_endpoint": {},
        }},
    )
    answer = compose(result, diagnose(result, None, make_plan()))

    assert answer["grounding"]["ok"] is True, answer["grounding"]["violations"]
    assert "MONDO:0060057" in render_text(answer)


def test_literature_is_quoted_only_for_edges_in_the_answer():
    """The bug a live run caught, in the shape that caused it.

    `evidence.literature.by_edge` covers every edge the executor checked, which
    is a wider set than the edges attached to candidates that survived ranking
    and filtering. Quoting an abstract about an edge that is not in the answer
    offers evidence for a claim the answer does not make — and it produced
    three violations at once, since the edge id, its PMID and its quote are all
    absent from a lexicon built from surviving edges.
    """
    verdict = {
        "verdict": "supports", "quote_verified": True,
        "quote": "Ivacaftor increased chloride transport in this cohort.",
        "pmid": "PMID:21602569",
    }
    dropped_verdict = {
        "verdict": "supports", "quote_verified": True,
        "quote": "VX-770, a CFTR potentiator, increases channel open probability.",
        "pmid": "PMID:21083385",
    }

    # A surviving candidate's edge carries its own literature inline — that is
    # how the executor writes it, and it is what puts the quote and the PMID
    # into the lexicon.
    served = make_edge(edge_id="e-served", literature={"verdicts": [verdict]})

    result = make_result(candidates=[make_candidate(edges=[served])])
    result["evidence"] = {"literature": {
        "summary": {"edges_checked": 2},
        "by_edge": {
            "e-served": {"status": "checked", "verdicts": [verdict]},
            # Checked by the executor, but its candidate did not survive.
            "infores:retriever:CHEBI:66901--biolink:affects--NCBIGene:1080":
                {"status": "checked", "verdicts": [dropped_verdict]},
        },
    }}

    answer = compose(result, diagnose(result, None, make_plan()))

    assert answer["grounding"]["ok"] is True, answer["grounding"]["violations"]
    assert answer["grounding"]["failed"] == 0

    text = render_text(answer)
    assert "PMID:21602569" in text          # the served edge's quote is shown
    assert "PMID:21083385" not in text      # the off-answer one is not
    assert "VX-770" not in text

    # Dropped silently would hide real evidence loss; it is counted.
    assert answer["grounding"]["quotes_skipped_off_answer_edges"] == 1


def test_no_skip_count_appears_when_nothing_was_skipped():
    result = make_result(candidates=[make_candidate()])
    answer = compose(result, diagnose(result, None, make_plan()))
    assert "quotes_skipped_off_answer_edges" not in answer["grounding"]


# ---------------------------------------------------------------------------
# A wrong anchor is not an absence
# ---------------------------------------------------------------------------


def _empty_run_with_anchor_types(types):
    return make_result(
        outcome="no_graph_data", verdict="no_answer",
        detail="the query returned no results",
        paths={"P1": make_path(verdict="no_answer", num_candidates=0)},
        resolution=make_resolution(types=types),
    )


def test_an_empty_run_with_a_sound_anchor_still_composes_an_absence():
    """The guard is narrow. Ordinary absences are unaffected."""
    result = _empty_run_with_anchor_types(["biolink:Disease"])
    answer = compose(result, diagnose(result, None, make_plan()))

    assert answer["answer_kind"] == KIND_ABSENCE


def test_a_mismatched_anchor_downgrades_an_absence_to_inconclusive(requires_biolink):
    """Checked here as well as in the policy layer, because `compose` is
    reached by routes that never consulted the policy layer."""
    result = _empty_run_with_anchor_types(["biolink:PhenotypicFeature"])
    answer = compose(result, diagnose(result, None, make_plan()))

    assert answer["answer_kind"] == KIND_INCONCLUSIVE
    text = render_text(answer)
    assert "could not have matched anything" in text
    assert "This is a statement about the graph" not in text


def test_the_anchor_caveat_travels_with_results_too(requires_biolink):
    """If the same run *did* return candidates, the reader still needs to know
    the plan's label for the concept disagrees with the identifier it landed
    on — but not in the words used for an empty run.

    "The query may therefore have been unable to match anything" is the point
    of the caveat on an absence and a falsehood here: the results in the same
    document show the constraint matched. Printing it anyway tells a reader to
    distrust an answer for a reason the answer itself refutes.
    """
    result = make_result(
        candidates=[make_candidate()],
        resolution=make_resolution(types=["biolink:PhenotypicFeature"]),
    )
    answer = compose(result, diagnose(result, None, make_plan()))

    assert answer["answer_kind"] == KIND_CANDIDATES
    caveats = " ".join(answer["caveats"])
    assert "not recorded under that category" in caveats
    assert "did not prevent a match" in caveats
    assert "unable to match anything" not in caveats


def test_the_empty_run_keeps_the_stronger_anchor_caveat(requires_biolink):
    """The wording that is true when nothing came back is unchanged."""
    result = make_result(
        outcome="no_graph_data", verdict="no_answer",
        resolution=make_resolution(types=["biolink:PhenotypicFeature"]),
    )
    answer = compose(result, diagnose(result, None, make_plan()))

    caveats = " ".join(answer["caveats"])
    assert "unable to match anything" in caveats
    assert "did not prevent a match" not in caveats


def test_a_low_confidence_anchor_is_caveated_without_blocking_the_absence():
    """Low confidence is doubt about which concept, not proof of a broken
    query. It is said, not acted on."""
    result = make_result(
        outcome="no_graph_data", verdict="no_answer",
        paths={"P1": make_path(verdict="no_answer", num_candidates=0)},
        resolution=make_resolution(confidence="low"),
    )
    answer = compose(result, diagnose(result, None, make_plan()))

    assert answer["answer_kind"] == KIND_ABSENCE
    assert any("low confidence" in c for c in answer["caveats"])


def test_the_anchor_statement_passes_the_grounding_gate(requires_biolink):
    """It names the resolved identifier, which is in the lexicon, and not the
    Biolink category, which is in no returned edge."""
    result = _empty_run_with_anchor_types(["biolink:PhenotypicFeature"])
    answer = compose(result, diagnose(result, None, make_plan()))

    assert answer["grounding"]["ok"] is True
    assert answer.get("withheld_statements") is None


# ---------------------------------------------------------------------------
# A name that did not ground
# ---------------------------------------------------------------------------
#
# Rebuilt from the live Q8 run: "which drugs treat neuropathic pain?" resolved
# nothing, the planner declined the repair, and the user was told the question
# was underspecified while twenty retrieved entries — one of them `neuralgia` —
# sat unused in the result document.


def _unresolved_run(considered, query="neuropathic pain"):
    return make_result(
        outcome="unresolved_grounding", replannable=True, verdict="unexecutable",
        detail=f"named entities did not resolve: ['{query}']",
        paths={},
        resolution={"disease_pain": {
            "entity_ref": "disease_pain",
            "query": query,
            "expected_category": "Disease",
            "is_variable": False,
            "resolved_curies": [],
            "confidence": "high",
            "reason": (
                "LLM rejected all 20 candidates: No candidate corresponds to "
                "the general disease term; all are specific subtypes or "
                "unrelated conditions."
            ),
            "considered": [
                {"curie": c, "label": l, "types": ["biolink:Disease"], "rank": i}
                for i, (l, c) in enumerate(considered, 1)
            ],
        }},
    )


_LIVE_CANDIDATES = [
    ("neuralgia", "MONDO:0021667"),
    ("Diabetic peripheral neuropathic pain", "UMLS:C1963916"),
    ("complex regional pain syndrome type 1", "MONDO:0011441"),
]


def test_an_unresolved_name_produces_a_clarification_not_an_inconclusive():
    result = _unresolved_run(_LIVE_CANDIDATES)
    answer = compose(result, diagnose(result, None, make_plan()))

    assert answer["answer_kind"] == KIND_CLARIFICATION
    text = render_text(answer)
    assert "No query was run" in text
    assert "could not be matched: 'neuropathic pain'" in text


def test_the_clarification_lists_the_entries_the_resolver_found():
    """The point of the whole change. These were retrieved and then discarded
    inside the system; they are what lets the user ask a better question."""
    result = _unresolved_run(_LIVE_CANDIDATES)
    text = render_text(compose(result, diagnose(result, None, make_plan())))

    assert "neuralgia (MONDO:0021667)" in text
    assert "returned 3 nearby entries" in text
    assert "asking again with that name" in text


def test_the_candidate_identifiers_pass_the_grounding_gate():
    """They occur in no result graph — no query ran — so they are grounded
    against the resolver's own retrieved list or not at all."""
    result = _unresolved_run(_LIVE_CANDIDATES)
    answer = compose(result, diagnose(result, None, make_plan()))

    assert answer["grounding"]["ok"] is True
    assert answer.get("withheld_statements") is None
    assert answer["grounding"]["lexicon"]["resolver_candidates"] == 3


def test_the_resolvers_own_rejection_prose_is_never_served():
    """`resolution.reason` is model-written. It is the most quotable sentence
    in the document and the one thing here that must not be reproduced."""
    result = _unresolved_run(_LIVE_CANDIDATES)
    answer = compose(result, diagnose(result, None, make_plan()))

    assert "LLM rejected" not in render_text(answer)
    assert "specific subtypes" not in render_text(answer)


def test_a_candidate_identifier_cannot_be_used_to_claim_something():
    """The exemption is scoped to offering a list. A statement that says what
    the graph holds about one of these is still refused, because none of them
    was ever queried."""
    result = _unresolved_run(_LIVE_CANDIDATES)
    lex = build_lexicon(result)

    assert check_statement(
        {"id": "A", "kind": "resolution_options",
         "text": "The search returned neuralgia (MONDO:0021667)."}, lex,
    ) == []

    claiming = check_statement(
        {"id": "B", "kind": "candidate",
         "text": "MONDO:0021667 is treated by dapsone."}, lex,
    )
    assert [v.kind for v in claiming] == ["ungrounded_identifier"]


def test_a_name_with_no_nearby_entries_says_so():
    result = _unresolved_run([], query="Zzyzx syndrome")
    text = render_text(compose(result, diagnose(result, None, make_plan())))

    assert "returned no nearby entries at all" in text
    assert "may not cover this concept" in text


def test_the_clarification_does_not_claim_a_result_graph_was_checked():
    """No query ran, so there is no result graph. The footer is the line a
    reader is most likely to take at face value."""
    result = _unresolved_run(_LIVE_CANDIDATES)
    text = render_text(compose(result, diagnose(result, None, make_plan())))

    assert "verified against the entries the resolver retrieved" in text
    assert "verified against the result graph" not in text


def test_a_long_candidate_list_is_truncated_with_a_count():
    result = _unresolved_run([(f"entry {i}", f"UMLS:C90000{i:02d}") for i in range(20)])
    text = render_text(compose(result, diagnose(result, None, make_plan())))

    assert "returned 20 nearby entries" in text
    assert "and 12 more." in text


# ---------------------------------------------------------------------------
# Explanation paths: found, filtered, grounded
# ---------------------------------------------------------------------------


def _path(*predicates, nodes=None, labels=None):
    nodes = nodes or [f"CHEBI:{i}" for i in range(len(predicates) + 1)]
    return {
        "predicates": list(predicates),
        "nodes": nodes,
        "labels": labels or [f"node {i}" for i in range(len(nodes))],
        "length": len(predicates),
        "description": " -> ".join(nodes),
        "edges": [
            {"subject": nodes[i], "predicate": p, "object": nodes[i + 1],
             "primary_source": "infores:ctd", "sources": ["infores:ctd"],
             "knowledge_level": "knowledge_assertion", "negated": False,
             "edge_id": f"x{i}", "publications": []}
            for i, p in enumerate(predicates)
        ],
    }


def _with_candidate_paths(paths):
    result = make_result(candidates=[make_candidate()])
    result["results"][0]["explanation_paths"] = paths
    result["evidence"] = {"explanations": {"paths_by_endpoint": {}}}
    return result


def test_explanation_paths_attached_to_candidates_reach_the_answer():
    """They used not to. The composer read only `paths_by_endpoint`, which a
    plan with `attach_explanations_to_candidates` leaves empty — so a live run
    spent 1,092 seconds on pathfinding and served no explanation at all."""
    result = _with_candidate_paths([
        _path("biolink:affects", "biolink:treats"),
    ])
    answer = compose(result, diagnose(result, None, make_plan()))

    assert any(s["kind"].startswith("explanation") for s in answer["statements"])


def test_those_paths_are_grounded_rather_than_withheld():
    """The lexicon reads the same place the composer does. When it did not,
    every explanation statement was composed and then dropped as ungrounded."""
    result = _with_candidate_paths([
        _path("biolink:affects", "biolink:treats"),
    ])
    answer = compose(result, diagnose(result, None, make_plan()))

    assert answer["grounding"]["ok"] is True
    assert answer.get("withheld_statements") is None


def test_a_path_reaching_the_disease_through_a_contraindication_is_not_served():
    result = _with_candidate_paths([
        _path("biolink:affects", "biolink:contraindicated_in"),
    ])
    answer = compose(result, diagnose(result, None, make_plan()))

    assert not any(s["kind"].startswith("explanation") for s in answer["statements"])
    report = answer["grounding"]["explanation_paths"]
    assert report["withheld"] == 1
    assert report["withheld_by_kind"] == {"contrary_direction": 1}


def test_the_reader_is_told_how_many_paths_were_held_back():
    """Withholding silently would leave an answer that looks like it found
    nothing, when it found something that pointed the other way."""
    result = _with_candidate_paths([
        _path("biolink:affects", "biolink:treats"),
        _path("biolink:affects", "biolink:contraindicated_in"),
        _path("biolink:subclass_of", "biolink:correlated_with"),
    ])
    answer = compose(result, diagnose(result, None, make_plan()))

    caveats = " ".join(answer["caveats"])
    assert "2 mechanistic path(s)" in caveats
    assert "not reasons" in caveats
    assert answer["grounding"]["explanation_paths"]["servable"] == 1


def test_a_run_with_only_good_paths_gets_no_withholding_caveat():
    result = _with_candidate_paths([_path("biolink:affects", "biolink:treats")])
    answer = compose(result, diagnose(result, None, make_plan()))

    assert "explanation_paths" not in answer["grounding"]
    assert not any("mechanistic path" in c for c in answer["caveats"])
