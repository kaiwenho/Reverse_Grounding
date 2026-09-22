"""Plan identity.

The loop's termination argument reduces to this file being right. If a reworded
plan counted as a new plan, "no repetition" would guarantee nothing.
"""

from __future__ import annotations

import copy
import json

from loop_controller.contracts import RelaxationAxis
from loop_controller.fingerprint import (
    describe_off_axis, diff_summary, executable_view, keyed_view,
    locked_constraints, orphan_entities, pinned_entity_refs, relaxation_diff,
    same_query,
)

from helpers import make_plan


def test_identical_plans_share_a_fingerprint():
    assert same_query(make_plan(), make_plan())


def test_prose_changes_do_not_make_a_new_plan():
    """The failure this exists to prevent: four iterations, one query."""
    original = make_plan()
    reworded = make_plan(restated="Identify drugs indicated for this condition.")
    reworded["confidence"] = {"level": "medium", "reasons": ["reworded"]}
    reworded["gaps"] = [{"gap_type": "none", "detail": "no gap"}]
    reworded["interpretation"]["intent"] = "something_else"

    assert same_query(original, reworded)
    assert diff_summary(original, reworded) == []


def test_version_stamps_are_not_part_of_identity():
    original = make_plan()
    restamped = make_plan()
    restamped["plan_version"] = "0.11.0"
    restamped["biolink_version"] = "4.9.9"
    assert same_query(original, restamped)


def test_a_different_predicate_is_a_different_plan():
    assert not same_query(
        make_plan(predicate="biolink:treats"),
        make_plan(predicate="biolink:affects"),
    )
    assert "paths: changed" in diff_summary(
        make_plan(predicate="biolink:treats"),
        make_plan(predicate="biolink:affects"),
    )


def test_a_widened_category_is_a_different_plan():
    assert not same_query(
        make_plan(answer_category="SmallMolecule"),
        make_plan(answer_category="ChemicalEntity"),
    )


def test_adding_an_evidence_policy_is_a_different_plan():
    assert not same_query(
        make_plan(),
        make_plan(evidence_policy={
            "origin": "user_requested",
            "application": "filter",
            "rationale": "the user asked for published evidence",
            "min_publications": 1,
        }),
    )


def test_hop_order_matters():
    plan = make_plan()
    plan["paths"][0]["hops"] = [
        {"subject_ref": "a", "predicate": "biolink:treats", "object_ref": "b"},
        {"subject_ref": "b", "predicate": "biolink:affects", "object_ref": "c"},
    ]
    reversed_plan = copy.deepcopy(plan)
    reversed_plan["paths"][0]["hops"].reverse()
    assert not same_query(plan, reversed_plan)


def test_integral_floats_do_not_change_identity():
    plan = make_plan()
    plan["paths"][0]["max_hops"] = 3
    other = copy.deepcopy(plan)
    other["paths"][0]["max_hops"] = 3.0
    assert same_query(plan, other)


def test_executable_view_drops_descriptive_blocks():
    view = executable_view(make_plan())
    assert "entities" in view and "paths" in view
    assert "interpretation" not in view
    assert "confidence" not in view


# ---------------------------------------------------------------------------
# Locked constraints
# ---------------------------------------------------------------------------


def test_pinned_anchors_are_locked_and_variables_are_not():
    locked = locked_constraints(make_plan())
    assert "anchor:disease" in locked
    assert "anchor:candidate_drug" not in locked
    assert pinned_entity_refs(make_plan()) == {"disease"}


def test_every_evidence_policy_filter_field_is_locked():
    plan = make_plan(evidence_policy={
        "origin": "user_requested",
        "application": "filter",
        "rationale": "the user asked for recent, published, curated evidence",
        "min_publications": 2,
        "min_year": 2015,
        "require_primary_knowledge_source": True,
        "excluded_knowledge_sources": ["infores:text-mining"],
    })
    locked = locked_constraints(plan)
    assert "evidence_policy:min_publications" in locked
    assert "evidence_policy:min_year" in locked
    assert "evidence_policy:require_primary_knowledge_source" in locked
    assert "evidence_policy:excluded_knowledge_sources" in locked


def test_a_policy_block_with_no_recognised_field_is_still_locked():
    """`origin` is fixed to `user_requested`, so presence alone means the user
    asked for it — even if the filtering field is one this package predates."""
    plan = make_plan(evidence_policy={
        "origin": "user_requested",
        "application": "filter",
        "rationale": "a filter from a future contract version",
        "some_future_filter": ["x"],
    })
    assert "evidence_policy" in locked_constraints(plan)


def test_entity_constraints_are_locked_per_entity():
    plan = make_plan()
    plan["entities"][0]["constraints"] = [
        {"field": "approval_status", "op": "eq", "value": "approved"},
    ]
    assert "constraint:disease:approval_status" in locked_constraints(plan)


# ---------------------------------------------------------------------------
# One-axis relaxation
# ---------------------------------------------------------------------------
#
# `same_query` answers "is this a different plan?", which is what termination
# needs. These answer "is this the one difference I asked for?", which is what
# the loop's attribution claim needs — and which nothing checked until now.


def test_the_keyed_view_survives_a_reordered_entity_list():
    """A planner that re-emits the entities in another order changed nothing.

    A positional diff would report every field of every entity as different,
    and the over-relaxation check would fire on every revision.
    """
    original = make_plan()
    shuffled = make_plan()
    shuffled["entities"] = list(reversed(shuffled["entities"]))

    assert keyed_view(original) == keyed_view(shuffled)


def test_a_widened_predicate_is_on_the_predicate_axis():
    before = make_plan()
    after = make_plan(predicate="biolink:affects")
    axis = RelaxationAxis(path_id="P1", axis="predicate")

    diff = relaxation_diff(before, after, axis)

    assert diff.ok
    assert diff.on_axis == ["paths[P1].hops[0].predicate"]
    assert diff.off_axis == []


def test_a_widened_category_is_on_the_category_axis_for_that_entity():
    before = make_plan()
    after = make_plan(answer_category="NamedThing")
    axis = RelaxationAxis(
        path_id="P1", axis="category", entity_ref="candidate_drug",
    )

    diff = relaxation_diff(before, after, axis)

    assert diff.ok
    assert diff.on_axis == ["entities[candidate_drug].biolink_category"]


def test_widening_a_different_entity_than_the_one_named_is_off_axis():
    """The axis key carries the entity, and so does the check.

    Two entities on one path can each be widened, and spending the axis for one
    of them does not license widening the other.
    """
    before = make_plan()
    after = make_plan(answer_category="NamedThing")
    axis = RelaxationAxis(path_id="P1", axis="category", entity_ref="disease")

    diff = relaxation_diff(before, after, axis)

    assert not diff.ok
    assert diff.off_axis == ["entities[candidate_drug].biolink_category"]


def test_loosening_two_constraints_at_once_is_reported():
    """The failure this whole section exists for.

    A revision that widens the category *and* broadens the predicate will often
    return results, and the loop would then attribute them to the category —
    which nobody could check and which would not be true.
    """
    before = make_plan()
    after = make_plan(answer_category="NamedThing", predicate="biolink:affects")
    axis = RelaxationAxis(
        path_id="P1", axis="category", entity_ref="candidate_drug",
    )

    diff = relaxation_diff(before, after, axis)

    assert not diff.ok
    assert diff.on_axis == ["entities[candidate_drug].biolink_category"]
    assert diff.off_axis == ["paths[P1].hops[0].predicate"]

    complaint = describe_off_axis(diff)
    assert "paths[P1].hops[0].predicate" in complaint
    assert "not asked for" in complaint


def test_prose_only_changes_are_not_off_axis_changes():
    """The cosmetic strip applies here too, or every revision would be over-broad."""
    before = make_plan()
    after = make_plan(predicate="biolink:affects",
                      restated="Some other wording entirely.")
    after["confidence"] = {"level": "low", "reasons": ["relaxed"]}

    diff = relaxation_diff(
        before, after, RelaxationAxis(path_id="P1", axis="predicate"),
    )

    assert diff.ok


def test_an_unrecognised_axis_makes_every_change_off_axis():
    """Safe direction: a new axis this module has no scope for is flagged for a
    human rather than waved through."""
    before = make_plan()
    after = make_plan(predicate="biolink:affects")

    diff = relaxation_diff(
        before, after, RelaxationAxis(path_id="P1", axis="something_new"),
    )

    assert not diff.axis_known
    assert not diff.ok
    assert diff.off_axis == ["paths[P1].hops[0].predicate"]


# ---------------------------------------------------------------------------
# Entities the plan declares and never queries
# ---------------------------------------------------------------------------


def test_a_plan_that_uses_all_its_entities_has_no_orphans():
    assert orphan_entities(make_plan()) == []


def test_a_fixed_entity_referenced_by_no_hop_is_an_orphan():
    """The Q9 shape. Asked "which drugs treat dermatitis herpetiformis by
    targeting CFTR?", the planner declared all three concepts and wrote one hop
    — drug to CFTR. The disease sat in the entity list, queried by nothing, and
    twenty well-evidenced drugs came back that nobody had asked about."""
    plan = make_plan()
    plan["entities"].append({
        "entity_ref": "derm_herp", "name": "dermatitis herpetiformis",
        "biolink_category": "Disease", "is_variable": False,
    })

    orphans = orphan_entities(plan)
    assert [o["entity_ref"] for o in orphans] == ["derm_herp"]
    assert orphans[0]["name"] == "dermatitis herpetiformis"


def test_an_unused_variable_entity_is_not_an_orphan():
    """A variable is the shape of what the plan is looking for. One nothing
    points at is inert; a fixed entity is a concept the user named."""
    plan = make_plan()
    plan["entities"].append({
        "entity_ref": "spare", "name": "any protein",
        "biolink_category": "Protein", "is_variable": True,
    })

    assert orphan_entities(plan) == []


def test_an_entity_used_only_by_an_explanation_query_is_not_an_orphan():
    """References are collected from the whole plan, not just from hops. A
    check that only looked at paths would report every hybrid plan's
    explanation endpoints as dropped."""
    plan = make_plan()
    plan["entities"].append({
        "entity_ref": "context_disease", "name": "coeliac disease",
        "biolink_category": "Disease", "is_variable": False,
    })
    plan["explanation_queries"] = [{
        "query_id": "EQ1",
        "endpoint_a": {"entity_ref": "disease"},
        "endpoint_b": {"entity_ref": "context_disease"},
        "max_hops": 3,
    }]

    assert orphan_entities(plan) == []


def test_an_entity_named_only_in_prose_is_still_an_orphan():
    """Mentioning the concept in the restated question is not querying it.

    This is the case that makes the check worth having: the plan reads as
    though it covers the question, because the prose says so.
    """
    plan = make_plan(restated="Find drugs treating this disease via CFTR.")
    plan["entities"].append({
        "entity_ref": "cftr", "name": "CFTR",
        "biolink_category": "Gene", "is_variable": False,
    })
    plan["notes"] = "CFTR is handled by the candidate_drug path."

    assert [o["entity_ref"] for o in orphan_entities(plan)] == ["cftr"]


# ---------------------------------------------------------------------------
# What the contract forces, and what a default means
# ---------------------------------------------------------------------------
#
# Both of these were found by measuring the one-axis rule against the live
# planner, and both were faults in the rule rather than in the model. The
# category axis failed 5 rounds out of 5 for doing the only thing that
# validates.


def test_a_field_at_its_contract_default_is_the_same_as_omitting_it():
    """`negated: false` and no `negated` are the same query.

    A revision that correctly widened a predicate was rejected as over-broad
    because it had tidied away `negated: false`. The rule was punishing plans
    for being tidy.
    """
    verbose = make_plan()
    verbose["paths"][0]["hops"][0]["negated"] = False
    verbose["paths"][0]["disabled"] = False

    terse = make_plan()
    terse["paths"][0]["hops"][0].pop("negated", None)
    terse["paths"][0].pop("disabled", None)

    assert keyed_view(verbose) == keyed_view(terse)
    assert same_query(verbose, terse)


def test_setting_a_default_valued_field_away_from_its_default_is_a_change():
    """The normalisation must not swallow a real one."""
    plain = make_plan()
    negated = make_plan()
    negated["paths"][0]["hops"][0]["negated"] = True

    assert not same_query(plain, negated)


def test_widening_a_category_may_move_the_paths_declared_result_category():
    """plan-core requires them to agree, so moving one forces the other.

    Rejecting that is rejecting the only revision that validates:

        expected category 'SmallMolecule' is incompatible with return entity
        'candidate_drug' category 'ChemicalEntity'
    """
    before = make_plan(answer_category="SmallMolecule")
    before["paths"][0]["expected_result_category"] = "SmallMolecule"

    after = make_plan(answer_category="ChemicalEntity")
    after["paths"][0]["expected_result_category"] = "ChemicalEntity"

    diff = relaxation_diff(before, after, RelaxationAxis(
        path_id="P1", axis="category", entity_ref="candidate_drug",
    ))

    assert diff.ok
    assert "paths[P1].expected_result_category" in diff.on_axis


def test_the_dependency_is_scoped_to_the_path_that_returns_that_entity():
    """A path returning something else does not get a free pass."""
    before = make_plan()
    before["paths"].append({
        "path_id": "P2", "return_entity_ref": "disease",
        "expected_result_category": "Disease",
        "hops": [{"subject_ref": "candidate_drug",
                  "predicate": "biolink:treats", "object_ref": "disease"}],
    })
    after = json.loads(json.dumps(before))
    after["entities"][1]["biolink_category"] = "ChemicalEntity"
    after["paths"][1]["expected_result_category"] = "ChemicalEntity"

    diff = relaxation_diff(before, after, RelaxationAxis(
        path_id="P1", axis="category", entity_ref="candidate_drug",
    ))

    assert not diff.ok
    assert diff.off_axis == ["paths[P2].expected_result_category"]


def test_a_category_widening_that_also_moves_the_predicate_is_still_caught():
    """The guarantee has to survive being loosened."""
    before = make_plan(answer_category="SmallMolecule")
    before["paths"][0]["expected_result_category"] = "SmallMolecule"

    after = make_plan(answer_category="ChemicalEntity",
                      predicate="biolink:related_to")
    after["paths"][0]["expected_result_category"] = "ChemicalEntity"

    diff = relaxation_diff(before, after, RelaxationAxis(
        path_id="P1", axis="category", entity_ref="candidate_drug",
    ))

    assert not diff.ok
    assert diff.off_axis == ["paths[P1].hops[0].predicate"]
