"""Scope decided on identifiers, not words.

The point of these tests is the one that keyword lists fail: a rule written once
against an ontology term covers every spelling of it and every subtype beneath
it. The rest cover the audit story — a block must name the rule, its owner and
its date — and the decorator, which turns a scope block into a refusal the
existing controller already knows how to handle.
"""

from __future__ import annotations

import json

import pytest

from query_intake.messages import RefusalReason
from query_intake.scope import (
    ScopeGate, ScopeGuardedPlanner, ScopeRule, ScopeRules, expand_closure,
    static_resolver, write_closure,
)

from helpers import make_plan


ROOT = "MONDO:9000"
CHILD = "MONDO:9001"
GRANDCHILD = "MONDO:9002"
UNRELATED = "MONDO:1"


def rule() -> ScopeRule:
    return ScopeRule(
        rule_id="R-001",
        curie=ROOT,
        label="example blocked area",
        owner="K. Ho",
        added="2026-03-14",
        review_by="2027-03-14",
        justification="example institutional requirement",
    )


def rules_with_closure() -> ScopeRules:
    return ScopeRules(
        rules=[rule()],
        closure={ROOT: {CHILD, GRANDCHILD}},
        closure_source="mondo-2026-01-05",
    )


class FakeAttempt:
    """Stands in for loop_controller.ports.PlanAttempt without importing it,
    so these tests exercise the decorator's duck typing rather than one class."""

    def __init__(self, plan):
        self.ok = True
        self.plan = plan
        self.refused = False
        self.refusal_reason = None
        self.errors = []
        self.attempts = 1
        self.raw_response = ""
        self.note = ""


class FakePlanner:
    def __init__(self, plan):
        # Not `self.plan` — that name is the method below, and the instance
        # attribute would shadow it.
        self._plan = plan
        self.calls = []

    def plan(self, question, *, available_inputs=None):
        self.calls.append(("plan", question))
        return FakeAttempt(dict(self._plan))

    def revise(self, request):
        self.calls.append(("revise", request))
        return FakeAttempt(dict(self._plan))


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("missing", [
    "rule_id", "curie", "label", "owner", "added", "justification",
])
def test_a_rule_without_its_accountability_fields_is_rejected(missing):
    """A scope entry nobody owns is one nobody will remove."""
    raw = {
        "rule_id": "R-1", "curie": ROOT, "label": "x", "owner": "someone",
        "added": "2026-01-01", "justification": "because",
    }
    raw.pop(missing)
    with pytest.raises(ValueError) as excinfo:
        ScopeRule.from_dict(raw)
    assert missing in str(excinfo.value)


def test_no_rules_file_means_everything_is_allowed():
    """The correct default. Entries get added because something requires them."""
    rules = ScopeRules.load(
        rules_path="/nonexistent/rules.json",
        closure_path="/nonexistent/closure.json",
    )
    assert rules.rules == []
    assert ScopeGate(rules).check(["MONDO:0005240"]).allowed


def test_rules_and_closure_load_from_disk(tmp_path):
    rules_path = tmp_path / "scope_rules.json"
    rules_path.write_text(json.dumps({
        "rules": [{
            "rule_id": "R-001", "curie": ROOT, "label": "area",
            "owner": "K. Ho", "added": "2026-03-14",
            "justification": "requirement",
        }],
    }), encoding="utf-8")

    closure_path = tmp_path / "scope_closure.json"
    write_closure({ROOT: [CHILD]}, closure_path, ontology_version="mondo-test")

    loaded = ScopeRules.load(rules_path, closure_path)
    assert loaded.rules[0].owner == "K. Ho"
    assert loaded.closure[ROOT] == {CHILD}
    assert loaded.closure_source == "mondo-test"


def test_a_root_with_no_closure_is_reported():
    """It blocks only itself, which looks like a working subtree rule and is
    not."""
    rules = ScopeRules(rules=[rule()], closure={})
    assert rules.unexpanded() == [ROOT]


def test_rules_past_their_review_date_are_reported():
    from datetime import date

    rules = ScopeRules(rules=[rule()])
    assert rules.stale_rules(today=date(2026, 1, 1)) == []
    assert rules.stale_rules(today=date(2028, 1, 1))[0].rule_id == "R-001"


# ---------------------------------------------------------------------------
# The gate
# ---------------------------------------------------------------------------


def test_one_rule_covers_the_whole_subtree():
    """The property a keyword list cannot have."""
    gate = ScopeGate(rules_with_closure())
    for curie in (ROOT, CHILD, GRANDCHILD):
        assert gate.check([curie]).allowed is False


def test_unrelated_concepts_pass():
    assert ScopeGate(rules_with_closure()).check([UNRELATED]).allowed


def test_a_block_names_the_rule_its_owner_and_its_date():
    """The sentence you can defend in a review, rather than 'the model said no'."""
    decision = ScopeGate(rules_with_closure()).check([GRANDCHILD])
    assert decision.reason is RefusalReason.OUT_OF_SCOPE
    assert decision.rule.rule_id == "R-001"
    assert decision.matched_curie == GRANDCHILD

    record = decision.to_dict()
    assert record["rule_owner"] == "K. Ho"
    assert record["rule_added"] == "2026-03-14"
    assert ROOT in record["detail"] and GRANDCHILD in record["detail"]


def test_any_blocked_anchor_blocks_the_question():
    assert ScopeGate(rules_with_closure()).check([UNRELATED, CHILD]).allowed is False


# ---------------------------------------------------------------------------
# Closure expansion
# ---------------------------------------------------------------------------


def test_closure_expansion_walks_the_hierarchy(tmp_path):
    obographs = tmp_path / "mondo.json"
    obographs.write_text(json.dumps({
        "graphs": [{
            "edges": [
                {"sub": "http://purl.obolibrary.org/obo/MONDO_9001",
                 "pred": "is_a",
                 "obj": "http://purl.obolibrary.org/obo/MONDO_9000"},
                {"sub": "http://purl.obolibrary.org/obo/MONDO_9002",
                 "pred": "is_a",
                 "obj": "http://purl.obolibrary.org/obo/MONDO_9001"},
                {"sub": "http://purl.obolibrary.org/obo/MONDO_0001",
                 "pred": "is_a",
                 "obj": "http://purl.obolibrary.org/obo/MONDO_0000"},
            ],
        }],
    }), encoding="utf-8")

    closure = expand_closure(obographs, [ROOT])
    assert closure[ROOT] == [CHILD, GRANDCHILD]   # transitive, and sorted


def test_closure_expansion_ignores_other_relations(tmp_path):
    obographs = tmp_path / "mondo.json"
    obographs.write_text(json.dumps({
        "graphs": [{
            "edges": [
                {"sub": "http://purl.obolibrary.org/obo/MONDO_9001",
                 "pred": "part_of",
                 "obj": "http://purl.obolibrary.org/obo/MONDO_9000"},
            ],
        }],
    }), encoding="utf-8")
    assert expand_closure(obographs, [ROOT])[ROOT] == []


def test_the_written_closure_is_readable(tmp_path):
    """A blocked set you cannot enumerate is one you cannot review."""
    path = write_closure({ROOT: [CHILD]}, tmp_path / "c.json",
                         ontology_version="mondo-2026-01-05")
    blob = json.loads(path.read_text(encoding="utf-8"))
    assert blob["ontology_version"] == "mondo-2026-01-05"
    assert blob["closure"][ROOT] == [CHILD]
    assert "do not edit by hand" in blob["generated_note"].lower()


# ---------------------------------------------------------------------------
# The decorator
# ---------------------------------------------------------------------------


def test_a_blocked_anchor_becomes_a_refusal_the_controller_understands():
    """`out_of_scope` is already in the plan contract and already rendered by
    the composer, so this needs no controller change at all."""
    planner = FakePlanner(make_plan(anchor_name="a blocked concept"))
    guarded = ScopeGuardedPlanner(
        planner, ScopeGate(rules_with_closure()),
        static_resolver({"a blocked concept": [CHILD]}),
    )
    attempt = guarded.plan("which drugs treat it?")

    assert attempt.ok is False
    assert attempt.refused is True
    assert attempt.refusal_reason == "out_of_scope"
    assert attempt.plan["refusal"]["reason"] == "out_of_scope"
    assert "R-001" in attempt.note


def test_an_allowed_anchor_passes_through_untouched():
    plan = make_plan()
    planner = FakePlanner(plan)
    guarded = ScopeGuardedPlanner(
        planner, ScopeGate(rules_with_closure()),
        static_resolver({"dermatitis herpetiformis": [UNRELATED]}),
    )
    attempt = guarded.plan("which drugs treat it?")
    assert attempt.ok is True
    assert attempt.plan == plan


def test_revisions_are_guarded_too():
    """A revision can move the anchor. A gate that only checked the first plan
    would be a gate with a door beside it."""
    planner = FakePlanner(make_plan(anchor_name="a blocked concept"))
    guarded = ScopeGuardedPlanner(
        planner, ScopeGate(rules_with_closure()),
        static_resolver({"a blocked concept": [ROOT]}),
    )
    attempt = guarded.revise(object())
    assert attempt.refused is True


def test_identifiers_pinned_in_the_plan_are_checked_without_a_resolver():
    plan = make_plan(anchor_curies=[GRANDCHILD])
    guarded = ScopeGuardedPlanner(
        FakePlanner(plan), ScopeGate(rules_with_closure()), resolver=None,
    )
    assert guarded.plan("q").refused is True


def test_without_a_resolver_a_name_only_plan_is_marked_partial():
    """Honest reporting: nothing was resolved, so nothing was really checked."""
    guarded = ScopeGuardedPlanner(
        FakePlanner(make_plan()), ScopeGate(rules_with_closure()), resolver=None,
    )
    attempt = guarded.plan("q")
    assert attempt.ok is True
    assert guarded.decisions[-1].partial is True


def test_variable_entities_are_never_scope_checked():
    """The answer entity is what the query is looking for, not a topic. A rule
    on 'any chemical this might return' would block the tool, not a subject."""
    plan = make_plan()
    plan["entities"][1]["name"] = "a blocked concept"   # the variable one
    guarded = ScopeGuardedPlanner(
        FakePlanner(plan), ScopeGate(rules_with_closure()),
        static_resolver({
            "a blocked concept": [CHILD],
            "dermatitis herpetiformis": [UNRELATED],
        }),
    )
    assert guarded.plan("q").ok is True


def test_a_failed_plan_is_passed_through_unchanged():
    class Failed:
        ok = False
        plan = None

    guarded = ScopeGuardedPlanner(
        FakePlanner(make_plan()), ScopeGate(rules_with_closure()),
    )
    guarded.inner = type("P", (), {"plan": lambda self, q, **k: Failed()})()
    assert guarded.plan("q").ok is False


def test_require_biomedical_is_off_by_default():
    """The controller's repair path handles an unresolvable anchor better: it
    gets the resolver's candidate list and asks the planner to try again."""
    guarded = ScopeGuardedPlanner(
        FakePlanner(make_plan()), ScopeGate(rules_with_closure()),
        static_resolver({}),          # resolves nothing
    )
    assert guarded.plan("q").ok is True

    strict = ScopeGuardedPlanner(
        FakePlanner(make_plan()), ScopeGate(rules_with_closure()),
        static_resolver({}), require_biomedical=True,
    )
    attempt = strict.plan("q")
    assert attempt.refused is True
    assert attempt.refusal_reason == "not_biomedical"
