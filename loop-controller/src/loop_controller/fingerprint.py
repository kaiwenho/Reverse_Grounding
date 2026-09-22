"""
fingerprint.py — Identity for a plan, based on what it would execute.

The loop's termination argument rests on "a revision must differ from every
plan already tried". That is only meaningful if difference is measured on the
part of the plan that changes the query. A planner asked to fix a plan will
often return prose that reads quite differently — a reworded
``restated_question``, a new confidence reason, reordered lists — while sending
byte-identical TRAPI to ARAX. Counting that as progress is how a loop spends
its whole budget re-running one query and reports four attempts.

So the fingerprint covers the executable content and nothing else:

  * entities: reference, surface name, category, variability, constraints,
    and any bound identifiers or external input
  * paths: hops in order, with subject, predicate, object, qualifiers,
    predicate expansion, and edge filters
  * explanation queries: endpoints, hop limits, and middle-category
    allow/blocklists
  * evidence policy, ranking specs, and aggregation settings, all of which
    change which candidates come back or in what order

and explicitly not: interpretation prose, confidence, gaps, notes, plan and
Biolink version stamps, or anything the executor ignores.

Field names are read defensively. The plan contract is versioned and shared,
but the controller reads plans it did not build, and a fingerprint that raises
on an unfamiliar shape would fail the loop for a cosmetic reason. Unknown keys
inside a covered block are included by name rather than dropped, so a contract
that grows a new executable field is over- rather than under-sensitive.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Set


#: Top-level plan keys whose content changes what runs.
EXECUTABLE_KEYS: tuple = (
    "plan_mode",
    "entities",
    "paths",
    "explanation_queries",
    "candidate_ranking",
    "explanation_ranking",
    "explanation_influence",
    "aggregation",
    "evidence_policy",
    "refusal",
)

#: The fields of an `evidence_policy` block that actually filter. The contract
#: fixes `origin` to `user_requested`, so every one of these is a constraint the
#: user asked for and none of them is ever relaxed.
EVIDENCE_POLICY_FILTERS: tuple = (
    "required_knowledge_sources",
    "excluded_knowledge_sources",
    "require_primary_knowledge_source",
    "min_publications",
    "min_year",
    "agent_type",
)

#: Fields whose contract default means the same as leaving them out, keyed by
#: field name with the default value.
#:
#: A planner that omits `negated` and one that writes `negated: false` have
#: written the same query, and the plan contract says so — pydantic fills the
#: default either way. Treating them as different makes every revision that
#: tidied an optional field look like a change to what runs.
#:
#: This is not hypothetical. Measuring the one-axis rule against the live
#: planner, a revision that correctly widened a predicate was rejected as
#: over-broad because it had dropped `negated: false`, and every category
#: revision was rejected partly for dropping `disabled: false`. The rule was
#: punishing plans for being tidy.
DEFAULT_VALUED_KEYS: Dict[str, Any] = {
    "negated": False,
    "disabled": False,
    "predicate_expansion": "descendants",
    "query_role": "queried",
}

#: Keys ignored wherever they appear. Descriptive, not executable.
COSMETIC_KEYS: frozenset = frozenset({
    "interpretation", "confidence", "gaps", "notes", "note", "rationale",
    "description", "plan_version", "biolink_version", "plan_id", "question",
    "restated_question", "evidence_recommendations", "generated_at",
    "user_request", "label",
})


def _canonical(value: Any) -> Any:
    """Recursively strip cosmetic keys and impose a stable ordering.

    Dicts are sorted by key. Lists are left in order, because order is
    executable in this contract: hops run in sequence, ranking criteria apply
    in priority order, and path priority is a real aggregation strategy.
    """
    if isinstance(value, dict):
        out = {}
        for key in sorted(value):
            if key in COSMETIC_KEYS:
                continue
            # A field explicitly set to its contract default is the same plan
            # as one that omits it, so both canonicalise to omitted.
            if key in DEFAULT_VALUED_KEYS and value[key] == DEFAULT_VALUED_KEYS[key]:
                continue
            inner = _canonical(value[key])
            if inner is None or inner == [] or inner == {}:
                continue
            out[key] = inner
        return out
    if isinstance(value, (list, tuple)):
        return [_canonical(v) for v in value]
    if isinstance(value, float) and value.is_integer():
        # 3 and 3.0 are the same hop limit.
        return int(value)
    return value


def _as_plain(plan: Any) -> Dict[str, Any]:
    """Accept a dict, a pydantic QueryPlan, or anything with a __dict__."""
    if isinstance(plan, dict):
        return plan
    dump = getattr(plan, "model_dump", None)
    if callable(dump):
        return dump(exclude_none=True)
    if hasattr(plan, "__dict__"):
        return {k: v for k, v in vars(plan).items() if not k.startswith("_")}
    raise TypeError(f"cannot fingerprint a {type(plan).__name__}")


def executable_view(plan: Any) -> Dict[str, Any]:
    """The part of a plan that decides what gets queried."""
    raw = _as_plain(plan)
    view: Dict[str, Any] = {}
    for key in EXECUTABLE_KEYS:
        if key in raw and raw[key] not in (None, [], {}):
            view[key] = _canonical(raw[key])
    return view


def fingerprint(plan: Any) -> str:
    """A short, stable hash of a plan's executable content."""
    view = executable_view(plan)
    blob = json.dumps(view, sort_keys=True, separators=(",", ":"), default=str)
    return "plan:" + hashlib.sha256(blob.encode("utf-8")).hexdigest()[:16]


def same_query(plan_a: Any, plan_b: Any) -> bool:
    """True when two plans would send the same queries."""
    return fingerprint(plan_a) == fingerprint(plan_b)


def diff_summary(plan_a: Any, plan_b: Any, max_items: int = 12) -> List[str]:
    """Which executable blocks differ, in words.

    Used in two places, both of which want a reason rather than a boolean: the
    trace, when an iteration produced a new plan, and the rejection message
    sent back to the planner when it produced an old one. "your revision is
    identical in paths and evidence_policy" is a correctable complaint;
    "rejected: duplicate" is not.
    """
    view_a = executable_view(plan_a)
    view_b = executable_view(plan_b)
    out: List[str] = []
    for key in sorted(set(view_a) | set(view_b)):
        left, right = view_a.get(key), view_b.get(key)
        if left == right:
            continue
        if left is None:
            out.append(f"{key}: added")
        elif right is None:
            out.append(f"{key}: removed")
        else:
            out.append(f"{key}: changed")
        if len(out) >= max_items:
            break
    return out


# ---------------------------------------------------------------------------
# Field-level diff: did the revision loosen only what it was asked to?
# ---------------------------------------------------------------------------
#
# `fingerprint` answers "is this a different query?", which is what termination
# needs. Relaxation needs a sharper question: "is this the *one* different
# query I asked for?" The loop tells the planner to loosen a single named
# constraint and change nothing else, so that an answer arriving after the
# revision can be attributed to that constraint. Until now that was a request
# in a prompt with nothing checking it, and a planner that quietly widened a
# category *and* broadened a predicate would produce results the loop would
# then explain with a claim that was no longer true.
#
# Checking it needs a diff at the level of individual fields rather than whole
# blocks, and one that survives the planner reordering a list — which it does
# freely, and which a positional diff would report as every field changing.

#: List blocks whose members carry their own identity in the plan contract.
#: Members are matched by that identifier and compared one to one; anything
#: else is compared by position, which is correct for hops because a path's
#: hops are an ordered sequence with no identifiers of their own.
IDENTIFIED_LISTS: Dict[str, tuple] = {
    "entities": ("entity_ref", "ref", "name"),
    "paths": ("path_id", "id"),
    "explanation_queries": ("query_id", "id", "endpoint_a"),
}

#: The flattened field paths each relaxation axis is entitled to change, with
#: `{path}` and `{entity}` filled from the axis. `*` matches one path segment;
#: `**` matches any number.
#:
#: Written as patterns rather than as a structural walk because the plan
#: contract will grow fields, and a pattern that names `qualifier**` keeps
#: covering a new qualifier field, where a hand-listed set would quietly start
#: reporting it as an off-axis change.
AXIS_SCOPE: Dict[str, tuple] = {
    "predicate": (
        "paths[{path}].hops[*].predicate",
    ),
    "predicate_expansion": (
        "paths[{path}].hops[*].predicate_expansion**",
    ),
    "qualifiers": (
        "paths[{path}].hops[*].qualifier**",
    ),
    "category": (
        "entities[{entity}].biolink_category",
        "entities[{entity}].categor**",
    ),
    "constraints": (
        "entities[{entity}].constraints**",
    ),
}

_MISSING = object()


def keyed_view(plan: Any) -> Dict[str, Any]:
    """The executable view flattened to one entry per leaf field.

    Keys read like ``paths[P1].hops[0].predicate`` and
    ``entities[candidate_drug].biolink_category``: identified list members by
    their identifier, hops by position, everything else by field name. Two
    plans that differ only in the order of their entities produce the same
    keyed view, which is the point.
    """
    out: Dict[str, Any] = {}
    _flatten(executable_view(plan), "", out)
    return out


def _flatten(node: Any, prefix: str, out: Dict[str, Any], name: str = "") -> None:
    if isinstance(node, dict):
        if not node:
            out[prefix or "."] = {}
            return
        for key in sorted(node):
            child = f"{prefix}.{key}" if prefix else key
            _flatten(node[key], child, out, name=key)
        return

    if isinstance(node, list):
        if not node:
            out[prefix or "."] = []
            return
        id_fields = IDENTIFIED_LISTS.get(name)
        for index, item in enumerate(node):
            label = str(index)
            if id_fields and isinstance(item, dict):
                for candidate in id_fields:
                    if item.get(candidate):
                        label = str(item[candidate])
                        break
            _flatten(item, f"{prefix}[{label}]", out, name=name)
        return

    out[prefix or "."] = node


def _compile(pattern: str) -> "re.Pattern[str]":
    """Turn a scope pattern into a regex over flattened field paths."""
    out: List[str] = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == "*":
            if pattern.startswith("**", index):
                out.append(".*")
                index += 2
            else:
                out.append(r"[^.\[\]]*")
                index += 1
            continue
        out.append(re.escape(char))
        index += 1
    return re.compile("^" + "".join(out) + "$")


def dependent_patterns(axis: Any, plan: Any) -> List[str]:
    """Fields the *contract* forces to move when this axis moves.

    Widening a variable entity's category is not a self-contained edit. If that
    entity is a path's `return_entity_ref`, plan-core requires the path's
    `expected_result_category` to stay compatible with it:

        expected category 'SmallMolecule' is incompatible with return entity
        'candidate_chemical' category 'ChemicalEntity'

    So a planner asked to widen the category has two choices: move the declared
    result category with it, or emit a plan that does not validate. Calling the
    first one over-relaxation punishes the only correct answer.

    It did, five times out of five, the first time the category axis was
    measured against the live planner. Every revision was rejected, re-asked,
    rejected again and then used with its attribution withdrawn — for doing
    exactly what the contract requires. The one-axis rule was wrong, not the
    model.

    Computed from the plan rather than listed as a pattern, because the
    dependency runs through `return_entity_ref` and only applies to the path
    that actually returns the widened entity.
    """
    if getattr(axis, "axis", "") != "category":
        return []
    entity_ref = getattr(axis, "entity_ref", None)
    if not entity_ref:
        return []

    raw = _as_plain(plan)
    out: List[str] = []
    for path in raw.get("paths") or []:
        item = path if isinstance(path, dict) else _as_plain(path)
        if item.get("return_entity_ref") != entity_ref:
            continue
        path_id = item.get("path_id") or item.get("id")
        if path_id:
            out.append(f"paths[{path_id}].expected_result_category")
    return out


def axis_patterns(axis: Any, plan: Any = None) -> List["re.Pattern[str]"]:
    """The compiled field-path patterns this axis may change.

    An axis this module does not recognise returns no patterns, which makes
    every change off-axis. That is the safe direction: an unknown axis produces
    a warning that a human reads, rather than a silent pass.

    ``plan`` is optional only so existing callers keep working; without it the
    contract-forced dependencies cannot be computed and a correct category
    relaxation will read as over-broad.
    """
    templates = list(AXIS_SCOPE.get(getattr(axis, "axis", ""), ()))
    path_id = str(getattr(axis, "path_id", "") or "")
    entity_ref = getattr(axis, "entity_ref", None)

    compiled: List["re.Pattern[str]"] = []
    for template in templates:
        filled = template.replace("{path}", path_id or "*")
        # An entity-scoped axis that arrived without an entity is scoped to any
        # entity rather than to none: the executor names one in practice, and
        # refusing every entity would turn a missing field into a false
        # over-relaxation report.
        filled = filled.replace("{entity}", str(entity_ref) if entity_ref else "*")
        compiled.append(_compile(filled))

    if plan is not None:
        for dependent in dependent_patterns(axis, plan):
            compiled.append(_compile(dependent))
    return compiled


@dataclass
class RelaxationDiff:
    """What a revision actually changed, split by whether it was asked for."""

    changed: List[str] = field(default_factory=list)
    on_axis: List[str] = field(default_factory=list)
    off_axis: List[str] = field(default_factory=list)
    axis_key: str = ""
    axis_known: bool = True

    @property
    def unchanged(self) -> bool:
        return not self.changed

    @property
    def ok(self) -> bool:
        """True when the requested axis moved and nothing else did."""
        return bool(self.on_axis) and not self.off_axis

    def to_dict(self) -> Dict[str, Any]:
        return {
            "axis_key": self.axis_key,
            "axis_known": self.axis_known,
            "changed": self.changed,
            "on_axis": self.on_axis,
            "off_axis": self.off_axis,
            "ok": self.ok,
        }


def relaxation_diff(before: Any, after: Any, axis: Any) -> RelaxationDiff:
    """Compare two plans against the one axis the planner was told to loosen."""
    view_a = keyed_view(before)
    view_b = keyed_view(after)
    # The dependencies are read off the plan being revised, so the scope covers
    # the path that actually returns the entity being widened.
    patterns = axis_patterns(axis, before)

    diff = RelaxationDiff(
        axis_key=str(getattr(axis, "key", "") or ""),
        axis_known=bool(patterns),
    )
    for key in sorted(set(view_a) | set(view_b)):
        if view_a.get(key, _MISSING) == view_b.get(key, _MISSING):
            continue
        diff.changed.append(key)
        if any(pattern.match(key) for pattern in patterns):
            diff.on_axis.append(key)
        else:
            diff.off_axis.append(key)
    return diff


def describe_off_axis(diff: RelaxationDiff, max_items: int = 8) -> str:
    """The complaint to send back to the planner, in its own field names.

    Specific rather than general, for the reason the duplicate rejection is:
    "you also changed paths[P1].hops[0].predicate" is a correctable mistake and
    "you changed too much" is an argument.
    """
    shown = diff.off_axis[:max_items]
    more = len(diff.off_axis) - len(shown)
    listed = ", ".join(shown) + (f", and {more} more" if more > 0 else "")

    if not diff.on_axis:
        return (
            f"the revision did not change the constraint it was asked to "
            f"loosen ({diff.axis_key}); instead it changed {listed}"
        )
    return (
        f"the revision loosened the requested constraint ({diff.axis_key}) but "
        f"also changed {listed}, which was not asked for"
    )


# ---------------------------------------------------------------------------
# Locked constraints
# ---------------------------------------------------------------------------


def locked_constraints(plan: Any) -> List[str]:
    """Constraints the loop may not relax, as human-readable keys.

    Two kinds, for the same underlying reason: they are the user's, not the
    planner's, so loosening them answers a question nobody asked.

    The evidence policy is the sharp case. The planner's design note says the
    executor "must not weaken the filter when no candidates pass", and the same
    rule has to hold one level up, because the controller is the component that
    would otherwise be tempted: a policy that removed every candidate is
    exactly the situation where relaxing it produces results. Those results
    would be the answer to a question with the user's filter deleted.

    Pinned anchors are the other. An anchor is the concept the question is
    about. Widening or replacing it produces confident results about something
    else, which is the failure mode the executor's LLM-reviewed resolution and
    concept checks exist to prevent.
    """
    raw = _as_plain(plan)
    locked: List[str] = []

    policy = raw.get("evidence_policy")
    if not isinstance(policy, dict):
        policy = _as_plain(policy) if policy else {}
    if policy:
        found = False
        for field_name in EVIDENCE_POLICY_FILTERS:
            value = policy.get(field_name)
            if value in (None, False, [], {}):
                continue
            locked.append(f"evidence_policy:{field_name}")
            found = True
        if not found:
            # `origin: user_requested` is the only value the contract allows,
            # so a policy block that is present at all came from the user even
            # when this package does not recognise which field carries it.
            locked.append("evidence_policy")

    for entity in raw.get("entities") or []:
        ent = entity if isinstance(entity, dict) else _as_plain(entity)
        ref = ent.get("entity_ref") or ent.get("ref") or ent.get("name")
        if not ref:
            continue
        if ent.get("is_variable"):
            continue
        locked.append(f"anchor:{ref}")
        for constraint in ent.get("constraints") or []:
            con = constraint if isinstance(constraint, dict) else {}
            field_name = con.get("field") or con.get("attribute") or "constraint"
            # An approval constraint is only ever present because the user
            # asked for approved drugs; the planner does not add entity
            # constraints on its own initiative.
            locked.append(f"constraint:{ref}:{field_name}")

    return locked


def referenced_entity_refs(plan: Any) -> Set[str]:
    """Every entity_ref named anywhere in the plan except the entities block.

    Walks for keys ending in `_ref` rather than visiting each site that can
    hold one, because that list is long and grows: hop endpoints, return
    entities, explanation endpoint bindings, from_discovery bindings, ranking
    scopes. A check that must be extended whenever the contract grows a field
    is one that will eventually be wrong in the direction of passing.
    """
    raw = _as_plain(plan)
    found: Set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str) and key.endswith("_ref"):
                    found.add(value)
                else:
                    walk(value)
        elif isinstance(node, (list, tuple)):
            for item in node:
                walk(item)

    walk({k: v for k, v in raw.items() if k != "entities"})
    return found


def orphan_entities(plan: Any) -> List[Dict[str, Any]]:
    """Fixed entities the plan declares and then never queries.

    The failure this catches does not look like a failure. Asked "which drugs
    treat dermatitis herpetiformis by targeting CFTR?", a planner declared all
    three concepts and wrote one hop, from the drug to CFTR. The disease
    appeared in the entity list and in no path. That plan validated, executed,
    and returned twenty well-evidenced drugs — every claim about them backed by
    a real edge, and not one of them an answer to the question.

    No grounding gate can catch it, because nothing in the answer is
    ungrounded. The answer is grounded, and it is to a narrower question than
    the one asked. The only place the difference is visible is the plan, where
    a concept from the question sits unused.

    Only non-variable entities count. A variable describes the shape of what
    the plan is looking for, and an unreferenced one is inert; a fixed entity
    is a concept the user named, and dropping it changes the question.
    """
    raw = _as_plain(plan)
    referenced = referenced_entity_refs(raw)
    orphans: List[Dict[str, Any]] = []

    for entity in raw.get("entities") or []:
        ent = entity if isinstance(entity, dict) else _as_plain(entity)
        ref = ent.get("entity_ref") or ent.get("ref")
        if not ref or ent.get("is_variable") or ref in referenced:
            continue
        orphans.append({
            "entity_ref": str(ref),
            "name": ent.get("name"),
            "biolink_category": ent.get("biolink_category"),
        })
    return orphans


def pinned_entity_refs(plan: Any) -> Set[str]:
    """References of the non-variable entities: the question's anchors."""
    raw = _as_plain(plan)
    refs: Set[str] = set()
    for entity in raw.get("entities") or []:
        ent = entity if isinstance(entity, dict) else _as_plain(entity)
        if ent.get("is_variable"):
            continue
        ref = ent.get("entity_ref") or ent.get("ref")
        if ref:
            refs.add(str(ref))
    return refs
