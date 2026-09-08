"""
postfilter.py — Apply the plan's evidence policy to retrieved edges.

Everything here runs against edges ARAX already returned. Nothing is
re-queried, so filters can be changed and reapplied against the same cached
results — which is why the cache stores raw responses rather than filtered
ones.

Two sources of rules
--------------------
  * `evidence_policy` from the plan: provenance, publication counts,
    knowledge level, agent type.
  * `PostCondition` records from trapi_builder: things TRAPI cannot express in
    a query graph, so they have to be checked after retrieval —
    `approval_status`, `negated`, and `predicate_expansion: self_only`.

Absent attributes
-----------------
The hardest question here is what to do with an edge that simply does not
declare the attribute a filter tests. A large share of real ARAX edges carry
no `knowledge_level` and no publications — often the ones from curated
databases, which are frequently the strongest evidence available.

An absent attribute is treated as the Biolink value `not_provided`. That
choice makes the plan's own enum the control: a policy listing
`not_provided` keeps undeclared edges, one omitting it drops them. Nothing is
hidden either way, because drops are counted separately as `mismatch` (the
edge declared a disqualifying value) versus `missing` (the edge declared
nothing). A rule whose drops are overwhelmingly `missing` is filtering on
metadata completeness rather than on evidence quality, and the report says so.

Derived evidence strength
-------------------------
`min_evidence_strength` has no counterpart in TRAPI — no edge carries an
"evidence strength" field. It is derived from knowledge level, publication
count, and provenance. The mapping is a judgment call rather than a standard,
so it is defined in one place, `STRENGTH_RULES`, and can be replaced whole.

Usage
-----
    policy = EvidencePolicy.from_plan(plan)
    result = apply_to_execution(execution, policy, post_conditions)
    print(result.report())
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence, Set,
)


# ---------------------------------------------------------------------------
# Attribute access
#
# TRAPI edges carry attributes as a list of {attribute_type_id, value}, and
# different knowledge providers use different type_ids for the same idea.
# These readers absorb that variation so the rules below stay simple.
# ---------------------------------------------------------------------------

_PUBLICATION_KEYS = {
    "biolink:publications",
    "biolink:publication",
    "biolink:has_supporting_publications",
    "biolink:supporting_publications",
    "publications",
}

_KNOWLEDGE_LEVEL_KEYS = {"biolink:knowledge_level", "knowledge_level"}
_AGENT_TYPE_KEYS = {"biolink:agent_type", "agent_type"}
_NEGATED_KEYS = {"biolink:negated", "negated"}

NOT_PROVIDED = "not_provided"


def _iter_attributes(edge: Dict[str, Any]):
    """Yield (type_id, value) for an edge, descending into nested attributes."""
    for attr in edge.get("attributes") or []:
        if not isinstance(attr, dict):
            continue
        yield attr.get("attribute_type_id"), attr.get("value")
        for sub in attr.get("attributes") or []:
            if isinstance(sub, dict):
                yield sub.get("attribute_type_id"), sub.get("value")


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    return [value]


def edge_publications(edge: Dict[str, Any]) -> List[str]:
    """Publication identifiers on an edge, deduplicated.

    Only PMID/PMC identifiers are counted: some providers put URLs or internal
    record ids in the same slot, and those cannot be resolved to a paper.
    """
    out: Set[str] = set()
    for type_id, value in _iter_attributes(edge):
        if type_id in _PUBLICATION_KEYS:
            for v in _as_list(value):
                if isinstance(v, str) and (
                    v.upper().startswith("PMID") or v.upper().startswith("PMC")
                ):
                    out.add(v)
    return sorted(out)


def edge_knowledge_level(edge: Dict[str, Any]) -> str:
    for type_id, value in _iter_attributes(edge):
        if type_id in _KNOWLEDGE_LEVEL_KEYS and value:
            return _as_list(value)[0]
    return NOT_PROVIDED


def edge_agent_type(edge: Dict[str, Any]) -> str:
    for type_id, value in _iter_attributes(edge):
        if type_id in _AGENT_TYPE_KEYS and value:
            return _as_list(value)[0]
    return NOT_PROVIDED


def edge_negated(edge: Dict[str, Any]) -> bool:
    """True when the edge asserts a negative claim ("X does not treat Y")."""
    if edge.get("negated") is True:
        return True
    for type_id, value in _iter_attributes(edge):
        if type_id in _NEGATED_KEYS:
            return value is True or str(value).lower() == "true"
    return False


def edge_sources(edge: Dict[str, Any]) -> Dict[str, List[str]]:
    """Group an edge's sources by role."""
    out: Dict[str, List[str]] = {}
    for src in edge.get("sources") or []:
        if not isinstance(src, dict):
            continue
        role = src.get("resource_role") or "unknown"
        rid = src.get("resource_id")
        if rid:
            out.setdefault(role, []).append(rid)
    return out


def edge_primary_source(edge: Dict[str, Any]) -> Optional[str]:
    return (edge_sources(edge).get("primary_knowledge_source") or [None])[0]


def edge_all_sources(edge: Dict[str, Any]) -> Set[str]:
    return {rid for ids in edge_sources(edge).values() for rid in ids}


# ---------------------------------------------------------------------------
# Derived evidence strength
# ---------------------------------------------------------------------------

STRENGTH_ORDER = ["any", "weak", "moderate", "strong", "asserted_only"]


def derive_strength(edge: Dict[str, Any]) -> str:
    """Map an edge onto the plan's evidence-strength vocabulary.

    TRAPI has no such field, so this is a convention, not a standard:

        asserted_only  a manually curated assertion
        strong         an assertion, or a well-published claim
        moderate       has publications, or a non-predicted claim with a
                       primary source
        weak           has a primary source
        any            everything else

    Replace this function wholesale to change the convention; every rule that
    depends on evidence strength routes through it.
    """
    level = edge_knowledge_level(edge)
    agent = edge_agent_type(edge)
    pubs = len(edge_publications(edge))
    has_primary = edge_primary_source(edge) is not None

    if level == "knowledge_assertion" and agent in (
        "manual_agent", "manual_validation_of_automated_agent"
    ):
        return "asserted_only"
    if level == "knowledge_assertion" or pubs >= 3:
        return "strong"
    if pubs >= 1 or (has_primary and level not in ("prediction", NOT_PROVIDED)):
        return "moderate"
    if has_primary:
        return "weak"
    return "any"


def strength_at_least(edge: Dict[str, Any], minimum: str) -> bool:
    if not minimum or minimum == "any":
        return True
    try:
        return STRENGTH_ORDER.index(derive_strength(edge)) >= STRENGTH_ORDER.index(minimum)
    except ValueError:
        return True


# ---------------------------------------------------------------------------
# Constraint operators
# ---------------------------------------------------------------------------


def apply_op(op: str, actual: Any, expected: Any) -> bool:
    """Evaluate one EntityConstraint / EdgeFilter operator."""
    if op == "exists":
        return actual is not None
    if op == "not_exists":
        return actual is None
    if actual is None:
        return False

    actuals = _as_list(actual)
    try:
        if op == "eq":
            return any(str(a).lower() == str(expected).lower() for a in actuals)
        if op == "neq":
            return all(str(a).lower() != str(expected).lower() for a in actuals)
        if op == "in":
            wanted = {str(v).lower() for v in _as_list(expected)}
            return any(str(a).lower() in wanted for a in actuals)
        if op == "not_in":
            unwanted = {str(v).lower() for v in _as_list(expected)}
            return all(str(a).lower() not in unwanted for a in actuals)
        if op == "contains":
            return any(str(expected).lower() in str(a).lower() for a in actuals)
        if op in ("gte", "lte", "gt", "lt"):
            nums = [float(a) for a in actuals]
            exp = float(expected)
            return {
                "gte": any(n >= exp for n in nums),
                "lte": any(n <= exp for n in nums),
                "gt": any(n > exp for n in nums),
                "lt": any(n < exp for n in nums),
            }[op]
    except (TypeError, ValueError):
        return False
    return False


#: Where an attribute's name can hide. ARAX frequently types node attributes
#: as the generic `biolink:Attribute` and keeps the real name in
#: `original_attribute_name`, so matching on `attribute_type_id` alone finds
#: nothing — the constraint then drops every candidate for lack of an
#: attribute that is in fact present.
_ATTRIBUTE_NAME_FIELDS = (
    "attribute_type_id", "original_attribute_name", "value_type_id",
    "attribute_source", "description",
)


def _name_matches(attr: Dict[str, Any], field_name: str) -> bool:
    """True if any of an attribute's naming fields denotes `field_name`.

    Compared on the local part of a CURIE and with separators normalised, so
    `biolink:approval_status`, `approval_status` and `approval status` all
    match a constraint written as `approval_status`.
    """
    want = field_name.replace("-", "_").replace(" ", "_").lower()
    for key in _ATTRIBUTE_NAME_FIELDS:
        raw = attr.get(key)
        if not isinstance(raw, str):
            continue
        local = raw.split(":")[-1].replace("-", "_").replace(" ", "_").lower()
        if local == want:
            return True
    return False


def node_field(node: Dict[str, Any], field_name: str) -> Any:
    """Read a named field from a KG node, checking attributes as well."""
    if field_name in node:
        return node[field_name]

    for attr in node.get("attributes") or []:
        if not isinstance(attr, dict):
            continue
        if _name_matches(attr, field_name):
            return attr.get("value")
        for sub in attr.get("attributes") or []:
            if isinstance(sub, dict) and _name_matches(sub, field_name):
                return sub.get("value")
    return None


def node_attribute_names(node: Dict[str, Any]) -> List[str]:
    """Every name a node's attributes go by.

    Reported when a constraint finds nothing, so the plan can be pointed at a
    field that exists rather than left guessing.
    """
    names: List[str] = []
    for attr in node.get("attributes") or []:
        if not isinstance(attr, dict):
            continue
        for key in ("original_attribute_name", "attribute_type_id", "value_type_id"):
            raw = attr.get(key)
            if isinstance(raw, str) and raw:
                local = raw.split(":")[-1]
                if local and local not in names and local != "Attribute":
                    names.append(local)
                    break
    return names


# ---------------------------------------------------------------------------
# Policy
# ---------------------------------------------------------------------------


@dataclass
class EvidenceRecommendation:
    """Metadata the plan asks to be reported alongside results.

    Reporting only. A recommendation names something a reader should be able
    to see — provenance, knowledge level, publication counts — so that they
    can judge the evidence themselves. It never removes a candidate and never
    changes an order. Ranking preferences travel as ranking criteria instead,
    and hard filters as evidence_policy.
    """

    feature: str
    origin: str = "planner_recommended"
    application: str = "report"
    preferred_values: List[str] = field(default_factory=list)
    rationale: Optional[str] = None

    @classmethod
    def from_plan(cls, plan) -> List["EvidenceRecommendation"]:
        raw = getattr(plan, "evidence_recommendations", None)
        if raw is None and hasattr(plan, "get"):
            raw = plan.get("evidence_recommendations")
        out: List["EvidenceRecommendation"] = []
        for item in raw or []:
            if hasattr(item, "model_dump"):
                item = item.model_dump(exclude_none=True)
            elif not isinstance(item, dict):
                item = {k: v for k, v in vars(item).items() if v is not None}
            feature = item.get("feature")
            if feature:
                out.append(cls(
                    feature=feature,
                    origin=item.get("origin") or "planner_recommended",
                    application=item.get("application") or "report",
                    preferred_values=list(item.get("preferred_values") or []),
                    rationale=item.get("rationale"),
                ))
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature": self.feature, "origin": self.origin,
            "application": self.application,
            "preferred_values": self.preferred_values or None,
            "rationale": self.rationale,
        }


@dataclass
class EvidencePolicy:
    """Hard filters the user explicitly asked for.

    The plan contract reserves this block for user-requested filters:
    `origin` is fixed to `user_requested` and `application` to `filter`. A
    planner's own preference about evidence quality is not expressible here,
    and belongs in ranking criteria or evidence recommendations instead.

    That division exists because a preference applied as a requirement removes
    answers silently. Requiring one publication per edge, for instance, drops
    curated drug-target assertions that carry no PMID — the strongest evidence
    available for that relationship. Only a user who asked for it should get
    that behaviour.

    `knowledge_level` and `min_evidence_strength` are deliberately absent:
    neither is a filter under this contract. Knowledge level is a ranking
    criterion with ordered preferences, and derived evidence strength is an
    executor-specific ranking input.
    """

    origin: Optional[str] = None
    application: Optional[str] = None
    rationale: Optional[str] = None

    require_primary_knowledge_source: bool = False
    min_publications: int = 0
    min_year: Optional[int] = None
    agent_type: List[str] = field(default_factory=list)
    required_knowledge_sources: List[str] = field(default_factory=list)
    excluded_knowledge_sources: List[str] = field(default_factory=list)

    #: Filled by evidence.py once publication years are known. Until then
    #: min_year cannot be enforced, since TRAPI edges carry identifiers, not
    #: dates.
    publication_years: Dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_plan(cls, plan) -> "EvidencePolicy":
        """Build from a plan's evidence_policy block, or an empty policy.

        Most plans carry no policy at all, which is the intended default: no
        hard filtering unless the user asked for it.
        """
        raw = getattr(plan, "evidence_policy", None)
        if raw is None and hasattr(plan, "get"):
            raw = plan.get("evidence_policy")
        raw = raw or {}
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(exclude_none=True)
        elif hasattr(raw, "__dict__") and not isinstance(raw, dict):
            raw = {k: v for k, v in vars(raw).items() if v is not None}

        return cls(
            origin=raw.get("origin"),
            application=raw.get("application"),
            rationale=raw.get("rationale"),
            require_primary_knowledge_source=bool(
                raw.get("require_primary_knowledge_source", False)
            ),
            min_publications=int(raw.get("min_publications") or 0),
            min_year=raw.get("min_year"),
            agent_type=list(raw.get("agent_type") or []),
            required_knowledge_sources=list(raw.get("required_knowledge_sources") or []),
            excluded_knowledge_sources=list(raw.get("excluded_knowledge_sources") or []),
        )

    @property
    def is_active(self) -> bool:
        return bool(
            self.require_primary_knowledge_source or self.min_publications
            or self.min_year or self.agent_type
            or self.required_knowledge_sources or self.excluded_knowledge_sources
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "origin": self.origin, "application": self.application,
            "rationale": self.rationale, "active": self.is_active,
            "require_primary_knowledge_source": self.require_primary_knowledge_source,
            "min_publications": self.min_publications or None,
            "min_year": self.min_year,
            "agent_type": self.agent_type or None,
            "required_knowledge_sources": self.required_knowledge_sources or None,
            "excluded_knowledge_sources": self.excluded_knowledge_sources or None,
        }

    def rules(self) -> List["Rule"]:
        """Compile into individual rules so each one's cost can be reported."""
        out: List[Rule] = []

        if self.require_primary_knowledge_source:
            out.append(Rule(
                "require_primary_knowledge_source",
                lambda e: edge_primary_source(e) is not None,
                lambda e: "missing",
            ))

        if self.min_publications:
            n = self.min_publications
            out.append(Rule(
                f"min_publications>={n}",
                lambda e: len(edge_publications(e)) >= n,
                lambda e: "missing" if not edge_publications(e) else "mismatch",
            ))

        if self.agent_type:
            allowed = set(self.agent_type)
            out.append(Rule(
                f"agent_type in {sorted(allowed)}",
                lambda e: edge_agent_type(e) in allowed,
                lambda e: "missing" if edge_agent_type(e) == NOT_PROVIDED
                else "mismatch",
            ))

        if self.required_knowledge_sources:
            required = set(self.required_knowledge_sources)
            out.append(Rule(
                f"knowledge_source in {sorted(required)}",
                lambda e: bool(edge_all_sources(e) & required),
                lambda e: "missing" if not edge_all_sources(e) else "mismatch",
            ))

        if self.excluded_knowledge_sources:
            excluded = set(self.excluded_knowledge_sources)
            out.append(Rule(
                f"knowledge_source not in {sorted(excluded)}",
                lambda e: not (edge_all_sources(e) & excluded),
                lambda e: "mismatch",
            ))

        if self.min_year and self.publication_years:
            year = self.min_year
            years = self.publication_years

            def recent_enough(e):
                pubs = edge_publications(e)
                known = [years[p] for p in pubs if p in years]
                return max(known) >= year if known else False

            out.append(Rule(
                f"min_year>={year}",
                recent_enough,
                lambda e: "missing" if not any(
                    p in years for p in edge_publications(e)
                ) else "mismatch",
            ))

        return out

    def deferred(self) -> List[str]:
        """Rules that cannot run yet, and why."""
        out = []
        if self.min_year and not self.publication_years:
            out.append(
                f"min_year>={self.min_year} not applied: publication years are "
                f"not on TRAPI edges. Populate policy.publication_years from "
                f"evidence.py first."
            )
        return out


@dataclass
class Rule:
    """One checkable condition, with a classifier for why an edge failed."""

    name: str
    test: Callable[[Dict[str, Any]], bool]
    reason: Callable[[Dict[str, Any]], str]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class RuleStats:
    name: str
    checked: int = 0
    dropped: int = 0
    dropped_missing: int = 0
    dropped_mismatch: int = 0

    @property
    def drop_rate(self) -> float:
        return self.dropped / self.checked if self.checked else 0.0

    @property
    def mostly_missing(self) -> bool:
        """True when this rule is chiefly filtering on absent metadata.

        Worth surfacing: such a rule is selecting for how completely a source
        annotates its edges, not for how good the evidence is.
        """
        return self.dropped > 0 and self.dropped_missing / self.dropped > 0.7

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rule": self.name,
            "checked": self.checked,
            "dropped": self.dropped,
            "dropped_missing_attribute": self.dropped_missing,
            "dropped_value_mismatch": self.dropped_mismatch,
            "drop_rate": round(self.drop_rate, 3),
            "mostly_missing_metadata": self.mostly_missing,
        }


@dataclass
class FilterResult:
    kept_edge_ids: Set[str] = field(default_factory=set)
    dropped_edge_ids: Set[str] = field(default_factory=set)
    kept_instances: List[Any] = field(default_factory=list)
    dropped_instances: int = 0
    instance_drop_reasons: Dict[str, int] = field(default_factory=dict)
    rule_stats: List[RuleStats] = field(default_factory=list)
    deferred: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def kept_candidates(self) -> List[str]:
        return sorted({
            i.return_curie for i in self.kept_instances
            if getattr(i, "return_curie", None)
        })

    def to_dict(self) -> Dict[str, Any]:
        return {
            "edges_kept": len(self.kept_edge_ids),
            "edges_dropped": len(self.dropped_edge_ids),
            "instances_kept": len(self.kept_instances),
            "instances_dropped": self.dropped_instances,
            "instance_drop_reasons": self.instance_drop_reasons,
            "candidates_kept": len(self.kept_candidates),
            "rules": [r.to_dict() for r in self.rule_stats],
            "deferred": self.deferred,
            "warnings": self.warnings,
        }

    def report(self) -> str:
        lines = [
            f"edges  {len(self.kept_edge_ids)} kept / "
            f"{len(self.dropped_edge_ids)} dropped",
            f"paths  {len(self.kept_instances)} kept / "
            f"{self.dropped_instances} dropped",
        ]
        for r in self.rule_stats:
            if not r.checked:
                continue
            flag = "  <- mostly missing metadata" if r.mostly_missing else ""
            lines.append(
                f"    {r.name}: dropped {r.dropped}/{r.checked} "
                f"(missing {r.dropped_missing}, mismatch {r.dropped_mismatch})"
                f"{flag}"
            )
        for reason, n in sorted(self.instance_drop_reasons.items()):
            lines.append(f"    path rule {reason}: dropped {n}")
        for d in self.deferred:
            lines.append(f"    DEFERRED {d}")
        for w in self.warnings:
            lines.append(f"    WARNING {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_edges(
    edges: Dict[str, Dict[str, Any]],
    policy: EvidencePolicy,
    post_conditions: Sequence[Any] = (),
) -> tuple:
    """Apply the policy and edge-level post-conditions.

    Returns (kept_ids, dropped_ids, rule_stats). Every rule is evaluated on
    every edge rather than short-circuiting, so the report shows what each
    rule would cost independently — a rule that drops nothing on its own can
    still be the one making a plan unanswerable when combined with others.
    """
    rules = policy.rules()

    for pc in post_conditions or ():
        kind = getattr(pc, "kind", None)
        detail = getattr(pc, "detail", {}) or {}

        if kind == "negated_edge" and detail.get("require_negated"):
            rules.append(Rule("require_negated_edge", edge_negated, lambda e: "mismatch"))
        elif kind == "predicate_self_only":
            exact = detail.get("predicate")
            rules.append(Rule(
                f"predicate=={exact} (self_only)",
                lambda e, exact=exact: e.get("predicate") == exact,
                lambda e: "mismatch",
            ))

    stats = [RuleStats(name=r.name) for r in rules]
    kept: Set[str] = set()
    dropped: Set[str] = set()

    for eid, edge in edges.items():
        failed = False
        for rule, st in zip(rules, stats):
            st.checked += 1
            if not rule.test(edge):
                st.dropped += 1
                if rule.reason(edge) == "missing":
                    st.dropped_missing += 1
                else:
                    st.dropped_mismatch += 1
                failed = True
        (dropped if failed else kept).add(eid)

    return kept, dropped, stats


def _check_constraint(node: Dict[str, Any], constraint: Dict[str, Any]) -> tuple:
    """Evaluate one entity constraint against a node.

    Returns (passed, outcome, note). `outcome` distinguishes a value that
    disqualifies the node from an attribute the node does not carry, since
    only the first is a statement about the candidate.
    """
    from .config import resolve_constraint

    field = constraint.get("field")
    fields, op, value, note = resolve_constraint(
        field, constraint.get("op"), constraint.get("value")
    )

    for candidate_field in fields:
        actual = node_field(node, candidate_field)
        if actual is None:
            continue
        return apply_op(op, actual, value), "mismatch", note

    return False, "missing", note


def filter_instances(
    instances: Sequence[Any],
    kept_edge_ids: Set[str],
    nodes: Dict[str, Dict[str, Any]],
    post_conditions: Sequence[Any] = (),
    node_map: Optional[Dict[str, str]] = None,
    drop_negated_paths: bool = True,
    unique_intermediate_nodes: bool = True,
    edges: Optional[Dict[str, Dict[str, Any]]] = None,
) -> tuple:
    """Drop path instances invalidated by edge filtering or node constraints.

    An instance survives only if every edge supporting it survived: a path is
    a conjunction, so one unsupported hop invalidates the whole traversal.
    """
    kept: List[Any] = []
    reasons: Dict[str, int] = {}
    edges = edges or {}

    # entity_ref -> constraints, from the builder's entity post-conditions.
    entity_constraints: Dict[str, List[Dict[str, Any]]] = {}
    for pc in post_conditions or ():
        if getattr(pc, "kind", None) == "entity_constraint":
            ref = getattr(pc, "entity_ref", None)
            if ref:
                entity_constraints.setdefault(ref, []).extend(
                    (getattr(pc, "detail", {}) or {}).get("constraints", [])
                )

    applied_mappings: Set[str] = set()

    def note(reason: str) -> None:
        reasons[reason] = reasons.get(reason, 0) + 1

    for inst in instances:
        edge_ids = getattr(inst, "edge_ids", []) or []
        bindings = getattr(inst, "bindings", {}) or {}

        if edge_ids and not all(eid in kept_edge_ids for eid in edge_ids):
            note("unsupported_edge")
            continue

        if drop_negated_paths and any(
            edge_negated(edges.get(eid, {})) for eid in edge_ids
        ):
            note("negated_edge")
            continue

        # The same node appearing twice means the path loops back on itself,
        # which is a traversal artefact rather than a mechanism.
        if unique_intermediate_nodes:
            curies = list(bindings.values())
            if len(curies) != len(set(curies)):
                note("repeated_node")
                continue

        violated = False
        for ref, constraints in entity_constraints.items():
            curie = bindings.get(ref)
            if not curie:
                continue
            node = nodes.get(curie) or {}
            for c in constraints:
                passed, kind, mapping_note = _check_constraint(node, c)
                if mapping_note:
                    applied_mappings.add(mapping_note)
                if not passed:
                    # An absent attribute and a disqualifying value both fail
                    # the check, but they mean different things: the first says
                    # the constraint cannot be evaluated against this data, the
                    # second that the candidate genuinely does not qualify.
                    note(f"constraint:{ref}.{c.get('field')}[{kind}]")
                    violated = True
                    break
            if violated:
                break
        if violated:
            continue

        kept.append(inst)

    return kept, reasons, sorted(applied_mappings)


def apply_to_execution(
    execution,
    policy: EvidencePolicy,
    post_conditions: Sequence[Any] = (),
    drop_negated_paths: bool = True,
    unique_intermediate_nodes: bool = True,
    mutate: bool = False,
) -> FilterResult:
    """Filter one PathExecution's edges and instances.

    Args:
        mutate: replace the execution's instances and edges with the filtered
            set. Off by default so the unfiltered result stays available for
            re-filtering under a revised policy without re-querying.
    """
    kept_ids, dropped_ids, stats = filter_edges(
        execution.kg_edges, policy, post_conditions
    )

    kept_instances, reasons, mappings = filter_instances(
        execution.instances,
        kept_edge_ids=kept_ids,
        nodes=execution.kg_nodes,
        post_conditions=post_conditions,
        drop_negated_paths=drop_negated_paths,
        unique_intermediate_nodes=unique_intermediate_nodes,
        edges=execution.kg_edges,
    )

    result = FilterResult(
        kept_edge_ids=kept_ids,
        dropped_edge_ids=dropped_ids,
        kept_instances=kept_instances,
        dropped_instances=len(execution.instances) - len(kept_instances),
        instance_drop_reasons=reasons,
        rule_stats=stats,
        deferred=policy.deferred(),
    )

    # Reported whether or not anything was dropped: expanding a plan's stated
    # value into the graph's own vocabulary changes what the constraint means,
    # and a reader comparing the plan against the result should not have to
    # infer that translation.
    for mapping in mappings:
        result.warnings.append(f"constraint vocabulary: {mapping}")

    # A constraint that drops everything because the attribute is absent is
    # not a strict filter — it is an unenforceable one, and saying so points at
    # a different fix than tightening or loosening a threshold would.
    for reason, count in reasons.items():
        if not (reason.startswith("constraint:") and reason.endswith("[missing]")):
            continue
        field_ref = reason[len("constraint:"):-len("[missing]")]
        if count == len(execution.instances):
            available = []
            for inst in execution.instances[:5]:
                ref_name = field_ref.split(".")[0]
                curie = (getattr(inst, "bindings", {}) or {}).get(ref_name)
                for name in node_attribute_names(execution.kg_nodes.get(curie) or {}):
                    if name not in available:
                        available.append(name)
            hint = (
                f" Attributes these nodes do carry: {available[:12]}."
                if available else ""
            )
            result.warnings.append(
                f"constraint on '{field_ref}' dropped all {count} candidate(s) "
                f"because no result node carries that attribute, so the "
                f"constraint cannot be evaluated against this knowledge "
                f"graph.{hint}"
            )
        elif count:
            result.warnings.append(
                f"constraint on '{field_ref}' dropped {count} candidate(s) that "
                f"do not declare the attribute at all, as distinct from "
                f"declaring a disqualifying value"
            )

    # Stated after the constraint diagnostics above, so the specific cause
    # leads and this general observation follows it.
    if execution.instances and not kept_instances:
        worst = max(
            (r for r in stats if r.dropped), key=lambda r: r.dropped, default=None
        )
        detail = f"; largest contributor: {worst.name}" if worst else ""
        result.warnings.append(
            f"evidence policy removed all {len(execution.instances)} path(s) "
            f"found by {execution.path_id}{detail}. Results exist but do not "
            f"meet the plan's evidence bar."
        )

    for r in stats:
        if r.mostly_missing and r.drop_rate > 0.3:
            result.warnings.append(
                f"rule '{r.name}' dropped {r.dropped} edges, "
                f"{r.dropped_missing} of them only because the attribute was "
                f"absent. This filters on annotation completeness rather than "
                f"evidence quality."
            )

    if mutate:
        execution.instances = kept_instances
        execution.kg_edges = {
            eid: e for eid, e in execution.kg_edges.items() if eid in kept_ids
        }

    return result


def summarize_edge(edge: Dict[str, Any]) -> Dict[str, Any]:
    """Compact view of an edge's evidence, for the ledger and LLM reranking."""
    return {
        "subject": edge.get("subject"),
        "predicate": edge.get("predicate"),
        "object": edge.get("object"),
        "primary_source": edge_primary_source(edge),
        "sources": sorted(edge_all_sources(edge))[:6],
        "knowledge_level": edge_knowledge_level(edge),
        "agent_type": edge_agent_type(edge),
        "publications": edge_publications(edge)[:10],
        "num_publications": len(edge_publications(edge)),
        "derived_strength": derive_strength(edge),
        "negated": edge_negated(edge),
    }


# ---------------------------------------------------------------------------
# EPC confidence scoring
#
# Evidence, provenance and confidence, reduced to one number per edge for
# ranking. This never removes anything: filtering is the thresholds' job, and
# scoring exists so that what survives can be ordered by how well supported it
# is. A low score is a statement about how much is known, not a verdict.
# ---------------------------------------------------------------------------

from .config import (                                          # noqa: E402
    AGENT_TYPE_SCORE, EPC_COMPONENT_WEIGHTS, KNOWLEDGE_LEVEL_SCORE,
    SOURCE_COUNT_MAX, SOURCE_COUNT_SCORE, primary_source_score,
    publication_score,
)


def score_epc(edge: Dict[str, Any]) -> Dict[str, Any]:
    """Score one edge's evidence, provenance and confidence.

    Returns the total in [0, 1] alongside each component, so a candidate can
    be explained by what supports it rather than only by a number.
    """
    level = edge_knowledge_level(edge)
    agent = edge_agent_type(edge)
    pubs = edge_publications(edge)
    sources = edge_all_sources(edge)
    primary = edge_primary_source(edge)

    components = {
        "knowledge_level": KNOWLEDGE_LEVEL_SCORE.get(level, 0.5),
        "agent_type": AGENT_TYPE_SCORE.get(agent, 0.5),
        "publications": publication_score(len(pubs)),
        "source_count": SOURCE_COUNT_SCORE.get(len(sources), SOURCE_COUNT_MAX),
        "primary_source": primary_source_score(primary),
    }
    total = sum(components[k] * w for k, w in EPC_COMPONENT_WEIGHTS.items())

    return {
        "epc_score": round(total, 4),
        "components": {k: round(v, 3) for k, v in components.items()},
        "knowledge_level": level,
        "agent_type": agent,
        "num_publications": len(pubs),
        "num_sources": len(sources),
        "primary_source": primary,
    }


def score_instance(
    instance,
    edges: Dict[str, Dict[str, Any]],
    aggregate: str = "min",
) -> Dict[str, Any]:
    """Score a whole path from its edges.

    Defaults to the minimum rather than the mean: a path is a conjunction, so
    its credibility is bounded by its weakest link. One well-evidenced hop
    cannot rescue a chain that depends on a speculative one, and averaging
    would let it appear to.
    """
    edge_ids = getattr(instance, "edge_ids", []) or []
    scores = [score_epc(edges[eid])["epc_score"] for eid in edge_ids if eid in edges]
    if not scores:
        return {"epc_score": 0.0, "num_edges": 0, "aggregate": aggregate}

    value = {
        "min": min(scores),
        "mean": sum(scores) / len(scores),
        "product": _product(scores),
    }.get(aggregate, min(scores))

    return {
        "epc_score": round(value, 4),
        "weakest_edge": round(min(scores), 4),
        "strongest_edge": round(max(scores), 4),
        "num_edges": len(scores),
        "aggregate": aggregate,
    }


def _product(values: Sequence[float]) -> float:
    out = 1.0
    for v in values:
        out *= v
    return out


# ---------------------------------------------------------------------------
# Distribution reporting
# ---------------------------------------------------------------------------


def distribution(edges: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize evidence metadata across edges, before any filtering.

    Reported pre-filter because its purpose is to let a controlling agent
    choose a threshold. A post-filter histogram only shows what already
    passed, which cannot tell anyone whether a threshold should be loosened.
    """
    from collections import Counter

    levels: Counter = Counter()
    agents: Counter = Counter()
    primaries: Counter = Counter()
    pub_buckets: Counter = Counter()
    epc_buckets: Counter = Counter()
    negated = 0
    total = len(edges)

    for edge in edges.values():
        levels[edge_knowledge_level(edge)] += 1
        agents[edge_agent_type(edge)] += 1
        primaries[edge_primary_source(edge) or "none"] += 1
        n = len(edge_publications(edge))
        pub_buckets[
            "0" if n == 0 else "1" if n == 1 else "2-4" if n < 5 else "5-9" if n < 10 else "10+"
        ] += 1
        score = score_epc(edge)["epc_score"]
        epc_buckets[f"{int(score * 10) / 10:.1f}"] += 1
        if edge_negated(edge):
            negated += 1

    return {
        "total_edges": total,
        "knowledge_level": dict(levels.most_common()),
        "agent_type": dict(agents.most_common()),
        "publications": dict(pub_buckets.most_common()),
        "primary_source": dict(primaries.most_common(15)),
        "epc_score_deciles": dict(sorted(epc_buckets.items())),
        "negated_edges": negated,
        "pct_knowledge_level_provided": round(
            100 * (total - levels.get(NOT_PROVIDED, 0)) / total, 1
        ) if total else 0.0,
        "pct_with_publications": round(
            100 * (total - pub_buckets.get("0", 0)) / total, 1
        ) if total else 0.0,
    }


def threshold_preview(
    edges: Dict[str, Dict[str, Any]],
    levels: Sequence[str] = ("knowledge_assertion", "logical_entailment",
                             "observation", "statistical_association",
                             "prediction", NOT_PROVIDED),
) -> Dict[str, Any]:
    """How many edges survive each candidate knowledge-level threshold.

    A controlling agent choosing a threshold needs to know its cost before
    imposing it; this answers that without re-querying.
    """
    out = {}
    allowed: Set[str] = set()
    for level in levels:
        allowed.add(level)
        out[f"allow {sorted(allowed)}"] = sum(
            1 for e in edges.values() if edge_knowledge_level(e) in allowed
        )
    return out


# ---------------------------------------------------------------------------
# Result-concept verification
#
# ARAX normalizes CURIEs inside its own NodeSynonymizer, and that
# normalization can merge concepts that the plan meant to keep apart. Pinning
# MONDO:0008345 is therefore no guarantee that results are about idiopathic
# pulmonary fibrosis.
#
# Checks escalate. The bound identifier and the returned node's name are
# already in the response, so drift is usually detectable for free; an LLM is
# consulted only when the cheap checks disagree.
# ---------------------------------------------------------------------------

CONCEPT_MATCH = "match"
CONCEPT_CURIE_DRIFT = "curie_drift"
CONCEPT_LABEL_DRIFT = "label_drift"
CONCEPT_CATEGORY_MISMATCH = "category_mismatch"
CONCEPT_ABSENT = "absent"
CONCEPT_LLM_CONFIRMED = "llm_confirmed"
CONCEPT_LLM_REJECTED = "llm_rejected"


@dataclass
class ConceptCheck:
    """Whether results are about the concept the plan asked for."""

    entity_ref: str
    requested_curie: Optional[str] = None
    requested_label: Optional[str] = None
    observed_curies: List[str] = field(default_factory=list)
    observed_labels: List[str] = field(default_factory=list)
    status: str = CONCEPT_MATCH
    needs_llm: bool = False
    detail: str = ""

    @property
    def ok(self) -> bool:
        return self.status in (CONCEPT_MATCH, CONCEPT_LLM_CONFIRMED)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "entity_ref": self.entity_ref,
            "requested_curie": self.requested_curie,
            "requested_label": self.requested_label,
            "observed_curies": self.observed_curies[:5],
            "observed_labels": self.observed_labels[:5],
            "status": self.status,
            "needs_llm": self.needs_llm,
            "ok": self.ok,
            "detail": self.detail,
        }


def check_anchor_concept(
    execution,
    entity_ref: str,
    requested_curie: str,
    requested_label: Optional[str] = None,
) -> ConceptCheck:
    """Verify results are bound to the concept that was pinned.

    A different identifier coming back for a pinned node means ARAX resolved
    it to something else. That may be a harmless synonym or a genuine
    conflation of two concepts, and only the second is a problem — which is
    why drift is flagged for judgement rather than treated as an error.
    """
    observed = sorted({
        inst.bindings.get(entity_ref)
        for inst in execution.instances
        if inst.bindings.get(entity_ref)
    })
    # A node with no name is left as None rather than falling back to its
    # CURIE: comparing a CURIE against a label would report drift on every
    # unnamed node, and unnamed nodes are common enough that the check would
    # become noise.
    labels = [(execution.kg_nodes.get(c) or {}).get("name") for c in observed]

    check = ConceptCheck(
        entity_ref=entity_ref,
        requested_curie=requested_curie,
        requested_label=requested_label,
        observed_curies=observed,
        observed_labels=[l or c for l, c in zip(labels, observed)],
    )

    if not observed:
        check.status = CONCEPT_ABSENT
        check.detail = f"no result bound '{entity_ref}'"
        return check

    if observed == [requested_curie]:
        observed_label = labels[0]
        if (
            requested_label
            and observed_label
            and observed_label != requested_curie
            and observed_label.strip().lower() != requested_label.strip().lower()
        ):
            check.status = CONCEPT_LABEL_DRIFT
            check.needs_llm = True
            check.detail = (
                f"CURIE matches but ARAX calls {requested_curie} "
                f"'{observed_label}' where the resolver called it "
                f"'{requested_label}'"
            )
        return check

    check.status = CONCEPT_CURIE_DRIFT
    check.needs_llm = True
    extra = [c for c in observed if c != requested_curie]
    check.detail = (
        f"pinned {requested_curie} but results bound {extra[:3]}; ARAX may "
        f"have conflated it with another concept"
    )
    return check


def check_result_category(
    execution,
    return_entity_ref: str,
    expected_category: Optional[str],
) -> ConceptCheck:
    """Verify returned candidates carry the category the path expects."""
    check = ConceptCheck(entity_ref=return_entity_ref,
                         requested_label=expected_category)
    if not expected_category:
        return check

    want = expected_category.replace("biolink:", "")
    wrong: List[str] = []
    for curie in execution.candidates:
        node = execution.kg_nodes.get(curie) or {}
        cats = [c.replace("biolink:", "") for c in (node.get("categories") or [])]
        if cats and want not in cats:
            wrong.append(f"{curie}({','.join(cats[:2])})")

    check.observed_curies = wrong[:5]
    if wrong:
        check.status = CONCEPT_CATEGORY_MISMATCH
        check.detail = (
            f"{len(wrong)} of {len(execution.candidates)} candidates are not "
            f"{want}: {wrong[:3]}"
        )
    return check


class ConceptVerifier(Protocol):
    """Judges whether two descriptions name the same concept. In llm.py."""

    def same_concept(
        self,
        question: str,
        requested_curie: str,
        requested_label: str,
        observed_curie: str,
        observed_label: str,
        observed_synonyms: Sequence[str] = (),
    ) -> tuple:
        """Returns (is_same: bool, reason: str)."""
        ...


def resolve_concept_check(
    check: ConceptCheck,
    question: str,
    verifier: Optional[ConceptVerifier],
    synonyms: Optional[Dict[str, List[str]]] = None,
) -> ConceptCheck:
    """Escalate a flagged check to the LLM.

    `synonyms` should come from ARAX's own /entity endpoint rather than a
    third-party normalizer: the question is what ARAX believes the CURIE
    means, since that belief is what produced these results.
    """
    if not check.needs_llm or verifier is None:
        return check

    observed_curie = check.observed_curies[0] if check.observed_curies else ""
    observed_label = check.observed_labels[0] if check.observed_labels else ""

    try:
        is_same, reason = verifier.same_concept(
            question=question,
            requested_curie=check.requested_curie or "",
            requested_label=check.requested_label or "",
            observed_curie=observed_curie,
            observed_label=observed_label,
            observed_synonyms=(synonyms or {}).get(observed_curie, []),
        )
    except Exception as e:
        check.detail += f"; concept verification unavailable ({e})"
        return check

    check.status = CONCEPT_LLM_CONFIRMED if is_same else CONCEPT_LLM_REJECTED
    check.needs_llm = False
    check.detail += f"; LLM: {reason}"
    return check
