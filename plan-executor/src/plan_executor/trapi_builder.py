"""
trapi_builder.py — Turn plan paths into TRAPI query graphs.

Two products, from one place so they cannot drift apart:

  1. `build_path_query()` — the full N-hop query graph for a path. This is
     what the executor tries first.
  2. `plan_decomposition()` + `build_hop_query()` — the single-hop pieces used
     when the full query times out. Decomposition runs the hop touching a
     resolved CURIE first, then feeds the intermediates it found into the next
     hop as pinned ids.

Deterministic node naming
-------------------------
Node keys are assigned structurally, never arbitrarily:

  * A standalone hop query is always `n00` (subject), `n01` (object), `e00`.
  * A full path query numbers nodes by order of first appearance walking hops
    in order.

This is what makes the cache work across paths. In the IPF example, P1 and P2
share an identical first hop; because both serialize to the same node keys,
the second path's hop 1 is a cache hit rather than a second ARAX call.

Post-conditions
---------------
Several things a plan can ask for have no TRAPI query-graph representation.
Rather than silently dropping them or pretending ARAX will honour them, the
builder emits them as `PostCondition` records for `postfilter.py` to apply
against returned edges:

  * `entity_constraint`   — e.g. approval_status == approved. TRAPI attribute
                            constraints exist but ARAX support is uneven, so
                            these are applied locally.
  * `predicate_self_only` — plans can request `predicate_expansion:
                            "self_only"`, but TRAPI has no per-edge flag to
                            suppress descendant expansion. Enforced by
                            dropping edges whose predicate is not the exact
                            one requested.
  * `negated_edge`        — a query graph cannot ask for negative assertions.
                            The edge is queried normally and the negation is
                            checked on the returned edge's Biolink `negated`
                            attribute.

Every post-condition carries the node or edge key it applies to, so filtering
never has to guess which part of the result it governs.

Usage
-----
    from .trapi_builder import build_path_query, plan_decomposition, to_trapi_message

    built = build_path_query(path, plan.entities, resolutions)
    message = to_trapi_message(built, max_results=5000)

    for step in plan_decomposition(path, plan.entities, resolutions):
        ...   # executor runs these in order, feeding results forward
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------

#: ARAX's auto-generated processing plan ends with
#: `filter_results(action=limit_number_of_results, max_results=500)`. When a
#: hop is run with hundreds of pinned intermediates, one result per
#: (subject, object) pair easily exceeds that, and the truncation is silent.
#: Submitting explicit operations with a raised ceiling avoids losing results
#: without knowing it; the executor compares `num_results` against this to
#: detect that truncation still happened.
DEFAULT_MAX_RESULTS = 5000

#: Biolink qualifier slot names are prefixed in TRAPI's `qualifier_type_id`
#: but written bare in the plan. Values are passed through untouched, since
#: some are CURIEs (`biolink:causes`) and some are plain enums (`decreased`).
_QUALIFIER_FIELDS = (
    "qualified_predicate",
    "object_aspect_qualifier",
    "object_direction_qualifier",
    "subject_aspect_qualifier",
    "subject_direction_qualifier",
    "causal_mechanism_qualifier",
    "anatomical_context_qualifier",
    "species_context_qualifier",
)


# ---------------------------------------------------------------------------
# Prefix normalization
# ---------------------------------------------------------------------------


def biolink_category(value: str) -> str:
    """Normalize a category to its prefixed form.

    Plans write `SmallMolecule` (the schema enumerates bare names) while TRAPI
    requires `biolink:SmallMolecule`.
    """
    if not value:
        return value
    return value if value.startswith("biolink:") else f"biolink:{value}"


def biolink_predicate(value: str) -> str:
    """Normalize a predicate to its prefixed form."""
    if not value:
        return value
    return value if value.startswith("biolink:") else f"biolink:{value}"


def _categories_of(entity) -> List[str]:
    cat = getattr(entity, "biolink_category", None)
    return [biolink_category(cat)] if cat else []


def _qualifiers_to_dict(qualifiers) -> Dict[str, str]:
    """Read a Qualifiers model (or plain dict) into a flat dict, dropping unset."""
    if qualifiers is None:
        return {}
    out: Dict[str, str] = {}
    for fname in _QUALIFIER_FIELDS:
        val = (
            qualifiers.get(fname)
            if isinstance(qualifiers, dict)
            else getattr(qualifiers, fname, None)
        )
        if val:
            out[fname] = val
    return out


def _constraints_to_dicts(constraints) -> List[Dict[str, Any]]:
    """Read EntityConstraint models (or dicts) into plain dicts."""
    out: List[Dict[str, Any]] = []
    for c in constraints or []:
        if isinstance(c, dict):
            out.append(dict(c))
        else:
            out.append(
                {
                    "field": getattr(c, "field", None),
                    "op": getattr(c, "op", None),
                    "value": getattr(c, "value", None),
                }
            )
    return out


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class PostCondition:
    """A plan requirement TRAPI cannot express, to be applied after retrieval."""

    kind: str                       # entity_constraint | predicate_self_only | negated_edge
    target_key: str                 # query-graph node or edge key it governs
    detail: Dict[str, Any] = field(default_factory=dict)
    entity_ref: Optional[str] = None
    hop_index: Optional[int] = None

    def __repr__(self) -> str:
        return f"<PostCondition {self.kind} on {self.target_key} {self.detail}>"


@dataclass
class BuiltQuery:
    """A TRAPI query graph plus the bookkeeping needed to interpret results."""

    query_graph: Dict[str, Any]
    node_map: Dict[str, str] = field(default_factory=dict)   # entity_ref -> node key
    edge_map: Dict[int, str] = field(default_factory=dict)   # hop index  -> edge key
    return_node_key: Optional[str] = None
    post_conditions: List[PostCondition] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    path_id: Optional[str] = None

    @property
    def node_keys(self) -> List[str]:
        return list(self.query_graph.get("nodes", {}))

    @property
    def edge_keys(self) -> List[str]:
        return list(self.query_graph.get("edges", {}))

    def entity_for_node(self, node_key: str) -> Optional[str]:
        for ref, key in self.node_map.items():
            if key == node_key:
                return ref
        return None

    def __repr__(self) -> str:
        return (
            f"<BuiltQuery path={self.path_id} nodes={len(self.node_keys)} "
            f"edges={len(self.edge_keys)} post_conditions={len(self.post_conditions)}>"
        )


@dataclass
class HopStep:
    """One stage of a pivot-first decomposition.

    `pinned_ref` is the endpoint whose CURIEs are known when this step runs —
    either a plan anchor or the output of an earlier step. `solve_ref` is the
    endpoint this step discovers.
    """

    order: int
    hop_index: int
    hop: Any
    pinned_ref: str
    solve_ref: str
    source: str                     # 'anchor' | 'previous_step'
    pinned_is_subject: bool

    def __repr__(self) -> str:
        arrow = "->" if self.pinned_is_subject else "<-"
        return (
            f"<HopStep {self.order}: {self.pinned_ref} {arrow} {self.solve_ref} "
            f"(hop {self.hop_index}, from {self.source})>"
        )


# ---------------------------------------------------------------------------
# Node construction
# ---------------------------------------------------------------------------


def _build_node(
    entity,
    ids: Optional[Sequence[str]] = None,
    is_set: bool = False,
) -> Dict[str, Any]:
    """Build a QNode.

    Only fields carrying meaning are emitted — TRAPI defaults such as
    `is_set: false` are omitted so the serialized graph matches what
    `cache.make_key` canonicalizes to.
    """
    node: Dict[str, Any] = {}
    cats = _categories_of(entity)
    if cats:
        node["categories"] = cats
    if ids:
        node["ids"] = sorted(set(ids))
    if is_set:
        node["is_set"] = True
    return node


def _build_edge(hop, subject_key: str, object_key: str) -> Dict[str, Any]:
    """Build a QEdge, including qualifier constraints when the hop has them."""
    edge: Dict[str, Any] = {
        "subject": subject_key,
        "object": object_key,
        "predicates": [biolink_predicate(hop.predicate)],
    }

    quals = _qualifiers_to_dict(getattr(hop, "qualifiers", None))
    if quals:
        edge["qualifier_constraints"] = [
            {
                "qualifier_set": [
                    {
                        "qualifier_type_id": biolink_predicate(k),
                        "qualifier_value": v,
                    }
                    for k, v in sorted(quals.items())
                ]
            }
        ]
    return edge


def _post_conditions_for_hop(hop, hop_index: int, edge_key: str) -> List[PostCondition]:
    """Collect the parts of a hop that TRAPI cannot carry."""
    out: List[PostCondition] = []

    if getattr(hop, "negated", False):
        out.append(
            PostCondition(
                kind="negated_edge",
                target_key=edge_key,
                hop_index=hop_index,
                detail={
                    "require_negated": True,
                    "predicate": biolink_predicate(hop.predicate),
                },
            )
        )

    if getattr(hop, "predicate_expansion", "descendants") == "self_only":
        out.append(
            PostCondition(
                kind="predicate_self_only",
                target_key=edge_key,
                hop_index=hop_index,
                detail={"predicate": biolink_predicate(hop.predicate)},
            )
        )

    return out


def _post_conditions_for_entity(entity, entity_ref: str, node_key: str) -> List[PostCondition]:
    constraints = _constraints_to_dicts(getattr(entity, "constraints", None))
    if not constraints:
        return []
    return [
        PostCondition(
            kind="entity_constraint",
            target_key=node_key,
            entity_ref=entity_ref,
            detail={"constraints": constraints},
        )
    ]


# ---------------------------------------------------------------------------
# Full path query
# ---------------------------------------------------------------------------


def assign_node_keys(path) -> Dict[str, str]:
    """Map entity_refs to node keys by order of first appearance.

    Structural rather than arbitrary, so two paths describing the same shape
    produce the same keys and therefore the same cache key.
    """
    keys: Dict[str, str] = {}
    for hop in path.hops:
        for ref in (hop.subject_ref, hop.object_ref):
            if ref not in keys:
                keys[ref] = f"n{len(keys):02d}"
    return keys


def build_path_query(
    path,
    entities: Dict[str, Any],
    resolutions: Optional[Dict[str, Sequence[str]]] = None,
    pin_variables: Optional[Dict[str, Sequence[str]]] = None,
) -> BuiltQuery:
    """Build the full N-hop query graph for a path.

    Args:
        path: a plan Path (needs .hops, .return_entity_ref, .path_id).
        entities: entity_ref -> Entity, from the plan.
        resolutions: entity_ref -> resolved CURIEs for non-variable entities.
            Produced by resolver.py.
        pin_variables: entity_ref -> CURIEs to pin onto an entity the plan
            marked variable. Used during decomposition to feed one hop's
            output into the next.

    Returns:
        BuiltQuery. Unresolved anchors are reported in `.warnings` rather than
        raised, so the caller can decide whether a partially pinned query is
        still worth running.
    """
    resolutions = resolutions or {}
    pin_variables = pin_variables or {}

    node_keys = assign_node_keys(path)
    nodes: Dict[str, Any] = {}
    edges: Dict[str, Any] = {}
    post_conditions: List[PostCondition] = []
    warnings: List[str] = []

    for ref, nkey in node_keys.items():
        entity = entities.get(ref)
        if entity is None:
            warnings.append(f"entity_ref '{ref}' not found in plan entities")
            continue

        ids = list(pin_variables.get(ref) or resolutions.get(ref) or [])
        is_variable = getattr(entity, "is_variable", False)

        if not is_variable and not ids:
            warnings.append(
                f"anchor entity '{ref}' has no resolved CURIE; node {nkey} "
                f"will be left open by category"
            )

        # A pinned node holding many CURIEs is a set: the question is "any of
        # these", not one result per member.
        nodes[nkey] = _build_node(entity, ids=ids, is_set=len(ids) > 1)
        post_conditions.extend(_post_conditions_for_entity(entity, ref, nkey))

    edge_map: Dict[int, str] = {}
    for i, hop in enumerate(path.hops):
        ekey = f"e{i:02d}"
        edge_map[i] = ekey
        edges[ekey] = _build_edge(hop, node_keys[hop.subject_ref], node_keys[hop.object_ref])
        post_conditions.extend(_post_conditions_for_hop(hop, i, ekey))

    return BuiltQuery(
        query_graph={"nodes": nodes, "edges": edges},
        node_map=node_keys,
        edge_map=edge_map,
        return_node_key=node_keys.get(getattr(path, "return_entity_ref", None)),
        post_conditions=post_conditions,
        warnings=warnings,
        path_id=getattr(path, "path_id", None),
    )


# ---------------------------------------------------------------------------
# Single-hop query, for decomposition
# ---------------------------------------------------------------------------


def build_hop_query(
    hop,
    subject_entity,
    object_entity,
    subject_ids: Optional[Sequence[str]] = None,
    object_ids: Optional[Sequence[str]] = None,
    hop_index: int = 0,
) -> BuiltQuery:
    """Build a standalone one-hop query graph.

    Always emits `n00` (subject), `n01` (object), `e00`, regardless of where
    the hop sat in its path. That fixed naming is what lets the same hop
    reached from different paths — or from a re-run of a revised plan — hit
    the same cache entry.
    """
    nodes = {
        "n00": _build_node(subject_entity, ids=subject_ids,
                           is_set=bool(subject_ids) and len(subject_ids) > 1),
        "n01": _build_node(object_entity, ids=object_ids,
                           is_set=bool(object_ids) and len(object_ids) > 1),
    }
    edges = {"e00": _build_edge(hop, "n00", "n01")}

    post_conditions = _post_conditions_for_hop(hop, hop_index, "e00")
    post_conditions.extend(
        _post_conditions_for_entity(subject_entity, hop.subject_ref, "n00")
    )
    post_conditions.extend(
        _post_conditions_for_entity(object_entity, hop.object_ref, "n01")
    )

    return BuiltQuery(
        query_graph={"nodes": nodes, "edges": edges},
        node_map={hop.subject_ref: "n00", hop.object_ref: "n01"},
        edge_map={hop_index: "e00"},
        post_conditions=post_conditions,
    )


# ---------------------------------------------------------------------------
# Decomposition planning
# ---------------------------------------------------------------------------


class DecompositionError(Exception):
    """Raised when a path cannot be decomposed into anchored single hops."""


def plan_decomposition(
    path,
    entities: Dict[str, Any],
    resolutions: Optional[Dict[str, Sequence[str]]] = None,
) -> List[HopStep]:
    """Order a path's hops for pivot-first execution.

    Starts from entities that resolve to concrete CURIEs and walks outward:
    each step must have one endpoint already known, so every query carries a
    pinned id list and stays bounded. A hop whose endpoints are both already
    known is emitted last — it constrains an existing candidate set rather
    than expanding it.

    Raises:
        DecompositionError: if no hop touches a resolvable entity. Such a path
            has no bounded starting point, so decomposition cannot help and
            the executor should report the plan as unexecutable rather than
            issue an unbounded query.
    """
    resolutions = resolutions or {}

    known = {
        ref
        for ref in {r for h in path.hops for r in (h.subject_ref, h.object_ref)}
        if resolutions.get(ref)
        or not getattr(entities.get(ref), "is_variable", True)
    }

    if not known:
        raise DecompositionError(
            f"path '{getattr(path, 'path_id', '?')}' has no anchored entity; "
            f"every endpoint is variable or unresolved"
        )

    remaining = list(enumerate(path.hops))
    steps: List[HopStep] = []
    deferred: List[tuple] = []
    order = 0

    while remaining:
        progressed = False
        for pos, (hop_index, hop) in enumerate(remaining):
            s_known = hop.subject_ref in known
            o_known = hop.object_ref in known

            if s_known and o_known:
                continue  # closing hop, handled after expansion finishes
            if not (s_known or o_known):
                continue  # not reachable yet

            pinned = hop.subject_ref if s_known else hop.object_ref
            solve = hop.object_ref if s_known else hop.subject_ref
            source = "anchor" if resolutions.get(pinned) else "previous_step"

            steps.append(
                HopStep(
                    order=order,
                    hop_index=hop_index,
                    hop=hop,
                    pinned_ref=pinned,
                    solve_ref=solve,
                    source=source,
                    pinned_is_subject=s_known,
                )
            )
            known.add(solve)
            remaining.pop(pos)
            order += 1
            progressed = True
            break

        if not progressed:
            # What is left either closes a cycle (both ends known) or is
            # unreachable. Closing hops are legitimate and run last.
            for hop_index, hop in remaining:
                if hop.subject_ref in known and hop.object_ref in known:
                    deferred.append((hop_index, hop))
                else:
                    raise DecompositionError(
                        f"path '{getattr(path, 'path_id', '?')}' hop {hop_index} "
                        f"({hop.subject_ref} -> {hop.object_ref}) is not reachable "
                        f"from any anchored entity"
                    )
            break

    for hop_index, hop in deferred:
        steps.append(
            HopStep(
                order=order,
                hop_index=hop_index,
                hop=hop,
                pinned_ref=hop.subject_ref,
                solve_ref=hop.object_ref,
                source="previous_step",
                pinned_is_subject=True,
            )
        )
        order += 1

    return steps


def build_step_query(
    step: HopStep,
    entities: Dict[str, Any],
    pinned_ids: Sequence[str],
    solve_ids: Optional[Sequence[str]] = None,
) -> BuiltQuery:
    """Build the query for one decomposition step.

    Args:
        step: from `plan_decomposition`.
        entities: entity_ref -> Entity.
        pinned_ids: CURIEs for the known endpoint — a plan anchor's resolution,
            or one batch of the previous step's output.
        solve_ids: optionally pin the far end too, which turns the step into a
            verification of specific pairs rather than an open search.
    """
    hop = step.hop
    subject_entity = entities.get(hop.subject_ref)
    object_entity = entities.get(hop.object_ref)

    if step.pinned_is_subject:
        subject_ids, object_ids = list(pinned_ids), list(solve_ids or [])
    else:
        subject_ids, object_ids = list(solve_ids or []), list(pinned_ids)

    built = build_hop_query(
        hop,
        subject_entity=subject_entity,
        object_entity=object_entity,
        subject_ids=subject_ids,
        object_ids=object_ids,
        hop_index=step.hop_index,
    )
    built.path_id = f"step{step.order}_hop{step.hop_index}"
    return built


def batch_ids(ids: Sequence[str], batch_size: int) -> List[List[str]]:
    """Split pinned ids into groups.

    Every group is executed — the executor does not stop at the first group
    that returns hits. Partial coverage would silently undercount the
    supporting-path totals that ranking depends on, and would make an empty
    result impossible to distinguish from an untested one.
    """
    ids = sorted(set(ids))
    return [list(ids[i:i + batch_size]) for i in range(0, len(ids), batch_size)]


# ---------------------------------------------------------------------------
# TRAPI message assembly
# ---------------------------------------------------------------------------


def default_operations(
    max_results: int = DEFAULT_MAX_RESULTS,
    compute_ngd: bool = False,
) -> List[str]:
    """ARAXi actions mirroring ARAX's auto-generated plan, with a raised cap.

    Left to itself ARAX appends `max_results=500`. A hop pinned with hundreds
    of intermediates produces one result per pair and is silently truncated at
    that ceiling. Sending explicit operations makes the limit visible and
    adjustable.

    `overlay(compute_ngd)` is off by default. It adds one virtual edge per
    result, sourced to `infores:arax` and typed `statistical_association` — a
    computed co-occurrence measure rather than retrieved knowledge. Those edges
    enter the knowledge graph alongside real ones, where they are scored as
    evidence and distort both the provenance distribution and the EPC ranking.
    Turn it on only when the NGD score is wanted for its own sake.
    """
    actions = ["expand()"]
    if compute_ngd:
        actions.append(
            "overlay(action=compute_ngd, virtual_relation_label=N1, "
            "subject_qnode_key=n00, object_qnode_key=n01)"
        )
    # Prunes hub nodes too generic to carry meaning for a specific question.
    # Its effect is uneven across query shapes: on a multi-hop query it removes
    # intermediates, and every path through a removed node dies with it,
    # whereas the same node arrives pinned by CURIE in a decomposed hop and
    # survives. Measured on a 2-hop NSCLC plan, the direct query returned 34
    # candidates and the decomposed form 57, the 34 a strict subset. The extra
    # 23 routed mostly through interleukin-10 — a cytokine touching nearly
    # every inflammatory process, and so connected to thousands of unrelated
    # drugs. The filter was right, and its absence is what makes a decomposed
    # result noisier than the direct one it replaces.
    actions += [
        "filter_kg(action=remove_general_concept_nodes,perform_action=True)",
        "resultify()",
        f"filter_results(action=limit_number_of_results, max_results={max_results})",
    ]
    return actions


def to_trapi_message(
    built: BuiltQuery,
    max_results: Optional[int] = DEFAULT_MAX_RESULTS,
    include_operations: bool = True,
    compute_ngd: bool = False,
) -> Dict[str, Any]:
    """Wrap a BuiltQuery into a submittable TRAPI request body."""
    message: Dict[str, Any] = {"message": {"query_graph": built.query_graph}}
    if include_operations and max_results:
        message["operations"] = {
            "actions": default_operations(max_results, compute_ngd=compute_ngd)
        }
    return message


def describe(built: BuiltQuery) -> str:
    """One-line human summary, for logs and the ledger."""
    qg = built.query_graph
    parts = []
    for ekey in sorted(qg.get("edges", {})):
        e = qg["edges"][ekey]
        s, o = qg["nodes"].get(e["subject"], {}), qg["nodes"].get(e["object"], {})

        def side(n, key):
            if n.get("ids"):
                ids = n["ids"]
                return ids[0] if len(ids) == 1 else f"{len(ids)} ids"
            return ",".join(c.replace("biolink:", "") for c in n.get("categories", [])) or key

        pred = ",".join(p.replace("biolink:", "") for p in e.get("predicates", []))
        q = "+q" if e.get("qualifier_constraints") else ""
        parts.append(f"{side(s, e['subject'])} --[{pred}{q}]--> {side(o, e['object'])}")
    return " ; ".join(parts)
