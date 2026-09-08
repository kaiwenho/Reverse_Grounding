"""
explain.py — Find mechanistic paths between two known endpoints.

Discovery paths answer "what candidates exist"; an explanation query answers
"by what route". Both endpoints are already known, so nothing is being
searched for — what is wanted is the chain connecting them, which is exactly
what ARAX's `connect(action=connect_nodes)` produces.

This is also the right tool when a question names two entities without stating
how they relate. Asserting a predicate between two pinned nodes is a guess,
and a guess that returns nothing is indistinguishable from the pair being
genuinely unconnected. Asking what paths exist has no such ambiguity.

Filtering happens here, not at ARAX
-----------------------------------
`connect_nodes` accepts only `max_path_length` (1..5) and
`max_pathfinder_paths`. Every constraint a plan expresses —
`middle_category_whitelist`, `predicate_whitelist`,
`min_evidence_strength_per_edge`, `min_publications_per_edge`,
`drop_paths_with_negated_edges`, `unique_intermediate_nodes` — has no
counterpart in the DSL and is applied to the returned paths. The plan's
`max_hops` is clamped to 5 with a warning rather than silently truncated,
since a plan asking for 6 would otherwise be rejected by ARAX at execution.

Fan-out and cost
----------------
A `from_discovery` endpoint means one connect() call per candidate, each a
pathfinding search. `top_k` on the binding bounds it, and results are cached
per endpoint pair, so a re-run or an overlapping plan costs nothing. Without
that bound this is the most expensive operation in the executor.

Usage
-----
    explainer = Explainer(client=arax_client)
    explanations = explainer.run(eq, plan_input, resolutions, executions)
    # -> {candidate_curie: [ExplanationPath, ...]}
"""

from __future__ import annotations

import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .arax_client import AraxClient, OUTCOME_SUCCESS
from .postfilter import (
    edge_knowledge_level, edge_negated, edge_publications, strength_at_least,
    summarize_edge,
)


#: ARAX's connect() ceiling (ARAX_connect.py, max_path_length_info).
MAX_CONNECT_HOPS = 5
DEFAULT_CONNECT_HOPS = 3

#: Paths returned per connect() call before local filtering. Generous, because
#: filtering happens afterwards and an over-tight cap would discard paths the
#: plan's own criteria would have kept.
DEFAULT_PATHFINDER_PATHS = 200

#: Candidates explained when a from_discovery binding names no top_k.
DEFAULT_TOP_K_CANDIDATES = 20


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class ExplanationPath:
    """One route between two endpoints."""

    node_curies: List[str] = field(default_factory=list)
    node_labels: List[str] = field(default_factory=list)
    node_categories: List[str] = field(default_factory=list)
    edge_ids: List[str] = field(default_factory=list)
    predicates: List[str] = field(default_factory=list)
    #: How many distinct edge instances attest this same route. Several
    #: sources asserting the same relationship is corroboration, not several
    #: different explanations.
    variant_count: int = 1

    @property
    def length(self) -> int:
        return len(self.edge_ids)

    @property
    def intermediates(self) -> List[str]:
        return self.node_curies[1:-1] if len(self.node_curies) > 2 else []

    @property
    def intermediate_categories(self) -> List[str]:
        return self.node_categories[1:-1] if len(self.node_categories) > 2 else []

    def describe(self) -> str:
        parts = []
        for i, label in enumerate(self.node_labels):
            parts.append(label)
            if i < len(self.predicates):
                parts.append(f"--[{self.predicates[i].replace('biolink:', '')}]-->")
        return " ".join(parts)

    def to_dict(self, edges: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        out = {
            "description": self.describe(),
            "length": self.length,
            "nodes": self.node_curies,
            "labels": self.node_labels,
            "intermediate_categories": self.intermediate_categories,
            "predicates": self.predicates,
        }
        if edges:
            out["edges"] = [
                dict(summarize_edge(edges[e]), edge_id=e)
                for e in self.edge_ids if e in edges
            ]
        if self.variant_count > 1:
            out["attesting_edge_sets"] = self.variant_count
        return out


@dataclass
class ExplanationResult:
    """Everything one explanation query produced."""

    query_id: str
    endpoint_b: Optional[str] = None
    by_candidate: Dict[str, List[ExplanationPath]] = field(default_factory=dict)
    kg_nodes: Dict[str, Any] = field(default_factory=dict)
    kg_edges: Dict[str, Any] = field(default_factory=dict)
    candidates_attempted: int = 0
    candidates_with_paths: int = 0
    paths_found: int = 0
    paths_kept: int = 0
    drop_reasons: Dict[str, int] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "query_id": self.query_id,
            "endpoint_b": self.endpoint_b,
            "candidates_attempted": self.candidates_attempted,
            "candidates_with_paths": self.candidates_with_paths,
            "paths_found": self.paths_found,
            "paths_kept": self.paths_kept,
            "drop_reasons": self.drop_reasons,
            "warnings": self.warnings,
            "elapsed_s": round(self.elapsed_s, 2),
        }

    def as_attachments(self) -> Dict[str, List[Dict[str, Any]]]:
        """Per-candidate paths, shaped for attaching to ranked results."""
        return {
            curie: [p.to_dict(self.kg_edges) for p in paths]
            for curie, paths in self.by_candidate.items()
        }


# ---------------------------------------------------------------------------
# Path extraction
# ---------------------------------------------------------------------------


def extract_paths(
    response: Dict[str, Any],
    start: str,
    end: str,
    max_paths: int = 500,
) -> Tuple[List[ExplanationPath], Dict[str, Any], Dict[str, Any]]:
    """Reconstruct ordered node-and-edge chains from a connect() response.

    A TRAPI result binds nodes and edges but does not order them, so the chain
    is recovered by walking the bound edges from one endpoint to the other.
    Each result is walked separately: edges from different results describe
    different routes, and merging them first would invent paths that were
    never returned.
    """
    message = (response or {}).get("message") or {}
    kg = message.get("knowledge_graph") or {}
    nodes = kg.get("nodes") or {}
    edges = kg.get("edges") or {}
    paths: List[ExplanationPath] = []
    seen: Dict[Tuple, ExplanationPath] = {}

    aux_graphs = message.get("auxiliary_graphs") or {}

    # A pathfinder result carries each route as an auxiliary graph referenced
    # by path_bindings, so the routes are already separated and must be read
    # one at a time. Pooling their edges first would let a walk cross from one
    # route into another and invent a path ARAX never returned.
    edge_groups: List[List[str]] = []
    for result in message.get("results") or []:
        for analysis in result.get("analyses") or []:
            for _, bindings in (analysis.get("path_bindings") or {}).items():
                for b in bindings or []:
                    aux_id = b.get("id") if isinstance(b, dict) else b
                    aux = aux_graphs.get(aux_id) or {}
                    group = [e for e in (aux.get("edges") or []) if e in edges]
                    if group:
                        edge_groups.append(group)

        # Older responses, and ordinary query results, bind edges directly.
        flat: List[str] = []
        for analysis in result.get("analyses") or []:
            for _, bindings in (analysis.get("edge_bindings") or {}).items():
                for b in bindings or []:
                    eid = b.get("id") if isinstance(b, dict) else b
                    if eid and eid in edges:
                        flat.append(eid)
        if flat:
            edge_groups.append(flat)

    for bound_edges in edge_groups:
        if not bound_edges:
            continue

        adjacency: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        for eid in bound_edges:
            edge = edges[eid]
            s, o = edge.get("subject"), edge.get("object")
            if not s or not o:
                continue
            # Traversed in both directions: a mechanistic chain reads through
            # an edge regardless of which way the assertion happens to point.
            adjacency[s].append((o, eid))
            adjacency[o].append((s, eid))

        for chain in _walk(adjacency, start, end, len(bound_edges)):
            node_curies, edge_ids = chain
            predicates = [edges[eid].get("predicate", "") for eid in edge_ids]

            # Keyed on the route itself — its nodes and relationships — rather
            # than on which edge records carry it. Two sources asserting the
            # same relationship between the same nodes are one explanation
            # attested twice, and listing them separately would fill a
            # top-5 with a single mechanism repeated.
            key = (tuple(node_curies), tuple(predicates))
            if key in seen:
                seen[key].variant_count += 1
                continue

            path = ExplanationPath(
                node_curies=node_curies,
                node_labels=[
                    (nodes.get(c) or {}).get("name") or c for c in node_curies
                ],
                node_categories=[
                    ((nodes.get(c) or {}).get("categories") or [""])[0].replace(
                        "biolink:", "")
                    for c in node_curies
                ],
                edge_ids=edge_ids,
                predicates=predicates,
            )
            seen[key] = path
            paths.append(path)
            if len(paths) >= max_paths:
                return paths, nodes, edges

    return paths, nodes, edges


def _walk(
    adjacency: Dict[str, List[Tuple[str, str]]],
    start: str,
    end: str,
    max_depth: int,
) -> Iterable[Tuple[List[str], List[str]]]:
    """Yield simple paths from start to end.

    Nodes are not revisited: a chain that loops back on itself is a traversal
    artefact rather than a mechanism.
    """
    if start not in adjacency:
        return

    stack: List[Tuple[str, List[str], List[str]]] = [(start, [start], [])]
    while stack:
        node, path_nodes, path_edges = stack.pop()
        if node == end and len(path_nodes) > 1:
            yield path_nodes, path_edges
            continue
        if len(path_edges) >= max_depth:
            continue
        for neighbour, eid in adjacency.get(node, []):
            if neighbour in path_nodes or eid in path_edges:
                continue
            stack.append((neighbour, path_nodes + [neighbour], path_edges + [eid]))


# ---------------------------------------------------------------------------
# Filtering
# ---------------------------------------------------------------------------


def filter_paths(
    paths: Sequence[ExplanationPath],
    edges: Dict[str, Any],
    middle_whitelist: Sequence[str] = (),
    middle_blacklist: Sequence[str] = (),
    post_filter: Optional[Dict[str, Any]] = None,
) -> Tuple[List[ExplanationPath], Dict[str, int]]:
    """Apply a plan's explanation constraints to returned paths.

    None of these can be pushed to ARAX — `connect_nodes` takes no whitelist —
    so they run here, with per-rule drop counts so an over-tight constraint is
    visible rather than showing up only as an empty result.
    """
    post_filter = post_filter or {}
    reasons: Dict[str, int] = defaultdict(int)

    allowed_middles = {c.replace("biolink:", "") for c in middle_whitelist or ()}
    banned_middles = {c.replace("biolink:", "") for c in middle_blacklist or ()}
    allowed_preds = set(post_filter.get("predicate_whitelist") or ())
    banned_preds = set(post_filter.get("predicate_blacklist") or ())
    required_preds = set(post_filter.get("required_predicates_anywhere") or ())
    min_strength = post_filter.get("min_evidence_strength_per_edge")
    min_pubs = int(post_filter.get("min_publications_per_edge") or 0)
    drop_negated = post_filter.get("drop_paths_with_negated_edges", True)
    unique_nodes = post_filter.get("unique_intermediate_nodes", True)

    kept: List[ExplanationPath] = []
    for path in paths:
        if unique_nodes and len(set(path.node_curies)) != len(path.node_curies):
            reasons["repeated_node"] += 1
            continue

        middles = {c for c in path.intermediate_categories if c}
        # Every intermediate must be permitted, not merely one of them. A
        # whitelist naming Gene and Protein is a statement about what may
        # appear in the chain, so a route passing through a Pathway breaches
        # it even though it also passes through a Gene.
        if allowed_middles and not middles.issubset(allowed_middles):
            reasons["middle_category_not_allowed"] += 1
            continue
        if banned_middles and (middles & banned_middles):
            reasons["middle_category_excluded"] += 1
            continue

        path_edges = [edges[e] for e in path.edge_ids if e in edges]

        if allowed_preds and not all(
            e.get("predicate") in allowed_preds for e in path_edges
        ):
            reasons["predicate_not_whitelisted"] += 1
            continue
        if banned_preds and any(
            e.get("predicate") in banned_preds for e in path_edges
        ):
            reasons["predicate_blacklisted"] += 1
            continue
        if required_preds and not any(
            e.get("predicate") in required_preds for e in path_edges
        ):
            reasons["required_predicate_absent"] += 1
            continue

        if drop_negated and any(edge_negated(e) for e in path_edges):
            reasons["negated_edge"] += 1
            continue
        if min_pubs and not all(
            len(edge_publications(e)) >= min_pubs for e in path_edges
        ):
            reasons["insufficient_publications"] += 1
            continue
        if min_strength and not all(
            strength_at_least(e, min_strength) for e in path_edges
        ):
            reasons["weak_edge"] += 1
            continue

        kept.append(path)

    return kept, dict(reasons)


def rank_paths(
    paths: Sequence[ExplanationPath],
    edges: Dict[str, Any],
    rank_by: str = "composite",
    top_k: int = 5,
    group_by_category: bool = False,
) -> List[ExplanationPath]:
    """Order paths and take the best.

    `composite` prefers short, well-evidenced chains: a three-hop route through
    a well-supported gene explains more than a five-hop route assembled from
    weak links.

    With `group_by_intermediate_category`, the selection takes the best path
    from each distinct category before taking a second from any — several
    mechanisms of different kinds say more than several variations on one.
    """
    from .postfilter import score_epc

    def edge_scores(path: ExplanationPath) -> List[float]:
        return [
            score_epc(edges[e])["epc_score"] for e in path.edge_ids if e in edges
        ]

    def key(path: ExplanationPath):
        scores = edge_scores(path)
        weakest = min(scores) if scores else 0.0
        mean = sum(scores) / len(scores) if scores else 0.0
        if rank_by == "shortest":
            return (path.length, -weakest)
        if rank_by == "evidence":
            return (-weakest, path.length)
        return (-round(mean - 0.05 * path.length, 4), path.length)

    ordered = sorted(paths, key=key)
    if not group_by_category:
        return ordered[:top_k]

    # Grouped by the intermediates themselves, not only their categories. A
    # category signature of "Gene" is shared by every route through any gene,
    # so grouping on it alone returns five variations on one mechanism —
    # which is the opposite of what asking for diversity means.
    by_category: Dict[Tuple, List[ExplanationPath]] = defaultdict(list)
    for path in ordered:
        signature = (
            tuple(sorted(set(path.intermediate_categories))),
            tuple(path.intermediates),
        )
        by_category[signature].append(path)

    out: List[ExplanationPath] = []
    round_index = 0
    while len(out) < top_k:
        added = False
        for paths_in_group in by_category.values():
            if round_index < len(paths_in_group):
                out.append(paths_in_group[round_index])
                added = True
                if len(out) >= top_k:
                    break
        if not added:
            break
        round_index += 1
    return out


# ---------------------------------------------------------------------------
# Explainer
# ---------------------------------------------------------------------------


class Explainer:
    """Runs a plan's explanation queries against ARAX."""

    def __init__(
        self,
        client: AraxClient,
        max_pathfinder_paths: int = DEFAULT_PATHFINDER_PATHS,
        default_top_k: int = DEFAULT_TOP_K_CANDIDATES,
        verbose: bool = True,
    ):
        self.client = client
        self.max_pathfinder_paths = max_pathfinder_paths
        self.default_top_k = default_top_k
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [explain] {msg}")

    # -- endpoint resolution ----------------------------------------------

    def resolve_endpoint(
        self,
        binding: Any,
        resolutions: Dict[str, Any],
        executions: Dict[str, Any],
        ranked: Optional[Sequence[Any]] = None,
    ) -> Tuple[List[str], List[str]]:
        """Turn an endpoint binding into CURIEs. Returns (curies, warnings)."""
        warnings: List[str] = []
        get = (lambda k: binding.get(k)) if hasattr(binding, "get") else (
            lambda k: getattr(binding, k, None)
        )
        kind = get("binding_type")

        if kind == "entity":
            ref = get("entity_ref")
            resolution = resolutions.get(ref)
            curies = list(getattr(resolution, "curies", None) or [])
            if not curies:
                warnings.append(f"entity '{ref}' did not resolve; nothing to explain")
            return curies, warnings

        if kind == "from_discovery":
            # A binding names several paths, each contributing candidates
            # through its own return_entity_ref. Two paths may use different
            # entity references while producing compatible categories, so the
            # candidates are pooled rather than taken from one path.
            path_ids = list(get("from_path_ids") or [])
            top_k = get("fanout_top_k") or self.default_top_k

            if not path_ids:
                warnings.append(
                    "from_discovery binding names no from_path_ids"
                )
                return [], warnings

            # Ranked order first: fanout_top_k means the best candidates, and
            # only ranking knows which those are. Pooling raw execution output
            # would take whichever happened to be found first.
            if ranked:
                curies = [
                    c.curie for c in ranked
                    if set(path_ids) & set(getattr(c, "path_ids", set()))
                ][:top_k]
                if curies:
                    return curies, warnings

            pooled: List[str] = []
            for path_id in path_ids:
                execution = executions.get(path_id)
                if execution is None:
                    warnings.append(
                        f"path '{path_id}' produced no execution to draw from"
                    )
                    continue
                for curie in getattr(execution, "candidates", []) or []:
                    if curie not in pooled:
                        pooled.append(curie)

            if not pooled:
                warnings.append(
                    f"paths {path_ids} produced no candidates to explain"
                )
            return pooled[:top_k], warnings

        warnings.append(f"unsupported binding_type '{kind}'")
        return [], warnings

    # -- connect -----------------------------------------------------------

    def connect(self, curie_a: str, curie_b: str, max_hops: int) -> Any:
        """Find routes between two pinned nodes."""
        return self.client.pathfinder(
            curie_a, curie_b, max_hops=max_hops,
            max_paths=self.max_pathfinder_paths,
        )

    # -- driver ------------------------------------------------------------

    def run(
        self,
        eq: Any,
        resolutions: Dict[str, Any],
        executions: Dict[str, Any],
        ranked: Optional[Sequence[Any]] = None,
    ) -> ExplanationResult:
        """Execute one explanation query across its endpoint pairs."""
        start = time.time()
        get = (lambda k: eq.get(k)) if hasattr(eq, "get") else (
            lambda k: getattr(eq, k, None)
        )
        query_id = get("query_id") or "?"
        result = ExplanationResult(query_id=query_id)

        max_hops = int(get("max_hops") or DEFAULT_CONNECT_HOPS)
        if max_hops > MAX_CONNECT_HOPS:
            result.warnings.append(
                f"max_hops={max_hops} exceeds what ARAX connect() accepts; "
                f"clamped to {MAX_CONNECT_HOPS}"
            )
            max_hops = MAX_CONNECT_HOPS

        a_curies, w1 = self.resolve_endpoint(
            get("endpoint_a"), resolutions, executions, ranked)
        b_curies, w2 = self.resolve_endpoint(
            get("endpoint_b"), resolutions, executions, ranked)
        result.warnings.extend(w1 + w2)

        if not a_curies or not b_curies:
            result.elapsed_s = time.time() - start
            return result

        # One endpoint is normally a single anchor and the other a candidate
        # list; whichever is singular becomes the fixed end.
        if len(b_curies) == 1:
            fixed, fanned = b_curies[0], a_curies
        elif len(a_curies) == 1:
            fixed, fanned = a_curies[0], b_curies
        else:
            fixed, fanned = b_curies[0], a_curies
            result.warnings.append(
                f"both endpoints resolved to multiple CURIEs; explaining "
                f"against {fixed} only"
            )

        result.endpoint_b = fixed
        result.candidates_attempted = len(fanned)
        self.log(f"{query_id}: connecting {len(fanned)} candidate(s) to {fixed} "
                 f"within {max_hops} hop(s)")

        return_spec = get("return") or get("return_spec") or {}
        if hasattr(return_spec, "get"):
            top_k_paths = int(return_spec.get("top_k_paths") or 5)
            rank_by = return_spec.get("rank_by") or "composite"
            group = bool(return_spec.get("group_by_intermediate_category", False))
        else:
            top_k_paths, rank_by, group = 5, "composite", False

        post_filter = get("post_filter") or {}
        if not hasattr(post_filter, "get"):
            post_filter = {}
        totals: Dict[str, int] = defaultdict(int)

        for curie in fanned:
            response = self.connect(curie, fixed, max_hops)
            if response.outcome != OUTCOME_SUCCESS or not response.response:
                if response.outcome != "empty":
                    result.warnings.append(
                        f"connect({curie}) {response.outcome}: "
                        f"{response.error_message}"
                    )
                continue

            paths, nodes, edges = extract_paths(response.response, curie, fixed)
            result.kg_nodes.update(nodes)
            result.kg_edges.update(edges)
            result.paths_found += len(paths)

            kept, reasons = filter_paths(
                paths, edges,
                middle_whitelist=get("middle_category_whitelist") or (),
                middle_blacklist=get("middle_category_blacklist") or (),
                post_filter=post_filter,
            )
            for reason, count in reasons.items():
                totals[reason] += count

            if not kept:
                continue

            best = rank_paths(kept, edges, rank_by=rank_by,
                              top_k=top_k_paths, group_by_category=group)
            result.by_candidate[curie] = best
            result.paths_kept += len(best)
            result.candidates_with_paths += 1

        result.drop_reasons = dict(totals)
        result.elapsed_s = time.time() - start

        # An explanation query that finds routes for nothing is worth calling
        # out: the candidates came from the same graph, so the absence usually
        # means the constraints are tighter than the mechanisms available.
        if result.candidates_attempted and not result.candidates_with_paths:
            biggest = max(totals.items(), key=lambda kv: kv[1], default=None)
            detail = f"; most paths dropped by {biggest[0]}" if biggest else ""
            result.warnings.append(
                f"no candidate could be connected to {fixed} within "
                f"{max_hops} hop(s) under this query's constraints{detail}"
            )

        self.log(f"{query_id}: {result.candidates_with_paths}/"
                 f"{result.candidates_attempted} candidate(s) explained, "
                 f"{result.paths_kept} path(s) kept of {result.paths_found}")
        return result

    def run_all(
        self,
        plan_input: Any,
        resolutions: Dict[str, Any],
        executions: Dict[str, Any],
        ranked: Optional[Sequence[Any]] = None,
    ) -> Tuple[Dict[str, List[Dict[str, Any]]], List[Dict[str, Any]]]:
        """Run every runnable explanation query.

        Returns (attachments keyed by candidate, per-query summaries). Paths
        from several queries accumulate on the same candidate rather than
        overwriting, since each query explains a different aspect.
        """
        attachments: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        summaries: List[Dict[str, Any]] = []

        for eq in getattr(plan_input, "runnable_explanation_queries", []) or []:
            result = self.run(eq, resolutions, executions, ranked)
            for curie, paths in result.as_attachments().items():
                attachments[curie].extend(paths)
            summaries.append(result.to_dict())
            for w in result.warnings:
                self.log(f"    warning: {w}")

        return dict(attachments), summaries
