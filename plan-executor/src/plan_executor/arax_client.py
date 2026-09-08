"""
arax_client.py — HTTP layer for ARAX, with outcome classification.

The executor's whole fallback strategy hinges on one question: did this query
fail in a way that decomposition can fix? Getting that classification right
matters more than the transport, so it lives here rather than being inferred
from exception types at the call site.

Outcomes
--------
    success   results returned
    empty     valid response, zero results — real evidence of "no answer"
    partial   results returned, but ARAX logged a degradation (a KP timed out,
              an edge was skipped). The data is incomplete.
    timeout   exceeded the wall clock, at HTTP or ARAX level
    error     everything else: 4xx, malformed response, transport failure

The empty/timeout split is the load-bearing one. A plan is declared
unanswerable only when a hop genuinely returns nothing; if the hop returned
nothing *because a knowledge provider timed out*, that is not evidence and
must not be reported as `no_answer`. This module therefore never classifies a
zero-result response as `empty` when a degradation was logged — it returns
`timeout` instead, so the executor retries or decomposes.

Timeout detection is deliberately broad, because ARAX signals slowness in at
least four different ways:

  1. a transport-level read timeout
  2. HTTP 502/503/504 from the gateway in front of ARAX
  3. a non-Success TRAPI `status` with timeout wording in `description`
  4. HTTP 200 with Success status, but ERROR/WARNING logs reporting that an
     expand step or a KP call timed out

Usage
-----
    client = AraxClient(cache=cache)
    resp = client.query(built, max_results=5000)

    if resp.outcome == "timeout":
        ...   # decompose
    elif resp.outcome == "empty":
        ...   # genuine no-answer, report it
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

from .cache import (
    QueryCache, STATUS_SUCCESS, STATUS_EMPTY, STATUS_TIMEOUT, STATUS_ERROR,
    KIND_ARAX_QUERY, KIND_ARAX_DSL,
)
from .trapi_builder import BuiltQuery, to_trapi_message, DEFAULT_MAX_RESULTS


# ---------------------------------------------------------------------------
# Endpoints and limits
# ---------------------------------------------------------------------------

ARAX_BASE = "https://arax.ncats.io/api/arax/v1.4"
ARAX_QUERY_URL = f"{ARAX_BASE}/query"
ARAX_META_KG_URL = f"{ARAX_BASE}/meta_knowledge_graph"

#: Wall clock for a single synchronous query. Past this the executor stops
#: waiting and decomposes, which is usually faster than a longer timeout.
REQUEST_TIMEOUT = 120

#: DSL workflows (including connect()) legitimately run longer.
DSL_TIMEOUT = 300

OUTCOME_SUCCESS = "success"
OUTCOME_EMPTY = "empty"
OUTCOME_PARTIAL = "partial"
OUTCOME_TIMEOUT = "timeout"
OUTCOME_ERROR = "error"

#: HTTP codes meaning "the server gave up", not "the request was wrong".
#: 598 is a non-standard read-timeout code some proxies emit.
_TIMEOUT_HTTP_CODES = {502, 503, 504, 522, 524, 598}

#: Matched against ARAX log messages and status descriptions. Broad on
#: purpose: a missed timeout gets misreported as `empty`, which would let the
#: executor claim "no answer" on a query that was never actually completed.
_TIMEOUT_PATTERN = re.compile(
    r"tim(?:e|ed)\s*[-_ ]?out|timeout|took too long|exceeded .{0,20}(?:time|limit)"
    r"|deadline|slow.{0,15}response|did not (?:respond|finish)",
    re.IGNORECASE,
)

#: TRAPI status values that mean the query completed normally.
_OK_STATUSES = {"Success", "OK", "QueryGraphZeroResults"}


# ---------------------------------------------------------------------------
# Response type
# ---------------------------------------------------------------------------


@dataclass
class AraxResponse:
    outcome: str
    response: Optional[Dict[str, Any]] = None
    num_results: int = 0
    num_kg_nodes: int = 0
    num_kg_edges: int = 0
    elapsed_s: float = 0.0
    http_status: Optional[int] = None
    trapi_status: Optional[str] = None
    error_message: Optional[str] = None
    arax_version: Optional[str] = None
    biolink_version: Optional[str] = None
    trapi_version: Optional[str] = None
    truncated: bool = False
    degraded_logs: List[str] = field(default_factory=list)
    from_cache: bool = False

    @property
    def is_answer(self) -> bool:
        """True when this response can be read as an answer, including zero results."""
        return self.outcome in (OUTCOME_SUCCESS, OUTCOME_EMPTY, OUTCOME_PARTIAL)

    @property
    def is_trustworthy_empty(self) -> bool:
        """True only when zero results genuinely means "nothing exists".

        The executor must consult this — not `num_results == 0` — before
        reporting `no_answer` back to the planner.
        """
        return self.outcome == OUTCOME_EMPTY

    @property
    def should_decompose(self) -> bool:
        return self.outcome in (OUTCOME_TIMEOUT, OUTCOME_PARTIAL)

    def cache_status(self) -> str:
        return {
            OUTCOME_SUCCESS: STATUS_SUCCESS,
            OUTCOME_EMPTY: STATUS_EMPTY,
            OUTCOME_PARTIAL: STATUS_SUCCESS,
            OUTCOME_TIMEOUT: STATUS_TIMEOUT,
            OUTCOME_ERROR: STATUS_ERROR,
        }[self.outcome]

    def __repr__(self) -> str:
        cached = " (cached)" if self.from_cache else ""
        trunc = " TRUNCATED" if self.truncated else ""
        return (
            f"<AraxResponse {self.outcome} results={self.num_results} "
            f"{self.elapsed_s:.1f}s{cached}{trunc}>"
        )


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def _extract_degraded_logs(response: Dict[str, Any]) -> List[str]:
    """Return ARAX log lines indicating a timeout or skipped work.

    ARAX will happily return HTTP 200 with `status: Success` while an
    individual knowledge provider timed out and contributed nothing. Those
    logs are the only evidence that the result is incomplete.
    """
    out: List[str] = []
    for entry in response.get("logs") or []:
        if not isinstance(entry, dict):
            continue
        if entry.get("level") not in ("ERROR", "WARNING"):
            continue
        msg = entry.get("message") or ""
        if _TIMEOUT_PATTERN.search(msg):
            out.append(f"[{entry.get('level')}] {msg[:300]}")
    return out


def _count(response: Dict[str, Any]) -> tuple:
    msg = response.get("message") or {}
    kg = msg.get("knowledge_graph") or {}
    return (
        len(msg.get("results") or []),
        len((kg.get("nodes") or {})),
        len((kg.get("edges") or {})),
    )


def classify_response(
    response: Dict[str, Any],
    max_results: Optional[int] = None,
) -> AraxResponse:
    """Turn a raw TRAPI response into a classified AraxResponse.

    Separated from transport so it can be unit-tested against recorded
    responses without a network.
    """
    num_results, n_nodes, n_edges = _count(response)
    trapi_status = response.get("status")
    degraded = _extract_degraded_logs(response)

    out = AraxResponse(
        outcome=OUTCOME_SUCCESS,
        response=response,
        num_results=num_results,
        num_kg_nodes=n_nodes,
        num_kg_edges=n_edges,
        trapi_status=trapi_status,
        arax_version=response.get("tool_version"),
        biolink_version=response.get("biolink_version"),
        trapi_version=response.get("schema_version"),
        degraded_logs=degraded,
        truncated=bool(max_results) and num_results >= max_results,
    )

    # A non-OK status whose description mentions time is a timeout, not a
    # generic error: those two route differently.
    if trapi_status and trapi_status not in _OK_STATUSES:
        description = response.get("description") or ""
        if _TIMEOUT_PATTERN.search(description) or degraded:
            out.outcome = OUTCOME_TIMEOUT
            out.error_message = f"{trapi_status}: {description[:300]}"
        else:
            out.outcome = OUTCOME_ERROR
            out.error_message = f"{trapi_status}: {description[:300]}"
        return out

    if degraded:
        # Zero results plus a degradation is not evidence of absence — the
        # query never really ran. Reporting `empty` here would let the
        # executor tell the planner "no answer exists" on the strength of a
        # failed KP call.
        out.outcome = OUTCOME_TIMEOUT if num_results == 0 else OUTCOME_PARTIAL
        out.error_message = "; ".join(degraded[:3])
        return out

    out.outcome = OUTCOME_SUCCESS if num_results else OUTCOME_EMPTY
    return out


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class AraxClient:
    """Submits TRAPI queries, classifies outcomes, and reads/writes the cache.

    Args:
        url: ARAX /query endpoint.
        timeout: per-request wall clock in seconds.
        cache: optional QueryCache. Cached timeouts are returned as hits so the
            executor can skip a known-slow query and decompose immediately.
        mock: optional MockArax for offline runs.
        on_version_change: called with (old, new) the first time a live
            response reports a different ARAX build than the cache holds.
    """

    def __init__(
        self,
        url: str = ARAX_QUERY_URL,
        timeout: float = REQUEST_TIMEOUT,
        cache: Optional[QueryCache] = None,
        mock: Optional["MockArax"] = None,
        on_version_change: Optional[Callable[[Optional[str], str], None]] = None,
        verbose: bool = True,
    ):
        if requests is None and mock is None:
            raise ImportError("arax_client needs `requests` (pip install requests) "
                              "unless a mock is supplied")
        self.url = url
        self.timeout = timeout
        self.cache = cache
        self.mock = mock
        self.on_version_change = on_version_change
        self.verbose = verbose
        self._seen_version: Optional[str] = None
        self.call_count = 0
        self.cache_hit_count = 0

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [arax] {msg}")

    # -- transport ---------------------------------------------------------

    def _post(self, body: Dict[str, Any], timeout: float) -> AraxResponse:
        """One HTTP round trip, with transport failures mapped to outcomes."""
        if self.mock is not None:
            return self.mock.submit(body, timeout=timeout)

        start = time.time()
        try:
            resp = requests.post(
                self.url,
                json=body,
                headers={"accept": "application/json"},
                timeout=timeout,
            )
        except requests.exceptions.Timeout:
            return AraxResponse(
                outcome=OUTCOME_TIMEOUT,
                elapsed_s=time.time() - start,
                error_message=f"read timeout after {timeout}s",
            )
        except requests.exceptions.RequestException as e:
            return AraxResponse(
                outcome=OUTCOME_ERROR,
                elapsed_s=time.time() - start,
                error_message=f"transport error: {e}",
            )

        elapsed = time.time() - start

        if resp.status_code != 200:
            # A gateway giving up is a timeout for routing purposes; a 4xx is
            # a malformed request and decomposing it would fail identically.
            outcome = (
                OUTCOME_TIMEOUT
                if resp.status_code in _TIMEOUT_HTTP_CODES
                else OUTCOME_ERROR
            )
            try:
                detail = json.dumps(resp.json())[:300]
            except Exception:
                detail = resp.text[:300]
            return AraxResponse(
                outcome=outcome,
                elapsed_s=elapsed,
                http_status=resp.status_code,
                error_message=f"HTTP {resp.status_code}: {detail}",
            )

        try:
            payload = resp.json()
        except ValueError as e:
            return AraxResponse(
                outcome=OUTCOME_ERROR,
                elapsed_s=elapsed,
                http_status=200,
                error_message=f"malformed JSON: {e}",
            )

        out = classify_response(payload)
        out.elapsed_s = elapsed
        out.http_status = 200
        return out

    # -- version drift -----------------------------------------------------

    def _check_version(self, resp: AraxResponse) -> None:
        """Invalidate the cache when the server build changes.

        A different ARAX build can answer the same query differently, so
        entries from the previous build are no longer trustworthy.
        """
        version = resp.arax_version
        if not version or version == self._seen_version:
            return
        previous, self._seen_version = self._seen_version, version
        if previous is None:
            return
        self.log(f"ARAX version changed {previous} -> {version}")
        if self.cache:
            dropped = self.cache.invalidate_by_version(version)
            self.log(f"invalidated {dropped} cache entries from the old build")
        if self.on_version_change:
            self.on_version_change(previous, version)

    # -- public API --------------------------------------------------------

    def query(
        self,
        built: BuiltQuery,
        max_results: int = DEFAULT_MAX_RESULTS,
        use_cache: bool = True,
        timeout: Optional[float] = None,
    ) -> AraxResponse:
        """Submit a BuiltQuery.

        A cached timeout is returned as a hit rather than re-executed: the
        executor reads it as "this shape already blew up" and decomposes
        without paying the wall clock again.
        """
        qg = built.query_graph
        timeout = timeout or self.timeout

        if use_cache and self.cache:
            hit = self.cache.get_arax_query(qg, endpoint=self.url)
            if hit is not None:
                self.cache_hit_count += 1
                out = (
                    classify_response(hit.response, max_results=max_results)
                    if hit.response is not None
                    else AraxResponse(
                        outcome=OUTCOME_TIMEOUT if hit.status == STATUS_TIMEOUT
                        else OUTCOME_ERROR,
                        error_message=hit.error_message,
                    )
                )
                out.from_cache = True
                out.elapsed_s = hit.elapsed_s or 0.0
                self.log(f"cache hit: {out}")
                return out

        body = to_trapi_message(built, max_results=max_results)
        self.call_count += 1
        resp = self._post(body, timeout=timeout)
        if resp.response is not None:
            resp.truncated = resp.num_results >= max_results

        self._check_version(resp)
        self._store(qg, resp, max_results)
        self.log(f"{resp}")
        return resp

    def dsl(
        self,
        actions: List[str],
        use_cache: bool = True,
        timeout: Optional[float] = None,
    ) -> AraxResponse:
        """Run an ARAXi DSL workflow, e.g. connect() for explanation queries."""
        # TRAPI requires `message` on every Query object, even when the whole
        # workflow is expressed as DSL: ARAX rejects the request outright
        # without it. The add_qnode actions populate the query graph inside
        # this empty message.
        body = {"message": {}, "operations": {"actions": actions}}
        request_key = {"actions": actions, "endpoint": self.url}

        if use_cache and self.cache:
            hit = self.cache.get(KIND_ARAX_DSL, request_key)
            if hit is not None:
                self.cache_hit_count += 1
                out = (
                    classify_response(hit.response)
                    if hit.response is not None
                    else AraxResponse(
                        outcome=OUTCOME_TIMEOUT if hit.status == STATUS_TIMEOUT
                        else OUTCOME_ERROR,
                        error_message=hit.error_message,
                    )
                )
                out.from_cache = True
                self.log(f"cache hit (dsl): {out}")
                return out

        self.call_count += 1
        resp = self._post(body, timeout=timeout or DSL_TIMEOUT)
        self._check_version(resp)

        if self.cache:
            self.cache.put(
                KIND_ARAX_DSL, request_key,
                response=resp.response, status=resp.cache_status(),
                num_results=resp.num_results, error_message=resp.error_message,
                endpoint=self.url, elapsed_s=resp.elapsed_s,
                arax_version=resp.arax_version,
                biolink_version=resp.biolink_version,
                trapi_version=resp.trapi_version,
            )

        self.log(f"dsl: {resp}")
        return resp

    def _store(self, qg: Dict[str, Any], resp: AraxResponse, max_results: int) -> None:
        if not self.cache:
            return
        meta = {"max_results": max_results, "truncated": resp.truncated}
        if resp.degraded_logs:
            meta["degraded_logs"] = resp.degraded_logs[:5]
        self.cache.put_arax_query(
            qg, response=resp.response, endpoint=self.url,
            status=resp.cache_status(), elapsed_s=resp.elapsed_s,
            error_message=resp.error_message, meta=meta,
        )

    def pathfinder(
        self,
        curie_a: str,
        curie_b: str,
        max_hops: int = 3,
        max_paths: int = 200,
        use_cache: bool = True,
        timeout: Optional[float] = None,
    ) -> AraxResponse:
        """Ask ARAX for routes between two pinned nodes.

        Sent as a native TRAPI pathfinder query — a query graph carrying
        `paths` rather than `edges` — because that is the shape ARAX's
        interpreter recognises. It then emits the `connect(action=connect_nodes)`
        step itself. Submitting that DSL directly fails: `connect_nodes` reads
        `query_graph.paths`, and `add_qnode` never creates one.

        `max_path_length` and `max_pathfinder_paths` are read from
        `query_options`, not from the action string.
        """
        query_graph = {
            "nodes": {"n0": {"ids": [curie_a]}, "n1": {"ids": [curie_b]}},
            "paths": {"p0": {"subject": "n0", "object": "n1"}},
        }
        body = {
            "message": {"query_graph": query_graph},
            "query_options": {
                "max_path_length": max_hops,
                "max_pathfinder_paths": max_paths,
            },
        }
        request_key = {"query_graph": query_graph, "endpoint": self.url,
                       "max_path_length": max_hops, "max_paths": max_paths}

        if use_cache and self.cache:
            hit = self.cache.get(KIND_ARAX_QUERY, request_key)
            if hit is not None:
                self.cache_hit_count += 1
                out = (
                    classify_response(hit.response) if hit.response is not None
                    else AraxResponse(
                        outcome=OUTCOME_TIMEOUT if hit.status == STATUS_TIMEOUT
                        else OUTCOME_ERROR,
                        error_message=hit.error_message,
                    )
                )
                out.from_cache = True
                return out

        self.call_count += 1
        resp = self._post(body, timeout=timeout or self.timeout)
        self._check_version(resp)

        if self.cache:
            self.cache.put(
                KIND_ARAX_QUERY, request_key, response=resp.response,
                status=resp.cache_status(), num_results=resp.num_results,
                error_message=resp.error_message, endpoint=self.url,
                elapsed_s=resp.elapsed_s, arax_version=resp.arax_version,
                biolink_version=resp.biolink_version,
                trapi_version=resp.trapi_version,
            )

        self.log(f"pathfinder {curie_a} -> {curie_b}: {resp}")
        return resp

    def meta_kg(self) -> Dict[str, Any]:
        """Fetch the meta knowledge graph.

        Used for diagnostics only: when a hop returns nothing, the executor can
        report whether ARAX even carries that subject/predicate/object triple.
        That is a fact for the planner to act on, not a decision the executor
        makes.
        """
        if self.mock is not None:
            return self.mock.meta_kg()
        try:
            resp = requests.get(ARAX_META_KG_URL, timeout=60)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            return {"error": str(e)}

    def stats(self) -> Dict[str, Any]:
        return {
            "live_calls": self.call_count,
            "cache_hits": self.cache_hit_count,
            "endpoint": self.url,
            "arax_version": self._seen_version,
        }


# ---------------------------------------------------------------------------
# Mock
# ---------------------------------------------------------------------------


class MockArax:
    """Offline ARAX substitute that can simulate timeouts.

    The point is testing the fallback path. Configure `timeout_when` to make
    multi-hop queries fail and single-hop queries succeed, and the executor's
    decomposition logic can be exercised with no network:

        mock = MockArax(timeout_when=lambda qg: len(qg["edges"]) > 1)

    `edges_for` supplies the synthetic graph: given a pinned CURIE and a
    predicate, return the neighbours to invent.
    """

    def __init__(
        self,
        timeout_when: Optional[Callable[[Dict[str, Any]], bool]] = None,
        edges_for: Optional[Callable[[str, str, str], List[tuple]]] = None,
        latency_s: float = 0.0,
        arax_version: str = "ARAX 1.6.2",
    ):
        self.timeout_when = timeout_when or (lambda qg: False)
        self.edges_for = edges_for
        self.latency_s = latency_s
        self.arax_version = arax_version
        self.submitted: List[Dict[str, Any]] = []

    def submit(self, body: Dict[str, Any], timeout: float = 120) -> AraxResponse:
        self.submitted.append(body)
        if self.latency_s:
            time.sleep(self.latency_s)

        qg = (body.get("message") or {}).get("query_graph") or {"nodes": {}, "edges": {}}

        if self.timeout_when(qg):
            return AraxResponse(
                outcome=OUTCOME_TIMEOUT,
                elapsed_s=float(timeout),
                error_message=f"[mock] simulated timeout after {timeout}s",
            )

        response = self._synthesize(qg)
        out = classify_response(response)
        out.elapsed_s = self.latency_s
        out.http_status = 200
        return out

    def _synthesize(self, qg: Dict[str, Any]) -> Dict[str, Any]:
        """Build a structurally valid TRAPI response for a one-hop graph."""
        nodes, edges = qg.get("nodes", {}), qg.get("edges", {})
        kg_nodes: Dict[str, Any] = {}
        kg_edges: Dict[str, Any] = {}
        results: List[Dict[str, Any]] = []

        for ekey, edge in edges.items():
            s_key, o_key = edge["subject"], edge["object"]
            s_node, o_node = nodes.get(s_key, {}), nodes.get(o_key, {})
            predicate = (edge.get("predicates") or ["biolink:related_to"])[0]

            # Whichever end is pinned drives generation.
            if s_node.get("ids"):
                pinned_key, pinned_ids, open_key, open_node = s_key, s_node["ids"], o_key, o_node
                pinned_is_subject = True
            elif o_node.get("ids"):
                pinned_key, pinned_ids, open_key, open_node = o_key, o_node["ids"], s_key, s_node
                pinned_is_subject = False
            else:
                continue

            open_category = (open_node.get("categories") or ["biolink:NamedThing"])[0]

            for pinned in pinned_ids:
                neighbours = (
                    self.edges_for(pinned, predicate, open_category)
                    if self.edges_for
                    else [(f"MOCK:{abs(hash((pinned, predicate))) % 9999}", "mock_neighbour")]
                )
                for curie, name in neighbours:
                    kg_nodes.setdefault(pinned, {"categories": (
                        nodes[pinned_key].get("categories") or ["biolink:NamedThing"]
                    ), "name": pinned})
                    kg_nodes.setdefault(curie, {"categories": [open_category], "name": name})

                    eid = f"mock_{ekey}_{len(kg_edges)}"
                    kg_edges[eid] = {
                        "subject": pinned if pinned_is_subject else curie,
                        "object": curie if pinned_is_subject else pinned,
                        "predicate": predicate,
                        "sources": [
                            {"resource_id": "infores:mock-kp",
                             "resource_role": "primary_knowledge_source"},
                        ],
                        "attributes": [
                            {"attribute_type_id": "biolink:publications",
                             "value": ["PMID:12345678"]},
                            {"attribute_type_id": "biolink:knowledge_level",
                             "value": "knowledge_assertion"},
                        ],
                    }
                    results.append({
                        "node_bindings": {
                            pinned_key: [{"id": pinned}],
                            open_key: [{"id": curie}],
                        },
                        "analyses": [{
                            "resource_id": "infores:arax",
                            "edge_bindings": {ekey: [{"id": eid}]},
                        }],
                    })

        return {
            "status": "Success",
            "description": "[mock] synthesized response",
            "tool_version": self.arax_version,
            "biolink_version": "4.2.5",
            "schema_version": "1.6.0",
            "logs": [],
            "message": {
                "query_graph": qg,
                "knowledge_graph": {"nodes": kg_nodes, "edges": kg_edges},
                "results": results,
            },
        }

    def meta_kg(self) -> Dict[str, Any]:
        return {"nodes": {}, "edges": [], "_mock": True}
