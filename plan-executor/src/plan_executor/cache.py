"""
cache.py — Local sqlite cache for ARAX queries and entity resolution.

Purpose
-------
Executing a plan means asking ARAX the same questions repeatedly: two paths in
a plan often share a hop, a decomposed query sweeps intermediates in groups,
and the planner may re-issue a revised plan that overlaps the previous one.
This module makes every such repeat free.

What is cached
--------------
Raw responses only — never post-filtered output. `evidence_policy`,
predicate whitelists, and ranking all change between plan iterations; if the
cache held filtered results, changing a filter would force a re-query. Filter
on read instead.

Negative results are cached deliberately:

  * `empty`   — a valid response with zero results. This is *evidence*, not a
                failure: it is what licenses a `no_answer` verdict. Cached at
                the same TTL as a success.
  * `timeout` — the query exceeded the wall clock. Cached so the executor can
                skip straight to decomposition instead of waiting again.
  * `error`   — transient failures, cached briefly so a retry storm does not
                hammer ARAX, but expiring soon since the cause usually clears.

Keying
------
The key is a sha256 over a canonicalized form of the request: dict keys
sorted, lists sorted, and empty/None fields dropped. That makes the key
insensitive to field ordering and to whether the builder wrote explicit TRAPI
defaults (`"ids": null`, `"constraints": []`) or omitted them.

Two assumptions the caller must uphold:

  1. CURIEs are already normalized before they reach the cache. Normalization
     is a network operation and belongs in the resolver; the cache does not
     do it. Un-normalized CURIEs simply produce a different key.
  2. Query-graph node keys are assigned deterministically by the builder
     (n00, n01, ... in hop order). The cache does not attempt graph
     isomorphism, so a graph relabelled nA/nB would key differently.

The ARAX endpoint URL is part of the key material: staging and production can
answer the same question differently, and silently sharing entries between
them would be wrong.

Usage
-----
    from .cache import QueryCache

    cache = QueryCache("runs/cache.sqlite")

    hit = cache.get_arax_query(query_graph, endpoint=ARAX_URL)
    if hit is None:
        response, elapsed = submit(query_graph)
        cache.put_arax_query(query_graph, response, endpoint=ARAX_URL,
                             elapsed_s=elapsed)
    elif hit.status == "timeout":
        ...   # skip the direct attempt, decompose immediately

CLI
---
    python cache.py stats  runs/cache.sqlite
    python cache.py purge  runs/cache.sqlite
    python cache.py clear  runs/cache.sqlite --kind arax_query
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import zlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Status vocabulary and default lifetimes
# ---------------------------------------------------------------------------

STATUS_SUCCESS = "success"    # results returned
STATUS_EMPTY = "empty"        # valid response, zero results — real evidence
STATUS_TIMEOUT = "timeout"    # exceeded wall clock
STATUS_ERROR = "error"        # HTTP / TRAPI-level failure

#: Seconds each status stays usable. A hit past its TTL is treated as a miss.
#: `empty` matches `success` because a zero-result answer is a finding, not a
#: failure. `timeout` is shorter because ARAX performance shifts. `error` is
#: very short because such failures are usually transient.
DEFAULT_TTL_SECONDS = {
    STATUS_SUCCESS: 30 * 24 * 3600,   # 30 days
    STATUS_EMPTY: 30 * 24 * 3600,     # 30 days
    STATUS_TIMEOUT: 7 * 24 * 3600,    # 7 days
    STATUS_ERROR: 3600,               # 1 hour
}

#: Cache kinds. Kept as constants so typos surface at import rather than as
#: silent misses.
KIND_ARAX_QUERY = "arax_query"
KIND_ARAX_DSL = "arax_dsl"
KIND_CONNECT = "connect"          # two-endpoint path queries
KIND_NAME_LOOKUP = "name_lookup"
KIND_NODE_NORM = "node_norm"
KIND_PUBLICATION = "publication"  # abstracts from docmetadata

ALL_KINDS = (
    KIND_ARAX_QUERY, KIND_ARAX_DSL, KIND_CONNECT,
    KIND_NAME_LOOKUP, KIND_NODE_NORM, KIND_PUBLICATION,
)


# ---------------------------------------------------------------------------
# Canonicalization and keying
# ---------------------------------------------------------------------------


#: TRAPI fields whose default value carries no meaning, so that writing the
#: default and omitting the field are the same query. ARAX echoes a submitted
#: query graph back with every default made explicit (`"is_set": false`,
#: `"constraints": []`, ...), while a hand-built graph omits them. Without
#: this, the same question keys two different ways depending on whether the
#: graph came from the builder or from a response.
_FIELD_DEFAULTS = {
    "is_set": False,
    "exclude": False,
    "set_interpretation": "BATCH",
}


def _is_default(key: str, value: Any) -> bool:
    """True when `value` is the meaningless default for `key`.

    The type check guards against Python treating `False == 0` as equal, which
    would otherwise strip a legitimate numeric zero.
    """
    if key not in _FIELD_DEFAULTS:
        return False
    default = _FIELD_DEFAULTS[key]
    return type(value) is type(default) and value == default


def _strip_empty(obj: Any) -> Any:
    """Drop None values, empty containers, and meaningless defaults.

    TRAPI serializers vary in whether they emit unset fields as `null`, as
    `[]`, as an explicit default, or not at all. All of these mean the same
    thing, so they must hash the same. Values that are genuinely meaningful —
    a numeric 0, a `False` on a field with no registered default — are kept.
    """
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if _is_default(k, v):
                continue
            cleaned = _strip_empty(v)
            if cleaned is None:
                continue
            if isinstance(cleaned, (list, dict)) and len(cleaned) == 0:
                continue
            out[k] = cleaned
        return out
    if isinstance(obj, list):
        cleaned_items = []
        for v in obj:
            cleaned = _strip_empty(v)
            if cleaned is None:
                continue
            if isinstance(cleaned, (list, dict)) and len(cleaned) == 0:
                continue
            cleaned_items.append(cleaned)
        return cleaned_items
    return obj


def _sort_key(obj: Any) -> str:
    """Stable ordering key for heterogeneous list elements."""
    return json.dumps(obj, sort_keys=True, default=str)


def canonicalize(obj: Any) -> Any:
    """Recursively sort dict keys and list elements for stable hashing.

    All lists are sorted, not just the ones TRAPI treats as sets (`ids`,
    `categories`, `predicates`). This is safe because the canonical form is
    only ever hashed, never used to reconstruct a query: sorting can merge two
    orderings of the same content, which is the intent, but cannot merge
    genuinely different content.
    """
    if isinstance(obj, dict):
        return {k: canonicalize(obj[k]) for k in sorted(obj)}
    if isinstance(obj, list):
        return sorted((canonicalize(x) for x in obj), key=_sort_key)
    return obj


def make_key(kind: str, request: Dict[str, Any]) -> str:
    """Compute the cache key for a request.

    The kind is folded in so that two different request types cannot collide
    even if their payloads happen to serialize identically.
    """
    canonical = canonicalize(_strip_empty(request))
    material = json.dumps(
        {"kind": kind, "request": canonical},
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Entry type
# ---------------------------------------------------------------------------


@dataclass
class CacheEntry:
    key: str
    kind: str
    request: Dict[str, Any]
    response: Optional[Dict[str, Any]]
    status: str
    fetched_at: str
    num_results: Optional[int] = None
    error_message: Optional[str] = None
    arax_version: Optional[str] = None
    biolink_version: Optional[str] = None
    trapi_version: Optional[str] = None
    endpoint: Optional[str] = None
    elapsed_s: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def age_seconds(self) -> float:
        try:
            fetched = datetime.fromisoformat(self.fetched_at)
        except ValueError:
            return float("inf")
        if fetched.tzinfo is None:
            fetched = fetched.replace(tzinfo=timezone.utc)
        return (datetime.now(timezone.utc) - fetched).total_seconds()

    @property
    def is_usable_result(self) -> bool:
        """True when this entry carries an answer, including a zero-result one.

        A `timeout` or `error` entry is a real cache hit — it tells the
        executor what happened last time — but it is not an answer.
        """
        return self.status in (STATUS_SUCCESS, STATUS_EMPTY)

    def __repr__(self) -> str:
        return (
            f"<CacheEntry {self.kind} {self.status} "
            f"results={self.num_results} age={self.age_seconds / 3600:.1f}h "
            f"key={self.key[:12]}>"
        )


# ---------------------------------------------------------------------------
# Store
# ---------------------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS cache_entries (
    key             TEXT PRIMARY KEY,
    kind            TEXT NOT NULL,
    request_json    TEXT NOT NULL,
    response_blob   BLOB,
    compressed      INTEGER NOT NULL DEFAULT 0,
    status          TEXT NOT NULL,
    num_results     INTEGER,
    error_message   TEXT,
    arax_version    TEXT,
    biolink_version TEXT,
    trapi_version   TEXT,
    endpoint        TEXT,
    elapsed_s       REAL,
    meta_json       TEXT,
    fetched_at      TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_cache_kind   ON cache_entries(kind);
CREATE INDEX IF NOT EXISTS idx_cache_status ON cache_entries(kind, status);
CREATE INDEX IF NOT EXISTS idx_cache_time   ON cache_entries(fetched_at);
"""


class QueryCache:
    """sqlite-backed cache keyed on canonicalized requests.

    Args:
        path: sqlite file. Parent directories are created. Use ":memory:" for
            an ephemeral cache in tests.
        ttl_seconds: per-status lifetimes; merged over DEFAULT_TTL_SECONDS.
        compress: zlib-compress stored responses. ARAX responses are commonly
            multi-megabyte JSON, which compresses well.
        enabled: when False, every read misses and every write is discarded.
            Lets callers honour a --no-cache flag without branching.
    """

    def __init__(
        self,
        path: str = "cache.sqlite",
        ttl_seconds: Optional[Dict[str, int]] = None,
        compress: bool = True,
        enabled: bool = True,
    ):
        self.path = path
        self.compress = compress
        self.enabled = enabled
        self.ttl = dict(DEFAULT_TTL_SECONDS)
        if ttl_seconds:
            self.ttl.update(ttl_seconds)

        # Counters for the run summary; the ledger reports these so a run's
        # cost is visible without instrumenting every call site.
        self.hits = 0
        self.misses = 0
        self.stale_misses = 0
        self.writes = 0

        if path != ":memory:":
            parent = os.path.dirname(os.path.abspath(path))
            os.makedirs(parent, exist_ok=True)

        self._conn = sqlite3.connect(path, timeout=30.0)
        self._conn.row_factory = sqlite3.Row
        # WAL lets readers proceed during a write, which matters once the
        # executor runs intermediate groups concurrently.
        if path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.execute("PRAGMA busy_timeout=30000")
        self._conn.executescript(_SCHEMA)
        self._conn.commit()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "QueryCache":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # -- serialization -----------------------------------------------------

    def _encode(self, response: Optional[Dict[str, Any]]):
        if response is None:
            return None, 0
        raw = json.dumps(response, separators=(",", ":"), default=str).encode("utf-8")
        if self.compress:
            return zlib.compress(raw, 6), 1
        return raw, 0

    @staticmethod
    def _decode(blob, compressed: int) -> Optional[Dict[str, Any]]:
        if blob is None:
            return None
        raw = zlib.decompress(blob) if compressed else blob
        return json.loads(raw.decode("utf-8"))

    def _row_to_entry(self, row: sqlite3.Row) -> CacheEntry:
        return CacheEntry(
            key=row["key"],
            kind=row["kind"],
            request=json.loads(row["request_json"]),
            response=self._decode(row["response_blob"], row["compressed"]),
            status=row["status"],
            fetched_at=row["fetched_at"],
            num_results=row["num_results"],
            error_message=row["error_message"],
            arax_version=row["arax_version"],
            biolink_version=row["biolink_version"],
            trapi_version=row["trapi_version"],
            endpoint=row["endpoint"],
            elapsed_s=row["elapsed_s"],
            meta=json.loads(row["meta_json"]) if row["meta_json"] else {},
        )

    # -- core get / put ----------------------------------------------------

    def get(
        self,
        kind: str,
        request: Dict[str, Any],
        ignore_ttl: bool = False,
    ) -> Optional[CacheEntry]:
        """Look up an entry. Returns None on miss or on an expired entry.

        An expired entry is left in place rather than deleted, so `purge()`
        remains the single point where data is removed.
        """
        if not self.enabled:
            self.misses += 1
            return None

        key = make_key(kind, request)
        row = self._conn.execute(
            "SELECT * FROM cache_entries WHERE key = ?", (key,)
        ).fetchone()

        if row is None:
            self.misses += 1
            return None

        entry = self._row_to_entry(row)

        if not ignore_ttl:
            ttl = self.ttl.get(entry.status, DEFAULT_TTL_SECONDS[STATUS_ERROR])
            if entry.age_seconds > ttl:
                self.stale_misses += 1
                self.misses += 1
                return None

        self.hits += 1
        return entry

    def put(
        self,
        kind: str,
        request: Dict[str, Any],
        response: Optional[Dict[str, Any]] = None,
        status: str = STATUS_SUCCESS,
        num_results: Optional[int] = None,
        error_message: Optional[str] = None,
        endpoint: Optional[str] = None,
        elapsed_s: Optional[float] = None,
        arax_version: Optional[str] = None,
        biolink_version: Optional[str] = None,
        trapi_version: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[CacheEntry]:
        """Store an entry, replacing any existing one for the same key."""
        if not self.enabled:
            return None

        key = make_key(kind, request)
        blob, compressed = self._encode(response)
        fetched_at = datetime.now(timezone.utc).isoformat()

        self._conn.execute(
            """
            INSERT OR REPLACE INTO cache_entries
                (key, kind, request_json, response_blob, compressed, status,
                 num_results, error_message, arax_version, biolink_version,
                 trapi_version, endpoint, elapsed_s, meta_json, fetched_at)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                key, kind,
                json.dumps(request, sort_keys=True, default=str),
                blob, compressed, status, num_results, error_message,
                arax_version, biolink_version, trapi_version,
                endpoint, elapsed_s,
                json.dumps(meta, default=str) if meta else None,
                fetched_at,
            ),
        )
        self._conn.commit()
        self.writes += 1

        return CacheEntry(
            key=key, kind=kind, request=request, response=response,
            status=status, fetched_at=fetched_at, num_results=num_results,
            error_message=error_message, arax_version=arax_version,
            biolink_version=biolink_version, trapi_version=trapi_version,
            endpoint=endpoint, elapsed_s=elapsed_s, meta=meta or {},
        )

    # -- ARAX-specific helpers --------------------------------------------

    @staticmethod
    def _extract_versions(response: Dict[str, Any]) -> Dict[str, Optional[str]]:
        """Pull version stamps out of a TRAPI response.

        ARAX reports `tool_version` ("ARAX 1.6.2"), `schema_version` (the
        TRAPI version) and `biolink_version`. Recording them lets a later run
        detect that the cache predates a server upgrade.
        """
        if not isinstance(response, dict):
            return {"arax_version": None, "biolink_version": None, "trapi_version": None}
        return {
            "arax_version": response.get("tool_version"),
            "biolink_version": response.get("biolink_version"),
            "trapi_version": response.get("schema_version"),
        }

    @staticmethod
    def _count_results(response: Dict[str, Any]) -> int:
        try:
            return len(response["message"]["results"] or [])
        except (KeyError, TypeError):
            return 0

    def get_arax_query(
        self,
        query_graph: Dict[str, Any],
        endpoint: Optional[str] = None,
        ignore_ttl: bool = False,
    ) -> Optional[CacheEntry]:
        """Look up a TRAPI query graph result.

        Check `.is_usable_result` on the return value: a `timeout` entry is a
        hit worth acting on (decompose immediately) but is not an answer.
        """
        return self.get(
            KIND_ARAX_QUERY,
            {"query_graph": query_graph, "endpoint": endpoint},
            ignore_ttl=ignore_ttl,
        )

    def put_arax_query(
        self,
        query_graph: Dict[str, Any],
        response: Optional[Dict[str, Any]] = None,
        endpoint: Optional[str] = None,
        status: Optional[str] = None,
        elapsed_s: Optional[float] = None,
        error_message: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> Optional[CacheEntry]:
        """Store a TRAPI query result.

        When `status` is omitted it is inferred: a response with results is
        `success`, one without is `empty`, and a missing response is `error`.
        The success/empty split is what lets a later run distinguish "no path
        exists" from "we never asked".
        """
        request = {"query_graph": query_graph, "endpoint": endpoint}
        versions = self._extract_versions(response or {})

        if status is None:
            if response is None:
                status = STATUS_ERROR
            else:
                status = STATUS_SUCCESS if self._count_results(response) else STATUS_EMPTY

        num_results = self._count_results(response) if response else 0

        return self.put(
            KIND_ARAX_QUERY, request, response=response, status=status,
            num_results=num_results, error_message=error_message,
            endpoint=endpoint, elapsed_s=elapsed_s, meta=meta, **versions,
        )

    def record_timeout(
        self,
        query_graph: Dict[str, Any],
        endpoint: Optional[str] = None,
        elapsed_s: Optional[float] = None,
        error_message: Optional[str] = None,
        kind: str = KIND_ARAX_QUERY,
    ) -> Optional[CacheEntry]:
        """Record that a query timed out.

        Worth its own method because the executor reads these entries as a
        routing signal: a query known to have timed out skips the direct
        attempt and goes straight to pivot-first decomposition.
        """
        return self.put(
            kind,
            {"query_graph": query_graph, "endpoint": endpoint},
            response=None,
            status=STATUS_TIMEOUT,
            num_results=0,
            error_message=error_message or "timeout",
            endpoint=endpoint,
            elapsed_s=elapsed_s,
        )

    # -- resolution helpers ------------------------------------------------

    def get_name_lookup(self, query: str, biolink_type: Optional[str] = None):
        return self.get(KIND_NAME_LOOKUP, {"query": query.strip().lower(),
                                           "biolink_type": biolink_type})

    def put_name_lookup(self, query: str, response: Dict[str, Any],
                        biolink_type: Optional[str] = None, **kw):
        return self.put(KIND_NAME_LOOKUP,
                        {"query": query.strip().lower(), "biolink_type": biolink_type},
                        response=response, **kw)

    def get_node_norm(self, curie: str):
        return self.get(KIND_NODE_NORM, {"curie": curie})

    def put_node_norm(self, curie: str, response: Dict[str, Any], **kw):
        return self.put(KIND_NODE_NORM, {"curie": curie}, response=response, **kw)

    def get_publication(self, pubid: str):
        return self.get(KIND_PUBLICATION, {"pubid": pubid})

    def put_publication(self, pubid: str, response: Dict[str, Any], **kw):
        return self.put(KIND_PUBLICATION, {"pubid": pubid}, response=response, **kw)

    def get_publications(self, pubids: List[str]):
        """Batch lookup. Returns (found, missing).

        Publications are cached one per ID rather than per batch, so a request
        for 50 IDs where 45 are cached only fetches the remaining 5.
        """
        found: Dict[str, Any] = {}
        missing: List[str] = []
        for pid in pubids:
            entry = self.get_publication(pid)
            if entry and entry.is_usable_result:
                found[pid] = entry.response
            else:
                missing.append(pid)
        return found, missing

    # -- maintenance -------------------------------------------------------

    def stats(self) -> Dict[str, Any]:
        rows = self._conn.execute(
            """
            SELECT kind, status, COUNT(*) n,
                   SUM(LENGTH(COALESCE(response_blob, ''))) bytes,
                   AVG(elapsed_s) avg_elapsed
            FROM cache_entries GROUP BY kind, status
            """
        ).fetchall()

        by_kind: Dict[str, Dict[str, Any]] = {}
        total_bytes = 0
        total_rows = 0
        for r in rows:
            by_kind.setdefault(r["kind"], {})[r["status"]] = {
                "count": r["n"],
                "bytes": r["bytes"] or 0,
                "avg_elapsed_s": round(r["avg_elapsed"], 2) if r["avg_elapsed"] else None,
            }
            total_bytes += r["bytes"] or 0
            total_rows += r["n"]

        return {
            "path": self.path,
            "entries": total_rows,
            "stored_bytes": total_bytes,
            "by_kind": by_kind,
            "session": {
                "hits": self.hits,
                "misses": self.misses,
                "stale_misses": self.stale_misses,
                "writes": self.writes,
                "hit_rate": round(self.hits / (self.hits + self.misses), 3)
                if (self.hits + self.misses) else None,
            },
        }

    def purge_expired(self) -> int:
        """Delete entries past their TTL. Returns the number removed."""
        now = datetime.now(timezone.utc)
        removed = 0
        for row in self._conn.execute(
            "SELECT key, status, fetched_at FROM cache_entries"
        ).fetchall():
            ttl = self.ttl.get(row["status"], DEFAULT_TTL_SECONDS[STATUS_ERROR])
            try:
                fetched = datetime.fromisoformat(row["fetched_at"])
                if fetched.tzinfo is None:
                    fetched = fetched.replace(tzinfo=timezone.utc)
                age = (now - fetched).total_seconds()
            except ValueError:
                age = float("inf")
            if age > ttl:
                self._conn.execute("DELETE FROM cache_entries WHERE key = ?", (row["key"],))
                removed += 1
        self._conn.commit()
        return removed

    def invalidate_by_version(self, current_arax_version: str) -> int:
        """Drop entries produced by a different ARAX build.

        Call this after any live response reveals a version other than the one
        the cache was populated under: an upgraded server can answer the same
        query differently, so old entries are no longer trustworthy.
        """
        cur = self._conn.execute(
            "DELETE FROM cache_entries WHERE arax_version IS NOT NULL "
            "AND arax_version != ?",
            (current_arax_version,),
        )
        self._conn.commit()
        return cur.rowcount

    def clear(self, kind: Optional[str] = None) -> int:
        if kind:
            cur = self._conn.execute("DELETE FROM cache_entries WHERE kind = ?", (kind,))
        else:
            cur = self._conn.execute("DELETE FROM cache_entries")
        self._conn.commit()
        return cur.rowcount

    def vacuum(self) -> None:
        """Reclaim disk space after large deletions."""
        self._conn.execute("VACUUM")
        self._conn.commit()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.1f}{unit}"
        n /= 1024.0
    return f"{n:.1f}TB"


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Inspect or maintain the query cache.")
    ap.add_argument("command", choices=["stats", "purge", "clear", "vacuum"])
    ap.add_argument("path", help="Path to cache.sqlite")
    ap.add_argument("--kind", choices=ALL_KINDS, help="Restrict 'clear' to one kind")
    args = ap.parse_args()

    if not os.path.exists(args.path):
        print(f"No cache at {args.path}")
        return 1

    with QueryCache(args.path) as cache:
        if args.command == "stats":
            s = cache.stats()
            print(f"{s['path']}: {s['entries']} entries, {_human(s['stored_bytes'])}")
            for kind, statuses in sorted(s["by_kind"].items()):
                print(f"  {kind}")
                for status, d in sorted(statuses.items()):
                    el = f", avg {d['avg_elapsed_s']}s" if d["avg_elapsed_s"] else ""
                    print(f"      {status:8s} {d['count']:6d}  {_human(d['bytes'])}{el}")

        elif args.command == "purge":
            print(f"Removed {cache.purge_expired()} expired entries")

        elif args.command == "clear":
            print(f"Removed {cache.clear(args.kind)} entries"
                  + (f" of kind {args.kind}" if args.kind else ""))

        elif args.command == "vacuum":
            cache.vacuum()
            print("Vacuumed")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
