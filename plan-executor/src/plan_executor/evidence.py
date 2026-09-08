"""
evidence.py — Fetch publications and check whether they support their edges.

Two jobs:

  1. Retrieve publication metadata (title, abstract, year) for PMIDs attached
     to edges. Years unblock `min_year`, which cannot be applied earlier
     because TRAPI edges carry identifiers, not dates.
  2. Judge whether a cited paper actually describes the relationship its edge
     asserts.

What literature support is for
------------------------------
It is a warrant, not a measurement. The aim is not to score how well studied a
claim is — that tracks research funding, and would systematically penalize
rare diseases — but to find one citation a reader can follow to see why the
edge is plausible. That makes this a search with a stopping condition rather
than a survey, so reading every abstract on an edge is unnecessary.

Provenance decides what a bad citation means
--------------------------------------------
A curated source asserts a fact on the curator's authority and attaches
references as support. If those references turn out to be tangential, the
annotation was sloppy but the assertion still stands on its own footing —
neutral.

A text-mined edge has no independent assertion. The claim *is* the extraction
from that paper. If the paper does not say it, nothing remains, so such an
edge is marked `should_drop`. This is the only case where failed verification
removes an edge, and it never applies to an edge that was not actually
checked: `unverified` and `verified unsupported` are kept apart throughout.

Hallucination control
---------------------
Every verdict must carry a verbatim span from the abstract, and that span is
checked mechanically against the source text. A span that does not appear is
discarded rather than down-weighted, so the verdict is only as good as
something that can be confirmed without trusting the model. Abstracts are
judged one at a time: batching lets a strong paper make the model lenient
about weak ones beside it, and the aggregate count is more reliable computed
by arithmetic than produced by a model reading everything at once.

Concepts are passed as labels and synonyms, never CURIEs — an abstract says
"pirfenidone" or "Esbriet", never CHEBI:32016.

Usage
-----
    evidence = EvidenceGatherer(cache=cache, verifier=llm_verifier)
    years = evidence.publication_years(all_pmids)     # unblocks min_year
    support = evidence.verify_edge(edge_id, edge, nodes)
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Protocol, Sequence

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None

from .cache import QueryCache, KIND_PUBLICATION, STATUS_EMPTY, STATUS_ERROR, STATUS_SUCCESS
from .config import PUBLICATION_BATCH_SIZE, PUBLICATION_TIMEOUT, PUBLICATION_URL
from .postfilter import (
    edge_all_sources, edge_primary_source, edge_publications,
)
from .config import CURATED_SOURCES, TEXT_MINED_SOURCES, source_matches


KIND_LIT_VERDICT = "lit_verdict"

# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

SUPPORTS_DIRECTLY = "supports_directly"
SUPPORTS_INDIRECTLY = "supports_indirectly"
REFUTES = "refutes"
MENTIONS_NO_RELATION = "mentions_both_no_relation"
UNRELATED = "unrelated"
INSUFFICIENT_TEXT = "insufficient_text"

VALID_VERDICTS = {
    SUPPORTS_DIRECTLY, SUPPORTS_INDIRECTLY, REFUTES,
    MENTIONS_NO_RELATION, UNRELATED, INSUFFICIENT_TEXT,
}

SUPPORTING = {SUPPORTS_DIRECTLY, SUPPORTS_INDIRECTLY}

STATUS_NO_PUBS = "no_publications"
STATUS_UNVERIFIED = "unverified"
STATUS_SUPPORTED = "supported"
STATUS_UNSUPPORTED = "unsupported"

SOURCE_CURATED = "curated"
SOURCE_TEXT_MINED = "text_mined"
SOURCE_OTHER = "other"

#: Abstracts read per edge before giving up on finding a warrant.
DEFAULT_MAX_READ = 5
#: Minimum read before an early stop is allowed. Stopping at the first hit
#: would make every supported edge show a count of exactly one, leaving
#: nothing for ranking to distinguish; reading a few gives the count meaning
#: without surveying the whole literature.
DEFAULT_MIN_READ = 3


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class Publication:
    pmid: str
    title: str = ""
    abstract: str = ""
    year: Optional[int] = None
    journal: str = ""

    @property
    def has_text(self) -> bool:
        return bool(self.abstract and len(self.abstract) > 50)


@dataclass
class PublicationVerdict:
    pmid: str
    verdict: str
    quote: str = ""
    quote_verified: bool = False
    reason: str = ""
    year: Optional[int] = None

    @property
    def counts_as_support(self) -> bool:
        """A verdict only counts once its quote is confirmed in the abstract."""
        return self.verdict in SUPPORTING and self.quote_verified

    def to_dict(self) -> Dict[str, Any]:
        return {
            "pmid": self.pmid,
            "verdict": self.verdict,
            "quote": self.quote[:300],
            "quote_verified": self.quote_verified,
            "reason": self.reason[:300],
            "year": self.year,
        }


@dataclass
class LiteratureSupport:
    """Whether an edge's citations back up what the edge asserts."""

    edge_id: str
    subject_label: str = ""
    predicate: str = ""
    object_label: str = ""
    source_type: str = SOURCE_OTHER
    primary_source: Optional[str] = None
    status: str = STATUS_UNVERIFIED
    verdicts: List[PublicationVerdict] = field(default_factory=list)
    pmids_total: int = 0
    pmids_read: int = 0
    stopped_early: bool = False
    elapsed_s: float = 0.0

    @property
    def supporting(self) -> List[PublicationVerdict]:
        return [v for v in self.verdicts if v.counts_as_support]

    @property
    def num_supporting(self) -> int:
        return len(self.supporting)

    @property
    def best_citation(self) -> Optional[PublicationVerdict]:
        """The strongest verified support, for showing a reader why to believe it."""
        direct = [v for v in self.supporting if v.verdict == SUPPORTS_DIRECTLY]
        pool = direct or self.supporting
        return max(pool, key=lambda v: v.year or 0) if pool else None

    @property
    def should_drop(self) -> bool:
        """True only for a text-mined edge whose citations were checked and failed.

        A curated edge stands on the curator's assertion regardless of how
        well its references were chosen, so it is never dropped here. An edge
        that was never checked is never dropped either — absence of
        verification is not evidence against it.
        """
        return self.source_type == SOURCE_TEXT_MINED and self.status == STATUS_UNSUPPORTED

    @property
    def score(self) -> Optional[float]:
        """Literature support in [0, 1], or None when unknown.

        Kept out of the EPC score deliberately. Verification only runs for the
        candidates the budget reaches, so folding it in would make two edges
        with identical metadata score differently according to whether they
        happened to be checked. `rank.py` weights this separately, and None
        means unknown rather than bad.
        """
        if self.status in (STATUS_NO_PUBS, STATUS_UNVERIFIED):
            return None
        if self.status == STATUS_UNSUPPORTED:
            return 0.0
        # The step from no warrant to one warrant is the meaningful one;
        # beyond that, count mostly tracks how well funded the field is.
        return round(min(1.0, 0.6 + 0.4 * min(1.0, (self.num_supporting - 1) / 2)), 3)

    def to_dict(self) -> Dict[str, Any]:
        best = self.best_citation
        return {
            "edge_id": self.edge_id,
            "assertion": f"{self.subject_label} --[{self.predicate}]--> {self.object_label}",
            "source_type": self.source_type,
            "primary_source": self.primary_source,
            "status": self.status,
            "score": self.score,
            "num_supporting": self.num_supporting,
            "pmids_total": self.pmids_total,
            "pmids_read": self.pmids_read,
            "stopped_early": self.stopped_early,
            "should_drop": self.should_drop,
            "best_citation": best.to_dict() if best else None,
            "verdicts": [v.to_dict() for v in self.verdicts],
        }


class LiteratureVerifier(Protocol):
    """Judges one abstract against one asserted relationship. In llm.py."""

    def judge(
        self,
        subject_label: str,
        predicate: str,
        object_label: str,
        title: str,
        abstract: str,
        subject_synonyms: Sequence[str] = (),
        object_synonyms: Sequence[str] = (),
    ) -> Dict[str, str]:
        """Returns {'verdict': ..., 'quote': ..., 'reason': ...}.

        The quote must be copied verbatim from the abstract; it is checked.
        """
        ...


# ---------------------------------------------------------------------------
# Quote verification
# ---------------------------------------------------------------------------


def _normalize(text: str) -> str:
    return " ".join((text or "").lower().split())


def verify_quote(quote: str, abstract: str) -> bool:
    """Confirm a quoted span really occurs in the abstract.

    This is what turns "trust the model" into "the model must point at
    something checkable". Whitespace and case are normalized because models
    reflow text; nothing else is relaxed.
    """
    if not quote or not abstract:
        return False
    q, a = _normalize(quote), _normalize(abstract)
    if len(q) < 15:
        return False
    return q in a


# ---------------------------------------------------------------------------
# Gatherer
# ---------------------------------------------------------------------------


class EvidenceGatherer:
    """Fetches publications and verifies that they support their edges.

    Args:
        cache: publications are cached per PMID rather than per batch, so a
            50-id request where 45 are already known fetches only 5.
        verifier: LLM hook. Without it, metadata still works and verification
            reports `unverified`.
        max_read / min_read: abstracts read per edge, and the floor before an
            early stop is permitted.
    """

    def __init__(
        self,
        cache: Optional[QueryCache] = None,
        verifier: Optional[LiteratureVerifier] = None,
        max_read: int = DEFAULT_MAX_READ,
        min_read: int = DEFAULT_MIN_READ,
        batch_size: int = PUBLICATION_BATCH_SIZE,
        timeout: float = PUBLICATION_TIMEOUT,
        request_id: str = "plan-executor",
        mock_fetch: Optional[Callable] = None,
        verbose: bool = True,
    ):
        self.cache = cache
        self.verifier = verifier
        self.max_read = max_read
        self.min_read = min_read
        self.batch_size = batch_size
        self.timeout = timeout
        self.request_id = request_id
        self.mock_fetch = mock_fetch
        self.verbose = verbose
        self.fetch_calls = 0
        self.llm_calls = 0

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [evidence] {msg}")

    # -- fetching ----------------------------------------------------------

    async def _fetch_async(self, pubids: List[str]) -> Dict[str, Any]:
        """One call to the docmetadata service.

        `PMC:` identifiers must lose the colon before the service accepts them.
        """
        sanitized = [
            p.replace("PMC:", "PMC", 1) if p.upper().startswith("PMC:") else p
            for p in pubids
        ]
        params = {"pubids": ",".join(sanitized), "request_id": self.request_id}
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            resp = await client.get(PUBLICATION_URL, params=params)
            resp.raise_for_status()
            return resp.json()

    def _fetch_sync(self, pubids: List[str]) -> Dict[str, Any]:
        """Sync bridge over the async client.

        The executor is synchronous throughout; only this service is async, so
        the boundary is crossed here rather than colouring everything above.
        """
        if self.mock_fetch is not None:
            return self.mock_fetch(pubids)
        if httpx is None:
            raise ImportError("evidence needs `httpx` (pip install httpx)")
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self._fetch_async(pubids))
        # Already inside an event loop: run in a private one on this thread so
        # a caller who happens to be async does not deadlock.
        import concurrent.futures

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            return pool.submit(asyncio.run, self._fetch_async(pubids)).result()

    def fetch_publications(self, pmids: Sequence[str]) -> Dict[str, Publication]:
        """Fetch metadata for PMIDs, using and filling the cache."""
        wanted = sorted({p for p in pmids if p})
        if not wanted:
            return {}

        out: Dict[str, Publication] = {}
        missing: List[str] = []

        for pmid in wanted:
            hit = self.cache.get_publication(pmid) if self.cache else None
            if hit is not None and hit.response is not None:
                r = hit.response
                out[pmid] = Publication(
                    pmid=pmid, title=r.get("title", ""), abstract=r.get("abstract", ""),
                    year=r.get("year"), journal=r.get("journal", ""),
                )
            else:
                missing.append(pmid)

        for i in range(0, len(missing), self.batch_size):
            batch = missing[i:i + self.batch_size]
            try:
                self.fetch_calls += 1
                raw = self._fetch_sync(batch)
            except Exception as e:
                self.log(f"publication fetch failed for {len(batch)} ids: {e}")
                continue

            results = (raw or {}).get("results") or {}
            for pmid in batch:
                rec = results.get(pmid) or {}
                pub = Publication(
                    pmid=pmid,
                    title=rec.get("article_title") or rec.get("title") or "",
                    abstract=rec.get("abstract") or "",
                    year=_extract_year(rec),
                    journal=rec.get("journal_name") or rec.get("journal") or "",
                )
                out[pmid] = pub
                if self.cache:
                    self.cache.put_publication(
                        pmid,
                        {"title": pub.title, "abstract": pub.abstract,
                         "year": pub.year, "journal": pub.journal},
                        status=STATUS_SUCCESS if pub.title or pub.abstract else STATUS_EMPTY,
                    )

        return out

    def publication_years(self, pmids: Sequence[str]) -> Dict[str, int]:
        """PMID -> year, for populating `EvidencePolicy.publication_years`."""
        return {
            pmid: pub.year
            for pmid, pub in self.fetch_publications(pmids).items()
            if pub.year
        }

    # -- verification ------------------------------------------------------

    @staticmethod
    def classify_source(edge: Dict[str, Any]) -> str:
        """Decide whether an edge's claim rests on curation or on extraction."""
        primary = edge_primary_source(edge)
        if primary and source_matches(primary, TEXT_MINED_SOURCES):
            return SOURCE_TEXT_MINED
        if primary and source_matches(primary, CURATED_SOURCES):
            return SOURCE_CURATED
        # Fall back to the wider source chain: an aggregator wrapping SemMedDB
        # still leaves the claim resting on text mining.
        for src in edge_all_sources(edge):
            if source_matches(src, TEXT_MINED_SOURCES):
                return SOURCE_TEXT_MINED
        return SOURCE_OTHER

    def verify_edge(
        self,
        edge_id: str,
        edge: Dict[str, Any],
        nodes: Optional[Dict[str, Any]] = None,
        synonyms: Optional[Dict[str, List[str]]] = None,
    ) -> LiteratureSupport:
        """Check whether an edge's citations support what it asserts.

        Abstracts are read newest first and one at a time, stopping once a
        verified warrant has been found and `min_read` abstracts have been
        seen.
        """
        start = time.time()
        nodes = nodes or {}
        synonyms = synonyms or {}

        subject_curie, object_curie = edge.get("subject", ""), edge.get("object", "")
        support = LiteratureSupport(
            edge_id=edge_id,
            subject_label=(nodes.get(subject_curie) or {}).get("name") or subject_curie,
            object_label=(nodes.get(object_curie) or {}).get("name") or object_curie,
            predicate=(edge.get("predicate") or "").replace("biolink:", ""),
            source_type=self.classify_source(edge),
            primary_source=edge_primary_source(edge),
        )

        pmids = edge_publications(edge)
        support.pmids_total = len(pmids)
        if not pmids:
            support.status = STATUS_NO_PUBS
            support.elapsed_s = time.time() - start
            return support

        if self.verifier is None:
            support.status = STATUS_UNVERIFIED
            support.elapsed_s = time.time() - start
            return support

        pubs = self.fetch_publications(pmids)
        # Newest first: a recent paper is the more useful citation to show a
        # reader, and ordering deterministically keeps runs reproducible.
        ordered = sorted(
            (pubs[p] for p in pmids if p in pubs),
            key=lambda pub: (pub.year or 0, pub.pmid),
            reverse=True,
        )

        for pub in ordered[: self.max_read]:
            verdict = self._judge_one(support, pub, synonyms, subject_curie, object_curie)
            support.verdicts.append(verdict)
            support.pmids_read += 1

            if support.pmids_read >= self.min_read and support.num_supporting >= 1:
                support.stopped_early = support.pmids_read < min(
                    len(ordered), self.max_read
                )
                break

        support.status = (
            STATUS_SUPPORTED if support.num_supporting else STATUS_UNSUPPORTED
        )
        support.elapsed_s = time.time() - start
        return support

    def _judge_one(
        self,
        support: LiteratureSupport,
        pub: Publication,
        synonyms: Dict[str, List[str]],
        subject_curie: str,
        object_curie: str,
    ) -> PublicationVerdict:
        """Judge one abstract, then confirm the quote it returned."""
        key = {
            "pmid": pub.pmid, "subject": subject_curie,
            "predicate": support.predicate, "object": object_curie,
        }
        if self.cache:
            hit = self.cache.get(KIND_LIT_VERDICT, key)
            if hit is not None and hit.response is not None:
                return PublicationVerdict(**hit.response)

        if not pub.has_text:
            verdict = PublicationVerdict(
                pmid=pub.pmid, verdict=INSUFFICIENT_TEXT, year=pub.year,
                reason="no abstract text available",
            )
        else:
            try:
                self.llm_calls += 1
                raw = self.verifier.judge(
                    subject_label=support.subject_label,
                    predicate=support.predicate,
                    object_label=support.object_label,
                    title=pub.title,
                    abstract=pub.abstract,
                    subject_synonyms=synonyms.get(subject_curie, []),
                    object_synonyms=synonyms.get(object_curie, []),
                )
            except Exception as e:
                return PublicationVerdict(
                    pmid=pub.pmid, verdict=INSUFFICIENT_TEXT, year=pub.year,
                    reason=f"verification unavailable: {e}",
                )

            label = (raw or {}).get("verdict", "")
            quote = (raw or {}).get("quote", "") or ""
            if label not in VALID_VERDICTS:
                label = INSUFFICIENT_TEXT

            verdict = PublicationVerdict(
                pmid=pub.pmid,
                verdict=label,
                quote=quote,
                quote_verified=verify_quote(quote, pub.abstract),
                reason=(raw or {}).get("reason", ""),
                year=pub.year,
            )

            # An unconfirmable quote means the justification cannot be checked,
            # so the verdict is discarded rather than merely discounted.
            if verdict.verdict in SUPPORTING and not verdict.quote_verified:
                verdict.reason = (
                    f"[quote not found in abstract; verdict discarded] "
                    f"{verdict.reason}"
                )

        if self.cache:
            self.cache.put(
                KIND_LIT_VERDICT, key, response=verdict.__dict__,
                status=STATUS_SUCCESS,
            )
        return verdict

    # -- batch -------------------------------------------------------------

    def verify_edges(
        self,
        edge_ids: Sequence[str],
        edges: Dict[str, Dict[str, Any]],
        nodes: Optional[Dict[str, Any]] = None,
        synonyms: Optional[Dict[str, List[str]]] = None,
    ) -> Dict[str, LiteratureSupport]:
        """Verify several edges, prefetching all their publications at once."""
        targets = [eid for eid in edge_ids if eid in edges]
        all_pmids = sorted({p for eid in targets for p in edge_publications(edges[eid])})
        if all_pmids:
            self.fetch_publications(all_pmids)
            self.log(f"fetched metadata for {len(all_pmids)} publication(s)")

        out: Dict[str, LiteratureSupport] = {}
        for eid in targets:
            out[eid] = self.verify_edge(eid, edges[eid], nodes, synonyms)
        return out

    @staticmethod
    def summarize(supports: Dict[str, LiteratureSupport]) -> Dict[str, Any]:
        """Aggregate view for the ledger."""
        from collections import Counter

        statuses: Counter = Counter()
        by_source: Counter = Counter()
        drops: List[str] = []
        for eid, s in supports.items():
            statuses[s.status] += 1
            by_source[s.source_type] += 1
            if s.should_drop:
                drops.append(eid)

        return {
            "edges_checked": len(supports),
            "status": dict(statuses),
            "source_type": dict(by_source),
            "edges_to_drop": drops,
            "drop_reason": (
                "text-mined edges whose cited papers were read and did not "
                "support the assertion"
            ),
            "total_abstracts_read": sum(s.pmids_read for s in supports.values()),
        }

    def stats(self) -> Dict[str, Any]:
        return {"publication_fetches": self.fetch_calls, "llm_judgements": self.llm_calls}


def _extract_year(record: Dict[str, Any]) -> Optional[int]:
    """Pull a publication year out of whichever field carries it."""
    for key in ("pub_year", "year", "publication_date", "pub_date", "date"):
        value = record.get(key)
        if value is None:
            continue
        if isinstance(value, int) and 1800 < value < 2200:
            return value
        text = str(value)
        for i in range(len(text) - 3):
            chunk = text[i:i + 4]
            if chunk.isdigit() and 1800 < int(chunk) < 2200:
                return int(chunk)
    return None
