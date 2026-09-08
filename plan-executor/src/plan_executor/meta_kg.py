"""
meta_kg.py — Check a plan's hops against the triples ARAX actually holds.

ARAX publishes a meta knowledge graph: every (subject category, predicate,
object category) combination its knowledge sources can answer. A hop naming a
combination absent from that list cannot return anything, no matter how the
rest of the plan is written.

Catching this before execution matters because the failure is otherwise
expensive and ambiguous. A signature-reversal plan asking for
`Gene positively_correlated_with Disease` spent four queries returning nothing
across two paths, and the result — `no_answer` — is indistinguishable from a
predicate that exists but has no data for the given entities. One meta-KG
fetch answers it up front.

Why this lives in the executor
------------------------------
The meta-KG is ARAX's own description of itself, in the same category as the
`connect()` hop ceiling and ChEMBL's attribute names. The planner reasons about
questions; what one knowledge graph happens to carry is not something it should
have to track.

What the check can and cannot do
--------------------------------
It establishes that a triple *type* is supported. It says nothing about whether
data exists for particular entities: `Gene contributes_to Disease` is in the
meta-KG and yet TNBC has four such edges. An absent triple is therefore
conclusive, while a present one is only permission to try — localization in the
executor remains the way to find an empty hop.

Substitutes are reported, never chosen. Replacing `positively_correlated_with`
with `contributes_to` turns "up-regulated in" into "causally contributes to",
which is a different claim about the biology. Listing what ARAX supports is
ARAX knowledge; deciding which one answers the question is not.

Usage
-----
    meta = MetaKnowledgeGraph.load(client, cache=cache)
    issues = meta.check_plan(plan_input)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .cache import QueryCache, STATUS_SUCCESS


KIND_META_KG = "meta_kg"

#: Categories too broad to constitute a match. A meta-edge declared on
#: BiologicalEntity says only that the endpoint is something biological, which
#: is true of nearly every node — accepting it would let any hop pass and the
#: check would confirm plans that return nothing. Observed directly: a hop
#: asking for Gene positively_correlated_with Disease matched a
#: Gene/BiologicalEntity meta-edge and was reported supported, while the same
#: query against ARAX returned zero results.
_TOO_GENERIC = {
    "NamedThing", "Entity", "BiologicalEntity", "OntologyClass",
    "ChemicalEntityOrGeneOrGeneProduct", "ChemicalEntityOrProteinOrPolypeptide",
    "PhysicalEssenceOrOccurrent", "ThingWithTaxon",
}

#: Categories interchangeable in practice, used when the Biolink model is
#: unavailable. Confined to siblings that ARAX's nodes routinely carry
#: together — a chemical is typically typed both SmallMolecule and Drug — and
#: deliberately excluding the broad parents above.
_FALLBACK_EQUIVALENTS = {
    "Gene": {"Gene", "GeneOrGeneProduct", "GenomicEntity"},
    "Protein": {"Protein", "GeneOrGeneProduct", "Polypeptide"},
    "SmallMolecule": {
        "SmallMolecule", "ChemicalEntity", "MolecularEntity", "Drug",
        "ChemicalOrDrugOrTreatment",
    },
    "Drug": {"Drug", "ChemicalEntity", "SmallMolecule", "ChemicalOrDrugOrTreatment"},
    "Disease": {"Disease", "DiseaseOrPhenotypicFeature"},
    "PhenotypicFeature": {"PhenotypicFeature", "DiseaseOrPhenotypicFeature"},
}


def _bare(value: str) -> str:
    return (value or "").replace("biolink:", "")


def _prefixed(value: str) -> str:
    v = value or ""
    return v if v.startswith("biolink:") else f"biolink:{v}"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class HopCheck:
    """Whether one hop names a triple ARAX supports."""

    path_id: str
    hop_index: int
    subject_category: str
    predicate: str
    object_category: str
    supported: bool = False
    matched_as: Optional[str] = None
    alternatives: List[str] = field(default_factory=list)
    reversed_available: bool = False

    @property
    def triple(self) -> str:
        return (f"{_bare(self.subject_category)} "
                f"--[{_bare(self.predicate)}]--> "
                f"{_bare(self.object_category)}")

    def message(self) -> str:
        if self.supported:
            return f"{self.triple} is supported"

        parts = [f"ARAX holds no {self.triple} edges, so this hop cannot return anything."]
        if self.reversed_available:
            parts.append(
                f"The reverse direction, {_bare(self.object_category)} "
                f"--[{_bare(self.predicate)}]--> {_bare(self.subject_category)}, "
                f"is supported — the hop may have subject and object swapped."
            )
        if self.alternatives:
            parts.append(
                f"Predicates ARAX does support between these categories: "
                f"{self.alternatives[:12]}."
            )
        else:
            parts.append(
                f"ARAX supports no predicate at all between "
                f"{_bare(self.subject_category)} and "
                f"{_bare(self.object_category)} in this direction."
            )
        return " ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "hop_index": self.hop_index,
            "triple": self.triple,
            "supported": self.supported,
            "matched_as": self.matched_as,
            "reverse_direction_available": self.reversed_available,
            "available_predicates": self.alternatives[:20],
        }


# ---------------------------------------------------------------------------
# Meta knowledge graph
# ---------------------------------------------------------------------------


class MetaKnowledgeGraph:
    """The set of triples ARAX can answer, indexed for lookup."""

    def __init__(self, raw: Dict[str, Any], vocab: Optional[Any] = None):
        self.raw = raw
        self.vocab = vocab
        self.triples: Set[Tuple[str, str, str]] = set()
        self.by_pair: Dict[Tuple[str, str], Set[str]] = {}
        self.qualifiers: Dict[Tuple[str, str, str], Dict[str, Set[str]]] = {}
        self._index()

    # -- construction ------------------------------------------------------

    def _index(self) -> None:
        """Read the meta-KG into lookup tables.

        `edges` is a list in the TRAPI specification, but implementations have
        been seen returning a mapping, so both are accepted.
        """
        edges = self.raw.get("edges")
        if isinstance(edges, dict):
            edges = list(edges.values())
        for edge in edges or []:
            if not isinstance(edge, dict):
                continue
            subject = _bare(edge.get("subject"))
            predicate = _bare(edge.get("predicate"))
            obj = _bare(edge.get("object"))
            if not (subject and predicate and obj):
                continue

            self.triples.add((subject, predicate, obj))
            self.by_pair.setdefault((subject, obj), set()).add(predicate)

            quals: Dict[str, Set[str]] = {}
            for q in edge.get("qualifiers") or []:
                if not isinstance(q, dict):
                    continue
                qtype = _bare(q.get("qualifier_type_id"))
                values = q.get("applicable_values") or []
                if qtype:
                    quals.setdefault(qtype, set()).update(_bare(v) for v in values)
            if quals:
                self.qualifiers[(subject, predicate, obj)] = quals

    @classmethod
    def load(
        cls,
        client: Any,
        cache: Optional[QueryCache] = None,
        vocab: Optional[Any] = None,
        refresh: bool = False,
    ) -> Optional["MetaKnowledgeGraph"]:
        """Fetch the meta-KG, using the cache unless asked to refresh.

        Returns None when it cannot be retrieved. The caller decides what that
        means — a missing meta-KG is a reason to skip the check, never a reason
        to fail a plan, since the graph itself may be perfectly able to answer
        it.
        """
        endpoint = getattr(client, "url", "arax")
        request = {"endpoint": endpoint, "resource": "meta_knowledge_graph"}

        if cache and not refresh:
            hit = cache.get(KIND_META_KG, request)
            if hit is not None and hit.response:
                return cls(hit.response, vocab=vocab)

        raw = client.meta_kg()
        if not raw or raw.get("error") or not raw.get("edges"):
            return None

        if cache:
            edges = raw.get("edges")
            cache.put(
                KIND_META_KG, request, response=raw, status=STATUS_SUCCESS,
                num_results=len(edges) if isinstance(edges, (list, dict)) else 0,
                endpoint=endpoint,
            )
        return cls(raw, vocab=vocab)

    # -- category matching -------------------------------------------------

    def _category_variants(self, category: str) -> Set[str]:
        """Categories that should be considered the same for matching.

        ARAX expands a queried category to its Biolink descendants, so a hop
        asking for Gene is answered by edges declared on GeneOrGeneProduct. The
        check has to allow the same latitude or it would reject hops that in
        fact run.
        """
        bare = _bare(category)
        out = {bare}

        # Descendants only. A query for Disease is answered by edges on
        # Disease and its subclasses, because every such node is a Disease.
        # Walking upward instead would accept an edge declared on a parent,
        # whose nodes need not be Diseases at all — which is how a hop with no
        # data came to be reported as supported.
        if self.vocab is not None:
            for method in ("get_descendants", "descendants"):
                fn = getattr(self.vocab, method, None)
                if not callable(fn):
                    continue
                try:
                    out.update(_bare(c) for c in fn(_prefixed(bare)) or [])
                    break
                except Exception:
                    continue

        out.update(_FALLBACK_EQUIVALENTS.get(bare, set()))
        return {c for c in out if c and c not in _TOO_GENERIC}

    def _predicate_variants(self, predicate: str, expansion: str) -> Set[str]:
        """The predicate, plus its Biolink descendants when expansion allows."""
        bare = _bare(predicate)
        out = {bare}
        if expansion == "self_only" or self.vocab is None:
            return out
        for method in ("get_descendants", "descendants"):
            fn = getattr(self.vocab, method, None)
            if callable(fn):
                try:
                    out.update(_bare(p) for p in fn(_prefixed(bare)) or [])
                    break
                except Exception:
                    continue
        return out

    # -- checking ----------------------------------------------------------

    def check_hop(
        self,
        path_id: str,
        hop_index: int,
        subject_category: str,
        predicate: str,
        object_category: str,
        predicate_expansion: str = "descendants",
    ) -> HopCheck:
        """Check one hop, and describe the alternatives when it is unsupported."""
        check = HopCheck(
            path_id=path_id, hop_index=hop_index,
            subject_category=subject_category, predicate=predicate,
            object_category=object_category,
        )

        subjects = self._category_variants(subject_category)
        objects = self._category_variants(object_category)
        predicates = self._predicate_variants(predicate, predicate_expansion)

        for s in subjects:
            for o in objects:
                for p in predicates:
                    if (s, p, o) in self.triples:
                        check.supported = True
                        if (s, p, o) != (_bare(subject_category),
                                         _bare(predicate),
                                         _bare(object_category)):
                            check.matched_as = f"{s} --[{p}]--> {o}"
                        return check

        available: Set[str] = set()
        for s in subjects:
            for o in objects:
                available |= self.by_pair.get((s, o), set())
        check.alternatives = sorted(available)

        for s in subjects:
            for o in objects:
                for p in predicates:
                    if (o, p, s) in self.triples:
                        check.reversed_available = True
                        break

        return check

    def check_plan(self, plan_input: Any) -> List[HopCheck]:
        """Check every hop of every runnable path."""
        entities = plan_input.entities_by_ref
        out: List[HopCheck] = []

        for path in plan_input.active_paths:
            path_id = getattr(path, "path_id", "?")
            for i, hop in enumerate(getattr(path, "hops", []) or []):
                subject = entities.get(getattr(hop, "subject_ref", None))
                obj = entities.get(getattr(hop, "object_ref", None))
                if subject is None or obj is None:
                    # An uncheckable hop is recorded as such rather than
                    # skipped: passing over it silently would leave the plan
                    # looking fully checked when part of it was not.
                    check = HopCheck(
                        path_id=path_id, hop_index=i,
                        subject_category="?", predicate=getattr(hop, "predicate", "?"),
                        object_category="?", supported=True,
                        matched_as="not checked — hop references an undeclared entity",
                    )
                    out.append(check)
                    continue
                out.append(self.check_hop(
                    path_id=path_id,
                    hop_index=i,
                    subject_category=getattr(subject, "biolink_category", "") or "",
                    predicate=getattr(hop, "predicate", "") or "",
                    object_category=getattr(obj, "biolink_category", "") or "",
                    predicate_expansion=getattr(hop, "predicate_expansion", "descendants"),
                ))
        return out

    def summary(self) -> Dict[str, Any]:
        return {
            "triples": len(self.triples),
            "category_pairs": len(self.by_pair),
            "triples_with_qualifiers": len(self.qualifiers),
        }

    def predicates_between(self, subject_category: str, object_category: str) -> List[str]:
        """Every predicate ARAX supports between two categories, either way round."""
        subjects = self._category_variants(subject_category)
        objects = self._category_variants(object_category)
        out: Set[str] = set()
        for s in subjects:
            for o in objects:
                out |= self.by_pair.get((s, o), set())
        return sorted(out)
