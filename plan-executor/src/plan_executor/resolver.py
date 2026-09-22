"""
resolver.py — Turn plan entity names into CURIEs, with the LLM deciding.

This is where "idiopathic pulmonary fibrosis" becomes MONDO:0008345. Every
query the executor issues is anchored on the output of this module, so a wrong
choice here does not fail loudly — it silently returns confident results about
the wrong disease.

Why the LLM is required, not advisory
-------------------------------------
The Name Resolver's own ranking is not reliable enough to trust unattended:
its top hit is sometimes an odd match, and taking it on rank order alone can
anchor an entire run on the wrong concept with nothing in the output to
indicate it. Silently wrong is worse than loudly stopped, so when a choice has
to be made and the LLM cannot make it, the run halts.

Two failure modes are kept distinct, because they call for different actions:

  * The LLM is unreachable or malfunctioning — retried, then raises
    `LLMUnavailableError`. The run stops and the operator checks the LLM.
    Nothing is resolved by fallback.
  * The LLM answers "none of these candidates is correct" — a legitimate
    answer about the data, not a malfunction. That entity is recorded as
    unresolvable and reported to the planner. The LLM is fine and is not
    blamed.

The LLM only ever selects from candidates the Name Resolver returned, or
rejects them all. It cannot invent a CURIE. Every decision is recorded with
the full candidate list and the stated reason.

No normalization step
---------------------
ARAX runs its own NodeSynonymizer over incoming CURIEs, so a separate Node
Normalizer round trip would duplicate work ARAX already does. The chosen CURIE
is sent as-is. One consequence: cache keys depend on which of several
equivalent CURIEs was chosen, so an equivalent-but-different CURIE causes a
cache miss, never a wrong answer.

Conflation
----------
Biolink draws distinctions that are often immaterial to a query: a gene versus
its protein product, a drug versus the small molecule it contains. When the
plan signals conflation, candidates from the sibling category are accepted
rather than filtered out, and no category-mismatch warning is raised.

Usage
-----
    resolver = EntityResolver(cache=cache, disambiguator=llm_disambiguator)
    resolutions = resolver.resolve_plan(plan)   # raises if the LLM is down
    curies = resolver.curie_map(resolutions)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Set

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

from .cache import QueryCache, STATUS_SUCCESS, STATUS_EMPTY, STATUS_ERROR


# ---------------------------------------------------------------------------
# Endpoints
#
# ARAX's own NodeSynonymizer reports using name-resolution-sri.renci.org.
# Resolving through the same service keeps the CURIEs we pin aligned with the
# ones its knowledge graph is keyed on.
# ---------------------------------------------------------------------------

NAME_RESOLVER_URL = "https://name-resolution-sri.renci.org/lookup"

#: Raised from 10 because gene symbols are shared across species: a lookup
#: for KDR returns the mouse, dog, cow, chicken, zebrafish and Xenopus
#: orthologs, and ten slots can be exhausted before the human gene appears.
LOOKUP_LIMIT = 20
REQUEST_TIMEOUT = 30

#: Default species scope. Biomedical questions concern humans unless they say
#: otherwise, and without this filter the Name Resolver readily returns an
#: ortholog — which resolves cleanly, queries cleanly, and answers a question
#: about the wrong organism.
HUMAN_TAXON = "NCBITaxon:9606"

#: Categories where a taxon filter is meaningful. Genes and their products
#: exist per-organism and share symbols across species, which is what makes
#: the ortholog trap possible. Diseases, chemicals, phenotypes and anatomy
#: are not taxon-scoped in the identifier systems used here — a MONDO term
#: or a CHEBI molecule carries no taxon at all, so filtering one by taxon
#: risks excluding every valid answer rather than narrowing to the right
#: one.
TAXON_SCOPED_CATEGORIES = {
    "Gene", "Protein", "GeneOrGeneProduct", "GeneProductMixin",
    "Polypeptide", "GeneFamily", "Transcript", "RNAProduct",
    "MacromolecularComplex", "NucleicAcidEntity", "Genome",
    "OrganismTaxon", "GenomicEntity", "ProteinFamily", "ProteinDomain",
}

#: Entity constraint fields a planner may use to scope one entity to a
#: species, overriding the run-wide default.
_TAXON_CONSTRAINT_FIELDS = {"taxon", "in_taxon", "species", "organism"}


def is_taxon_scoped(category: str) -> bool:
    """True when a taxon filter can sensibly narrow this category."""
    return (category or "").replace("biolink:", "") in TAXON_SCOPED_CATEGORIES


def taxon_from_entity(entity) -> Optional[List[str]]:
    """Read a species scope from an entity.

    `taxa` is the contract's field. Entity constraints are still read as a
    fallback, since that was the only way to express species before the field
    existed.
    """
    taxa = (
        entity.get("taxa") if hasattr(entity, "get")
        else getattr(entity, "taxa", None)
    )
    if taxa:
        return [str(t) for t in (taxa if isinstance(taxa, (list, tuple)) else [taxa])]
    return _taxon_from_constraints(entity)


def _taxon_from_constraints(entity) -> Optional[List[str]]:
    """Read a species scope from an entity's constraints, if it states one.

    The plan schema has no dedicated species field, but `Entity.constraints`
    can carry one, which lets a plan about a model organism scope a single
    entity without changing the default for the whole run.
    """
    out: List[str] = []
    for c in getattr(entity, "constraints", None) or []:
        field_name = (
            c.get("field") if isinstance(c, dict) else getattr(c, "field", None)
        )
        if (field_name or "").lower() not in _TAXON_CONSTRAINT_FIELDS:
            continue
        value = c.get("value") if isinstance(c, dict) else getattr(c, "value", None)
        for v in (value if isinstance(value, (list, tuple)) else [value]):
            if v:
                out.append(str(v))
    return out or None

#: Attempts after the first before giving up on the LLM.
DEFAULT_LLM_RETRIES = 2
DEFAULT_LLM_BACKOFF_S = 2.0


# ---------------------------------------------------------------------------
# Conflation
# ---------------------------------------------------------------------------

#: Category groups treated as interchangeable when the plan asks for it. Bare
#: names, matching how the plan writes categories.
CONFLATION_GROUPS: Dict[str, Set[str]] = {
    "gene_protein": {
        "Gene", "Protein", "GeneOrGeneProduct", "GeneProductMixin",
        "Polypeptide", "GeneFamily",
    },
    "drug_chemical": {
        "Drug", "SmallMolecule", "ChemicalEntity", "MolecularMixture",
        "ChemicalMixture", "ChemicalOrDrugOrTreatment", "MolecularEntity",
    },
}


#: Fields a plan may use to state identifiers directly. Several are accepted
#: because the schema has no such field yet and different plans spell it
#: differently.
_CURIE_FIELDS = ("curies", "curie", "ids", "id")


#: Direction words the schema treats as the same claim. A signature may
#: describe a gene as increased or upregulated in the disease state; both mean
#: the same thing and a plan may use either.
_DIRECTION_SYNONYMS = {
    "increased": {"increased", "upregulated", "up"},
    "upregulated": {"increased", "upregulated", "up"},
    "decreased": {"decreased", "downregulated", "down"},
    "downregulated": {"decreased", "downregulated", "down"},
}


class DirectionalInputError(RuntimeError):
    """An entity asks for one direction of a signature that carries none.

    Fatal rather than falling back to the whole set, because two entities in a
    signature-reversal plan reference the same input and differ only by
    direction. Handing both the full list would make the up-regulated and
    down-regulated arms identical, and a reversal query built from that says
    nothing while looking like it ran.
    """

    def __init__(self, entity_ref: str, input_ref: str, direction: str):
        super().__init__(
            f"entity '{entity_ref}' asks for the '{direction}' rows of input "
            f"'{input_ref}', but the supplied value carries no direction. "
            f"Provide it as {{'increased': [...], 'decreased': [...]}} or as "
            f"rows of {{'id': ..., 'direction': ...}} — an undirected list "
            f"would give every direction the same genes."
        )


def _filter_by_direction(
    value: Any,
    direction: Optional[str],
    entity_ref: str,
    input_ref: str,
) -> List[str]:
    """Select the rows of a directional signature matching `direction`.

    The direction is the one observed in the disease state, not the effect
    wanted from a drug: a reversal path pairs an increased-in-disease gene
    with a drug edge that decreases it.
    """
    if direction is None:
        if isinstance(value, dict):
            # No direction asked for, so every row qualifies.
            return [str(v).strip() for group in value.values()
                    for v in (group or []) if v]
        rows = value if isinstance(value, (list, tuple)) else [value]
        return [
            str(r.get("id") if isinstance(r, dict) else r).strip()
            for r in rows if r
        ]

    wanted = _DIRECTION_SYNONYMS.get(direction.lower(), {direction.lower()})

    if isinstance(value, dict):
        out: List[str] = []
        for key, group in value.items():
            if str(key).lower() in wanted:
                out.extend(str(v).strip() for v in (group or []) if v)
        if out:
            return out
        raise DirectionalInputError(entity_ref, input_ref, direction)

    rows = value if isinstance(value, (list, tuple)) else [value]
    if not any(isinstance(r, dict) and "direction" in r for r in rows):
        raise DirectionalInputError(entity_ref, input_ref, direction)

    return [
        str(r.get("id") or r.get("curie") or "").strip()
        for r in rows
        if isinstance(r, dict)
        and str(r.get("direction", "")).lower() in wanted
        and (r.get("id") or r.get("curie"))
    ]


class MissingExternalInput(RuntimeError):
    """An entity references an external input the caller did not supply.

    Raised rather than resolved by other means: the plan states that this data
    comes from outside the knowledge graph — a signature, a variant list, a
    screening panel — so substituting a name lookup would answer a different
    question. The contract is explicit that the LLM must never be asked to
    recreate it.
    """

    def __init__(self, entity_ref: str, input_ref: str, expected_format: str):
        self.entity_ref = entity_ref
        self.input_ref = input_ref
        self.expected_format = expected_format
        super().__init__(
            f"entity '{entity_ref}' requires external input '{input_ref}' "
            f"({expected_format}), which was not supplied. Provide it through "
            f"the input registry; it cannot be looked up or inferred."
        )


def _supplied_curies(
    entity,
    input_registry: Optional[Dict[str, Any]] = None,
    entity_ref: str = "?",
) -> List[str]:
    """Read identifiers an entity supplies directly or through an input binding.

    `input_binding` is the contract's form; the bare fields are accepted too
    for plans written before it existed.
    """
    binding = (
        entity.get("input_binding") if hasattr(entity, "get")
        else getattr(entity, "input_binding", None)
    )
    if binding is not None:
        get = (lambda k: binding.get(k)) if hasattr(binding, "get") else (
            lambda k: getattr(binding, k, None)
        )
        kind = get("binding_type")

        if kind == "identifiers":
            return [str(v).strip() for v in (get("identifiers") or []) if v]

        if kind == "external_input":
            input_ref = get("input_ref")
            expected = get("expected_format") or "curie_list"
            supplied = (input_registry or {}).get(input_ref)
            if supplied is None:
                raise MissingExternalInput(entity_ref, input_ref, expected)
            # Several entities may share one input and differ only by which
            # direction of it they want, so the filter is applied per entity
            # rather than to the registry entry.
            return [
                v for v in _filter_by_direction(
                    supplied, get("direction_filter"), entity_ref, input_ref,
                ) if v
            ]

    out: List[str] = []
    for field_name in _CURIE_FIELDS:
        value = (
            entity.get(field_name) if hasattr(entity, "get")
            else getattr(entity, field_name, None)
        )
        if not value:
            continue
        for v in (value if isinstance(value, (list, tuple)) else [value]):
            v = str(v).strip()
            # A CURIE has a prefix; a bare word here is a name in the wrong
            # field and would silently become an unqueryable identifier.
            if v and ":" in v and v not in out:
                out.append(v)
        if out:
            break
    return out


def acceptable_categories(category: str, conflation: Sequence[str] = ()) -> Set[str]:
    """Categories that satisfy a request for `category` under `conflation`.

    With no conflation this is just the category itself; with `gene_protein`
    active, a request for Gene is also satisfied by Protein.
    """
    bare = (category or "").replace("biolink:", "")
    out = {bare} if bare else set()
    for flag in conflation or ():
        group = CONFLATION_GROUPS.get(flag)
        if group and bare in group:
            out |= group
    return out


def conflation_from_plan(plan, override: Optional[Sequence[str]] = None) -> List[str]:
    """Read conflation flags from a plan.

    Accepts several shapes because the planner's field is still settling: a
    list of flag names, or a mapping of flag -> bool. An explicit override
    always wins.

    NOTE: as of this writing no plan can reach any of these three places.
    plan-core's schema sets `additionalProperties: false` at the top level
    and inside `interpretation`, and declares no `conflation` field in
    either, so a plan carrying one fails validation before it gets here.
    In practice this function returns the override or nothing. Per-entity
    defaults are applied separately, in `EntityResolver.DEFAULT_CONFLATION`;
    do not assume a plan can turn conflation on until the schema says so.
    """
    if override is not None:
        return list(override)

    raw = getattr(plan, "raw", None) or {}
    value = (
        getattr(plan, "conflation", None)
        or raw.get("conflation")
        or (raw.get("interpretation") or {}).get("conflation")
    )
    if not value:
        return []
    if isinstance(value, dict):
        return [k for k, v in value.items() if v]
    if isinstance(value, str):
        return [value]
    return list(value)


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class Candidate:
    """One option returned by the Name Resolver."""

    curie: str
    label: str
    types: List[str] = field(default_factory=list)
    taxa: List[str] = field(default_factory=list)
    synonyms: List[str] = field(default_factory=list)
    rank: int = 0

    @property
    def bare_types(self) -> List[str]:
        return [t.replace("biolink:", "") for t in self.types]

    def matches(self, acceptable: Set[str]) -> bool:
        return bool(acceptable) and bool(acceptable & set(self.bare_types))

    def to_dict(self) -> Dict[str, Any]:
        return {"curie": self.curie, "label": self.label,
                "types": self.bare_types[:4], "rank": self.rank,
                "synonyms": self.synonyms[:3]}


@dataclass
class DisambiguationChoice:
    """An LLM's answer.

    `index` of None means "none of these candidates is correct" — a real
    answer about the data, distinct from the LLM failing to answer at all.
    """

    index: Optional[int]
    reason: str
    confidence: Optional[str] = None


@dataclass
class Resolution:
    """The outcome of resolving one plan entity, with its decision trail."""

    entity_ref: str
    query: str
    expected_category: str
    is_variable: bool = False

    curies: List[str] = field(default_factory=list)
    label: Optional[str] = None
    categories: List[str] = field(default_factory=list)

    candidates: List[Candidate] = field(default_factory=list)
    method: str = "unresolved"  # single | llm | variable | rejected | failed
    reason: str = ""
    confidence: Optional[str] = None
    llm_used: bool = False
    llm_attempts: int = 0
    alias_used: Optional[str] = None
    conflation: List[str] = field(default_factory=list)

    #: Every category that was allowed to satisfy `expected_category` on this
    #: entity, conflation included. Stated rather than left to be recomputed:
    #: a reader that re-derives it needs both the flags and the group table,
    #: and the loop controller has neither. Without this field the controller
    #: compares a resolved gene against a plan that said Protein, calls it a
    #: mismatch, and repairs a plan that was answered correctly — which is
    #: exactly what it did. Empty for a variable or plan-supplied entity,
    #: where nothing was accepted because nothing was looked up.
    accepted_categories: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    elapsed_s: float = 0.0

    @property
    def resolved(self) -> bool:
        return bool(self.curies)

    @property
    def primary(self) -> Optional[str]:
        return self.curies[0] if self.curies else None

    def to_dict(self) -> Dict[str, Any]:
        """Ledger form: what was chosen, and what it was chosen over."""
        return {
            "entity_ref": self.entity_ref,
            "query": self.query,
            "expected_category": self.expected_category,
            "is_variable": self.is_variable,
            "resolved_curies": self.curies,
            "resolved_label": self.label,
            "resolved_categories": self.categories,
            "method": self.method,
            "reason": self.reason,
            "confidence": self.confidence,
            "llm_used": self.llm_used,
            "llm_attempts": self.llm_attempts,
            "alias_used": self.alias_used,
            "conflation": self.conflation,
            "accepted_categories": self.accepted_categories,
            "considered": [c.to_dict() for c in self.candidates],
            "warnings": self.warnings,
            "elapsed_s": round(self.elapsed_s, 2),
        }

    def __repr__(self) -> str:
        if self.is_variable:
            return f"<Resolution {self.entity_ref} VARIABLE ({self.expected_category})>"
        if not self.resolved:
            return f"<Resolution {self.entity_ref} UNRESOLVED ({self.method})>"
        return (f"<Resolution {self.entity_ref} -> {self.primary} "
                f"({self.label}) via {self.method}>")


class LLMUnavailableError(RuntimeError):
    """The LLM could not be consulted, so no choice can be trusted.

    Raised only for malfunction — never when the LLM successfully answers
    "none of these". The run stops here rather than falling back to rank
    order, because an unverified anchor silently corrupts every result
    downstream.
    """

    def __init__(self, entity_ref: str, attempts: int, last_error: str):
        self.entity_ref = entity_ref
        self.attempts = attempts
        self.last_error = last_error
        super().__init__(
            f"LLM disambiguation failed for entity '{entity_ref}' after "
            f"{attempts} attempt(s). Last error: {last_error}\n"
            f"Resolution requires the LLM — the Name Resolver's ranking is not "
            f"reliable enough to use unattended. Check that the model is "
            f"running and reachable, then re-run. Cached lookups mean the "
            f"re-run will not repeat completed work."
        )


class Disambiguator(Protocol):
    """Chooses among candidates, or rejects them all. Implemented by llm.py."""

    def choose(
        self,
        question: str,
        entity_name: str,
        expected_category: str,
        candidates: Sequence[Candidate],
        aliases: Sequence[str] = (),
        conflation: Sequence[str] = (),
    ) -> DisambiguationChoice:
        ...


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


class EntityResolver:
    """Resolves plan entities to CURIEs, with the LLM making every real choice.

    Args:
        disambiguator: required. Without it, any choice raises
            `LLMUnavailableError`.
        confirm_single: ask the LLM to confirm even a lone candidate. Defaults
            to True: one odd result from the Name Resolver is still odd, and
            confirmation turns a silent wrong anchor into a clear rejection.
        llm_retries: retries before raising `LLMUnavailableError`.
        conflation: override the plan's conflation flags.
        default_conflation: apply `DEFAULT_CONFLATION` to entities whose
            category falls in one of those groups. Defaults to True. Set
            False to reproduce a run made before this default existed.
    """

    #: Conflation groups applied automatically, per entity, when that
    #: entity's own category belongs to the group.
    #:
    #: A plan cannot ask for conflation: `conflation_from_plan` reads three
    #: places on the plan and the schema forbids all of them, so the flags
    #: were only ever reachable from the command line. Without a default, a
    #: category of Gene pins the lookup to genes and a category of Protein
    #: pins it to proteins, and the LLM is then told to reject anything of
    #: the other form. The graph does not draw that line — the node
    #: normalizer treats a gene and its product as one concept, and target
    #: edges are indexed against genes far more often than proteins — so the
    #: pin loses real answers. `drug_chemical` is deliberately not defaulted
    #: on; it has not been shown to be needed, and one behaviour change at a
    #: time is testable.
    DEFAULT_CONFLATION: Sequence[str] = ("gene_protein",)

    def __init__(
        self,
        cache: Optional[QueryCache] = None,
        disambiguator: Optional[Disambiguator] = None,
        name_resolver_url: str = NAME_RESOLVER_URL,
        confirm_single: bool = True,
        only_taxa: Optional[Sequence[str]] = (HUMAN_TAXON,),
        llm_retries: int = DEFAULT_LLM_RETRIES,
        llm_backoff_s: float = DEFAULT_LLM_BACKOFF_S,
        conflation: Optional[Sequence[str]] = None,
        default_conflation: bool = True,
        input_registry: Optional[Dict[str, Any]] = None,
        mock_lookup: Optional[Callable] = None,
        verbose: bool = True,
    ):
        if requests is None and mock_lookup is None:
            raise ImportError("resolver needs `requests` unless mock_lookup is supplied")
        self.cache = cache
        self.disambiguator = disambiguator
        self.name_resolver_url = name_resolver_url
        self.confirm_single = confirm_single
        self.only_taxa = list(only_taxa) if only_taxa else []
        self.llm_retries = llm_retries
        self.llm_backoff_s = llm_backoff_s
        self.conflation_override = conflation
        self.default_conflation = default_conflation
        #: input_ref -> identifiers, supplied by the caller for entities bound
        #: to external inputs.
        self.input_registry = dict(input_registry or {})
        self.mock_lookup = mock_lookup
        self.verbose = verbose
        self.lookup_calls = 0
        self.llm_calls = 0

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [resolve] {msg}")

    # -- name lookup -------------------------------------------------------

    def name_lookup(
        self,
        query: str,
        biolink_type: Optional[str] = None,
        limit: int = LOOKUP_LIMIT,
        only_taxa: Optional[Sequence[str]] = None,
    ) -> List[Candidate]:
        """Look up a name, returning ranked candidates.

        Rank is retained for the record but never used to decide.
        """
        taxa = list(only_taxa) if only_taxa is not None else self.only_taxa
        # Only narrow categories where taxon means something. Sending a
        # taxon with a disease or chemical lookup can exclude every valid
        # answer, since those identifiers carry no taxon to match.
        if taxa and biolink_type and not is_taxon_scoped(biolink_type):
            taxa = []
        # Taxa are folded into the cache key: the same name filtered to human
        # and filtered to nothing are different questions with different
        # answers.
        cache_type = f"{biolink_type}|taxa={','.join(taxa)}" if taxa else biolink_type
        if self.cache:
            hit = self.cache.get_name_lookup(query, biolink_type=cache_type)
            if hit is not None and hit.response is not None:
                return [Candidate(**c) for c in hit.response.get("candidates", [])]

        if self.mock_lookup is not None:
            raw = self.mock_lookup(query, biolink_type)
        else:
            params = {"string": query, "autocomplete": "false", "limit": limit}
            if biolink_type:
                params["biolink_type"] = biolink_type
            if taxa:
                params["only_taxa"] = "|".join(taxa)
            self.log(f"lookup {params}")
            try:
                self.lookup_calls += 1
                resp = requests.get(self.name_resolver_url, params=params,
                                    timeout=REQUEST_TIMEOUT)
                resp.raise_for_status()
                raw = resp.json()
            except Exception as e:
                self.log(f"name_lookup('{query}') failed: {e}")
                if self.cache:
                    self.cache.put_name_lookup(
                        query, {"candidates": [], "error": str(e)},
                        biolink_type=cache_type, status=STATUS_ERROR,
                        error_message=str(e))
                return []

        candidates = [
            Candidate(
                curie=r.get("curie", ""),
                label=r.get("label", ""),
                types=r.get("types", []) or [],
                taxa=r.get("taxa", []) or [],
                synonyms=(r.get("synonyms") or [])[:5],
                rank=i,
            )
            for i, r in enumerate(raw or [])
        ]

        # The service may ignore only_taxa, or not support it under this
        # name. Each record carries its own taxa, so the filter is applied
        # again locally: server-side narrowing is an optimisation, this is
        # the guarantee. Records with no taxa at all are kept, since an
        # unannotated record is unknown rather than wrong.
        if taxa:
            wanted = set(taxa)
            matching = [
                c for c in candidates
                if not c.taxa or (set(c.taxa) & wanted)
            ]
            if matching and len(matching) < len(candidates):
                self.log(
                    f"taxon filter kept {len(matching)} of "
                    f"{len(candidates)} candidate(s) for '{query}'"
                )
            elif not matching:
                self.log(
                    f"no candidate for '{query}' is in {taxa}; the service "
                    f"returned {len(candidates)} record(s), all other species"
                )
            candidates = matching
            for i, c in enumerate(candidates):
                c.rank = i

        if self.cache:
            self.cache.put_name_lookup(
                query, {"candidates": [c.__dict__ for c in candidates]},
                biolink_type=cache_type,
                status=STATUS_SUCCESS if candidates else STATUS_EMPTY,
                num_results=len(candidates),
            )
        return candidates

    # -- disambiguation ----------------------------------------------------

    def _ask_llm(
        self,
        entity_ref: str,
        question: str,
        entity,
        pool: List[Candidate],
        conflation: Sequence[str],
    ) -> tuple:
        """Consult the LLM, retrying transient failures.

        Returns (choice, attempts).

        Raises:
            LLMUnavailableError: after retries are exhausted. Deliberately not
                caught below — a failure to verify the anchor must stop the run
                rather than degrade it.
        """
        if self.disambiguator is None:
            raise LLMUnavailableError(
                entity_ref, 0,
                "no disambiguator configured; resolution requires an LLM",
            )

        last_error = "unknown"
        attempts = 0
        for attempt in range(self.llm_retries + 1):
            attempts += 1
            try:
                self.llm_calls += 1
                choice = self.disambiguator.choose(
                    question=question,
                    entity_name=getattr(entity, "name", ""),
                    expected_category=getattr(entity, "biolink_category", "") or "",
                    candidates=pool,
                    aliases=getattr(entity, "aliases", None) or [],
                    conflation=list(conflation),
                )
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                self.log(f"LLM attempt {attempts} failed: {last_error}")
                if attempt < self.llm_retries:
                    time.sleep(self.llm_backoff_s * (attempt + 1))
                continue

            if choice is None:
                last_error = "LLM returned no choice"
                self.log(f"LLM attempt {attempts}: no choice returned")
                if attempt < self.llm_retries:
                    time.sleep(self.llm_backoff_s * (attempt + 1))
                continue

            # An explicit rejection is a valid answer about the candidates,
            # not a malfunction, so it is returned rather than retried.
            if choice.index is None:
                return choice, attempts

            if not (0 <= choice.index < len(pool)):
                last_error = (
                    f"index {choice.index} out of range for {len(pool)} candidates"
                )
                self.log(f"LLM attempt {attempts}: {last_error}")
                if attempt < self.llm_retries:
                    time.sleep(self.llm_backoff_s * (attempt + 1))
                continue

            return choice, attempts

        raise LLMUnavailableError(entity_ref, attempts, last_error)

    # -- entity resolution -------------------------------------------------

    def _effective_conflation(
        self, category: str, conflation: Sequence[str],
    ) -> List[str]:
        """Conflation flags for one entity: the plan's, plus any default.

        Per-entity rather than per-plan, and only for an entity whose own
        category is in the group: a Disease or ChemicalEntity anchor in the
        same plan is left exactly as it was.
        """
        out = list(conflation)
        if not self.default_conflation:
            return out
        bare = (category or "").replace("biolink:", "")
        if not bare:
            return out
        for flag in self.DEFAULT_CONFLATION:
            group = CONFLATION_GROUPS.get(flag) or set()
            if bare in group and flag not in out:
                out.append(flag)
        return out

    def resolve_entity(
        self,
        entity,
        question: str = "",
        conflation: Sequence[str] = (),
    ) -> Resolution:
        """Resolve one plan entity.

        Raises:
            LLMUnavailableError: if a choice is needed and the LLM cannot make
                it. Callers should let this propagate.
        """
        start = time.time()
        ref = getattr(entity, "entity_ref", "?")
        name = getattr(entity, "name", "")
        category = getattr(entity, "biolink_category", "") or ""
        is_variable = getattr(entity, "is_variable", False)

        res = Resolution(entity_ref=ref, query=name, expected_category=category,
                         is_variable=is_variable, conflation=list(conflation))

        # Variable entities describe a slot filled by query results, not a
        # concept to look up.
        if is_variable:
            res.method = "variable"
            res.reason = "variable entity; left open by category"
            res.categories = [category] if category else []
            res.elapsed_s = time.time() - start
            return res

        # A plan may state identifiers outright. Some entity sets cannot be
        # reached by name at all — an expression signature, a screening panel,
        # or a concept the resolver does not index under the name in use — and
        # a stated CURIE is a deliberate act by the plan's author rather than
        # a guess to be adjudicated, so no lookup or disambiguation runs.
        supplied = _supplied_curies(
            entity, input_registry=self.input_registry, entity_ref=ref,
        )
        if supplied:
            res.curies = list(supplied)
            res.label = getattr(entity, "name", None) or supplied[0]
            res.categories = [category] if category else []
            res.method = "supplied"
            res.reason = (
                f"{len(supplied)} CURIE(s) given in the plan; no lookup "
                f"performed"
            )
            res.elapsed_s = time.time() - start
            return res

        # Defaults are applied here, after the early returns: a variable
        # entity is never looked up, and a supplied CURIE is the plan
        # author's decision, so neither should carry a flag that had no
        # effect on it. The ledger's `conflation` field records what
        # actually applied to this entity, default included.
        conflation = self._effective_conflation(category, conflation)
        if list(conflation) != list(res.conflation):
            self.log(
                f"{ref}: conflation {list(conflation)} "
                f"(category {category or 'unspecified'})"
            )
        res.conflation = list(conflation)

        acceptable = acceptable_categories(category, conflation)
        res.accepted_categories = sorted(acceptable)

        # Under conflation a server-side type filter would exclude valid
        # siblings, so filtering happens locally instead.
        biolink_type = None
        if category and len(acceptable) <= 1:
            biolink_type = (
                category if category.startswith("biolink:") else f"biolink:{category}"
            )

        entity_taxa = taxon_from_entity(entity)
        if entity_taxa:
            res.warnings.append(
                f"entity constrains taxon to {entity_taxa}, overriding the "
                f"run default {self.only_taxa or 'any'}"
            )
        candidates = self.name_lookup(
            name, biolink_type=biolink_type, only_taxa=entity_taxa,
        )

        # A plan about a model organism would find nothing under the human
        # filter. Retrying without it keeps such plans workable, but the
        # species scope of the result is then unverified, so it is recorded
        # rather than passed off as an ordinary resolution.
        taxa_applied = bool(entity_taxa or self.only_taxa) and is_taxon_scoped(
            biolink_type or category
        )
        if not candidates and taxa_applied:
            widened = self.name_lookup(name, biolink_type=biolink_type, only_taxa=[])
            if widened:
                candidates = widened
                res.warnings.append(
                    f"no match for '{name}' in {entity_taxa or self.only_taxa}; the candidates below are from other species and the LLM must confirm the "
                    f"species is right for this question"
                )

        # An alias often succeeds where the formal name fails: "IPF" and
        # "idiopathic pulmonary fibrosis" are indexed differently.
        if not candidates:
            for alias in (getattr(entity, "aliases", None) or []):
                candidates = self.name_lookup(alias, biolink_type=biolink_type)
                if candidates:
                    res.alias_used = alias
                    res.warnings.append(
                        f"name '{name}' returned nothing; resolved via alias '{alias}'"
                    )
                    break

        # Retrying unfiltered separates "this concept does not exist" from
        # "the plan gave it the wrong category" — only the planner can fix the
        # second, and it needs to be told which one happened.
        if not candidates and biolink_type:
            unfiltered = self.name_lookup(name, biolink_type=None)
            if unfiltered:
                found = sorted({t for c in unfiltered[:3] for t in c.bare_types})
                res.warnings.append(
                    f"no {category} matches for '{name}', but the name resolves "
                    f"to other categories: {found[:6]}. The plan's category may "
                    f"be wrong."
                )
                res.candidates = unfiltered

        if not candidates:
            res.method = "failed"
            res.reason = f"no candidates for '{name}'" + (
                f" as {category}" if category else ""
            )
            res.elapsed_s = time.time() - start
            return res

        # Local category narrowing, conflation-aware. If it would leave
        # nothing, the unnarrowed list is passed to the LLM instead: an empty
        # pool means the plan's category disagrees with every candidate, and
        # the LLM should see that rather than have it hidden.
        pool = [c for c in candidates if c.matches(acceptable)] if acceptable else candidates
        if not pool:
            pool = candidates
            res.warnings.append(
                f"no candidate is typed as {category}"
                + (f" (conflation: {list(conflation)})" if conflation else "")
            )
        res.candidates = pool

        if len(pool) == 1 and not self.confirm_single:
            chosen = pool[0]
            res.method = "single"
            res.reason = "only candidate returned; LLM confirmation disabled"
        else:
            choice, attempts = self._ask_llm(ref, question, entity, pool, conflation)
            res.llm_used = True
            res.llm_attempts = attempts
            res.confidence = choice.confidence

            if choice.index is None:
                # Ten unusable candidates is functionally the same as none,
                # and the aliases exist for exactly this case. The primary
                # name returning wrong results never triggered the alias
                # path before, because that only fired on an empty list.
                aliases = [
                    a for a in (getattr(entity, "aliases", None) or [])
                    if a and a != res.alias_used
                ]
                for alias in aliases:
                    alt = self.name_lookup(
                        alias, biolink_type=biolink_type, only_taxa=entity_taxa,
                    )
                    new = [c for c in alt if c.curie not in
                           {p.curie for p in pool}]
                    if not new:
                        continue
                    self.log(
                        f"all candidates for '{name}' were rejected; "
                        f"retrying with alias '{alias}'"
                    )
                    alt_choice, alt_attempts = self._ask_llm(
                        ref, question, entity, alt, conflation,
                    )
                    res.llm_attempts += alt_attempts
                    if alt_choice.index is not None:
                        pool = alt
                        res.candidates = alt
                        res.alias_used = alias
                        res.warnings.append(
                            f"candidates for '{name}' were all rejected; "
                            f"resolved via alias '{alias}' instead"
                        )
                        choice = alt_choice
                        break

            if choice.index is None:
                res.method = "rejected"
                res.reason = f"LLM rejected all {len(pool)} candidates: {choice.reason}"
                res.warnings.append(
                    "no candidate matched the intended concept; the planner may "
                    "need a different name, alias, or category"
                )
                res.elapsed_s = time.time() - start
                return res

            chosen = pool[choice.index]
            res.method = "llm"
            res.reason = f"chose {choice.index + 1} of {len(pool)}: {choice.reason}"

        res.curies = [chosen.curie]
        res.label = chosen.label
        res.categories = chosen.types

        # Surfaced, never corrected: the plan decides intent, the executor
        # reports what it observed.
        if acceptable and not chosen.matches(acceptable):
            res.warnings.append(
                f"chosen CURIE {chosen.curie} is typed {chosen.bare_types[:4]}, "
                f"which does not include {category}"
            )

        res.elapsed_s = time.time() - start
        return res

    def resolve_plan(
        self,
        plan,
        entities: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Resolution]:
        """Resolve every entity in a plan.

        Raises:
            LLMUnavailableError: on the first entity whose choice cannot be
                made. Stopping early is deliberate — later entities would meet
                the same broken LLM, and a partial resolution is not usable.
        """
        if entities is None:
            entities = getattr(plan, "entities", {}) or {}
            if isinstance(entities, list):
                entities = {getattr(e, "entity_ref", None): e for e in entities}

        question = getattr(plan, "question", "") or ""
        conflation = conflation_from_plan(plan, self.conflation_override)
        if conflation:
            self.log(f"conflation active: {conflation}")

        out: Dict[str, Resolution] = {}
        for ref, entity in entities.items():
            res = self.resolve_entity(entity, question=question, conflation=conflation)
            out[ref] = res
            self.log(str(res))
            for w in res.warnings:
                self.log(f"    warning: {w}")
        return out

    # -- reporting ---------------------------------------------------------

    @staticmethod
    def curie_map(resolutions: Dict[str, Resolution]) -> Dict[str, List[str]]:
        """Reduce resolutions to the {entity_ref: [curie]} map the builder wants."""
        return {ref: r.curies for ref, r in resolutions.items() if r.curies}

    @staticmethod
    def unresolved(resolutions: Dict[str, Resolution]) -> List[str]:
        """Non-variable entities that failed to resolve.

        Any of these makes the plan unexecutable: without a CURIE there is
        nothing to pin, and pivot-first decomposition has no pivot.
        """
        return [ref for ref, r in resolutions.items()
                if not r.is_variable and not r.resolved]

    def stats(self) -> Dict[str, Any]:
        return {"name_lookup_calls": self.lookup_calls, "llm_calls": self.llm_calls}
