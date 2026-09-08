"""
config.py — Endpoints, limits, and scoring tables.

Two kinds of weight are kept apart on purpose:

  * EPC weights (here) express how good a single edge's evidence is. They are
    the executor's convention and apply to every plan.
  * `plan.ranking.criteria` weights express how to order candidates
    (num_supporting_paths, path_length, ...). Those are the planner's
    decision and arrive with the plan.

Mixing them would let executor conventions silently override what a plan
asked for.
"""

from __future__ import annotations

from math import log10
from typing import Dict, Set


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

ARAX_BASE = "https://arax.ncats.io/api/arax/v1.4"
ARAX_QUERY_URL = f"{ARAX_BASE}/query"
ARAX_ENTITY_URL = f"{ARAX_BASE}/entity"
ARAX_META_KG_URL = f"{ARAX_BASE}/meta_knowledge_graph"

NAME_RESOLVER_URL = "https://name-resolution-sri.renci.org/lookup"
PUBLICATION_URL = "https://docmetadata.transltr.io/publications"

REQUEST_TIMEOUT = 120        # single synchronous ARAX query
DSL_TIMEOUT = 300            # ARAXi workflows, including connect()
RESOLVER_TIMEOUT = 30
PUBLICATION_TIMEOUT = 20     # docmetadata's own default of 4s is too tight
                             # for the batch sizes used here


# ---------------------------------------------------------------------------
# Execution limits
# ---------------------------------------------------------------------------

DEFAULT_MAX_RESULTS = 5000   # overrides ARAX's silent 500-result ceiling
DEFAULT_BATCH_SIZE = 500     # intermediates pinned per decomposed query
PUBLICATION_BATCH_SIZE = 50

LLM_RETRIES = 2
LLM_BACKOFF_S = 2.0


# ---------------------------------------------------------------------------
# EPC scoring
#
# Component weights sum to 1.0, so an edge's score lands in [0, 1].
# ---------------------------------------------------------------------------

EPC_COMPONENT_WEIGHTS: Dict[str, float] = {
    "knowledge_level": 0.30,
    "agent_type": 0.20,
    "publications": 0.20,
    "source_count": 0.15,
    "primary_source": 0.15,
}

#: TRAPI knowledge levels. `not_provided` sits at the midpoint rather than at
#: zero: an edge that declines to state its knowledge level is of unknown
#: quality, not of poor quality, and a great many real edges omit it.
KNOWLEDGE_LEVEL_SCORE: Dict[str, float] = {
    "knowledge_assertion": 1.00,
    "logical_entailment": 0.90,
    "observation": 0.70,
    "statistical_association": 0.60,
    "prediction": 0.30,
    "not_provided": 0.50,
}

AGENT_TYPE_SCORE: Dict[str, float] = {
    "manual_agent": 1.00,
    "manual_validation_of_automated_agent": 0.90,
    "data_analysis_pipeline": 0.70,
    "computational_model": 0.50,
    "automated_agent": 0.50,
    "text_mining_agent": 0.40,
    "image_processing_agent": 0.40,
    "not_provided": 0.50,
}

#: Zero publications scores 0.5, not 0. Expert-curated databases routinely
#: assert facts without attaching a PMID, and those assertions are often the
#: strongest evidence available for drug-target relationships.
PUBLICATION_FLOOR = 0.50


def publication_score(n: int) -> float:
    """Diminishing returns on publication count: 0->0.50, 1->0.65, 10->1.00."""
    if n <= 0:
        return PUBLICATION_FLOOR
    return min(1.0, PUBLICATION_FLOOR + (1 - PUBLICATION_FLOOR) * log10(1 + n) / log10(11))


#: Independent sources corroborating the same claim.
SOURCE_COUNT_SCORE: Dict[int, float] = {0: 0.40, 1: 0.60, 2: 0.80}
SOURCE_COUNT_MAX = 1.00     # 3 or more


PRIMARY_SOURCE_DEFAULT = 0.70
PRIMARY_SOURCE_CURATED = 0.90
PRIMARY_SOURCE_TEXT_MINED = 0.50

#: Matched against TRAPI `infores:` identifiers via `source_matches` below,
#: never by raw substring — see the note there.
CURATED_SOURCES: Set[str] = {
    "drugcentral", "drugbank", "chembl", "ctd", "disgenet", "hmdb", "kegg",
    "reactome", "go", "gocam", "uniprot", "omim", "orphanet", "hpo", "mondo",
    "clinicaltrials", "dgidb", "pharmgkb", "gtopdb", "sider", "panther",
    "intact", "biogrid", "string-db", "monarchinitiative",
}

TEXT_MINED_SOURCES: Set[str] = {
    "semmeddb", "text-mining-provider", "text-mining-provider-targeted",
    "text-mining-provider-cooccurrence", "pubtator", "chemotext",
}


def _normalize_source(source_id: str) -> str:
    """Strip the infores prefix and separators: 'infores:drug-central' -> 'drugcentral'."""
    s = (source_id or "").strip().lower()
    if ":" in s:
        s = s.split(":", 1)[1]
    return s.replace("-", "").replace("_", "").replace(".", "")


def _tokens(source_id: str) -> Set[str]:
    """Split an infores id on its separators: 'automat-ctd' -> {'automat', 'ctd'}."""
    s = (source_id or "").strip().lower()
    if ":" in s:
        s = s.split(":", 1)[1]
    return {t for t in s.replace("_", "-").replace(".", "-").split("-") if t}


def source_matches(source_id: str, keywords: Set[str]) -> bool:
    """Test an infores identifier against a keyword set.

    Raw substring matching is unsafe here. TRAPI source ids are compound
    (`infores:automat-ctd`, `infores:text-mining-provider-targeted`) and short
    keywords appear inside unrelated names — a substring test for `go` would
    match `infores:mygene-info` and `infores:gtopdb`. Three tiers instead:

        1. exact match on the normalized id
        2. the keyword is a whole token of the id, which catches aggregator
           wrappers like `automat-ctd` around a curated source
        3. substring, but only for keywords of five characters or more, where
           an accidental match is implausible

    Short keywords therefore never match by substring, which is what keeps
    `go` from matching half the registry.
    """
    normalized = _normalize_source(source_id)
    if not normalized:
        return False
    token_set = _tokens(source_id)

    for kw in keywords:
        kw_norm = _normalize_source(kw)
        if not kw_norm:
            continue
        if normalized == kw_norm:
            return True
        if kw_norm in token_set or kw.lower() in token_set:
            return True
        if len(kw_norm) >= 5 and kw_norm in normalized:
            return True
    return False


def primary_source_score(source_id: str) -> float:
    """Score a primary knowledge source by how it was produced."""
    if not source_id:
        return PRIMARY_SOURCE_DEFAULT
    if source_matches(source_id, TEXT_MINED_SOURCES):
        return PRIMARY_SOURCE_TEXT_MINED
    if source_matches(source_id, CURATED_SOURCES):
        return PRIMARY_SOURCE_CURATED
    return PRIMARY_SOURCE_DEFAULT


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

CACHE_PATH = "runs/cache.sqlite"


# ---------------------------------------------------------------------------
# Constraint vocabulary mapping
#
# A plan states intent — "approved" — while the knowledge graph stores a
# particular source's encoding of it. ARAX surfaces ChEMBL's availability
# column as `chembl_availability_type` with values like "prescription only".
# Making the planner emit that would push a storage detail of one knowledge
# source into the layer that reasons about questions.
#
# This is the same translation `derive_strength` performs for
# `min_evidence_strength`, which likewise has no counterpart in TRAPI.
#
# Two rules keep it honest: a plan naming a raw field is used verbatim, with
# no mapping applied; and every mapping that fires is reported in the result,
# because expanding "approved" into a specific value set is a judgement the
# plan should be able to see and override.
# ---------------------------------------------------------------------------

#: Semantic field name -> node attributes to look for, in order. The semantic
#: name itself is tried first, so a graph that does carry it wins.
CONSTRAINT_FIELD_ALIASES: Dict[str, list] = {
    "approval_status": [
        "approval_status", "chembl_availability_type", "max_phase",
        "highest_clinical_phase",
    ],
    "clinical_trial_phase": ["max_phase", "highest_clinical_phase", "clinical_trial_phase"],
    "molecular_weight": ["molecular_weight", "mw_freebase"],
    "description": ["description"],
    "synonym": ["synonym", "synonyms"],
    "xref": ["xref", "equivalent_identifiers"],
}

#: (semantic field, plan value) -> the graph values that satisfy it.
#:
#: `approved` means currently marketed. Drugs that were withdrawn or
#: discontinued were once approved, and for a repurposing question they may be
#: the interesting ones — but that is a different claim, so it has its own key
#: rather than being folded in silently.
CONSTRAINT_VALUE_ALIASES: Dict[tuple, list] = {
    ("approval_status", "approved"): ["prescription only", "over the counter", "4"],
    ("approval_status", "marketed"): ["prescription only", "over the counter", "4"],
    ("approval_status", "ever_approved"): [
        "prescription only", "over the counter", "withdrawn", "discontinued", "4",
    ],
    ("approval_status", "withdrawn"): ["withdrawn"],
    ("approval_status", "discontinued"): ["discontinued"],
    ("approval_status", "investigational"): ["unknown", "0", "1", "2", "3"],
}


def resolve_constraint(field: str, op: str, value) -> tuple:
    """Translate a plan constraint into the graph's own vocabulary.

    Returns (candidate_fields, op, value, note). `note` is None when nothing
    was translated, and otherwise describes the substitution for the result.

    Value expansion turns an equality into a membership test, since one
    semantic value usually covers several stored ones.
    """
    fields = CONSTRAINT_FIELD_ALIASES.get(field, [field])
    key = (field, str(value).strip().lower()) if value is not None else None
    expansion = CONSTRAINT_VALUE_ALIASES.get(key) if key else None

    if expansion is None:
        note = None if fields == [field] else (
            f"'{field}' also searched as {fields[1:]}"
        )
        return fields, op, value, note

    new_op = "in" if op in ("eq", "in") else op
    note = (
        f"'{field} {op} {value}' interpreted as '{fields[0]}"
        + (f"/{fields[1]}" if len(fields) > 1 else "")
        + f" {new_op} {expansion}'"
    )
    return fields, new_op, expansion, note
