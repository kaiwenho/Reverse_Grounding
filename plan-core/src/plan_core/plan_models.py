"""
Pydantic models for the biomedical query plan.

These models mirror the JSON Schema at /schema/query_plan.schema.json.
They're used for:

- Programmatic construction of plans (in tests, in code paths that don't
  go through the LLM).
- Type-safe manipulation of plans after validation.
- Serialization to JSON matching the schema.

Runtime validation of LLM output is done by validators.py against the
compiled JSON Schema (which pulls Biolink enums dynamically). These
Pydantic models are intentionally *slightly permissive* — they use
`str` for predicate/category/qualifier fields rather than enums —
because the Biolink enums are large and change per model version.
The JSON Schema layer is where those constraints are enforced.

If you want strict Pydantic-side enum enforcement, wire the Biolink
vocab into a pre-validator; keep the model definitions themselves stable.
"""

from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, List, Literal, Optional, Union

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    field_validator,
    model_validator,
)


# ------------------------------------------------------------------------
# Enums (small, stable enums only — large Biolink enums stay as `str`)
# ------------------------------------------------------------------------

class PlanMode(str, Enum):
    discovery = "discovery"
    explanation = "explanation"
    hybrid = "hybrid"


class Intent(str, Enum):
    discovery = "discovery"
    validation = "validation"
    mixed = "mixed"


class ConfidenceLevel(str, Enum):
    high = "high"
    medium = "medium"
    low = "low"


class RefusalReason(str, Enum):
    out_of_scope = "out_of_scope"
    unsafe_or_clinical_advice = "unsafe_or_clinical_advice"
    insufficient_information = "insufficient_information"
    needs_clarification = "needs_clarification"
    requires_capability_not_available = "requires_capability_not_available"


class EdgeFilterOp(str, Enum):
    eq = "eq"
    neq = "neq"
    in_ = "in"
    not_in = "not_in"
    gte = "gte"
    lte = "lte"
    gt = "gt"
    lt = "lt"
    exists = "exists"
    not_exists = "not_exists"
    contains = "contains"


class RankingStrategy(str, Enum):
    path_priority = "path_priority"
    evidence_weighted = "evidence_weighted"
    shortest_path_first = "shortest_path_first"
    multi_path_consensus = "multi_path_consensus"
    genetic_evidence_boosted = "genetic_evidence_boosted"
    explanation_diversity = "explanation_diversity"
    custom = "custom"


# ------------------------------------------------------------------------
# Core building blocks
# ------------------------------------------------------------------------

_CFG = ConfigDict(extra="forbid", use_enum_values=True, populate_by_name=True)
Curie = Annotated[str, StringConstraints(pattern=r"^[A-Za-z][A-Za-z0-9._-]*:[^\s]+$")]
TaxonCurie = Annotated[str, StringConstraints(pattern=r"^NCBITaxon:[0-9]+$")]
SignatureDirection = Literal[
    "increased", "upregulated", "decreased", "downregulated"
]


class Confidence(BaseModel):
    model_config = _CFG
    level: ConfidenceLevel
    reasons: List[str] = Field(min_length=1)


class EntityConstraint(BaseModel):
    """Portable hard constraint supported by the current plan contract."""

    model_config = _CFG
    field: Literal["approval_status"]
    op: Literal["eq"]
    value: Literal["approved", "ever_approved"]


class IdentifiersInputBinding(BaseModel):
    model_config = _CFG
    binding_type: Literal["identifiers"] = "identifiers"
    identifiers: List[Curie] = Field(min_length=1)

    @field_validator("identifiers")
    @classmethod
    def identifiers_are_unique(cls, value: List[str]) -> List[str]:
        if len(value) != len(set(value)):
            raise ValueError("identifiers must be unique")
        return value


class ExternalInputBinding(BaseModel):
    model_config = _CFG
    binding_type: Literal["external_input"] = "external_input"
    input_ref: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_.-]*$")
    expected_format: Literal[
        "curie_list",
        "directional_gene_signature",
        "variant_list",
        "screening_panel",
    ]
    direction_filter: Optional[SignatureDirection] = None

    @model_validator(mode="after")
    def direction_filter_matches_format(self):
        is_directional_signature = (
            self.expected_format == "directional_gene_signature"
        )
        if is_directional_signature and self.direction_filter is None:
            raise ValueError(
                "directional_gene_signature bindings require direction_filter"
            )
        if not is_directional_signature and self.direction_filter is not None:
            raise ValueError(
                "direction_filter is valid only for directional_gene_signature"
            )
        return self


EntityInputBinding = Union[IdentifiersInputBinding, ExternalInputBinding]


class Entity(BaseModel):
    model_config = _CFG
    entity_ref: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    name: str = Field(min_length=1)
    aliases: Optional[List[str]] = None
    biolink_category: str  # runtime-checked against Biolink vocab
    is_variable: bool
    #: 'queried' (the default) or 'context'. See the schema description and
    #: `_check_entities_are_used` in validators.py.
    query_role: Literal["queried", "context"] = "queried"
    taxa: Optional[List[TaxonCurie]] = Field(default=None, min_length=1)
    input_binding: Optional[EntityInputBinding] = None
    constraints: Optional[List[EntityConstraint]] = None
    notes: Optional[str] = None

    @field_validator("taxa")
    @classmethod
    def taxa_are_unique(cls, value: Optional[List[str]]) -> Optional[List[str]]:
        if value is not None and len(value) != len(set(value)):
            raise ValueError("taxa must be unique")
        return value


class Qualifiers(BaseModel):
    model_config = _CFG
    qualified_predicate: Optional[str] = None
    object_aspect_qualifier: Optional[str] = None
    object_direction_qualifier: Optional[str] = None
    subject_aspect_qualifier: Optional[str] = None
    subject_direction_qualifier: Optional[str] = None
    causal_mechanism_qualifier: Optional[str] = None
    anatomical_context_qualifier: Optional[str] = None
    species_context_qualifier: Optional[str] = None


class EdgeFilter(BaseModel):
    model_config = _CFG
    field: str
    op: EdgeFilterOp
    value: Optional[Union[str, float, bool, List[Any]]] = None


class Hop(BaseModel):
    model_config = _CFG
    subject_ref: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    predicate: str = Field(pattern=r"^biolink:[a-z][a-z0-9_]*$")
    object_ref: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    negated: bool = False
    qualifiers: Optional[Qualifiers] = None
    predicate_expansion: Literal["self_only", "descendants"] = "descendants"
    edge_filters: Optional[List[EdgeFilter]] = None
    notes: Optional[str] = None

    @model_validator(mode="after")
    def endpoints_must_differ(self):
        if self.subject_ref == self.object_ref:
            raise ValueError("self-loop hops are unsupported: subject_ref must differ from object_ref")
        return self


class Path(BaseModel):
    model_config = _CFG
    path_id: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    archetype_tag: Optional[str] = None
    rationale: str = Field(min_length=1)
    hops: List[Hop] = Field(min_length=1, max_length=5)
    return_entity_ref: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")
    expected_result_category: Optional[str] = None
    disabled: bool = False
    notes: Optional[str] = None


# ------------------------------------------------------------------------
# Explanation-mode blocks
# ------------------------------------------------------------------------

class EntityEndpointBinding(BaseModel):
    model_config = _CFG
    binding_type: Literal["entity"] = "entity"
    entity_ref: str = Field(pattern=r"^[A-Za-z_][A-Za-z0-9_]*$")


class FromDiscoveryEndpointBinding(BaseModel):
    model_config = _CFG
    binding_type: Literal["from_discovery"] = "from_discovery"
    from_path_ids: List[
        Annotated[str, StringConstraints(pattern=r"^[A-Za-z0-9_\-]+$")]
    ] = Field(min_length=1)
    fanout_top_k: int = Field(ge=1)

    @field_validator("from_path_ids")
    @classmethod
    def path_ids_are_unique(cls, value: List[str]) -> List[str]:
        if len(value) != len(set(value)):
            raise ValueError("from_path_ids must be unique")
        return value


EndpointBinding = Union[EntityEndpointBinding, FromDiscoveryEndpointBinding]


class ExplanationPostFilter(BaseModel):
    model_config = _CFG
    predicate_whitelist: Optional[List[str]] = None
    predicate_blacklist: Optional[List[str]] = None
    required_predicates_anywhere: Optional[List[str]] = None
    required_qualifier_on_any_edge: Optional[Qualifiers] = None
    forbidden_qualifier_on_any_edge: Optional[Qualifiers] = None
    min_publications_per_edge: Optional[int] = Field(default=None, ge=0)
    required_knowledge_sources_per_edge: Optional[List[str]] = None
    excluded_knowledge_sources_per_edge: Optional[List[str]] = None
    drop_paths_with_negated_edges: bool = True
    unique_intermediate_nodes: bool = True


class ExplanationReturn(BaseModel):
    model_config = _CFG
    top_k_paths: Optional[int] = Field(default=None, ge=1)
    rank_by: Literal[
        "shortest_first",
        "evidence_strength",
        "num_publications",
        "num_knowledge_sources",
        "composite",
    ] = "composite"
    group_by_intermediate_category: bool = True


class ExplanationQuery(BaseModel):
    model_config = _CFG
    query_id: str = Field(pattern=r"^[A-Za-z0-9_\-]+$")
    archetype_tag: Optional[str] = None
    rationale: str = Field(min_length=1)
    endpoint_a: EndpointBinding
    endpoint_b: EndpointBinding
    max_hops: int = Field(ge=1, le=5)
    middle_category_whitelist: Optional[List[str]] = None
    middle_category_blacklist: Optional[List[str]] = None
    post_filter: Optional[ExplanationPostFilter] = None
    return_: Optional[ExplanationReturn] = Field(default=None, alias="return")
    notes: Optional[str] = None

    @model_validator(mode="after")
    def endpoints_must_differ(self):
        same_entity = (
            isinstance(self.endpoint_a, EntityEndpointBinding)
            and isinstance(self.endpoint_b, EntityEndpointBinding)
            and self.endpoint_a.entity_ref == self.endpoint_b.entity_ref
        )
        shared_discovery_sources = (
            isinstance(self.endpoint_a, FromDiscoveryEndpointBinding)
            and isinstance(self.endpoint_b, FromDiscoveryEndpointBinding)
            and bool(
                set(self.endpoint_a.from_path_ids)
                & set(self.endpoint_b.from_path_ids)
            )
        )
        if same_entity or shared_discovery_sources:
            raise ValueError(
                "self-loop explanation queries are unsupported: endpoint_a "
                "and endpoint_b must use different entity bindings and "
                "non-overlapping discovery sources"
            )
        return self


# ------------------------------------------------------------------------
# Top-level plan
# ------------------------------------------------------------------------

class Interpretation(BaseModel):
    model_config = _CFG
    archetypes: List[str] = Field(min_length=1)
    restated_question: str = Field(min_length=1)
    intent: Optional[Intent] = None
    assumptions: Optional[List[str]] = None


class RankingCriterion(BaseModel):
    model_config = _CFG
    name: str
    direction: Literal["asc", "desc"]
    weight: Optional[float] = Field(default=None, ge=0)
    origin: Literal["user_requested", "planner_recommended"]
    application: Literal["rank"]
    scope: Literal["portable", "executor_specific"]
    profile_id: Optional[str] = Field(default=None, min_length=1)
    preferred_values: Optional[List[str]] = Field(default=None, min_length=1)
    rationale: str = Field(min_length=1)


class RankingSpec(BaseModel):
    model_config = _CFG
    strategy: RankingStrategy
    criteria: List[RankingCriterion] = Field(min_length=1)
    top_k: int = Field(ge=1)
    tie_breaker: Optional[str] = None


class CandidateRanking(BaseModel):
    model_config = _CFG
    explanation_influence: Literal["none", "annotate_only", "rerank"]
    rationale: str = Field(min_length=1)
    discovery: RankingSpec
    final: Optional[RankingSpec] = None


class RankingPlan(BaseModel):
    model_config = _CFG
    candidate_ranking: Optional[CandidateRanking] = None
    explanation_ranking: Optional[RankingSpec] = None


class EvidenceRecommendation(BaseModel):
    model_config = _CFG
    feature: Literal[
        "primary_knowledge_source",
        "knowledge_level",
        "agent_type",
        "publications",
        "knowledge_sources",
        "recency",
    ]
    origin: Literal["user_requested", "planner_recommended"]
    application: Literal["report"]
    preferred_values: Optional[List[str]] = Field(default=None, min_length=1)
    rationale: str = Field(min_length=1)


class EvidencePolicy(BaseModel):
    model_config = _CFG
    origin: Literal["user_requested"]
    application: Literal["filter"]
    rationale: str = Field(min_length=1)
    required_knowledge_sources: Optional[List[str]] = None
    excluded_knowledge_sources: Optional[List[str]] = None
    require_primary_knowledge_source: bool = False
    min_publications: Optional[int] = Field(default=None, ge=0)
    min_year: Optional[int] = Field(default=None, ge=1900)
    agent_type: Optional[List[str]] = None


class Aggregation(BaseModel):
    model_config = _CFG
    combine: Literal["union", "intersection", "weighted_union", "path_priority_union"] = "union"
    deduplicate_on: Literal["entity_ref", "name", "resolved_id"] = "resolved_id"
    group_evidence_across_paths: bool = True
    attach_explanations_to_candidates: bool = False


class Gap(BaseModel):
    model_config = _CFG
    kind: Literal[
        "biolink_predicate_missing",
        "biolink_qualifier_missing",
        "kg_data_missing",
        "requires_external_computation",
        "ambiguous_question",
        "out_of_scope",
        "tool_cannot_constrain_predicates",
    ]
    description: str
    affected_path_ids: Optional[List[str]] = None
    affected_query_ids: Optional[List[str]] = None
    workaround: Optional[str] = None


class Refusal(BaseModel):
    model_config = _CFG
    reason: RefusalReason
    message: str


class QueryPlan(BaseModel):
    model_config = _CFG
    plan_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    biolink_version: str = Field(pattern=r"^\d+\.\d+\.\d+$")
    plan_id: Optional[str] = None
    created_at: Optional[str] = None
    question: str = Field(min_length=1)
    plan_mode: PlanMode
    interpretation: Interpretation
    entities: List[Entity]
    paths: Optional[List[Path]] = None
    explanation_queries: Optional[List[ExplanationQuery]] = None
    ranking: Optional[RankingPlan] = None
    evidence_policy: Optional[EvidencePolicy] = None
    evidence_recommendations: Optional[List[EvidenceRecommendation]] = None
    aggregation: Optional[Aggregation] = None
    confidence: Confidence
    gaps: Optional[List[Gap]] = None
    refusal: Optional[Refusal] = None
    notes: Optional[str] = None

    def to_json_dict(self) -> dict:
        """Serialize with `return_` -> `return` alias handling."""
        return self.model_dump(by_alias=True, exclude_none=True)
