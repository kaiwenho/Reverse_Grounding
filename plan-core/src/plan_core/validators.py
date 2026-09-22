"""
Plan validation: JSON Schema validation + semantic (cross-field) checks.

This is the quality gate every LLM output passes through. Order:

1. `validate_schema()` runs the Biolink-compiled JSON Schema check.
2. `validate_semantics()` runs the checks that JSON Schema can't express
   cleanly (identifier uniqueness, graph connectivity, hybrid-mode wiring,
   answer-variable rules, atomic fixed-entity names, fixed-endpoint explanation
   fallbacks, ranking-shape rules, complete explanation fan-out, terminal
   refusal shape,
   deprecated-predicate rejection, qualifier-family compatibility, and
   predicate direction against Biolink metadata).

Callers should use `validate_plan()`, which runs both and returns a single
`ValidationResult`. If it comes back not-ok, the planner should log the
errors and either retry with them appended to the prompt or refuse.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from jsonschema import Draft202012Validator

from .archetype_catalog import archetype_tags
from .biolink_vocab import (
    QUALIFIER_PREDICATE_FAMILY_ROOTS,
    BiolinkVocabulary,
    load_biolink_vocabulary,
)
from .compile_schema import compile_schema


# ------------------------------------------------------------------------
# Result types
# ------------------------------------------------------------------------

@dataclass
class ValidationError:
    location: str  # "paths/0/hops/1/predicate" or similar
    message: str
    kind: str      # 'schema' | 'semantic'


@dataclass
class ValidationResult:
    ok: bool
    errors: List[ValidationError] = field(default_factory=list)

    def format(self) -> str:
        if self.ok:
            return "OK"
        lines = [f"{len(self.errors)} validation error(s):"]
        for e in self.errors:
            lines.append(f"  [{e.kind}] {e.location}: {e.message}")
        return "\n".join(lines)


# ------------------------------------------------------------------------
# Schema validation
# ------------------------------------------------------------------------

_compiled_validator: Optional[Draft202012Validator] = None


def get_compiled_validator(force_reload: bool = False) -> Draft202012Validator:
    """Cached Draft 2020-12 validator against the Biolink-compiled schema."""
    global _compiled_validator
    if _compiled_validator is None or force_reload:
        _compiled_validator = Draft202012Validator(compile_schema())
    return _compiled_validator


def _actionable_schema_errors(error):
    """Prefer the matching discriminated ``oneOf`` branch's errors.

    ``jsonschema`` normally summarizes a failed union as "not valid under any
    of the given schemas".  Plan unions carry a ``binding_type`` discriminator,
    so errors from the branch whose constant matches the instance are both
    unambiguous and much more useful to the planner's repair prompt.  For
    example, a directional signature missing ``direction_filter`` now reports
    that required property instead of only the generic union failure.
    """
    if (
        getattr(error, "validator", None) == "oneOf"
        and isinstance(getattr(error, "instance", None), dict)
        and isinstance(getattr(error, "validator_value", None), list)
        and getattr(error, "context", None)
    ):
        discriminator = error.instance.get("binding_type")
        matching_branch = None
        for index, branch in enumerate(error.validator_value):
            if not isinstance(branch, dict):
                continue
            binding_schema = (
                branch.get("properties", {}).get("binding_type", {})
            )
            if binding_schema.get("const") == discriminator:
                matching_branch = index
                break

        if matching_branch is not None:
            matching_errors = []
            for child in error.context:
                branch_indexes = [
                    token for token in child.schema_path
                    if isinstance(token, int)
                ]
                if branch_indexes and branch_indexes[0] == matching_branch:
                    matching_errors.extend(_actionable_schema_errors(child))
            if matching_errors:
                return matching_errors

    return [error]


def validate_schema(plan: Dict[str, Any]) -> List[ValidationError]:
    v = get_compiled_validator()
    errors: List[ValidationError] = []
    seen: set[tuple[str, str]] = set()
    for err in v.iter_errors(plan):
        for actionable in _actionable_schema_errors(err):
            loc = (
                "/".join(str(p) for p in actionable.absolute_path)
                or "<root>"
            )
            key = (loc, actionable.message)
            if key not in seen:
                errors.append(ValidationError(loc, actionable.message, "schema"))
                seen.add(key)
    return errors


# ------------------------------------------------------------------------
# Semantic validation
# ------------------------------------------------------------------------

def _err(location: str, message: str) -> ValidationError:
    return ValidationError(location, message, "semantic")

def _as_list(value: Any) -> List[Any]:
    """Coerce a field that should be a list into one.

    LLM output sometimes emits a bare string where the schema wants an array.
    Iterating that string yields one error per character, so normalize first.
    A lone string is wrapped rather than dropped, so its content still gets
    checked — the schema layer separately reports the type error.
    Non-list, non-string values yield [] and are left to the schema layer.
    """
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        return [value]
    return []


def _as_dict(value: Any) -> Dict[str, Any]:
    """Coerce a field that should be an object into one. See _as_list."""
    return value if isinstance(value, dict) else {}


_RELATIONAL_ENTITY_CONNECTOR_RE = re.compile(
    r"^\s*(?P<left>.+?)\s+"
    r"(?P<cue>"
    r"in\s+(?:patients|people|individuals)\s+with|"
    r"among\s+(?:patients|people|individuals)\s+with|"
    r"due\s+to|caused\s+by|resulting\s+from|secondary\s+to|"
    r"triggered\s+by|induced\s+by|arising\s+from|attributable\s+to|"
    r"derived\s+from|associated\s+with|linked\s+to|related\s+to|from"
    r")\s+(?P<right>.+?)\s*$",
    re.IGNORECASE,
)

_RELATIONAL_ENTITY_OF_RE = re.compile(
    r"^\s*(?P<left>.+?\b(?P<relation_noun>"
    r"manifestations?|complications?|sequelae?))\s+"
    r"(?P<cue>of)\s+(?P<right>.+?)\s*$",
    re.IGNORECASE,
)


def _split_relational_entity_name(
    value: Any,
) -> Optional[tuple[str, str, str]]:
    """Return two concepts and their high-confidence relational cue.

    This is deliberately narrower than general natural-language parsing. It
    catches constructions that should not be sent to an entity resolver as one
    name, while avoiding bare ``with`` and ``of`` because they occur in valid
    biomedical concept labels. The validator reports the problem; it does not
    choose a predicate or rewrite the plan.
    """
    if not isinstance(value, str):
        return None

    for pattern in (
        _RELATIONAL_ENTITY_OF_RE,
        _RELATIONAL_ENTITY_CONNECTOR_RE,
    ):
        match = pattern.match(value)
        if match is None:
            continue
        left = match.group("left").strip(" \t\r\n,;:.")
        cue = " ".join(match.group("cue").lower().split())
        relation_noun = match.groupdict().get("relation_noun")
        if relation_noun is not None:
            cue = f"{relation_noun.lower()} {cue}"
        right = match.group("right").strip(" \t\r\n,;:.")
        if left and right:
            return left, cue, right
    return None


def _validate_predicate_value(
    errors: List[ValidationError],
    vocab: BiolinkVocabulary,
    location: str,
    value: Any,
) -> bool:
    """Validate one predicate and distinguish deprecated from unknown terms."""
    if not isinstance(value, str):
        return False  # JSON Schema reports the type/required-field error.
    if vocab.is_valid_predicate(value):
        return True
    if vocab.is_deprecated_predicate(value):
        errors.append(_err(
            location,
            f"'{value}' is deprecated in Biolink v{vocab.biolink_version}; "
            "use an active predicate from the loaded vocabulary",
        ))
    else:
        errors.append(_err(
            location,
            f"'{value}' is not a valid active Biolink predicate "
            f"in loaded model (v{vocab.biolink_version})",
        ))
    return False


def _validate_qualified_predicate(
    errors: List[ValidationError],
    vocab: BiolinkVocabulary,
    qualifiers: Any,
    location: str,
) -> None:
    qualifier_object = _as_dict(qualifiers)
    if "qualified_predicate" in qualifier_object:
        _validate_predicate_value(
            errors,
            vocab,
            f"{location}/qualified_predicate",
            qualifier_object.get("qualified_predicate"),
        )


def _validate_hop_qualifiers(
    errors: List[ValidationError],
    vocab: BiolinkVocabulary,
    predicate: Any,
    qualifiers: Any,
    location: str,
) -> None:
    """Validate qualifier contents and the base predicate's Biolink family."""
    qualifier_object = _as_dict(qualifiers)
    _validate_qualified_predicate(errors, vocab, qualifier_object, location)

    # Empty objects are discouraged by the prompt but do not constitute a
    # qualified relationship. JSON Schema reports malformed non-object values.
    if not qualifier_object or not isinstance(predicate, str):
        return
    if not vocab.is_valid_predicate(predicate):
        return  # The predicate validator reports unknown/deprecated terms.
    if vocab.qualifier_predicate_family(predicate) is None:
        families = ", ".join(QUALIFIER_PREDICATE_FAMILY_ROOTS)
        errors.append(_err(
            location,
            f"predicate '{predicate}' cannot carry qualifiers under this "
            "Biolink plan contract; qualified hops are limited to the "
            f"{families} families and their active descendants",
        ))


def _direction_matches(
    vocab: BiolinkVocabulary,
    predicate: str,
    subject_category: str,
    object_category: str,
) -> bool:
    """Check a predicate's subject/object categories against its domain/range."""
    domain = vocab.predicate_domains.get(predicate)
    range_ = vocab.predicate_ranges.get(predicate)
    domain_ok = domain is None or vocab.category_satisfies(subject_category, domain)
    range_ok = range_ is None or vocab.category_satisfies(object_category, range_)
    return domain_ok and range_ok


_INCREASED_DIRECTIONS = {"increased", "upregulated"}
_DECREASED_DIRECTIONS = {"decreased", "downregulated"}


def _validate_signature_reversal_hop(
    errors: List[ValidationError],
    entities: Dict[str, Dict[str, Any]],
    hop: Dict[str, Any],
    location: str,
) -> None:
    """Check that a directional-signature partition is reversed, not copied."""
    qualifiers = _as_dict(hop.get("qualifiers"))
    for side in ("subject", "object"):
        entity_ref = hop.get(f"{side}_ref")
        entity = entities.get(entity_ref) if isinstance(entity_ref, str) else None
        binding = _as_dict(entity.get("input_binding")) if entity else {}
        if binding.get("expected_format") != "directional_gene_signature":
            continue

        source_direction = binding.get("direction_filter")
        if source_direction in _INCREASED_DIRECTIONS:
            valid_drug_directions = _DECREASED_DIRECTIONS
        elif source_direction in _DECREASED_DIRECTIONS:
            valid_drug_directions = _INCREASED_DIRECTIONS
        else:
            continue  # JSON Schema reports a missing/invalid direction_filter.

        aspect_key = f"{side}_aspect_qualifier"
        direction_key = f"{side}_direction_qualifier"
        if qualifiers.get(aspect_key) != "expression":
            errors.append(_err(
                f"{location}/qualifiers/{aspect_key}",
                "a Q3 directional-signature partition must use the expression "
                "aspect on the signature-gene endpoint",
            ))
        effect_direction = qualifiers.get(direction_key)
        if effect_direction not in valid_drug_directions:
            errors.append(_err(
                f"{location}/qualifiers/{direction_key}",
                f"signature source direction '{source_direction}' requires an "
                "opposite drug expression direction from "
                f"{sorted(valid_drug_directions)}",
            ))


_DISCOVERY_RETURN_CATEGORY_FAMILIES = (
    "ChemicalEntity",
    "GeneOrGeneProduct",
    "DiseaseOrPhenotypicFeature",
    "BiologicalProcessOrActivity",
    "AnatomicalEntity",
    "SequenceVariant",
)


def _return_categories_are_compatible(
    vocab: BiolinkVocabulary,
    categories: List[str],
) -> bool:
    """Whether categories can form one semantically coherent candidate set."""
    normalized = {category.removeprefix("biolink:") for category in categories}
    if len(normalized) <= 1:
        return True

    # Accept descendants of one useful Biolink candidate family, such as Drug
    # plus SmallMolecule or Gene plus Protein.
    if any(
        all(vocab.category_satisfies(category, family) for category in normalized)
        for family in _DISCOVERY_RETURN_CATEGORY_FAMILIES
    ):
        return True

    # Also accept a direct ancestor/descendant chain outside the common families.
    return any(
        all(vocab.category_satisfies(category, possible_ancestor)
            for category in normalized)
        for possible_ancestor in normalized
    )


_EXPLANATION_DERIVED_CRITERIA = {
    "num_explanation_paths",
    "distinct_intermediate_categories",
}


def _validate_ranking_spec(
    errors: List[ValidationError],
    vocab: BiolinkVocabulary,
    spec: Any,
    location: str,
    stage: str,
    active_paths: List[Dict[str, Any]],
    has_user_ranking_override: bool,
) -> set[str]:
    """Validate one stage-specific ranking specification."""
    if not isinstance(spec, dict):
        errors.append(_err(
            location,
            "this ranking stage must be an explicit object",
        ))
        return set()

    criteria = [
        criterion for criterion in _as_list(spec.get("criteria"))
        if isinstance(criterion, dict)
    ]
    criterion_names = {
        criterion.get("name") for criterion in criteria
        if isinstance(criterion.get("name"), str)
    }
    strategy = spec.get("strategy")

    if not criteria:
        errors.append(_err(
            f"{location}/criteria",
            "ranking criteria must contain at least one explicit criterion",
        ))
    if "top_k" not in spec:
        errors.append(_err(
            f"{location}/top_k",
            "top_k is required for every ranking stage",
        ))

    if stage == "explanation":
        if "num_explanation_paths" in criterion_names:
            errors.append(_err(
                f"{location}/criteria",
                "num_explanation_paths counts explanation support per "
                "candidate and belongs only in "
                "ranking.candidate_ranking.final; explanation_ranking "
                "orders individual paths",
            ))
        if strategy not in {"explanation_diversity", "custom"}:
            errors.append(_err(
                f"{location}/strategy",
                "explanation_ranking must use explanation_diversity, or custom "
                "when the user explicitly requests another supported priority",
            ))
        if (
            strategy == "explanation_diversity"
            and "distinct_intermediate_categories" not in criterion_names
        ):
            errors.append(_err(
                f"{location}/criteria",
                "explanation_diversity requires the "
                "distinct_intermediate_categories criterion",
            ))
    else:
        if strategy == "explanation_diversity":
            errors.append(_err(
                f"{location}/strategy",
                "explanation_diversity ranks explanation paths, not candidates; "
                "place it in ranking.explanation_ranking",
            ))
        if strategy == "shortest_path_first" and active_paths:
            errors.append(_err(
                f"{location}/strategy",
                "shortest_path_first cannot rank candidates from fixed "
                "discovery paths; use evidence_weighted for one path or "
                "multi_path_consensus for multiple paths",
            ))
        if strategy == "multi_path_consensus":
            if len(active_paths) < 2:
                errors.append(_err(
                    f"{location}/strategy",
                    "multi_path_consensus requires at least two active "
                    "discovery paths",
                ))
            if "num_supporting_paths" not in criterion_names:
                errors.append(_err(
                    f"{location}/criteria",
                    "multi_path_consensus requires the num_supporting_paths "
                    "criterion",
                ))

    if stage == "candidate_discovery":
        early_explanation_criteria = sorted(
            criterion_names & _EXPLANATION_DERIVED_CRITERIA
        )
        if early_explanation_criteria:
            errors.append(_err(
                f"{location}/criteria",
                "candidate_ranking.discovery cannot use explanation-derived "
                f"criteria that do not exist yet: {early_explanation_criteria}",
            ))
        if len(active_paths) == 1:
            for constant_name in ("num_supporting_paths", "path_length"):
                if constant_name in criterion_names:
                    errors.append(_err(
                        f"{location}/criteria",
                        f"{constant_name} cannot distinguish candidates from "
                        "one fixed discovery path",
                    ))

    if strategy == "evidence_weighted":
        portable_evidence_names = {
            criterion.get("name")
            for criterion in criteria
            if criterion.get("scope") == "portable"
        }.intersection({
            "knowledge_level",
            "num_publications",
            "num_knowledge_sources",
        })
        if not portable_evidence_names:
            errors.append(_err(
                f"{location}/criteria",
                "evidence_weighted requires at least one portable evidence "
                "criterion: knowledge_level, num_publications, or "
                "num_knowledge_sources",
            ))

    if strategy == "genetic_evidence_boosted" and "genetic_support" not in criterion_names:
        errors.append(_err(
            f"{location}/criteria",
            "genetic_evidence_boosted requires the genetic_support criterion",
        ))

    if strategy == "custom":
        if not has_user_ranking_override:
            errors.append(_err(
                "confidence/reasons",
                "custom ranking requires a confidence reason beginning "
                "user_requested_ranking_override:",
            ))
        missing_weights = [
            i for i, criterion in enumerate(criteria)
            if "weight" not in criterion
        ]
        if missing_weights:
            errors.append(_err(
                f"{location}/criteria",
                "custom ranking requires an explicit weight for every "
                f"criterion; missing at indexes {missing_weights}",
            ))

    for i, criterion in enumerate(criteria):
        name = criterion.get("name")
        preferred_values = _as_list(criterion.get("preferred_values"))
        if name == "knowledge_level":
            if not preferred_values:
                errors.append(_err(
                    f"{location}/criteria/{i}/preferred_values",
                    "portable knowledge_level ranking must state its "
                    "preferred values for a standalone user",
                ))
            for j, value in enumerate(preferred_values):
                if (
                    isinstance(value, str)
                    and value not in vocab.knowledge_level_values
                ):
                    errors.append(_err(
                        f"{location}/criteria/{i}/preferred_values/{j}",
                        f"'{value}' is not a Biolink knowledge level",
                    ))

        if name == "edge_evidence_strength":
            if criterion.get("scope") != "executor_specific":
                errors.append(_err(
                    f"{location}/criteria/{i}/scope",
                    "edge_evidence_strength is an executor-specific derived "
                    "criterion, not a Biolink field",
                ))
            if criterion.get("profile_id") != "executor_epc_v1":
                errors.append(_err(
                    f"{location}/criteria/{i}/profile_id",
                    "edge_evidence_strength must identify the versioned "
                    "executor_epc_v1 profile",
                ))

    return criterion_names


def validate_semantics(
    plan: Dict[str, Any],
    vocab: Optional[BiolinkVocabulary] = None,
) -> List[ValidationError]:
    """
    Cross-field checks not expressible in JSON Schema:

    - entity_ref, path_id, and query_id values are unique.
    - Every entity_ref referenced in paths/queries exists in `entities`.
    - Every path's return_entity_ref exists AND appears in a hop.
    - Discovery paths are connected, return a variable entity, and each enabled
      path contains at least one non-variable fixed anchor.
    - Path expected-result categories and explanation middle-node category
      constraints occur in the loaded Biolink Model.
    - A Path expected-result category is compatible with its return entity.
    - Explanation-query entity endpoints reference non-variable entities.
    - `from_discovery` bindings only in hybrid mode, reference enabled paths,
      and aggregate compatible return-entity categories.
    - Hybrid plans that attach explanations cover every enabled discovery path.
    - Every active predicate hop between two distinct fixed entities has an
      independent predicate-unconstrained explanation fallback in hybrid mode.
    - Ranking subjects and stages match plan mode; hybrid plans explicitly state
      whether explanations are independent, annotate candidates, or rerank them.
    - Every ranking stage has explicit criteria and top_k. Consensus ranking has
      at least two active discovery paths, and fixed discovery paths do not use
      shortest-path ranking.
    - Predicate CURIEs are active (not deprecated) and their direction matches
      Biolink domain/range.
    - Qualifier-bearing hops use an active descendant of affects, regulates,
      or interacts_with.
    - Q3 directional-signature partitions use the opposite expression direction.
    - biolink_category values are valid Biolink categories.
    - Fixed resolver-grounded entity names do not combine separate concepts in
      high-confidence relational constructions.
    - Input-bound entities are non-variable.
    - The only supported entity constraint is semantic `approval_status`, using
      `approved` or `ever_approved` with eq.
    - Hop endpoints and explanation-query endpoint bindings are not self-loops.
    - archetype_tag values agree with the archetype catalog.
    - If a `refusal` is present, path/query semantic checks are skipped.
    """
    v = vocab or load_biolink_vocabulary()
    errors: List[ValidationError] = []

    is_refused = "refusal" in plan
    mode = plan.get("plan_mode")
    entities: Dict[str, Dict[str, Any]] = {}
    for i, ent in enumerate(_as_list(plan.get("entities"))):
        if not isinstance(ent, dict):
            continue
        ref = ent.get("entity_ref")
        if isinstance(ref, str):
            if ref in entities:
                errors.append(_err(
                    f"entities/{i}/entity_ref",
                    f"duplicate entity_ref '{ref}'",
                ))
            else:
                entities[ref] = ent

        cat = ent.get("biolink_category")
        if isinstance(cat, str) and not v.is_valid_category(cat):
            errors.append(_err(
                f"entities/{i}/biolink_category",
                f"category '{cat}' is not a valid Biolink category (v{v.biolink_version})",
            ))

        # Variable labels describe answer sets, and input-bound entities
        # describe supplied identifier sets. Only a fixed entity that will be
        # sent to a name resolver must denote one atomic concept.
        if ent.get("is_variable") is False and "input_binding" not in ent:
            relational_parts = _split_relational_entity_name(ent.get("name"))
            if relational_parts is not None:
                left, cue, right = relational_parts
                errors.append(_err(
                    f"entities/{i}/name",
                    f"fixed entity name '{ent.get('name')}' appears to combine "
                    f"separate concepts '{left}' and '{right}' using the "
                    f"relational cue '{cue}'; create separate Entity objects "
                    "and represent their relationship in graph structure. "
                    "Normally use an open ExplanationQuery; use a predicate "
                    "hop only when the question states or clearly implies an "
                    "active Biolink predicate. If the concepts cannot be "
                    "separated confidently, emit a needs_clarification refusal",
                ))

        if "input_binding" in ent and ent.get("is_variable") is True:
            errors.append(_err(
                f"entities/{i}/input_binding",
                "an input-bound entity must have is_variable=false",
            ))

        input_binding = _as_dict(ent.get("input_binding"))
        if input_binding.get("binding_type") == "external_input":
            expected_format = input_binding.get("expected_format")
            direction_filter = input_binding.get("direction_filter")
            if (
                expected_format == "directional_gene_signature"
                and direction_filter not in (
                    _INCREASED_DIRECTIONS | _DECREASED_DIRECTIONS
                )
            ):
                errors.append(_err(
                    f"entities/{i}/input_binding/direction_filter",
                    "a directional_gene_signature binding requires one valid "
                    "source-partition direction_filter",
                ))
            elif (
                expected_format != "directional_gene_signature"
                and direction_filter is not None
            ):
                errors.append(_err(
                    f"entities/{i}/input_binding/direction_filter",
                    "direction_filter is valid only for a "
                    "directional_gene_signature binding",
                ))

        for j, constraint in enumerate(_as_list(ent.get("constraints"))):
            if not isinstance(constraint, dict):
                continue
            if constraint.get("field") != "approval_status":
                errors.append(_err(
                    f"entities/{i}/constraints/{j}/field",
                    "EntityConstraint currently supports only the portable "
                    "'approval_status' field",
                ))
                continue
            if constraint.get("op") != "eq":
                errors.append(_err(
                    f"entities/{i}/constraints/{j}/op",
                    "approval_status is a hard semantic constraint and requires op='eq'",
                ))
            approval_value = constraint.get("value")
            if not isinstance(approval_value, str) or approval_value not in {
                "approved",
                "ever_approved",
            }:
                errors.append(_err(
                    f"entities/{i}/constraints/{j}/value",
                    "approval_status must be 'approved' (currently marketed) or "
                    "'ever_approved' (also discontinued or withdrawn)",
                ))

    entity_refs = set(entities.keys())
    valid_archetypes = set(archetype_tags())

    # Archetype tag check on interpretation
    interp = _as_dict(plan.get("interpretation"))
    interpretation_archetypes = {
        tag for tag in _as_list(interp.get("archetypes"))
        if isinstance(tag, str)
    }
    for i, tag in enumerate(_as_list(interp.get("archetypes"))):
        if isinstance(tag, str) and tag not in valid_archetypes:
            errors.append(_err(
                f"interpretation/archetypes/{i}",
                f"archetype tag '{tag}' not in catalog {sorted(valid_archetypes)}",
            ))

    if is_refused:
        # A refusal is a terminal, non-executable result. Keep its shape
        # unambiguous for standalone readers and downstream consumers.
        if _as_list(plan.get("entities")):
            errors.append(_err(
                "entities",
                "a refusal must use an empty entities array because it does "
                "not define an executable graph query",
            ))
        for field_name in (
            "paths",
            "explanation_queries",
            "ranking",
            "aggregation",
            "evidence_policy",
        ):
            if field_name in plan:
                errors.append(_err(
                    field_name,
                    f"a refusal must omit the executable '{field_name}' field",
                ))
        return errors  # Skip path/query semantic checks for refusals

    raw_paths = [
        path for path in _as_list(plan.get("paths"))
        if isinstance(path, dict)
    ]
    active_paths = [
        path for path in raw_paths
        if path.get("disabled", False) is not True
    ]
    confidence_reasons = {
        reason for reason in _as_list(_as_dict(plan.get("confidence")).get("reasons"))
        if isinstance(reason, str)
    }
    has_user_ranking_override = any(
        reason.startswith("user_requested_ranking_override:")
        for reason in confidence_reasons
    )

    # Ranking is stage-specific so a standalone reader can tell whether a
    # criterion orders candidate entities or explanation paths, and whether
    # candidate-specific explanations may change the candidate order.
    ranking = plan.get("ranking")
    candidate_ranking: Dict[str, Any] = {}
    explanation_ranking: Dict[str, Any] = {}
    explanation_influence: Optional[str] = None
    discovery_top_k: Optional[int] = None
    if not isinstance(ranking, dict):
        errors.append(_err(
            "ranking",
            "every executable plan must include an explicit ranking object",
        ))
    else:
        candidate_value = ranking.get("candidate_ranking")
        explanation_value = ranking.get("explanation_ranking")
        candidate_ranking = _as_dict(candidate_value)
        explanation_ranking = _as_dict(explanation_value)

        if mode in {"discovery", "hybrid"} and not candidate_ranking:
            errors.append(_err(
                "ranking/candidate_ranking",
                f"{mode} mode requires candidate_ranking",
            ))
        if mode == "explanation" and candidate_value is not None:
            errors.append(_err(
                "ranking/candidate_ranking",
                "pure explanation mode has no candidate entities to rank",
            ))
        if mode in {"explanation", "hybrid"} and not explanation_ranking:
            errors.append(_err(
                "ranking/explanation_ranking",
                f"{mode} mode requires explanation_ranking",
            ))
        if mode == "discovery" and explanation_value is not None:
            errors.append(_err(
                "ranking/explanation_ranking",
                "pure discovery mode has no explanation paths to rank",
            ))

        if candidate_ranking:
            explanation_influence = candidate_ranking.get("explanation_influence")
            discovery_spec = _as_dict(candidate_ranking.get("discovery"))
            if isinstance(discovery_spec.get("top_k"), int):
                discovery_top_k = discovery_spec["top_k"]
            _validate_ranking_spec(
                errors,
                v,
                discovery_spec,
                "ranking/candidate_ranking/discovery",
                "candidate_discovery",
                active_paths,
                has_user_ranking_override,
            )
            final_value = candidate_ranking.get("final")
            if explanation_influence == "rerank":
                if not isinstance(final_value, dict):
                    errors.append(_err(
                        "ranking/candidate_ranking/final",
                        "final ranking is required when explanation_influence "
                        "is 'rerank'",
                    ))
                else:
                    final_names = _validate_ranking_spec(
                        errors,
                        v,
                        final_value,
                        "ranking/candidate_ranking/final",
                        "candidate_final",
                        active_paths,
                        has_user_ranking_override,
                    )
                    if not (final_names & _EXPLANATION_DERIVED_CRITERIA):
                        errors.append(_err(
                            "ranking/candidate_ranking/final/criteria",
                            "a final candidate reranking must include at least "
                            "one explanation-derived criterion",
                        ))
            elif final_value is not None:
                errors.append(_err(
                    "ranking/candidate_ranking/final",
                    "final ranking is allowed only when "
                    "explanation_influence is 'rerank'",
                ))

            if mode == "discovery" and explanation_influence != "none":
                errors.append(_err(
                    "ranking/candidate_ranking/explanation_influence",
                    "discovery mode requires explanation_influence='none'",
                ))

        if explanation_ranking:
            _validate_ranking_spec(
                errors,
                v,
                explanation_ranking,
                "ranking/explanation_ranking",
                "explanation",
                active_paths,
                has_user_ranking_override,
            )

    # Reporting recommendations are portable, non-filtering instructions for
    # users who take the plan to another Biolink-compatible system.
    for i, recommendation in enumerate(_as_list(plan.get("evidence_recommendations"))):
        if not isinstance(recommendation, dict):
            continue
        if recommendation.get("feature") == "knowledge_level":
            for j, value in enumerate(_as_list(recommendation.get("preferred_values"))):
                if isinstance(value, str) and value not in v.knowledge_level_values:
                    errors.append(_err(
                        f"evidence_recommendations/{i}/preferred_values/{j}",
                        f"'{value}' is not a Biolink knowledge level",
                    ))

    # evidence_policy is reserved for user-requested hard filters. Require at
    # least one actual filter in addition to its origin/application metadata.
    evidence_policy = plan.get("evidence_policy")
    if isinstance(evidence_policy, dict):
        policy_metadata = {"origin", "application", "rationale"}
        if not (set(evidence_policy) - policy_metadata):
            errors.append(_err(
                "evidence_policy",
                "evidence_policy must contain at least one user-requested hard filter",
            ))

    path_ids: set[str] = set()
    paths_by_id: Dict[str, Dict[str, Any]] = {}
    disabled_path_ids: set[str] = set()
    fixed_endpoint_hops: Dict[tuple[str, str], str] = {}
    # Discovery paths
    for i, path in enumerate(_as_list(plan.get("paths"))):
        if not isinstance(path, dict) or "path_id" not in path:
            continue  # schema layer will report this
        pid = path["path_id"]
        if isinstance(pid, str):
            if pid in path_ids:
                errors.append(_err(
                    f"paths/{i}/path_id",
                    f"duplicate path_id '{pid}'",
                ))
            path_ids.add(pid)
            paths_by_id.setdefault(pid, path)
            if path.get("disabled", False) is True:
                disabled_path_ids.add(pid)

        # archetype_tag on path
        atag = path.get("archetype_tag")
        if isinstance(atag, str) and atag not in valid_archetypes:
            errors.append(_err(
                f"paths/{i}/archetype_tag",
                f"archetype tag '{atag}' not in catalog",
            ))
        is_signature_reversal_path = (
            atag == "Q3_signature_reversal"
            or "Q3_signature_reversal" in interpretation_archetypes
        )

        refs_in_hops: set[str] = set()
        adjacency: Dict[str, set[str]] = {}
        for j, hop in enumerate(_as_list(path.get("hops"))):
            if not isinstance(hop, dict):
                continue
            # Predicate validity
            pred = hop.get("predicate")
            pred_is_valid = _validate_predicate_value(
                errors,
                v,
                f"paths/{i}/hops/{j}/predicate",
                pred,
            )
            _validate_hop_qualifiers(
                errors,
                v,
                pred,
                hop.get("qualifiers"),
                f"paths/{i}/hops/{j}/qualifiers",
            )
            if is_signature_reversal_path:
                _validate_signature_reversal_hop(
                    errors,
                    entities,
                    hop,
                    f"paths/{i}/hops/{j}",
                )
            # Endpoint refs
            for role in ("subject_ref", "object_ref"):
                ref = hop.get(role)
                if isinstance(ref, str) and ref not in entity_refs:
                    errors.append(_err(
                        f"paths/{i}/hops/{j}/{role}",
                        f"unknown entity_ref '{ref}'",
                    ))
                if isinstance(ref, str):
                    refs_in_hops.add(ref)

            subject_ref = hop.get("subject_ref")
            object_ref = hop.get("object_ref")
            if isinstance(subject_ref, str) and isinstance(object_ref, str):
                if subject_ref == object_ref:
                    errors.append(_err(
                        f"paths/{i}/hops/{j}",
                        "self-loop hops are unsupported: subject_ref and "
                        "object_ref must identify different entities",
                    ))
                adjacency.setdefault(subject_ref, set()).add(object_ref)
                adjacency.setdefault(object_ref, set()).add(subject_ref)

                # A predicate between two pinned endpoints may be the user's
                # intended relationship, but it can still be unsupported or
                # absent in the selected KG. Require an independent open-ended
                # explanation query over the same endpoints as a fallback.
                subject = entities.get(subject_ref)
                object_ = entities.get(object_ref)
                if (
                    path.get("disabled", False) is not True
                    and subject_ref != object_ref
                    and subject is not None
                    and object_ is not None
                    and subject.get("is_variable", False) is not True
                    and object_.get("is_variable", False) is not True
                ):
                    pair = tuple(sorted((subject_ref, object_ref)))
                    fixed_endpoint_hops.setdefault(
                        pair,
                        f"paths/{i}/hops/{j}",
                    )

            # Predicate direction against inherited Biolink domain/range metadata.
            subject = entities.get(subject_ref) if isinstance(subject_ref, str) else None
            object_ = entities.get(object_ref) if isinstance(object_ref, str) else None
            subject_cat = subject.get("biolink_category") if subject else None
            object_cat = object_.get("biolink_category") if object_ else None
            if (
                pred_is_valid
                and isinstance(subject_cat, str)
                and isinstance(object_cat, str)
                and v.is_valid_category(subject_cat)
                and v.is_valid_category(object_cat)
                and not _direction_matches(v, pred, subject_cat, object_cat)
            ):
                reverse_ok = _direction_matches(v, pred, object_cat, subject_cat)
                if pred in v.symmetric_predicates and reverse_ok:
                    continue
                domain = v.predicate_domains.get(pred) or "any category"
                range_ = v.predicate_ranges.get(pred) or "any category"
                message = (
                    f"predicate '{pred}' expects {domain} -> {range_}, "
                    f"but this hop is {subject_cat} -> {object_cat}"
                )
                inverse = v.predicate_inverses.get(pred)
                if inverse and _direction_matches(v, inverse, subject_cat, object_cat):
                    message += f"; use inverse predicate '{inverse}' for this direction"
                elif reverse_ok:
                    message += "; swap subject_ref and object_ref"
                errors.append(_err(
                    f"paths/{i}/hops/{j}/predicate",
                    message,
                ))

        # All hops in one path must form one connected component.
        if len(refs_in_hops) > 1:
            start = min(refs_in_hops)
            visited: set[str] = set()
            pending = [start]
            while pending:
                current = pending.pop()
                if current in visited:
                    continue
                visited.add(current)
                pending.extend(adjacency.get(current, set()) - visited)
            disconnected = sorted(refs_in_hops - visited)
            if disconnected:
                errors.append(_err(
                    f"paths/{i}/hops",
                    "path contains disconnected hop fragments; "
                    f"not connected to '{start}': {disconnected}",
                ))

        # A discovery pattern made entirely of variables is not grounded in the
        # user's question or an upstream fixed input. Check this per enabled
        # path so one anchored path cannot hide an unconstrained sibling path.
        if path.get("disabled", False) is not True:
            fixed_anchor_refs = sorted(
                ref for ref in refs_in_hops
                if ref in entities
                and entities[ref].get("is_variable") is False
            )
            if not fixed_anchor_refs:
                errors.append(_err(
                    f"paths/{i}/hops",
                    "every enabled discovery path must contain at least one "
                    "non-variable entity as a fixed query anchor",
                ))

        # return_entity_ref reachability
        ret = path.get("return_entity_ref")
        if isinstance(ret, str) and ret not in entity_refs:
            errors.append(_err(
                f"paths/{i}/return_entity_ref",
                f"unknown entity_ref '{ret}'",
            ))
        elif isinstance(ret, str) and ret not in refs_in_hops:
            errors.append(_err(
                f"paths/{i}/return_entity_ref",
                f"'{ret}' is not used in any hop of this path",
            ))
        elif (
            isinstance(mode, str)
            and mode in {"discovery", "hybrid"}
            and isinstance(ret, str)
            and ret in entities
            and entities[ret].get("is_variable") is not True
        ):
            errors.append(_err(
                f"paths/{i}/return_entity_ref",
                f"discovery answer entity '{ret}' must have is_variable=true",
            ))

        expected_category = path.get("expected_result_category")
        expected_category_is_valid = (
            isinstance(expected_category, str)
            and v.is_valid_category(expected_category)
        )
        if isinstance(expected_category, str) and not expected_category_is_valid:
            errors.append(_err(
                f"paths/{i}/expected_result_category",
                f"category '{expected_category}' is not a valid Biolink "
                f"category (v{v.biolink_version})",
            ))
        if (
            expected_category_is_valid
            and isinstance(ret, str)
            and ret in entities
        ):
            return_category = entities[ret].get("biolink_category")
            if (
                isinstance(return_category, str)
                and v.is_valid_category(return_category)
                and not v.category_satisfies(
                    return_category,
                    expected_category,
                )
            ):
                errors.append(_err(
                    f"paths/{i}/expected_result_category",
                    f"expected category '{expected_category}' is incompatible "
                    f"with return entity '{ret}' category '{return_category}'; "
                    "the return category must equal or descend from the "
                    "expected category in Biolink",
                ))

    # Explanation queries
    query_ids: set[str] = set()
    open_fixed_endpoint_explanations: set[tuple[str, str]] = set()
    from_discovery_path_coverage: set[str] = set()
    has_from_discovery_binding = False
    for i, eq in enumerate(_as_list(plan.get("explanation_queries"))):
        if not isinstance(eq, dict) or "query_id" not in eq:
            continue
        query_id = eq.get("query_id")
        if isinstance(query_id, str):
            if query_id in query_ids:
                errors.append(_err(
                    f"explanation_queries/{i}/query_id",
                    f"duplicate query_id '{query_id}'",
                ))
            query_ids.add(query_id)
        atag = eq.get("archetype_tag")
        if isinstance(atag, str) and atag not in valid_archetypes:
            errors.append(_err(
                f"explanation_queries/{i}/archetype_tag",
                f"archetype tag '{atag}' not in catalog",
            ))
        direct_endpoint_refs: Dict[str, str] = {}
        direct_binding_refs: Dict[str, str] = {}
        discovery_endpoint_sources: Dict[str, set[str]] = {}
        for side_name in ("endpoint_a", "endpoint_b"):
            b = eq.get(side_name)
            if not isinstance(b, dict) or "binding_type" not in b:
                continue  # schema layer will report this
            if b["binding_type"] == "entity":
                ref = b.get("entity_ref")
                if ref is None:
                    continue  # schema layer will report the missing field
                if not isinstance(ref, str):
                    continue  # schema layer reports the wrong type
                direct_binding_refs[side_name] = ref
                if ref not in entity_refs:
                    errors.append(_err(
                        f"explanation_queries/{i}/{side_name}/entity_ref",
                        f"unknown entity_ref '{ref}'",
                    ))
                else:
                    ent = entities[ref]
                    direct_endpoint_refs[side_name] = ref
                    if ent.get("is_variable", False):
                        errors.append(_err(
                            f"explanation_queries/{i}/{side_name}/entity_ref",
                            f"endpoint entity '{ref}' is variable; the "
                            "two-endpoint path tool requires resolved specific entities",
                        ))
            elif b["binding_type"] == "from_discovery":
                has_from_discovery_binding = True
                if mode != "hybrid":
                    errors.append(_err(
                        f"explanation_queries/{i}/{side_name}",
                        f"from_discovery binding only valid in hybrid mode (got '{mode}')",
                    ))
                from_path_ids = b.get("from_path_ids")
                if from_path_ids is None:
                    continue  # schema layer will report the missing field
                if not isinstance(from_path_ids, list):
                    continue  # schema layer reports the wrong type
                fanout_top_k = b.get("fanout_top_k")
                if (
                    explanation_influence in {"annotate_only", "rerank"}
                    and isinstance(discovery_top_k, int)
                    and isinstance(fanout_top_k, int)
                    and fanout_top_k < discovery_top_k
                ):
                    purpose = (
                        "every returned candidate can receive its requested "
                        "explanation"
                        if explanation_influence == "annotate_only"
                        else "every preliminary candidate is explained before "
                        "final reranking"
                    )
                    errors.append(_err(
                        f"explanation_queries/{i}/{side_name}/fanout_top_k",
                        f"fanout_top_k ({fanout_top_k}) must be at least "
                        "candidate_ranking.discovery.top_k "
                        f"({discovery_top_k}) so {purpose}",
                    ))
                string_path_ids = [
                    path_id for path_id in from_path_ids
                    if isinstance(path_id, str)
                ]
                discovery_endpoint_sources[side_name] = set(string_path_ids)
                source_categories: Dict[str, str] = {}
                for j, from_path_id in enumerate(string_path_ids):
                    location = (
                        f"explanation_queries/{i}/{side_name}/"
                        f"from_path_ids/{j}"
                    )
                    if from_path_id not in path_ids:
                        errors.append(_err(
                            location,
                            f"unknown from_path_id '{from_path_id}'",
                        ))
                    elif from_path_id in disabled_path_ids:
                        errors.append(_err(
                            location,
                            f"from_path_id '{from_path_id}' references a disabled path",
                        ))
                    else:
                        from_discovery_path_coverage.add(from_path_id)
                        source_path = paths_by_id.get(from_path_id, {})
                        return_ref = source_path.get("return_entity_ref")
                        return_entity = entities.get(return_ref)
                        category = (
                            return_entity.get("biolink_category")
                            if isinstance(return_entity, dict)
                            else None
                        )
                        if isinstance(category, str) and v.is_valid_category(category):
                            source_categories[from_path_id] = category

                if (
                    len(source_categories) > 1
                    and not _return_categories_are_compatible(
                        v,
                        list(source_categories.values()),
                    )
                ):
                    details = ", ".join(
                        f"{path_id}={category}"
                        for path_id, category in sorted(source_categories.items())
                    )
                    errors.append(_err(
                        f"explanation_queries/{i}/{side_name}/from_path_ids",
                        "from_path_ids select incompatible return-entity "
                        f"categories ({details}); use separate explanation queries",
                    ))

        same_direct_entity = (
            set(direct_binding_refs) == {"endpoint_a", "endpoint_b"}
            and direct_binding_refs["endpoint_a"]
            == direct_binding_refs["endpoint_b"]
        )
        shared_discovery_sources = (
            set(discovery_endpoint_sources) == {"endpoint_a", "endpoint_b"}
            and bool(
                discovery_endpoint_sources["endpoint_a"]
                & discovery_endpoint_sources["endpoint_b"]
            )
        )
        if same_direct_entity or shared_discovery_sources:
            errors.append(_err(
                f"explanation_queries/{i}/endpoint_b",
                "self-loop explanation queries are unsupported: endpoint_a "
                "and endpoint_b must identify different entities and use "
                "non-overlapping discovery sources",
            ))

        # Every middle-node constraint must use the loaded Biolink vocabulary;
        # backend-local category extensions are outside this planner contract.
        for category_field in (
            "middle_category_whitelist",
            "middle_category_blacklist",
        ):
            for j, cat in enumerate(_as_list(eq.get(category_field))):
                if isinstance(cat, str) and not v.is_valid_category(cat):
                    errors.append(_err(
                        f"explanation_queries/{i}/{category_field}/{j}",
                        f"category '{cat}' is not a valid Biolink category "
                        f"(v{v.biolink_version})",
                    ))

        # Predicate whitelist/blacklist values in post_filter
        pf = _as_dict(eq.get("post_filter"))
        for key in ("predicate_whitelist", "predicate_blacklist",
                    "required_predicates_anywhere"):
            for j, p in enumerate(_as_list(pf.get(key))):
                _validate_predicate_value(
                    errors,
                    v,
                    f"explanation_queries/{i}/post_filter/{key}/{j}",
                    p,
                )

        for key in (
            "required_qualifier_on_any_edge",
            "forbidden_qualifier_on_any_edge",
        ):
            _validate_qualified_predicate(
                errors,
                v,
                pf.get(key),
                f"explanation_queries/{i}/post_filter/{key}",
            )

        # A fixed-endpoint fallback must stay open to predicates other than the
        # one used in the primary hop. Negative filters may still be used, but a
        # whitelist or required-predicate constraint would defeat the fallback.
        has_positive_predicate_constraint = bool(
            _as_list(pf.get("predicate_whitelist"))
            or _as_list(pf.get("required_predicates_anywhere"))
        )
        if (
            set(direct_endpoint_refs) == {"endpoint_a", "endpoint_b"}
            and not has_positive_predicate_constraint
        ):
            pair = tuple(sorted((
                direct_endpoint_refs["endpoint_a"],
                direct_endpoint_refs["endpoint_b"],
            )))
            open_fixed_endpoint_explanations.add(pair)

    aggregation = _as_dict(plan.get("aggregation"))
    attach_explanations = (
        aggregation.get("attach_explanations_to_candidates") is True
    )

    if mode == "hybrid":
        if explanation_influence == "none":
            if has_from_discovery_binding:
                errors.append(_err(
                    "ranking/candidate_ranking/explanation_influence",
                    "hybrid candidate-specific explanations require "
                    "explanation_influence='annotate_only' or 'rerank'; use "
                    "'none' only for independent fixed-endpoint context",
                ))
            if attach_explanations:
                errors.append(_err(
                    "aggregation/attach_explanations_to_candidates",
                    "independent explanations with explanation_influence='none' "
                    "must not be attached to candidates",
                ))
        elif explanation_influence in {"annotate_only", "rerank"}:
            if not has_from_discovery_binding:
                errors.append(_err(
                    "ranking/candidate_ranking/explanation_influence",
                    f"explanation_influence='{explanation_influence}' requires "
                    "at least one from_discovery endpoint binding",
                ))
            if not attach_explanations:
                errors.append(_err(
                    "aggregation/attach_explanations_to_candidates",
                    f"explanation_influence='{explanation_influence}' requires "
                    "attach_explanations_to_candidates=true",
                ))

    if attach_explanations and not (
        mode == "hybrid" and has_from_discovery_binding
    ):
        errors.append(_err(
            "aggregation/attach_explanations_to_candidates",
            "candidate explanations can be attached only in hybrid mode with "
            "a from_discovery endpoint binding",
        ))

    if (
        mode == "hybrid"
        and attach_explanations
    ):
        enabled_path_ids = path_ids - disabled_path_ids
        missing_explanation_sources = sorted(
            enabled_path_ids - from_discovery_path_coverage
        )
        if missing_explanation_sources:
            errors.append(_err(
                "aggregation/attach_explanations_to_candidates",
                "candidate explanations do not cover enabled discovery paths: "
                f"{missing_explanation_sources}",
            ))

    if fixed_endpoint_hops and mode != "hybrid":
        errors.append(_err(
            "plan_mode",
            "an active predicate hop between two fixed entities requires "
            "plan_mode='hybrid' so the plan can include its explanation fallback",
        ))

    for pair, location in fixed_endpoint_hops.items():
        if pair not in open_fixed_endpoint_explanations:
            errors.append(_err(
                location,
                "active predicate hop joins two non-variable entities "
                f"'{pair[0]}' and '{pair[1]}'; add an independent, "
                "predicate-unconstrained explanation_query for these endpoints. "
                "If the user did not state or imply this predicate, remove the "
                "hop and use only the explanation query instead",
            ))

    if not is_refused:
        errors.extend(_check_entities_are_used(plan, entities))

    return errors


def _referenced_entity_refs(plan: Dict[str, Any]) -> set:
    """Every entity_ref named anywhere in the plan except the entities block.

    Collected by walking the structure for keys ending in `_ref` rather than by
    visiting each site that can hold one. The list of such sites is long and
    grows — hop endpoints, return entities, explanation endpoint bindings,
    from_discovery bindings, ranking scopes — and a check that has to be
    extended every time the contract grows a field is a check that will
    eventually be wrong in the direction of passing.
    """
    found: set = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if isinstance(value, str) and key.endswith("_ref"):
                    found.add(value)
                else:
                    walk(value)
        elif isinstance(node, list):
            for item in node:
                walk(item)

    walk({k: v for k, v in plan.items() if k != "entities"})
    return found


def _check_entities_are_used(
    plan: Dict[str, Any], entities: Dict[str, Dict[str, Any]],
) -> List[ValidationError]:
    """An entity the plan declares and never queries.

    The schema checks that every entity_ref used exists. Nothing checked the
    other direction, and the other direction is where a plan quietly stops
    answering the question it was given.

    Seen live. Asked "which drugs treat dermatitis herpetiformis by targeting
    CFTR?", the planner declared all three concepts and then wrote a single hop
    from the drug to CFTR. `dermatitis herpetiformis` appeared in the entity
    list and in no path. The plan validated, executed, and returned twenty
    well-evidenced drugs that affect CFTR — every claim about them grounded in
    a real edge, and none of them an answer to the question asked. A grounding
    gate cannot catch that, because nothing in the answer is ungrounded; the
    answer is simply to a narrower question.

    Only non-variable entities are checked. A variable is the shape of a thing
    the plan is looking for, and one that nothing points at is inert; a fixed
    entity is a concept the user named, and dropping it changes the question.
    """
    referenced = _referenced_entity_refs(plan)
    errors: List[ValidationError] = []

    for i, ent in enumerate(_as_list(plan.get("entities"))):
        if not isinstance(ent, dict):
            continue
        ref = ent.get("entity_ref")
        if not isinstance(ref, str) or ref in referenced:
            continue
        if ent.get("is_variable"):
            continue

        name = ent.get("name") or ref
        role = ent.get("query_role") or "queried"

        if role == "context":
            # A declared omission, which is a different thing from a silent
            # one. Q3_signature_reversal is the honest case: the query runs
            # over the expression signature's genes, and the disease names
            # where that signature came from. The note is required because the
            # marker is only worth anything if setting it is a decision — an
            # unexplained 'context' is how this check gets neutralised.
            if not str(ent.get("notes") or "").strip():
                errors.append(_err(
                    f"entities/{i}/notes",
                    f"entity '{ref}' ({name}) is marked query_role='context', which "
                    f"exempts it from being queried; say in `notes` why the "
                    f"question does not require querying it",
                ))
            continue

        errors.append(_err(
            f"entities/{i}/entity_ref",
            f"entity '{ref}' ({name}) is declared but never used: no hop, "
            f"explanation query or binding references it. A fixed entity names "
            f"a concept from the question, so a plan that does not query it is "
            f"answering something narrower than what was asked. Add the hop or "
            f"explanation query that uses it; if the question genuinely does "
            f"not require querying it, set query_role='context' on the entity and "
            f"say why in its `notes`",
        ))

    return errors


# ------------------------------------------------------------------------
# Combined entry point
# ------------------------------------------------------------------------

def validate_plan(
    plan: Dict[str, Any],
    vocab: Optional[BiolinkVocabulary] = None,
) -> ValidationResult:
    """Run schema then semantic validation. Returns combined result."""
    errors = validate_schema(plan)
    # Only run semantic checks if the schema check passed enough to make sense.
    # If plan_mode is missing, semantic checks would crash — skip.
    if isinstance(plan, dict) and "plan_mode" in plan:
        errors.extend(validate_semantics(plan, vocab=vocab))
    return ValidationResult(ok=(len(errors) == 0), errors=errors)


if __name__ == "__main__":
    import glob
    import json

    for ex_path in sorted(glob.glob("/home/claude/plan_schema/example_*.json")):
        with open(ex_path) as f:
            plan = json.load(f)
        result = validate_plan(plan)
        print(f"{'[OK]  ' if result.ok else '[FAIL]'} {ex_path}")
        if not result.ok:
            for e in result.errors[:5]:
                print(f"   {e.kind}: {e.location}: {e.message}")
