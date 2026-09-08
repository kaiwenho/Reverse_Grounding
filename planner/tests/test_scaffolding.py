"""
Unit tests for the planner scaffolding.

These tests verify:

- The Biolink vocabulary loads and reports expected counts.
- The compiled JSON Schema has Biolink enums injected.
- All bundled example plans validate.
- Pydantic models round-trip through JSON without loss.
- The planner agent handles: happy path, JSON parse failure with retry,
  validation failure with retry, and refusal on repeated failure.
- The archetype catalog and the schema archetype_tag enum agree.
"""

import json
import glob
import copy
import tempfile
import pytest
from pydantic import ValidationError as PydanticValidationError

from plan_core import (
    PLAN_VERSION, QueryPlan, compile_schema, load_biolink_vocabulary,
    validate_plan,
)
from plan_core.archetype_catalog import archetype_tags, load_archetype_catalog
from planner_agent.planner_agent import (
    EchoJSONClient,
    PlannerAgent,
    _build_retry_message,
)

from pathlib import Path
EXAMPLE_DIR = Path(__file__).parent / "fixtures"
EXAMPLE_GLOB = str(EXAMPLE_DIR / "example_*.json")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
PLANNER_ARTIFACT_ROOTS = (
    PROJECT_ROOT / "plan-core",
    PROJECT_ROOT / "planner",
)


# ---------------------------------------------------------------------
# Biolink vocab
# ---------------------------------------------------------------------

def test_biolink_vocab_loads():
    v = load_biolink_vocabulary()
    assert v.biolink_version.startswith("4.")
    assert "biolink:treats" in v.predicate_curies
    assert "biolink:affects" in v.predicate_curies
    assert "biolink:directly_physically_interacts_with" in v.predicate_curies
    assert "biolink:gene_associated_with_condition" in v.predicate_curies
    assert "SmallMolecule" in v.categories
    assert "Gene" in v.categories
    assert "Disease" in v.categories
    assert "decreased" in v.direction_qualifier_values
    assert "inhibition" in v.causal_mechanism_qualifier_values
    assert "activity" in v.aspect_qualifier_values


def test_biolink_vocab_rejects_unknown():
    v = load_biolink_vocabulary()
    assert not v.is_valid_predicate("biolink:not_a_thing")
    assert not v.is_valid_category("NotACategory")


def test_biolink_vocab_excludes_deprecated_predicates():
    v = load_biolink_vocabulary()
    assert v.deprecated_predicate_curies
    assert v.predicate_curies.isdisjoint(v.deprecated_predicate_curies)
    for predicate in v.deprecated_predicate_curies:
        assert v.is_deprecated_predicate(predicate)
        assert not v.is_valid_predicate(predicate)


def test_planner_artifacts_do_not_name_deprecated_predicates():
    """A Biolink upgrade cannot silently reintroduce one into planner code."""
    deprecated = load_biolink_vocabulary().deprecated_predicate_curies
    checked_suffixes = {".json", ".md", ".py", ".toml"}
    ignored_parts = {
        ".git", ".pytest_cache", "__pycache__", "raw_responses",
    }
    occurrences = []

    # The Executor is a separate consumer with its own migration history and
    # tests. In particular, archived execution plans and run output must not
    # determine whether the planner itself passes its vocabulary checks.
    for artifact_root in PLANNER_ARTIFACT_ROOTS:
        for path in artifact_root.rglob("*"):
            if (
                not path.is_file()
                or path.suffix not in checked_suffixes
                or ignored_parts.intersection(path.parts)
            ):
                continue
            text = path.read_text(errors="ignore")
            for predicate in deprecated:
                if predicate in text:
                    occurrences.append((path.relative_to(PROJECT_ROOT), predicate))

    assert not occurrences, "Deprecated predicates found in planner artifacts: " + "; ".join(
        f"{path}: {predicate}" for path, predicate in occurrences
    )


def test_biolink_vocab_carries_direction_metadata():
    v = load_biolink_vocabulary()
    predicate = "biolink:gene_associated_with_condition"
    assert v.predicate_domains[predicate] == "Gene"
    assert v.predicate_ranges[predicate] == "DiseaseOrPhenotypicFeature"
    assert v.predicate_inverses[predicate] == "biolink:condition_associated_with_gene"
    assert v.category_satisfies("Disease", "DiseaseOrPhenotypicFeature")
    assert v.category_satisfies("SmallMolecule", "ChemicalOrDrugOrTreatment")
    assert "biolink:treats" not in v.symmetric_predicates


def test_biolink_vocab_expands_mixin_to_concrete_implementers():
    v = load_biolink_vocabulary()
    implementers = v.concrete_implementers("GeneOrGeneProduct")

    assert "GeneOrGeneProduct" in v.mixin_categories
    assert {"Gene", "Protein"}.issubset(implementers)
    assert implementers == v.concrete_implementers(
        "biolink:GeneOrGeneProduct"
    )
    assert implementers.isdisjoint(v.mixin_categories)
    assert implementers.isdisjoint(v.abstract_categories)
    assert implementers.isdisjoint(v.deprecated_categories)
    assert v.category_satisfies("GeneOrGeneProduct", "NamedThing")


def test_biolink_vocab_resolves_qualifier_predicate_families():
    v = load_biolink_vocabulary()
    assert v.qualifier_predicate_family("biolink:regulates") == "biolink:regulates"
    assert v.qualifier_predicate_family("biolink:affects") == "biolink:affects"
    assert (
        v.qualifier_predicate_family("biolink:directly_physically_interacts_with")
        == "biolink:interacts_with"
    )
    assert (
        v.qualifier_predicate_family("biolink:has_adverse_event")
        == "biolink:affects"
    )
    assert v.qualifier_predicate_family("biolink:treats") is None

def test_catalog_path_template_categories_are_biolink_valid():
    """
    Categories referenced in path templates must exist in Biolink after
    stripping placeholder suffixes (e.g. `Drug_1` -> `Drug`, `Disease_X` -> `Disease`).
    Biolink category names are CamelCase with no underscores, so anything
    from the first underscore onward is a placeholder suffix by convention.
    """
    import re
    cat = load_archetype_catalog()
    vocab = load_biolink_vocabulary()

    # Match a capitalized identifier optionally followed by a placeholder suffix.
    # `Disease_X`, `Gene_1`, `Drug_comed`, or plain `SmallMolecule` all match.
    token_re = re.compile(r"\b([A-Z][A-Za-z]+)(?:_[A-Za-z0-9]+)?\b")

    bad = []
    for a in cat["archetypes"]:
        for p in a.get("biolink_paths", []) or []:
            template = p.get("template", "")
            for base in token_re.findall(template):
                if not vocab.is_valid_category(base):
                    bad.append((a["tag"], p["label"], base))
    assert not bad, "Invalid categories in path templates: " + "; ".join(
        f"{tag}/{label}: {base}" for tag, label, base in bad
    )


# ---------------------------------------------------------------------
# Compiled schema
# ---------------------------------------------------------------------

def test_compiled_schema_has_biolink_enums():
    schema = compile_schema()
    defs = schema["$defs"]
    entity_cats = defs["Entity"]["properties"]["biolink_category"]["enum"]
    aspect_vals = schema["$defs"]["Qualifiers"]["properties"]["object_aspect_qualifier"]["enum"]
    # Biolink v4.4.x has hundreds of categories and dozens of aspects
    assert len(entity_cats) > 100
    assert len(aspect_vals) > 20
    assert "SmallMolecule" in entity_cats
    assert "activity" in aspect_vals
    category_ref = "#/$defs/Entity/properties/biolink_category"
    assert (
        defs["Path"]["properties"]["expected_result_category"]["$ref"]
        == category_ref
    )
    explanation_properties = defs["ExplanationQuery"]["properties"]
    assert (
        explanation_properties["middle_category_whitelist"]["items"]["$ref"]
        == category_ref
    )
    assert (
        explanation_properties["middle_category_blacklist"]["items"]["$ref"]
        == category_ref
    )


def test_v010_contract_fields_ranking_inputs_and_evidence_guidance():
    schema = compile_schema()
    defs = schema["$defs"]
    entity_properties = defs["Entity"]["properties"]
    assert "taxa" in entity_properties
    assert "input_binding" in entity_properties
    assert "role" not in entity_properties
    assert "is_variable" in defs["Entity"]["required"]
    assert defs["ExplanationQuery"]["properties"]["max_hops"]["maximum"] == 5
    assert defs["Path"]["properties"]["hops"]["maxItems"] == 5
    external_input = next(
        branch for branch in defs["EntityInputBinding"]["oneOf"]
        if branch["properties"]["binding_type"]["const"] == "external_input"
    )
    assert "direction_filter" in external_input["properties"]
    assert set(external_input["properties"]["direction_filter"]["enum"]) == set(
        load_biolink_vocabulary().direction_qualifier_values
    )
    assert "min_evidence_strength" not in defs["Path"]["properties"]
    assert "min_evidence_strength_per_edge" not in defs["ExplanationPostFilter"]["properties"]
    assert "knowledge_level_per_edge" not in defs["ExplanationPostFilter"]["properties"]
    assert "min_evidence_strength" not in defs["EvidencePolicy"]["properties"]
    assert "knowledge_level" not in defs["EvidencePolicy"]["properties"]
    assert "require_primary_source" not in defs["EvidencePolicy"]["properties"]
    assert "require_primary_knowledge_source" in defs["EvidencePolicy"]["properties"]
    assert set(defs["EvidencePolicy"]["required"]) == {
        "origin", "application", "rationale",
    }
    criterion_schema = defs["RankingCriterion"]
    assert set(criterion_schema["required"]) == {
        "name", "direction", "origin", "application", "scope", "rationale",
    }
    ranking_names = criterion_schema["properties"]["name"]["enum"]
    assert "knowledge_level" in ranking_names
    assert set(defs["RankingSpec"]["required"]) == {
        "strategy", "criteria", "top_k",
    }
    assert set(defs["RankingPlan"]["properties"]) == {
        "candidate_ranking", "explanation_ranking",
    }
    assert set(defs["CandidateRanking"]["required"]) == {
        "explanation_influence", "rationale", "discovery",
    }
    assert defs["CandidateRanking"]["properties"]["explanation_influence"]["enum"] == [
        "none", "annotate_only", "rerank",
    ]
    assert "EvidenceRecommendation" in defs
    assert "evidence_recommendations" in schema["properties"]
    entity_constraint = defs["EntityConstraint"]
    constraint_properties = entity_constraint["properties"]
    assert set(constraint_properties) == {"field", "op", "value"}
    assert constraint_properties["field"]["const"] == "approval_status"
    assert constraint_properties["op"]["const"] == "eq"
    assert constraint_properties["value"]["enum"] == ["approved", "ever_approved"]
    assert "allOf" not in entity_constraint
    discovery_binding = next(
        binding for binding in defs["EndpointBinding"]["oneOf"]
        if binding["properties"]["binding_type"]["const"] == "from_discovery"
    )
    assert discovery_binding["required"] == [
        "binding_type", "from_path_ids", "fanout_top_k",
    ]
    assert discovery_binding["properties"]["from_path_ids"]["minItems"] == 1
    assert discovery_binding["properties"]["from_path_ids"]["uniqueItems"] is True
    assert "from_path_id" not in discovery_binding["properties"]
    assert "fanout_top_k" in discovery_binding["properties"]
    assert "default" not in discovery_binding["properties"]["fanout_top_k"]
    assert "top_k" not in discovery_binding["properties"]
    assert (
        defs["Aggregation"]["properties"]
        ["attach_explanations_to_candidates"]["default"]
        is False
    )
    assert PLAN_VERSION == "0.10.0"
    assert load_archetype_catalog()["schema_version"] == PLAN_VERSION
    assert schema["$id"].endswith("/v0.10.0.json")
    assert "self-loop" in defs["Hop"]["description"]
    assert "qualifier-bearing hops" in defs["Hop"]["description"]
    assert "self-loop" in defs["ExplanationQuery"]["description"]


# ---------------------------------------------------------------------
# Examples validate
# ---------------------------------------------------------------------

@pytest.mark.parametrize("path", sorted(glob.glob(EXAMPLE_GLOB)))
def test_example_validates(path):
    with open(path) as f:
        plan = json.load(f)
    result = validate_plan(plan)
    assert result.ok, result.format()


# ---------------------------------------------------------------------
# Pydantic round-trip
# ---------------------------------------------------------------------

@pytest.mark.parametrize("path", sorted(glob.glob(EXAMPLE_GLOB)))
def test_pydantic_round_trip(path):
    with open(path) as f:
        raw = json.load(f)
    parsed = QueryPlan.model_validate(raw)
    dumped = parsed.to_json_dict()
    result = validate_plan(dumped)
    assert result.ok, result.format()


# ---------------------------------------------------------------------
# Archetype catalog / schema enum agreement
# ---------------------------------------------------------------------

def test_archetype_catalog_matches_schema_enum():
    catalog_tags = set(archetype_tags())
    schema = compile_schema()
    schema_tags = set(
        schema["properties"]["interpretation"]["properties"]["archetypes"]["items"]["enum"]
    )
    assert catalog_tags == schema_tags, (
        f"Mismatch — in catalog only: {catalog_tags - schema_tags}, "
        f"in schema only: {schema_tags - catalog_tags}"
    )


# ---------------------------------------------------------------------
# Enriched catalog content (v0.2.0 of catalog)
# ---------------------------------------------------------------------

def test_catalog_carries_key_predicates_and_paths_for_repurposing_archetypes():
    """Every archetype except OTHER should have key_predicates and biolink_paths."""
    from plan_core.archetype_catalog import (
        archetype_by_tag,
        load_archetype_catalog,
    )
    cat = load_archetype_catalog()
    for a in cat["archetypes"]:
        if a["tag"] == "OTHER":
            continue
        assert a.get("key_predicates"), f"{a['tag']} missing key_predicates"
        assert a.get("biolink_paths"), f"{a['tag']} missing biolink_paths"
        # every path should have the required structural fields
        for p in a["biolink_paths"]:
            assert "label" in p and "template" in p and "rationale" in p
            assert "hop_count" in p


def test_q1_catalog_separates_known_target_discovery_from_disease_context():
    from plan_core.archetype_catalog import archetype_by_tag

    q1 = archetype_by_tag("Q1_target_based")
    variants = {item["when"]: item for item in q1["planning_variants"]}
    known = variants["Both the target and disease are known"]
    assert "only the drug-target hop" in known["candidate_query"]
    assert "Do not make a fixed target-disease predicate" in known["candidate_query"]
    assert "cannot distinguish or rerank candidate drugs" in known["explanation_query"]
    assert any(
        path["hop_count"] == 1
        and path["template"] == "SmallMolecule -[biolink:affects]-> Gene"
        for path in q1["biolink_paths"]
    )


def test_catalog_key_predicates_are_biolink_valid():
    """All key_predicates and path-template predicates in the catalog must exist in loaded Biolink."""
    import re
    cat = load_archetype_catalog()
    vocab = load_biolink_vocabulary()
    pred_re = re.compile(r"biolink:[a-z_][a-z0-9_]*")
    for a in cat["archetypes"]:
        for p in a.get("key_predicates", []):
            assert vocab.is_valid_predicate(p), (
                f"{a['tag']}: key_predicate '{p}' is not in loaded Biolink"
            )
        for path in a.get("biolink_paths", []) or []:
            for m in pred_re.findall(path.get("template", "")):
                assert vocab.is_valid_predicate(m), (
                    f"{a['tag']}: path '{path.get('label')}' uses '{m}' which is not in loaded Biolink"
                )


def test_catalog_qualifier_templates_use_supported_predicate_families():
    """Every catalog edge carrying `{...}` must use a validated family."""
    import re

    cat = load_archetype_catalog()
    vocab = load_biolink_vocabulary()
    decorated_edge_re = re.compile(
        r"-\[(biolink:[a-z][a-z0-9_]*)\s+\{([^}]*)\}"
    )

    for archetype in cat["archetypes"]:
        for path in archetype.get("biolink_paths", []) or []:
            for predicate, contents in decorated_edge_re.findall(path.get("template", "")):
                if "qualifier" not in contents:
                    continue
                assert vocab.qualifier_predicate_family(predicate) is not None, (
                    f"{archetype['tag']}: path '{path.get('label')}' attaches "
                    f"qualifiers to unsupported predicate '{predicate}'"
                )


def test_catalog_high_priority_predicates_are_biolink_valid():
    """Every predicate in `high_priority_predicates` must exist in loaded Biolink."""
    cat = load_archetype_catalog()
    vocab = load_biolink_vocabulary()
    hpp = cat.get("high_priority_predicates", [])
    assert hpp, "high_priority_predicates is missing or empty"
    for p in hpp:
        assert vocab.is_valid_predicate(p), (
            f"high_priority_predicates: '{p}' is not in loaded Biolink"
        )


def test_catalog_gap_flag_kinds_match_schema_enum():
    """Gap flag `kind` values in the catalog must exist in the Gap.kind enum."""
    schema = compile_schema()
    allowed = set(schema["$defs"]["Gap"]["properties"]["kind"]["enum"])
    cat = load_archetype_catalog()
    for a in cat["archetypes"]:
        for g in a.get("gap_flags", []) or []:
            assert g["kind"] in allowed, (
                f"{a['tag']}: gap kind '{g['kind']}' not in schema enum"
            )

def test_useful_qualifier_values_are_biolink_valid():
    cat = load_archetype_catalog()
    vocab = load_biolink_vocabulary()
    useful = cat.get("useful_qualifier_values", {})
    qualifier_to_enum = {
        "object_aspect_qualifier": vocab.aspect_qualifier_values,
        "subject_aspect_qualifier": vocab.aspect_qualifier_values,
        "object_direction_qualifier": vocab.direction_qualifier_values,
        "subject_direction_qualifier": vocab.direction_qualifier_values,
        "causal_mechanism_qualifier": vocab.causal_mechanism_qualifier_values,
    }
    for qual_name, groups in useful.items():
        if qual_name == "note":
            continue
        allowed = qualifier_to_enum.get(qual_name)
        assert allowed is not None, f"unknown qualifier '{qual_name}'"
        for group_name, values in groups.items():
            for v in values:
                assert v in allowed, f"'{v}' not in Biolink {qual_name} enum"


# ---------------------------------------------------------------------
# Prompt detail levels
# ---------------------------------------------------------------------

def test_prompt_detail_levels_are_monotonic_in_size():
    """slim < standard < full."""
    from plan_core.archetype_catalog import archetype_summary_for_prompt
    slim = archetype_summary_for_prompt("slim")
    standard = archetype_summary_for_prompt("standard")
    full = archetype_summary_for_prompt("full")
    assert len(slim) < len(standard) < len(full)


def test_standard_prompt_includes_key_predicates_and_gaps():
    from plan_core.archetype_catalog import archetype_summary_for_prompt
    text = archetype_summary_for_prompt("standard")
    # Key predicates section shows up for at least Q1
    assert "biolink:gene_associated_with_condition" in text
    # Global qualifier note shows up
    assert "Qualifier" in text
    # At least one gap flag description surfaces (Q3 signature)
    assert "signature" in text.lower()
    assert "Both the target and disease are known" in text
    assert "Do not make a fixed target-disease predicate" in text
    assert "Inline members:" in text
    assert "one fixed Gene entity per named member" in text


def test_full_prompt_includes_worked_paths():
    from plan_core.archetype_catalog import archetype_summary_for_prompt
    text = archetype_summary_for_prompt("full")
    # Path templates use `-[biolink:...]->`; look for a Q1 path signature
    assert "biolink:directly_physically_interacts_with" in text
    assert "hop" in text.lower()


def test_q3_catalog_distinguishes_inline_members_from_external_inputs():
    catalog = load_archetype_catalog()
    q3 = next(
        archetype for archetype in catalog["archetypes"]
        if archetype["tag"] == "Q3_signature_reversal"
    )

    assert "inline_members" in q3["input_contract"]
    assert "one fixed Gene entity per named member" in (
        q3["input_contract"]["inline_members"]
    )
    assert "no input_binding" in q3["input_contract"]["inline_members"]
    assert any("IL4" in question for question in q3["example_questions"])


def test_planner_prompt_states_v010_semantic_rules():
    from planner_agent.prompt_assembly import build_system_prompt
    text = build_system_prompt()
    assert "independently useful to a user" in text
    assert "Do not rely on\nan executor default" in text
    assert "Do NOT emit `plan_version` or `biolink_version`" in text
    assert "orchestrator supplies required `plan_version`" in text
    assert "do not replace it with a" in text
    assert "guessed \"official\" symbol" in text
    assert "taxon-aware resolver" in text
    assert "Variable gene entities do not require an" in text
    assert "canonical Biolink direction" in text
    assert "A hop may carry a\n   qualifier block only when" in text
    assert "`biolink:regulates`" in text
    assert "`biolink:affects`" in text
    assert "`biolink:interacts_with`" in text
    assert "Never attach qualifiers\n   to another predicate" in text
    assert "active, non-deprecated Biolink predicate" in text
    assert "marked deprecated in the loaded Biolink Model" in text
    assert "`Path.expected_result_category`" in text
    assert "`middle_category_whitelist` or `middle_category_blacklist`" in text
    assert "must equal it or be one of its Biolink descendants" in text
    assert "Do NOT emit `Entity.role`" in text
    assert "`Path.return_entity_ref` identifies that path's target" in text
    assert "`endpoint_a` and `endpoint_b` bindings identify its two" in text
    assert "an all-variable path is invalid" in text
    assert "Set `is_variable` explicitly" in text
    assert "NCBITaxon:9606" in text
    assert "Do NOT emit `knowledge_level` as a hard filter" in text
    assert "explicitly requests a minimum publication count" in text
    assert "input-bound entity MUST have `is_variable=false`" in text
    assert "Inline signature members are NOT an external input" in text
    assert "create one fixed Gene" in text
    assert "do not ask for a manifest" in text
    assert "A phrase such\n      as \"using the supplied dataset\" is not proof" in text
    assert "expected_format=\"directional_gene_signature\"" in text
    assert "MUST include\n      `direction_filter`" in text
    assert "source `increased` or `upregulated` genes" in text
    assert "It CANNOT\n      infer missing gene members or directions" in text
    assert "A Path has NO `max_hops` field and NO" in text
    assert "Only an `ExplanationQuery` has `max_hops`" in text
    assert "Never guess its predicate" in text
    assert "emit BOTH the predicate-constrained" in text
    assert "Q1 known-target exception" in text
    assert "do not\n      make a target-disease predicate a mandatory hop" in text
    assert "label disease relevance\n      unverified" in text
    assert "two ordered explanation queries" in text
    assert "predicate-unconstrained explanation fallback" in text
    assert "check whether a phrase names two concepts" in text
    assert "MUST denote one atomic biomedical" in text
    assert "`eczema from gluten allergy`" in text
    assert "`eczema from gluten sensitivity`" in text
    assert "Do not mechanically split on bare `with` or bare `of`" in text
    assert "Every resulting entity must correspond to a concept stated by the user" in text
    assert "Do not\n      infer an unnamed diagnosis, subtype, or relationship" in text
    assert "Decomposition does not authorize inventing a" in text
    assert "Exactly one fixed discovery path" in text
    assert "stage-specific `ranking` object" in text
    assert "`ranking.candidate_ranking`" in text
    assert "`ranking.explanation_ranking`" in text
    assert "`annotate_only`" in text
    assert "`rerank`" in text
    assert "cannot distinguish drugs" in text
    assert "scope=\"executor_specific\"" in text
    assert "profile_id=\"executor_epc_v1\"" in text
    assert "evidence_recommendations" in text
    assert "primary_knowledge_source" in text
    assert "NOT primary literature" in text
    assert "origin=\"user_requested\"" in text
    assert "application=\"filter\"" in text
    assert "When the user asks for approved drugs" in text
    assert "{\"field\":\"approval_status\", \"op\":\"eq\", \"value\":\"approved\"}" in text
    assert "Use `ever_approved`" in text
    assert "empty result rather than silently relaxing it" in text
    assert "the ONLY EntityConstraint" in text
    assert "detailed Biolink\n    ApprovalStatusEnum value" in text
    assert "`chembl_availability_type`" in text
    assert "Do not emit self-loop queries" in text
    assert "uses a non-empty `from_path_ids` list" in text
    assert "never a singular `from_path_id`" in text
    assert "binding's\n    `fanout_top_k`" in text
    assert "greater than\n    or equal to `candidate_ranking.discovery.top_k`" in text
    assert "Preserve\n    discovery order for `annotate_only`" in text
    assert "apply `candidate_ranking.final` for\n    `rerank`" in text
    assert "MUST NOT contain\n      `num_explanation_paths`" in text
    assert "collectively cover every enabled\n    discovery path" in text
    assert "true ONLY for candidate fan-out" in text
    assert "it defaults to false" in text
    assert "requires_capability_not_available" in text
    assert "ground to the same CURIE" in text
    assert "user_requested_ranking_override:" in text
    assert "Do not add clinical-ranking" in text
    assert "deterministically expands" in text
    assert "MUST include `entities: []`" in text
    assert "`unsafe_or_clinical_advice`" in text
    assert "never emit JSON `null`" in text


def test_planner_prompt_includes_compact_canonical_shapes():
    from planner_agent.prompt_assembly import build_system_prompt

    text = build_system_prompt()
    assert "## Canonical JSON shapes" in text
    assert "### Entity constraint and discovery Path" in text
    assert '"constraints": [' in text
    assert "There is no root\n`entity_constraints`" in text
    assert "does not have `max_hops` or `constraints`" in text
    assert "### Directional signature partitions" in text
    assert '"direction_filter": "increased"' in text
    assert "### Inline directional signature members" in text
    assert '"name": "IL4"' in text
    assert "they do not use\n`input_binding`" in text
    assert '"candidate_ranking": {' in text
    assert '"discovery": {' in text
    assert '"strategy": "evidence_weighted"' in text
    assert '"preferred_values": [' in text
    assert '"return": {' in text
    assert '"top_k_paths": 5' in text
    assert "Do not put `top_k_paths` directly on an" in text
    assert '"from_path_ids": ["P1", "P2"]' in text
    assert '"fanout_top_k": 20' in text
    assert "### Refusal" in text
    assert '"entities": []' in text
    assert '"reason": "unsafe_or_clinical_advice"' in text
    assert "Explain why explanations do not affect candidate order." not in text
    assert "This discovery-only plan has no explanation queries" in text


def test_prompt_hides_runtime_versions_from_exemplars():
    from planner_agent.prompt_assembly import build_system_prompt

    exemplar = json.loads(_load_example("example_q1_target_based.json"))
    exemplar["plan_version"] = "9.9.9"
    exemplar["biolink_version"] = "8.8.8"
    text = build_system_prompt(exemplar_plans=[exemplar])

    assert "9.9.9" not in text
    assert "8.8.8" not in text


def test_agent_accepts_archetype_detail():
    from planner_agent.planner_agent import PlannerAgent, EchoJSONClient
    canned = _load_example("example_q1_target_based.json")
    a = PlannerAgent(llm=EchoJSONClient(canned), archetype_detail="full")
    # Prompt should contain worked-path content from the catalog
    assert "biolink:directly_physically_interacts_with" in a.system_prompt


# ---------------------------------------------------------------------
# Planner agent — happy path, retry paths, refusal
# ---------------------------------------------------------------------
def _load_example(name: str) -> str:
    with open(EXAMPLE_DIR / name) as f:
        return f.read()


def _criterion(
    name: str,
    direction: str,
    weight: float | None = None,
    *,
    origin: str = "planner_recommended",
    scope: str = "portable",
    rationale: str = "Test ranking recommendation.",
    preferred_values: list[str] | None = None,
    profile_id: str | None = None,
) -> dict:
    criterion = {
        "name": name,
        "direction": direction,
        "origin": origin,
        "application": "rank",
        "scope": scope,
        "rationale": rationale,
    }
    if weight is not None:
        criterion["weight"] = weight
    if preferred_values is not None:
        criterion["preferred_values"] = preferred_values
    if profile_id is not None:
        criterion["profile_id"] = profile_id
    return criterion


def _candidate_discovery_ranking(plan: dict) -> dict:
    return plan["ranking"]["candidate_ranking"]["discovery"]


def _explanation_ranking(plan: dict) -> dict:
    return plan["ranking"]["explanation_ranking"]


def _add_fixed_target_disease_hop(plan: dict) -> None:
    """Create the legacy gated-Q1 shape for fixed-hop fallback tests."""
    plan["paths"][0]["hops"].append({
        "subject_ref": "target_gene",
        "predicate": "biolink:gene_associated_with_condition",
        "object_ref": "disease_x",
        "predicate_expansion": "descendants",
    })


def test_planner_happy_path():
    canned = _load_example("example_q1_target_based.json")
    agent = PlannerAgent(llm=EchoJSONClient(canned))
    result = agent.plan("What approved drugs target VEGFR2 and could be repurposed for diabetic retinopathy?")
    assert result.ok
    assert result.attempts == 1
    assert result.plan is not None
    assert result.plan.plan_mode == "hybrid"


def test_planner_strips_markdown_fences():
    canned = _load_example("example_q1_target_based.json")
    fenced = f"```json\n{canned}\n```"
    agent = PlannerAgent(llm=EchoJSONClient(fenced))
    result = agent.plan("test question")
    assert result.ok
    assert result.attempts == 1


def test_planner_stamps_versions_when_model_omits_them():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    plan.pop("plan_version")
    plan.pop("biolink_version")
    agent = PlannerAgent(llm=EchoJSONClient(json.dumps(plan)))

    result = agent.plan("test question")

    assert result.ok
    assert result.attempts == 1
    assert result.plan is not None
    assert result.plan.plan_version == PLAN_VERSION
    assert result.plan.biolink_version == load_biolink_vocabulary().biolink_version


def test_generate_complete_plan_returns_versioned_json():
    from generate_plan import generate_complete_plan

    plan = json.loads(_load_example("example_q1_target_based.json"))
    plan.pop("plan_version")
    plan.pop("biolink_version")

    complete = generate_complete_plan(
        "test question",
        llm=EchoJSONClient(json.dumps(plan)),
    )

    assert complete["plan_version"] == PLAN_VERSION
    assert complete["biolink_version"] == load_biolink_vocabulary().biolink_version
    assert "return" in complete["explanation_queries"][0]
    assert "return_" not in complete["explanation_queries"][0]
    assert validate_plan(complete).ok


def test_planner_retries_on_validation_failure_then_succeeds():
    """First response is invalid (bad predicate), second is valid."""
    canned_bad = json.loads(_load_example("example_q1_target_based.json"))
    canned_bad["paths"][0]["hops"][0]["predicate"] = "biolink:definitely_not_a_predicate"
    canned_good = _load_example("example_q1_target_based.json")

    class TwoShotClient:
        def __init__(self):
            self.n = 0
        def complete(self, system, user):
            self.n += 1
            return json.dumps(canned_bad) if self.n == 1 else canned_good

    agent = PlannerAgent(llm=TwoShotClient())
    result = agent.plan("test")
    assert result.ok
    assert result.attempts == 2
    assert len(result.attempt_history) == 2
    assert result.attempt_history[0].validation is not None
    assert not result.attempt_history[0].validation.ok
    assert result.attempt_history[1].validation is not None
    assert result.attempt_history[1].validation.ok
    assert "definitely_not_a_predicate" in result.attempt_history[0].raw_response


def test_planner_verifies_external_input_against_authoritative_manifest():
    canned = _load_example("example_q3_signature_reversal.json")
    question = json.loads(canned)["question"]
    agent = PlannerAgent(llm=EchoJSONClient(canned))

    result = agent.plan(
        question,
        available_inputs=[{
            "input_ref": "tnbc_signature",
            "expected_format": "directional_gene_signature",
            "directions": ["increased", "decreased"],
        }],
    )

    assert result.ok, result.validation.format()
    assert result.attempts == 1
    assert result.attempt_history[0].validation is not None
    assert result.attempt_history[0].validation.ok


def test_planner_retries_missing_external_input_as_clarification_refusal():
    signature_plan = _load_example("example_q3_signature_reversal.json")
    refusal = _load_example("example_refusal_missing_signature_input.json")

    class MissingInputThenRefusalClient:
        def __init__(self):
            self.responses = iter([signature_plan, refusal])
            self.calls = []

        def complete(self, system, user):
            self.calls.append((system, user))
            return next(self.responses)

    client = MissingInputThenRefusalClient()
    agent = PlannerAgent(llm=client)
    result = agent.plan(
        "Using the supplied dataset asthma_signature, which contains a "
        "directional human gene signature, find compounds predicted to reverse it."
    )

    assert result.ok, result.validation.format()
    assert result.attempts == 2
    assert result.plan is not None
    assert result.plan.refusal is not None
    assert result.plan.refusal.reason == "needs_clarification"
    first_validation = result.attempt_history[0].validation
    assert first_validation is not None and not first_validation.ok
    assert any(
        "not present in the authoritative available-input manifest" in error.message
        for error in first_validation.errors
    )
    assert result.attempt_history[1].validation is not None
    assert result.attempt_history[1].validation.ok
    assert "AVAILABLE EXTERNAL INPUTS (authoritative manifest):\n[]" in client.calls[0][1]
    assert "External-input rules" in client.calls[1][1]


def test_retry_message_adds_ranking_shape_for_observed_nesting_errors():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    plan["plan_version"] = "1.0"
    candidate = plan["ranking"]["candidate_ranking"]
    candidate["strategy"] = candidate["discovery"].pop("strategy")
    knowledge_level = next(
        criterion
        for criterion in candidate["discovery"]["criteria"]
        if criterion["name"] == "knowledge_level"
    )
    knowledge_level.pop("preferred_values")

    result = validate_plan(plan)
    assert not result.ok
    message = _build_retry_message("test question", json.dumps(plan), result)

    assert "Relevant canonical schema shapes" in message
    assert "Ranking blocks (place only the mode-appropriate blocks" in message
    assert '"discovery": {' in message
    assert '"strategy": "evidence_weighted"' in message
    assert '"preferred_values": [' in message
    assert "Never put `strategy`" in message
    assert '"plan_version": "1.0"' not in message
    assert "Do not emit `plan_version` or `biolink_version`" in message
    assert "Explain why explanations do not affect candidate order." not in message


def test_retry_message_adds_explanation_return_shape_for_observed_field_errors():
    plan = json.loads(_load_example("example_q7_explanation.json"))
    query = plan["explanation_queries"][0]
    query["top_k_paths"] = query["return"].pop("top_k_paths")
    query["return"]["top_k"] = 5

    result = validate_plan(plan)
    assert not result.ok
    message = _build_retry_message("test question", json.dumps(plan), result)

    assert "Relevant canonical schema shapes" in message
    assert "Explanation-query return shape" in message
    assert '"top_k_paths": 5' in message
    assert "`return.top_k` is not valid" in message
    assert "Ranking blocks (place only the mode-appropriate blocks" not in message


def test_retry_message_adds_exact_path_shape_for_observed_path_errors():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    plan["paths"][0]["max_hops"] = 1

    result = validate_plan(plan)
    assert not result.ok
    message = _build_retry_message("test question", json.dumps(plan), result)

    assert "Discovery Path shape" in message
    assert '"hops": [' in message
    assert "no `max_hops` field and no `constraints` object" in message
    assert "Only an `ExplanationQuery`\nhas `max_hops`" in message


def test_retry_message_places_root_entity_constraints_on_entity():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    candidate = next(
        entity for entity in plan["entities"]
        if entity["entity_ref"] == "candidate_drug"
    )
    plan["entity_constraints"] = candidate.pop("constraints")

    result = validate_plan(plan)
    assert not result.ok
    message = _build_retry_message("test question", json.dumps(plan), result)

    assert "Entity constraint placement" in message
    assert "There is no top-level\n`entity_constraints` field" in message
    assert '"field": "approval_status"' in message


def test_planner_returns_refusal_object_when_llm_produces_refusal():
    canned = _load_example("example_refusal_needs_clarification.json")
    agent = PlannerAgent(llm=EchoJSONClient(canned))
    result = agent.plan("Are there any drugs I could try for it?")
    assert result.ok
    assert result.plan is not None
    assert result.plan.refusal is not None
    assert result.plan.refusal.reason == "needs_clarification"


def test_clinical_advice_refusal_uses_specific_reason_and_complete_shape():
    plan = json.loads(_load_example("example_refusal_clinical_advice.json"))

    assert plan["entities"] == []
    assert plan["refusal"]["reason"] == "unsafe_or_clinical_advice"
    assert not {
        "paths", "explanation_queries", "ranking", "aggregation",
    }.intersection(plan)
    assert validate_plan(plan).ok
    QueryPlan.model_validate(plan)


def test_retry_message_adds_complete_refusal_shape():
    plan = json.loads(_load_example("example_refusal_clinical_advice.json"))
    plan.pop("entities")

    result = validate_plan(plan)
    assert not result.ok
    message = _build_retry_message(plan["question"], json.dumps(plan), result)

    assert "Refusal shape" in message
    assert '"entities": []' in message
    assert "A refusal still requires" in message
    assert "Never emit those fields as null" in message
    assert "`unsafe_or_clinical_advice`, not `out_of_scope`" in message


def test_validator_rejects_executable_content_on_refusal():
    plan = json.loads(_load_example("example_refusal_clinical_advice.json"))
    executable = json.loads(_load_example("example_q1_target_based.json"))
    plan["entities"] = copy.deepcopy(executable["entities"])
    plan["paths"] = copy.deepcopy(executable["paths"])
    plan["ranking"] = copy.deepcopy(executable["ranking"])

    result = validate_plan(plan)

    assert not result.ok
    assert any(
        error.location == "entities" and "empty entities array" in error.message
        for error in result.errors
    )
    assert any(
        error.location == "paths" and "must omit" in error.message
        for error in result.errors
    )
    assert any(
        error.location == "ranking" and "must omit" in error.message
        for error in result.errors
    )


def test_planner_reports_failure_when_both_attempts_fail():
    bad = '{"not_a_plan": true}'
    agent = PlannerAgent(llm=EchoJSONClient(bad))
    result = agent.plan("test")
    assert not result.ok
    assert result.attempts == 2
    assert result.plan is None
    assert len(result.attempt_history) == 2
    assert all(
        attempt.validation is not None and not attempt.validation.ok
        for attempt in result.attempt_history
    )


def test_planner_reports_parse_error():
    agent = PlannerAgent(llm=EchoJSONClient("this is not JSON"))
    result = agent.plan("test")
    assert not result.ok
    assert result.error is not None
    assert "JSON parse" in result.error
    assert len(result.attempt_history) == 1
    assert result.attempt_history[0].validation is None
    assert result.attempt_history[0].error == result.error


def test_smoke_result_saves_both_attempts_and_validation_results():
    from smoke_test import write_smoke_result

    canned_bad = json.loads(_load_example("example_q1_target_based.json"))
    canned_bad["paths"][0]["hops"][0]["predicate"] = (
        "biolink:definitely_not_a_predicate"
    )
    canned_good = _load_example("example_q1_target_based.json")

    class TwoShotClient:
        def __init__(self):
            self.responses = iter([json.dumps(canned_bad), canned_good])

        def complete(self, system, user):
            return next(self.responses)

    result = PlannerAgent(llm=TwoShotClient()).plan("test question")
    assert result.ok and result.attempts == 2

    with tempfile.TemporaryDirectory() as directory:
        output = Path(directory) / "smoke.txt"
        write_smoke_result(output, "test question", result)
        text = output.read_text()

    assert "ATTEMPT 1 RAW RESPONSE" in text
    assert "ATTEMPT 1 VALIDATION" in text
    assert "definitely_not_a_predicate" in text
    assert "ATTEMPT 2 RAW RESPONSE" in text
    assert "ATTEMPT 2 VALIDATION" in text
    assert text.rstrip().endswith("OK")


# ---------------------------------------------------------------------
# Bonus: validators reject bad predicates
# ---------------------------------------------------------------------

def test_validator_flags_unknown_predicate():
    with open(str(EXAMPLE_DIR)+"/example_q1_target_based.json") as f:
        plan = json.load(f)
    plan["paths"][0]["hops"][0]["predicate"] = "biolink:definitely_not_a_predicate"
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        "not a valid active Biolink predicate" in e.message for e in result.errors
    )


@pytest.mark.parametrize(
    "predicate",
    ["biolink:affects", "biolink:regulates", "biolink:interacts_with"],
)
def test_validator_accepts_qualifiers_on_supported_predicate_families(predicate):
    plan = json.loads(_load_example("example_hybrid_ipf.json"))
    hop = plan["paths"][1]["hops"][1]
    hop["predicate"] = predicate
    result = validate_plan(plan)
    assert not any(
        error.location == "paths/1/hops/1/qualifiers"
        and "cannot carry qualifiers" in error.message
        for error in result.errors
    )


def test_validator_rejects_qualifiers_on_other_predicate_families():
    plan = json.loads(_load_example("example_hybrid_ipf.json"))
    hop = plan["paths"][1]["hops"][1]
    hop["predicate"] = "biolink:treats"
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        error.location == "paths/1/hops/1/qualifiers"
        and "cannot carry qualifiers" in error.message
        for error in result.errors
    )


@pytest.mark.parametrize("placement", ["hop", "qualified", "explanation_filter"])
def test_validator_rejects_deprecated_predicates_everywhere(placement):
    plan = json.loads(_load_example("example_q1_target_based.json"))
    predicate = next(iter(load_biolink_vocabulary().deprecated_predicate_curies))

    if placement == "hop":
        plan["paths"][0]["hops"][0]["predicate"] = predicate
    elif placement == "qualified":
        plan["paths"][0]["hops"][0]["qualifiers"] = {
            "qualified_predicate": predicate,
        }
    else:
        post_filter = plan["explanation_queries"][0].setdefault("post_filter", {})
        post_filter["predicate_blacklist"] = [predicate]

    result = validate_plan(plan)
    assert not result.ok
    assert any(
        predicate in error.message and "deprecated" in error.message
        for error in result.errors
    )


def test_validator_flags_reversed_predicate_direction_and_suggests_inverse():
    plan = json.loads(_load_example("example_hybrid_ipf.json"))
    hop = plan["paths"][0]["hops"][0]
    hop["subject_ref"], hop["object_ref"] = hop["object_ref"], hop["subject_ref"]
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        "use inverse predicate 'biolink:condition_associated_with_gene'" in e.message
        for e in result.errors
    )


def test_validator_accepts_mixin_category_against_named_thing_range():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    target = next(
        entity for entity in plan["entities"]
        if entity["entity_ref"] == "target_gene"
    )
    target["biolink_category"] = "GeneOrGeneProduct"
    plan["paths"][0]["hops"][0]["predicate"] = (
        "biolink:directly_physically_interacts_with"
    )

    result = validate_plan(plan)

    assert result.ok, result.format()


def test_validator_requires_variable_discovery_answer():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    next(e for e in plan["entities"] if e["entity_ref"] == "candidate_drug")["is_variable"] = False
    result = validate_plan(plan)
    assert not result.ok
    assert any("must have is_variable=true" in e.message for e in result.errors)


def test_validator_accepts_broad_expected_result_category():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    plan["paths"][0]["expected_result_category"] = "ChemicalEntity"
    result = validate_plan(plan)
    assert result.ok, result.format()


def test_validator_rejects_unknown_expected_result_category():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    plan["paths"][0]["expected_result_category"] = "NotACategory"
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        error.kind == "schema"
        and error.location == "paths/0/expected_result_category"
        for error in result.errors
    )
    assert any(
        error.kind == "semantic"
        and error.location == "paths/0/expected_result_category"
        and "not a valid Biolink category" in error.message
        for error in result.errors
    )


def test_validator_rejects_incompatible_expected_result_category():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    plan["paths"][0]["expected_result_category"] = "Gene"
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        error.kind == "semantic"
        and error.location == "paths/0/expected_result_category"
        and "incompatible with return entity" in error.message
        for error in result.errors
    )


@pytest.mark.parametrize(
    "category_field",
    ["middle_category_whitelist", "middle_category_blacklist"],
)
def test_validator_rejects_unknown_explanation_middle_category(category_field):
    plan = json.loads(_load_example("example_q7_explanation.json"))
    plan["explanation_queries"][0][category_field] = ["NotACategory"]
    result = validate_plan(plan)
    assert not result.ok
    expected_location = f"explanation_queries/0/{category_field}/0"
    assert any(
        error.kind == "schema" and error.location == expected_location
        for error in result.errors
    )
    assert any(
        error.kind == "semantic"
        and error.location == expected_location
        and "not a valid Biolink category" in error.message
        for error in result.errors
    )


def test_validator_requires_open_explanation_fallback_for_fixed_hop():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    _add_fixed_target_disease_hop(plan)
    plan["plan_mode"] = "discovery"
    plan.pop("explanation_queries")
    plan["ranking"].pop("explanation_ranking")
    result = validate_plan(plan)
    assert not result.ok
    assert any("plan_mode='hybrid'" in e.message for e in result.errors)
    assert any(
        "predicate-unconstrained explanation_query" in e.message
        for e in result.errors
    )


def test_validator_rejects_predicate_constrained_fixed_hop_fallback():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    _add_fixed_target_disease_hop(plan)
    plan["explanation_queries"][0]["post_filter"]["predicate_whitelist"] = [
        "biolink:gene_associated_with_condition"
    ]
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        "predicate-unconstrained explanation_query" in e.message
        for e in result.errors
    )


def test_q1_known_target_disease_does_not_gate_candidates_on_fixed_edge():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    assert len(plan["paths"]) == 1
    assert len(plan["paths"][0]["hops"]) == 1
    assert plan["paths"][0]["hops"][0] == {
        "subject_ref": "candidate_drug",
        "predicate": "biolink:affects",
        "object_ref": "target_gene",
        "predicate_expansion": "descendants",
    }
    assert plan["explanation_queries"][0]["endpoint_a"]["entity_ref"] == "target_gene"
    assert plan["explanation_queries"][0]["endpoint_b"]["entity_ref"] == "disease_x"
    assert plan["ranking"]["candidate_ranking"]["explanation_influence"] == "none"


def test_ranking_subjects_and_hybrid_influence_are_explicit_by_mode():
    discovery = json.loads(_load_example("example_q3_signature_reversal.json"))
    assert set(discovery["ranking"]) == {"candidate_ranking"}
    assert discovery["ranking"]["candidate_ranking"]["explanation_influence"] == "none"

    explanation = json.loads(_load_example("example_q7_explanation.json"))
    assert set(explanation["ranking"]) == {"explanation_ranking"}

    annotate = json.loads(_load_example("example_hybrid_ipf.json"))
    assert set(annotate["ranking"]) == {
        "candidate_ranking", "explanation_ranking",
    }
    candidate = annotate["ranking"]["candidate_ranking"]
    assert candidate["explanation_influence"] == "annotate_only"
    assert "final" not in candidate

    independent = json.loads(_load_example("example_q1_target_based.json"))
    assert independent["ranking"]["candidate_ranking"]["explanation_influence"] == "none"
    assert independent["aggregation"]["attach_explanations_to_candidates"] is False


def test_validator_accepts_candidate_reranking_by_explanations():
    plan = json.loads(_load_example("example_hybrid_ipf.json"))
    candidate = plan["ranking"]["candidate_ranking"]
    candidate["explanation_influence"] = "rerank"
    candidate["rationale"] = (
        "The user explicitly asks to prioritize candidates by mechanistic support."
    )
    candidate["final"] = {
        "strategy": "multi_path_consensus",
        "criteria": [
            _criterion("num_supporting_paths", "desc", 2),
            _criterion("num_explanation_paths", "desc", 1.5),
            _criterion("distinct_intermediate_categories", "desc", 1),
            _criterion(
                "knowledge_level",
                "desc",
                1,
                preferred_values=["knowledge_assertion", "logical_entailment"],
            ),
        ],
        "top_k": 20,
    }
    result = validate_plan(plan)
    assert result.ok, result.format()
    QueryPlan.model_validate(plan)


def test_validator_requires_final_stage_only_for_reranking():
    plan = json.loads(_load_example("example_hybrid_ipf.json"))
    plan["ranking"]["candidate_ranking"]["explanation_influence"] = "rerank"
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        "final ranking is required" in error.message
        for error in result.errors
    )

    plan = json.loads(_load_example("example_hybrid_ipf.json"))
    plan["ranking"]["candidate_ranking"]["final"] = copy.deepcopy(
        _candidate_discovery_ranking(plan)
    )
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        "allowed only when explanation_influence is 'rerank'" in error.message
        for error in result.errors
    )


def test_validator_requires_explanation_derived_final_ranking_criterion():
    plan = json.loads(_load_example("example_hybrid_ipf.json"))
    candidate = plan["ranking"]["candidate_ranking"]
    candidate["explanation_influence"] = "rerank"
    candidate["final"] = copy.deepcopy(candidate["discovery"])
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        "must include at least one explanation-derived criterion" in error.message
        for error in result.errors
    )


def test_validator_matches_hybrid_influence_to_endpoint_binding():
    candidate_specific = json.loads(_load_example("example_hybrid_ipf.json"))
    candidate_specific["ranking"]["candidate_ranking"]["explanation_influence"] = "none"
    result = validate_plan(candidate_specific)
    assert not result.ok
    assert any(
        "candidate-specific explanations require" in error.message
        for error in result.errors
    )

    shared_context = json.loads(_load_example("example_q1_target_based.json"))
    shared_context["ranking"]["candidate_ranking"]["explanation_influence"] = "annotate_only"
    shared_context["aggregation"]["attach_explanations_to_candidates"] = True
    result = validate_plan(shared_context)
    assert not result.ok
    assert any(
        "requires at least one from_discovery endpoint binding" in error.message
        for error in result.errors
    )


def test_validator_matches_ranking_subjects_to_plan_mode():
    explanation = json.loads(_load_example("example_q7_explanation.json"))
    candidate_source = json.loads(_load_example("example_q1_target_based.json"))
    explanation["ranking"]["candidate_ranking"] = copy.deepcopy(
        candidate_source["ranking"]["candidate_ranking"]
    )
    result = validate_plan(explanation)
    assert not result.ok
    assert any("no candidate entities to rank" in error.message for error in result.errors)

    discovery = json.loads(_load_example("example_q3_signature_reversal.json"))
    discovery["ranking"]["explanation_ranking"] = copy.deepcopy(
        explanation["ranking"]["explanation_ranking"]
    )
    result = validate_plan(discovery)
    assert not result.ok
    assert any("no explanation paths to rank" in error.message for error in result.errors)


def test_validator_rejects_explanation_features_before_explanation_stage():
    plan = json.loads(_load_example("example_hybrid_ipf.json"))
    _candidate_discovery_ranking(plan)["criteria"].append(
        _criterion("num_explanation_paths", "desc", 1)
    )
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        "cannot use explanation-derived criteria" in error.message
        for error in result.errors
    )


def test_validator_rejects_candidate_path_count_in_explanation_ranking():
    plan = json.loads(_load_example("example_q7_explanation.json"))
    _explanation_ranking(plan)["criteria"].append(
        _criterion("num_explanation_paths", "desc", 1)
    )

    result = validate_plan(plan)

    assert not result.ok
    assert any(
        error.location == "ranking/explanation_ranking/criteria"
        and "belongs only in ranking.candidate_ranking.final" in error.message
        for error in result.errors
    )


def test_schema_rejects_legacy_ambiguous_ranking_shape():
    plan = json.loads(_load_example("example_q3_signature_reversal.json"))
    plan["ranking"] = copy.deepcopy(_candidate_discovery_ranking(plan))
    result = validate_plan(plan)
    assert not result.ok
    assert any(error.kind == "schema" for error in result.errors)
    with pytest.raises(PydanticValidationError):
        QueryPlan.model_validate(plan)


def test_validator_requires_explicit_ranking_criteria_and_top_k():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    plan.pop("ranking")
    result = validate_plan(plan)
    assert not result.ok
    assert any("explicit ranking object" in e.message for e in result.errors)

    plan = json.loads(_load_example("example_q1_target_based.json"))
    discovery_ranking = _candidate_discovery_ranking(plan)
    discovery_ranking.pop("criteria")
    discovery_ranking.pop("top_k")
    result = validate_plan(plan)
    assert not result.ok
    assert any("at least one explicit criterion" in e.message for e in result.errors)
    assert any("top_k is required" in e.message for e in result.errors)


def test_validator_rejects_consensus_for_one_active_path():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    discovery_ranking = _candidate_discovery_ranking(plan)
    discovery_ranking["strategy"] = "multi_path_consensus"
    discovery_ranking["criteria"].insert(
        0, _criterion("num_supporting_paths", "desc", 2)
    )
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        "multi_path_consensus requires at least two active" in e.message
        for e in result.errors
    )


def test_validator_accepts_evidence_ranking_for_one_active_path():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    plan["ranking"]["candidate_ranking"]["discovery"] = {
        "strategy": "evidence_weighted",
        "criteria": [
            _criterion(
                "knowledge_level",
                "desc",
                1,
                preferred_values=[
                    "knowledge_assertion",
                    "logical_entailment",
                    "observation",
                    "statistical_association",
                ],
            )
        ],
        "top_k": 30,
        "tie_breaker": "alphabetical",
    }
    result = validate_plan(plan)
    assert result.ok, result.format()


def test_validator_rejects_shortest_path_ranking_for_fixed_discovery_path():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    plan["ranking"]["candidate_ranking"]["discovery"] = {
        "strategy": "shortest_path_first",
        "criteria": [
            _criterion("path_length", "asc", 1)
        ],
        "top_k": 30,
    }
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        "cannot rank candidates from fixed discovery paths" in e.message
        for e in result.errors
    )


def test_validator_requires_strategy_feature_criteria():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    plan["ranking"]["candidate_ranking"]["discovery"] = {
        "strategy": "evidence_weighted",
        "criteria": [_criterion("recency", "desc", 1)],
        "top_k": 30,
    }
    result = validate_plan(plan)
    assert not result.ok
    assert any("portable evidence criterion" in e.message for e in result.errors)

    plan["ranking"]["candidate_ranking"]["discovery"] = {
        "strategy": "genetic_evidence_boosted",
        "criteria": [
            _criterion("num_knowledge_sources", "desc", 1)
        ],
        "top_k": 30,
    }
    result = validate_plan(plan)
    assert not result.ok
    assert any("genetic_support criterion" in e.message for e in result.errors)


def test_validator_accepts_supported_user_ranking_override():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    plan["ranking"]["candidate_ranking"]["discovery"] = {
        "strategy": "custom",
        "criteria": [
            _criterion("recency", "desc", 3, origin="user_requested"),
            _criterion(
                "knowledge_level",
                "desc",
                1,
                preferred_values=["knowledge_assertion", "logical_entailment"],
            ),
        ],
        "top_k": 30,
    }
    plan["confidence"]["reasons"].append(
        "user_requested_ranking_override: prioritize recent evidence"
    )
    result = validate_plan(plan)
    assert result.ok, result.format()


def test_validator_requires_documented_and_weighted_custom_ranking():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    plan["ranking"]["candidate_ranking"]["discovery"] = {
        "strategy": "custom",
        "criteria": [
            _criterion("recency", "desc", origin="user_requested"),
            _criterion("num_knowledge_sources", "desc", 1),
        ],
        "top_k": 30,
    }
    result = validate_plan(plan)
    assert not result.ok
    assert any("custom ranking requires a confidence reason" in e.message for e in result.errors)
    assert any("explicit weight for every criterion" in e.message for e in result.errors)


def test_validator_accepts_explicit_portable_knowledge_level_ranking():
    plan = json.loads(_load_example("example_q7_explanation.json"))
    knowledge_level = next(
        criterion for criterion in _explanation_ranking(plan)["criteria"]
        if criterion["name"] == "knowledge_level"
    )
    assert knowledge_level["scope"] == "portable"
    assert knowledge_level["application"] == "rank"
    assert knowledge_level["preferred_values"]
    result = validate_plan(plan)
    assert result.ok, result.format()


def test_validator_requires_executor_profile_for_epc_ranking():
    plan = json.loads(_load_example("example_q7_explanation.json"))
    explanation_ranking = _explanation_ranking(plan)
    explanation_ranking["criteria"].append(
        _criterion("edge_evidence_strength", "desc", 1)
    )
    result = validate_plan(plan)
    assert not result.ok
    assert any("executor-specific derived criterion" in e.message for e in result.errors)
    assert any("executor_epc_v1" in e.message for e in result.errors)

    explanation_ranking["criteria"][-1] = _criterion(
        "edge_evidence_strength",
        "desc",
        1,
        scope="executor_specific",
        profile_id="executor_epc_v1",
        rationale="Use the project Executor's versioned composite evidence score.",
    )
    result = validate_plan(plan)
    assert result.ok, result.format()


def test_evidence_recommendations_are_non_filtering_and_portable():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    recommendations = {
        item["feature"]: item for item in plan["evidence_recommendations"]
    }
    assert recommendations["primary_knowledge_source"]["application"] == "report"
    assert recommendations["knowledge_level"]["application"] == "report"
    assert "evidence_policy" not in plan
    result = validate_plan(plan)
    assert result.ok, result.format()


def test_evidence_policy_requires_a_user_requested_hard_filter():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    plan["evidence_policy"] = {
        "origin": "user_requested",
        "application": "filter",
        "rationale": "The user requested edges with declared source provenance.",
    }
    result = validate_plan(plan)
    assert not result.ok
    assert any("at least one user-requested hard filter" in e.message for e in result.errors)

    plan["evidence_policy"]["require_primary_knowledge_source"] = True
    result = validate_plan(plan)
    assert result.ok, result.format()


@pytest.mark.parametrize("status", ["approved", "ever_approved"])
def test_validator_accepts_portable_hard_approval_status(status):
    plan = json.loads(_load_example("example_q1_target_based.json"))
    candidate = next(
        entity for entity in plan["entities"]
        if entity["entity_ref"] == "candidate_drug"
    )
    constraint = candidate["constraints"][0]
    constraint["value"] = status
    assert constraint["field"] == "approval_status"
    assert constraint["op"] == "eq"
    assert all(
        criterion["name"] != "approval_status"
        for criterion in _candidate_discovery_ranking(plan)["criteria"]
    )
    result = validate_plan(plan)
    assert result.ok, result.format()
    QueryPlan.model_validate(plan)


@pytest.mark.parametrize(
    ("op", "value"),
    [
        ("in", ["approved"]),
        ("eq", "marketed"),
        ("eq", "prescription only"),
    ],
)
def test_validator_rejects_nonsemantic_approval_constraint(op, value):
    plan = json.loads(_load_example("example_q1_target_based.json"))
    candidate = next(
        entity for entity in plan["entities"]
        if entity["entity_ref"] == "candidate_drug"
    )
    candidate["constraints"][0].update(op=op, value=value)
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        "approval_status" in error.message
        for error in result.errors
        if error.kind == "semantic"
    )
    with pytest.raises(PydanticValidationError):
        QueryPlan.model_validate(plan)


def test_validator_rejects_unsupported_entity_constraint():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    candidate = next(
        entity for entity in plan["entities"]
        if entity["entity_ref"] == "candidate_drug"
    )
    candidate["constraints"][0] = {
        "field": "max_phase",
        "op": "gte",
        "value": 4,
    }
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        "only the portable 'approval_status' field" in error.message
        for error in result.errors
        if error.kind == "semantic"
    )
    with pytest.raises(PydanticValidationError):
        QueryPlan.model_validate(plan)


def test_schema_rejects_planner_recommended_hard_filter_and_old_source_name():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    plan["evidence_policy"] = {
        "origin": "planner_recommended",
        "application": "filter",
        "rationale": "Invalid silent planner filter.",
        "require_primary_source": True,
    }
    result = validate_plan(plan)
    assert not result.ok
    assert any(error.kind == "schema" for error in result.errors)


def test_validator_rejects_variable_input_binding():
    plan = json.loads(_load_example("example_q3_signature_reversal.json"))
    next(
        e for e in plan["entities"]
        if e["entity_ref"] == "sig_increased_gene"
    )["is_variable"] = True
    result = validate_plan(plan)
    assert not result.ok
    assert any("input-bound entity must have is_variable=false" in e.message for e in result.errors)


def test_directional_signature_requires_partition_filter():
    plan = json.loads(_load_example("example_q3_signature_reversal.json"))
    binding = next(
        entity["input_binding"] for entity in plan["entities"]
        if entity["entity_ref"] == "sig_increased_gene"
    )
    binding.pop("direction_filter")

    result = validate_plan(plan)

    assert not result.ok
    assert any(
        error.location.endswith("input_binding/direction_filter")
        and "requires one valid" in error.message
        for error in result.errors
    )
    with pytest.raises(PydanticValidationError):
        QueryPlan.model_validate(plan)


def test_direction_filter_is_rejected_for_non_directional_input():
    plan = json.loads(_load_example("example_q3_signature_reversal.json"))
    binding = next(
        entity["input_binding"] for entity in plan["entities"]
        if entity["entity_ref"] == "sig_increased_gene"
    )
    binding["expected_format"] = "curie_list"

    result = validate_plan(plan)

    assert not result.ok
    assert any(error.kind == "schema" for error in result.errors)
    with pytest.raises(PydanticValidationError):
        QueryPlan.model_validate(plan)


def test_signature_reversal_uses_opposite_expression_direction():
    plan = json.loads(_load_example("example_q3_signature_reversal.json"))
    increased_binding = next(
        entity["input_binding"] for entity in plan["entities"]
        if entity["entity_ref"] == "sig_increased_gene"
    )
    decreased_binding = next(
        entity["input_binding"] for entity in plan["entities"]
        if entity["entity_ref"] == "sig_decreased_gene"
    )
    assert increased_binding["input_ref"] == decreased_binding["input_ref"]
    assert increased_binding["direction_filter"] == "increased"
    assert decreased_binding["direction_filter"] == "decreased"
    assert (
        plan["paths"][0]["hops"][0]["qualifiers"]
        ["object_direction_qualifier"]
        == "decreased"
    )
    assert (
        plan["paths"][1]["hops"][0]["qualifiers"]
        ["object_direction_qualifier"]
        == "increased"
    )
    assert validate_plan(plan).ok

    plan["paths"][0]["hops"][0]["qualifiers"][
        "object_direction_qualifier"
    ] = "increased"
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        "requires an opposite drug expression direction" in error.message
        for error in result.errors
    )


def test_inline_directional_signature_uses_surface_names_without_bindings():
    plan = json.loads(_load_example("example_q3_inline_signature.json"))
    genes = [
        entity for entity in plan["entities"]
        if entity["entity_ref"].startswith("gene_")
    ]

    assert {entity["name"] for entity in genes} == {"IL4", "IL5", "IL13"}
    assert all("input_binding" not in entity for entity in genes)
    assert all(entity["is_variable"] is False for entity in genes)
    assert all(entity["taxa"] == ["NCBITaxon:9606"] for entity in genes)
    assert len(plan["paths"]) == 3
    assert all(
        path["hops"][0]["qualifiers"]["object_direction_qualifier"]
        == "decreased"
        for path in plan["paths"]
    )
    assert validate_plan(plan).ok


def test_planner_accepts_inline_directional_signature_without_manifest():
    canned = _load_example("example_q3_inline_signature.json")
    question = json.loads(canned)["question"]
    agent = PlannerAgent(llm=EchoJSONClient(canned))

    result = agent.plan(question, available_inputs=[])

    assert result.ok, result.validation.format()
    assert result.attempts == 1
    assert all(
        entity.input_binding is None
        for entity in result.plan.entities
        if entity.entity_ref.startswith("gene_")
    )


def test_schema_and_pydantic_reject_legacy_entity_role():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    plan["entities"][0]["role"] = "seed"
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        error.kind == "schema"
        and error.location == "entities/0"
        and "role" in error.message
        for error in result.errors
    )
    with pytest.raises(PydanticValidationError):
        QueryPlan.model_validate(plan)


def test_schema_and_pydantic_require_explicit_is_variable():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    plan["entities"][0].pop("is_variable")
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        error.kind == "schema"
        and error.location == "entities/0"
        and "is_variable" in error.message
        for error in result.errors
    )
    with pytest.raises(PydanticValidationError):
        QueryPlan.model_validate(plan)


def test_validator_requires_fixed_anchor_in_each_enabled_discovery_path():
    plan = json.loads(_load_example("example_q3_signature_reversal.json"))
    signature_gene = next(
        entity for entity in plan["entities"]
        if entity["entity_ref"] == "sig_increased_gene"
    )
    signature_gene["is_variable"] = True
    signature_gene.pop("input_binding")
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        error.kind == "semantic"
        and error.location == "paths/0/hops"
        and "fixed query anchor" in error.message
        for error in result.errors
    )


def test_validator_does_not_require_anchor_for_disabled_path():
    plan = json.loads(_load_example("example_q3_signature_reversal.json"))
    signature_gene = next(
        entity for entity in plan["entities"]
        if entity["entity_ref"] == "sig_increased_gene"
    )
    signature_gene["is_variable"] = True
    signature_gene.pop("input_binding")
    plan["paths"][0]["disabled"] = True
    result = validate_plan(plan)
    assert not any(
        error.kind == "semantic"
        and error.location == "paths/0/hops"
        and "fixed query anchor" in error.message
        for error in result.errors
    )


def test_validator_rejects_disconnected_path_fragments():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    plan["entities"].extend([
        {
            "entity_ref": "other_gene",
            "name": "another gene",
            "biolink_category": "Gene",
            "taxa": ["NCBITaxon:9606"],
            "is_variable": True,
        },
        {
            "entity_ref": "other_disease",
            "name": "another disease",
            "biolink_category": "Disease",
            "is_variable": False,
        },
    ])
    plan["paths"][0]["hops"].append({
        "subject_ref": "other_gene",
        "predicate": "biolink:gene_associated_with_condition",
        "object_ref": "other_disease",
    })
    result = validate_plan(plan)
    assert not result.ok
    assert any("disconnected hop fragments" in e.message for e in result.errors)


@pytest.mark.parametrize(
    ("name", "left", "cue", "right"),
    [
        ("eczema from gluten allergy", "eczema", "from", "gluten allergy"),
        (
            "eczema from gluten sensitivity",
            "eczema",
            "from",
            "gluten sensitivity",
        ),
        (
            "rash due to celiac disease",
            "rash",
            "due to",
            "celiac disease",
        ),
        (
            "skin manifestation of celiac disease",
            "skin manifestation",
            "manifestation of",
            "celiac disease",
        ),
    ],
)
def test_validator_rejects_relational_fixed_entity_names(
    name, left, cue, right,
):
    plan = json.loads(_load_example("example_q1_target_based.json"))
    entity_index, disease = next(
        (index, entity)
        for index, entity in enumerate(plan["entities"])
        if entity["entity_ref"] == "disease_x"
    )
    disease["name"] = name

    result = validate_plan(plan)

    assert not result.ok
    matching_errors = [
        error for error in result.errors
        if error.location == f"entities/{entity_index}/name"
        and "appears to combine separate concepts" in error.message
    ]
    assert len(matching_errors) == 1
    message = matching_errors[0].message
    assert f"separate concepts '{left}' and '{right}'" in message
    assert f"relational cue '{cue}'" in message
    assert "needs_clarification" in message


@pytest.mark.parametrize(
    "name",
    [
        "idiopathic pulmonary fibrosis",
        "diabetes mellitus with complications",
    ],
)
def test_validator_allows_nonrelational_fixed_entity_names(name):
    plan = json.loads(_load_example("example_q1_target_based.json"))
    disease = next(
        entity for entity in plan["entities"]
        if entity["entity_ref"] == "disease_x"
    )
    disease["name"] = name

    result = validate_plan(plan)

    assert result.ok, result.format()


def test_validator_rejects_self_loop_hop():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    hop = plan["paths"][0]["hops"][0]
    hop["object_ref"] = hop["subject_ref"]
    result = validate_plan(plan)
    assert not result.ok
    assert any("self-loop hops are unsupported" in error.message for error in result.errors)
    with pytest.raises(PydanticValidationError):
        QueryPlan.model_validate(plan)


def test_validator_rejects_identical_explanation_entity_endpoints():
    plan = json.loads(_load_example("example_q7_explanation.json"))
    query = plan["explanation_queries"][0]
    query["endpoint_b"] = copy.deepcopy(query["endpoint_a"])
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        "self-loop explanation queries are unsupported" in error.message
        for error in result.errors
    )
    with pytest.raises(PydanticValidationError):
        QueryPlan.model_validate(plan)


def test_validator_rejects_same_discovery_source_at_both_endpoints():
    plan = json.loads(_load_example("example_hybrid_ipf.json"))
    query = plan["explanation_queries"][0]
    query["endpoint_b"] = copy.deepcopy(query["endpoint_a"])
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        "self-loop explanation queries are unsupported" in error.message
        for error in result.errors
    )
    with pytest.raises(PydanticValidationError):
        QueryPlan.model_validate(plan)


def test_validator_rejects_disabled_from_discovery_path():
    plan = json.loads(_load_example("example_hybrid_ipf.json"))
    plan["paths"][0]["disabled"] = True
    result = validate_plan(plan)
    assert not result.ok
    assert any("references a disabled path" in e.message for e in result.errors)


def test_validator_rejects_partial_hybrid_explanation_coverage():
    plan = json.loads(_load_example("example_hybrid_ipf.json"))
    plan["explanation_queries"][0]["endpoint_a"]["from_path_ids"] = [
        "P1_target_binds"
    ]
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        "candidate explanations do not cover enabled discovery paths" in e.message
        and "P2_target_inhibits" in e.message
        for e in result.errors
    )


def test_pydantic_round_trip_does_not_enable_candidate_explanations():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    plan["aggregation"].pop("attach_explanations_to_candidates")
    assert "attach_explanations_to_candidates" not in plan["aggregation"]
    round_tripped = QueryPlan.model_validate(plan).to_json_dict()
    assert (
        round_tripped["aggregation"]["attach_explanations_to_candidates"]
        is False
    )
    result = validate_plan(round_tripped)
    assert result.ok, result.format()


def test_validator_accepts_multiple_return_entity_refs_in_one_fanout():
    plan = json.loads(_load_example("example_hybrid_ipf.json"))
    original = next(
        entity for entity in plan["entities"]
        if entity["entity_ref"] == "candidate_drug"
    )
    alternate = copy.deepcopy(original)
    alternate["entity_ref"] = "candidate_drug_by_inhibition"
    alternate["name"] = "any approved small molecule found by inhibition"
    plan["entities"].append(alternate)

    second_path = next(
        path for path in plan["paths"]
        if path["path_id"] == "P2_target_inhibits"
    )
    for hop in second_path["hops"]:
        if hop.get("subject_ref") == "candidate_drug":
            hop["subject_ref"] = "candidate_drug_by_inhibition"
        if hop.get("object_ref") == "candidate_drug":
            hop["object_ref"] = "candidate_drug_by_inhibition"
    second_path["return_entity_ref"] = "candidate_drug_by_inhibition"

    result = validate_plan(plan)
    assert result.ok, result.format()
    QueryPlan.model_validate(plan)


def test_validator_rejects_incompatible_fanout_return_categories():
    plan = json.loads(_load_example("example_hybrid_ipf.json"))
    second_path = next(
        path for path in plan["paths"]
        if path["path_id"] == "P2_target_inhibits"
    )
    second_path["return_entity_ref"] = "ipf_gene"
    second_path["expected_result_category"] = "Gene"
    result = validate_plan(plan)
    assert not result.ok
    assert any(
        "incompatible return-entity categories" in e.message
        for e in result.errors
    )


def test_schema_and_pydantic_reject_legacy_singular_from_path_id():
    plan = json.loads(_load_example("example_hybrid_ipf.json"))
    binding = plan["explanation_queries"][0]["endpoint_a"]
    first_path_id = binding.pop("from_path_ids")[0]
    binding["from_path_id"] = first_path_id
    result = validate_plan(plan)
    assert not result.ok
    assert any(error.kind == "schema" for error in result.errors)
    with pytest.raises(PydanticValidationError):
        QueryPlan.model_validate(plan)


def test_schema_and_pydantic_reject_ambiguous_fanout_top_k_name():
    plan = json.loads(_load_example("example_hybrid_ipf.json"))
    binding = plan["explanation_queries"][0]["endpoint_a"]
    binding["top_k"] = binding.pop("fanout_top_k")
    result = validate_plan(plan)
    assert not result.ok
    assert any(error.kind == "schema" for error in result.errors)
    with pytest.raises(PydanticValidationError):
        QueryPlan.model_validate(plan)


def test_schema_and_pydantic_require_explicit_fanout_top_k():
    plan = json.loads(_load_example("example_hybrid_ipf.json"))
    binding = plan["explanation_queries"][0]["endpoint_a"]
    binding.pop("fanout_top_k")

    result = validate_plan(plan)

    assert not result.ok
    assert any(
        error.kind == "schema"
        and error.location == "explanation_queries/0/endpoint_a"
        and "fanout_top_k" in error.message
        for error in result.errors
    )
    with pytest.raises(PydanticValidationError):
        QueryPlan.model_validate(plan)


@pytest.mark.parametrize("influence", ["annotate_only", "rerank"])
def test_validator_requires_fanout_for_full_preliminary_candidate_pool(influence):
    plan = json.loads(_load_example("example_hybrid_ipf.json"))
    candidate = plan["ranking"]["candidate_ranking"]
    candidate["explanation_influence"] = influence
    candidate["discovery"]["top_k"] = 25
    if influence == "rerank":
        candidate["rationale"] = (
            "The user asks to rerank candidates using their explanations."
        )
        candidate["final"] = {
            "strategy": "multi_path_consensus",
            "criteria": [
                _criterion("num_supporting_paths", "desc", 2),
                _criterion("num_explanation_paths", "desc", 1.5),
            ],
            "top_k": 20,
        }

    result = validate_plan(plan)

    assert not result.ok
    assert any(
        error.location
        == "explanation_queries/0/endpoint_a/fanout_top_k"
        and "must be at least candidate_ranking.discovery.top_k (25)"
        in error.message
        for error in result.errors
    )


def test_retry_message_explains_full_candidate_fanout():
    plan = json.loads(_load_example("example_hybrid_ipf.json"))
    plan["ranking"]["candidate_ranking"]["discovery"]["top_k"] = 25

    result = validate_plan(plan)
    message = _build_retry_message(plan["question"], json.dumps(plan), result)

    assert not result.ok
    assert "Endpoint-binding shapes" in message
    assert "must be at least `candidate_ranking.discovery.top_k`" in message


@pytest.mark.parametrize("id_kind", ["entity_ref", "path_id", "query_id"])
def test_validator_rejects_duplicate_ids(id_kind):
    fixture = "example_q7_explanation.json" if id_kind == "query_id" else "example_q1_target_based.json"
    plan = json.loads(_load_example(fixture))
    if id_kind == "entity_ref":
        plan["entities"].append(copy.deepcopy(plan["entities"][0]))
    elif id_kind == "path_id":
        plan["paths"].append(copy.deepcopy(plan["paths"][0]))
    else:
        plan["explanation_queries"].append(copy.deepcopy(plan["explanation_queries"][0]))
    result = validate_plan(plan)
    assert not result.ok
    assert any(f"duplicate {id_kind}" in e.message for e in result.errors)


def test_explanation_max_hops_is_five_in_schema_and_pydantic():
    plan = json.loads(_load_example("example_q7_explanation.json"))
    plan["explanation_queries"][0]["max_hops"] = 6
    result = validate_plan(plan)
    assert not result.ok
    with pytest.raises(PydanticValidationError):
        QueryPlan.model_validate(plan)


def test_discovery_path_has_at_most_five_explicit_hops():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    hop = copy.deepcopy(plan["paths"][0]["hops"][0])
    plan["paths"][0]["hops"] = [copy.deepcopy(hop) for _ in range(6)]

    result = validate_plan(plan)

    assert not result.ok
    assert any(
        error.kind == "schema"
        and error.location == "paths/0/hops"
        and "too long" in error.message
        for error in result.errors
    )
    with pytest.raises(PydanticValidationError):
        QueryPlan.model_validate(plan)


def test_schema_rejects_non_ncb_taxon():
    plan = json.loads(_load_example("example_q1_target_based.json"))
    next(e for e in plan["entities"] if e["entity_ref"] == "target_gene")["taxa"] = ["human"]
    result = validate_plan(plan)
    assert not result.ok
    assert any(e.kind == "schema" and "NCBITaxon" in e.message for e in result.errors)

def test_static_schema_enums_are_biolink_subsets():
    """Every hard-coded enum value in query_plan.schema.json must exist in Biolink."""
    import json
    with open("../plan-core/schema/query_plan.schema.json") as f:
        schema = json.load(f)
    vocab = load_biolink_vocabulary()

    entity_cats = schema["$defs"]["Entity"]["properties"]["biolink_category"]["enum"]
    for c in entity_cats:
        assert vocab.is_valid_category(c), f"static schema has invalid category '{c}'"

    aspect = schema["$defs"]["Qualifiers"]["properties"]["object_aspect_qualifier"]["enum"]
    for v in aspect:
        assert v in vocab.aspect_qualifier_values, f"static schema has invalid aspect '{v}'"
