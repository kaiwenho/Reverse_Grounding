"""
Compile the base query plan JSON Schema by substituting Biolink-derived
enums for the hard-coded ones.

The base schema on disk contains illustrative hard-coded enums (small
subsets of Biolink for readability). At runtime, this module reads the
Biolink YAML via `biolink_vocab` and produces a *compiled schema* where:

- Entity.biolink_category enum -> all valid Biolink categories.
- Qualifiers.object_aspect_qualifier / subject_aspect_qualifier enums ->
  full aspect enum from Biolink.
- Qualifiers.object_direction_qualifier / subject_direction_qualifier enums ->
  direction enum from Biolink.
- ExternalInputBinding.direction_filter enum -> direction enum from Biolink.
- Qualifiers.causal_mechanism_qualifier enum -> causal mechanism enum from Biolink.
- EvidencePolicy.agent_type items enum -> Biolink AgentTypeEnum.
- Hop.predicate pattern is left alone (already `^biolink:[a-z][a-z0-9_]*$`)
  because enumerating every active predicate in-schema is noisy;
  predicate validity is checked separately by validators.py.
- Path.expected_result_category and ExplanationQuery.middle-category lists
  share Entity.biolink_category through local JSON Schema references, so the
  same injected model-derived category vocabulary applies to all of them.

The base schema is not modified on disk. This module returns a new dict.

Callers must use the compiled schema for validation, not the base one.
The base schema still validates syntactically (it's a legal JSON Schema)
but rejects some legal Biolink values that were not hard-coded originally.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .biolink_vocab import BiolinkVocabulary, load_biolink_vocabulary

BASE_SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "query_plan.schema.json"


def _set_enum(node: Dict[str, Any], values):
    """Replace the `enum` list in a JSON Schema node with `values` (sorted for determinism)."""
    node["enum"] = sorted(values)


def compile_schema(
    base_schema_path: Optional[Path] = None,
    vocab: Optional[BiolinkVocabulary] = None,
) -> Dict[str, Any]:
    """
    Load the base schema, inject Biolink-derived enums, return the compiled schema.

    :param base_schema_path: Path to base schema JSON. Defaults to the bundled copy.
    :param vocab: Preloaded vocabulary (for tests). Defaults to load from disk.
    :return: The compiled schema as a dict.
    """
    path = base_schema_path or BASE_SCHEMA_PATH
    with path.open() as f:
        schema = json.load(f)

    v = vocab or load_biolink_vocabulary()

    defs = schema["$defs"]

    # --- Entity.biolink_category ---
    defs["Entity"]["properties"]["biolink_category"]["enum"] = sorted(v.categories)

    # --- Qualifiers ---
    q = defs["Qualifiers"]["properties"]
    _set_enum(q["object_aspect_qualifier"], v.aspect_qualifier_values)
    _set_enum(q["subject_aspect_qualifier"], v.aspect_qualifier_values)
    _set_enum(q["object_direction_qualifier"], v.direction_qualifier_values)
    _set_enum(q["subject_direction_qualifier"], v.direction_qualifier_values)
    _set_enum(q["causal_mechanism_qualifier"], v.causal_mechanism_qualifier_values)

    # --- Directional external-input partition ---
    external_input = next(
        branch["properties"]
        for branch in defs["EntityInputBinding"]["oneOf"]
        if branch["properties"]["binding_type"].get("const")
        == "external_input"
    )
    _set_enum(external_input["direction_filter"], v.direction_qualifier_values)

    # --- EvidencePolicy.agent_type ---
    at = defs["EvidencePolicy"]["properties"]["agent_type"]["items"]
    _set_enum(at, v.agent_type_values)

    # --- ExplanationPostFilter qualifier sub-schemas ---
    # These inherit from Qualifiers already via $ref, so nothing to do.

    # Record the injected Biolink version for downstream introspection
    schema.setdefault("x-biolink", {})["version"] = v.biolink_version

    return schema


def compile_schema_to_file(
    output_path: Path,
    base_schema_path: Optional[Path] = None,
    vocab: Optional[BiolinkVocabulary] = None,
) -> Path:
    """Write the compiled schema to disk. Useful for debugging or shipping a snapshot."""
    schema = compile_schema(base_schema_path=base_schema_path, vocab=vocab)
    output_path.write_text(json.dumps(schema, indent=2))
    return output_path


if __name__ == "__main__":
    schema = compile_schema()
    print(f"Compiled schema against Biolink v{schema['x-biolink']['version']}")
    entity_cats = schema["$defs"]["Entity"]["properties"]["biolink_category"]["enum"]
    aspect_vals = schema["$defs"]["Qualifiers"]["properties"]["object_aspect_qualifier"]["enum"]
    causal_vals = schema["$defs"]["Qualifiers"]["properties"]["causal_mechanism_qualifier"]["enum"]
    print(f"Category enum:            {len(entity_cats)} values")
    print(f"Aspect qualifier enum:    {len(aspect_vals)} values")
    print(f"Causal mechanism enum:    {len(causal_vals)} values")
