"""
Query Planner Agent — orchestration.

The agent:

1. Assembles the system prompt (from prompt_assembly.build_system_prompt).
2. Calls the LLM with the user's question and authoritative external-input manifest.
3. Parses the response as JSON.
4. Validates the parsed plan (schema + semantics).
5. On validation failure, retries ONCE with the errors appended to the prompt,
   asking the model to fix them.
6. If the retry also fails, returns a refusal with reason
   `requires_capability_not_available`.

Every attempted response and its own validation result are retained in
`PlannerResult.attempt_history` for smoke-test and production diagnostics.

The agent is model-agnostic — it takes any object satisfying the
`LLMClient` protocol (a single `complete(system, user) -> str` method).
Wiring gpt-oss (or any other model) is a matter of writing a 20-line adapter.

Nothing in this module makes network calls directly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol

from plan_core import PLAN_VERSION, QueryPlan
from .prompt_assembly import build_system_prompt
from plan_core import ValidationError, ValidationResult, validate_plan

from plan_core import load_biolink_vocabulary


_RANKING_REPAIR_HINT = """\
Ranking blocks (place only the mode-appropriate blocks inside `ranking`):

Discovery-mode candidate block:
```json
{
  "candidate_ranking": {
    "explanation_influence": "none",
    "rationale": "This discovery-only plan has no explanation queries, so candidate order is determined only by discovery evidence.",
    "discovery": {
      "strategy": "evidence_weighted",
      "criteria": [
        {
          "name": "knowledge_level",
          "direction": "desc",
          "origin": "planner_recommended",
          "application": "rank",
          "scope": "portable",
          "preferred_values": [
            "knowledge_assertion",
            "logical_entailment",
            "observation",
            "statistical_association"
          ],
          "rationale": "Prefer directly asserted or observed evidence."
        }
      ],
      "top_k": 25
    }
  }
}
```

Explanation-mode block:
```json
{
  "explanation_ranking": {
    "strategy": "explanation_diversity",
    "criteria": [
      {
        "name": "distinct_intermediate_categories",
        "direction": "desc",
        "origin": "planner_recommended",
        "application": "rank",
        "scope": "portable",
        "rationale": "Prefer diverse explanation paths."
      }
    ],
    "top_k": 5
  }
}
```

Discovery mode uses only `candidate_ranking`; explanation mode uses only
`explanation_ranking`; hybrid mode uses both. In a hybrid repair, preserve the
existing question-supported `explanation_influence` value rather than copying
`none` from the discovery example. Never put `strategy`, `criteria`, or `top_k`
directly under `candidate_ranking`. A `knowledge_level` criterion always
includes ordered `preferred_values`. `num_explanation_paths` counts paths per
candidate and is valid only in `candidate_ranking.final`; never put it in
`explanation_ranking`, which ranks individual paths.
"""


_EXPLANATION_RETURN_REPAIR_HINT = """\
Explanation-query return shape:
```json
{
  "query_id": "EQ1",
  "rationale": "Retrieve KG-supported paths between the fixed endpoints without asserting an unstated relationship.",
  "endpoint_a": {"binding_type": "entity", "entity_ref": "E1"},
  "endpoint_b": {"binding_type": "entity", "entity_ref": "E2"},
  "max_hops": 4,
  "return": {
    "top_k_paths": 5,
    "rank_by": "composite",
    "group_by_intermediate_category": true
  }
}
```
`top_k_paths` belongs only inside `return`; `return.top_k` is not valid.
"""


_ENDPOINT_REPAIR_HINT = """\
Endpoint-binding shapes:
```json
{"binding_type": "entity", "entity_ref": "E1"}
```
```json
{"binding_type": "from_discovery", "from_path_ids": ["P1", "P2"], "fanout_top_k": 20}
```
Use the second shape only in hybrid mode. Do not use singular `from_path_id`
or endpoint `top_k`. For `annotate_only` or `rerank`, its explicit
`fanout_top_k` must be at least `candidate_ranking.discovery.top_k`, so every
returned or preliminary candidate receives the explanation query.
"""


_PATH_REPAIR_HINT = """\
Discovery Path shape:
```json
{
  "path_id": "P1",
  "archetype_tag": "Q1_target_based",
  "rationale": "Find candidate drugs connected to the fixed target.",
  "hops": [
    {
      "subject_ref": "candidate_drug",
      "predicate": "biolink:directly_physically_interacts_with",
      "object_ref": "target_gene"
    }
  ],
  "return_entity_ref": "candidate_drug",
  "expected_result_category": "SmallMolecule"
}
```
A discovery `Path` has an explicit `hops` array of one to five edges. It has
no `max_hops` field and no `constraints` object. Only an `ExplanationQuery`
has `max_hops`.
"""


_ENTITY_CONSTRAINT_REPAIR_HINT = """\
Entity constraint placement:
```json
{
  "entity_ref": "candidate_drug",
  "name": "any approved drug",
  "biolink_category": "Drug",
  "is_variable": true,
  "constraints": [
    {"field": "approval_status", "op": "eq", "value": "approved"}
  ]
}
```
`constraints` is an array inside the affected Entity. There is no top-level
`entity_constraints` field and a Path cannot contain entity constraints.
"""


_CORE_GRAPH_REPAIR_HINT = """\
Core entity and hop fields:
- Every entity keeps its existing question-derived values and uses the exact
  keys `entity_ref`, `name`, `biolink_category`, and `is_variable`.
- Every hop keeps its existing endpoints and uses the exact keys `subject_ref`,
  `predicate`, and `object_ref`.
Do not emit `Entity.role`; do not rename identifiers or replace surface names
while repairing structure.
"""


_EXTERNAL_INPUT_REPAIR_HINT = """\
External-input rules:
- Gene names and source directions listed directly in the question are inline
  values, not an external input. Create one fixed Gene entity and one opposite-
  direction expression path per named member; preserve surface names, omit
  input_binding, and let the resolver ground them.
- An external `input_ref` is usable only when it appears in the authoritative
  available-input manifest included with this request. Wording such as
  "the supplied dataset" does not establish availability by itself.
- If the required input is absent, emit a `needs_clarification` refusal asking
  for the dataset or a stable input binding; never invent an `input_ref`.
- A `directional_gene_signature` binding requires `direction_filter` with one
  source-signature direction. Reuse the same input_ref in two fixed Gene
  entities when both increased and decreased partitions are needed.
"""


_REFUSAL_REPAIR_HINT = """\
Refusal shape:
```json
{
  "question": "Preserve the user's question verbatim.",
  "plan_mode": "discovery",
  "interpretation": {
    "archetypes": ["OTHER"],
    "restated_question": "State what was understood.",
    "intent": "discovery"
  },
  "entities": [],
  "confidence": {
    "level": "low",
    "reasons": ["State the concrete refusal anchor."]
  },
  "refusal": {
    "reason": "needs_clarification",
    "message": "State exactly what information is needed."
  }
}
```
A refusal still requires `entities: []`, `interpretation`, and `confidence`.
Omit paths, explanation_queries, ranking, aggregation, and every other unused
optional field. Never emit those fields as null. Use
`unsafe_or_clinical_advice`, not `out_of_scope`, for individualized medication,
dosing, or treatment advice.
"""


_EVIDENCE_REPAIR_HINT = """\
Evidence metadata shapes:
```json
{"feature": "primary_knowledge_source", "origin": "planner_recommended", "application": "report", "rationale": "Report KG provenance."}
```
```json
{"origin": "user_requested", "application": "filter", "rationale": "The user explicitly requested this hard filter."}
```
Use `evidence_recommendations` for reporting/ranking support. Use
`evidence_policy` only for a hard filter explicitly requested by the user.
"""

# ------------------------------------------------------------------------
# LLM client protocol — wire in gpt-oss (or anything else) via adapter
# ------------------------------------------------------------------------

class LLMClient(Protocol):
    def complete(self, system: str, user: str) -> str:
        """Return the model's response text. Should aim for JSON-only output."""
        ...


# ------------------------------------------------------------------------
# Agent result
# ------------------------------------------------------------------------

@dataclass(frozen=True)
class PlannerAttempt:
    """One LLM response and the validation outcome produced from it."""

    number: int
    raw_response: str
    validation: Optional[ValidationResult]
    error: Optional[str] = None


@dataclass
class PlannerResult:
    ok: bool
    plan: Optional[QueryPlan]
    raw_response: str
    validation: ValidationResult
    attempts: int  # 1 or 2
    error: Optional[str] = None  # non-validation failure (JSON parse, LLM error)
    attempt_history: List[PlannerAttempt] = field(default_factory=list)


# ------------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------------

class PlannerAgent:
    def __init__(
        self,
        llm: LLMClient,
        exemplar_plans: Optional[Iterable[dict]] = None,
        include_full_schema: bool = False,
        archetype_detail: str = "standard",
    ):
        self.llm = llm
        self._system_prompt = build_system_prompt(
            include_full_schema=include_full_schema,
            exemplar_plans=exemplar_plans,
            archetype_detail=archetype_detail,
        )

    @property
    def system_prompt(self) -> str:
        return self._system_prompt

    # ------------------------------------------------------------------

    def plan(
        self,
        question: str,
        *,
        available_inputs: Optional[Iterable[dict]] = None,
    ) -> PlannerResult:
        """
        Run the full planner flow for one question.

        ``available_inputs`` is an authoritative manifest of external inputs
        that the eventual query workflow can access. A dataset merely named in
        question text is not considered available unless listed here.

        Attempt 1: send prompt + question, parse + validate.
        Attempt 2 (only on validation failure): append errors, ask for a fix.
        """
        input_manifest = _normalize_available_inputs(available_inputs)
        user_request = _build_user_request(question, input_manifest)
        raw1 = self.llm.complete(self._system_prompt, user_request)
        parsed, parse_err = _try_parse(raw1)
        if parse_err:
            attempt1 = PlannerAttempt(1, raw1, None, parse_err)
            return PlannerResult(
                ok=False, plan=None, raw_response=raw1,
                validation=ValidationResult(ok=False, errors=[]),
                attempts=1, error=parse_err,
                attempt_history=[attempt1],
            )
        _stamp_versions(parsed)
        result1 = _validate_with_available_inputs(parsed, input_manifest)
        attempt1 = PlannerAttempt(1, raw1, result1)
        if result1.ok:
            plan = QueryPlan.model_validate(parsed)
            return PlannerResult(
                ok=True, plan=plan, raw_response=raw1,
                validation=result1, attempts=1,
                attempt_history=[attempt1],
            )

        # Retry once with errors appended
        retry_user = _build_retry_message(
            question,
            raw1,
            result1,
            available_inputs=input_manifest,
        )
        raw2 = self.llm.complete(self._system_prompt, retry_user)
        parsed2, parse_err2 = _try_parse(raw2)
        if parse_err2:
            attempt2 = PlannerAttempt(2, raw2, None, parse_err2)
            return PlannerResult(
                ok=False, plan=None, raw_response=raw2,
                validation=result1, attempts=2, error=parse_err2,
                attempt_history=[attempt1, attempt2],
            )
        _stamp_versions(parsed2)
        result2 = _validate_with_available_inputs(parsed2, input_manifest)
        attempt2 = PlannerAttempt(2, raw2, result2)
        if result2.ok:
            plan = QueryPlan.model_validate(parsed2)
            return PlannerResult(
                ok=True, plan=plan, raw_response=raw2,
                validation=result2, attempts=2,
                attempt_history=[attempt1, attempt2],
            )

        return PlannerResult(
            ok=False, plan=None, raw_response=raw2,
            validation=result2, attempts=2,
            attempt_history=[attempt1, attempt2],
        )


# ------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------

_EXTERNAL_INPUT_FORMATS = {
    "curie_list",
    "directional_gene_signature",
    "variant_list",
    "screening_panel",
}
_SIGNATURE_DIRECTIONS = {
    "increased", "upregulated", "decreased", "downregulated",
}


def _normalize_available_inputs(
    available_inputs: Optional[Iterable[dict]],
) -> List[Dict[str, Any]]:
    """Validate and normalize the authoritative external-input manifest."""
    normalized: List[Dict[str, Any]] = []
    seen_refs: set[str] = set()
    for index, item in enumerate(available_inputs or []):
        if not isinstance(item, dict):
            raise TypeError(f"available_inputs[{index}] must be an object")
        unknown = set(item) - {
            "input_ref", "expected_format", "description", "directions",
        }
        if unknown:
            raise ValueError(
                f"available_inputs[{index}] has unsupported fields: "
                f"{sorted(unknown)}"
            )
        input_ref = item.get("input_ref")
        expected_format = item.get("expected_format")
        if not isinstance(input_ref, str) or not input_ref:
            raise ValueError(
                f"available_inputs[{index}].input_ref must be a non-empty string"
            )
        if input_ref in seen_refs:
            raise ValueError(f"duplicate available input_ref '{input_ref}'")
        if expected_format not in _EXTERNAL_INPUT_FORMATS:
            raise ValueError(
                f"available_inputs[{index}].expected_format must be one of "
                f"{sorted(_EXTERNAL_INPUT_FORMATS)}"
            )

        entry: Dict[str, Any] = {
            "input_ref": input_ref,
            "expected_format": expected_format,
        }
        description = item.get("description")
        if description is not None:
            if not isinstance(description, str) or not description.strip():
                raise ValueError(
                    f"available_inputs[{index}].description must be a "
                    "non-empty string"
                )
            entry["description"] = description

        directions = item.get("directions")
        if directions is not None:
            if expected_format != "directional_gene_signature":
                raise ValueError(
                    f"available_inputs[{index}].directions is valid only for "
                    "directional_gene_signature"
                )
            if (
                not isinstance(directions, list)
                or not directions
                or any(
                    not isinstance(direction, str)
                    or direction not in _SIGNATURE_DIRECTIONS
                    for direction in directions
                )
            ):
                raise ValueError(
                    f"available_inputs[{index}].directions must be a non-empty "
                    f"list drawn from {sorted(_SIGNATURE_DIRECTIONS)}"
                )
            if len(directions) != len(set(directions)):
                raise ValueError(
                    f"available_inputs[{index}].directions must be unique"
                )
            entry["directions"] = list(directions)

        normalized.append(entry)
        seen_refs.add(input_ref)
    return normalized


def _build_user_request(
    question: str,
    available_inputs: List[Dict[str, Any]],
) -> str:
    """Attach an authoritative, compact input manifest to the LLM request."""
    return (
        "BIOMEDICAL QUESTION (copy only this text into plan.question):\n"
        f"{question}\n\n"
        "AVAILABLE EXTERNAL INPUTS (authoritative manifest):\n"
        f"{json.dumps(available_inputs, indent=2)}\n\n"
        "Only input_ref values listed in this manifest are actually supplied. "
        "A dataset mentioned only in the biomedical question is unavailable."
    )


def _validate_with_available_inputs(
    plan: Dict[str, Any],
    available_inputs: List[Dict[str, Any]],
) -> ValidationResult:
    """Run plan validation, then verify external bindings against the manifest."""
    result = validate_plan(plan)
    errors = list(result.errors)
    manifest_by_ref = {
        item["input_ref"]: item for item in available_inputs
    }

    entities = plan.get("entities")
    if not isinstance(entities, list):
        return ValidationResult(ok=not errors, errors=errors)

    for entity_index, entity in enumerate(entities):
        if not isinstance(entity, dict):
            continue
        binding = entity.get("input_binding")
        if not isinstance(binding, dict):
            continue
        if binding.get("binding_type") != "external_input":
            continue

        input_ref = binding.get("input_ref")
        if not isinstance(input_ref, str):
            continue  # The JSON Schema reports missing/wrong types.
        available = manifest_by_ref.get(input_ref)
        if available is None:
            errors.append(ValidationError(
                location=f"entities/{entity_index}/input_binding/input_ref",
                message=(
                    f"external input_ref '{input_ref}' is not present in the "
                    "authoritative available-input manifest; do not infer "
                    "availability from question wording—emit a "
                    "needs_clarification refusal asking for the input"
                ),
                kind="semantic",
            ))
            continue

        expected_format = binding.get("expected_format")
        if (
            isinstance(expected_format, str)
            and expected_format != available["expected_format"]
        ):
            errors.append(ValidationError(
                location=f"entities/{entity_index}/input_binding/expected_format",
                message=(
                    f"input_ref '{input_ref}' is declared as "
                    f"'{available['expected_format']}' in the available-input "
                    f"manifest, not '{expected_format}'"
                ),
                kind="semantic",
            ))

        direction_filter = binding.get("direction_filter")
        manifest_directions = available.get("directions")
        if (
            isinstance(direction_filter, str)
            and isinstance(manifest_directions, list)
            and direction_filter not in manifest_directions
        ):
            errors.append(ValidationError(
                location=f"entities/{entity_index}/input_binding/direction_filter",
                message=(
                    f"direction '{direction_filter}' is not declared for "
                    f"input_ref '{input_ref}' in the available-input manifest"
                ),
                kind="semantic",
            ))

    return ValidationResult(ok=not errors, errors=errors)


def _try_parse(raw: str) -> tuple[Optional[dict], Optional[str]]:
    """Extract the first JSON object from `raw`. LLMs sometimes wrap with fences."""
    text = raw.strip()
    if text.startswith("```"):
        # strip the first fence line and the trailing fence
        first_nl = text.find("\n")
        text = text[first_nl + 1:] if first_nl != -1 else text
        if text.endswith("```"):
            text = text[: -3].rstrip()
    try:
        return json.loads(text), None
    except json.JSONDecodeError as e:
        return None, f"JSON parse error: {e}"

def _stamp_versions(parsed):
    """Override any versions the LLM guessed. Authoritative values come from us."""
    if isinstance(parsed, dict):
        parsed["plan_version"] = PLAN_VERSION
        parsed["biolink_version"] = load_biolink_vocabulary().biolink_version


def _schema_repair_hints(result: ValidationResult) -> List[str]:
    """Return only canonical fragments relevant to the failed schema paths."""
    hints: List[str] = []

    def add_once(hint: str) -> None:
        if hint not in hints:
            hints.append(hint)

    for error in result.errors[:20]:
        location = error.location
        message = error.message

        if location == "ranking" or location.startswith("ranking/"):
            add_once(_RANKING_REPAIR_HINT)

        if location.startswith("explanation_queries/"):
            if (
                "/return" in location
                or "top_k_paths" in message
                or "'top_k'" in message
                or "rank_by" in message
                or "group_by_intermediate_category" in message
            ):
                add_once(_EXPLANATION_RETURN_REPAIR_HINT)

            if (
                "/endpoint_a" in location
                or "/endpoint_b" in location
                or "endpoint_a" in message
                or "endpoint_b" in message
                or "from_path_id" in message
                or "fanout_top_k" in message
            ):
                add_once(_ENDPOINT_REPAIR_HINT)

        if location.startswith("paths/"):
            add_once(_PATH_REPAIR_HINT)
            add_once(_CORE_GRAPH_REPAIR_HINT)

        if location.startswith("entities/"):
            add_once(_CORE_GRAPH_REPAIR_HINT)
            if "/constraints" in location or "constraint" in message.lower():
                add_once(_ENTITY_CONSTRAINT_REPAIR_HINT)

        if location == "entity_constraints" or "entity_constraints" in message:
            add_once(_ENTITY_CONSTRAINT_REPAIR_HINT)

        if (
            "/input_binding" in location
            or "external input" in message.lower()
            or "available-input" in message.lower()
            or "direction_filter" in message
        ):
            add_once(_EXTERNAL_INPUT_REPAIR_HINT)

        if (
            location == "evidence_policy"
            or location.startswith("evidence_policy/")
            or location.startswith("evidence_recommendations/")
        ):
            add_once(_EVIDENCE_REPAIR_HINT)

    return hints


def _build_retry_message(
    question: str,
    first_response: str,
    result: ValidationResult,
    *,
    available_inputs: Optional[Iterable[dict]] = None,
) -> str:
    """Format the retry message with structured error feedback."""
    input_manifest = _normalize_available_inputs(available_inputs)
    request_context = _build_user_request(question, input_manifest)
    previous_response = first_response
    previous_plan, previous_parse_error = _try_parse(first_response)
    if previous_parse_error is None and isinstance(previous_plan, dict):
        # Versions belong to the orchestrator, not to model repair. Omitting
        # them here prevents a guessed value from being copied into attempt 2.
        previous_plan = dict(previous_plan)
        previous_plan.pop("plan_version", None)
        previous_plan.pop("biolink_version", None)
        previous_response = json.dumps(previous_plan, indent=2)

    error_lines = "\n".join(
        f"- [{e.kind}] {e.location}: {e.message}"
        for e in result.errors[:20]  # cap to keep prompt manageable
    )
    hints = _schema_repair_hints(result)
    if (
        isinstance(previous_plan, dict)
        and isinstance(previous_plan.get("refusal"), dict)
        and _REFUSAL_REPAIR_HINT not in hints
    ):
        hints.append(_REFUSAL_REPAIR_HINT)
    hint_block = ""
    if hints:
        hint_block = (
            "Relevant canonical schema shapes:\n\n"
            + "\n\n".join(hints)
            + "\n\n"
        )
    return (
        f"The previous plan failed validation.\n\n{request_context}\n\n"
        f"Your previous response, with runtime-owned version fields removed, "
        f"was:\n\n{previous_response}\n\n"
        f"Validation errors:\n\n{error_lines}\n\n"
        f"{hint_block}"
        "Correct the existing plan while preserving every semantic choice that "
        "was not identified as invalid, including entity and path identifiers. "
        "Remove unexpected fields and place each required field exactly as shown "
        "by the relevant canonical shape. Omit unused optional fields instead "
        "of emitting JSON null. Do not emit `plan_version` or "
        "`biolink_version`; the orchestrator supplies them. Every rationale must "
        "be a factual explanation specific to this question, never an instruction "
        "such as 'explain why'. "
        "Emit a corrected JSON plan. Do not include any commentary. "
        "If the errors indicate the question is fundamentally ambiguous or "
        "you cannot ground it, emit a refusal instead."
    )


# ------------------------------------------------------------------------
# Stub / echo client for wiring tests without a real model
# ------------------------------------------------------------------------

class EchoJSONClient:
    """Testing double: returns a fixed pre-canned JSON string, ignoring inputs."""
    def __init__(self, canned_response: str):
        self.canned_response = canned_response
        self.calls: List[tuple[str, str]] = []

    def complete(self, system: str, user: str) -> str:
        self.calls.append((system, user))
        return self.canned_response
