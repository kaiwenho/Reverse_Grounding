# Biomedical Query Planner Agent

The Biomedical Query Planner Agent turns a natural-language biomedical
question into a validated, Biolink-compliant JSON query plan.

The plan has two uses. It explains the query to a reader who does not know
Biolink, and it gives a knowledge-graph executor a machine-readable contract.
The planner does not resolve entity names or run graph queries.

```text
Natural-language question
          ↓
     Planner agent
          ↓
 Validated QueryPlan JSON
          ↓
 Knowledge-graph executor
```

## What it does

The planner maps a question to one of eleven biomedical archetypes:
target-based repurposing, human genetic evidence, signature reversal,
side-effect repurposing, disease similarity, pathway or mechanism of action,
real-world evidence, drug combination, safety or pharmacokinetics, clinical
trial landscape, and indication lookup.

It then creates a plan with the Biolink paths, ranking rules, evidence
guidance, and known modeling gaps needed to answer the question.

Plans are independent of any specific knowledge graph. They use Biolink
categories and predicates instead of backend-specific vocabulary. Names are
resolved and queries are executed downstream. Identifiers and external data
appear only when the caller supplies them through an `input_binding`.

### Plan modes

| Mode | Use it when | Output |
|---|---|---|
| `discovery` | The question asks which entities match | Paths with variable answer entities |
| `explanation` | The question asks why two known entities are related | Two-endpoint path-finding queries |
| `hybrid` | The question combines discovery and explanation | Discovery paths plus linked explanation queries |

## Install

Requires Python 3.10 or later. Main dependencies are `pyyaml`, `jsonschema`,
and `pydantic`.

```bash
git clone <repo-url>
cd planner_agent
pip install -e ".[dev]"
```

## Quick start

The planner can use any LLM client that implements
`complete(system, user) -> str`.

```python
from planner_agent import PlannerAgent
from planner_agent.ollama_client import OllamaGPTOSSClient

agent = PlannerAgent(llm=OllamaGPTOSSClient(model="gpt-oss:120b"))
result = agent.plan("Which drugs treat dermatitis herpetiformis?")

if result.ok:
    print(result.plan.plan_mode)                  # "discovery"
    print(result.plan.interpretation.archetypes)  # ["Q11_indication_lookup"]
    print(result.plan.confidence.level)           # "high"
else:
    print(result.validation.format())
```

### Authoritative inputs

Pass an input manifest when a question depends on an external dataset:

```python
result = agent.plan(
    "Using tnbc_signature, find compounds predicted to reverse it.",
    available_inputs=[{
        "input_ref": "tnbc_signature",
        "expected_format": "directional_gene_signature",
        "directions": ["increased", "decreased"],
    }],
)
```

Mentioning a “supplied” dataset in the question does not make it available.
The manifest must identify it. If the question lists the signature members and
their directions directly, no manifest is needed. The planner keeps those
names as fixed entities, and the downstream resolver supplies identifiers.

### Run with Ollama

```bash
ollama pull gpt-oss:120b
ollama serve
PYTHONPATH=../plan-core/src:src python smoke_test.py
```

Use `generate_plan.py` to produce one complete, validated JSON plan:

```bash
PYTHONPATH=../plan-core/src:src python generate_plan.py \
  "Which approved drugs inhibit human JAK2? Return the top 25 candidates." \
  --out ../plan-executor/plans/jak2_plan.json
```

Omit `--out` to print only the JSON plan to standard output. For a
dataset-backed question, use `--available-inputs FILE` with a JSON-array
manifest.

The planner adds the authoritative `plan_version` and `biolink_version`
values. The LLM does not generate these values.

## Example plans

Complete examples are available in [`tests/fixtures`](tests/fixtures/):

- [Target-based discovery](tests/fixtures/example_q1_target_based.json)
- [Signature reversal](tests/fixtures/example_q3_signature_reversal.json)
- [Explanation](tests/fixtures/example_q7_explanation.json)
- [Hybrid planning](tests/fixtures/example_hybrid_ipf.json)
- [Clarification refusal](tests/fixtures/example_refusal_needs_clarification.json)

## Plan contents

The exact fields depend on the plan mode and question. These are the main
parts of the contract:

| Field | Purpose |
|---|---|
| `plan_version`, `biolink_version` | Identify the schema and vocabulary used |
| `interpretation` | Records the question archetype and planner interpretation |
| `entities` | Defines names, Biolink categories, and variable status |
| `paths` | Describes discovery paths and their answer entities |
| `explanation_queries` | Connects fixed endpoints or discovered candidates |
| `ranking`, `aggregation` | Defines how to combine and order results |
| `evidence_recommendations` | Lists evidence fields to rank or display |
| `evidence_policy` | Holds hard evidence filters requested by the user |
| `gaps` | Records concepts that Biolink cannot represent directly |
| `confidence` | Gives a categorical level with structural reasons |
| `refusal` | Explains why the planner cannot produce an executable plan |

A plan includes only the executable fields it needs. For example, an
explanation-only plan does not need discovery paths. A refusal keeps the
required metadata but omits paths, ranking, and aggregation.

## How it works

```text
question → prompt assembly → LLM draft → validation → QueryPlan
              ↑                         │
       catalog + Biolink + schema       └─ retry once on failure
```

1. **Build the prompt.** The planner combines the question with the archetype
   catalog, Biolink vocabulary, and plan schema.
2. **Create a draft.** The LLM returns a proposed JSON plan.
3. **Validate the plan.** JSON Schema checks its structure. The semantic
   validator checks relationships across fields, including path connectivity,
   entity roles, Biolink direction, qualifiers, and hybrid bindings.
4. **Retry once if needed.** The planner sends structured validation errors and
   relevant canonical shapes back to the LLM. `attempt_history` keeps each raw
   response and validation result.

Biolink is the source of truth for categories, predicates, and qualifier
values. The vocabulary loader excludes deprecated terms. The schema compiler
injects the active values into the plan schema. To upgrade Biolink, replace the
downloaded, version-pinned model file.

### Planner boundaries

The planner defines query intent and semantic structure. It does not:

- resolve surface names to CURIEs
- choose backend-specific categories or predicates
- run graph queries or join graph results
- apply ranking rules to returned candidates
- give a biomedical or clinical answer to the user

These tasks belong to the resolver, executor, or controlling application. This
separation keeps the plan portable across compatible knowledge graphs.

See [Design decisions](docs/design.md) for the complete planning and validation
rules.

## Configuration

Choose how much archetype guidance to include in the prompt:

```python
PlannerAgent(llm=..., archetype_detail="standard")
```

| Level | Approximate tokens | Contents |
|---|---:|---|
| `slim` | 3.2K | Definitions, signals, and examples |
| `standard` | 5.7K | Slim content plus key predicates, qualifiers, and modeling gaps |
| `full` | 8.9K | Standard content plus all worked Biolink path templates |

Set `include_full_schema=True` to include the complete compiled JSON Schema.
This adds about 17K tokens and is mainly useful for debugging format errors.

## Key behavior

- The planner uses surface names by default. Name resolution belongs
  downstream.
- It never pauses for user input. An underspecified question produces a typed
  `needs_clarification` refusal.
- Confidence is `high`, `medium`, or `low` and is tied to structural features
  of the plan.
- Unsupported Biolink relationships are recorded as modeling gaps. The planner
  does not silently substitute a similar predicate.
- Candidate ranking and explanation ranking are separate. Explanations affect
  candidate order only when the plan explicitly requests reranking.
- Evidence recommendations describe what to rank or display. Hard evidence
  filters are used only when the user requests them.

The detailed rules for entity roles, qualified hops, fallbacks, approvals,
hybrid fan-out, refusals, and self-loops are in
[docs/design.md](docs/design.md).

## Project layout

```text
data/
  biolink-model.yaml          Downloaded Biolink vocabulary (not committed)
  archetype_catalog.json      Definitions and paths for 11 archetypes
schema/
  query_plan.schema.json      Plan schema
src/planner_agent/
  biolink_vocab.py            Loads Biolink vocabulary
  compile_schema.py           Adds Biolink enums to the schema
  archetype_catalog.py        Loads and formats the catalog
  plan_models.py              Pydantic models
  validators.py               Schema and semantic validation
  prompt_assembly.py          Builds the system prompt
  planner_agent.py            Draft, validate, and retry workflow
  ollama_client.py            Ollama adapter
docs/
  design.md                   Detailed planning and validation rules
generate_plan.py              One-question JSON plan generator
tests/
  test_scaffolding.py         Test suite
  fixtures/                   Complete example plans
```

## Testing

From the `planner/` directory, run:

```bash
PYTHONPATH=../plan-core/src:src python -m pytest tests/test_scaffolding.py -q
````

The suite covers vocabulary loading, schema compilation, example plans,
Pydantic round trips, prompt assembly, retries, and refusals. Audit tests also
check catalog terms against the downloaded Biolink Model and schema. These tests
help detect drift when the catalog or Biolink version changes.

## Versioning

| Artifact | Version | Notes |
|---|---|---|
| Plan schema | 0.10.0 | Inline signatures, explanation fan-out, and canonical refusals |
| Archetype catalog | 0.5.0 | Inline and manifest-backed signature inputs |
| Biolink Model | 4.4.3 | [Download instructions](../plan-core/data/README.md) |

Every plan records `plan_version` and `biolink_version`. The planner stamps
both values programmatically.

## Roadmap

- Few-shot prompt examples after domain-expert review
- Gold-set evaluation harness
- Downstream query-executor agent

## License

TBD
