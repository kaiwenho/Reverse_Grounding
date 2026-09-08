# Planner design decisions

This document explains the detailed rules behind the Biomedical Query Planner
Agent. See the [README](../README.md) for installation and basic use.

## Biolink vocabulary

Biolink is the only source of truth for semantic vocabulary.
`biolink_vocab.py` reads the downloaded, version-pinned model from `plan-core/data/biolink-model.yaml` at startup. It extracts all
active predicates, categories, and qualifier enum values. It excludes
deprecated predicates.

The loader expands category mixins through both `is_a` and `mixins`. It removes
mixin, abstract, and deprecated classes from the resulting concrete
implementers. An adapter that requires concrete categories can call:

```python
load_biolink_vocabulary().concrete_implementers("GeneOrGeneProduct")
```

The adapter can use the returned active classes as a union. The plan keeps the
original mixin because it is the concise Biolink category intended by the
planner.

`compile_schema.py` adds the model-derived enums to the plan schema at runtime.
Upgrading Biolink therefore requires a new YAML model file, not a vocabulary
change in the planner code.

## Validation and retry

Validation has two layers.

The compiled JSON Schema catches structural errors. These include unknown
fields, wrong types, and invalid enum values.

The semantic validator checks rules that JSON Schema cannot express. It checks
for:

- duplicate IDs and dangling references
- disconnected paths and fixed discovery answers
- disabled hybrid inputs
- incomplete candidate-explanation fan-out
- executable content attached to a refusal
- predicate directions that conflict with Biolink domain and range metadata
- qualifier blocks outside the allowed predicate families

If validation fails, the planner retries once. It adds the structured errors
and relevant canonical shapes to the next prompt. `PlannerResult.attempt_history`
keeps every raw response, parse error, and validation result. The smoke test
writes every attempt to its output file.

The test suite also acts as a drift audit. It checks every predicate, category,
qualifier value, and gap type in the archetype catalog against the Biolink
model and plan schema. It checks result categories and explanation middle
categories through the Biolink class hierarchy. Deprecated predicates cannot
appear in planner source or examples.

## Entities and external inputs

### Surface names by default

The planner emits names such as `metformin` or `Alzheimer's disease` with a
Biolink category. A downstream service resolves those names. The planner does
not invent identifiers.

An `input_binding` can preserve an identifier list or external dataset supplied
by the caller. Every external reference must appear in the `available_inputs`
manifest passed to `PlannerAgent.plan`. Otherwise, the planner returns
`needs_clarification`.

### No human-in-the-loop

The pipeline never waits for a user. If a question lacks required information,
the planner returns `reason: "needs_clarification"`. Its message contains the
questions that would need answers.

### Contextual entity roles

Entity roles belong to each query context, not to the entity globally. Every
entity states whether it is variable. A discovery path uses
`return_entity_ref` to name its result. An explanation query uses `endpoint_a`
and `endpoint_b`.

This structure lets one entity be a fixed discovery anchor and an explanation
endpoint. Every active discovery path must contain at least one non-variable
entity. This prevents connected but unanchored all-variable queries.

A `from_discovery` binding becomes a concrete endpoint after the executor
selects and resolves a candidate. The explanation query then runs once for
each selected candidate.

### Self-loops

Self-loops are outside the planner's scope. Every hop must use different
subject and object references. An explanation query must use different direct
entities or non-overlapping discovery-path lists.

Questions that require a self-relationship are refused instead of approximated.
Multi-entity cycles remain valid within the hop limits. After grounding, the
executor must also reject different references that resolve to the same CURIE.

## Modeling graph paths

### Explicit modeling gaps

Biolink does not have a predicate for transcriptomic signature reversal or
therapeutic drug synergy. The planner records a `gaps` entry instead of using a
nearby but incorrect predicate. The entry names the gap and the chosen proxy.

For a directional signature binding, `direction_filter` selects one source
partition. Reversal maps increased genes to decreased drug-expression effects,
and decreased genes to increased effects.

If the question lists the genes and directions, the planner creates one fixed,
surface-name Gene entity and one opposite-direction path per member. It does
not create an external input binding. A downstream resolver grounds the names.

The planner can state the paths and required aggregation. It cannot invent
missing genes or directions, and it cannot calculate a connectivity score with
ordinary graph traversal.

### Canonical edge direction

Plans use canonical Biolink edge direction. The semantic validator compares
hop categories with inherited Biolink domain and range metadata. When useful,
it suggests a declared inverse. An executor may still search inverse edges
because knowledge graphs store some relationships in different directions.

### Category constraints

All categories come from the downloaded, version-pinned Biolink Model. This includes entity
categories, `expected_result_category`, and explanation-query middle-category
allowlists and blocklists.

An expected result may be broader than the returned entity. For example,
`ChemicalEntity` may describe returned `SmallMolecule` candidates. The two
categories cannot contradict each other. The planner does not emit
backend-specific category extensions.

### Qualified hops

Qualifier blocks are allowed only on `biolink:regulates`, `biolink:affects`,
and `biolink:interacts_with`. They are also allowed on active descendants found
through the downloaded model’s predicate hierarchy. Other predicates can still be used
without qualifiers.

If a relationship needs an unsupported predicate and qualifier combination,
the planner records a modeling limitation. It does not emit an invalid plan.

### Pinned-relationship fallbacks

The planner never invents a predicate between two fixed entities.

For a Q1 question with a known target and disease, discovery contains only the
drug-to-target hop. An open explanation query checks target-to-disease
relevance separately. Missing target-to-disease data therefore cannot remove
valid target-engaging candidates. If the explanation is empty, the candidates
remain and their disease relevance is marked unverified.

Other patterns may keep a user-stated, fixed relationship in discovery. Those
relationships also receive an open fallback. A question with only two fixed
endpoints gets a predicate-specific explanation query followed by an open
fallback.

A different fallback path provides diagnostic context. It does not prove the
relationship requested by the user.

## Confidence, ranking, and evidence

### Categorical confidence

`confidence.level` is `high`, `medium`, or `low`. It is based on structural
features such as the number of archetypes, entity ambiguity, and declared
gaps. `confidence.reasons` records which features applied.

The planner does not emit LLM-generated probabilities because those values are
not comparable across calls.

### Separate ranking stages

Candidate ranking and explanation ranking have different purposes.

`candidate_ranking.discovery` orders discovered entities. A single fixed
discovery path uses `evidence_weighted`. Consensus ranking needs at least two
materially different paths.

`explanation_ranking` orders paths across explanation queries. Each query's
`return` block also ranks and limits its own paths.

A hybrid plan sets `explanation_influence` to one of three values:

- `none`: explanations provide independent shared context
- `annotate_only`: explanations attach to candidates but do not change order
- `rerank`: explanation evidence may change candidate order

Only `rerank` includes `candidate_ranking.final`. Its criteria must use
explanation-derived information. Every ranking stage includes explicit
criteria and `top_k`.

Portable evidence criteria use returned fields such as `knowledge_level` and
knowledge-source counts. The executor-derived `edge_evidence_strength` value
is optional and versioned.

### Evidence guidance and policy

`evidence_recommendations` tells any user or query system which metadata to
rank or display. It normally includes `primary_knowledge_source` and
`knowledge_level`.

A primary knowledge source is the upstream graph source that asserted an edge.
It does not mean primary literature.

`evidence_policy` is reserved for hard filters that the user requested. It
records the user request and its rationale. The planner may recommend ranking
or displaying evidence, but it never removes evidence without a user-requested
policy.

### Approval constraints

When the user asks for approved drugs, the plan includes
`approval_status eq approved`. This means currently marketed for any
indication. It does not mean approved for the disease in the question.

The planner uses `ever_approved` only when the user includes formerly approved,
discontinued, or withdrawn drugs. Approval is the only entity-level constraint
in the current plan contract.

These values are plan-level meanings, not Biolink slots or detailed
`ApprovalStatusEnum` values. An executor may map them to available backend
attributes. For example, ARAX may use ChEMBL availability. The executor must
report the mapping and must not weaken the filter when no candidates pass.
The planner reports other requested entity filters as unsupported instead of
inventing them.

### Hybrid explanation coverage

A `from_discovery` endpoint names one or more discovery paths with
`from_path_ids`. Each path still controls its own `return_entity_ref`.
Candidate sets are combined, deduplicated, and ordered by
`candidate_ranking.discovery`.

The binding's `fanout_top_k` then selects candidates for explanation. This
limit differs from the final result limit. For `annotate_only` and `rerank`, it
must be at least `candidate_ranking.discovery.top_k`. Every returned candidate
must be annotated, and every preliminary candidate must be explained before a
fair reranking.

With `annotate_only`, explanations do not change candidate order. With
`rerank`, the executor applies `candidate_ranking.final` afterward.

One binding may combine different entity references only when their Biolink
categories are compatible. `attach_explanations_to_candidates` is true only
for candidate fan-out. Together, the bindings must cover every active
discovery path.

## Refusals

A refusal still has a complete schema shape. It includes `entities`, usually
as an empty list, plus `interpretation`, `confidence`, and a typed `refusal`
object. It leaves out paths, ranking, aggregation, and other unused optional
fields.

The planner uses `unsafe_or_clinical_advice` for individualized medication,
dose, and treatment-selection requests. It uses `out_of_scope` for
non-clinical scope mismatches.
