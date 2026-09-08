# plan-executor

`plan-executor` runs a query plan against the
[ARAX](https://arax.ncats.io) biomedical knowledge graph. It returns ranked
answers with supporting evidence.

The [planner](https://github.com/kaiwenho/Biolink_Planner_Agent) turns a
question into a plan. This project executes that plan. It resolves names to
identifiers, sends queries, applies the plan's filters and ranking rules, and
reports the result.

The planner and executor share the plan contract through `plan-core`. They are
otherwise independent. You can execute a hand-written plan without installing
the planner.

---

## Design guarantees

### Clear outcomes

A query can return no candidates for several reasons. The graph may have no
matching data. A multi-hop query may find data for each hop but no shared
intermediate. A knowledge provider may time out. The plan's filters may remove
all candidates.

The executor reports a typed outcome for each case. It also states whether the
planner may be able to fix the problem.

### Reviewed entity resolution

The name resolver may rank the wrong entity first. Using that result without
review can produce confident answers about the wrong disease, gene, or drug.

The executor therefore requires an LLM to choose among resolution candidates.
If the LLM is unavailable, the run stops. It does not fall back to the first
candidate. A plan may also provide an identifier directly or refer to an
external input.

### Evidence-grounded reranking

The LLM may reorder leading candidates. Each reason it gives must cite
retrieved edges that support that candidate. It must also refer to information
contained in those edges. The executor rejects reasons based on recalled facts
or source names alone. If a reason fails these checks, the deterministic order
remains in place.

### Checked literature evidence

When literature verification is enabled, the executor checks abstracts linked
to leading edges. A citation alone is not enough because the cited abstract
may not support the edge. A literature verdict requires a verbatim quote that
the executor can find in the retrieved abstract.

---

## Install

Requirements:

- Python 3.10 or later
- A running [Ollama](https://ollama.com) instance with a tool-capable model
- Network access to ARAX

Clone this repository and `plan-core` as sibling directories:

```bash
git clone <this repo> plan-executor
git clone <plan-core repo> plan-core

cd plan-executor
pip install requests httpx
```

`plan-core` is required. It provides the plan schema and Biolink model. Its
validation prevents invalid categories or predicates from reaching ARAX.

You can run the executor without installing either project:

```bash
PYTHONPATH=../plan-core/src:src python -m plan_executor.run_plan plan.json --check
```

You can also install both projects in editable mode:

```bash
pip install -e ../plan-core
pip install -e .
```

Changes to `plan-core` will then be available to both projects immediately.

---

## Examples

Complete examples are stored separately because execution output can be large.

- [Example plan](examples/gluten_plan.json)
- [Example output](examples/gluten_output.json)

---

## Run a plan

Start with a check:

```bash
python -m plan_executor.run_plan plan.json --check
```

This command validates the plan, checks the LLM connection, and confirms that
ARAX supports each hop. It does not run a graph query.

Next, run discovery without the slower LLM stages:

```bash
python -m plan_executor.run_plan plan.json \
    --no-explain --no-verify-literature --no-rerank \
    --out runs/result.json --hints runs/hints.json
```

This step helps separate plan or graph problems from model problems.

Then run the full workflow:

```bash
python -m plan_executor.run_plan plan.json \
    --verify-top-k 20 --rerank-top-k 20 --max-abstracts 3 \
    --out runs/result.json --hints runs/hints.json
```

A full run may take several minutes. Each pathfinder search usually takes
30–60 seconds, and each abstract requires a separate LLM call.

You can safely interrupt a run. Completed work is cached, so the next run can
resume without repeating it.

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | Results were found, or the run established `no_answer` |
| 1 | The run was inconclusive, unexecutable, or based on an invalid plan |
| 2 | The LLM was unavailable, so entity resolution could not continue |

---

## Execution flow

The executor runs inexpensive checks before expensive queries and LLM calls.
Any stage may stop the run when it finds a blocking problem.

1. **Load and validate the plan.** `plan-core` checks the schema and Biolink
   values. The executor then checks its own requirements. Every path needs a
   pinned anchor, a discovery answer must be variable, and `max_hops` must stay
   within the pathfinder limit.
2. **Check services and capabilities.** The executor checks the LLM connection
   and compares each hop with ARAX's meta knowledge graph. If ARAX does not
   support a hop, the report shows the relationships it does support for those
   categories.
3. **Resolve entities.** The executor converts names to CURIEs and asks the LLM
   to choose among candidates. It uses human entities by default unless the
   plan requests another taxon.
4. **Execute paths.** The executor tries the full query first. If it times out,
   it runs each hop separately and joins them through a shared intermediate.
   If a multi-hop query is empty, it also checks each hop to locate the gap.
5. **Check concepts.** The executor confirms that results refer to the pinned
   concepts. This matters because ARAX may normalize or merge identifiers.
6. **Apply the evidence policy.** The executor applies hard filters requested
   by the user. It counts removals by rule and distinguishes a failing
   attribute from a missing attribute.
7. **Rank candidates.** The executor first applies the plan's ranking rules. It
   may then verify literature for leading candidates and ask the LLM to reorder
   them.
8. **Find explanations.** If the plan requests explanations, the executor
   searches for mechanistic paths between the specified endpoints.
9. **Assemble the output.** The executor writes one JSON result document.

---

## Read the output

`--out` writes the full result. It includes:

- ranked candidates and their supporting evidence
- per-path outcomes and hop counts
- entity-resolution choices and alternatives
- filter counts
- execution cost information

`--hints` writes a smaller file for the agent that controls the workflow. It
contains only information that may require action.

Read the `outcome` field first:

| Outcome | Meaning | Replan? |
|---------|---------|---------|
| `results` | The query returned candidates | — |
| `plan_refused` | The planner declined the question | No |
| `invalid_plan` | Schema or Biolink validation failed | **Yes** |
| `unresolved_grounding` | A named entity could not be resolved | **Yes** |
| `unsupported_backend_capability` | ARAX cannot answer a requested hop | **Yes** |
| `join_failure` | The hops returned data but had no shared intermediate | No |
| `filters_removed_all_candidates` | The evidence policy removed all candidates | No |
| `truncated_or_timed_out` | A query did not finish, so absence is not proven | No |
| `no_graph_data` | The graph had no data for the query as written | No |

The `replannable` field is true for the three outcomes marked **Yes**. The
other outcomes describe the backend, execution, or available data rather than
a plan error.

---

## Useful options

| Option | Purpose |
|--------|---------|
| `--check` | Validate the plan and check dependencies without running a query |
| `--mock` | Run offline with synthetic data |
| `--no-explain` | Skip explanation queries, which are often the most expensive stage |
| `--no-verify-literature` | Skip abstract verification |
| `--no-rerank` | Keep the order produced by the plan's ranking rules |
| `--taxon NCBITaxon:10090` | Resolve genes in another species; human is the default |
| `--conflate gene_protein` | Treat a gene and its protein product as one concept |
| `--timeout 3` | Set the query timeout; a short value can test decomposition |
| `--skip-direct-over-hops 3` | Decompose long paths without waiting for a direct-query timeout |
| `--dump-queries FILE` | Save the exact TRAPI queries sent to ARAX |
| `--cache PATH` | Set the SQLite cache path; the default is `runs/cache.sqlite` |
| `--no-cache` | Bypass the cache |

### Cache behavior

The cache stores raw responses, not filtered results. A new evidence policy
can therefore reuse a response and apply new filters without sending the query
again.

The cache also stores empty responses and timeouts. An empty response is a
useful result. A cached timeout tells the executor to skip the direct query
and start with decomposition on the next run.

Use these commands to inspect or clear the cache:

```bash
python -m plan_executor.cache stats runs/cache.sqlite
python -m plan_executor.cache purge runs/cache.sqlite
```

---

## Modules

| File | Responsibility |
|------|----------------|
| `run_plan.py` | Command-line interface and stage orchestration |
| `plan_input.py` | Plan loading, validation, and executability checks |
| `meta_kg.py` | Checks whether ARAX supports each hop |
| `resolver.py` | Entity resolution |
| `trapi_builder.py` | Converts plan paths into TRAPI query graphs |
| `arax_client.py` | Sends requests and classifies responses |
| `executor.py` | Executes, decomposes, and joins queries |
| `postfilter.py` | Applies evidence policies and concept checks |
| `evidence.py` | Retrieves and verifies publication evidence |
| `rank.py` | Ranks and reranks candidates |
| `explain.py` | Finds paths between endpoints |
| `aggregate.py` | Assembles the final result |
| `cache.py` | Stores cached responses |
| `config.py` | Defines endpoints, limits, and scoring settings |
| `llm.py` | Provides the Ollama LLM client |
