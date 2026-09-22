# Reverse Grounding

Ask a biomedical question in English. Get an answer built from a knowledge
graph, where every claim is checked against the graph that produced it.

A language model drafts the query and helps judge evidence. It never writes
the answer. Each sentence is assembled from a template and verified —
identifier by identifier — against the result graph, and anything that cannot
be verified is withheld and reported as withheld.

```bash
make install
reverse-grounding --question "Which drugs treat psoriasis?"
```

---

## What it does

```
question → intake → plan → execute → diagnose → decide → (revise → …) → compose → answer
```

1. **Intake** checks the request: shape, rate limits, screening, topic scope,
   and how much evidence review to spend.
2. **The planner** turns the question into a validated, Biolink-compliant JSON
   **query plan** — entities, the paths between them, ranking rules.
3. **The executor** compiles the plan to TRAPI, queries
   [ARAX](https://arax.ncats.io), and returns ranked candidates with evidence.
4. **The controller** reads the result and picks exactly one move:

   | move | when |
   |---|---|
   | `accept` | results came back and the plan asked the right question |
   | `repair_plan` | the plan is at fault — a name that did not resolve, a triple the backend cannot answer, an entity declared and never used |
   | `relax_plan` | the query completed and found nothing, and one constraint can be loosened |
   | `retry_execution` | the run did not finish, so the plan has not been tested |
   | `report_absence` | the query was sound, complete, and the graph holds nothing |
   | `abandon` | no move remains |

5. **The composer** builds the answer from the returned graph and runs it
   through the grounding gate.

Revision is deliberately narrow. A relaxation names **one** constraint, and a
structural diff rejects any revision that changed more — which is what makes
"these results appeared once this constraint was loosened" a claim worth
making. Every plan is fingerprinted on its executable content, so the loop
cannot ask the same question twice.

---

## Install

Python 3.10+ and an OpenAI-compatible chat endpoint (a local
[Ollama](https://ollama.com) serving `gpt-oss:120b` by default).

```bash
git clone https://github.com/kaiwenho/Reverse_Grounding
cd Reverse_Grounding
make install        # the Biolink Model, then five editable installs
make test           # 559 tests; no model needed
```

`make install` first downloads `plan-core/data/biolink-model.yaml`, pinned to
`v4.4.3`. It is a dependency rather than a source file, so it is gitignored
and a fresh clone does not have it — and nothing works without it, because the
schema compiles its category and qualifier enums from that file at runtime.
`make biolink` fetches it on its own if you need to.

Editable installs matter beyond convenience: the controller runs the executor
as a **subprocess**, and with them installed that subprocess finds
`plan_executor` on its own. Without them every command needs a `PYTHONPATH`
and two `--executor-pythonpath` flags.

---

## Usage

### `reverse-grounding` — the front door

Enters the system where a request is meant to enter it, through intake.

```bash
reverse-grounding --question "Which drugs treat psoriasis?" --out runs/demo
```

Prints a JSON result with a fixed status (`completed`, `no_data`, `refused`,
`needs_clarification`, `rate_limited`, `invalid_request`, `budget_exhausted`,
`unavailable`), a message from a fixed table, and the answer document when
there is one. Exit code is `0` for an answer or an established absence, `1`
for anything not answered, `2` if the service could not run.

It accepts the same body an HTTP caller would send, and rejects the same
unknown fields:

```bash
echo '{"question": "Which drugs treat psoriasis?", "review_depth": "deep"}' \
  | reverse-grounding --payload -
```

`--review-depth` is `off`, `light`, `standard` or `deep`, and maps to fixed
server-side limits. A caller sets the name, never the numbers.

### `loop-controller` — the loop, without intake

One layer down. Useful in development; it skips rate limiting, screening and
scope, so it is not what a run representing the whole system should use.

```bash
loop-controller --question "Which drugs treat psoriasis?" --out runs/dev
loop-controller --plan my_plan.json --out runs/dev      # skip the planner
loop-controller --question "..." --dry-run              # offline, synthetic data
```

### `plan-executor` — one plan, once

```bash
plan-executor my_plan.json --out result.json
plan-executor my_plan.json --check                      # validate, no query
```

---

## Output

**The answer.** Statements, the candidate table, caveats, and a grounding
report saying what was checked and what was withheld.

Through `reverse-grounding` it arrives inside the result document, under
`answer`. It is not also written to disk: intake decides whether an answer
leaves at all, and a copy sitting beside the result would be the answer
existing outside the door that controls it. Through `loop-controller`, which
has no door above it, it is written as `answer.json`.

**`trace.json`** — the whole decision history. Per iteration: what the executor
reported, which moves were legal and *why each of the others was not*, what
was chosen, what the deterministic policy would have chosen instead, and what
it cost. Read this when a run surprises you; it is designed to answer "why did
it do that" without rerunning anything.

Through intake the trace is **redacted** before it is written: the question is
replaced by a salted digest, and the planner's prompts and raw responses are
dropped because they quote it. A biomedical question can identify a person,
and the digest is enough to correlate runs or follow up a report. Pass
`--log-raw-question` to keep the text.

**`iter*.plan.json` / `iter*.result.json` / `iter*.hints.json`** — the plan
sent on each iteration, what the executor returned, and what it suggested to
the controller.

See [`examples/`](examples) for one run of each file, with a walkthrough.

---

## Configuration

**A different model or endpoint.** The client posts OpenAI-shaped `messages`
to a configurable URL, so any compatible server works — you do not need a GPU
that fits a 120B model.

```bash
reverse-grounding --question "..." --model your-model-name \
    --executor-arg --ollama-url \
    --executor-arg https://your-endpoint/v1/chat/completions
```

**Let a model choose the loop's moves** with `--llm-controller`. Off by
default, so runs are reproducible. Either way the trace records what the
deterministic policy would have done, so the two can be compared.

**Budgets** are per run: `--max-iterations`, `--max-planner-calls`,
`--max-wall-clock`, `--max-arax-calls`, `--max-llm-calls` on `loop-controller`.

A question costs roughly 200 seconds, most of it in evidence review. Lower
`--review-depth`, or pass `--executor-arg --no-verify-literature`.

---

## Packages

| package | what it owns |
|---|---|
| [`plan-core`](plan-core) | the plan contract: JSON schema, pinned Biolink model, validators |
| [`planner`](planner) | natural language → a validated plan |
| [`plan-executor`](plan-executor) | plan → TRAPI → ARAX → ranked candidates with evidence |
| [`loop-controller`](loop-controller) | diagnose, decide, revise, compose; owns the grounding gate |
| [`query-intake`](query-intake) | request policy, identity, rate limits, screening, scope |

Each has its own README. `plan-core` is the only dependency the others share.

---

## Design notes

**Why ARAX rather than the whole Tier-0 graph.** ARAX serves a subset of
Tier-0 and exposes endpoints the full graph does not — pathfinding,
meta-knowledge lookups, per-hop decomposition. The plan contract is
backend-agnostic: swapping in another Translator endpoint means writing a new
executor, not a new planner or controller.

**TRAPI is generated, not drafted.** The model writes a plan against a schema
with a pinned Biolink version; `plan_executor/trapi_builder.py` derives TRAPI
from it. Malformed TRAPI is impossible by construction, and a plan is
something a human can read and argue with.

**Why the answer is templated.** The grounding gate builds a lexicon from the
graph ARAX returned — node identifiers, edge predicates, publication ids,
sources — and checks every statement against it. Identifiers a model wrote are
quarantined and may only appear in statements describing the *question*, never
in claims about what the graph asserts. An absence is treated as a claim about
the world and is refused unless the query was sound and ran to completion.

---

## Development

```bash
make test                       # all suites
cd loop-controller && pytest -q # one package
make clean
```

559 tests across four packages: 251 controller, 147 intake, 144 planner,
17 executor. They need no network and no model — the executor takes a mock
lookup and any object with a `choose()` in place of the model.

`loop-controller/tools/` holds the benchmark and the relaxation probes:

```bash
make probe                      # the plan that exercises relax_plan
```

`loop-controller/tools/probes/README.md` explains what each probe targets and
what to read afterwards.

---

## Roadmap

- **Contradiction detection.** The loop revises on lack of support. Detecting
  that the graph asserts something *contrary* to the hypothesis, and revising
  in response, needs a definition of what counts as a conflict and which move
  it maps to.
- **Evaluation harness.** The trace was designed as its input — per-iteration
  decisions, the policy counterfactual, override counts, grounding violations,
  cost. A gold set turns "does the loop help?" into a number.
- **Tests for `plan-core`.** The shared contract is the one package without a
  suite, and two bugs so far have lived in the seam between it and a consumer.
- **An HTTP server.** `query-intake` is built for one, and currently has a CLI.
- **Cost.** Literature checks run sequentially and dominate the wall clock.

---

## Licence

MIT.
