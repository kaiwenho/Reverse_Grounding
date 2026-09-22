# loop-controller

`loop-controller` runs a question to a graph-grounded answer. It asks the
[planner](../planner) for a plan, runs it through the
[executor](../plan-executor), reads what came back, decides what to do next,
and — when there is an answer — composes it from the result graph without a
language model writing a word of it.

```text
question ──► planner ──► plan ──► executor ──► result + hints
                ▲                                   │
                │                                   ▼
                └──── repair / relax ◄──────── diagnose ──► decide
                                                            │
                       accept / absence / abandon ◄──────────┘
                                    │
                                    ▼
                            answer, grounded
```

The planner, the executor and this controller share the plan contract through
`plan-core`. The executor is reached as a subprocess, so the two programs stay
independent: you can run a plan by hand, and the controller can drive an
executor it did not import.

---

## Design guarantees

### The loop cannot repeat itself

A plan is identified by what it would query — entities, paths, hops, filters,
ranking, aggregation — and not by how it is described. A revision that reads
differently but sends the same TRAPI is rejected before it is executed, and the
planner is told specifically what did not change.

Without this, the loop's most common failure is invisible: four iterations,
four identical queries, one answer, and a trace that looks like diligence.

### Relaxation is monotone, and never touches the user's constraints

When a plan runs to completion and finds nothing, the executor names the axes
that could be loosened, tightest first. The controller picks one — exactly one
— per iteration, so an answer found afterwards can be attributed to the
constraint that was blocking it.

An axis is spent once and never re-offered. Two constraints are never
loosened at all: a pinned anchor's category, because the anchor is what the
question is about, and a user-requested `evidence_policy`, because it records
what the user asked for. The second matters most in the case where relaxing
would work — the filter removed every candidate, and dropping it would produce
results. Those results would answer a question with the user's filter deleted.

### The cheap check runs first

Before the first search, the loop runs the executor's `--check`: it validates
the plan and asks the backend whether it can answer each hop, without sending a
single graph query. Seconds rather than minutes. The two faults it catches — an
invalid plan and an unsupported triple — are both repairable, so finding them
after a multi-minute pathfinder search is pure waiting.

A clean check is never mistaken for a result. It means "nothing obviously
wrong", not "this will work": the backend supporting a hop shape says nothing
about whether it holds data for these entities, so the run proceeds as normal.
Only a repairable verdict short-circuits it.

### Every move is bounded

Iterations, wall clock, planner calls, ARAX calls and model calls all have
ceilings. ARAX calls come from the executor's own ledger rather than an
estimate, and model calls are counted per component — the executor's, the
controller's own decision calls, and the planner's drafts and revisions — so
the ceiling bounds the loop's spend rather than one component's. The trace
breaks the total down, because the three answer different questions: a high
controller count means the decision prompt is being rejected and retried, a
high planner count means revisions are not validating first time.

Execution retries walk a three-rung ladder and then stop being legal. Every
non-terminal move either produces a new plan, spends an axis, or climbs a rung
— all three finite.

### An LLM chooses the move, inside a fence

The controller asks a model what to do next, because reading a thin evidence
distribution alongside a concept warning and judging which matters more is not
something a rule table gets right in every case.

The model chooses from the moves the policy layer has already declared legal.
Its choice is checked against the same rules before it is acted on, and a
choice that fails twice is replaced by the deterministic ordering. That
ordering is a complete controller on its own: the loop is correct without a
model and better with one, and `--no-llm-controller` runs it that way.

### No model-written text reaches a user

The result document is not model-free — the executor keeps what its model said
in `rerank_reason`, in each resolution's `reason`, and in the literature
verdicts, and the plan carries the planner's `restated_question` and refusal
message. All of them read exactly like the prose an answer wants.

The composer reads none of them. It builds each sentence from typed values —
identifiers, labels, Biolink predicates, knowledge levels, source identifiers,
counts, and quotes verified present in a retrieved abstract — and then checks
what it built: every cited edge must exist in the result graph, every
identifier must occur in it, every quotation must be a verified quote, and no
statement may reproduce a quarantined string. Statements that fail are dropped
and recorded rather than repaired.

### An absence is a finding

"No results" covers a timeout and an exhaustive empty traversal, and only one
of them says anything about the world. The controller reports an absence only
for the second, and says so in those terms. For the first it says the opposite:
that nothing was established.

---

## Install

Requirements:

- Python 3.10 or later
- `plan-core` (the plan contract)
- `plan-executor`, importable by some interpreter — as a subprocess, not a
  Python dependency
- Optionally the planner, and a running [Ollama](https://ollama.com) instance

Clone as siblings:

```bash
git clone <this repo> loop-controller
cd loop-controller
pip install -e ../plan-core
pip install -e .
```

You can also run without installing anything:

```bash
PYTHONPATH=../plan-core/src:src python -m loop_controller.run_loop --help
```

---

## Run

Start offline. `--dry-run` puts the executor in its mock mode, which answers
with synthetic data and needs neither ARAX nor a model:

```bash
python -m loop_controller.run_loop \
    --plan ../plan-executor/examples/gluten_plan.json \
    --dry-run --no-llm-controller \
    --executor-pythonpath ../plan-core/src \
    --executor-pythonpath ../plan-executor/src \
    --out runs/
```

Then a real question, end to end:

```bash
python -m loop_controller.run_loop \
    --question "Which approved drugs target IL-17A in psoriasis?" \
    --executor-pythonpath ../plan-core/src \
    --executor-pythonpath ../plan-executor/src \
    --executor-arg --no-verify-literature \
    --max-iterations 4 \
    --out runs/
```

A hand-written plan needs no planner at all. The loop executes, diagnoses and
composes; repairing and relaxing are reported as unavailable.

```bash
python -m loop_controller.run_loop --plan plan.json --out runs/
```

### Exit codes

| Code | Meaning |
|------|---------|
| 0 | An answer was composed, or an absence was established |
| 1 | The loop ended without either |
| 3 | The planner declined the question |

---

## Output

Three documents land in `--out`.

`answer.json` is the deliverable. Every statement carries the edge identifiers
that support it, and a `grounding` block reports what was checked and what — if
anything — was withheld.

```json
{
  "answer_kind": "ranked_candidates",
  "statements": [
    {
      "id": "C1",
      "kind": "candidate",
      "text": "1. Petrolatum [DRUGBANK:DB11058]\n     Petrolatum [DRUGBANK:DB11058] —[biolink:treats]→ dermatitis [MONDO:0002406] (asserted by infores:multiomics-clinicaltrials; knowledge level knowledge_assertion)",
      "supported_by": ["infores:retriever:DRUGBANK:DB11058--biolink:treats--..."]
    }
  ],
  "caveats": ["..."],
  "grounding": {"checked": 11, "passed": 11, "failed": 0, "ok": true}
}
```

`trace.json` is the record of why the loop did what it did: the plan and result
for each iteration, the diagnosis, the moves that were legal and why the others
were not, the decision and who made it, and every rejected model output. It
also records what the deterministic policy *would* have chosen on each
iteration, so the model's contribution is a measured agreement rate rather than
an impression.

`iterN.plan.json`, `iterN.result.json` and `iterN.hints.json` are the
executor's own artefacts, kept per iteration.

---

## The six moves

| Move | When | Planner call |
|------|------|--------------|
| `accept` | There are results worth serving | — |
| `repair_plan` | The executor marked the outcome replannable | yes |
| `relax_plan` | Valid plan, complete coverage, empty graph | yes |
| `retry_execution` | The run did not finish; the plan is fine | no |
| `report_absence` | The traversal completed and the graph holds nothing | — |
| `abandon` | Refusal, locked filter, exhausted budget, no move left | — |

`repair_plan` is the highest-value move because two of the three replannable
outcomes arrive with a grounded menu: the resolver's ranked candidate list for
a name that did not resolve, and the meta knowledge graph's supported
relationships for a hop ARAX cannot answer. Both are passed to the planner, so
a repair is a choice among known options rather than a resample.

`retry_execution` makes no planner call and is usually cheap: the executor
caches raw responses, so completing an unfinished traversal repeats only the
work that did not finish.

---

## Options

| Option | Purpose |
|--------|---------|
| `--question` / `--plan` | Plan a question, or execute a plan you wrote |
| `--dry-run` | Run the executor offline against synthetic data |
| `--no-llm-controller` | Choose moves deterministically; fully reproducible |
| `--max-iterations` | Loop ceiling (default 4) |
| `--max-planner-calls` | Planner ceiling (default 3) |
| `--max-arax-calls` / `--max-llm-calls` | Cost ceilings; model calls counted across executor, controller and planner |
| `--executor-arg ARG` | Passed through to `run_plan.py` verbatim; repeatable |
| `--executor-pythonpath PATH` | Prepended to the executor's `PYTHONPATH`; repeatable |
| `--process-timeout` | Wall-clock ceiling for one executor process |
| `--top-k` | Candidates to include in the answer |
| `--no-quotes` | Omit verified literature quotes |

The executor's own options are not re-exposed. Anything this CLI does not name
goes through `--executor-arg`, so the two programs cannot drift into
disagreeing about what an option means.

---

## Modules

| File | Responsibility |
|------|----------------|
| `loop.py` | The state machine, and the no-repetition guard |
| `contracts.py` | Actions, diagnosis, decision, budget, loop state |
| `diagnose.py` | Reads one result document; decides nothing |
| `policy.py` | Legal moves, hard invariants, deterministic default |
| `decide.py` | The LLM decision maker, and its fallback |
| `revision.py` | The revision request and the message that carries it |
| `fingerprint.py` | Plan identity, and which constraints are locked |
| `ports.py` | What the loop needs from a planner and an executor |
| `executor_cli.py` | Runs `plan-executor` as a subprocess |
| `planner_port.py` | Wraps `PlannerAgent` and adds `revise()` |
| `compose.py` | Builds the answer from typed values |
| `grounding.py` | The gate: lexicon, quarantine, per-statement checks |
| `trace.py` | The audit record |
| `run_loop.py` | Command-line interface |

---

## Tests

```bash
PYTHONPATH=src:tests python -m pytest tests -q
```

Most tests replay recorded result documents through a scripted executor, which
exercises the real diagnosis, policy, decision maker, revision assembly and
composer. `test_compose.py` runs the composer against
`plan-executor/examples/gluten_output.json` — a genuine ARAX run — and
`test_integration_executor.py` drives the real executor subprocess in mock
mode. The latter skips unless `plan-core`, `plan-executor` and the pinned
Biolink model are present.

The tests worth reading first are the termination ones in `test_loop.py`. They
run the loop against decision makers that are actively trying to keep going,
which is the only useful way to check that a bounded loop is bounded.
