# query-intake

`query-intake` is the front door. It owns everything that happens before a
question reaches the planner and everything that happens after an answer leaves
the loop: request shape, caller identity, rate limits, review depth, screening,
topic scope, and what a caller is allowed to be told.

```text
request + context
      ↓
 rate limit → parse → intake policy ─── refused ──► fixed message
      ↓
 planner (scope-guarded) ──► loop-controller ──► composed answer
      ↓
 fixed public status + trimmed answer
```

The [controller](../loop-controller) runs the plan/execute loop and composes a
grounded answer. This package decides whether a question gets that far, and what
comes back.

---

## Why it exists separately

The controller has a CLI, and a CLI is the right interface for a developer
running the loop by hand. It is the wrong interface for anything facing a
network: it exposes `--top-k`, `--max-results` and a passthrough for arbitrary
executor arguments, and every one of those is a cost lever in the caller's hands.

This package narrows the surface to two fields and puts a server-owned ceiling
behind each of them.

---

## Design guarantees

### A caller controls exactly two fields

```json
{ "question": "Which drugs treat dermatitis herpetiformis?", "review_depth": "standard" }
```

Anything else is refused as an invalid request — including `caller_id`. Identity
comes from whatever authenticated the request, never from the body. If it could
arrive in the payload, a caller could spend someone else's quota and the rate
limiter would be decorative.

### Review depth is a name, not a number

| Depth | Reranking | Abstract review | Abstracts per edge |
|---|---|---|---|
| `off` | no | no | — |
| `light` | top 5 | top 5 | 1 |
| `standard` | top 20 | top 20 | 3 |
| `deep` | top 50 | top 50 | 5 |

Each maps to a frozen set of server-side limits covering executor review
settings, answer size, loop iterations, and call ceilings. A caller picks a
level; they do not pick the numbers behind it.

`off` means no reranking and no abstract review. It does **not** mean no
language model anywhere — the executor requires one to disambiguate entity
names and stops rather than guessing. The deterministic grounding check on the
composed answer runs at every depth, including `off`.

### The deterministic checks are the floor

Length, Unicode normalisation, invisible and bidirectional characters, control
characters. These need no model, cannot fail open, and run before anything
expensive. A 40kB paste never reaches a prompt.

### Ambiguity reaches the planner

Every pattern in `checks.py` is written for precision, not recall. A question
that merely brushes against a topic goes through. This is the design decision
most likely to be quietly reversed after one bad report, and it should not be: a
filter tuned until nothing questionable gets through has stopped answering
research questions, and the loop behind it already bounds what a bad question
can do.

Safety, contraindication and adverse-event research are in scope and tested to
stay that way. What is refused is the *shape* of a personal care decision —
"what dose should I take", "my prescription" — not the subject matter.

### A missing detector fails open, loudly

The rule-based detector always works. Prompt Guard is optional and lazily
loaded; if it cannot be reached, the request proceeds and the trace records that
detection was degraded.

That is deliberate. Failing closed would let an 86M classifier being unreachable
take down a research service — turning a small, bounded risk into a total
outage. The bound is real: an injected question can at worst produce a
Biolink-valid plan for a *different biomedical question*, answered entirely from
graph edges, with no model-written text reaching the caller. If that bound ever
weakens, revisit this decision first.

Note which model does which job. **Llama Guard** is a content-safety classifier
over hazard categories and does not detect injection. **Prompt Guard** is the
injection and jailbreak classifier. Using the first for the second gives you a
guard that does not guard the thing you meant.

### Scope is decided on identifiers, not words

A keyword list fails on the first synonym. "Abortion", "pregnancy termination",
"TOP", the same word in another language, the identifier typed directly — one
concept, many strings.

So a scope rule names an ontology term, and blocks that term plus everything
beneath it. One rule, every phrasing, every subtype. And a block produces a
sentence you can defend in a review:

> the anchor resolved to `MONDO:0000123`, a descendant of `MONDO:0005240`,
> covered by rule R-004, added 2026-03-14, owner K. Ho

rather than *"the classifier said no"*, which is reproducible by nobody and
shifts the day the model is updated.

The gate wraps the planner rather than sitting in front of it, because the
planner is what actually decides which entities a question is about — it does
the extraction that would otherwise need NER or another model. A blocked
question costs one planner call and no graph queries. Because it inspects what
the plan *pins*, a question that tries to talk its way into a different anchor
is judged on what ended up in the plan.

### Raw questions stay out of persisted logs

Traces carry a salted digest by default. A biomedical question can identify a
person even in a service that refuses patient-specific ones, and the trace is
what ends up in a log store. `log_raw_question` on the context turns it off
where that is appropriate.

### What leaves is narrow

A status from a closed set, a message from a fixed table, and the answer the
controller already ran through its grounding gate. Not the internal reason
("planner call budget spent (3)"), not the rule that fired, not a classifier's
explanation, not the trace.

The answer is trimmed further than the controller produces it: the grounding
counts survive, the per-violation detail does not — that describes where the
checker is weak and belongs in the trace.

---

## Install

```bash
git clone <this repo> query-intake
cd query-intake
pip install -e ../loop-controller
pip install -e .
```

Optional injection classifier:

```bash
pip install -e ".[promptguard]"
```

Or run without installing:

```bash
PYTHONPATH=src:../loop-controller/src python -c "import query_intake"
```

---

## Use

```python
from loop_controller.executor_cli import ExecutorSettings, SubprocessExecutor
from loop_controller.planner_port import PlannerAgentPort
from planner_agent import PlannerAgent
from planner_agent.ollama_client import OllamaGPTOSSClient

from query_intake import (
    IntakePolicy, QueryService, RequestContext, ScopeGate, ScopeRules,
    ServiceConfig, default_detector,
)
from query_intake.scope import SriNameResolver

service = QueryService(
    executor=SubprocessExecutor(ExecutorSettings(
        pythonpath=["../plan-core/src", "../plan-executor/src"],
        runs_dir="runs",
    )),
    planner=PlannerAgentPort(PlannerAgent(llm=OllamaGPTOSSClient())),
    config=ServiceConfig(
        intake=IntakePolicy(default_detector(use_prompt_guard=True)),
        scope=ScopeGate(ScopeRules.load()),
        resolver=SriNameResolver(),
        runs_dir="runs",
    ),
)

result = service.handle_payload(
    {"question": "Which approved drugs target IL-17A in psoriasis?",
     "review_depth": "standard"},
    RequestContext(caller_id=authenticated_user_id),   # never from the body
)

print(result.to_dict())
```

### Statuses

| Status | Meaning |
|---|---|
| `completed` | Results found; every statement is graph-supported |
| `no_data` | The query ran to completion and the graph holds nothing — a finding |
| `refused` | Declined at intake, by scope, or by the planner |
| `needs_clarification` | The question lacks information needed to build a query |
| `rate_limited` | Caller over their limit; `retry_after_s` says how long |
| `invalid_request` | Malformed body or unknown field |
| `budget_exhausted` | The loop stopped without reaching a conclusion |
| `unavailable` | A component could not be reached |

---

## The scope rule file

No `rules/scope_rules.json` means everything is allowed, which is the right
default. `rules/scope_rules.example.json` documents the format.

Every entry requires `rule_id`, `curie`, `label`, `owner`, `added` and
`justification`, and `ScopeRule.from_dict` refuses one that is missing any of
them — an entry nobody owns is one nobody will remove. `review_by` is optional
and recommended; `ScopeRules.stale_rules()` reports entries past it.

Descendants are precomputed offline, not walked per request:

```python
from query_intake.scope import expand_closure, write_closure

closure = expand_closure("mondo.json", ["MONDO:0005240"])
write_closure(closure, "rules/scope_closure.json", ontology_version="mondo-2026-01-05")
```

`mondo.json` is a downloaded, version-pinned dependency — the same convention
`plan-core` uses for `biolink-model.yaml`. Without a closure file a rule blocks
only its exact term, which looks like a working subtree rule and is not;
`ScopeRules.unexpanded()` lists any root in that state.

**A caution worth keeping in the file.** A topic blocklist in a biomedical tool
decides which researchers the tool is useless for. Reproductive health, mental
health, substance use, stigmatised infectious disease — these are areas where
researchers genuinely need graph queries. Keep the list as short as you can
justify, tie each entry to a specific institutional requirement rather than
general discomfort, and put a review date on it.

---

## Modules

| File | Responsibility |
|---|---|
| `service.py` | The one public entry point, and the fixed processing order |
| `models.py` | Request, context, review depth, limits, public result |
| `messages.py` | Every sentence this service is allowed to say |
| `checks.py` | Deterministic text checks and the pattern lists |
| `detectors.py` | Pluggable attack detection, including Prompt Guard |
| `policy.py` | Checks and detector combined into one intake decision |
| `scope.py` | Scope rules, ontology closure, the planner decorator |
| `ratelimit.py` | Per-caller request and concurrency limits |
| `redact.py` | Question digests and trace redaction |

---

## Tests

```bash
PYTHONPATH=src:../loop-controller/src:tests python -m pytest tests -q
```

No live ARAX and no live model. The executor and planner are the scripted
doubles from `loop-controller`, so what runs is the real intake policy, the real
scope gate, the real loop and the real composer.

The two tests worth reading first are
`test_caller_identity_is_never_read_from_the_request_body` and the leakage tests
at the end of `test_service.py`. Everything else could be wrong and produce a
bad answer; those being wrong produce a service where quotas are decorative and
internals are on the wire.

---

## Known limits

- **The rate limiter is in-memory and per-process.** Correct for one worker;
  with several, each keeps its own counters and a caller gets the limit
  multiplied by the worker count. Swap in a shared store before scaling out.
- **Digests do not correlate across restarts** unless `INTAKE_LOG_SALT` is set.
- **The scope gate resolves names with one lookup** and checks the resolver's
  top candidates. The executor later chooses under model review and may pick
  differently.
- **Injection detection catches clumsy attacks.** Meta says so of Prompt Guard
  in its own model card, and it is more true of the pattern list. The real
  defence is the small blast radius downstream.
