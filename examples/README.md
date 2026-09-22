# A worked example

One real run, start to finish. Read this if you want to know what the system
does without installing it or finding a GPU.

```bash
reverse-grounding --question "Which drugs treat psoriasis?" \
    --out examples/psoriasis --result examples/psoriasis/result.json
```

One iteration, 284 seconds, one ARAX call, four model calls. 25 drugs came
back. Every statement passed the grounding check.

## The files

| file | what it is |
|---|---|
| `psoriasis/result.json` | what the caller got back, answer included |
| `psoriasis/trace.json` | every decision, and why the alternatives were not taken |
| `psoriasis/iter1.plan.json` | the plan the model wrote |
| `psoriasis/iter1.result.json` | what the executor returned |
| `psoriasis/iter1.hints.json` | what the executor suggested to the controller |

## What the caller gets

`result.json` is the public document. A fixed status, a message from a fixed
table, a request id, and the answer:

```json
{
  "status": "completed",
  "message": "Results were found. Every statement in the answer is supported by edges present in the returned graph.",
  "request_id": "b01221db029d4d0dbad0a16007177145",
  "review_depth": "standard",
  "answer": { ... }
}
```

The message is not assembled from anything that happened inside. It is picked
from `query_intake/messages.py` by status. Whatever a model said, whatever a
classifier scored, whatever the planner's internal refusal reason was — that
goes to the trace, which stays inside.

## What a claim looks like

Every statement carries the edge ids that support it:

```json
{
  "id": "C1",
  "kind": "candidate",
  "text": "1. Ustekinumab [UNII:FU77B4U5Z0]\n     Ustekinumab [UNII:FU77B4U5Z0] —[biolink:treats]→ psoriasis [MONDO:0005083] (asserted by infores:multiomics-clinicaltrials, infores:drugcentral, infores:multiomics-drugapprovals; knowledge level knowledge_assertion)",
  "supported_by": [
    "infores:retriever:UNII:FU77B4U5Z0--biolink:treats--None--None--None--MONDO:0005083--infores:multiomics-clinicaltrials",
    "..."
  ]
}
```

`UNII:FU77B4U5Z0`, `biolink:treats`, `MONDO:0005083` and every source named
are in the graph ARAX returned. The grounding gate checked all of it:

```json
"grounding": { "checked": 26, "passed": 26, "withheld": 0, "ok": true }
```

Nothing was withheld here. When something is, the count and the reason appear
in the caveats — silence about a gap would be its own ungrounded claim.

## The caveats are not decoration

```
On path P1, 0.0% of edges carry publications; most support here is curated
assertion rather than cited literature.

The order of 20 candidate(s) was adjusted by a model whose stated reasons were
checked against the retrieved edges (20 passed). The reasons themselves are
recorded in the execution result and are not reproduced here.

Literature was checked for 49 edge(s); only quotes verified present in the
retrieved abstract are shown.
```

The first says the evidence is curation, not papers — true of this question,
and the kind of thing a reader should weigh. The second is the interesting
one: **a model reordered the list, and its reasons were checked against the
retrieved edges before the reordering was allowed to stand.** The reasons
themselves are not shown, because they are model prose and model prose does
not reach a reader.

## What the model did, and what it did not

`provenance.model_assisted_steps` says it outright:

```
entity disambiguation (choice recorded per entity, with alternatives)
candidate reranking (reasons required to cite retrieved edges; ungrounded reasons discarded)
literature quote selection (quotes verified present in the retrieved abstract before use)
```

Three jobs, each with a check. The resolution is recorded with its
alternatives, so the choice can be second-guessed:

```json
"psoriasis": {
  "resolved_curies": ["MONDO:0005083"],
  "confidence": "high",
  "chosen_by_model": true,
  "alternatives_considered": 20
}
```

And `composed_by` states the boundary:

> deterministic template renderer; no model-authored text is read or emitted,
> and every statement was checked against the result graph

`graph.quarantined_strings: 22` is that boundary in numbers — 22 identifiers a
model wrote, held apart from the lexicon so they cannot be mistaken for
something the graph asserted.

## What the trace shows

The question is not in it:

```json
"question": "q:d08928560ca59a74",
"planner_exchanges_note": "prompts and raw responses omitted; they quote the question"
```

A salted digest instead of the text, and the planner's prompts dropped for
quoting it. `request_id` ties this trace to the result the caller holds.
`--log-raw-question` turns that off.

The decision itself, with the alternatives and why each was unavailable:

```json
"allowed": ["abandon", "accept"],
"refusals": {
  "repair_plan":     "the executor did not mark 'results' replannable",
  "relax_plan":      "relaxing is for an empty result, and there are results",
  "retry_execution": "nothing about this run suggests it was execution that failed",
  "report_absence":  "there are results, so there is no absence to report"
}
```

This run was easy, so the loop ran once and accepted. The refusals are still
recorded, because on a hard question the reason a move was *not* available is
usually what you need.

`decision_maker_health.attempts: 0` — no model was asked. With
`--llm-controller` off the deterministic policy decides, and the
`counterfactual` block records what the other one would have done either way.

## A word on what this does not claim

Ammonia is rank 21. Allantoin is 24. Both are there because `infores:drugcentral`
asserts an edge to psoriasis, and both are honestly reported as single-source
curated assertions with no publications.

That is the system working. It reports what the graph holds, with the
provenance attached, and does not decide whether the graph is right. Judging
whether an assertion is biologically meaningful is a different problem, and
this system does not pretend to solve it — it makes the evidence legible
enough that you can.
