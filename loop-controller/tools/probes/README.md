# Relaxation probes

Three hand-written plans whose only job is to reach `relax_plan`.

**`probe_predicate.json` reaches it.** `relax_plan → accept`, 20 results, zero
over-broad relaxations, on live ARAX. That closes the last unmeasured path in
the loop: every one of the six actions has now been taken on real data.

Getting there took seven attempts — four live runs on natural-language
questions, then three probes. The other two probes still miss, and the table
below says why each one did.

That is not bad luck. Relaxation needs a narrow conjunction:

* the plan validates and the `--check` probe passes, **and**
* every entity resolves, **and**
* the query runs to completion with `coverage_complete`, **and**
* it returns nothing, **and**
* a loosenable axis remains that is not a pinned anchor.

Anything that breaks earlier goes to `repair_plan` and never arrives. Steering
a *question* into that state means guessing what the planner will write and
what the graph holds, which is two guesses; writing the plan directly is none.

The planner is still exercised where it counts: `--plan` fixes only the opening
plan, and the revision goes through the real `PlannerAgentPort`, with the real
one-axis instruction and the real structural diff.

That was not true when these probes were first written. `bench.py` skipped
building the planner whenever `--plan` was given, which made `relax_plan`
illegal — "no planner is configured to choose the replacement" — so a probe
that came back empty abandoned instead of relaxing. Three runs were spent
finding that out. Use a bench with `--no-planner` in its `--help`; an older one
cannot run these.

## Running them

One at a time, each against a one-line question file. The question text is
ignored when `--plan` is given, but the bench still wants a file:

```bash
cd loop-controller
echo "probe" > /tmp/probe.txt

for p in qualifiers expansion predicate; do
  PYTHONPATH=../plan-core/src:src python tools/bench.py /tmp/probe.txt \
    --plan tools/probes/probe_$p.json \
    --out runs/probe-$p \
    --executor-pythonpath ../plan-core/src \
    --executor-pythonpath ../plan-executor/src
done
```

The `--executor-pythonpath` flags are not optional. The executor runs as a
subprocess and does not inherit this process's `PYTHONPATH`, so without them it
dies at import.

Leave `--llm-controller` **off**. The question here is what the planner does
with a relaxation instruction, and a second model choosing the moves only
makes a surprising result harder to attribute.

## What each one targets, and what actually happened

| plan | axis | live result |
|---|---|---|
| `probe_predicate.json` | `P1:predicate` | **`relax_plan` → `accept`, 20 results, 0 over-broad.** The one that works. |
| `probe_qualifiers.json` | `P1:qualifiers` | `accept` on iteration 1. Not empty. |
| `probe_expansion.json` | `P1:predicate_expansion` | `accept` on iteration 1. Not empty. |

`probe_predicate.json` took two attempts, and the first failure is the useful
part. It originally asked for `preventative_for_condition` between
ChemicalEntity and Disease. ARAX does not model that triple at all, so the
meta knowledge graph rejected the hop before any query ran:
`unsupported_backend_capability`, replannable, `repair_plan`, relaxation never
reached, `arax_calls: 0`.

**An unsupported triple is a plan fault, not an over-tight query.** They look
similar from outside — both return nothing — and only one of them relaxes.
That is the trap every earlier probe fell into, and it is the first thing to
check when a new one misses.

The rejection message is also the fix. It lists what ARAX *does* support
between those categories:

```
affects, applied_to_treat, causes, coexists_with, contraindicated_in,
contributes_to, correlated_with, disrupts, exacerbates_condition,
has_side_effect, in_clinical_trials_for, in_preclinical_trials_for
```

Pick the sparsest entry on that list and aim it at a rare disease. The current
probe uses `in_preclinical_trials_for` against dermatitis herpetiformis:
supported by construction, empty in practice. It carries no qualifiers and no
entity constraints, so `predicate` is the tightest axis
`suggest_relaxations` can offer and the policy has to pick it first.

The two probes that missed are kept as they are. `probe_expansion` failing
says something worth knowing about the graph: most `biolink:affects` edges in
ARAX are recorded as `affects` itself, not as descendants, so `self_only` is
not the tight constraint that probe assumed.

All three validate against plan-core, and the axis in the table is what
`plan_executor.suggest_relaxations` ranks first for that hop. What is *not*
checked is the biology — whether each query is actually empty against live
ARAX. If one comes back with results on iteration
1, it has told you nothing; tighten it further, or say so and it can be
rewritten.

The anchors (`tnf`, `psoriasis`) are pinned, so their `category` axes are
offered by the executor and refused by the controller. That refusal appearing
in `legal.cautions` is itself worth seeing.

## What to read afterwards

For each run, in `runs/probe-*/summary.json`:

* `action_counts` — `relax_plan` must appear, or the probe missed and nothing
  below means anything.
* `over_broad_relaxations.rejected_and_corrected` — the planner over-reached
  and took the correction. The rule works.
* `over_broad_relaxations.accepted_with_attribution_withdrawn` — it would not
  take the correction. Read `over_relaxations[].diff.off_axis` in the trace:
  those field names are what the relaxation prompt has to stop the model
  touching.
* `revision_verdicts` — `planner_refused` here would be interesting. It would
  mean the model declines to loosen a constraint it is explicitly told it may
  loosen, which is a prompt problem, not a policy one.
