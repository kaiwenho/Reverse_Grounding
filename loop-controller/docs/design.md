# Controller design decisions

This document explains the reasoning behind the loop controller. See the
[README](../README.md) for installation and use.

## The gap this component fills

The planner repairs its own plans, but only against *validation* errors: it can
fix a plan that does not conform to the contract. It cannot learn that a
conforming plan named a hop the backend cannot answer, resolved an anchor to
the wrong disease, or ran to completion against an empty region of the graph.
That information exists only after execution.

The executor produces that information deliberately and in a form built to be
acted on — a typed outcome, a `replannable` flag, per-path relaxation axes,
evidence distributions, resolution alternatives, a cost ledger — and then
stops. It does not replan, and it does not auto-relax. Its own note says why:
choosing a replacement predicate or a parent category "needs the Biolink model
and the question's intent, and both belong to the planner".

So there was a gap with two halves. Nothing carried execution feedback back to
the planner, and nothing decided whether feedback was warranted in the first
place. This component is both halves, plus the composition step that turns a
result into an answer.

## The action taxonomy

`replannable` is the executor's judgement about whether a *plan* error
occurred. It is necessary for deciding what to do next and not sufficient,
because it says nothing about a run that returned twenty-five candidates
alongside a failed concept check, and nothing about a run that returned nothing
because a query timed out.

Six moves cover what can actually be done:

| Move | The situation it addresses |
|------|---------------------------|
| `accept` | The results answer the question |
| `repair_plan` | The plan is broken and a planner can fix it |
| `relax_plan` | The plan is valid and tighter than the graph |
| `retry_execution` | The plan is fine and the run did not finish |
| `report_absence` | The traversal completed and the graph holds nothing |
| `abandon` | Nothing above applies |

The set is closed. A decision maker chooses from it or is overridden; there is
no free-text move and no seventh option. That is what makes the loop's
behaviour enumerable, and it is why a model can be given the choice safely.

Two of these are easy to conflate and must not be. An unfinished run and an
empty completed run both return zero results, and they license opposite moves:
one wants a longer timeout, the other wants a looser constraint. Treating them
alike produces the two characteristic failures of a naive loop — relaxing a
plan that was never actually tested, and reporting a timeout to a user as
evidence that the graph is empty.

## Why an LLM decides, and what stops it

A rule table gets the common cases right and the interesting ones wrong. Twenty
candidates with a concept warning and 4% knowledge-level coverage, against a
repair that would cost one planner call — the right move depends on how much
the anchor is in doubt and how much the ranking rests on the missing metadata,
and those are judgements. The model makes them.

What the model cannot do is act outside the rules, because it never chooses
from the full move set. `policy.legal_actions` computes what is permitted for
the current diagnosis and state; the model chooses from that list; and
`policy.validate_decision` checks the choice against the same rules before the
loop acts on it. A choice that fails is returned to the model once, with the
structured problems, mirroring the planner's own retry. A second failure hands
control to the deterministic ordering.

The refusals travel with the list. A model told only "choose accept or abandon"
invents a third option; a model told "relaxing is unavailable because the only
remaining axis would widen the pinned anchor *eczema*" argues with the reason
instead, which is both a better decision and a legible disagreement in the
trace.

Everything the model produces is trace-only. Its `rationale` is written for
developers and never reaches a user, and the composer enforces that
independently rather than trusting the convention.

### Measuring whether it helps

The deterministic policy is evaluated on every iteration whether or not it is
used, because it is the fallback. Recording its choice next to the actual
choice makes "is the model earning its place?" a measured agreement rate rather
than an impression, and makes a systematically overridden model visible as a
prompt problem rather than a mystery.

## Termination

Four invariants, all enforced in code rather than in prompt text.

**No repetition.** A plan is identified by its executable content —
`plan_mode`, entities, paths, explanation queries, ranking, aggregation,
evidence policy — canonicalised and hashed. Interpretation prose, confidence
reasons, gaps, notes and version stamps are excluded. A revision whose
fingerprint matches one already run is rejected before execution, and the
planner is told what did not change. This is the single most valuable guard in
the component: without it, the loop's most common failure looks exactly like
diligence.

**Monotone relaxation.** Each `relax_plan` spends exactly one axis, keyed by
path, axis and entity. Spent axes are never re-offered. The axis set is finite,
so relaxation terminates. An axis is marked spent when the relaxation is
*requested*, not when it succeeds — an axis that was tried and could not be
carried out has had its turn, and re-offering it is the one way the set could
stop shrinking.

**Bounded escalation.** `retry_execution` walks a three-rung ladder of
progressively looser execution settings. Past the top rung, retrying is no
longer legal. The ladder is the retry counter.

**Budgets.** Iterations, wall clock, planner calls, ARAX calls and LLM calls.
The last two are read from the executor's ledger, because a budget the loop
cannot measure is a comment rather than a limit.

Together: every non-terminal move adds a fingerprint, spends an axis, or climbs
a rung; all three are finite; and the budget bounds them again from outside.
A non-terminal move is also refused on the last permitted iteration, since a
plan the loop has no room to run is a planner call spent for nothing.

## Constraints that are never relaxed

Two, for the same reason: they are the user's, not the planner's.

A **user-requested evidence policy** is never weakened. The planner's design
note already says the executor "must not weaken the filter when no candidates
pass", and the rule has to hold one level up as well, because the controller is
the component that would otherwise be tempted. A policy that removed every
candidate is exactly the situation where relaxing it produces results — and
those results would answer a question with the user's filter deleted. The
correct output there is a narrower finding: nothing in the graph met the
evidence standard that was asked for. The composer says that in those words.

A **pinned anchor's category** is never widened. The anchor is the concept the
question is about. The executor resolves anchors under LLM review and checks
afterwards that results concern the intended concept; widening the anchor's
category re-opens the door those two stages exist to close, at the moment the
loop is least able to notice, because a broader anchor usually does return
something.

The answer entity's category is a different matter and is freely widened —
`SmallMolecule` to `ChemicalEntity` narrows nothing the user asked about.

## The revision contract

A `RevisionRequest` carries the prior plan, the failure in the executor's typed
vocabulary, the grounded alternatives, the constraints that may not move, and
the fingerprints already tried. It exists because a bare resample of the
planner's prompt reproduces the same plan often enough that a loop built on one
spends its budget confirming its first answer.

The division of labour follows the executor's. The controller names *which*
constraint gives way; the planner chooses what it becomes, because that needs
the Biolink model and the question's intent. For relaxation the request names
one axis and asks for everything else to stay identical, so that if the revised
plan returns results, the loosened constraint is what was blocking them.

Revisions go through the planner's own system prompt, parser, version stamping
and validators. There is no second, looser path into the graph: a revised plan
is a plan in exactly the sense the executor already relies on.

## The subprocess boundary

The executor runs as a process, not an import. That preserves what makes it
usable: a run that hangs is killed by a timeout rather than blocking the loop,
a crash in a knowledge-provider client cannot take the controller down, and the
exact command is reproducible from the trace by hand.

The cost is a coupling the type system no longer checks. Two things pay it
down. The outcome vocabulary is mirrored in `contracts.py` and an unrecognised
value is reported loudly and treated conservatively — the loop narrows to
accepting or stopping rather than falling through a branch. And every run
produces a result document even when the process produced none: a crashed,
killed or never-started run is written up in the executor's own shape, so
nothing downstream needs a branch for "there is no result".

## Composition, and what counts as model-written

The project's constraint is that no model response is served to a user. Meeting
it takes more than not calling a model in the composer, because `result.json`
is full of model-written text — and the most quotable text in the file is
model-written.

The executor keeps what its model said, correctly, so the reasoning stays
auditable: `rerank_reason` on each candidate, `reason` on each resolution,
per-abstract rationales in the literature verdicts. The plan adds
`interpretation.restated_question`, `interpretation.intent`, the confidence
reasons, the declared gaps, the evidence policy's `rationale`, and a refusal's
message. `rerank_reason` reads exactly like the one-line justification a
candidate list wants. `restated_question` would make an excellent headline.

So the composer does two things. It reads none of those fields, building each
sentence from typed values instead — identifiers, labels, predicates, knowledge
levels, source identifiers, counts, verified quotes. And it checks what it
built against a **quarantine** of every model-written string in the document,
so a future composer that reached for one would fail the check rather than pass
it silently.

The quarantine check is a substring test with a 40-character floor. Below that,
model-written and graph-derived text are not distinguishable by content — a
rerank reason legitimately contains "Petrolatum" — and above it a match means
composed prose reached the answer. The floor makes it a detector of copied
sentences rather than of shared vocabulary.

Composition happens first and filtering second, so the grounding report counts
what composition actually produced. A composer that checked as it went could
never report producing something ungrounded, and the count of withheld
statements is the only visible evidence that the gate does anything.

### The gate's own failure mode

An upstream change could add a model-written field the quarantine does not
know. `unquarantined_prose_fields` does not prevent that; it surfaces the
candidates, so a reviewer sees "three long free-text fields exist that the gate
does not classify" rather than nothing at all.

### Quotes

Verified literature quotes are included. A quote is the abstract's own text and
the executor checked byte-for-byte that it appears there, so it is evidence
rather than composition. The *selection* of quote was model-assisted, and the
answer's provenance block says so alongside the other two model-assisted steps.

## Refusals and absences

A refusal is rendered from its typed `reason`, not from the planner's message.
An unrecognised reason gets a generic sentence rather than a wrong one.

An absence names what was established: the query that ran, the anchors it ran
against, and that the traversal completed. It is bounded to the query as
written, because that is all that was checked. A join failure gets its own
sentence — the individual relationships exist, the chain does not — because
that is a materially different finding from an empty graph.

An unfinished run gets the opposite treatment. The temptation is to report the
empty result as an absence, which is the most useful-sounding claim available
and one the run does not support, so the composed statement says in as many
words that nothing was established.

## What is deliberately not here

**No auto-relaxation without the planner.** The controller could walk the
Biolink hierarchy itself and pick a parent predicate. It does not, because that
would duplicate planner knowledge and ignore the question's intent, which is
how a loop ends up matching a broader relation that no longer means what was
asked.

**No answer synthesis.** There is no "Petrolatum is a plausible candidate
because it is already used topically". That sentence requires a model and would
be exactly the fluent, unfalsifiable claim the architecture exists to keep out.
What replaces it is narrower and duller and checkable.

**No user interaction.** The pipeline never waits for a person, matching the
planner's own rule. An underspecified question produces a refusal, and the
refusal is rendered from its type.

## Open work

- **Evaluation harness.** The trace is designed as its input: per-iteration
  decisions, the deterministic counterfactual, override counts, duplicate
  rejections, cost. A gold set of questions with expected outcome classes would
  turn "does the loop help?" into a number, and would measure the LLM decision
  maker against the policy baseline directly.
- **`revise()` upstream.** The adapter reuses the planner's system prompt,
  parser and validators through borrowed private helpers. Promoting
  `PlannerAgent.revise(RevisionRequest)` into the planner, and the request
  schema into `plan-core`, would make the contract shared rather than inferred.
- **Concept-check repair.** A failed concept check is currently a caveat and a
  reason to prefer repairing. When the anchor resolved to a plausible neighbour
  the resolver's alternatives are the obvious repair menu, and that path is not
  yet distinguished from an ordinary unresolved-grounding repair.
- **Cross-iteration evidence.** Each iteration is composed from its own result
  document. A plan that was relaxed after an empty run discards the earlier
  run's distributions, which would be useful context in the answer's caveats.
