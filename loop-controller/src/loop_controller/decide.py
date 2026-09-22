"""
decide.py — Who chooses the next move.

Two decision makers implement the same small interface. `PolicyDecisionMaker`
returns the deterministic priority ordering from `policy.default_action`.
`LLMDecisionMaker` asks a model, and falls back to that same ordering whenever
the model cannot be reached, cannot be parsed, or proposes something the policy
layer refuses twice.

The model's job is narrow on purpose. It is given the diagnosis, the moves the
policy layer has declared legal, the reason each other move is not, and what
the loop has already tried. It returns one move. It does not name predicates,
it does not write anything a user will read, and it cannot widen its own remit,
because everything it returns is checked against the same rules that produced
the list it chose from.

The retry mirrors the planner's: one more attempt, with the structured problems
appended. Two attempts and then a deterministic fallback is a bounded cost per
iteration, which matters because this runs inside a loop that is itself bounded
— an unbounded repair here would quietly become the loop's real budget.

`rationale` is written to the trace and never to an answer. That separation is
load-bearing for the project's constraint that no model-authored text reaches a
user, and it is enforced twice: here by convention, and in `compose.py` by a
gate that refuses to emit any field this package marks as model-authored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Protocol

from .contracts import ABANDON, ACTIONS, Decision, Diagnosis, LoopState
from .policy import LegalActions, default_action, validate_decision


SCHEMA_PATH = Path(__file__).resolve().parents[2] / "schema" / "controller_decision.schema.json"

_VALIDATOR = None


def _validator():
    """Compile the decision schema once, and tolerate its absence.

    The schema is the structural half of the check; `validate_decision` is the
    half that matters more, since legality, monotonicity and budget cannot be
    expressed against a single document. If jsonschema or the file is missing,
    structural checking degrades to the dataclass conversion and the semantic
    checks still run — a decision can be malformed and rejected, but never
    illegal and accepted.
    """
    global _VALIDATOR
    if _VALIDATOR is not None:
        return _VALIDATOR
    try:
        import jsonschema  # type: ignore

        with open(SCHEMA_PATH, "r", encoding="utf-8") as handle:
            schema = json.load(handle)
        _VALIDATOR = jsonschema.Draft202012Validator(schema)
    except Exception:  # pragma: no cover - environment-dependent
        _VALIDATOR = False
    return _VALIDATOR


class LLMClient(Protocol):
    """The planner's client protocol, reused unchanged."""

    def complete(self, system: str, user: str) -> str: ...


@dataclass
class DecisionAttempt:
    """One exchange with the decision maker, kept for the trace."""

    index: int
    raw_response: str = ""
    parse_error: Optional[str] = None
    problems: List[str] = field(default_factory=list)
    accepted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index": self.index,
            "raw_response": self.raw_response[:4000],
            "parse_error": self.parse_error,
            "problems": self.problems,
            "accepted": self.accepted,
        }


class DecisionMaker(Protocol):
    def decide(
        self, diag: Diagnosis, state: LoopState, legal: LegalActions,
    ) -> Decision: ...


# ---------------------------------------------------------------------------
# Deterministic
# ---------------------------------------------------------------------------


class PolicyDecisionMaker:
    """The priority ordering, and nothing else.

    Useful in three places: as the fallback for the LLM maker, as the decision
    maker for a reproducible evaluation run, and as the thing an LLM maker's
    choices are compared against when judging whether the model is earning its
    place in the loop.
    """

    name = "policy"

    def __init__(self) -> None:
        self.attempts: List[DecisionAttempt] = []

    def decide(
        self, diag: Diagnosis, state: LoopState, legal: LegalActions,
    ) -> Decision:
        self.attempts = []
        return default_action(diag, state, legal)


# ---------------------------------------------------------------------------
# LLM
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are the Loop Controller for a biomedical knowledge-graph question
answering system. A planner turns a question into a query plan; an executor
runs that plan against the graph and reports what happened. You decide what the
system does next.

You choose exactly one move, from the list of legal moves you are given. You
may not invent a move, and you may not choose one that is not in that list. The
list is computed from the current state, and every move it excludes is
excluded for a stated reason, which you are shown.

The moves:

  accept           Serve these results as the answer.
  repair_plan      The plan is broken in a way the planner can fix: an invalid
                   field, a name that did not resolve, a hop the backend cannot
                   answer. Where alternatives are known they are listed for you.
  relax_plan       The plan is valid, ran to completion, and found nothing. Its
                   constraints may be tighter than the graph. Name exactly ONE
                   axis to loosen, by its key. The planner chooses what to
                   replace it with; you choose only which constraint gives way.
  retry_execution  The plan is fine and the run did not finish. Re-run the same
                   plan with looser execution settings. No planner call is made.
  report_absence   The query ran to completion and the graph holds no data for
                   it. This is a finding, not a failure, and is a legitimate
                   answer to give a user.
  abandon          Stop without an answer.

How to choose:

- Prefer the move that addresses what actually went wrong. An unfinished run is
  not evidence that a plan is over-constrained; an over-constrained plan is not
  fixed by waiting longer.
- Prefer the smallest change. Relaxation axes are listed tightest first. An
  answer obtained after loosening one constraint can be attributed to that
  constraint; an answer obtained after loosening four cannot.
- Results in hand usually beat another iteration. Take one only when you can
  say what the next run would produce that this one did not.
- A concept-check warning is serious: it means the results may concern a
  related but different concept than the question asked about. Prefer repairing
  the anchor over accepting, when repair is legal.
- Thin knowledge-level annotation is a caveat, not a defect. It travels with
  the answer either way and is rarely a reason to iterate.
- report_absence is not a failure and not a last resort. Establishing that the
  graph does not contain something is often the most useful true statement the
  system can make.

Your rationale is written to an audit trace for developers. It is never shown
to a user, and no text you write ever reaches a user: the answer is composed
separately, from the graph, by a deterministic renderer. Do not write for an
end reader and do not summarise the biology.

Return a single JSON object and nothing else:

{
  "action": "<one of the legal moves>",
  "rationale": "<why, in one or two sentences, referring to the diagnosis>",
  "expected_change": "<what the next run should produce that this one did not>",
  "relaxation_key": "<required for relax_plan: one key from relaxation_options>",
  "repair_focus": ["<optional for repair_plan>"],
  "execution_overrides": {"timeout": 300}
}

Omit the fields that do not apply to your chosen move.
"""


class LLMDecisionMaker:
    """Asks a model, checks the answer, falls back when it does not hold up."""

    name = "llm"

    def __init__(
        self,
        llm: LLMClient,
        *,
        max_attempts: int = 2,
        history_limit: int = 4,
    ) -> None:
        self.llm = llm
        self.max_attempts = max_attempts
        self.history_limit = history_limit
        self.attempts: List[DecisionAttempt] = []

    # -- public ------------------------------------------------------------

    def decide(
        self, diag: Diagnosis, state: LoopState, legal: LegalActions,
    ) -> Decision:
        self.attempts = []
        fallback = default_action(diag, state, legal)

        if not legal.allowed or legal.allowed == [ABANDON]:
            # Nothing to choose between. Asking anyway spends a call to be told
            # the only thing that was ever going to happen.
            fallback.rationale += " (no alternative move was available)"
            return fallback

        user_message = build_user_message(diag, state, legal, self.history_limit)
        problems: List[str] = []

        for index in range(1, self.max_attempts + 1):
            attempt = DecisionAttempt(index=index)
            self.attempts.append(attempt)

            message = user_message
            if problems:
                message = user_message + _retry_block(problems)

            try:
                raw = self.llm.complete(SYSTEM_PROMPT, message)
            except Exception as exc:
                attempt.parse_error = f"llm call failed: {exc}"
                return _override(fallback, None, attempt.parse_error)

            attempt.raw_response = raw or ""
            parsed, parse_error = _parse(raw)
            if parse_error:
                attempt.parse_error = parse_error
                problems = [parse_error]
                continue

            structural = _structural_problems(parsed)
            decision = _to_decision(parsed)
            semantic = validate_decision(decision, diag, state, legal)
            problems = structural + semantic
            attempt.problems = problems

            if not problems:
                attempt.accepted = True
                decision.source = "llm"
                return decision

        proposed = self.attempts[-1]
        last_action = None
        if not proposed.parse_error and proposed.raw_response:
            parsed, _ = _parse(proposed.raw_response)
            if isinstance(parsed, dict):
                last_action = parsed.get("action")
        return _override(
            fallback, last_action,
            "; ".join(problems) or proposed.parse_error or "no valid decision",
        )


def _override(
    fallback: Decision, attempted: Optional[str], reason: str,
) -> Decision:
    """Take the deterministic move, and say loudly that it was a fallback.

    Kept visible rather than smoothed over. A loop where the model is
    frequently overridden is a loop whose prompt, or whose reporting of legal
    moves, needs work — and that is only diagnosable if every override survives
    into the trace with the rejected choice next to the reason.
    """
    fallback.source = "policy"
    fallback.overridden_from = attempted
    fallback.override_reason = reason
    return fallback


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------


def build_user_message(
    diag: Diagnosis,
    state: LoopState,
    legal: LegalActions,
    history_limit: int = 4,
) -> str:
    """Assemble the decision request.

    Compacted deliberately. The full result document runs to megabytes and
    almost none of it bears on the choice; sending it would bury the four or
    five facts that do. What is included is the executor's own classification,
    the material each legal move would use, and what has already been tried —
    the last of these because a model that cannot see its previous iterations
    will propose them again.
    """
    lines: List[str] = []

    lines.append(f"QUESTION\n{state.question or '(not recorded)'}\n")

    lines.append("WHERE THE LOOP IS")
    lines.append(f"  iteration {state.iteration + 1} of at most {state.budget.max_iterations}")
    lines.append(f"  planner calls used: {state.planner_calls}/{state.budget.max_planner_calls}")
    lines.append(f"  elapsed: {state.elapsed_s:.0f}s of {state.budget.max_wall_clock_s:.0f}s")
    if state.execution_overrides:
        lines.append(f"  execution settings in force: {state.execution_overrides}")
    lines.append("")

    if state.attempts:
        lines.append("WHAT HAS BEEN TRIED")
        for attempt in state.attempts[-history_limit:]:
            d = attempt.diagnosis
            decision = attempt.decision
            outcome = f"{d.verdict}/{d.outcome}, {d.num_results} result(s)" if d else "?"
            chose = decision.action if decision else "?"
            lines.append(f"  {attempt.index}. {outcome} -> chose {chose}")
            if decision and decision.relaxation_key:
                lines.append(f"     loosened {decision.relaxation_key}")
            if decision and decision.expected_change:
                lines.append(f"     expected: {decision.expected_change}")
        if state.spent_axes:
            lines.append(f"  axes already loosened: {sorted(state.spent_axes)}")
        lines.append("")

    lines.append("DIAGNOSIS OF THE RUN THAT JUST FINISHED")
    lines.append(f"  verdict: {diag.verdict}")
    lines.append(f"  outcome: {diag.outcome} — {diag.outcome_detail}")
    lines.append(f"  replannable: {diag.replannable}")
    lines.append(f"  results: {diag.num_results}")
    if diag.concept_warning:
        lines.append(f"  CONCEPT WARNING: {diag.concept_warning}")
    if diag.unresolved_entities:
        lines.append(f"  unresolved entities: {diag.unresolved_entities}")
    if diag.incomplete_coverage:
        lines.append(f"  incomplete coverage on: {diag.incomplete_coverage}")
    if diag.timed_out_paths:
        lines.append(f"  timed out: {diag.timed_out_paths}")
    if diag.filters_dropped_all:
        lines.append("  the evidence policy removed every candidate")
    if diag.thin_annotation_paths:
        lines.append(
            f"  thin knowledge-level annotation on: {diag.thin_annotation_paths}"
        )
    if diag.ungrounded_rerank_count:
        lines.append(
            f"  {diag.ungrounded_rerank_count} rerank reason(s) failed their "
            f"grounding check and were discarded"
        )
    if diag.locked_constraints:
        lines.append(f"  constraints that may not be relaxed: {diag.locked_constraints}")
    lines.append("")

    if diag.resolution_alternatives:
        lines.append("RESOLUTION ALTERNATIVES (what the resolver considered)")
        for ref, options in diag.resolution_alternatives.items():
            shown = ", ".join(
                f"{o.get('curie')} ({o.get('label')})" for o in options[:6]
            )
            lines.append(f"  {ref}: {shown or '(none)'}")
        lines.append("")

    if diag.unsupported_hops:
        lines.append("HOPS THE BACKEND CANNOT ANSWER")
        for hop in diag.unsupported_hops:
            lines.append(f"  {hop.get('path_id')}: {hop.get('message')}")
            if hop.get("supported_alternatives"):
                lines.append(
                    f"    it does support: {hop['supported_alternatives'][:10]}"
                )
        lines.append("")

    if diag.plan_validation_errors:
        lines.append("VALIDATION ERRORS")
        for err in diag.plan_validation_errors[:10]:
            lines.append(f"  {err}")
        lines.append("")

    lines.append("LEGAL MOVES")
    for action in legal.allowed:
        lines.append(f"  {action}")
    lines.append("")

    if legal.refusals:
        lines.append("MOVES THAT ARE NOT AVAILABLE, AND WHY")
        for action in ACTIONS:
            if action in legal.refusals:
                lines.append(f"  {action}: {legal.refusals[action]}")
        lines.append("")

    if legal.relaxation_options:
        lines.append("RELAXATION OPTIONS (tightest first; choose at most one key)")
        for axis in legal.relaxation_options:
            current = f" currently {axis.current!r}" if axis.current is not None else ""
            lines.append(f"  {axis.key} — {axis.detail}{current}")
        lines.append("")

    if legal.escalation:
        lines.append(
            f"IF YOU RETRY EXECUTION, the next escalation step is "
            f"{legal.escalation}\n"
        )

    if legal.cautions:
        lines.append("CAUTIONS")
        for caution in legal.cautions:
            lines.append(f"  - {caution}")
        lines.append("")

    lines.append("Return the JSON decision object now.")
    return "\n".join(lines)


def _retry_block(problems: List[str]) -> str:
    body = "\n".join(f"  - {p}" for p in problems)
    return (
        "\n\nYOUR PREVIOUS DECISION WAS REJECTED\n"
        f"{body}\n\n"
        "Choose again from the legal moves listed above. If none of them fits "
        "what you believe should happen, choose the closest legal move and say "
        "so in the rationale.\n"
    )


# ---------------------------------------------------------------------------
# Parsing and checking
# ---------------------------------------------------------------------------


def _parse(raw: str) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Parse the model's reply, tolerating a fenced or padded object."""
    if raw is None:
        return None, "empty response"
    text = raw.strip()
    if not text:
        return None, "empty response"
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.strip()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            return None, "response was not JSON and contained no JSON object"
        try:
            parsed = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            return None, f"response was not valid JSON: {exc}"
    if not isinstance(parsed, dict):
        return None, f"expected a JSON object, got {type(parsed).__name__}"
    return parsed, None


def _structural_problems(parsed: Dict[str, Any]) -> List[str]:
    validator = _validator()
    if not validator:
        return []
    problems: List[str] = []
    for error in sorted(validator.iter_errors(parsed), key=lambda e: list(e.path)):
        location = "/".join(str(p) for p in error.path) or "(root)"
        problems.append(f"{location}: {error.message}")
    return problems[:8]


def _to_decision(parsed: Dict[str, Any]) -> Decision:
    overrides = parsed.get("execution_overrides") or {}
    if not isinstance(overrides, dict):
        overrides = {}
    focus = parsed.get("repair_focus") or []
    if not isinstance(focus, list):
        focus = []
    return Decision(
        action=str(parsed.get("action") or ""),
        rationale=str(parsed.get("rationale") or ""),
        expected_change=str(parsed.get("expected_change") or ""),
        relaxation_key=parsed.get("relaxation_key") or None,
        repair_focus=[str(f) for f in focus][:8],
        execution_overrides=overrides,
        source="llm",
    )
