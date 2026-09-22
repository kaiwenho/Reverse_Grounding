"""
trace.py — The record of why the loop did what it did.

An iterative system that sometimes calls a model is only trustworthy to the
extent its decisions can be reconstructed afterwards. The trace is what makes
that possible, and it is written for three readers.

A developer debugging a bad answer needs the chain: which plan ran, what came
back, what was diagnosed, which moves were legal, which was taken and by whom.

An evaluation harness needs the same records as data. Every field here is
machine-readable, and the ones that matter for evaluation are the ones a
summary would drop — how often the model's choice was overridden, how often a
revision duplicated a plan already tried, how many iterations ended in an
accept versus an absence versus an exhausted budget.

And a reviewer asking whether the model is earning its place needs the
counterfactual. The deterministic policy is evaluated on every iteration
regardless, because it is the fallback, so the trace can record what the policy
*would* have chosen next to what was actually chosen. Agreement rate is then a
measured number rather than an impression.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List

from .contracts import CONTROLLER_SCHEMA_VERSION, LoopOutcome


@dataclass
class Trace:
    """Accumulates the loop's record and writes it out."""

    question: str = ""
    started_at: str = ""
    decision_attempts: List[Dict[str, Any]] = field(default_factory=list)
    planner_exchanges: List[Dict[str, Any]] = field(default_factory=list)
    executor_commands: List[Dict[str, Any]] = field(default_factory=list)
    legality: List[Dict[str, Any]] = field(default_factory=list)
    counterfactual: List[Dict[str, Any]] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def note(self, message: str) -> None:
        self.notes.append(message)

    def warn(self, message: str) -> None:
        self.warnings.append(message)

    def record_legality(self, iteration: int, legal: Any) -> None:
        self.legality.append({"iteration": iteration, **legal.to_dict()})

    def record_decision_attempts(self, iteration: int, attempts: List[Any]) -> None:
        for attempt in attempts or []:
            self.decision_attempts.append({
                "iteration": iteration, **attempt.to_dict()
            })

    def record_counterfactual(
        self, iteration: int, chosen: str, policy_choice: str, source: str,
    ) -> None:
        self.counterfactual.append({
            "iteration": iteration,
            "chosen": chosen,
            "policy_would_choose": policy_choice,
            "chosen_by": source,
            "agreed": chosen == policy_choice,
        })

    # -- output ------------------------------------------------------------

    def to_dict(self, outcome: LoopOutcome) -> Dict[str, Any]:
        state = outcome.state
        agreements = [c for c in self.counterfactual if c["agreed"]]
        overrides = [
            a for a in self.decision_attempts if a.get("problems") or a.get("parse_error")
        ]
        return {
            "schema": CONTROLLER_SCHEMA_VERSION,
            "question": self.question,
            "started_at": self.started_at,
            "outcome": {
                "status": outcome.status,
                "reason": outcome.reason,
            },
            "state": state.to_dict() if state else None,
            "decision_attempts": self.decision_attempts,
            "legality": self.legality,
            "counterfactual": {
                "per_iteration": self.counterfactual,
                "agreement_rate": (
                    round(len(agreements) / len(self.counterfactual), 3)
                    if self.counterfactual else None
                ),
            },
            "decision_maker_health": {
                "attempts": len(self.decision_attempts),
                "rejected_attempts": len(overrides),
            },
            "planner_exchanges": self.planner_exchanges,
            "executor_commands": self.executor_commands,
            "notes": self.notes,
            "warnings": self.warnings,
        }

    def write(self, path: str, outcome: LoopOutcome) -> str:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(self.to_dict(outcome), indent=2, default=str),
            encoding="utf-8",
        )
        return str(target)


def summary_lines(outcome: LoopOutcome) -> List[str]:
    """A short human account of the run, one line per iteration."""
    lines: List[str] = [f"status: {outcome.status} — {outcome.reason}"]
    state = outcome.state
    if not state:
        return lines

    for attempt in state.attempts:
        diag = attempt.diagnosis
        decision = attempt.decision
        head = (
            f"  [{attempt.index}] {diag.verdict}/{diag.outcome} "
            f"({diag.num_results} result(s))" if diag else f"  [{attempt.index}] ?"
        )
        if decision:
            head += f" -> {decision.action} [{decision.source}]"
            if decision.overridden_from:
                head += f" (overrode '{decision.overridden_from}')"
        lines.append(head)
        if decision and decision.rationale:
            lines.append(f"        {decision.rationale}")
        if attempt.planner_note:
            lines.append(f"        planner: {attempt.planner_note}")

    lines.append(
        f"  {state.iteration} iteration(s), {state.planner_calls} planner call(s), "
        f"{state.arax_calls} ARAX call(s), {state.llm_calls} LLM call(s), "
        f"{state.elapsed_s:.0f}s"
    )
    return lines
