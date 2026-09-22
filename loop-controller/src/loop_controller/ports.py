"""
ports.py — What the loop needs from a planner and an executor.

The loop is written against these two interfaces and never against a concrete
planner or executor. That is worth the small indirection for three reasons.

The executor runs as a subprocess, so there is no import to write the loop
against in the first place; something has to describe the boundary, and a
protocol describes it in the type system rather than in a comment.

Both dependencies are expensive and stateful — one calls a knowledge graph and
a local model, the other calls a model — so every test that exercises the loop
needs a substitute. Scripted implementations of these two protocols are the
whole of the test harness, and a recorded result document replayed through a
scripted executor exercises the real diagnosis, policy and composition code.

And the planner is optional. Without one the controller can still execute a
hand-written plan, diagnose the run, and compose an answer; it simply cannot
repair or relax. The policy layer already reports that as a refusal reason
rather than a crash, which only works because "no planner" is representable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Sequence


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------


@dataclass
class PlanAttempt:
    """A plan, or the reason there isn't one.

    ``plan`` is the plain dict form, because that is what gets fingerprinted,
    written to disk and handed to the executor. ``errors`` carries validation
    failures so the loop can report *why* a revision could not be produced —
    which is a different situation from a revision that ran and found nothing,
    and the trace should not blur them.
    """

    ok: bool
    plan: Optional[Dict[str, Any]] = None
    refused: bool = False
    refusal_reason: Optional[str] = None
    errors: List[str] = field(default_factory=list)
    attempts: int = 0
    raw_response: str = ""
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ok": self.ok,
            "refused": self.refused,
            "refusal_reason": self.refusal_reason,
            "errors": self.errors[:20],
            "attempts": self.attempts,
            "note": self.note,
        }


class PlannerPort(Protocol):
    """A source of validated plans."""

    def plan(
        self,
        question: str,
        *,
        available_inputs: Optional[Sequence[dict]] = None,
    ) -> PlanAttempt:
        """Draft a plan for a question."""
        ...

    def revise(self, request: "Any") -> PlanAttempt:
        """Produce a new plan addressing a diagnosed failure.

        Takes a ``RevisionRequest``. Typed loosely here to keep this module
        free of a circular import with ``revision``.
        """
        ...


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


@dataclass
class ExecutionRun:
    """One execution of one plan.

    ``result`` is always a document, even when the run crashed: a synthesised
    backend-failure result rather than None. The alternative is a null check in
    every stage downstream, and a loop that can crash on the path it takes when
    something has already gone wrong is a loop that fails exactly when its
    reporting matters most.
    """

    result: Dict[str, Any]
    hints: Dict[str, Any] = field(default_factory=dict)
    exit_code: int = 0
    plan_path: Optional[str] = None
    result_path: Optional[str] = None
    hints_path: Optional[str] = None
    stderr_tail: str = ""
    llm_unavailable: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "exit_code": self.exit_code,
            "plan_path": self.plan_path,
            "result_path": self.result_path,
            "hints_path": self.hints_path,
            "llm_unavailable": self.llm_unavailable,
            "stderr_tail": self.stderr_tail[-1000:],
        }


class ExecutorPort(Protocol):
    """A way to run a plan against the graph."""

    def run(
        self,
        plan: Dict[str, Any],
        *,
        overrides: Optional[Dict[str, Any]] = None,
        tag: str = "",
    ) -> ExecutionRun:
        ...


# ---------------------------------------------------------------------------
# Synthesised results
# ---------------------------------------------------------------------------


def failure_result(
    plan: Optional[Dict[str, Any]],
    outcome: str,
    detail: str,
    *,
    replannable: bool = False,
    verdict: str = "error",
) -> Dict[str, Any]:
    """A result document for a run that never produced one.

    Shaped like the executor's own output so that diagnosis, policy and the
    composer treat it as an ordinary result rather than a special case. The
    executor cannot report on a run it failed to start, so something has to
    write that report, and writing it in the executor's own vocabulary keeps
    the number of shapes the rest of the package handles at one.
    """
    plan = plan or {}
    return {
        "schema": "plan-executor-result/0.1.0",
        "plan": {
            "plan_id": plan.get("plan_id"),
            "plan_version": plan.get("plan_version"),
            "biolink_version": plan.get("biolink_version"),
            "question": plan.get("question"),
            "plan_mode": plan.get("plan_mode"),
            "interpretation": plan.get("interpretation"),
            "confidence": plan.get("confidence"),
            "gaps": plan.get("gaps") or [],
        },
        "verdict": verdict,
        "verdict_reasons": [detail],
        "outcome": {
            "outcome": outcome,
            "detail": detail,
            "replannable": replannable,
        },
        "results": [],
        "paths": {},
        "elapsed_s": 0.0,
    }
