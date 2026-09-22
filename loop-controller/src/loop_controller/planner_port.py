"""
planner_port.py — Adapting the planner, and giving it a way to revise.

`PlannerAgent.plan` already does most of what a revision needs: it sends the
system prompt, parses the reply, stamps the runtime-owned version fields,
validates against the schema and the semantic rules, and retries once with the
structured errors appended. What it does not have is an entry point that takes
execution feedback instead of a bare question.

This adapter adds one, and adds it by reusing the planner's own machinery
rather than reimplementing it. The revision goes through the same system
prompt, the same parser, the same version stamping and the same validators, so
a revised plan is a plan in exactly the sense the executor already relies on —
there is no second, looser path into the graph. The only new thing is the user
message, which `revision.build_revision_message` assembles.

Those helpers are private to the planner module. They are imported by name
where available and re-implemented against `plan_core` where they are not, so
this package works with a planner checkout that has moved on, and says so
plainly instead of failing at the point of use.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .ports import PlanAttempt
from .revision import RevisionRequest, build_revision_message


# ---------------------------------------------------------------------------
# Planner internals, borrowed where possible
# ---------------------------------------------------------------------------

def _load_planner_helpers() -> Tuple[Callable, Callable, Callable, bool]:
    """Return (parse, stamp, validate, borrowed).

    ``borrowed`` records whether the planner's own helpers were found. It ends
    up in the trace: a revision validated by a locally re-implemented stamp is
    a slightly different guarantee from one validated by the planner's, and the
    difference should be visible rather than assumed away.
    """
    try:
        from planner_agent.planner_agent import (  # type: ignore
            _try_parse, _stamp_versions, _validate_with_available_inputs,
        )
        return _try_parse, _stamp_versions, _validate_with_available_inputs, True
    except Exception:
        pass

    from plan_core import PLAN_VERSION, load_biolink_vocabulary, validate_plan

    def parse(raw: str):
        text = (raw or "").strip()
        if text.startswith("```"):
            first_nl = text.find("\n")
            text = text[first_nl + 1:] if first_nl != -1 else text
            if text.endswith("```"):
                text = text[:-3].rstrip()
        try:
            return json.loads(text), None
        except json.JSONDecodeError as exc:
            return None, f"JSON parse error: {exc}"

    def stamp(parsed):
        if isinstance(parsed, dict):
            parsed["plan_version"] = PLAN_VERSION
            parsed["biolink_version"] = load_biolink_vocabulary().biolink_version

    def validate(plan, available_inputs):
        # The manifest cross-check lives in the planner. Without it, an
        # external input binding cannot be verified here, so a revision that
        # introduces one is reported as unvalidatable rather than accepted.
        result = validate_plan(plan)
        refs = {i.get("input_ref") for i in (available_inputs or [])}
        for entity in plan.get("entities") or []:
            binding = (entity or {}).get("input_binding") or {}
            ref = binding.get("input_ref")
            if ref and ref not in refs:
                result.ok = False
                result.errors.append(_SimpleError(
                    "external_input",
                    f"entities/{entity.get('entity_ref')}",
                    f"input_ref '{ref}' is not in the available-inputs manifest",
                ))
        return result

    return parse, stamp, validate, False


@dataclass
class _SimpleError:
    kind: str
    location: str
    message: str


# ---------------------------------------------------------------------------
# Adapter
# ---------------------------------------------------------------------------


class PlannerAgentPort:
    """Wraps a `PlannerAgent` and adds `revise`."""

    def __init__(self, agent: Any, *, max_revision_attempts: int = 2) -> None:
        self.agent = agent
        self.max_revision_attempts = max_revision_attempts
        (
            self._parse, self._stamp, self._validate, self.borrowed_helpers,
        ) = _load_planner_helpers()
        self.exchanges: List[Dict[str, Any]] = []

    # -- plan --------------------------------------------------------------

    def plan(
        self,
        question: str,
        *,
        available_inputs: Optional[Sequence[dict]] = None,
    ) -> PlanAttempt:
        result = self.agent.plan(question, available_inputs=available_inputs)
        return _from_planner_result(result)

    # -- revise ------------------------------------------------------------

    def revise(self, request: RevisionRequest) -> PlanAttempt:
        """Ask for a plan that addresses a diagnosed execution failure.

        Mirrors `plan()`'s flow deliberately: one attempt, and on a validation
        failure one more with the errors appended. Bounded for the same reason
        the decision maker is — this runs inside a loop that already has a
        budget, and an unbounded repair here would quietly become that budget.
        """
        message = build_revision_message(request)
        errors: List[str] = []

        for attempt in range(1, self.max_revision_attempts + 1):
            body = message
            if errors:
                body = message + _validation_feedback(errors)

            try:
                raw = self.agent.llm.complete(self.agent.system_prompt, body)
            except Exception as exc:
                return PlanAttempt(
                    ok=False, attempts=attempt,
                    errors=[f"planner call failed: {exc}"],
                    note="the planner could not be reached",
                )

            self.exchanges.append({
                "action": request.action,
                "attempt": attempt,
                "raw_response": (raw or "")[:8000],
            })

            parsed, parse_error = self._parse(raw)
            if parse_error:
                errors = [parse_error]
                continue

            self._stamp(parsed)

            if isinstance(parsed.get("refusal"), dict):
                refusal = parsed["refusal"]
                return PlanAttempt(
                    ok=False, plan=parsed, refused=True,
                    refusal_reason=str(refusal.get("reason") or "refused"),
                    attempts=attempt, raw_response=raw or "",
                    note="the planner declined to revise rather than emit a "
                         "plan it does not believe can work",
                )

            validation = self._validate(parsed, list(request.available_inputs))
            if getattr(validation, "ok", False):
                return PlanAttempt(
                    ok=True, plan=parsed, attempts=attempt,
                    raw_response=raw or "",
                    note=f"revised for {request.action}",
                )

            errors = [
                f"[{getattr(e, 'kind', '?')}] {getattr(e, 'location', '?')}: "
                f"{getattr(e, 'message', e)}"
                for e in getattr(validation, "errors", [])[:20]
            ]

        return PlanAttempt(
            ok=False, attempts=self.max_revision_attempts, errors=errors,
            note="the revision did not validate",
        )


def _validation_feedback(errors: Sequence[str]) -> str:
    body = "\n".join(f"- {e}" for e in errors[:20])
    return (
        "\n\nYOUR REVISION FAILED VALIDATION\n\n"
        f"{body}\n\n"
        "Correct these while preserving every semantic choice that was not "
        "identified as invalid, including entity and path identifiers. Omit "
        "unused optional fields instead of emitting JSON null. Emit the "
        "corrected plan and no commentary.\n"
    )


def _from_planner_result(result: Any) -> PlanAttempt:
    """Normalise a `PlannerResult` into the loop's own shape."""
    plan_obj = getattr(result, "plan", None)
    plan_dict: Optional[Dict[str, Any]] = None
    if plan_obj is not None:
        dump = getattr(plan_obj, "model_dump", None)
        plan_dict = dump(exclude_none=True) if callable(dump) else dict(plan_obj)

    refused = bool(plan_dict and plan_dict.get("refusal"))
    refusal_reason = None
    if refused:
        refusal_reason = str((plan_dict.get("refusal") or {}).get("reason") or "refused")

    validation = getattr(result, "validation", None)
    errors = [
        f"[{getattr(e, 'kind', '?')}] {getattr(e, 'location', '?')}: "
        f"{getattr(e, 'message', e)}"
        for e in (getattr(validation, "errors", None) or [])[:20]
    ]
    if getattr(result, "error", None):
        errors.insert(0, str(result.error))

    return PlanAttempt(
        ok=bool(getattr(result, "ok", False)) and not refused,
        plan=plan_dict,
        refused=refused,
        refusal_reason=refusal_reason,
        errors=errors,
        attempts=int(getattr(result, "attempts", 0) or 0),
        raw_response=str(getattr(result, "raw_response", "") or ""),
    )


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class ScriptedPlanner:
    """Returns prepared plans in order, recording what it was asked for.

    The recorded requests are the point: a test asserts not only that the loop
    asked for a revision but that it asked for the right one — the tightest
    unspent axis, the locked constraints listed, the prior fingerprint
    forbidden.
    """

    def __init__(self, attempts: Sequence[PlanAttempt]) -> None:
        self.attempts = list(attempts)
        self.requests: List[Any] = []
        self.index = 0

    def _next(self) -> PlanAttempt:
        if self.index < len(self.attempts):
            attempt = self.attempts[self.index]
        else:
            attempt = PlanAttempt(
                ok=False, errors=["the scripted planner ran out of plans"],
            )
        self.index += 1
        return attempt

    def plan(
        self,
        question: str,
        *,
        available_inputs: Optional[Sequence[dict]] = None,
    ) -> PlanAttempt:
        self.requests.append({"kind": "plan", "question": question})
        return self._next()

    def revise(self, request: RevisionRequest) -> PlanAttempt:
        self.requests.append(request)
        return self._next()


class NoPlanner:
    """A planner-shaped absence.

    Used when the controller is given a hand-written plan and no planner. The
    policy layer already refuses `repair_plan` and `relax_plan` when no planner
    is configured, so this exists to make "not configured" a value rather than
    a None check at each call site.
    """

    def plan(
        self,
        question: str,
        *,
        available_inputs: Optional[Sequence[dict]] = None,
    ) -> PlanAttempt:
        return PlanAttempt(ok=False, errors=["no planner is configured"])

    def revise(self, request: RevisionRequest) -> PlanAttempt:
        return PlanAttempt(ok=False, errors=["no planner is configured"])
