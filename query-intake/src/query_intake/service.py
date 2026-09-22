"""
service.py — The one public entry point.

Everything in this package exists to make this function short and its order
fixed:

    rate limit → parse → intake policy → plan (scope-guarded) → loop →
    public result

The order is the security property. Rate limiting is first because a caller
sending garbage should still be counted; parsing is before policy because there
is nothing to check until the request is a request; policy is before the planner
because a refused question should cost nothing; and the scope gate sits around
the planner rather than after it, so a blocked concept never reaches a graph
query. A single method means that order is stated once and tested once, instead
of being implied by the sequence of statements in a CLI.

**What leaves here is narrow on purpose.** A fixed status from a closed set, a
message from a fixed table, and — when there is one — the answer the controller
already put through its grounding gate. Not the internal reason, not the rule
that fired, not a classifier's explanation, not the trace. Those are written to
disk where an operator can read them.

**The public result is trimmed further than the answer document.** The composed
answer carries a `grounding` block listing every statement that failed its check
and why. That is exactly right for the trace and wrong for a caller: it is a
description of the checker's weak points. The counts survive, the detail does
not.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from loop_controller.compose import ComposerSettings
from loop_controller.contracts import (
    Budget, LOOP_ABSENCE, LOOP_ANSWERED, LOOP_EXHAUSTED, LOOP_FAILED,
    LOOP_REFUSED,
)
from loop_controller.loop import ControllerConfig, LoopController

from . import messages
from .messages import RefusalReason
from .models import (
    DepthLimits, InvalidRequest, PublicResult, PublicStatus, RequestContext,
    UserRequest,
)
from .policy import IntakePolicy
from .ratelimit import InMemoryRateLimiter, RateLimiter
from .redact import redact_trace, summarise_for_log
from .scope import Resolver, ScopeGate, ScopeGuardedPlanner

log = logging.getLogger("query_intake")


#: How the planner's typed refusal reasons map to what a caller is told.
#:
#: `needs_clarification` and `insufficient_information` are not refusals in the
#: same sense as the others — the question could work with more detail — so they
#: get their own status rather than being flattened into "refused". A client
#: should be able to tell "ask me again with more information" from "we will not
#: answer this".
_REFUSAL_MAP: Dict[str, tuple] = {
    "needs_clarification": (
        PublicStatus.NEEDS_CLARIFICATION, messages.NEEDS_CLARIFICATION,
    ),
    "insufficient_information": (
        PublicStatus.NEEDS_CLARIFICATION, messages.NEEDS_CLARIFICATION,
    ),
    "unsafe_or_clinical_advice": (
        PublicStatus.REFUSED, messages.resolve(RefusalReason.CLINICAL_ADVICE),
    ),
    "out_of_scope": (
        PublicStatus.REFUSED, messages.resolve(RefusalReason.OUT_OF_SCOPE),
    ),
    "not_biomedical": (
        PublicStatus.REFUSED, messages.resolve(RefusalReason.NOT_BIOMEDICAL),
    ),
    "requires_capability_not_available": (
        PublicStatus.REFUSED, messages.resolve(RefusalReason.OUT_OF_SCOPE),
    ),
}


@dataclass
class ServiceConfig:
    """Everything the service holds that is not per-request."""

    intake: IntakePolicy = field(default_factory=IntakePolicy)
    rate_limiter: RateLimiter = field(default_factory=InMemoryRateLimiter)
    scope: ScopeGate = field(default_factory=ScopeGate)
    resolver: Optional[Resolver] = None
    runs_dir: str = "runs"
    write_traces: bool = True
    verbose: bool = False
    #: Included in the public result when detection was degraded. Off by
    #: default: telling a caller that the attack classifier is down is telling
    #: the wrong audience.
    disclose_degraded: bool = False


class QueryService:
    """The front door."""

    def __init__(
        self,
        executor: Any,
        planner: Optional[Any] = None,
        decision_maker: Optional[Any] = None,
        config: Optional[ServiceConfig] = None,
    ) -> None:
        self.executor = executor
        self.planner = planner
        self.decision_maker = decision_maker
        self.config = config or ServiceConfig()

    # -- entry points ------------------------------------------------------

    def handle_payload(
        self, payload: Any, context: RequestContext,
    ) -> PublicResult:
        """Entry for an untrusted JSON body."""
        rate = self.config.rate_limiter.check(context.caller_id)
        if not rate.allowed:
            self._log(context, None, "rate_limited", reason=rate.reason)
            return PublicResult(
                status=PublicStatus.RATE_LIMITED,
                message=messages.RATE_LIMITED,
                request_id=context.request_id,
                retry_after_s=rate.retry_after_s,
            )
        try:
            try:
                request = UserRequest.from_payload(payload)
            except InvalidRequest as exc:
                self._log(context, None, "invalid_request", reason=str(exc))
                return PublicResult(
                    status=PublicStatus.INVALID_REQUEST,
                    message=str(exc),
                    request_id=context.request_id,
                )
            return self._process(request, context)
        finally:
            self.config.rate_limiter.release(context.caller_id)

    def handle(
        self, request: UserRequest, context: RequestContext,
    ) -> PublicResult:
        """Entry for an already-typed request."""
        rate = self.config.rate_limiter.check(context.caller_id)
        if not rate.allowed:
            self._log(context, request, "rate_limited", reason=rate.reason)
            return PublicResult(
                status=PublicStatus.RATE_LIMITED,
                message=messages.RATE_LIMITED,
                request_id=context.request_id,
                retry_after_s=rate.retry_after_s,
            )
        try:
            return self._process(request, context)
        finally:
            self.config.rate_limiter.release(context.caller_id)

    # -- the order ---------------------------------------------------------

    def _process(
        self, request: UserRequest, context: RequestContext,
    ) -> PublicResult:
        started = time.time()
        limits = request.limits

        decision = self.config.intake.evaluate(request.question)
        if not decision.allowed:
            self._log(
                context, request, "refused_at_intake",
                reason=decision.reason.value if decision.reason else "unknown",
                matched=decision.matched, detail=decision.detail,
                elapsed_s=round(time.time() - started, 3),
            )
            return PublicResult(
                status=PublicStatus.REFUSED,
                message=messages.resolve(decision.reason)
                if decision.reason else messages.UNAVAILABLE,
                request_id=context.request_id,
                review_depth=request.review_depth,
            )

        if decision.degraded:
            log.warning(
                "request %s: attack detection unavailable (%s)",
                context.request_id,
                (decision.attack.error if decision.attack else "unknown"),
            )

        controller = self._build_controller(context, limits)
        outcome = controller.run(decision.question)

        self._persist_trace(controller, outcome, context)
        result = self._to_public(outcome, request, context, decision.degraded)

        state = outcome.state
        self._log(
            context, request, result.status.value,
            loop_status=outcome.status,
            iterations=state.iteration if state else 0,
            planner_calls=state.planner_calls if state else 0,
            arax_calls=state.arax_calls if state else 0,
            executor_llm_calls=state.llm_calls if state else 0,
            grounded=(outcome.answer or {}).get("grounding", {}).get("ok"),
            withheld=(outcome.answer or {}).get("grounding", {}).get("failed"),
            degraded_detection=decision.degraded,
            elapsed_s=round(time.time() - started, 3),
        )
        return result

    # -- wiring ------------------------------------------------------------

    def _build_controller(
        self, context: RequestContext, limits: DepthLimits,
    ) -> LoopController:
        """One controller per request, configured from the depth alone.

        Built per request rather than shared, because `LoopState` and the trace
        are per-run and a shared controller would interleave them. The executor
        is shared — it is stateless apart from its cache, which is meant to be
        shared.
        """
        planner = self.planner
        if planner is not None:
            planner = ScopeGuardedPlanner(
                planner, self.config.scope, self.config.resolver,
            )

        run_dir = Path(self.config.runs_dir) / context.request_id

        config = ControllerConfig(
            budget=Budget(
                max_iterations=limits.max_iterations,
                max_planner_calls=limits.max_planner_calls,
                max_arax_calls=limits.max_arax_calls,
                max_llm_calls=limits.max_llm_calls,
            ),
            composer=ComposerSettings(top_k=limits.answer_top_k),
            runs_dir=str(run_dir),
            trace_path=None,   # written here, after redaction
            answer_path=None,
            verbose=self.config.verbose,
        )

        self._apply_depth_to_executor(limits)

        return LoopController(
            executor=self.executor,
            planner=planner,
            decision_maker=self.decision_maker,
            config=config,
        )

    def _apply_depth_to_executor(self, limits: DepthLimits) -> None:
        """Put the depth's review settings on the executor's argument list.

        The knobs are replaced rather than appended, so consecutive requests at
        different depths do not accumulate contradictory flags on a shared
        executor. Anything the deployment set for its own reasons — endpoints,
        cache paths, taxon — is left alone.
        """
        settings = getattr(self.executor, "settings", None)
        if settings is None:
            return

        owned = {
            "--no-rerank", "--rerank-top-k", "--no-verify-literature",
            "--verify-top-k", "--max-abstracts", "--max-verify-edges",
        }
        kept: list = []
        skip_next = False
        for arg in list(getattr(settings, "base_args", [])):
            if skip_next:
                skip_next = False
                continue
            if arg in owned:
                skip_next = not arg.startswith("--no-")
                continue
            kept.append(arg)

        settings.base_args = kept + limits.executor_args()

    # -- output ------------------------------------------------------------

    def _to_public(
        self,
        outcome: Any,
        request: UserRequest,
        context: RequestContext,
        degraded: bool,
    ) -> PublicResult:
        status, message = self._classify(outcome)

        answer = None
        if status in (PublicStatus.COMPLETED, PublicStatus.NO_DATA):
            answer = _public_answer(outcome.answer)

        if degraded and self.config.disclose_degraded:
            message = f"{message} (some optional screening was unavailable)"

        return PublicResult(
            status=status,
            message=message,
            request_id=context.request_id,
            review_depth=request.review_depth,
            answer=answer,
        )

    def _classify(self, outcome: Any) -> tuple:
        """Map the loop's ending to a fixed public status and message.

        `outcome.reason` is deliberately not used. It is assembled from
        internals — "planner call budget spent (3)" — which is the right level
        of detail for a trace and an unnecessary description of the
        architecture to hand a stranger.
        """
        if outcome.status == LOOP_ANSWERED:
            return PublicStatus.COMPLETED, messages.COMPLETED

        if outcome.status == LOOP_ABSENCE:
            return PublicStatus.NO_DATA, messages.NO_DATA

        if outcome.status == LOOP_REFUSED:
            reason = _refusal_reason(outcome)
            mapped = _REFUSAL_MAP.get(reason)
            if mapped:
                return mapped
            return PublicStatus.REFUSED, messages.resolve(
                RefusalReason.OUT_OF_SCOPE
            )

        if outcome.status == LOOP_EXHAUSTED:
            return PublicStatus.BUDGET_EXHAUSTED, messages.BUDGET_EXHAUSTED

        if outcome.status == LOOP_FAILED:
            return PublicStatus.UNAVAILABLE, messages.UNAVAILABLE

        return PublicStatus.UNAVAILABLE, messages.UNAVAILABLE

    # -- records -----------------------------------------------------------

    def _persist_trace(
        self, controller: Any, outcome: Any, context: RequestContext,
    ) -> None:
        if not self.config.write_traces:
            return
        try:
            raw = controller.trace.to_dict(outcome)
            redacted = redact_trace(raw, raw=context.log_raw_question)
            redacted["request_id"] = context.request_id
            redacted["caller_id"] = context.caller_id

            target = Path(self.config.runs_dir) / context.request_id / "trace.json"
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(redacted, indent=2, default=str), encoding="utf-8",
            )
        except Exception as exc:  # pragma: no cover - never fail a request on logging
            log.warning("request %s: could not write trace (%s)",
                        context.request_id, exc)

    def _log(
        self,
        context: RequestContext,
        request: Optional[UserRequest],
        event: str,
        **fields: Any,
    ) -> None:
        record = summarise_for_log(
            request_id=context.request_id,
            caller_id=context.caller_id,
            question=request.question if request else "",
            raw=context.log_raw_question,
            event=event,
            review_depth=request.review_depth.value if request else None,
            **fields,
        )
        log.info("%s", json.dumps(record, default=str))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _refusal_reason(outcome: Any) -> str:
    result = getattr(outcome, "final_result", None) or {}
    refusal = result.get("refusal") or {}
    if refusal.get("reason"):
        return str(refusal["reason"])
    plan = getattr(outcome, "final_plan", None) or {}
    return str((plan.get("refusal") or {}).get("reason") or "")


#: Keys of the composed answer that may leave the service.
_PUBLIC_ANSWER_KEYS = (
    "schema", "question", "answer_kind", "statements", "caveats",
    "candidates", "provenance", "composed_by",
)


def _public_answer(answer: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Trim the composed answer to what a caller should see.

    `withheld_statements` and the per-violation detail inside `grounding` are
    dropped. A caller is entitled to know that every statement was checked and
    how many passed — that is the claim the system is making about itself. The
    list of what failed and why is a description of where the checker is weak,
    which belongs in the trace.
    """
    if not answer:
        return None

    out = {k: answer[k] for k in _PUBLIC_ANSWER_KEYS if k in answer}

    grounding = answer.get("grounding") or {}
    out["grounding"] = {
        "checked": grounding.get("checked", 0),
        "passed": grounding.get("passed", 0),
        "withheld": grounding.get("failed", 0),
        "ok": grounding.get("ok", False),
    }
    return out
