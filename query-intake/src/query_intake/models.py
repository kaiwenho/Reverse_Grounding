"""
models.py — What a caller may send, and what they get back.

Two types carry the whole security posture of this layer, and the difference
between them is the point.

``UserRequest`` is everything a caller controls: a question, and one of four
named review depths. Nothing else. Not candidate counts, not iteration limits,
not model names, not endpoints, not prompts. Every one of those is a cost lever
or a behaviour lever, and a lever the caller can pull is a lever an attacker can
pull. The controller's own CLI exposes several of them, which is right for a
developer running it by hand and wrong for anything facing a network.

``RequestContext`` is everything the *server* knows: who is calling, what
they've already spent, a request id. It is constructed by whatever authenticates
the caller — never from the request body. If identity could arrive in the
payload, a caller could claim someone else's quota, and rate limiting would be
decorative. There is a test for exactly this.

``ReviewDepth`` deserves its own note. It names how much optional model work the
executor may do: reranking candidates, and checking the abstracts attached to
returned edges. It does **not** mean "no language model anywhere" — the executor
requires one to disambiguate entity names and refuses to run without it, because
an unreviewed anchor silently produces confident results about the wrong
concept. `off` means no reranking and no abstract review. Deterministic
graph-grounding of the final answer runs at every depth, including `off`.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Dict, List, Mapping, Optional


INTAKE_SCHEMA_VERSION = "query-intake/0.1.0"


# ---------------------------------------------------------------------------
# Review depth
# ---------------------------------------------------------------------------


class ReviewDepth(str, Enum):
    """How much optional model review the executor may perform."""

    OFF = "off"
    LIGHT = "light"
    STANDARD = "standard"
    DEEP = "deep"

    @classmethod
    def parse(cls, value: Any) -> "ReviewDepth":
        """Accept only the four names. Anything else is an invalid request.

        Not coerced, not defaulted. A caller who sends `review_depth: 99` has
        sent a request this service does not understand, and quietly treating
        it as `standard` would hide a client bug and bill someone for the
        difference.
        """
        if isinstance(value, cls):
            return value
        if isinstance(value, str):
            try:
                return cls(value.strip().lower())
            except ValueError:
                pass
        raise InvalidRequest(
            f"review_depth must be one of "
            f"{', '.join(d.value for d in cls)}"
        )


@dataclass(frozen=True)
class DepthLimits:
    """The server-owned work ceiling for one depth.

    Frozen, and built once at import. These are not defaults a caller may
    override; they are the definition of what the named depth means.
    """

    rerank: bool
    verify_literature: bool
    rerank_top_k: int
    verify_top_k: int
    max_abstracts: int
    max_verify_edges: int
    answer_top_k: int
    max_iterations: int
    max_planner_calls: int
    max_arax_calls: int
    max_llm_calls: int

    def executor_args(self) -> List[str]:
        """Render as arguments for `plan_executor.run_plan`.

        The executor's flags are the authority on what these mean; this is a
        translation, not a second set of defaults. Anything not named here
        keeps the executor's own default.
        """
        argv: List[str] = []
        if not self.rerank:
            argv.append("--no-rerank")
        else:
            argv += ["--rerank-top-k", str(self.rerank_top_k)]
        if not self.verify_literature:
            argv.append("--no-verify-literature")
        else:
            argv += [
                "--verify-top-k", str(self.verify_top_k),
                "--max-abstracts", str(self.max_abstracts),
                "--max-verify-edges", str(self.max_verify_edges),
            ]
        return argv


#: The four levels. `off` still runs entity resolution (the executor requires
#: it) and still runs the deterministic grounding check on the composed answer.
#:
#: The `max_llm_calls` figures are deliberately generous relative to the review
#: settings, because the controller's own decision calls and the planner's
#: revision calls are not yet counted against that ceiling — see the project's
#: open-issues register. Treat them as a backstop, not as an accurate budget.
_LIMITS: Dict[ReviewDepth, DepthLimits] = {
    ReviewDepth.OFF: DepthLimits(
        rerank=False, verify_literature=False,
        rerank_top_k=0, verify_top_k=0, max_abstracts=0, max_verify_edges=0,
        answer_top_k=20,
        max_iterations=3, max_planner_calls=2,
        max_arax_calls=60, max_llm_calls=40,
    ),
    ReviewDepth.LIGHT: DepthLimits(
        rerank=True, verify_literature=True,
        rerank_top_k=5, verify_top_k=5, max_abstracts=1, max_verify_edges=25,
        answer_top_k=20,
        max_iterations=3, max_planner_calls=2,
        max_arax_calls=80, max_llm_calls=120,
    ),
    ReviewDepth.STANDARD: DepthLimits(
        rerank=True, verify_literature=True,
        rerank_top_k=20, verify_top_k=20, max_abstracts=3, max_verify_edges=100,
        answer_top_k=25,
        max_iterations=4, max_planner_calls=3,
        max_arax_calls=150, max_llm_calls=400,
    ),
    ReviewDepth.DEEP: DepthLimits(
        rerank=True, verify_literature=True,
        rerank_top_k=50, verify_top_k=50, max_abstracts=5, max_verify_edges=200,
        answer_top_k=50,
        max_iterations=4, max_planner_calls=3,
        max_arax_calls=250, max_llm_calls=900,
    ),
}

#: Read-only view. Handed out rather than the dict, so a caller holding a
#: reference cannot edit the meaning of a depth for everyone else in-process.
DEPTH_LIMITS: Mapping[ReviewDepth, DepthLimits] = MappingProxyType(_LIMITS)


def limits_for(depth: ReviewDepth) -> DepthLimits:
    return DEPTH_LIMITS[depth]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class InvalidRequest(ValueError):
    """The request is not one this service can interpret.

    Distinct from a refusal. A refusal means "we understood and declined"; this
    means "this is not a well-formed request", and the two should not look the
    same to a client trying to fix its code.
    """


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class UserRequest:
    """Everything the caller controls. Two fields, and no more."""

    question: str
    review_depth: ReviewDepth = ReviewDepth.STANDARD

    @classmethod
    def from_payload(cls, payload: Any) -> "UserRequest":
        """Build from an untrusted JSON body.

        Unknown keys are rejected rather than ignored. A caller sending
        `max_results` should be told the field does not exist, not silently
        served something other than what they asked for — and a caller sending
        `caller_id` should certainly not be humoured.
        """
        if not isinstance(payload, dict):
            raise InvalidRequest("request body must be a JSON object")

        allowed = {"question", "review_depth"}
        unknown = sorted(set(payload) - allowed)
        if unknown:
            raise InvalidRequest(
                f"unknown field(s) {unknown}; this endpoint accepts only "
                f"{sorted(allowed)}"
            )

        question = payload.get("question")
        if not isinstance(question, str):
            raise InvalidRequest("question must be a string")

        depth = payload.get("review_depth", ReviewDepth.STANDARD.value)
        return cls(question=question, review_depth=ReviewDepth.parse(depth))

    @property
    def limits(self) -> DepthLimits:
        return limits_for(self.review_depth)


@dataclass
class RequestContext:
    """Everything the server knows. Never built from the request body.

    ``caller_id`` comes from whatever authenticated the request. If it could
    arrive in the payload, a caller could spend someone else's quota and the
    rate limiter would be decorative.

    ``log_raw_question`` defaults to False. The trace is the artefact that ends
    up in a log store or in front of a reviewer, and a biomedical question can
    identify a person even in a service that refuses patient-specific ones. The
    digest is enough to spot a repeated question or correlate a report; the
    text is not needed for either.
    """

    caller_id: str = "anonymous"
    request_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    received_at: float = 0.0
    log_raw_question: bool = False
    attributes: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


class PublicStatus(str, Enum):
    """What a caller is told. A closed set with fixed meanings.

    `NO_DATA` is separate from `COMPLETED` and from every failure, because an
    established absence is a finding: the query ran to completion and the graph
    holds nothing. Collapsing it into an error would throw away the most
    defensible thing the system can say when it cannot answer.
    """

    COMPLETED = "completed"
    NO_DATA = "no_data"
    REFUSED = "refused"
    NEEDS_CLARIFICATION = "needs_clarification"
    RATE_LIMITED = "rate_limited"
    INVALID_REQUEST = "invalid_request"
    BUDGET_EXHAUSTED = "budget_exhausted"
    UNAVAILABLE = "unavailable"


@dataclass
class PublicResult:
    """What leaves this service.

    ``message`` is always drawn from the fixed table in `messages.py`. It is
    never assembled from an internal reason, a classifier's explanation, or
    anything a model wrote — those go to the trace, which stays inside.

    ``answer`` is the composed answer document when there is one, already
    checked by the controller's grounding gate before it got here.
    """

    status: PublicStatus
    message: str
    request_id: str
    review_depth: Optional[ReviewDepth] = None
    answer: Optional[Dict[str, Any]] = None
    retry_after_s: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "schema": INTAKE_SCHEMA_VERSION,
            "status": self.status.value,
            "message": self.message,
            "request_id": self.request_id,
        }
        if self.review_depth is not None:
            out["review_depth"] = self.review_depth.value
        if self.answer is not None:
            out["answer"] = self.answer
        if self.retry_after_s is not None:
            out["retry_after_s"] = round(self.retry_after_s, 1)
        return out
