"""query_intake — the front door: request policy, scope, and one entry point."""

__version__ = "0.1.0"

from .models import (
    DEPTH_LIMITS, DepthLimits, INTAKE_SCHEMA_VERSION, InvalidRequest,
    PublicResult, PublicStatus, RequestContext, ReviewDepth, UserRequest,
    limits_for,
)
from .messages import RefusalReason, REFUSAL_MESSAGES, resolve
from .checks import MAX_QUESTION_CHARS, TextCheck, check_text, normalise
from .detectors import (
    AttackVerdict, CompositeDetector, NullDetector, PromptAttackDetector,
    PromptGuardDetector, RuleBasedDetector, default_detector,
)
from .policy import IntakeDecision, IntakePolicy
from .ratelimit import (
    InMemoryRateLimiter, NoRateLimit, RateDecision, RateLimiter, RateLimits,
)
from .redact import digest, question_for_log, redact_trace, summarise_for_log
from .scope import (
    ScopeDecision, ScopeGate, ScopeGuardedPlanner, ScopeRule, ScopeRules,
    expand_closure, static_resolver, write_closure,
)
from .service import QueryService, ServiceConfig

__all__ = [
    "DEPTH_LIMITS", "DepthLimits", "INTAKE_SCHEMA_VERSION", "InvalidRequest",
    "PublicResult", "PublicStatus", "RequestContext", "ReviewDepth",
    "UserRequest", "limits_for",
    "RefusalReason", "REFUSAL_MESSAGES", "resolve",
    "MAX_QUESTION_CHARS", "TextCheck", "check_text", "normalise",
    "AttackVerdict", "CompositeDetector", "NullDetector",
    "PromptAttackDetector", "PromptGuardDetector", "RuleBasedDetector",
    "default_detector",
    "IntakeDecision", "IntakePolicy",
    "InMemoryRateLimiter", "NoRateLimit", "RateDecision", "RateLimiter",
    "RateLimits",
    "digest", "question_for_log", "redact_trace", "summarise_for_log",
    "ScopeDecision", "ScopeGate", "ScopeGuardedPlanner", "ScopeRule",
    "ScopeRules", "expand_closure", "static_resolver", "write_closure",
    "QueryService", "ServiceConfig",
]
