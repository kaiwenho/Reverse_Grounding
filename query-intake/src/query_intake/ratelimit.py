"""
ratelimit.py — Bounding what one caller can spend.

This is the guard that matters most in practice, and it is the one that was
missing entirely. One question costs several ARAX queries and a handful of local
model calls, and a deep review reads abstracts one model call at a time. Nobody
needs to defeat a prompt filter to make that expensive; they only need a loop
and a few minutes. Cost abuse is the cheap attack, and no classifier addresses
it.

Two limits, because they bound different things:

**Requests per window** stops a flood. **Concurrent requests** stops a caller
holding open several multi-minute runs at once, which a per-window count does
not catch — ten requests an hour is a reasonable rate and ten simultaneous deep
searches is not.

The implementation is in-memory and per-process. That is honest about what it
is: correct for a single worker, and wrong the moment there are two, because
each keeps its own counters and a caller gets N times the allowance. The
`RateLimiter` protocol exists so a deployment can put a shared store behind it;
`InMemoryRateLimiter` is what you develop against, not what you scale on. The
docstring says so rather than leaving someone to find out.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Deque, Dict, Protocol


@dataclass
class RateDecision:
    allowed: bool
    retry_after_s: float = 0.0
    reason: str = ""
    remaining: int = 0


class RateLimiter(Protocol):
    def check(self, caller_id: str) -> RateDecision: ...

    def release(self, caller_id: str) -> None: ...


@dataclass(frozen=True)
class RateLimits:
    """Per-caller ceilings.

    Defaults are deliberately low. A researcher asking questions by hand will
    never notice thirty an hour; a script will. Raising them is a decision
    someone should make on purpose.
    """

    requests_per_window: int = 30
    window_s: float = 3600.0
    max_concurrent: int = 2


class InMemoryRateLimiter:
    """Sliding window plus a concurrency count, in this process only.

    Correct for one worker. With several, each holds its own counters and a
    caller effectively gets the limit multiplied by the worker count — swap in a
    shared implementation before running more than one.
    """

    def __init__(self, limits: RateLimits | None = None) -> None:
        self.limits = limits or RateLimits()
        self._events: Dict[str, Deque[float]] = defaultdict(deque)
        self._active: Dict[str, int] = defaultdict(int)
        self._lock = threading.Lock()

    def check(self, caller_id: str) -> RateDecision:
        now = time.time()
        limits = self.limits

        with self._lock:
            events = self._events[caller_id]
            cutoff = now - limits.window_s
            while events and events[0] < cutoff:
                events.popleft()

            if self._active[caller_id] >= limits.max_concurrent:
                return RateDecision(
                    allowed=False,
                    retry_after_s=30.0,
                    reason=(
                        f"{self._active[caller_id]} request(s) already running "
                        f"for this caller, limit {limits.max_concurrent}"
                    ),
                    remaining=max(0, limits.requests_per_window - len(events)),
                )

            if len(events) >= limits.requests_per_window:
                oldest = events[0]
                return RateDecision(
                    allowed=False,
                    retry_after_s=max(0.0, (oldest + limits.window_s) - now),
                    reason=(
                        f"{len(events)} request(s) in the last "
                        f"{limits.window_s:.0f}s, limit "
                        f"{limits.requests_per_window}"
                    ),
                    remaining=0,
                )

            events.append(now)
            self._active[caller_id] += 1
            return RateDecision(
                allowed=True,
                remaining=limits.requests_per_window - len(events),
            )

    def release(self, caller_id: str) -> None:
        """Give back the concurrency slot.

        Must run whether the request succeeded, refused or raised — the service
        calls it in a `finally`. A slot leaked on an error path locks a caller
        out permanently, and the symptom (one user cannot make requests, nothing
        in the logs) is unpleasant to diagnose.
        """
        with self._lock:
            if self._active[caller_id] > 0:
                self._active[caller_id] -= 1


class NoRateLimit:
    """Allows everything. For tests and single-user local runs."""

    def check(self, caller_id: str) -> RateDecision:
        return RateDecision(allowed=True, remaining=1)

    def release(self, caller_id: str) -> None:
        return None
