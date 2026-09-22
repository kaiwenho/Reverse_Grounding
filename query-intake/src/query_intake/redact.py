"""
redact.py — Keeping raw questions out of persisted logs.

The controller's trace is the artefact that ends up in a log store, in a bug
report, or in front of a reviewer. It currently records the question verbatim,
which is the right default for a developer debugging on their own machine and
the wrong one for a service.

A biomedical question can identify a person even in a service that refuses
patient-specific ones — a rare disease plus a treatment plus a timeframe is
often enough, and the refusal happens *after* the text has already been handled.
Storing a digest instead keeps what operators actually use the field for
(spotting a repeated question, correlating a report with a run) and drops what
they don't need.

The digest is salted. Without a salt, a short question drawn from a small space
of likely phrasings can be recovered by hashing candidates until one matches,
which makes the digest a thin disguise rather than a redaction. Set
`INTAKE_LOG_SALT`; the default is a per-process random value, which means
digests do not correlate across restarts — safe, and a limitation worth knowing
before someone builds a dashboard on them.

`log_raw_question` on the context turns this off where it is appropriate: local
development, or a deployment whose callers have consented and whose log store is
governed accordingly. It is per-request rather than global so the decision sits
with whatever authenticated the caller.
"""

from __future__ import annotations

import hashlib
import os
import secrets
from typing import Any, Dict


_DEFAULT_SALT = secrets.token_hex(16)


def _salt() -> str:
    return os.environ.get("INTAKE_LOG_SALT") or _DEFAULT_SALT


def digest(text: str, *, length: int = 16) -> str:
    """A salted, truncated digest of a question."""
    payload = f"{_salt()}::{text}".encode("utf-8")
    return "q:" + hashlib.sha256(payload).hexdigest()[:length]


def question_for_log(text: str, *, raw: bool = False) -> str:
    """The question as it should be persisted."""
    return text if raw else digest(text)


def redact_trace(trace: Dict[str, Any], *, raw: bool = False) -> Dict[str, Any]:
    """Replace question text in a controller trace with a digest.

    Shallow by design: it rewrites the fields known to carry the question rather
    than scanning the whole document for anything resembling one. A scan would
    be guesswork, and this needs to be predictable — an operator should be able
    to say exactly which fields are redacted.

    The planner exchanges are the important case. Each carries the full revision
    message, which quotes the question inside a much longer prompt, so those are
    dropped rather than rewritten. What they are useful for — did the planner
    produce a valid revision, was it a duplicate — survives in the loop state.
    """
    if raw:
        return trace

    out = dict(trace)
    if out.get("question"):
        out["question"] = digest(str(out["question"]))

    state = out.get("state")
    if isinstance(state, dict) and state.get("question"):
        state = dict(state)
        state["question"] = digest(str(state["question"]))
        out["state"] = state

    if out.get("planner_exchanges"):
        out["planner_exchanges"] = [
            {
                k: v for k, v in exchange.items()
                if k in {"kind", "round", "ok", "refused", "refusal_reason",
                         "attempts", "note", "errors"}
            }
            for exchange in out["planner_exchanges"]
            if isinstance(exchange, dict)
        ]
        out["planner_exchanges_note"] = (
            "prompts and raw responses omitted; they quote the question"
        )

    if out.get("decision_attempts"):
        out["decision_attempts"] = [
            {k: v for k, v in attempt.items() if k != "raw_response"}
            for attempt in out["decision_attempts"]
            if isinstance(attempt, dict)
        ]

    return out


def summarise_for_log(
    *,
    request_id: str,
    caller_id: str,
    question: str,
    raw: bool = False,
    **fields: Any,
) -> Dict[str, Any]:
    """One structured log line for a request.

    Everything an operator needs to answer "what happened to request X" without
    holding the question: who asked, what was decided, what it cost, how it
    ended.
    """
    record: Dict[str, Any] = {
        "request_id": request_id,
        "caller_id": caller_id,
        "question": question_for_log(question, raw=raw),
        "question_chars": len(question),
    }
    record.update(fields)
    return record
