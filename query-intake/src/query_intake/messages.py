"""
messages.py — Every sentence this service is allowed to say.

One table, one place. A refusal reason and the words the caller sees live
side by side so they cannot drift apart, and nothing else in the package is
permitted to build a message from an internal detail.

This exists for the same reason the controller's composer renders refusals from
a typed reason rather than from the planner's prose: the moment a message is
assembled from whatever went wrong, it starts leaking. Internal reasons say
things like "planner call budget spent (3)" — accurate, useful in a trace, and
an unnecessary description of your architecture to hand a stranger. A
classifier's explanation is worse, because it is model-written text going
straight to a user, which is the one thing this whole system is built to
prevent.

So: the reason is a typed value, the message is a constant, and the detail is
written to the trace. `resolve()` is the only way to get a message, and an
unrecognised reason yields a deliberately bland fallback rather than an
exception — a service that crashes while explaining a refusal has turned a
handled case into an outage.
"""

from __future__ import annotations

from enum import Enum
from typing import Dict


class RefusalReason(str, Enum):
    """Why a request was declined, as a typed value."""

    EMPTY_QUESTION = "empty_question"
    TOO_LONG = "too_long"
    MALFORMED_TEXT = "malformed_text"
    CLINICAL_ADVICE = "clinical_advice"
    PROMPT_ATTACK = "prompt_attack"
    OUT_OF_SCOPE = "out_of_scope"
    NOT_BIOMEDICAL = "not_biomedical"


#: The public wording. Written to be true, useful, and free of internals.
#:
#: Each says what happened and, where there is one, what the caller could do
#: instead. None names a component, a model, a limit value, or a rule id — the
#: rule that fired goes to the trace, where an operator can look it up.
REFUSAL_MESSAGES: Dict[RefusalReason, str] = {
    RefusalReason.EMPTY_QUESTION: (
        "No question was provided."
    ),
    RefusalReason.TOO_LONG: (
        "That question is longer than this service accepts. Please shorten it "
        "to a single research question."
    ),
    RefusalReason.MALFORMED_TEXT: (
        "That question contains characters this service cannot process. Please "
        "send plain text."
    ),
    RefusalReason.CLINICAL_ADVICE: (
        "This service answers population-level biomedical research questions "
        "about what a knowledge graph asserts. It cannot advise on the "
        "diagnosis, treatment, or medication of a particular person. A "
        "clinician is the right source for that. The same underlying topic "
        "phrased as a research question — for example, which drugs the graph "
        "records as treating a condition — is in scope."
    ),
    RefusalReason.PROMPT_ATTACK: (
        "That request could not be processed as a research question."
    ),
    RefusalReason.OUT_OF_SCOPE: (
        "This question falls outside the topics this service covers."
    ),
    RefusalReason.NOT_BIOMEDICAL: (
        "This service answers biomedical research questions about entities and "
        "their asserted relationships. That question does not appear to be one."
    ),
}

#: Messages for outcomes that are not refusals.
RATE_LIMITED = (
    "Too many requests. Please wait before sending another."
)
BUDGET_EXHAUSTED = (
    "The search did not reach a conclusion within the work this service "
    "allows for one question. A narrower question may succeed."
)
UNAVAILABLE = (
    "This service is temporarily unable to answer. Please try again later."
)
NEEDS_CLARIFICATION = (
    "That question is missing information needed to build a graph query. "
    "Please state the entities or relationship you want examined."
)
NO_DATA = (
    "The query ran to completion and the knowledge graph holds no matching "
    "data. This is a result about the graph, not a failure of the search."
)
COMPLETED = (
    "Results were found. Every statement in the answer is supported by edges "
    "present in the returned graph."
)

_FALLBACK = "This request could not be completed."


def resolve(reason: RefusalReason) -> str:
    """The public wording for a refusal reason.

    Falls back rather than raising. A reason added upstream and not yet given
    wording is a documentation bug; turning it into an unhandled exception at
    the moment of refusing turns it into an outage.
    """
    return REFUSAL_MESSAGES.get(reason, _FALLBACK)
