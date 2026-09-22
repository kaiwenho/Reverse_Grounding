"""
checks.py — Deterministic checks on the raw question text.

These run before any model, cost nothing, and cannot fail open. They are the
floor: whatever else is configured, these always hold.

**The governing rule is that ambiguity passes through.** Every pattern here is
written for precision, not recall. A question that is merely unusual, or that
brushes against a topic, reaches the planner — which has its own refusal for
clinical advice, and which can only emit a Biolink-valid plan in any case. This
is deliberate and it is the design decision most likely to be quietly reversed
by a future maintainer tightening the rules after one bad report. Don't. A
filter tuned until nothing questionable gets through is a filter that has
stopped answering research questions, and the loop behind it already bounds
what a bad question can do.

**There is no "is this biomedical?" text check here, on purpose.** Deciding that
from raw words needs either a keyword list that fails on every synonym, or a
model whose judgement nobody can reproduce. The reliable version happens later:
the planner names the entities, they resolve to identifiers, and a question with
nothing biomedical in it produces nothing to resolve. That check lives in
`scope.py`, where it works on identifiers instead of spellings.
"""

from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from .messages import RefusalReason


#: A research question, not a document. Long enough for a detailed one with
#: several named entities; short enough that a pasted corpus is refused before
#: it reaches a prompt.
MAX_QUESTION_CHARS = 600

#: Below this there is nothing to plan.
MIN_QUESTION_CHARS = 3

#: Characters with no place in a typed question and a long history of use in
#: smuggling: zero-width joiners and spaces, bidirectional overrides, the byte
#: order mark. They render as nothing, which is the point of them.
_INVISIBLE = {
    "​", "‌", "‍", "⁠", "﻿",
    "‪", "‫", "‬", "‭", "‮",
    "⁦", "⁧", "⁨", "⁩",
}

#: Control characters that survive normalisation. Newline and tab are kept —
#: people paste questions with line breaks — and everything else in the C0 and
#: C1 ranges is refused.
_ALLOWED_CONTROL = {"\n", "\t", "\r"}


@dataclass
class TextCheck:
    """The outcome of the deterministic pass."""

    ok: bool
    text: str = ""
    reason: Optional[RefusalReason] = None
    detail: str = ""
    notes: List[str] = field(default_factory=list)


def normalise(raw: str) -> Tuple[str, List[str]]:
    """Canonicalise the text and report what changed.

    NFKC folds the lookalike forms — full-width letters, ligatures, the various
    compatibility characters — so that later pattern matching sees one spelling
    rather than a dozen. The notes exist so the trace can record that a question
    arrived containing invisible characters even when it was otherwise fine;
    that is a signal worth having when reviewing traffic later.
    """
    notes: List[str] = []

    if any(ch in _INVISIBLE for ch in raw):
        notes.append("invisible or bidirectional characters were removed")
    stripped = "".join(ch for ch in raw if ch not in _INVISIBLE)

    normalised = unicodedata.normalize("NFKC", stripped)
    if normalised != stripped:
        notes.append("unicode was normalised (NFKC)")

    collapsed = re.sub(r"[ \t ]+", " ", normalised)
    collapsed = re.sub(r"\n{3,}", "\n\n", collapsed).strip()

    return collapsed, notes


def check_text(raw: str) -> TextCheck:
    """Length, shape and character sanity. No patterns, no models."""
    if not isinstance(raw, str):
        return TextCheck(
            ok=False, reason=RefusalReason.EMPTY_QUESTION,
            detail="question was not a string",
        )

    text, notes = normalise(raw)

    if len(text) < MIN_QUESTION_CHARS:
        return TextCheck(
            ok=False, reason=RefusalReason.EMPTY_QUESTION,
            detail=f"question is {len(text)} character(s) after normalisation",
            notes=notes,
        )

    if len(text) > MAX_QUESTION_CHARS:
        return TextCheck(
            ok=False, reason=RefusalReason.TOO_LONG,
            detail=f"{len(text)} characters, limit {MAX_QUESTION_CHARS}",
            notes=notes,
        )

    bad = sorted({
        ch for ch in text
        if unicodedata.category(ch) in {"Cc", "Cf", "Co", "Cs"}
        and ch not in _ALLOWED_CONTROL
    })
    if bad:
        return TextCheck(
            ok=False, reason=RefusalReason.MALFORMED_TEXT,
            detail=f"control characters present: {[hex(ord(c)) for c in bad]}",
            notes=notes,
        )

    return TextCheck(ok=True, text=text, notes=notes)


# ---------------------------------------------------------------------------
# Patterns
# ---------------------------------------------------------------------------
#
# High precision, low recall, by design. Each of these should read as obviously
# what it is; if a pattern needs an argument to justify, it belongs to the
# planner's judgement instead.


#: Someone asking about their own care. The shape is first person plus a
#: treatment decision — "should I take", "my dose" — not the topic. The same
#: subject asked as research ("which drugs treat X") is in scope and must
#: continue to be.
_CLINICAL_ADVICE = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\b(should|shall|can|could|must|ought)\s+i\b[^.?!]{0,60}"
        r"\b(take|use|stop|start|switch|try|continue|avoid|increase|reduce)\b",
        r"\bwhat\s+(dose|dosage|amount|strength)\s+(should|do|can)\s+i\b",
        r"\bhow\s+(much|many|often)\s+(should|do|can)\s+i\s+(take|use|inject|apply)\b",
        r"\bmy\s+(dose|dosage|prescription|medication|treatment plan|"
        r"diagnosis|symptoms?|condition|test results?)\b",
        r"\bi\s+(was\s+diagnosed\s+with|have\s+been\s+diagnosed)\b",
        r"\bis\s+it\s+safe\s+for\s+me\s+to\b",
        r"\b(diagnose|treat)\s+(me|my)\b",
        r"\bwhat\s+(should|do)\s+i\s+do\s+about\s+my\b",
    )
]

#: Acute first-person distress. Narrow on purpose: "suicidality as a drug
#: adverse event" is an ordinary pharmacovigilance research question and must
#: not be caught, so these require a first-person present-tense framing.
_EMERGENCY = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\bi\s+(am|'m)\s+(having|experiencing)\s+[^.?!]{0,40}"
        r"\b(chest pain|stroke|heart attack|seizure|anaphylaxis)\b",
        r"\bi\s+(just\s+)?(took|swallowed)\s+[^.?!]{0,40}\b(too many|overdose)\b",
        r"\bi\s+(can'?t|cannot)\s+breathe\b",
    )
]

#: Attempts to talk to the machinery rather than ask it something. These target
#: instruction override and prompt extraction — the two things a question has no
#: legitimate reason to contain.
_PROMPT_ATTACK = [
    re.compile(p, re.IGNORECASE) for p in (
        r"\b(ignore|disregard|forget|override)\b[^.?!]{0,30}"
        r"\b(previous|prior|above|earlier|preceding|all)\b[^.?!]{0,30}"
        r"\b(instruction|prompt|direction|rule|constraint|message)s?\b",
        r"\b(system|developer|initial|original)\s+prompt\b",
        r"\b(reveal|show|print|repeat|output|disclose)\b[^.?!]{0,30}"
        r"\b(your|the)\s+(prompt|instructions|system message|rules|configuration)\b",
        r"\bwhat\s+(are|were)\s+your\s+(instructions|rules|system prompt)\b",
        r"\b(bypass|circumvent|disable|turn off|get past)\b[^.?!]{0,30}"
        r"\b(safety|safeguard|guard|filter|check|validation|restriction)s?\b",
        r"\b(pretend|act)\s+(to\s+be|as\s+if|as\s+though)\b",
        r"\byou\s+are\s+now\s+(a|an|in)\b",
        r"\b(developer|debug|god)\s+mode\b",
        r"\bjailbreak\b",
        r"</?\s*(system|admin|instruction|im_start|im_end)\s*>",
        r"\b(begin|end)\s+(system|admin)\s+(prompt|instruction)",
        r"\buse\s+the\s+\w+\s+tool\b",
        r"\bcall\s+the\s+\w+\s+(function|tool|api)\b",
    )
]


def match_clinical_advice(text: str) -> Optional[str]:
    """The pattern that fired, or None. Returned for the trace, not the caller."""
    for pattern in _EMERGENCY:
        if pattern.search(text):
            return f"emergency:{pattern.pattern[:60]}"
    for pattern in _CLINICAL_ADVICE:
        if pattern.search(text):
            return f"clinical_advice:{pattern.pattern[:60]}"
    return None


def match_prompt_attack(text: str) -> Optional[str]:
    for pattern in _PROMPT_ATTACK:
        if pattern.search(text):
            return f"prompt_attack:{pattern.pattern[:60]}"
    return None
