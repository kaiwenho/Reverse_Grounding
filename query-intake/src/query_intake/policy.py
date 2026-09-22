"""
policy.py — Turning the checks into one decision.

Cheapest first, and stop at the first refusal. Text sanity costs nothing, the
pattern lists cost almost nothing, the classifier costs a model call — so a
question that is 40kB of pasted text never reaches a model, and a question that
is empty never reaches a pattern.

**Clinical-advice patterns run before the attack detector**, which looks odd
until you consider who is on the other end. A question can match both; the
attacker doesn't care which message they get, and the person asking about their
own medication does. Putting the useful, specific message first serves the
person who is genuinely confused about what this tool is for, at no cost to the
other case.

**A degraded detector does not block.** If the classifier could not load, the
question proceeds and the decision records `degraded=True`. The reasoning is in
`detectors.py`; the short version is that the loop behind this bounds the damage
an injected question can do, and a research service that stops answering because
a small classifier is unreachable has made things worse rather than better.

This module does not decide scope. Scope is about *which concepts* a question is
about, and that is answerable from identifiers rather than words — see
`scope.py`, which runs after the planner has named the entities.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from .checks import TextCheck, check_text, match_clinical_advice
from .detectors import AttackVerdict, PromptAttackDetector, default_detector
from .messages import RefusalReason


@dataclass
class IntakeDecision:
    """Allow or refuse, plus everything the trace should remember.

    ``detail`` and ``matched`` never reach the caller. They name the rule that
    fired, which is what an operator reviewing traffic needs and what an
    attacker probing the filter would love to have.
    """

    allowed: bool
    question: str = ""
    reason: Optional[RefusalReason] = None
    detail: str = ""
    matched: Optional[str] = None
    notes: List[str] = field(default_factory=list)
    attack: Optional[AttackVerdict] = None
    degraded: bool = False

    def to_dict(self) -> dict:
        return {
            "allowed": self.allowed,
            "reason": self.reason.value if self.reason else None,
            "detail": self.detail,
            "matched": self.matched,
            "notes": self.notes,
            "degraded_detection": self.degraded,
            "attack": self.attack.to_dict() if self.attack else None,
        }


class IntakePolicy:
    """Runs the deterministic checks, then whatever detector is configured."""

    def __init__(
        self,
        detector: Optional[PromptAttackDetector] = None,
        *,
        screen_clinical_advice: bool = True,
    ) -> None:
        self.detector = detector or default_detector()
        self.screen_clinical_advice = screen_clinical_advice

    def evaluate(self, raw: str) -> IntakeDecision:
        text_check: TextCheck = check_text(raw)
        if not text_check.ok:
            return IntakeDecision(
                allowed=False,
                reason=text_check.reason,
                detail=text_check.detail,
                notes=text_check.notes,
            )

        question = text_check.text
        notes = list(text_check.notes)

        if self.screen_clinical_advice:
            hit = match_clinical_advice(question)
            if hit:
                return IntakeDecision(
                    allowed=False,
                    question=question,
                    reason=RefusalReason.CLINICAL_ADVICE,
                    detail="matched a first-person care-decision pattern",
                    matched=hit,
                    notes=notes,
                )

        verdict = self.detector.inspect(question)
        if verdict.malicious:
            return IntakeDecision(
                allowed=False,
                question=question,
                reason=RefusalReason.PROMPT_ATTACK,
                detail=f"flagged by {verdict.detector}",
                matched=verdict.evidence,
                notes=notes,
                attack=verdict,
            )

        if not verdict.available:
            notes.append(
                "attack detection was unavailable; the request proceeded on "
                "the deterministic checks alone"
            )

        # Everything else goes through. The planner has its own refusal for a
        # question it cannot ground, and it can only emit a Biolink-valid plan
        # in any case — so an unclear question costs one planner call, and
        # over-blocking costs a research question that should have been
        # answered.
        return IntakeDecision(
            allowed=True,
            question=question,
            notes=notes,
            attack=verdict,
            degraded=not verdict.available,
        )
