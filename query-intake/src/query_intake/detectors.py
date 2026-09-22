"""
detectors.py — Pluggable prompt-attack detection.

One interface, three implementations, and a stated policy about what happens
when a detector is not there.

**The rule-based detector always works.** It has no dependencies and no network,
so there is always a floor. It catches clumsy attacks and nothing else, which is
an honest description of what regular expressions can do here.

**Prompt Guard is optional.** Meta's classifier is a better detector than a
pattern list, and its own model card is candid that adaptive attackers get past
it and that performance varies by domain. It is worth having and it is not worth
depending on.

**A missing detector fails open, loudly.** If the model cannot be loaded, the
request proceeds, the result records that detection was degraded, and the trace
says so. That is a deliberate trade. Failing closed would mean an 86M classifier
being unreachable takes down a research service — converting a small,
well-bounded risk into a total outage. The bound is real: an injected question
can at worst produce a Biolink-valid plan for a different biomedical question,
answered entirely from graph edges, with no model-written text reaching the
caller. If that bound ever weakens, revisit this decision first.

Fail-open applies only to the *detectors*. The deterministic checks in
`checks.py` have nothing to fail, and they always run.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Protocol


@dataclass
class AttackVerdict:
    """What a detector concluded, and whether it was able to conclude anything.

    ``available`` is separate from ``malicious`` on purpose. "Not an attack" and
    "could not tell" are different facts, and collapsing them is how a service
    ends up reporting clean traffic while its classifier has been down for a
    week.
    """

    malicious: bool = False
    available: bool = True
    detector: str = "none"
    score: Optional[float] = None
    evidence: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "detector": self.detector,
            "malicious": self.malicious,
            "available": self.available,
            "score": self.score,
            "evidence": self.evidence,
            "error": self.error,
        }


class PromptAttackDetector(Protocol):
    """Classify one question as an attack or not."""

    name: str

    def inspect(self, text: str) -> AttackVerdict: ...


# ---------------------------------------------------------------------------
# Rule-based
# ---------------------------------------------------------------------------


class RuleBasedDetector:
    """Pattern matching. Always available, catches only the obvious.

    Its value is not its hit rate — it is that there is never a configuration
    in which nothing is checked.
    """

    name = "rules"

    def inspect(self, text: str) -> AttackVerdict:
        from .checks import match_prompt_attack

        hit = match_prompt_attack(text)
        return AttackVerdict(
            malicious=hit is not None,
            available=True,
            detector=self.name,
            evidence=hit,
        )


class NullDetector:
    """Checks nothing and says so.

    For deployments that want only the deterministic checks. It reports
    ``available=False`` rather than a clean verdict, so "detection is switched
    off" shows up in the trace instead of looking like "nothing was found".
    """

    name = "null"

    def inspect(self, text: str) -> AttackVerdict:
        return AttackVerdict(
            malicious=False, available=False, detector=self.name,
            error="no attack detector configured",
        )


# ---------------------------------------------------------------------------
# Prompt Guard
# ---------------------------------------------------------------------------


class PromptGuardDetector:
    """Meta's Llama Prompt Guard 2, loaded lazily.

    Note which model this is. Llama Guard is a *content safety* classifier over
    hazard categories and does not detect injection; Prompt Guard is the
    injection and jailbreak classifier, and at 86M parameters it is cheap enough
    to run on every request. Confusing the two is easy and gives you a guard
    that does not guard the thing you meant.

    Loaded on first use rather than at construction, so importing this module
    costs nothing and a service that never receives a request never pays for the
    weights. A load failure is recorded once and every later call short-circuits
    to an unavailable verdict rather than retrying an import that will fail
    again.
    """

    name = "prompt-guard-2"

    def __init__(
        self,
        model_id: str = "meta-llama/Llama-Prompt-Guard-2-86M",
        threshold: float = 0.9,
        device: str = "cpu",
    ) -> None:
        self.model_id = model_id
        self.threshold = threshold
        self.device = device
        self._pipeline = None
        self._load_error: Optional[str] = None

    def _load(self):
        if self._pipeline is not None or self._load_error is not None:
            return self._pipeline
        try:
            from transformers import pipeline  # type: ignore

            self._pipeline = pipeline(
                "text-classification",
                model=self.model_id,
                device=self.device,
                truncation=True,
            )
        except Exception as exc:
            self._load_error = f"{type(exc).__name__}: {exc}"
        return self._pipeline

    def inspect(self, text: str) -> AttackVerdict:
        classifier = self._load()
        if classifier is None:
            return AttackVerdict(
                malicious=False, available=False, detector=self.name,
                error=self._load_error or "model unavailable",
            )
        try:
            top = classifier(text)[0]
        except Exception as exc:
            return AttackVerdict(
                malicious=False, available=False, detector=self.name,
                error=f"{type(exc).__name__}: {exc}",
            )

        label = str(top.get("label", "")).upper()
        score = float(top.get("score", 0.0))
        malicious = label in {"MALICIOUS", "LABEL_1", "INJECTION", "JAILBREAK"}
        return AttackVerdict(
            malicious=malicious and score >= self.threshold,
            available=True,
            detector=self.name,
            score=score,
            evidence=f"{label} ({score:.3f})" if malicious else None,
        )


# ---------------------------------------------------------------------------
# Composite
# ---------------------------------------------------------------------------


class CompositeDetector:
    """Run several detectors; any hit blocks.

    Keeps every verdict rather than the first hit, because the useful operating
    question later is not "was this blocked" but "which detectors agree" — a
    pattern firing while the model says benign, repeatedly, means the pattern is
    too broad.
    """

    name = "composite"

    def __init__(self, detectors: List[PromptAttackDetector]) -> None:
        self.detectors = list(detectors)
        self.last_verdicts: List[AttackVerdict] = []

    def inspect(self, text: str) -> AttackVerdict:
        verdicts = [d.inspect(text) for d in self.detectors]
        self.last_verdicts = verdicts

        hit = next((v for v in verdicts if v.malicious), None)
        if hit is not None:
            return AttackVerdict(
                malicious=True,
                available=True,
                detector=f"{self.name}:{hit.detector}",
                score=hit.score,
                evidence=hit.evidence,
            )

        any_available = any(v.available for v in verdicts)
        errors = "; ".join(v.error for v in verdicts if v.error) or None
        return AttackVerdict(
            malicious=False,
            available=any_available,
            detector=self.name,
            error=errors,
        )


def default_detector(use_prompt_guard: bool = False) -> PromptAttackDetector:
    """The rule-based floor, optionally with Prompt Guard above it."""
    detectors: List[PromptAttackDetector] = [RuleBasedDetector()]
    if use_prompt_guard:
        detectors.append(PromptGuardDetector())
    return CompositeDetector(detectors)
