"""
path_quality.py — Which explanation paths are worth showing, and why not.

An explanation path connects a candidate to the concept the question is about.
Every edge in one is real, so the grounding gate passes it: the identifiers
occur in the result graph, nothing is invented, nothing is model-written. That
is necessary and it is not sufficient. A chain of true edges can still say
nothing, and it can say the opposite of what it is being used to support.

Both were found in one live run — 25 candidates, 12,113 paths found, 125 kept,
1,092 seconds:

    Resveratrol —[subclass_of]→ phenols —[subclass_of]→ Phenylephrine
                —[contraindicated_in]→ dermatitis herpetiformis

    Nimodipine —[correlated_with]→ hypotension
               —[treats_or_applied_or_studied_to_treat]→ Phenylephrine
               —[contraindicated_in]→ dermatitis herpetiformis

Read them as sentences. The first says Resveratrol is a phenol, and some other
phenol must not be given to patients with this disease. The second says
Nimodipine can cause low blood pressure, that a drug for low blood pressure
exists, and that *that* drug is contraindicated here. Neither is a reason to
try the candidate. The second is closer to a reason not to.

41 of the 125 kept paths ended in `contraindicated_in`, and 19 more passed
through `has_side_effect`.

Two rules, both structural, both deterministic. This module reads predicates
and negation flags; it does not judge biology, and it is not a substitute for
the EPC-confidence and knowledge-level thresholds that rank what survives.

**Direction.** A path that reaches the destination through a relation meaning
"must not be used", "causes", or "is a symptom of" is not support. It may be
worth seeing — a contraindication is a real finding — but not filed under
evidence that something helps.

**Mechanism.** A path built only from taxonomic and statistical links —
`subclass_of`, `correlated_with`, `related_to` — connects two things without
explaining anything. In a graph this dense, almost any pair of nodes is
reachable that way, which is why 12,113 paths existed to be narrowed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


#: Relations that carry an explicit negative or non-therapeutic meaning.
#: Arriving at the question's concept through one of these is not support for
#: the candidate, whatever the rest of the chain says.
CONTRARY_PREDICATES: frozenset = frozenset({
    "biolink:contraindicated_in",
    "biolink:contraindicated_for",
    "biolink:has_adverse_event",
    "biolink:causes_adverse_event",
    "biolink:has_side_effect",
    "biolink:causes",
    "biolink:contributes_to",
    "biolink:exacerbates",
    "biolink:exacerbates_condition",
    "biolink:has_phenotype",
})

#: Relations that place two things in the same class or note that they vary
#: together, without asserting that one does anything to the other. Real
#: edges, and no mechanism.
NON_MECHANISTIC_PREDICATES: frozenset = frozenset({
    "biolink:subclass_of",
    "biolink:superclass_of",
    "biolink:related_to",
    "biolink:related_to_at_concept_level",
    "biolink:correlated_with",
    "biolink:positively_correlated_with",
    "biolink:negatively_correlated_with",
    "biolink:associated_with",
    "biolink:coexists_with",
    "biolink:close_match",
    "biolink:same_as",
    "biolink:broad_match",
    "biolink:narrow_match",
    "biolink:has_member",
    "biolink:member_of",
})

VERDICT_OK = "ok"
VERDICT_CONTRARY = "contrary_direction"
VERDICT_NO_MECHANISM = "no_mechanism"
VERDICT_NEGATED = "negated_edge"
VERDICT_MALFORMED = "malformed"


@dataclass
class PathVerdict:
    """Why a path may or may not be offered as support."""

    verdict: str = VERDICT_OK
    detail: str = ""
    #: The predicate that decided it, when one did.
    predicate: Optional[str] = None

    @property
    def servable(self) -> bool:
        return self.verdict == VERDICT_OK

    def to_dict(self) -> Dict[str, Any]:
        return {
            "verdict": self.verdict,
            "detail": self.detail,
            "predicate": self.predicate,
        }


def judge_path(path: Dict[str, Any]) -> PathVerdict:
    """Classify one explanation path.

    Order matters only in what gets reported, since a path can fail more than
    one way. Negation first because it is the flattest contradiction, then
    direction, then mechanism — most specific reason first, so the count of
    each kind means something.
    """
    predicates = [str(p) for p in (path.get("predicates") or [])]
    edges = path.get("edges") or []

    if not predicates:
        return PathVerdict(VERDICT_MALFORMED, "the path records no predicates")

    for edge in edges:
        if isinstance(edge, dict) and edge.get("negated"):
            return PathVerdict(
                VERDICT_NEGATED,
                f"an edge in the chain is negated "
                f"({edge.get('predicate')}), so the chain asserts the "
                f"relationship does not hold",
                edge.get("predicate"),
            )

    # The last predicate is the one that reaches the concept the question is
    # about. That is the claim the path is being used to make.
    final = predicates[-1]
    if final in CONTRARY_PREDICATES:
        return PathVerdict(
            VERDICT_CONTRARY,
            f"the chain reaches the destination through '{final}', which is "
            f"not a claim that the candidate helps",
            final,
        )

    # Contrary relations earlier in the chain are not disqualifying on their
    # own — "drug causes X, X is treated by Y" is a real, if weak, route — but
    # they are worth surfacing, so they are reported rather than dropped.

    if all(p in NON_MECHANISTIC_PREDICATES for p in predicates):
        return PathVerdict(
            VERDICT_NO_MECHANISM,
            f"every link is taxonomic or statistical "
            f"({', '.join(sorted(set(predicates)))}), so the chain connects "
            f"without explaining",
            predicates[0],
        )

    return PathVerdict()


@dataclass
class PathReview:
    """What a set of explanation paths was allowed to say."""

    servable: List[Dict[str, Any]] = field(default_factory=list)
    withheld: List[Dict[str, Any]] = field(default_factory=list)

    @property
    def counts(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for item in self.withheld:
            kind = item["verdict"]["verdict"]
            out[kind] = out.get(kind, 0) + 1
        return out

    def to_dict(self) -> Dict[str, Any]:
        return {
            "servable": len(self.servable),
            "withheld": len(self.withheld),
            "withheld_by_kind": self.counts,
            "examples": [
                {
                    "path": w["path"].get("description")
                           or " → ".join(w["path"].get("labels") or []),
                    "why": w["verdict"]["detail"],
                }
                for w in self.withheld[:5]
            ],
        }


def review_paths(paths: Sequence[Dict[str, Any]]) -> PathReview:
    """Split explanation paths into what may be offered and what may not."""
    review = PathReview()
    for path in paths or []:
        verdict = judge_path(path)
        entry = {"path": path, "verdict": verdict.to_dict()}
        if verdict.servable:
            review.servable.append(path)
        else:
            review.withheld.append(entry)
    return review


def collect_explanation_paths(result: Dict[str, Any]) -> Dict[str, List[Dict[str, Any]]]:
    """Every explanation path in the result, keyed by where it leads.

    The executor writes these in one of two places depending on the plan.
    With `aggregation.attach_explanations_to_candidates: false` they sit under
    `evidence.explanations.paths_by_endpoint`; with it true they are attached
    to each candidate as `explanation_paths` and the endpoint map is left
    empty.

    The composer only ever read the first, so a hybrid plan that attached its
    explanations produced an answer with no explanation section at all — in one
    live run, after 1,092 seconds of pathfinding. Nothing reported the silence,
    because an empty endpoint map is also what a run with no explanations looks
    like.
    """
    out: Dict[str, List[Dict[str, Any]]] = {}

    explanations = (result.get("evidence") or {}).get("explanations") or {}
    for endpoint, paths in (explanations.get("paths_by_endpoint") or {}).items():
        if paths:
            out.setdefault(str(endpoint), []).extend(paths)

    for candidate in result.get("results") or []:
        paths = candidate.get("explanation_paths") or []
        if not paths:
            continue
        key = str(candidate.get("curie") or candidate.get("label") or "candidate")
        out.setdefault(key, []).extend(paths)

    return out
