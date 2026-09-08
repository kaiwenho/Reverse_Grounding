"""
rank.py — Order candidates by the plan's own ranking criteria.

Three kinds of signal, deliberately kept apart:

  * `plan.ranking.criteria` — the planner's decision about what matters for
    this question, with its own weights. Authoritative.
  * EPC score — the executor's convention for how well evidenced an edge is.
    Enters ranking only through the `edge_evidence_strength` criterion, so a
    plan that never asks for it never gets it.
  * Literature support — a separate signal, only present for candidates the
    verification budget reached. Never folded into EPC, because two edges with
    identical metadata would otherwise score differently according to whether
    they happened to be checked.

Unknown is not zero
-------------------
A candidate whose literature was never verified has `None`, not 0.0, and is
excluded from that criterion's normalization rather than placed at the bottom
of it. Treating unchecked as unsupported would rank candidates by whether the
budget reached them.

Normalization
-------------
Criteria arrive on incompatible scales — path counts, hop counts, scores in
[0, 1]. Each is min-max normalized across the candidate pool before weighting,
and `direction: asc` inverts it so that lower is better. Normalizing within
the pool means scores are comparative, not absolute: a candidate's 0.8 says it
ranks well among these candidates, not that it is good in isolation.

Grounded reranking
------------------
An LLM may reorder the top-K after reading evidence, but every reason it gives
must cite edge ids drawn from the evidence it was shown, and those citations
are checked against the candidate's own supporting edges. A reranking whose
citations do not check out is discarded for that candidate and the
deterministic position stands. This is the same principle as quote
verification in evidence.py: the model points at something confirmable rather
than being taken at its word.

Usage
-----
    ranker = CandidateRanker(plan.ranking, reranker=llm_reranker)
    ranked = ranker.rank(candidates, question=plan.question)
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Protocol, Sequence, Set, Tuple


# ---------------------------------------------------------------------------
# Criterion vocabulary
# ---------------------------------------------------------------------------

CRITERION_NAMES = {
    "path_priority_order", "num_supporting_paths", "num_publications",
    "num_knowledge_sources", "knowledge_level", "edge_evidence_strength",
    "path_length", "clinical_trial_phase", "approval_status",
    "genetic_support", "recency", "num_explanation_paths",
    "distinct_intermediate_categories",
}

#: Criteria that cannot be computed before explanations exist. A discovery
#: ranking naming one would score every candidate identically, so the plan
#: contract forbids it there.
EXPLANATION_DERIVED_CRITERIA = {
    "num_explanation_paths", "distinct_intermediate_categories",
}

#: Lowest score a value named in `preferred_values` can take, and the score
#: for one that is not named. Kept apart so that everything the plan preferred
#: outranks everything it did not, while an unnamed value — including
#: `not_provided` — still scores above zero and is retained.
PREFERRED_FLOOR = 0.4
UNLISTED_VALUE_SCORE = 0.2

INFLUENCE_NONE = "none"
INFLUENCE_ANNOTATE = "annotate_only"
INFLUENCE_RERANK = "rerank"

#: Criteria where a smaller value is better, unless the plan says otherwise.
NATURALLY_ASCENDING = {"path_length", "path_priority_order"}

STRATEGY_DEFAULT_CRITERIA = {
    "path_priority": [("path_priority_order", "asc", 3.0),
                      ("edge_evidence_strength", "desc", 1.0)],
    "evidence_weighted": [("edge_evidence_strength", "desc", 3.0),
                          ("num_publications", "desc", 1.0),
                          ("num_knowledge_sources", "desc", 1.0)],
    "shortest_path_first": [("path_length", "asc", 3.0),
                            ("edge_evidence_strength", "desc", 1.0)],
    "multi_path_consensus": [("num_supporting_paths", "desc", 2.0),
                             ("distinct_intermediate_categories", "desc", 1.0),
                             ("edge_evidence_strength", "desc", 1.0),
                             ("path_length", "asc", 0.5)],
    "genetic_evidence_boosted": [("genetic_support", "desc", 3.0),
                                 ("num_supporting_paths", "desc", 1.0),
                                 ("edge_evidence_strength", "desc", 1.0)],
    "explanation_diversity": [("num_explanation_paths", "desc", 2.0),
                              ("distinct_intermediate_categories", "desc", 2.0),
                              ("edge_evidence_strength", "desc", 1.0)],
}

#: Weight given to verified literature support when the plan does not name a
#: criterion for it. Modest by design: it is a warrant that a reader can
#: follow, not a measure of how well studied a claim is.
DEFAULT_LITERATURE_WEIGHT = 1.0


# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------


@dataclass
class Criterion:
    """One ranking criterion as the plan states it.

    `preferred_values` carries an ordered vocabulary — knowledge levels, say —
    so a portable criterion can be scored anywhere without the consumer
    needing the executor's own conventions. `scope` marks the exception:
    `edge_evidence_strength` is `executor_specific`, meaning the plan selects
    and weights it but the formula stays in executor configuration.
    """

    name: str
    direction: str = "desc"
    weight: float = 1.0
    origin: Optional[str] = None
    scope: Optional[str] = None
    profile_id: Optional[str] = None
    preferred_values: List[str] = field(default_factory=list)
    rationale: Optional[str] = None

    def value_score(self, value: Any) -> Optional[float]:
        """Score a categorical value against `preferred_values`.

        Listed values are ranked by position and span [PREFERRED_FLOOR, 1.0];
        an unlisted value scores below all of them but above zero.

        Both halves of that matter. Scoring unlisted at zero would make
        missing metadata disqualifying, which the contract forbids — absent
        annotation is unknown quality, not poor quality. But scoring it at the
        midpoint would put it above the plan's own lowest preference, so a
        value the plan explicitly ranked last would lose to one it never
        mentioned.
        """
        if not self.preferred_values:
            return None
        try:
            rank = self.preferred_values.index(value)
        except (ValueError, TypeError):
            return UNLISTED_VALUE_SCORE
        if len(self.preferred_values) == 1:
            return 1.0
        span = 1.0 - PREFERRED_FLOOR
        return 1.0 - span * rank / (len(self.preferred_values) - 1)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name, "direction": self.direction,
            "weight": self.weight, "origin": self.origin, "scope": self.scope,
            "preferred_values": self.preferred_values or None,
            "rationale": self.rationale,
        }


@dataclass
class Candidate:
    """One answer, with everything that reached it across all paths."""

    curie: str
    label: str = ""
    categories: List[str] = field(default_factory=list)

    path_ids: Set[str] = field(default_factory=set)
    instances: List[Any] = field(default_factory=list)
    edge_ids: Set[str] = field(default_factory=set)
    intermediate_curies: Set[str] = field(default_factory=set)
    intermediate_categories: Set[str] = field(default_factory=set)

    epc_scores: List[float] = field(default_factory=list)
    knowledge_levels: List[str] = field(default_factory=list)
    literature_scores: List[float] = field(default_factory=list)
    literature: Dict[str, Any] = field(default_factory=dict)
    explanation_path_count: int = 0
    path_priority: Optional[int] = None
    node_attributes: Dict[str, Any] = field(default_factory=dict)

    metrics: Dict[str, float] = field(default_factory=dict)
    normalized: Dict[str, float] = field(default_factory=dict)
    score: float = 0.0
    rank: int = 0
    deterministic_rank: int = 0
    rerank_reason: Optional[str] = None
    rerank_grounded: Optional[bool] = None
    notes: List[str] = field(default_factory=list)

    @property
    def num_supporting_paths(self) -> int:
        return len(self.path_ids)

    @property
    def mean_epc(self) -> Optional[float]:
        return sum(self.epc_scores) / len(self.epc_scores) if self.epc_scores else None

    @property
    def literature_score(self) -> Optional[float]:
        """Mean verified literature support, or None when nothing was checked."""
        known = [s for s in self.literature_scores if s is not None]
        return sum(known) / len(known) if known else None

    @property
    def best_citation(self) -> Optional[Dict[str, Any]]:
        """The strongest checkable citation, for showing a reader why to believe it."""
        best = None
        for support in self.literature.values():
            cite = support.get("best_citation") if isinstance(support, dict) else None
            if cite and (best is None or (cite.get("year") or 0) > (best.get("year") or 0)):
                best = cite
        return best

    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "curie": self.curie,
            "label": self.label,
            "categories": self.categories[:3],
            "score": round(self.score, 4),
            "supporting_paths": sorted(self.path_ids),
            "num_supporting_paths": self.num_supporting_paths,
            "num_instances": len(self.instances),
            "metrics": {k: round(v, 3) for k, v in self.metrics.items()},
            "normalized": {k: round(v, 3) for k, v in self.normalized.items()},
            "mean_epc": round(self.mean_epc, 3) if self.mean_epc is not None else None,
            "literature_score": (
                round(self.literature_score, 3)
                if self.literature_score is not None else None
            ),
            "best_citation": self.best_citation,
            "deterministic_rank": self.deterministic_rank,
            "rerank_reason": self.rerank_reason,
            "rerank_grounded": self.rerank_grounded,
            "notes": self.notes,
        }

    def __repr__(self) -> str:
        return (f"<Candidate #{self.rank} {self.curie} ({self.label}) "
                f"score={self.score:.3f} paths={self.num_supporting_paths}>")


#: Words that name a source rather than describe evidence. A reason built
#: only from these restates provenance, which EPC scoring already weighs
#: deterministically — the point of asking a model to read is to get something
#: metadata cannot supply.
_PROVENANCE_WORDS = {
    "chembl", "drugcentral", "gtopdb", "dgidb", "ttd", "signor", "drugbank",
    "semmeddb", "rtx", "kg2", "arax", "infores", "database", "databases",
    "source", "sources", "curated", "knowledge", "assertion", "approved",
    "supported", "evidence", "confidence", "multiple", "several", "high",
}

#: Minimum length for a word to count as substantive when matched against a
#: verified quote. Short words are too common to indicate that the reason
#: actually drew on the quoted text.
_MIN_CONTENT_WORD = 5


def _content_tokens(
    candidate: "Candidate",
    cited_edge_ids: Set[str],
    edges: Dict[str, Any],
) -> Dict[str, Set[str]]:
    """Terms a reason could legitimately be grounded in, by kind.

    Quotes are the strongest: they exist only because an abstract was read.
    Failing a quote, the assertion itself counts — the predicate and its
    qualifiers, which say what the edge claims — because most edges have no
    verified literature and would otherwise be unrankable.

    Knowledge level, agent type and publication counts are deliberately not
    accepted. They describe how well annotated a record is rather than what it
    claims, and they are already scored numerically before the model sees
    them. Allowing a reason to rest on them would readmit exactly the
    metadata restatement this check exists to reject.
    """
    quote_words: Set[str] = set()
    structure: Set[str] = set()

    for eid in cited_edge_ids:
        edge = edges.get(eid) or {}

        predicate = (edge.get("predicate") or "").replace("biolink:", "")
        for part in predicate.replace("_", " ").split():
            if len(part) >= 4:
                structure.add(part.lower())

        for constraint in edge.get("qualifiers") or []:
            if isinstance(constraint, dict):
                value = constraint.get("qualifier_value") or ""
                if not isinstance(value, str) or not value:
                    continue
                # Identifiers are not descriptive content. A species context
                # of NCBITaxon:9606 is retrieved data, but no reason would
                # reference it by CURIE, so offering it as something to ground
                # in only clutters the diagnostic.
                if ":" in value and not value.startswith("biolink:"):
                    continue
                for part in value.replace("biolink:", "").replace("_", " ").split():
                    if len(part) >= 4 and not any(ch.isdigit() for ch in part):
                        structure.add(part.lower())

        support = candidate.literature.get(eid)
        if isinstance(support, dict):
            for verdict in support.get("verdicts") or []:
                if not verdict.get("quote_verified"):
                    continue
                for word in (verdict.get("quote") or "").lower().split():
                    word = "".join(ch for ch in word if ch.isalpha())
                    if len(word) >= _MIN_CONTENT_WORD:
                        quote_words.add(word)

    return {
        "quote": quote_words - _PROVENANCE_WORDS,
        "structure": structure - _PROVENANCE_WORDS,
    }


class CandidateReranker(Protocol):
    """Reorders the top-K after reading evidence. Implemented by llm.py."""

    def rerank(
        self,
        question: str,
        candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Returns [{'curie', 'rank', 'reason', 'cited_edge_ids'}, ...].

        Cited edge ids must come from the evidence supplied for that
        candidate; they are checked.
        """
        ...


# ---------------------------------------------------------------------------
# Assembling candidates
# ---------------------------------------------------------------------------


def collect_candidates(
    executions: Dict[str, Any],
    path_priority: Optional[Dict[str, int]] = None,
    epc_by_edge: Optional[Dict[str, float]] = None,
    literature_by_edge: Optional[Dict[str, Any]] = None,
) -> Dict[str, Candidate]:
    """Merge per-path executions into one candidate set.

    This is the cross-path join: a drug found by both P1 and P2 becomes one
    candidate with two supporting paths, which is what
    `num_supporting_paths` measures. Doing it here rather than inside the
    executor keeps each path's result independently inspectable.
    """
    path_priority = path_priority or {}
    epc_by_edge = epc_by_edge or {}
    literature_by_edge = literature_by_edge or {}
    out: Dict[str, Candidate] = {}

    for path_id, execution in executions.items():
        return_ref = getattr(execution, "return_entity_ref", None)
        nodes = getattr(execution, "kg_nodes", {}) or {}
        priority = path_priority.get(path_id)

        for inst in getattr(execution, "instances", []) or []:
            curie = getattr(inst, "return_curie", None) or inst.bindings.get(return_ref)
            if not curie:
                continue

            node = nodes.get(curie) or {}
            cand = out.get(curie)
            if cand is None:
                cand = Candidate(
                    curie=curie,
                    label=node.get("name") or curie,
                    categories=[
                        c.replace("biolink:", "") for c in (node.get("categories") or [])
                    ],
                    node_attributes=node,
                )
                out[curie] = cand

            cand.path_ids.add(path_id)
            cand.instances.append(inst)
            if priority is not None:
                cand.path_priority = (
                    priority if cand.path_priority is None
                    else min(cand.path_priority, priority)
                )

            for eid in getattr(inst, "edge_ids", []) or []:
                if eid in cand.edge_ids:
                    continue
                cand.edge_ids.add(eid)
                if eid in epc_by_edge:
                    cand.epc_scores.append(epc_by_edge[eid])
                edge = (getattr(execution, "kg_edges", {}) or {}).get(eid)
                if edge is not None:
                    from .postfilter import edge_knowledge_level
                    cand.knowledge_levels.append(edge_knowledge_level(edge))
                support = literature_by_edge.get(eid)
                if support is not None:
                    score = (
                        support.get("score") if isinstance(support, dict)
                        else getattr(support, "score", None)
                    )
                    cand.literature_scores.append(score)
                    cand.literature[eid] = (
                        support if isinstance(support, dict) else support.to_dict()
                    )

            # Intermediates are every bound node other than the answer itself;
            # their variety is what `distinct_intermediate_categories` rewards,
            # since agreement across mechanisms is stronger than repetition of
            # one.
            for ref, bound in (getattr(inst, "bindings", {}) or {}).items():
                if bound == curie:
                    continue
                cand.intermediate_curies.add(bound)
                inode = nodes.get(bound) or {}
                for cat in inode.get("categories") or []:
                    cand.intermediate_categories.add(cat.replace("biolink:", ""))

    return out


def compute_metrics(candidate: Candidate) -> Dict[str, float]:
    """Compute every criterion the schema allows, from what was gathered.

    Criteria the data cannot support are left absent rather than filled with
    zero, so they are skipped in normalization instead of silently ranking
    every candidate equally badly.
    """
    metrics: Dict[str, float] = {
        "num_supporting_paths": float(len(candidate.path_ids)),
        "distinct_intermediate_categories": float(len(candidate.intermediate_categories)),
        "num_explanation_paths": float(candidate.explanation_path_count),
    }

    if candidate.instances:
        lengths = [
            len(getattr(i, "edge_ids", []) or []) or len(getattr(i, "bindings", {})) - 1
            for i in candidate.instances
        ]
        lengths = [x for x in lengths if x > 0]
        if lengths:
            metrics["path_length"] = float(min(lengths))

    if candidate.epc_scores:
        metrics["edge_evidence_strength"] = float(candidate.mean_epc)

    if candidate.path_priority is not None:
        metrics["path_priority_order"] = float(candidate.path_priority)

    pubs, sources, years = 0, set(), []
    for support in candidate.literature.values():
        if not isinstance(support, dict):
            continue
        pubs += support.get("num_supporting", 0)
        if support.get("primary_source"):
            sources.add(support["primary_source"])
        cite = support.get("best_citation") or {}
        if cite.get("year"):
            years.append(cite["year"])
    if pubs:
        metrics["num_publications"] = float(pubs)
    if sources:
        metrics["num_knowledge_sources"] = float(len(sources))
    if years:
        metrics["recency"] = float(max(years))

    for field_name, key in (
        ("approval_status", "approval_status"),
        ("clinical_trial_phase", "max_phase"),
        ("genetic_support", "genetic_support"),
    ):
        value = _node_numeric(candidate.node_attributes, key)
        if value is not None:
            metrics[field_name] = value

    return metrics


def _node_numeric(node: Dict[str, Any], key: str) -> Optional[float]:
    """Read a numeric or boolean-ish node attribute, if present."""
    def coerce(v):
        if isinstance(v, bool):
            return 1.0 if v else 0.0
        if isinstance(v, (int, float)):
            return float(v)
        text = str(v).strip().lower()
        if text in ("approved", "true", "yes"):
            return 1.0
        if text in ("investigational", "false", "no", "withdrawn"):
            return 0.0
        try:
            return float(text)
        except ValueError:
            return None

    if key in node:
        return coerce(node[key])
    for attr in node.get("attributes") or []:
        if not isinstance(attr, dict):
            continue
        tid = attr.get("attribute_type_id") or ""
        if tid.split(":")[-1] == key:
            return coerce(attr.get("value"))
    return None


# ---------------------------------------------------------------------------
# Ranker
# ---------------------------------------------------------------------------


class CandidateRanker:
    """Scores and orders candidates by the plan's ranking block."""

    def __init__(
        self,
        ranking: Optional[Any] = None,
        reranker: Optional[CandidateReranker] = None,
        literature_weight: float = DEFAULT_LITERATURE_WEIGHT,
        rerank_top_k: int = 20,
        require_reranker: bool = False,
        quiet_warnings: bool = False,
        stage: str = "discovery",
        seed: int = 0,
        verbose: bool = True,
    ):
        """Rank candidates for one stage of a plan.

        Args:
            ranking: a RankingSpec — `{strategy, criteria, top_k, tie_breaker}`.
                Callers pass the block for the stage they are running rather
                than the whole ranking plan, which `RankingPlan` splits.
            stage: `discovery`, `final` or `explanation`. A discovery ranking
                cannot use explanation-derived criteria, because those values
                do not exist until explanations have run.
        """
        raw = ranking or {}
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(exclude_none=True)
        elif not isinstance(raw, dict) and hasattr(raw, "__dict__"):
            raw = {k: v for k, v in vars(raw).items() if v is not None}

        # The pre-0.8 shape put strategy and criteria at the top level of
        # `ranking`. Accepting it silently would rank by executor defaults
        # while appearing to honour the plan, so it is refused outright.
        if "candidate_ranking" in raw or "explanation_ranking" in raw:
            raise ValueError(
                "CandidateRanker takes one RankingSpec, not the whole ranking "
                "plan. Use RankingPlan.from_plan(...) and pass .discovery, "
                ".final or .explanation."
            )

        self.stage = stage
        self.warnings: List[str] = []
        self.strategy = raw.get("strategy") or "multi_path_consensus"
        self.top_k = raw.get("top_k")
        self.tie_breaker = raw.get("tie_breaker") or "path_priority_order"
        self.criteria = self._compile_criteria(raw.get("criteria"))
        self.reranker = reranker
        self.literature_weight = literature_weight
        self.rerank_top_k = rerank_top_k
        self.require_reranker = require_reranker
        # The preview pass runs before literature exists, so its complaints
        # about missing publication data describe the pass rather than the
        # plan. Reported only on the pass whose result is kept.
        self.quiet_warnings = quiet_warnings
        self.rng = random.Random(seed)
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [rank] {msg}")

    def _compile_criteria(self, criteria) -> List["Criterion"]:
        """Normalize the plan's criteria, falling back to the strategy default.

        A strategy with no criteria is not an error: the strategy name is
        itself a statement of intent, and each has a documented default here
        rather than silently ranking on nothing.
        """
        out: List[Tuple[str, str, float]] = []
        for c in criteria or []:
            if hasattr(c, "model_dump"):
                c = c.model_dump()
            elif not isinstance(c, dict):
                c = vars(c)
            name = c.get("name")
            if name not in CRITERION_NAMES:
                self.warnings.append(f"unknown ranking criterion '{name}', ignored")
                continue
            direction = c.get("direction") or (
                "asc" if name in NATURALLY_ASCENDING else "desc"
            )
            out.append(Criterion(
                name=name,
                direction=direction,
                weight=float(c.get("weight", 1.0) or 1.0),
                origin=c.get("origin"),
                scope=c.get("scope"),
                profile_id=c.get("profile_id"),
                preferred_values=list(c.get("preferred_values") or []),
                rationale=c.get("rationale"),
            ))

        if not out:
            out = [
                Criterion(name=n, direction=d, weight=w)
                for n, d, w in STRATEGY_DEFAULT_CRITERIA.get(self.strategy, [])
            ]
            if out:
                self.warnings.append(
                    f"no criteria given; using defaults for strategy "
                    f"'{self.strategy}'"
                )

        if self.stage == "discovery":
            blocked = [c for c in out if c.name in EXPLANATION_DERIVED_CRITERIA]
            for c in blocked:
                self.warnings.append(
                    f"criterion '{c.name}' is explanation-derived and cannot be "
                    f"computed during discovery ranking; ignored"
                )
            out = [c for c in out if c.name not in EXPLANATION_DERIVED_CRITERIA]

        return out

    # -- scoring -----------------------------------------------------------

    def score_all(self, candidates: Dict[str, Candidate]) -> List[Candidate]:
        """Compute metrics, normalize, and produce a weighted score."""
        pool = list(candidates.values())
        if not pool:
            return []

        for cand in pool:
            cand.metrics = compute_metrics(cand)

        # knowledge_level is categorical, so its score depends on the ordering
        # the plan supplies rather than on any convention here. Computed once
        # the criteria are known, and only when a criterion asks for it.
        kl_criterion = next(
            (c for c in self.criteria if c.name == "knowledge_level"), None
        )
        if kl_criterion is not None:
            for cand in pool:
                scores = [
                    kl_criterion.value_score(level)
                    for level in cand.knowledge_levels
                ]
                scores = [x for x in scores if x is not None]
                if scores:
                    cand.metrics["knowledge_level"] = sum(scores) / len(scores)

        ranges: Dict[str, Tuple[float, float]] = {}
        for criterion in self.criteria:
            name = criterion.name
            values = [c.metrics[name] for c in pool if name in c.metrics]
            if values:
                ranges[name] = (min(values), max(values))
            else:
                self.warnings.append(
                    f"criterion '{name}' has no data on any candidate; skipped"
                )

        lit_values = [
            c.literature_score for c in pool if c.literature_score is not None
        ]
        lit_range = (min(lit_values), max(lit_values)) if lit_values else None

        for cand in pool:
            total, total_weight = 0.0, 0.0

            for criterion in self.criteria:
                name, direction, weight = (
                    criterion.name, criterion.direction, criterion.weight,
                )
                if name not in ranges or name not in cand.metrics:
                    continue
                low, high = ranges[name]
                value = cand.metrics[name]
                # A criterion identical across all candidates cannot separate
                # them; scoring it 0.5 keeps it from tilting the total either
                # way.
                norm = 0.5 if high == low else (value - low) / (high - low)
                if direction == "asc":
                    norm = 1.0 - norm
                cand.normalized[name] = norm
                total += norm * weight
                total_weight += weight

            # Literature enters as its own term rather than through EPC, and
            # only for candidates that were actually verified. An unchecked
            # candidate simply does not carry this term, instead of carrying a
            # zero that would rank it below one that failed verification.
            if lit_range and cand.literature_score is not None and self.literature_weight:
                low, high = lit_range
                norm = 0.5 if high == low else (cand.literature_score - low) / (high - low)
                cand.normalized["literature_support"] = norm
                total += norm * self.literature_weight
                total_weight += self.literature_weight
            elif cand.literature_score is None and self.literature_weight:
                cand.notes.append("literature not verified; excluded from that term")

            cand.score = total / total_weight if total_weight else 0.0

        return pool

    def _tie_break_key(self, cand: Candidate):
        if self.tie_breaker == "alphabetical":
            return cand.label.lower() or cand.curie
        if self.tie_breaker == "recency":
            return -(cand.metrics.get("recency") or 0)
        if self.tie_breaker == "random":
            return self.rng.random()
        # path_priority_order: a candidate found by an earlier-listed path
        # wins, since path order in the plan expresses the planner's own
        # preference.
        return (cand.path_priority if cand.path_priority is not None else 10**6)

    def rank(
        self,
        candidates: Dict[str, Candidate],
        question: str = "",
        edges: Optional[Dict[str, Any]] = None,
    ) -> List[Candidate]:
        """Score, order, optionally rerank, and truncate to top_k."""
        pool = self.score_all(candidates)
        if not pool:
            return []

        pool.sort(key=lambda c: (-c.score, self._tie_break_key(c)))
        for i, cand in enumerate(pool, 1):
            cand.rank = i
            cand.deterministic_rank = i

        if not self.quiet_warnings:
            for w in self.warnings:
                self.log(f"warning: {w}")

        if self.reranker is not None:
            pool = self._apply_rerank(pool, question, edges or {})
        elif self.require_reranker:
            raise RuntimeError(
                "ranking requires an LLM reranker but none was configured"
            )

        if self.top_k:
            pool = pool[: self.top_k]

        self.log(f"{len(pool)} candidate(s) ranked "
                 f"(stage={self.stage}, strategy={self.strategy}, "
                 f"criteria={[c.name for c in self.criteria]})")
        return pool

    # -- reranking ---------------------------------------------------------

    def _apply_rerank(
        self,
        pool: List[Candidate],
        question: str,
        edges: Dict[str, Any],
    ) -> List[Candidate]:
        """Let an LLM reorder the top-K, keeping only grounded reorderings.

        Only the head of the list is reranked: reordering candidates nobody
        will read costs tokens without changing what anyone sees.
        """
        head = pool[: self.rerank_top_k]
        tail = pool[self.rerank_top_k:]
        by_curie = {c.curie: c for c in head}

        payload = [self._rerank_payload(c, edges) for c in head]
        try:
            proposals = self.reranker.rerank(question=question, candidates=payload)
        except Exception as e:
            self.log(f"reranker failed, deterministic order stands: {e}")
            for c in head:
                c.notes.append(f"reranker unavailable ({e})")
            if self.require_reranker:
                raise
            return pool

        accepted = 0
        for proposal in proposals or []:
            curie = proposal.get("curie")
            cand = by_curie.get(curie)
            if cand is None:
                self.warnings.append(
                    f"reranker named unknown candidate '{curie}', ignored"
                )
                continue

            cited = set(proposal.get("cited_edge_ids") or [])
            grounded, detail = self._check_grounding(
                cand, cited, edges, reason=proposal.get("reason", ""),
            )
            cand.rerank_grounded = grounded

            if not grounded:
                # An ungrounded justification is discarded rather than
                # discounted: a reason that cannot be traced to a retrieved
                # edge is not evidence about this candidate at all.
                cand.notes.append(f"reranking rejected: {detail}")
                continue

            cand.rerank_reason = proposal.get("reason", "")
            new_rank = proposal.get("rank")
            if isinstance(new_rank, int) and new_rank > 0:
                cand.rank = new_rank
                accepted += 1

        head.sort(key=lambda c: (c.rank, c.deterministic_rank))
        for i, cand in enumerate(head, 1):
            cand.rank = i
        for i, cand in enumerate(tail, len(head) + 1):
            cand.rank = i

        self.log(f"reranked {accepted}/{len(head)} candidate(s); "
                 f"{sum(1 for c in head if c.rerank_grounded is False)} rejected "
                 f"as ungrounded")
        return head + tail

    @staticmethod
    def _check_grounding(
        candidate: Candidate,
        cited_edge_ids: Set[str],
        edges: Dict[str, Any],
        reason: str = "",
    ) -> Tuple[bool, str]:
        """Verify a reranking is both correctly cited and drawn from the evidence.

        Citation checks come first. Two ways to fail there: citing an edge
        absent from the result graph, or citing a real edge belonging to a
        different candidate — a genuine edge about another drug says nothing
        about this one.

        Then the content check. A citation proves the edge exists; it does not
        prove the reason followed from it. "Classic approved EGFR inhibitor
        [cites e42]" passes every citation test while being recalled
        pharmacology with a reference stapled on — and an ordering built from
        recall rather than retrieval is one the knowledge graph did not
        produce.

        So the reason must reference something the cited edges actually
        contain: a word from a verified quote, or failing that a predicate,
        qualifier or knowledge level. Naming the source databases does not
        count, since provenance is already scored deterministically and
        restating it adds nothing a script could not do.
        """
        if not cited_edge_ids:
            return False, "no edge ids cited"

        unknown = [e for e in cited_edge_ids if edges and e not in edges]
        if unknown:
            return False, f"cited edges absent from the result graph: {unknown[:3]}"

        foreign = [e for e in cited_edge_ids if e not in candidate.edge_ids]
        if foreign:
            return False, (
                f"cited edges do not support this candidate: {foreign[:3]}"
            )

        if not reason.strip():
            return False, "citations resolve, but no reason was given"

        tokens = _content_tokens(candidate, cited_edge_ids, edges)
        # Separators are normalised on both sides so that a reason writing
        # "decreased activity" matches a qualifier stored as
        # "decreased_activity".
        flat = reason.lower().replace("_", " ").replace("-", " ")
        words = {"".join(ch for ch in w if ch.isalpha()) for w in flat.split()}
        words.discard("")

        hit_quote = sorted(words & tokens["quote"])[:3]
        if hit_quote:
            return True, f"reason draws on verified quoted text ({hit_quote})"

        hit_structure = sorted(
            t for t in tokens["structure"] if t in words or t in flat
        )[:3]
        if hit_structure:
            return True, f"reason references retrieved edge content ({hit_structure})"

        available = sorted(tokens["quote"])[:6] or sorted(tokens["structure"])[:6]
        return False, (
            "reason cites valid edges but does not reference anything they "
            "contain — it restates provenance or recalls outside knowledge. "
            f"Content available to draw on: {available}"
        )

    def _rerank_payload(self, cand: Candidate, edges: Dict[str, Any]) -> Dict[str, Any]:
        """What the reranker sees: only edges that actually support this candidate."""
        from .postfilter import summarize_edge

        evidence = []
        for eid in sorted(cand.edge_ids)[:12]:
            edge = edges.get(eid)
            if edge:
                item = summarize_edge(edge)
                item["edge_id"] = eid
                support = cand.literature.get(eid)
                if isinstance(support, dict):
                    # Whether an edge's literature was checked, and what came
                    # of it, are different facts. Without this the reranker
                    # cannot tell an edge whose citations failed from one that
                    # never had any, and ranks both below an edge that merely
                    # happened to fall inside the verification budget.
                    item["literature_status"] = support.get("status")
                    # The quoted text is the only part of the payload that
                    # came from reading rather than from metadata, so it is
                    # lifted to a field of its own instead of being nested
                    # inside the citation record where it was overlooked.
                    #
                    # Taken from best_citation rather than from the first
                    # verified verdict, so the paper the model quotes is the
                    # same one the result displays. Selecting each by its own
                    # rule put two different real PMIDs on one candidate and
                    # made a correct citation look invented.
                    best = support.get("best_citation") or {}
                    if best:
                        item["verified_citation"] = best
                    if best.get("quote"):
                        item["verified_quote"] = best["quote"][:400]
                        item["verified_quote_pmid"] = best.get("pmid")
                    else:
                        for verdict in support.get("verdicts") or []:
                            if verdict.get("quote_verified") and verdict.get("quote"):
                                item["verified_quote"] = verdict["quote"][:400]
                                item["verified_quote_pmid"] = verdict.get("pmid")
                                break
                else:
                    item["literature_status"] = "not_checked"
                evidence.append(item)

        return {
            "curie": cand.curie,
            "label": cand.label,
            "deterministic_rank": cand.deterministic_rank,
            "supporting_paths": sorted(cand.path_ids),
            "num_supporting_paths": cand.num_supporting_paths,
            "metrics": {k: round(v, 3) for k, v in cand.metrics.items()},
            "evidence": evidence,
        }


# ---------------------------------------------------------------------------
# Ranking plan
# ---------------------------------------------------------------------------


@dataclass
class RankingPlan:
    """The plan's ranking block, split by stage.

    Ranking is stage-specific: discovery orders candidates before explanations
    exist, `final` may reorder them afterwards, and explanation ranking orders
    paths. Keeping them apart is what makes `explanation_influence` meaningful
    — without it there is no way to say that explanations may annotate
    candidates without moving them.
    """

    explanation_influence: str = INFLUENCE_NONE
    rationale: Optional[str] = None
    discovery: Optional[Dict[str, Any]] = None
    final: Optional[Dict[str, Any]] = None
    explanation: Optional[Dict[str, Any]] = None
    warnings: List[str] = field(default_factory=list)

    @property
    def attaches_explanations(self) -> bool:
        return self.explanation_influence in (INFLUENCE_ANNOTATE, INFLUENCE_RERANK)

    @property
    def reorders_after_explanation(self) -> bool:
        """True only for `rerank`.

        Under `annotate_only` the discovery order is the answer and
        explanations are context attached to it; moving a candidate would
        contradict what the plan asked for.
        """
        return self.explanation_influence == INFLUENCE_RERANK

    @classmethod
    def from_plan(cls, ranking: Any) -> "RankingPlan":
        raw = ranking or {}
        if hasattr(raw, "model_dump"):
            raw = raw.model_dump(exclude_none=True)
        elif not isinstance(raw, dict) and hasattr(raw, "__dict__"):
            raw = {k: v for k, v in vars(raw).items() if v is not None}

        warnings: List[str] = []
        if "strategy" in raw or "criteria" in raw:
            warnings.append(
                "ranking block uses the pre-0.8 top-level shape "
                "({strategy, criteria, top_k}); v0.8.0 requires "
                "candidate_ranking and/or explanation_ranking"
            )

        candidate = raw.get("candidate_ranking") or {}
        if hasattr(candidate, "model_dump"):
            candidate = candidate.model_dump(exclude_none=True)

        plan = cls(
            explanation_influence=(
                candidate.get("explanation_influence") or INFLUENCE_NONE
            ),
            rationale=candidate.get("rationale"),
            discovery=candidate.get("discovery"),
            final=candidate.get("final"),
            explanation=raw.get("explanation_ranking"),
            warnings=warnings,
        )

        # `rerank` promises a second ordering that uses something explanations
        # produced. Without a final block there is nothing to apply, and
        # falling back to the discovery order would silently turn this into
        # annotate_only.
        if plan.explanation_influence == INFLUENCE_RERANK and not plan.final:
            plan.warnings.append(
                "explanation_influence is 'rerank' but no candidate_ranking."
                "final block was given; candidate order will not change"
            )
        if plan.explanation_influence == INFLUENCE_ANNOTATE and plan.final:
            plan.warnings.append(
                "explanation_influence is 'annotate_only', so the "
                "candidate_ranking.final block is not applied"
            )
        if plan.final:
            names = {
                (c.get("name") if isinstance(c, dict) else getattr(c, "name", None))
                for c in (plan.final.get("criteria") or [])
            }
            if not (names & EXPLANATION_DERIVED_CRITERIA):
                plan.warnings.append(
                    "candidate_ranking.final names no explanation-derived "
                    "criterion, so reranking would reproduce the discovery "
                    "order"
                )
        return plan

    def spec(self, stage: str) -> Optional[Dict[str, Any]]:
        return {"discovery": self.discovery, "final": self.final,
                "explanation": self.explanation}.get(stage)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "explanation_influence": self.explanation_influence,
            "rationale": self.rationale,
            "has_discovery": self.discovery is not None,
            "has_final": self.final is not None,
            "has_explanation": self.explanation is not None,
            "warnings": self.warnings,
        }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------


def ranking_report(ranked: Sequence[Candidate], limit: int = 10) -> str:
    lines = []
    for c in ranked[:limit]:
        moved = ""
        if c.rerank_reason and c.deterministic_rank != c.rank:
            moved = f"  (was #{c.deterministic_rank})"
        cite = c.best_citation
        lines.append(
            f"  #{c.rank:2d} {c.label[:38]:38s} {c.score:.3f} "
            f"paths={c.num_supporting_paths}{moved}"
        )
        if cite:
            lines.append(f"        cite {cite.get('pmid')} ({cite.get('year')})")
        if c.rerank_reason:
            lines.append(f"        {c.rerank_reason[:100]}")
    return "\n".join(lines)
