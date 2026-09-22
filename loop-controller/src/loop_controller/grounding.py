"""
grounding.py — The gate. Nothing leaves without a path in the graph behind it.

This is where the project's hard constraint is enforced rather than intended.
Two claims have to hold about every sentence the system serves:

  1. Every identifier, label, relation and source it names occurs in the result
     graph this answer was composed from.
  2. No part of it was written by a language model.

The first is the reverse-grounding requirement stated directly. The second is
subtler and is the reason this module exists as a checker rather than as a
convention. `result.json` is not model-free. The executor uses a model in three
places and keeps what it said, correctly, because the reasoning should be
auditable: `rerank_reason` on each candidate, `reason` and `alias_used` on each
resolution, and per-abstract `rationale` in the literature verdicts. All three
sit in fields a composer would naturally reach for — `rerank_reason` in
particular reads exactly like the one-line justification a candidate list wants
next to each row.

So the gate builds two things. A **lexicon** of everything the graph actually
contains, and a **quarantine** of every model-authored string in the document.
A statement passes when all its tokens are in the first and none of its text
overlaps the second. Statements are checked individually and a failing one is
dropped and recorded rather than silently repaired, because a gate that edits
its input cannot be used as evidence that its input was correct.

The quarantine check is a substring test with a length floor. Below the floor,
model-written text and graph-derived text are not distinguishable by content —
a rerank reason may legitimately contain "Petrolatum" — and above it, a match
means composed prose reached the answer. The floor makes the check a detector
of copied sentences, which is what actually goes wrong, rather than of shared
vocabulary, which is unavoidable.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Sequence, Set


#: Fields in a result document whose values a model wrote. Quarantined
#: wholesale: never rendered, and checked against everything that is.
#:
#: The executor's three are the obvious ones. The plan's are easier to miss and
#: more tempting to use: `interpretation.restated_question` reads like a clean
#: restatement of what the user asked and would make an excellent headline, and
#: it was written by the planner's model. So was the refusal message, which is
#: why a refusal is rendered from its typed `reason` here rather than by
#: passing the planner's sentence through.
LLM_AUTHORED_FIELDS: Dict[str, Sequence[str]] = {
    "results[].rerank_reason": ("results", "rerank_reason"),
    "resolution{}.reason": ("resolution", "reason"),
    "evidence.literature.by_edge{}.verdicts[].rationale": (
        "evidence", "literature", "rationale",
    ),
    "plan.interpretation.restated_question": ("plan", "interpretation"),
    "plan.interpretation.intent": ("plan", "interpretation"),
    "plan.confidence.reasons[]": ("plan", "confidence"),
    "plan.gaps[].*": ("plan", "gaps"),
    "refusal.message": ("refusal",),
}

#: Minimum length for a quarantined string to be worth checking as a substring.
#: Shorter fragments collide with graph vocabulary and would fail every answer
#: that named an entity the reranker also named.
QUARANTINE_MIN_LEN = 40

_CURIE_RE = re.compile(r"\b[A-Za-z][A-Za-z0-9_.]*:[A-Za-z0-9_.\-]+\b")
_QUOTE_RE = re.compile(r"[“\"]([^”\"]{8,})[”\"]")


# ---------------------------------------------------------------------------
# Lexicon
# ---------------------------------------------------------------------------


@dataclass
class Lexicon:
    """Everything the result document actually contains.

    Built from the evidence attached to candidates and to explanation paths —
    that is, from edges that were returned — rather than from the plan. A plan
    names what was asked for; only the result says what came back, and an
    answer may only speak about what came back.
    """

    curies: Set[str] = field(default_factory=set)
    labels: Set[str] = field(default_factory=set)
    predicates: Set[str] = field(default_factory=set)
    sources: Set[str] = field(default_factory=set)
    knowledge_levels: Set[str] = field(default_factory=set)
    agent_types: Set[str] = field(default_factory=set)
    edge_ids: Set[str] = field(default_factory=set)
    publications: Set[str] = field(default_factory=set)
    verified_quotes: List[str] = field(default_factory=list)
    quarantine: List[str] = field(default_factory=list)

    #: Terms from the query that was *sent*, rather than from what came back:
    #: the categories, predicates and pinned identifiers in the submitted TRAPI
    #: graphs.
    #:
    #: These are needed because some statements are about the question rather
    #: than the answer. "The query that returned nothing was: any ChemicalEntity
    #: —[biolink:treats]→ dermatitis" is the whole content of an absence, and
    #: every term in it is one the graph did not return — by definition, since
    #: the graph returned nothing. Checking it against the result graph alone
    #: rejects the one sentence an absence exists to say.
    #:
    #: Kept separate rather than merged into `curies` and `predicates`, because
    #: the distinction is the point: a claim about what the graph *asserts* must
    #: come from a returned edge, and only a claim about what was *asked* may
    #: draw on this.
    query_terms: Set[str] = field(default_factory=set)

    #: Identifiers the name resolver retrieved and did *not* accept — the
    #: entries it considered for a name that failed to ground.
    #:
    #: These occur in no result graph, because a run that could not resolve its
    #: anchor never sent a query. They are still retrieved facts rather than
    #: invention: the resolver looked the name up and these are what came back,
    #: recorded in the result document with their own identifiers and labels.
    #:
    #: Kept apart from `curies` for the same reason `query_terms` is. A claim
    #: that the graph asserts something must rest on a returned edge. Offering
    #: the user a list of entries to choose between asserts nothing about any
    #: of them, so it may draw on this set and only this set.
    resolver_candidates: Set[str] = field(default_factory=set)

    def known_identifier(self, token: str) -> bool:
        return (
            token in self.curies
            or token in self.sources
            or token in self.predicates
            or token in self.publications
            or token in self.knowledge_levels
            or token in self.edge_ids
        )

    def stats(self) -> Dict[str, int]:
        return {
            "curies": len(self.curies),
            "labels": len(self.labels),
            "predicates": len(self.predicates),
            "sources": len(self.sources),
            "edges": len(self.edge_ids),
            "publications": len(self.publications),
            "verified_quotes": len(self.verified_quotes),
            "quarantined_strings": len(self.quarantine),
            "query_terms": len(self.query_terms),
            "resolver_candidates": len(self.resolver_candidates),
        }


def build_lexicon(result: Dict[str, Any]) -> Lexicon:
    """Walk the result document and record what is in it."""
    lex = Lexicon()

    for candidate in result.get("results") or []:
        _add_candidate(candidate, lex)

    # Wherever the executor wrote them. A hybrid plan with
    # `attach_explanations_to_candidates` puts the paths on the candidates and
    # leaves the endpoint map empty, and reading only the map left every
    # intermediate node out of the lexicon — so a composer that *did* find the
    # paths would have every one of its statements withheld as ungrounded.
    from .path_quality import collect_explanation_paths

    for paths in collect_explanation_paths(result).values():
        for path in paths or []:
            _add_explanation_path(path, lex)

    for record in (result.get("resolution") or {}).values():
        for curie in record.get("resolved_curies") or []:
            lex.curies.add(str(curie))
        if record.get("resolved_label"):
            lex.labels.add(str(record["resolved_label"]))
        # Entries the resolver retrieved for this name. Recorded whether or not
        # one was accepted, because the list is what a clarification offers.
        for considered in record.get("considered") or []:
            if considered.get("curie"):
                lex.resolver_candidates.add(str(considered["curie"]))
            if considered.get("label"):
                lex.labels.add(str(considered["label"]))

    for check in result.get("concept_checks") or []:
        for curie in (check.get("observed_curies") or []):
            lex.curies.add(str(curie))
        for label in (check.get("observed_labels") or []):
            lex.labels.add(str(label))
        if check.get("requested_curie"):
            lex.curies.add(str(check["requested_curie"]))
        if check.get("requested_label"):
            lex.labels.add(str(check["requested_label"]))

    _collect_query_terms(result, lex)
    _collect_quarantine(result, lex)
    return lex


def _collect_query_terms(result: Dict[str, Any], lex: Lexicon) -> None:
    """Everything the submitted queries named.

    Read from `paths[].submitted_queries[].query_graph` — what was actually
    sent — rather than from the plan, which says what was asked for. When a
    multi-hop query is decomposed those differ, and a description of the query
    should match the one that ran.
    """
    for path in (result.get("paths") or {}).values():
        for submitted in path.get("submitted_queries") or []:
            graph = submitted.get("query_graph") or {}
            for node in (graph.get("nodes") or {}).values():
                for curie in node.get("ids") or []:
                    lex.query_terms.add(str(curie))
                for category in node.get("categories") or []:
                    lex.query_terms.add(str(category))
            for edge in (graph.get("edges") or {}).values():
                for predicate in edge.get("predicates") or []:
                    lex.query_terms.add(str(predicate))

    # An explanation query that found nothing still names the endpoint it
    # failed to reach, and that endpoint is not in any returned edge.
    explanations = (result.get("evidence") or {}).get("explanations") or {}
    for summary in explanations.get("summaries") or []:
        if summary.get("endpoint_b"):
            lex.query_terms.add(str(summary["endpoint_b"]))

    # The user's own evidence filters name knowledge sources that, by having
    # removed everything, appear in no surviving edge.
    policy = (result.get("plan") or {}).get("evidence_policy") or {}
    if isinstance(policy, dict):
        for key in ("required_knowledge_sources", "excluded_knowledge_sources"):
            for source in policy.get(key) or []:
                lex.query_terms.add(str(source))


def _add_candidate(candidate: Dict[str, Any], lex: Lexicon) -> None:
    if candidate.get("curie"):
        lex.curies.add(str(candidate["curie"]))
    if candidate.get("label"):
        lex.labels.add(str(candidate["label"]))
    for evidence in (candidate.get("evidence_by_path") or {}).values():
        for instance in evidence or []:
            for curie in (instance.get("bindings") or {}).values():
                if curie:
                    lex.curies.add(str(curie))
            for edge in instance.get("edges") or []:
                _add_edge(edge, lex)


def _add_explanation_path(path: Dict[str, Any], lex: Lexicon) -> None:
    for curie in path.get("nodes") or []:
        lex.curies.add(str(curie))
    for label in path.get("labels") or []:
        lex.labels.add(str(label))
    for predicate in path.get("predicates") or []:
        lex.predicates.add(str(predicate))
    for edge in path.get("edges") or []:
        _add_edge(edge, lex)


def _add_edge(edge: Dict[str, Any], lex: Lexicon) -> None:
    for key in ("subject", "object"):
        if edge.get(key):
            lex.curies.add(str(edge[key]))
    if edge.get("predicate"):
        lex.predicates.add(str(edge["predicate"]))
    if edge.get("primary_source"):
        lex.sources.add(str(edge["primary_source"]))
    for source in edge.get("sources") or []:
        lex.sources.add(str(source))
    if edge.get("knowledge_level"):
        lex.knowledge_levels.add(str(edge["knowledge_level"]))
    if edge.get("agent_type"):
        lex.agent_types.add(str(edge["agent_type"]))
    if edge.get("edge_id"):
        lex.edge_ids.add(str(edge["edge_id"]))
    for pmid in edge.get("publications") or []:
        lex.publications.add(str(pmid))

    literature = edge.get("literature") or {}
    for verdict in literature.get("verdicts") or []:
        if verdict.get("quote") and verdict.get("quote_verified"):
            lex.verified_quotes.append(str(verdict["quote"]))
        if verdict.get("pmid"):
            lex.publications.add(str(verdict["pmid"]))
    best = literature.get("best_citation")
    if isinstance(best, dict):
        if best.get("pmid"):
            lex.publications.add(str(best["pmid"]))
        if best.get("quote") and best.get("quote_verified"):
            lex.verified_quotes.append(str(best["quote"]))


def _collect_quarantine(result: Dict[str, Any], lex: Lexicon) -> None:
    """Gather every model-authored string in the document.

    Collected by walking the known field names rather than by scanning for
    prose, so a new model-authored field added upstream is *not* caught here.
    That is the honest failure mode and it is checked separately by
    `unquarantined_prose_fields`, which flags long free text in places this
    module does not know about, so the gap is reported rather than assumed
    away.
    """
    for candidate in result.get("results") or []:
        reason = candidate.get("rerank_reason")
        if reason:
            lex.quarantine.append(str(reason))

    for record in (result.get("resolution") or {}).values():
        for key in ("reason",):
            if record.get(key):
                lex.quarantine.append(str(record[key]))

    by_edge = ((result.get("evidence") or {}).get("literature") or {}).get("by_edge") or {}
    for support in by_edge.values():
        for verdict in support.get("verdicts") or []:
            for key in ("rationale", "reason", "explanation"):
                if verdict.get(key):
                    lex.quarantine.append(str(verdict[key]))

    # The plan's prose. Written by the planner's model, carried into the result
    # document for auditability, and the most quotable text in the file.
    plan = result.get("plan") or {}
    interpretation = plan.get("interpretation") or {}
    for key in ("restated_question", "intent"):
        if interpretation.get(key):
            lex.quarantine.append(str(interpretation[key]))
    for reason in (plan.get("confidence") or {}).get("reasons") or []:
        lex.quarantine.append(str(reason))
    for gap in plan.get("gaps") or []:
        if isinstance(gap, dict):
            for value in gap.values():
                if isinstance(value, str):
                    lex.quarantine.append(value)
    refusal = result.get("refusal") or {}
    for key in ("message", "detail", "explanation"):
        if refusal.get(key):
            lex.quarantine.append(str(refusal[key]))

    # The evidence policy's `rationale` restates the user's request in the
    # planner's words. The request is the user's; the words are the model's.
    policy = plan.get("evidence_policy") or {}
    if isinstance(policy, dict) and policy.get("rationale"):
        lex.quarantine.append(str(policy["rationale"]))

    lex.quarantine = [q for q in lex.quarantine if len(q) >= QUARANTINE_MIN_LEN]


# ---------------------------------------------------------------------------
# Checking
# ---------------------------------------------------------------------------


@dataclass
class Violation:
    statement_id: str
    kind: str
    detail: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "statement_id": self.statement_id,
            "kind": self.kind,
            "detail": self.detail,
        }


#: Statement kinds that describe the query rather than the answer, and may
#: therefore name terms from the submitted query graph.
#:
#: Deliberately a short, explicit list. Every kind added here is one whose
#: claims are no longer checked against returned edges, so the bar for adding
#: one is that the statement genuinely talks about what was *asked* — never
#: about what the graph says.
QUERY_DESCRIBING_KINDS = frozenset({
    "query_description",
    "anchors",
    "policy_description",
    "explanation_absence",
})

#: Statement kinds that offer the user a list of entries to choose between,
#: and may therefore name identifiers the resolver retrieved but did not accept.
#:
#: One kind, and it should stay that way. A statement of this kind asserts
#: nothing about any entry it names — it says "the search returned these" and
#: asks which was meant. The moment a kind here started saying what the graph
#: holds about one of them, it would be making a claim out of a list of
#: candidates that were never queried.
CANDIDATE_OFFERING_KINDS = frozenset({
    "resolution_options",
})


def check_statement(
    statement: Dict[str, Any], lex: Lexicon, *, allow_quotes: bool = True,
) -> List[Violation]:
    """Every reason this statement may not be served."""
    sid = str(statement.get("id") or "?")
    text = str(statement.get("text") or "")
    kind = str(statement.get("kind") or "")
    describes_query = kind in QUERY_DESCRIBING_KINDS
    offers_candidates = kind in CANDIDATE_OFFERING_KINDS
    violations: List[Violation] = []

    for edge_id in statement.get("supported_by") or []:
        if str(edge_id) not in lex.edge_ids:
            violations.append(Violation(
                sid, "unknown_edge",
                f"cites edge '{edge_id}', which is not in the result graph",
            ))

    for token in _CURIE_RE.findall(text):
        if token in lex.labels:
            continue
        if lex.known_identifier(token):
            continue
        if describes_query and token in lex.query_terms:
            continue
        if offers_candidates and token in lex.resolver_candidates:
            continue
        if describes_query:
            where = "the result graph or the submitted query"
        elif offers_candidates:
            where = "the result graph or the resolver's candidates"
        else:
            where = "the result graph"
        violations.append(Violation(
            sid, "ungrounded_identifier",
            f"names '{token}', which does not occur in {where}",
        ))

    for quoted in _QUOTE_RE.findall(text):
        if not allow_quotes:
            violations.append(Violation(
                sid, "quote_not_permitted",
                "contains a quotation, which this answer kind does not allow",
            ))
            continue
        if not any(quoted.strip() in q or q in quoted for q in lex.verified_quotes):
            violations.append(Violation(
                sid, "unverified_quote",
                f"quotes text that was not verified present in a retrieved "
                f"abstract: {quoted[:60]!r}",
            ))

    for authored in lex.quarantine:
        if authored in text:
            violations.append(Violation(
                sid, "model_authored_text",
                f"reproduces model-written text from the result document: "
                f"{authored[:60]!r}",
            ))

    return violations


def check_answer(
    answer: Dict[str, Any], lex: Lexicon, *, allow_quotes: bool = True,
) -> Dict[str, Any]:
    """Check every statement and report what passed.

    Returns a report rather than raising. The composer uses it to drop failing
    statements and to record why, and the report is written into the answer
    itself, so an answer document carries the evidence that it was checked
    alongside the claims that were checked.
    """
    violations: List[Violation] = []
    passed: List[str] = []
    for statement in answer.get("statements") or []:
        found = check_statement(statement, lex, allow_quotes=allow_quotes)
        if found:
            violations.extend(found)
        else:
            passed.append(str(statement.get("id")))

    return {
        "checked": len(answer.get("statements") or []),
        "passed": len(passed),
        "failed": len(violations),
        "ok": not violations,
        "violations": [v.to_dict() for v in violations],
        "lexicon": lex.stats(),
    }


def unquarantined_prose_fields(
    result: Dict[str, Any], *, min_len: int = 120,
) -> List[str]:
    """Long free text in fields this module does not know to quarantine.

    A safety net for the one way the gate can be wrong: an upstream change adds
    a model-authored field, the quarantine does not know it, and a composer
    that reached for it would pass the check. This does not prevent that — it
    surfaces the candidates, so a reviewer sees "three long text fields exist
    that the gate does not classify" rather than nothing at all.

    Known-deterministic fields are excluded by name: they are assembled by the
    executor's own code from counts and identifiers, so their length says
    nothing about their authorship.
    """
    deterministic = {
        "detail", "description", "message", "concept_warning", "restated_question",
        "question", "note", "notes", "warnings", "verdict_reasons", "assertion",
        "coverage_notes", "drop_reason", "error", "path", "endpoint",
    }
    known = {"rerank_reason", "reason", "rationale", "explanation"}
    found: List[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                walk(value, f"{path}.{key}" if path else str(key))
        elif isinstance(node, list):
            for item in node[:50]:
                walk(item, f"{path}[]")
        elif isinstance(node, str) and len(node) >= min_len:
            leaf = path.rsplit(".", 1)[-1].replace("[]", "")
            if leaf in deterministic or leaf in known:
                return
            if " " not in node:
                return  # an identifier or a serialized blob, not prose
            found.append(path)

    walk(result, "")
    seen, out = set(), []
    for item in found:
        if item not in seen:
            seen.add(item)
            out.append(item)
    return out[:20]
