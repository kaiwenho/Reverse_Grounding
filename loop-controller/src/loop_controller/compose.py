"""
compose.py — Turning a result document into something a person can read,
without a language model anywhere in the sentence.

Every statement here is assembled from typed values: identifiers, labels,
Biolink predicates, knowledge levels, source identifiers, counts, and verbatim
quotes that were verified present in a retrieved abstract. The templates are in
this file and they are fixed. Nothing is summarised, nothing is paraphrased,
and no field written by a model is read — those are enumerated in
`grounding.LLM_AUTHORED_FIELDS` and every composed statement is checked against
them before it is kept.

That constraint shapes the output more than it might seem. There is no
"Petrolatum is a plausible candidate because it is already used topically" —
that sentence requires a model, and would be exactly the kind of fluent,
unfalsifiable claim the whole architecture exists to keep out. What replaces it
is narrower and duller and checkable: this edge, from this source, at this
knowledge level, with this publication behind it.

Refusals and absences get the same treatment. A refusal is rendered from its
typed `reason`, not from the planner's message. An absence names the query that
was run and the concepts it was anchored on, and says what was established
rather than that nothing was found.

Two smaller decisions worth stating. Verified quotes are included, because a
quote is the abstract's text and its presence there was checked byte-for-byte
by the executor — but the answer records that the *choice* of quote was
model-assisted, because it was. And every candidate carries its supporting
edges by identifier, so a reader who wants to check a row does not have to take
the row's word for anything.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .contracts import (
    Diagnosis, OUTCOME_FILTERED_OUT, OUTCOME_JOIN_FAILURE, OUTCOME_NO_DATA,
    OUTCOME_REFUSED, OUTCOME_TRUNCATED, OUTCOME_UNRESOLVED,
)
from .grounding import Lexicon, build_lexicon, check_answer, unquarantined_prose_fields
from .path_quality import PathReview, collect_explanation_paths, review_paths


ANSWER_SCHEMA_VERSION = "loop-controller-answer/0.1.0"

KIND_CANDIDATES = "ranked_candidates"
KIND_EXPLANATION = "explanation"
KIND_ABSENCE = "absence"
KIND_FILTERED_ABSENCE = "filtered_absence"
KIND_REFUSAL = "refusal"
KIND_CLARIFICATION = "clarification_needed"
KIND_INCONCLUSIVE = "inconclusive"

#: How many nearby entries a clarification lists before it summarises the rest.
#: Twenty is what the resolver returns and twenty is too many to read; the ones
#: past the first few are ranked lower and rarely what was meant.
MAX_RESOLUTION_OPTIONS = 8


#: What each refusal reason means, in the system's own words rather than the
#: planner's. Keyed on the contract's enum, so a new reason produces a generic
#: sentence instead of a wrong one.
REFUSAL_TEXT: Dict[str, str] = {
    "out_of_scope": (
        "This question falls outside what this system answers: it works over a "
        "biomedical knowledge graph of entities and their asserted "
        "relationships."
    ),
    "unsafe_or_clinical_advice": (
        "This question asks for individualised clinical advice — a medication, "
        "a dose, or a treatment choice for a particular person. This system "
        "reports what a knowledge graph asserts and is not a source of "
        "clinical guidance."
    ),
    "insufficient_information": (
        "This question does not carry enough information to build a query "
        "against the knowledge graph."
    ),
    "needs_clarification": (
        "This question is underspecified. A query could not be built without "
        "guessing at what was meant."
    ),
    "requires_capability_not_available": (
        "Answering this question needs a capability this system does not have; "
        "the relationship it asks about is not one the knowledge graph models."
    ),
}


@dataclass
class ComposerSettings:
    top_k: int = 20
    max_edges_per_candidate: int = 3
    max_explanations: int = 5
    max_quotes: int = 5
    include_quotes: bool = True


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def compose(
    result: Dict[str, Any],
    diag: Optional[Diagnosis] = None,
    *,
    question: Optional[str] = None,
    settings: Optional[ComposerSettings] = None,
    iterations: int = 1,
    plan: Optional[Dict[str, Any]] = None,
    extra_caveats: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Build the answer document, then check it, then drop what failed.

    The order matters. Statements are composed first and filtered second, so
    the grounding report counts what composition actually produced. A composer
    that checked as it went could never report that it had produced something
    ungrounded, because it would never have produced it — and the count of
    dropped statements is the only visible evidence that the gate does
    anything.

    ``plan`` is the plan that was executed. It is accepted because the result
    document's own plan block does not carry everything the plan does — see
    `_with_plan_policy`. ``extra_caveats`` carries what only the controller
    knows, which is the shape of the run rather than the content of the result.
    """
    settings = settings or ComposerSettings()
    result = _with_plan_policy(result, plan)

    lex = build_lexicon(result)
    labels = _label_index(result, lex)

    result_plan = result.get("plan") or {}
    question = question or result_plan.get("question") or ""
    kind = _answer_kind(result, diag)

    statements: List[Dict[str, Any]] = []
    caveats: List[str] = []

    quotes_skipped = 0
    path_review = PathReview()
    if kind == KIND_REFUSAL:
        statements += _refusal_statements(result)
    elif kind == KIND_CANDIDATES:
        statements += _candidate_statements(result, labels, settings)
        explanations, path_review = _explanation_statements(result, labels, settings)
        statements += explanations
        literature, quotes_skipped = _literature_statements(result, settings, lex)
        statements += literature
    elif kind == KIND_EXPLANATION:
        explanations, path_review = _explanation_statements(result, labels, settings)
        statements += explanations
        literature, quotes_skipped = _literature_statements(result, settings, lex)
        statements += literature
    elif kind == KIND_FILTERED_ABSENCE:
        statements += _filtered_absence_statements(result, labels)
    elif kind == KIND_ABSENCE:
        statements += _absence_statements(result, labels)
    elif kind == KIND_CLARIFICATION:
        statements += _clarification_statements(result)
    else:
        statements += _inconclusive_statements(result, diag)

    caveats += _caveats(result, diag)
    caveats += [str(c) for c in (extra_caveats or []) if str(c).strip()]

    answer: Dict[str, Any] = {
        "schema": ANSWER_SCHEMA_VERSION,
        "question": question,
        "answer_kind": kind,
        "statements": statements,
        "caveats": caveats,
        "candidates": _candidate_rows(result, settings),
        "provenance": _provenance(result, lex, iterations),
    }

    report = check_answer(answer, lex, allow_quotes=settings.include_quotes)
    if not report["ok"]:
        failed = {v["statement_id"] for v in report["violations"]}
        answer["withheld_statements"] = [
            s for s in statements if str(s.get("id")) in failed
        ]
        answer["statements"] = [
            s for s in statements if str(s.get("id")) not in failed
        ]
    answer["grounding"] = report

    unclassified = unquarantined_prose_fields(result)
    if unclassified:
        answer["grounding"]["unclassified_prose_fields"] = unclassified

    if path_review.withheld:
        # Not a grounding failure: every edge in these chains is in the result
        # graph. They are withheld because a chain of true edges can still fail
        # to be support — it can reach the destination through a
        # contraindication, or connect two things without explaining anything.
        answer["grounding"]["explanation_paths"] = path_review.to_dict()
        caveats.append(
            f"{len(path_review.withheld)} mechanistic path(s) linking these "
            f"candidates to the question's concept were found and not shown, "
            f"because the chain does not support the candidate: "
            f"{path_review.counts}. They are real paths in the graph; they are "
            f"not reasons."
        )

    if quotes_skipped:
        # Not a failure: these are verified quotes about edges that did not
        # survive into the answer. Counted because a large number means
        # literature was checked for candidates that were then discarded, which
        # is wasted model time worth noticing.
        answer["grounding"]["quotes_skipped_off_answer_edges"] = quotes_skipped

    answer["composed_by"] = (
        "deterministic template renderer; no model-authored text is read or "
        "emitted, and every statement was checked against the result graph"
    )
    return answer


def _with_plan_policy(
    result: Dict[str, Any], plan: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Put the executed plan's evidence policy back into the result document.

    The executor's result carries a plan block, but a partial one: plan_id, the
    versions, the question, the mode, the interpretation, the confidence and
    the gaps. Not `evidence_policy`. That is a reasonable thing for the
    executor to leave out — it is the thing that *applied* the policy, and its
    filter accounting already records what the policy did.

    It is not reasonable here, because of one answer kind. A `filtered_absence`
    exists to say that the graph held matches and the user's own evidence
    filters removed all of them, and the only useful content of that sentence
    is *which* filters. Without the policy the composer could say that filters
    removed everything and then fall silent about what they were, which is the
    least helpful form of an already unwelcome answer.

    Only the one key, and only when the result does not already carry it: the
    result's plan block is the executor's record of what it actually ran, and
    overwriting any of it with what the controller believes it sent would turn
    a record into an assumption. The lexicon reads the same field, so grafting
    it here also lets the filter's knowledge sources be named — they appear in
    no returned edge, having removed every one of them.
    """
    if not plan:
        return result
    policy = plan.get("evidence_policy")
    if not policy:
        return result

    result_plan = dict(result.get("plan") or {})
    if result_plan.get("evidence_policy"):
        return result

    result_plan["evidence_policy"] = policy
    grafted = dict(result)
    grafted["plan"] = result_plan
    return grafted


def _answer_kind(result: Dict[str, Any], diag: Optional[Diagnosis]) -> str:
    """Which answer this run licenses.

    The anchor check is repeated here rather than left to the policy layer, and
    the repetition is the point. Policy decides what the *loop* may do; this
    decides what the *document* may say, and they are reached by different
    routes — `compose` is called directly by tools and tests, and a loop that
    abandons for an unrelated reason still ends up here with the same empty
    result in hand. An absence is the one answer kind that makes a claim about
    the world on the strength of nothing coming back, so it is the one that has
    to be unavailable in both places.
    """
    outcome = (result.get("outcome") or {}).get("outcome")
    if result.get("refusal") or outcome == OUTCOME_REFUSED:
        return KIND_REFUSAL
    if result.get("results"):
        mode = (result.get("plan") or {}).get("plan_mode")
        if mode == "explanation":
            return KIND_EXPLANATION
        return KIND_CANDIDATES
    if _has_explanation_paths(result):
        return KIND_EXPLANATION
    if outcome == OUTCOME_UNRESOLVED and _unresolved_anchors(result):
        # A name that did not ground is the one dead end where the system knows
        # something useful to say back: what it searched for, and what it found
        # instead. Answering "no answer was established" would be true and
        # would throw that away.
        return KIND_CLARIFICATION
    if diag is not None and diag.anchor_category_mismatch:
        return KIND_INCONCLUSIVE
    if outcome == OUTCOME_FILTERED_OUT:
        return KIND_FILTERED_ABSENCE
    if outcome in (OUTCOME_NO_DATA, OUTCOME_JOIN_FAILURE):
        return KIND_ABSENCE
    return KIND_INCONCLUSIVE


def _has_explanation_paths(result: Dict[str, Any]) -> bool:
    """Whether the run produced explanation paths, wherever they were written."""
    return any(collect_explanation_paths(result).values())


# ---------------------------------------------------------------------------
# Labels
# ---------------------------------------------------------------------------


def _label_index(result: Dict[str, Any], lex: Lexicon) -> Dict[str, str]:
    """CURIE to preferred label, from what the graph returned.

    Not from the plan. The plan carries the surface name the user or the
    planner wrote; the result carries what the identifier actually resolved to,
    and those differ in exactly the cases where the difference matters.
    """
    index: Dict[str, str] = {}

    for record in (result.get("resolution") or {}).values():
        label = record.get("resolved_label")
        for curie in record.get("resolved_curies") or []:
            if label:
                index[str(curie)] = str(label)

    for check in result.get("concept_checks") or []:
        observed = check.get("observed_curies") or []
        labels = check.get("observed_labels") or []
        for curie, label in zip(observed, labels):
            index.setdefault(str(curie), str(label))

    explanations = (result.get("evidence") or {}).get("explanations") or {}
    for paths in (explanations.get("paths_by_endpoint") or {}).values():
        for path in paths or []:
            for curie, label in zip(path.get("nodes") or [], path.get("labels") or []):
                index.setdefault(str(curie), str(label))

    for candidate in result.get("results") or []:
        if candidate.get("curie") and candidate.get("label"):
            index[str(candidate["curie"])] = str(candidate["label"])

    return index


def _name(curie: Optional[str], labels: Dict[str, str]) -> str:
    """Render an entity as 'label [CURIE]', or just the CURIE.

    Both, always, when a label is known: the label is what a reader
    understands, the identifier is what makes the claim checkable, and the
    gate only has purchase on the identifier.
    """
    if not curie:
        return "an unnamed entity"
    label = labels.get(str(curie))
    return f"{label} [{curie}]" if label else str(curie)


# Predicates are rendered verbatim inside an arrow — `—[biolink:treats]→` —
# rather than converted to English. A predicate name does not reliably become a
# verb in the right direction, and a fluent sentence that quietly inverts
# subject and object is exactly the failure this architecture exists to make
# impossible. The arrow is uglier and cannot be read backwards.


# ---------------------------------------------------------------------------
# Candidates
# ---------------------------------------------------------------------------


def _candidate_statements(
    result: Dict[str, Any], labels: Dict[str, str], settings: ComposerSettings,
) -> List[Dict[str, Any]]:
    candidates = (result.get("results") or [])[: settings.top_k]
    total = len(result.get("results") or [])
    statements: List[Dict[str, Any]] = []

    shown = len(candidates)
    headline = (
        f"The knowledge graph returned {total} candidate"
        f"{'' if total == 1 else 's'}"
    )
    if shown < total:
        headline += f"; the {shown} highest ranked are listed"
    statements.append(_statement("S1", "summary", headline + "."))

    for index, candidate in enumerate(candidates, start=1):
        statements.append(_candidate_statement(index, candidate, labels, settings))

    return statements


def _candidate_statement(
    index: int,
    candidate: Dict[str, Any],
    labels: Dict[str, str],
    settings: ComposerSettings,
) -> Dict[str, Any]:
    """One row: the entity, the assertions behind it, and who asserted them.

    Assertions are grouped by the triple they state, not listed per edge. A
    knowledge graph commonly holds the same triple from six sources, and six
    near-identical clauses read as six findings when they are one finding with
    six attestations — which is also the more useful fact, since agreement
    across independent sources is what a reader is weighing.

    The triple is rendered as an arrow rather than as a sentence. Predicate
    names do not reliably become English verbs in the right direction, and a
    plausible-sounding sentence that inverts subject and object is exactly the
    kind of error this architecture is built to make impossible.
    """
    curie = candidate.get("curie")
    edges = _candidate_edges(candidate)
    groups = _group_edges(edges, curie)[: settings.max_edges_per_candidate]

    parts = [f"{index}. {_name(curie, labels)}"]

    for group in groups:
        sources = ", ".join(group["sources"]) or "an unnamed source"
        detail = f"asserted by {sources}"
        if group["knowledge_levels"]:
            detail += f"; knowledge level {', '.join(group['knowledge_levels'])}"
        if group["publications"]:
            count = len(group["publications"])
            detail += (
                f"; {count} publication{'' if count == 1 else 's'} "
                f"({', '.join(sorted(group['publications'])[:3])}"
                + (" …" if count > 3 else "") + ")"
            )
        parts.append(
            f"\n     {_name(group['subject'], labels)} "
            f"—[{group['predicate']}]→ {_name(group['object'], labels)} "
            f"({detail})"
        )
        if group["negated"]:
            parts.append(
                "\n     NOTE: this edge is negated — the source asserts the "
                "relationship does NOT hold"
            )

    counts = []
    if candidate.get("num_supporting_paths"):
        counts.append(
            f"{candidate['num_supporting_paths']} supporting path"
            f"{'' if candidate['num_supporting_paths'] == 1 else 's'}"
        )
    remaining = len(_group_edges(edges, curie)) - len(groups)
    if remaining > 0:
        counts.append(f"{remaining} further assertion(s) not shown")
    if counts:
        parts.append(f"\n     ({'; '.join(counts)})")

    return _statement(
        f"C{index}", "candidate", "".join(parts),
        supported_by=[
            edge_id for group in groups for edge_id in group["edge_ids"]
        ],
        subject=curie,
    )


def _group_edges(
    edges: Sequence[Dict[str, Any]], curie: Optional[str],
) -> List[Dict[str, Any]]:
    """Collapse edges stating the same triple, keeping every attestation.

    Ordered by how many sources assert the triple, so the best-attested
    statement about a candidate is the one a reader sees first.
    """
    grouped: Dict[Tuple[Any, Any, Any], Dict[str, Any]] = {}
    for edge in edges:
        key = (edge.get("subject"), edge.get("predicate"), edge.get("object"))
        group = grouped.setdefault(key, {
            "subject": edge.get("subject"),
            "predicate": edge.get("predicate"),
            "object": edge.get("object"),
            "sources": [],
            "knowledge_levels": [],
            "publications": set(),
            "edge_ids": [],
            "negated": False,
        })
        source = edge.get("primary_source")
        if source and source not in group["sources"]:
            group["sources"].append(str(source))
        level = edge.get("knowledge_level")
        if level and level not in group["knowledge_levels"]:
            group["knowledge_levels"].append(str(level))
        for pmid in edge.get("publications") or []:
            group["publications"].add(str(pmid))
        if edge.get("edge_id"):
            group["edge_ids"].append(str(edge["edge_id"]))
        if edge.get("negated"):
            group["negated"] = True

    return sorted(
        grouped.values(),
        key=lambda g: (-len(g["sources"]), str(g["predicate"])),
    )


def _candidate_edges(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    edges: List[Dict[str, Any]] = []
    for instances in (candidate.get("evidence_by_path") or {}).values():
        for instance in instances or []:
            for edge in instance.get("edges") or []:
                edges.append(edge)
    return edges


def _candidate_rows(
    result: Dict[str, Any], settings: ComposerSettings,
) -> List[Dict[str, Any]]:
    """The structured table behind the sentences.

    Deliberately excludes `rerank_reason` and every other model-authored field,
    even though this block is data rather than prose: a consumer rendering the
    table would put whatever is here in front of a reader.
    """
    rows: List[Dict[str, Any]] = []
    for candidate in (result.get("results") or [])[: settings.top_k]:
        edges = _candidate_edges(candidate)
        rows.append({
            "rank": candidate.get("rank"),
            "curie": candidate.get("curie"),
            "label": candidate.get("label"),
            "categories": candidate.get("categories") or [],
            "score": candidate.get("score"),
            "num_supporting_paths": candidate.get("num_supporting_paths"),
            "mean_epc": candidate.get("mean_epc"),
            "knowledge_levels": sorted({
                e["knowledge_level"] for e in edges if e.get("knowledge_level")
            }),
            "primary_sources": sorted({
                e["primary_source"] for e in edges if e.get("primary_source")
            }),
            "publications": sorted({
                pmid for e in edges for pmid in (e.get("publications") or [])
            })[:20],
            "supporting_edge_ids": [
                e["edge_id"] for e in edges if e.get("edge_id")
            ][:20],
            "reranked": candidate.get("deterministic_rank") != candidate.get("rank"),
        })
    return rows


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------


def _explanation_statements(
    result: Dict[str, Any], labels: Dict[str, str], settings: ComposerSettings,
) -> Tuple[List[Dict[str, Any]], PathReview]:
    """Mechanistic paths, rebuilt here rather than copied.

    The executor writes a `description` for each path and it is deterministic,
    but rebuilding it from `nodes`, `labels` and `predicates` means composition
    depends on the structured fields the gate can check rather than on a string
    it cannot.
    """
    explanations = (result.get("evidence") or {}).get("explanations") or {}
    by_endpoint = collect_explanation_paths(result)
    statements: List[Dict[str, Any]] = []
    counter = 0
    review_total = PathReview()

    for endpoint, paths in by_endpoint.items():
        review = review_paths(paths)
        review_total.servable.extend(review.servable)
        review_total.withheld.extend(review.withheld)
        for path in review.servable[: settings.max_explanations]:
            counter += 1
            nodes = path.get("nodes") or []
            predicates = path.get("predicates") or []
            if not nodes:
                continue

            chain: List[str] = [_name(nodes[0], labels)]
            for i, predicate in enumerate(predicates):
                if i + 1 >= len(nodes):
                    break
                chain.append(f"—[{predicate}]→")
                chain.append(_name(nodes[i + 1], labels))

            edge_ids = [
                e["edge_id"] for e in (path.get("edges") or []) if e.get("edge_id")
            ]
            sources = sorted({
                e["primary_source"] for e in (path.get("edges") or [])
                if e.get("primary_source")
            })

            text = (
                f"{_name(nodes[0], labels)} and {_name(nodes[-1], labels)} are "
                f"connected by a path of length "
                f"{path.get('length', len(predicates))}: " + " ".join(chain)
            )
            if sources:
                text += f". Asserted by: {', '.join(sources)}"

            statements.append(_statement(
                f"E{counter}", "explanation", text + ".",
                supported_by=edge_ids, subject=str(endpoint),
            ))

    for summary in explanations.get("summaries") or []:
        if summary.get("candidates_with_paths") == 0 and summary.get("endpoint_b"):
            counter += 1
            statements.append(_statement(
                f"E{counter}", "explanation_absence",
                f"No connecting path was found to "
                f"{_name(summary['endpoint_b'], labels)} under this query's "
                f"hop limit and constraints, across "
                f"{summary.get('candidates_attempted', 0)} candidate(s) "
                f"attempted.",
            ))

    return statements, review_total


# ---------------------------------------------------------------------------
# Literature
# ---------------------------------------------------------------------------


def _literature_statements(
    result: Dict[str, Any],
    settings: ComposerSettings,
    lex: Lexicon,
) -> Tuple[List[Dict[str, Any]], int]:
    """Verified quotes, for edges that are actually in the answer.

    The executor's rule is that a literature verdict requires a verbatim quote
    it can find in the retrieved abstract, because a citation alone does not
    show the cited paper supports the edge. Only quotes that passed that check
    are rendered, and the quote is reproduced rather than characterised: the
    abstract's words are evidence, a summary of them would not be.

    The second filter is the one a live run caught. `evidence.literature.
    by_edge` covers **every edge the executor checked**, and that is a wider set
    than the edges attached to the candidates that survived ranking and
    filtering. Quoting an abstract about an edge that is not in the answer
    offers evidence for a claim the answer is not making — and it was producing
    three violations at once, since the edge id, its PMID and its quote are all
    absent from a lexicon built from surviving edges.

    Skipped quotes are counted rather than silently dropped. If many are being
    skipped, real evidence is being lost somewhere upstream and that is worth
    seeing.
    """
    if not settings.include_quotes:
        return [], 0

    by_edge = ((result.get("evidence") or {}).get("literature") or {}).get("by_edge") or {}
    statements: List[Dict[str, Any]] = []
    counter = 0
    skipped = 0

    for edge_id, support in by_edge.items():
        if str(edge_id) not in lex.edge_ids:
            skipped += sum(
                1 for v in support.get("verdicts") or []
                if v.get("quote_verified") and v.get("quote")
                and str(v.get("verdict", "")).startswith("supports")
            )
            continue

        for verdict in support.get("verdicts") or []:
            if not verdict.get("quote_verified") or not verdict.get("quote"):
                continue
            if not str(verdict.get("verdict", "")).startswith("supports"):
                continue
            counter += 1
            if counter > settings.max_quotes:
                return statements, skipped
            pmid = verdict.get("pmid") or "an unrecorded PMID"
            statements.append(_statement(
                f"L{counter}", "literature",
                f"{pmid} states: \"{verdict['quote']}\"",
                supported_by=[edge_id],
            ))
    return statements, skipped


# ---------------------------------------------------------------------------
# Absence, refusal, inconclusive
# ---------------------------------------------------------------------------


def _absence_statements(
    result: Dict[str, Any], labels: Dict[str, str],
) -> List[Dict[str, Any]]:
    """State what was established, not merely that nothing was found.

    The distinction is the point of the whole outcome taxonomy. "No results"
    covers a timeout and an exhaustive empty traversal, and only one of them
    says anything about the world. This branch is reached only for the second,
    so the sentence can be about the graph — and it is bounded to the query as
    written, because that is all that was actually checked.
    """
    outcome = result.get("outcome") or {}
    statements = [_statement(
        "S1", "absence",
        "The query ran to completion and the knowledge graph returned no "
        "matching results. This is a statement about the graph, not a failure "
        "of the search: the queried relationships were traversed and nothing "
        "satisfied them.",
    )]

    anchors = _anchor_names(result, labels)
    query_lines = _query_description(result, labels)
    if query_lines:
        statements.append(_statement(
            "S2", "query_description",
            "The query that returned nothing was: " + "; ".join(query_lines) + ".",
        ))
    if anchors:
        statements.append(_statement(
            "S3", "anchors",
            "It was anchored on " + ", ".join(anchors) + ".",
        ))

    if outcome.get("outcome") == OUTCOME_JOIN_FAILURE:
        statements.append(_statement(
            "S4", "absence_detail",
            "Each hop returned data on its own, but no single intermediate "
            "entity connected them. The individual relationships exist in the "
            "graph; the chain the question asks about does not.",
        ))

    return statements


def _filtered_absence_statements(
    result: Dict[str, Any], labels: Dict[str, str],
) -> List[Dict[str, Any]]:
    """The narrower claim: nothing survived the user's own filter."""
    accounting = (result.get("evidence") or {}).get("filter_accounting") or {}
    dropped = sum(int(a.get("instances_dropped") or 0) for a in accounting.values())

    statements = [_statement(
        "S1", "filtered_absence",
        f"The graph returned matching results, and the evidence filters "
        f"requested in the question removed all of them "
        f"({dropped} instance(s) dropped). The graph is not empty here; "
        f"nothing in it met the evidence standard that was asked for.",
    )]

    policy = _policy_description(result)
    if policy:
        statements.append(_statement(
            "S2", "policy_description",
            "The filters applied were: " + "; ".join(policy) + ".",
        ))

    reasons: Dict[str, int] = {}
    for acc in accounting.values():
        for reason, count in (acc.get("instance_drop_reasons") or {}).items():
            reasons[str(reason)] = reasons.get(str(reason), 0) + int(count or 0)
    if reasons:
        ordered = sorted(reasons.items(), key=lambda kv: -kv[1])
        statements.append(_statement(
            "S3", "filter_accounting",
            "Removals by rule: "
            + ", ".join(f"{name} ({count})" for name, count in ordered) + ".",
        ))

    return statements


def _unresolved_anchors(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Fixed entities whose name did not ground, with what was searched for."""
    out: List[Dict[str, Any]] = []
    for ref, record in (result.get("resolution") or {}).items():
        if not isinstance(record, dict) or record.get("is_variable"):
            continue
        if record.get("resolved_curies"):
            continue
        out.append({
            "entity_ref": ref,
            "query": record.get("query") or ref,
            "considered": record.get("considered") or [],
        })
    return out


def _clarification_statements(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Say which term failed, show what the search returned, and ask.

    This is the one dead end where the system is holding something the user can
    act on. A name that did not ground means no query ran, so there is nothing
    to report about the graph — but the resolver did look the name up, and what
    came back is a list of real entries with real identifiers, sitting in the
    result document.

    Two things are deliberately not said. The resolver's own account of why it
    rejected the candidates is model-written and quarantined, so nothing here
    reproduces it. And no entry is described as similar, related, or likely:
    none of them was queried, so the only honest claim is that the search
    returned them. The sentence says that and stops.
    """
    unresolved = _unresolved_anchors(result)
    if not unresolved:  # pragma: no cover - guarded by `_answer_kind`
        return []

    terms = ", ".join(f"'{u['query']}'" for u in unresolved)
    plural = len(unresolved) > 1
    statements = [_statement(
        "S1", "clarification",
        f"No query was run. Building one requires matching every concept in "
        f"the question to an entry in the knowledge graph, and "
        f"{'these terms' if plural else 'this term'} could not be matched: "
        f"{terms}.",
    )]

    counter = 2
    for entry in unresolved:
        considered = entry["considered"]
        if not considered:
            statements.append(_statement(
                f"S{counter}", "clarification",
                f"Searching for '{entry['query']}' returned no nearby entries "
                f"at all, so the graph may not cover this concept under a name "
                f"close to the one used.",
            ))
            counter += 1
            continue

        shown = considered[:MAX_RESOLUTION_OPTIONS]
        listed = "; ".join(
            f"{c.get('label')} ({c.get('curie')})"
            for c in shown if c.get("curie")
        )
        more = len(considered) - len(shown)
        statements.append(_statement(
            f"S{counter}", "resolution_options",
            f"Searching for '{entry['query']}' returned "
            f"{len(considered)} nearby entries, and none was accepted as a "
            f"match for the term as written. The closest were: {listed}"
            + (f"; and {more} more." if more > 0 else "."),
        ))
        counter += 1

    statements.append(_statement(
        f"S{counter}", "clarification",
        "If one of these is the concept meant, asking again with that name in "
        "place of the original term will build a query against it. If none of "
        "them is, the graph may hold the concept under a different name, or "
        "may not cover it — naming a more specific condition is usually what "
        "works.",
    ))
    return statements


def _refusal_statements(result: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Rendered from the typed reason, never from the planner's message."""
    refusal = result.get("refusal") or {}
    reason = str(refusal.get("reason") or "")
    text = REFUSAL_TEXT.get(
        reason,
        "This question was declined before any query was built.",
    )
    statements = [_statement("S1", "refusal", text)]

    questions = refusal.get("clarifying_questions") or refusal.get("questions") or []
    if reason == "needs_clarification" and questions:
        # The questions themselves are planner-written prose. They are not
        # reproduced. What is reported is that clarification is needed and how
        # many points are open, which is a fact about the refusal.
        statements.append(_statement(
            "S2", "refusal_detail",
            f"{len(questions)} point(s) would need to be specified before a "
            f"query could be built.",
        ))
    return statements


def _inconclusive_statements(
    result: Dict[str, Any], diag: Optional[Diagnosis] = None,
) -> List[Dict[str, Any]]:
    """When nothing was established, say that, and say why not.

    The temptation here is to report the empty result as an absence, which
    would be the most useful-sounding thing available and would be a claim the
    run does not support. The sentence says the opposite in as many words.
    """
    outcome = result.get("outcome") or {}
    detail = outcome.get("detail") or ""
    statements = [_statement(
        "S1", "inconclusive",
        "No answer was established. The run did not complete in a way that "
        "would support either a positive answer or a statement that the graph "
        "contains nothing.",
    )]
    if detail:
        statements.append(_statement(
            "S2", "inconclusive_detail",
            f"Reported by the query executor as: {outcome.get('outcome')} — {detail}.",
        ))
    if outcome.get("outcome") == OUTCOME_TRUNCATED:
        statements.append(_statement(
            "S3", "inconclusive_detail",
            "A query did not finish, so the absence of results here is not "
            "evidence that the graph lacks the data.",
        ))

    # Named without its Biolink category, deliberately: the category the plan
    # pinned appears in no returned edge — there are none — and this statement
    # is about the answer rather than the query, so it is checked strictly. The
    # identifier is enough to make the problem findable, and the category is in
    # the caveats and the trace.
    for mismatch in (diag.anchor_category_mismatch if diag else []):
        statements.append(_statement(
            f"S{len(statements) + 1}", "inconclusive_detail",
            f"The question's anchor was searched for under a Biolink category "
            f"that {mismatch.get('curie')} is not recorded under, so this "
            f"query could not have matched anything whatever the graph "
            f"contains. That is a fault in the plan, not a finding about the "
            f"data.",
        ))

    return statements


# ---------------------------------------------------------------------------
# Query and policy description
# ---------------------------------------------------------------------------


def _query_description(
    result: Dict[str, Any], labels: Dict[str, str],
) -> List[str]:
    """The executed query, from the submitted TRAPI graphs.

    Read from `paths[].submitted_queries` rather than from the plan, because
    the plan says what was asked for and the submitted query says what was
    sent. When a run decomposed a multi-hop query, those differ.
    """
    lines: List[str] = []
    for path_id, path in (result.get("paths") or {}).items():
        for submitted in (path.get("submitted_queries") or [])[:2]:
            graph = submitted.get("query_graph") or {}
            edges = graph.get("edges") or {}
            nodes = graph.get("nodes") or {}
            for edge in edges.values():
                subject = _node_description(nodes.get(edge.get("subject")), labels)
                obj = _node_description(nodes.get(edge.get("object")), labels)
                predicates = edge.get("predicates") or []
                relation = ", ".join(predicates) if predicates else "any relation"
                lines.append(f"{path_id}: {subject} —[{relation}]→ {obj}")
    seen, out = set(), []
    for line in lines:
        if line not in seen:
            seen.add(line)
            out.append(line)
    return out[:6]


def _node_description(node: Optional[Dict[str, Any]], labels: Dict[str, str]) -> str:
    if not node:
        return "an unspecified node"
    ids = node.get("ids") or []
    if ids:
        return ", ".join(_name(str(i), labels) for i in ids[:3])
    categories = node.get("categories") or []
    if categories:
        return f"any {', '.join(str(c) for c in categories[:3])}"
    return "any entity"


def _policy_description(result: Dict[str, Any]) -> List[str]:
    """The user's evidence filters, from their typed fields.

    Not from `evidence_policy.rationale`, which restates the user's request in
    the planner's words and is quarantined for that reason.
    """
    policy = (result.get("plan") or {}).get("evidence_policy") or {}
    if not isinstance(policy, dict):
        return []
    lines: List[str] = []
    if policy.get("required_knowledge_sources"):
        lines.append(
            f"edges must come from {', '.join(policy['required_knowledge_sources'])}"
        )
    if policy.get("excluded_knowledge_sources"):
        lines.append(
            f"edges from {', '.join(policy['excluded_knowledge_sources'])} "
            f"are excluded"
        )
    if policy.get("require_primary_knowledge_source"):
        lines.append("edges must declare a primary knowledge source")
    if policy.get("min_publications") is not None:
        lines.append(f"at least {policy['min_publications']} publication(s) per edge")
    if policy.get("min_year") is not None:
        lines.append(f"publications from {policy['min_year']} or later")
    if policy.get("agent_type"):
        lines.append(f"agent type in {', '.join(policy['agent_type'])}")
    return lines


def _anchor_names(result: Dict[str, Any], labels: Dict[str, str]) -> List[str]:
    names: List[str] = []
    for record in (result.get("resolution") or {}).values():
        if record.get("is_variable"):
            continue
        for curie in record.get("resolved_curies") or []:
            names.append(_name(str(curie), labels))
    return names[:6]


# ---------------------------------------------------------------------------
# Caveats
# ---------------------------------------------------------------------------


def _caveats(result: Dict[str, Any], diag: Optional[Diagnosis]) -> List[str]:
    """What a reader needs in order to weigh the answer.

    Attached to every answer that has them, whatever the loop decided. A
    concept-check failure in particular is not a footnote — it means the
    results may concern a different concept than the question asked about — so
    it is stated first and in plain terms.
    """
    caveats: List[str] = []

    if result.get("concept_warning"):
        caveats.append(
            "A concept check did not confirm that the results concern the "
            "intended entity. They may describe a related but different "
            "concept; the resolved identifiers are listed in the provenance "
            "block and should be checked before the answer is relied on."
        )

    for path_id, path in (result.get("paths") or {}).items():
        if not path.get("coverage_complete", True):
            caveats.append(
                f"Coverage of path {path_id} was incomplete, so these results "
                f"are a partial view rather than everything the graph holds."
            )

    distribution = (result.get("evidence") or {}).get("distribution_pre_filter") or {}
    for path_id, dist in distribution.items():
        pct = dist.get("pct_knowledge_level_provided")
        if pct is not None and pct < 30:
            caveats.append(
                f"On path {path_id}, only {pct}% of edges declare a knowledge "
                f"level, so ranking terms that use it rest on sparse metadata."
            )
        pubs = dist.get("pct_with_publications")
        if pubs is not None and pubs < 10:
            caveats.append(
                f"On path {path_id}, {pubs}% of edges carry publications; most "
                f"support here is curated assertion rather than cited "
                f"literature."
            )

    reranked = [
        c for c in result.get("results") or []
        if c.get("rerank_reason")
    ]
    if reranked:
        grounded = sum(1 for c in reranked if c.get("rerank_grounded"))
        caveats.append(
            f"The order of {len(reranked)} candidate(s) was adjusted by a "
            f"model whose stated reasons were checked against the retrieved "
            f"edges ({grounded} passed). The reasons themselves are recorded "
            f"in the execution result and are not reproduced here."
        )

    literature = ((result.get("evidence") or {}).get("literature") or {}).get("summary") or {}
    if literature.get("edges_checked"):
        caveats.append(
            f"Literature was checked for {literature['edges_checked']} edge(s); "
            f"only quotes verified present in the retrieved abstract are shown."
        )

    if diag and diag.filters_dropped_all:
        caveats.append(
            "The evidence filters requested in the question were applied as "
            "given and were not weakened."
        )

    caveats += _anchor_caveats(diag)
    caveats += _orphan_caveats(diag)
    return caveats


def _orphan_caveats(diag: Optional[Diagnosis]) -> List[str]:
    """A concept from the question that the query never used.

    First in nothing and last in the list, but the most important caveat here,
    because it is the only one describing a problem invisible everywhere else.
    A concept-check warning, thin annotation, an incomplete path — each of
    those leaves a mark on the result. This one does not: the run succeeds, the
    candidates are real, every statement about them passes the grounding gate,
    and the answer is to a smaller question. Nothing but this sentence tells
    the reader.
    """
    if diag is None or not diag.orphan_entities:
        return []

    named = ", ".join(
        str(o.get("name") or o.get("entity_ref"))
        for o in diag.orphan_entities[:4]
    )
    return [
        f"This answer does not account for {named}, which the question "
        f"mentions. The plan named the concept but built no query step that "
        f"used it, so the results below answer a narrower question: they are "
        f"supported by real evidence for what was asked, and that is less than "
        f"what was asked for."
    ]


def _anchor_caveats(diag: Optional[Diagnosis]) -> List[str]:
    """Doubt about the concepts the question is anchored on.

    Worth saying whatever the answer turned out to be, but not in the same
    words. On an empty run the mismatch is the likely cause: the node
    constraint may have excluded everything for a reason about the plan rather
    than the graph, which is why an absence cannot be claimed. On a run that
    returned results that sentence is simply false — the constraint plainly
    matched something — and printing it anyway would tell the reader to
    distrust an answer for a reason the same document disproves. What is left
    to say there is smaller and still true: the plan's label for the concept
    disagrees with the identifier it resolved to, so check that the identifier
    is the one the question meant.
    """
    if diag is None:
        return []
    caveats: List[str] = []

    for mismatch in diag.anchor_category_mismatch:
        opening = (
            f"The question's anchor '{mismatch.get('entity_ref')}' was searched "
            f"for as {mismatch.get('expected_category')}, but the identifier it "
            f"resolved to ({mismatch.get('curie')}) is not recorded under that "
            f"category."
        )
        if diag.num_results:
            caveats.append(
                f"{opening} The query still returned results, so the category "
                f"did not prevent a match; what it does mean is that the plan "
                f"described this concept in terms the identifier does not "
                f"carry, which is worth checking against the question."
            )
        else:
            caveats.append(
                f"{opening} The query may therefore have been unable to match "
                f"anything for a reason unrelated to what the graph contains."
            )

    low = sorted(
        ref for ref, level in diag.anchor_confidence.items() if level == "low"
    )
    if low:
        caveats.append(
            f"The resolver had low confidence in the identifier chosen for "
            f"{', '.join(low)}. The answer is about whatever that identifier "
            f"names, which may not be what the question meant."
        )

    return caveats


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def _provenance(
    result: Dict[str, Any], lex: Lexicon, iterations: int,
) -> Dict[str, Any]:
    resolution = {}
    for ref, record in (result.get("resolution") or {}).items():
        if record.get("is_variable"):
            continue
        resolution[ref] = {
            "query": record.get("query"),
            "resolved_curies": record.get("resolved_curies") or [],
            "resolved_label": record.get("resolved_label"),
            "confidence": record.get("confidence"),
            "chosen_by_model": bool(record.get("llm_used")),
            "alternatives_considered": len(record.get("considered") or []),
        }

    return {
        "executor_schema": result.get("schema"),
        "biolink_version": (result.get("plan") or {}).get("biolink_version"),
        "plan_version": (result.get("plan") or {}).get("plan_version"),
        "plan_mode": (result.get("plan") or {}).get("plan_mode"),
        "loop_iterations": iterations,
        "verdict": result.get("verdict"),
        "outcome": (result.get("outcome") or {}).get("outcome"),
        "entity_resolution": resolution,
        "knowledge_sources": sorted(lex.sources)[:50],
        "graph": lex.stats(),
        "model_assisted_steps": [
            "entity disambiguation (choice recorded per entity, with "
            "alternatives)",
            "candidate reranking (reasons required to cite retrieved edges; "
            "ungrounded reasons discarded)",
            "literature quote selection (quotes verified present in the "
            "retrieved abstract before use)",
        ],
    }


# ---------------------------------------------------------------------------
# Statements and rendering
# ---------------------------------------------------------------------------


def _statement(
    sid: str,
    kind: str,
    text: str,
    *,
    supported_by: Optional[Sequence[str]] = None,
    subject: Optional[str] = None,
) -> Dict[str, Any]:
    statement: Dict[str, Any] = {
        "id": sid,
        "kind": kind,
        "text": text,
        "supported_by": [str(e) for e in (supported_by or [])],
    }
    if subject:
        statement["subject"] = subject
    return statement


def render_text(answer: Dict[str, Any]) -> str:
    """A plain-text rendering of the answer document.

    A view of the statements, not a second composition: it concatenates text
    that already passed the gate and adds no sentence of its own beyond fixed
    headings. Anything that could not be said in a checked statement is not
    said here either.
    """
    lines: List[str] = []
    question = answer.get("question")
    if question:
        lines.append(f"Question: {question}")
        lines.append("")

    for statement in answer.get("statements") or []:
        lines.append(statement.get("text", ""))
        if statement.get("supported_by"):
            lines.append(
                f"    supported by {len(statement['supported_by'])} edge(s): "
                f"{', '.join(statement['supported_by'][:3])}"
                + (" …" if len(statement["supported_by"]) > 3 else "")
            )
    lines.append("")

    caveats = answer.get("caveats") or []
    if caveats:
        lines.append("Caveats")
        for caveat in caveats:
            lines.append(f"  - {caveat}")
        lines.append("")

    grounding = answer.get("grounding") or {}
    # A clarification has no result graph to have been checked against — no
    # query ran. Its identifiers were checked against what the resolver
    # retrieved, and saying otherwise would overstate what was verified in the
    # one line a reader is most likely to trust.
    checked_against = (
        "the entries the resolver retrieved"
        if answer.get("answer_kind") == KIND_CLARIFICATION
        else "the result graph"
    )
    lines.append(
        f"Grounding: {grounding.get('passed', 0)}/{grounding.get('checked', 0)} "
        f"statements verified against {checked_against}"
        + (
            f"; {grounding.get('failed')} withheld"
            if grounding.get("failed") else ""
        )
    )
    provenance = answer.get("provenance") or {}
    if provenance.get("knowledge_sources"):
        lines.append(
            f"Sources: {', '.join(provenance['knowledge_sources'][:10])}"
            + (" …" if len(provenance["knowledge_sources"]) > 10 else "")
        )
    return "\n".join(lines)
