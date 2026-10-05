"""Closing hops in a decomposed path must pin both ends.

The case that surfaced it, a star-shaped path with two anchors:

    drug -[directly_physically_interacts_with]-> NCBIGene:1080   (CFTR)
    drug -[treats]-> MONDO:0015614                               (dermatitis herpetiformis)

Run directly, this is one TRAPI graph with the drug node shared, and ARAX does
the intersection. It came back empty, which was the right answer. The executor
then decomposed the path to locate the break: hop 0 pinned CFTR and found 13
drugs, and hop 1 — a *closing* hop, both ends already known — was sent as
`drugs -treats-> ?` with the disease left open. The join bound the disease
anchor to whatever each drug treats, and the path reported `success` with nine
candidates, every one of them bound to a disease other than the anchor
(`dermatitis_herpetiformis: UMLS:C0392164`, among others). A warning even
blamed the combined query for the difference.

`plan_decomposition` always documented a closing hop as constraining an
existing candidate set; the executor never passed the far end's ids. These
tests pin the fix: a closing hop is queried with both ends pinned, and the join
refuses to bind an anchor to anything the resolver did not choose.

No network: `GraphDouble` answers TRAPI one-hop and multi-hop queries from a
small in-memory edge list, honouring node `ids` the way ARAX does.
"""

import json
from types import SimpleNamespace

import pytest

from plan_executor.arax_client import (
    AraxResponse, OUTCOME_SUCCESS, OUTCOME_EMPTY, OUTCOME_TIMEOUT,
)
from plan_executor.executor import PathExecutor, StepResult
from plan_executor.trapi_builder import HopStep, plan_decomposition


CFTR = "NCBIGene:1080"
DH = "MONDO:0015614"
OTHER_DISEASE = "UMLS:C0392164"

INTERACTS = "biolink:directly_physically_interacts_with"
TREATS = "biolink:treats"

# DRUG:A is the only real answer: it binds CFTR and treats DH.
# DRUG:B binds CFTR but treats a different disease — the false positive.
# DRUG:C treats DH but does not bind CFTR.
GRAPH = {
    INTERACTS: {("DRUG:A", CFTR), ("DRUG:B", CFTR)},
    TREATS: {("DRUG:A", DH), ("DRUG:B", OTHER_DISEASE), ("DRUG:C", DH)},
}
# The live case: no CFTR binder treats DH, so the direct query is empty.
GRAPH_NO_ANSWER = {
    INTERACTS: {("DRUG:B", CFTR)},
    TREATS: {("DRUG:B", OTHER_DISEASE), ("DRUG:C", DH)},
}


# --- doubles ---------------------------------------------------------------


def _entity(category, variable):
    return SimpleNamespace(
        biolink_category=category, is_variable=variable, constraints=None,
    )


def _hop(subject, obj, predicate):
    return SimpleNamespace(
        subject_ref=subject, object_ref=obj, predicate=predicate,
        qualifiers=None, predicate_expansion=None, negated=False,
    )


ENTITIES = {
    "candidate_drug": _entity("Drug", True),
    "cftr": _entity("Gene", False),
    "dermatitis_herpetiformis": _entity("Disease", False),
}
STAR = SimpleNamespace(
    path_id="P1",
    hops=[
        _hop("candidate_drug", "cftr", INTERACTS),
        _hop("candidate_drug", "dermatitis_herpetiformis", TREATS),
    ],
    return_entity_ref="candidate_drug",
    disabled=False,
)
RESOLUTIONS = {"cftr": [CFTR], "dermatitis_herpetiformis": [DH]}


class GraphDouble:
    """Answers a TRAPI query graph from an edge list, honouring pinned ids."""

    def __init__(self, graph, direct="answer", time_out_hops=()):
        self.graph = graph
        self.direct = direct          # 'answer' | 'timeout'
        self.time_out_hops = set(time_out_hops)   # predicates of 1-hop queries
        self.queries = []

    def query(self, built, max_results=None):
        qg = built.query_graph
        self.queries.append(json.loads(json.dumps(qg)))
        if len(qg["edges"]) > 1 and self.direct == "timeout":
            return AraxResponse(outcome=OUTCOME_TIMEOUT)
        if len(qg["edges"]) == 1 and (
            list(qg["edges"].values())[0]["predicates"][0] in self.time_out_hops
        ):
            return AraxResponse(outcome=OUTCOME_TIMEOUT)

        nodes = qg["nodes"]

        def ok(key, curie):
            ids = nodes[key].get("ids")
            return not ids or curie in ids

        # Enumerate assignments edge by edge; enough for stars and chains.
        partials = [({}, {})]
        for qkey, qedge in qg["edges"].items():
            s_key, o_key = qedge["subject"], qedge["object"]
            pred = qedge["predicates"][0]
            nxt = []
            for nb, eb in partials:
                for s, o in sorted(self.graph.get(pred, ())):
                    if not (ok(s_key, s) and ok(o_key, o)):
                        continue
                    if nb.get(s_key, s) != s or nb.get(o_key, o) != o:
                        continue
                    nxt.append(({**nb, s_key: s, o_key: o},
                                {**eb, qkey: f"{pred}|{s}|{o}"}))
            partials = nxt

        results, kg_edges = [], {}
        for nb, eb in partials:
            for eid in eb.values():
                pred, s, o = eid.split("|")
                kg_edges[eid] = {"subject": s, "object": o, "predicate": pred}
            results.append({
                "node_bindings": {k: [{"id": v}] for k, v in nb.items()},
                "analyses": [{"edge_bindings": {
                    k: [{"id": v}] for k, v in eb.items()}}],
            })
        return AraxResponse(
            outcome=OUTCOME_SUCCESS if results else OUTCOME_EMPTY,
            response={"message": {
                "results": results,
                "knowledge_graph": {"nodes": {}, "edges": kg_edges},
            }},
            num_results=len(results),
        )


def _run(graph, direct="answer", resolutions=RESOLUTIONS, time_out_hops=()):
    client = GraphDouble(graph, direct=direct, time_out_hops=time_out_hops)
    executor = PathExecutor(client, verbose=False)
    return executor.execute_path(STAR, ENTITIES, resolutions), client


def _anchor_bindings(execution):
    return {i.bindings.get("dermatitis_herpetiformis")
            for i in execution.instances}


# --- decomposition ---------------------------------------------------------


def test_the_second_anchor_hop_is_planned_as_a_closing_step():
    steps = plan_decomposition(STAR, ENTITIES, RESOLUTIONS)
    assert [(s.pinned_ref, s.solve_ref, s.closing) for s in steps] == [
        ("cftr", "candidate_drug", False),
        ("candidate_drug", "dermatitis_herpetiformis", True),
    ]


# --- the live failure ------------------------------------------------------


def test_localizing_an_empty_star_does_not_invent_candidates():
    """The run that surfaced this: direct empty, localization found 'answers'."""
    execution, client = _run(GRAPH_NO_ANSWER)

    assert execution.mode == "direct_then_localized"
    assert execution.instances == []
    assert not execution.candidates
    assert execution.verdict == "no_answer"
    assert OTHER_DISEASE not in _anchor_bindings(execution)
    assert not any("combined query" in w for w in execution.warnings)


def test_the_closing_query_pins_the_disease():
    _, client = _run(GRAPH_NO_ANSWER)
    closing = [q for q in client.queries
               if len(q["edges"]) == 1
               and list(q["edges"].values())[0]["predicates"] == [TREATS]]
    assert closing, "the treats hop was never run"
    for q in closing:
        edge = list(q["edges"].values())[0]
        assert q["nodes"][edge["object"]].get("ids") == [DH]


def test_decomposing_after_a_timeout_returns_only_the_real_answer():
    execution, _ = _run(GRAPH, direct="timeout")

    assert execution.mode == "decomposed"
    assert execution.verdict == "success"
    assert {i.bindings["candidate_drug"] for i in execution.instances} == {"DRUG:A"}
    assert _anchor_bindings(execution) == {DH}


def test_direct_and_decomposed_agree():
    direct, _ = _run(GRAPH)
    decomposed, _ = _run(GRAPH, direct="timeout")
    key = lambda ex: sorted(
        (i.bindings["candidate_drug"], i.bindings["dermatitis_herpetiformis"])
        for i in ex.instances
    )
    assert direct.mode == "direct"
    assert key(direct) == key(decomposed) == [("DRUG:A", DH)]


# --- localization is diagnostic --------------------------------------------
#
# The live rerun after the fix: the direct query was empty on complete
# coverage, hop 0 returned 13 drugs, and the pinned closing hop timed out on
# the ARAX side after 76.6s. Localization then overwrote the direct no_answer
# with `inconclusive`, and the plan outcome became truncated_or_timed_out —
# discarding a real absence because a follow-up diagnostic did not finish.


def test_a_stalled_localization_keeps_the_direct_absence():
    execution, _ = _run(GRAPH_NO_ANSWER, time_out_hops={TREATS})

    assert execution.mode == "direct_then_localized"
    assert execution.verdict == "no_answer"
    assert execution.coverage_complete is True
    assert execution.failure_kind != "timeout"
    assert not execution.candidates


def test_a_stalled_localization_still_says_where_it_got_to():
    """Hop 0 returned data, so hop 1 is the first hop not confirmed."""
    execution, _ = _run(GRAPH_NO_ANSWER, time_out_hops={TREATS})

    assert execution.failed_at_hop == 1
    assert any(
        "complete empty answer stands" in n and "hop 1" in n
        for n in execution.coverage_notes
    )


def test_the_plan_outcome_is_an_absence_not_a_timeout():
    from plan_executor.aggregate import OUTCOME_NO_DATA, typed_outcome

    execution, _ = _run(GRAPH_NO_ANSWER, time_out_hops={TREATS})
    outcome = typed_outcome(
        plan_input=SimpleNamespace(issues=[]),
        executions={"P1": execution},
        resolutions=None, filter_results=None, num_results=0,
    )
    assert outcome["outcome"] == OUTCOME_NO_DATA


def test_a_direct_timeout_is_still_inconclusive():
    """The guard protects a direct *answer*; it does not invent one."""
    execution, _ = _run(GRAPH, direct="timeout", time_out_hops={TREATS})

    assert execution.mode == "decomposed"
    assert execution.verdict == "inconclusive"
    assert execution.coverage_complete is False


# --- refusing to run open-ended --------------------------------------------


def test_a_closing_hop_with_nothing_to_pin_is_inconclusive_not_open():
    """A fixed entity the resolver could not ground must not be left open."""
    execution, client = _run(
        GRAPH, direct="timeout", resolutions={"cftr": [CFTR]},
    )
    assert execution.verdict == "inconclusive"
    assert execution.instances == []
    assert any("no CURIEs to pin" in n for n in execution.coverage_notes)
    assert not any(
        list(q["edges"].values())[0]["predicates"] == [TREATS]
        for q in client.queries if len(q["edges"]) == 1
    )


# --- the join's own guard --------------------------------------------------


def _step_result(step, pairs, solve_pinned):
    return StepResult(step=step, outcome=OUTCOME_SUCCESS, pairs=pairs,
                      solve_pinned=solve_pinned)


def test_the_join_refuses_an_anchor_bound_to_another_curie():
    """Defence in depth: even if a hop ran open, the anchor cannot be swapped."""
    s0, s1 = plan_decomposition(STAR, ENTITIES, RESOLUTIONS)
    open_close = _step_result(
        s1,
        [("DRUG:A", DH, "e1"), ("DRUG:B", OTHER_DISEASE, "e2")],
        solve_pinned=False,
    )
    steps = [
        _step_result(s0, [(CFTR, "DRUG:A", "e3"), (CFTR, "DRUG:B", "e4")],
                     solve_pinned=False),
        open_close,
    ]
    instances = PathExecutor._chain_steps(steps, RESOLUTIONS)

    assert [i.bindings["candidate_drug"] for i in instances] == ["DRUG:A"]
    assert open_close.anchor_rejections == 1


def test_the_join_trusts_a_pinned_end_including_subclass_matches():
    """When the query pinned the anchor, the backend's binding stands."""
    s0, s1 = plan_decomposition(STAR, ENTITIES, RESOLUTIONS)
    subclass = "MONDO:0099999"   # e.g. a descendant ARAX matched for DH
    pinned_close = _step_result(s1, [("DRUG:A", subclass, "e1")],
                                solve_pinned=True)
    steps = [
        _step_result(s0, [(CFTR, "DRUG:A", "e2")], solve_pinned=False),
        pinned_close,
    ]
    instances = PathExecutor._chain_steps(steps, RESOLUTIONS)

    assert len(instances) == 1
    assert pinned_close.anchor_rejections == 0
