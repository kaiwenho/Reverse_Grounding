"""
executor.py — Run a plan path against ARAX, falling back to decomposition.

Strategy
--------
    1. Submit the full N-hop query.
    2. If it times out, decompose: run the hop touching a resolved CURIE
       first, then feed the intermediates it found into the next hop as pinned
       ids, in batches.
    3. Chain the per-hop results back into whole paths by joining on the
       shared intermediate node.

Every batch is executed. Stopping at the first batch that returns hits would
undercount the supporting-path totals that ranking depends on, and would make
an empty result impossible to distinguish from an untested one. Partial
coverage is therefore never silent: it downgrades the verdict.

Verdicts
--------
    success       results found
    no_answer     genuinely nothing, on complete coverage — the planner can
                  act on this
    inconclusive  nothing found, but coverage was incomplete: a batch failed,
                  or ARAX truncated a hop's results. Looks identical to
                  no_answer in the data and means something entirely
                  different, so it is reported separately.
    unexecutable  no anchor to pivot from
    error         a failure decomposition cannot fix

The distinction between `no_answer` and `inconclusive` is the point of most of
the bookkeeping here. `no_answer` is a claim about biology; `inconclusive` is
a claim about the run. Reporting the second as the first would tell the
planner that no drug exists when the query simply never completed.

What this module does not do
----------------------------
It never relaxes a constraint and retries. Deciding which constraint is
load-bearing needs the question's intent and the Biolink model, both of which
live with the planner. The executor reports what failed, at which hop, and
which axes could be loosened — then stops. Auto-relaxation would also mean the
results no longer answer the plan as written, which quietly breaks provenance.

Usage
-----
    executor = PathExecutor(client=arax_client)
    execution = executor.execute_path(path, entities, resolutions)

    if execution.verdict == "no_answer":
        report(execution.to_dict())      # includes suggested_relaxations
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .arax_client import (
    AraxClient, AraxResponse, OUTCOME_EMPTY, OUTCOME_ERROR, OUTCOME_PARTIAL,
    OUTCOME_SUCCESS, OUTCOME_TIMEOUT,
)
from .trapi_builder import (
    BuiltQuery, DecompositionError, HopStep, batch_ids, build_path_query,
    build_step_query, describe, plan_decomposition, DEFAULT_MAX_RESULTS,
)


#: Intermediates pinned into one query. 500 keeps each query bounded while
#: keeping the number of round trips low; the right value is a latency
#: tradeoff worth measuring against a live endpoint rather than assuming.
DEFAULT_BATCH_SIZE = 500

VERDICT_SUCCESS = "success"
VERDICT_NO_ANSWER = "no_answer"
VERDICT_INCONCLUSIVE = "inconclusive"
VERDICT_UNEXECUTABLE = "unexecutable"
VERDICT_ERROR = "error"

#: Why a path produced nothing, when it produced nothing. The distinction
#: the executor exists to preserve: a missing edge, hops that do not meet, a
#: query that never finished, and a filter that removed everything all look
#: identical in the results and call for different responses.
FAILURE_NO_DATA = "no_graph_data"
FAILURE_JOIN = "join_failure"
FAILURE_TIMEOUT = "timeout"
FAILURE_TRUNCATED = "truncation"
FAILURE_UNRESOLVED = "unresolved_grounding"
FAILURE_UNSUPPORTED = "unsupported_capability"
FAILURE_BACKEND = "backend_failure"

MODE_DIRECT = "direct"
MODE_DECOMPOSED = "decomposed"


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass
class StepResult:
    """One hop of a decomposition, across all its batches."""

    step: HopStep
    outcome: str
    pairs: List[Tuple[str, str, str]] = field(default_factory=list)  # pinned, solved, edge_id
    batches_total: int = 0
    batches_ok: int = 0
    batches_failed: int = 0
    truncated: bool = False
    pinned_count: int = 0
    elapsed_s: float = 0.0
    errors: List[str] = field(default_factory=list)
    #: True when the query pinned the solve end as well, so the backend has
    #: already restricted what that end can bind to.
    solve_pinned: bool = False
    #: Pairs the join refused because they bound an anchor to a CURIE the
    #: resolver did not choose. Nonzero means a hop ran with an anchor end open.
    anchor_rejections: int = 0

    @property
    def solved_curies(self) -> List[str]:
        return sorted({s for _, s, _ in self.pairs})

    @property
    def complete(self) -> bool:
        """True when every batch ran cleanly and nothing was truncated.

        Only a complete step can support a `no_answer` verdict: if a batch
        failed, the absence of results says nothing about the biology.
        """
        return self.batches_failed == 0 and not self.truncated

    def to_dict(self) -> Dict[str, Any]:
        return {
            "order": self.step.order,
            "hop_index": self.step.hop_index,
            "pinned_ref": self.step.pinned_ref,
            "solve_ref": self.step.solve_ref,
            "source": self.step.source,
            "closing": getattr(self.step, "closing", False),
            "solve_pinned": self.solve_pinned,
            "anchor_rejections": self.anchor_rejections,
            "outcome": self.outcome,
            "pinned_count": self.pinned_count,
            "solved_count": len(self.solved_curies),
            "pairs_found": len(self.pairs),
            "batches": {"total": self.batches_total, "ok": self.batches_ok,
                        "failed": self.batches_failed},
            "truncated": self.truncated,
            "complete": self.complete,
            "elapsed_s": round(self.elapsed_s, 2),
            "errors": self.errors[:5],
        }


@dataclass
class PathInstance:
    """One complete traversal of a path: a CURIE bound to each entity_ref."""

    bindings: Dict[str, str]
    edge_ids: List[str] = field(default_factory=list)
    return_curie: Optional[str] = None

    def key(self) -> Tuple:
        return tuple(sorted(self.bindings.items()))


@dataclass
class PathExecution:
    """Everything that happened while running one path."""

    path_id: str
    verdict: str
    mode: str = MODE_DIRECT
    instances: List[PathInstance] = field(default_factory=list)
    steps: List[StepResult] = field(default_factory=list)
    kg_nodes: Dict[str, Any] = field(default_factory=dict)
    kg_edges: Dict[str, Any] = field(default_factory=dict)
    return_entity_ref: Optional[str] = None
    failed_at_hop: Optional[int] = None
    failed_at_step: Optional[int] = None
    coverage_complete: bool = True
    coverage_notes: List[str] = field(default_factory=list)
    join_failed: bool = False
    failure_kind: Optional[str] = None
    suggested_relaxations: List[Dict[str, Any]] = field(default_factory=list)
    direct_outcome: Optional[str] = None
    query_description: Optional[str] = None
    submitted_queries: List[Dict[str, Any]] = field(default_factory=list)
    #: Requirements TRAPI cannot express, carried from the builder so that
    #: postfilter can apply them. Without this the plan's entity constraints
    #: — approval_status, and anything else stated on an entity — are computed
    #: and then silently discarded.
    post_conditions: List[Any] = field(default_factory=list)
    elapsed_s: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error_message: Optional[str] = None

    @property
    def candidates(self) -> List[str]:
        """Distinct CURIEs bound to the path's return entity."""
        return sorted({i.return_curie for i in self.instances if i.return_curie})

    def hop_counts(self) -> Dict[str, Any]:
        """Bindings found per hop, and how many survived the join.

        Reported together because the pair is what distinguishes a missing
        edge from hops that do not meet: bindings at every hop with zero
        joined instances is a join failure, while a zero at one hop is missing
        data at that hop.
        """
        return {
            "per_hop": [
                {
                    "order": sr.step.order,
                    "hop_index": sr.step.hop_index,
                    "pinned_ref": sr.step.pinned_ref,
                    "pinned_count": sr.pinned_count,
                    "solve_ref": sr.step.solve_ref,
                    "solved_count": len(sr.solved_curies),
                    "pairs": len(sr.pairs),
                    "complete": sr.complete,
                }
                for sr in self.steps
            ],
            "joined_instances": len(self.instances),
            "join_failed": self.join_failed,
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "path_id": self.path_id,
            "verdict": self.verdict,
            "mode": self.mode,
            "direct_outcome": self.direct_outcome,
            "query": self.query_description,
            "return_entity_ref": self.return_entity_ref,
            "num_instances": len(self.instances),
            "num_candidates": len(self.candidates),
            "kg_nodes": len(self.kg_nodes),
            "kg_edges": len(self.kg_edges),
            "coverage_complete": self.coverage_complete,
            "coverage_notes": self.coverage_notes,
            "failed_at_hop": self.failed_at_hop,
            "failed_at_step": self.failed_at_step,
            "failure_kind": self.failure_kind,
            "hop_counts": self.hop_counts(),
            "steps": [s.to_dict() for s in self.steps],
            "submitted_queries": self.submitted_queries,
            "suggested_relaxations": self.suggested_relaxations,
            "post_conditions": [
                {"kind": pc.kind, "target_key": pc.target_key,
                 "entity_ref": pc.entity_ref, "detail": pc.detail}
                for pc in self.post_conditions
            ],
            "warnings": self.warnings,
            "error": self.error_message,
            "elapsed_s": round(self.elapsed_s, 2),
        }

    def __repr__(self) -> str:
        return (f"<PathExecution {self.path_id} {self.verdict} via {self.mode}: "
                f"{len(self.candidates)} candidates, "
                f"{len(self.instances)} instances>")


# ---------------------------------------------------------------------------
# Relaxation suggestions
# ---------------------------------------------------------------------------


def suggest_relaxations(hop, entities: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Describe which constraints on a failed hop could be loosened.

    Ordered from tightest to loosest, so the planner sees the smallest useful
    change first. These are observations about the hop as written — the
    executor does not name replacement predicates or parent categories, since
    choosing them needs the Biolink model and the question's intent, and both
    belong to the planner.
    """
    out: List[Dict[str, Any]] = []

    quals = getattr(hop, "qualifiers", None)
    if quals:
        active = (
            {k: v for k, v in quals.items() if v}
            if isinstance(quals, dict)
            else {
                f: getattr(quals, f)
                for f in dir(quals)
                if not f.startswith("_") and getattr(quals, f, None)
                and isinstance(getattr(quals, f), str)
            }
        )
        if active:
            out.append({
                "axis": "qualifiers",
                "detail": f"hop constrains {len(active)} qualifier(s): "
                          f"{sorted(active)}",
                "current": active,
            })

    predicate = getattr(hop, "predicate", None)
    if predicate:
        out.append({
            "axis": "predicate",
            "detail": f"'{predicate}' may be narrower than the knowledge graph "
                      f"records; a broader relation may match",
            "current": predicate,
        })

    if getattr(hop, "predicate_expansion", "descendants") == "self_only":
        out.append({
            "axis": "predicate_expansion",
            "detail": "hop requests self_only; allowing descendants would widen "
                      "the match",
            "current": "self_only",
        })

    for ref in (getattr(hop, "subject_ref", None), getattr(hop, "object_ref", None)):
        entity = entities.get(ref)
        if entity is None:
            continue
        category = getattr(entity, "biolink_category", None)
        if category:
            out.append({
                "axis": "category",
                "detail": f"entity '{ref}' is restricted to {category}",
                "entity_ref": ref,
                "current": category,
            })
        constraints = getattr(entity, "constraints", None)
        if constraints:
            out.append({
                "axis": "entity_constraint",
                "detail": f"entity '{ref}' carries {len(constraints)} constraint(s)",
                "entity_ref": ref,
            })

    return out


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------


class PathExecutor:
    """Executes plan paths, decomposing when a direct query is too slow.

    Args:
        client: an AraxClient.
        batch_size: intermediates pinned per query during decomposition.
        max_results: per-query result ceiling. Reaching it means truncation,
            which makes coverage incomplete.
        skip_direct_over_hops: go straight to decomposition for paths with at
            least this many hops. A long chain will almost certainly time out,
            and paying the full wall clock to learn that wastes minutes.
        max_intermediates: optional safety ceiling on how many intermediates
            feed the next hop. None means no cap — capping would break the
            completeness that `no_answer` depends on, so a cap that does fire
            downgrades the verdict.
    """

    def __init__(
        self,
        client: AraxClient,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_results: int = DEFAULT_MAX_RESULTS,
        skip_direct_over_hops: Optional[int] = None,
        max_intermediates: Optional[int] = None,
        localize_empty: bool = True,
        verbose: bool = True,
    ):
        self.client = client
        self.batch_size = batch_size
        self.max_results = max_results
        self.skip_direct_over_hops = skip_direct_over_hops
        self.max_intermediates = max_intermediates
        self.localize_empty = localize_empty
        self.verbose = verbose

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [exec] {msg}")

    # -- entry point -------------------------------------------------------

    def execute_path(
        self,
        path,
        entities: Dict[str, Any],
        resolutions: Dict[str, Sequence[str]],
    ) -> PathExecution:
        """Run one path, decomposing if the direct query is too slow."""
        start = time.time()
        path_id = getattr(path, "path_id", "?")
        return_ref = getattr(path, "return_entity_ref", None)

        execution = PathExecution(
            path_id=path_id, verdict=VERDICT_ERROR, return_entity_ref=return_ref,
        )

        built = build_path_query(path, entities, resolutions)
        execution.query_description = describe(built)
        execution.warnings.extend(built.warnings)
        execution.post_conditions.extend(built.post_conditions)
        self.log(f"{path_id}: {execution.query_description}")

        n_hops = len(path.hops)
        go_direct = not (
            self.skip_direct_over_hops and n_hops >= self.skip_direct_over_hops
        )

        if go_direct:
            execution.submitted_queries.append({
                "stage": "direct", "query_graph": built.query_graph,
            })
            resp = self.client.query(built, max_results=self.max_results)
            execution.direct_outcome = resp.outcome

            if resp.outcome in (OUTCOME_SUCCESS, OUTCOME_EMPTY):
                self._finish_direct(execution, built, resp, path, entities)
                self._bind_return(execution)

                # An empty multi-hop result says the chain found nothing; it
                # does not say where the chain broke. Running the hops
                # separately costs a few cheap queries and distinguishes three
                # very different findings: the first hop has no data, the
                # second has none for what the first returned, or both have
                # data but share no intermediate. Only the last is a statement
                # about the pair rather than about a predicate, and relaxation
                # advice aimed at the wrong hop is wasted.
                # Triggered on the absence of candidates rather than on an
                # empty response: a query can return results that bind only
                # part of the chain, which yields no candidates and is just as
                # opaque about where the break is.
                if (
                    self.localize_empty
                    and not execution.candidates
                    and len(path.hops) > 1
                ):
                    self.log(f"{path_id}: empty — decomposing to locate the "
                             f"break")
                    execution.warnings.append(
                        "direct query returned nothing; hops were run "
                        "separately to identify which one is empty"
                    )
                    direct_state = self._localization_snapshot(execution)
                    self._run_decomposition(execution, path, entities, resolutions)
                    execution.mode = "direct_then_localized"
                    self._keep_direct_absence(execution, direct_state)

                self._bind_return(execution)
                execution.elapsed_s = time.time() - start
                return execution

            if resp.outcome == OUTCOME_ERROR:
                # Decomposition reissues the same constraints, so a malformed
                # or rejected query fails identically piece by piece.
                execution.verdict = VERDICT_ERROR
                execution.error_message = resp.error_message
                execution.elapsed_s = time.time() - start
                self.log(f"{path_id}: error, not decomposing — {resp.error_message}")
                return execution

            reason = "cached timeout" if resp.from_cache else resp.outcome
            self.log(f"{path_id}: {reason} — decomposing")
        else:
            execution.direct_outcome = "skipped"
            self.log(f"{path_id}: {n_hops} hops, skipping direct attempt")

        self._run_decomposition(execution, path, entities, resolutions)
        self._bind_return(execution)
        execution.elapsed_s = time.time() - start
        return execution

    # -- localization -------------------------------------------------------

    @staticmethod
    def _localization_snapshot(execution: PathExecution) -> Dict[str, Any]:
        """The direct query's verdict fields, before localization touches them."""
        return {
            "verdict": execution.verdict,
            "failure_kind": execution.failure_kind,
            "coverage_complete": execution.coverage_complete,
            "error_message": execution.error_message,
        }

    def _keep_direct_absence(
        self, execution: PathExecution, direct: Dict[str, Any],
    ) -> None:
        """Stop a diagnostic run that did not finish from overturning an answer.

        Localization runs the hops separately only to find *where* an empty
        chain breaks. When the direct query was a complete, trustworthy empty
        result, that is the answer: the closing hop now re-asks a strict
        subset of the same question, so it cannot legitimately find more. If
        the localization times out or errors, it has established nothing, and
        reporting the path as `inconclusive` would discard a real absence on
        the strength of a diagnostic that never completed.

        What localization did establish is kept. If earlier steps returned
        data, the step that failed is the first hop not confirmed to hold
        any, which is a better place to aim relaxation than the direct query's
        default of hop 0.
        """
        if direct["verdict"] != VERDICT_NO_ANSWER:
            return
        if execution.verdict not in (
            VERDICT_INCONCLUSIVE, VERDICT_ERROR, VERDICT_UNEXECUTABLE,
        ):
            return

        stalled = execution.verdict
        execution.verdict = direct["verdict"]
        execution.failure_kind = direct["failure_kind"]
        execution.coverage_complete = direct["coverage_complete"]
        execution.error_message = direct["error_message"]
        where = (
            f" at step {execution.failed_at_step}; hop "
            f"{execution.failed_at_hop} is the first hop not confirmed to "
            f"hold data"
            if execution.failed_at_hop is not None
            else "; the break was not located"
        )
        execution.coverage_notes.append(
            f"localization ended {stalled}{where}. The direct query's "
            f"complete empty answer stands."
        )
        self.log(f"{execution.path_id}: localization {stalled}; keeping the "
                 f"direct no_answer")

    @staticmethod
    def _bind_return(execution: PathExecution) -> None:
        """Populate each instance's return_curie from the path's return entity."""
        ref = execution.return_entity_ref
        if not ref:
            return
        for inst in execution.instances:
            if inst.return_curie is None:
                inst.return_curie = inst.bindings.get(ref)

    # -- direct mode -------------------------------------------------------

    def _finish_direct(
        self,
        execution: PathExecution,
        built: BuiltQuery,
        resp: AraxResponse,
        path,
        entities: Dict[str, Any],
    ) -> None:
        execution.mode = MODE_DIRECT
        message = (resp.response or {}).get("message") or {}
        kg = message.get("knowledge_graph") or {}
        execution.kg_nodes = kg.get("nodes") or {}
        execution.kg_edges = kg.get("edges") or {}
        execution.instances = self._instances_from_trapi(
            message.get("results") or [], built
        )

        if resp.truncated:
            execution.coverage_complete = False
            execution.coverage_notes.append(
                f"ARAX returned {resp.num_results} results, at the requested "
                f"ceiling of {self.max_results}; more may exist"
            )

        if execution.instances and execution.candidates:
            execution.verdict = VERDICT_SUCCESS
        elif execution.instances:
            # Results came back but nothing bound the return entity, so the
            # path produced no candidates even though it matched something.
            execution.verdict = VERDICT_INCONCLUSIVE
            execution.coverage_complete = False
            execution.coverage_notes.append(
                f"{len(execution.instances)} result(s) returned, but none bound "
                f"'{execution.return_entity_ref}'; the path yielded no candidates"
            )
        elif resp.is_trustworthy_empty and execution.coverage_complete:
            execution.verdict = VERDICT_NO_ANSWER
            execution.failed_at_hop = 0
            execution.suggested_relaxations = suggest_relaxations(
                path.hops[0], entities
            ) if path.hops else []
        else:
            execution.verdict = VERDICT_INCONCLUSIVE

        self.log(f"{execution.path_id}: direct -> {execution.verdict} "
                 f"({len(execution.candidates)} candidates)")

    @staticmethod
    def _instances_from_trapi(results: List[Dict], built: BuiltQuery) -> List[PathInstance]:
        """Convert TRAPI results into PathInstances via the builder's node map."""
        out: List[PathInstance] = []
        return_ref = None
        for ref, key in built.node_map.items():
            if key == built.return_node_key:
                return_ref = ref
                break

        for result in results:
            bindings: Dict[str, str] = {}
            for node_key, binds in (result.get("node_bindings") or {}).items():
                ref = built.entity_for_node(node_key)
                if not ref or not binds:
                    continue
                first = binds[0]
                bindings[ref] = first.get("id") if isinstance(first, dict) else first

            edge_ids: List[str] = []
            for analysis in result.get("analyses") or []:
                for _, ebinds in (analysis.get("edge_bindings") or {}).items():
                    for eb in ebinds or []:
                        eid = eb.get("id") if isinstance(eb, dict) else eb
                        if eid:
                            edge_ids.append(eid)

            if bindings:
                out.append(PathInstance(
                    bindings=bindings,
                    edge_ids=edge_ids,
                    return_curie=bindings.get(return_ref),
                ))
        return out

    # -- decomposition -----------------------------------------------------

    def _run_decomposition(
        self,
        execution: PathExecution,
        path,
        entities: Dict[str, Any],
        resolutions: Dict[str, Sequence[str]],
    ) -> None:
        execution.mode = MODE_DECOMPOSED

        try:
            steps = plan_decomposition(path, entities, resolutions)
        except DecompositionError as e:
            execution.verdict = VERDICT_UNEXECUTABLE
            execution.failure_kind = FAILURE_UNRESOLVED
            execution.error_message = str(e)
            self.log(f"{execution.path_id}: unexecutable — {e}")
            return

        self.log(f"{execution.path_id}: {len(steps)} step(s): "
                 f"{' then '.join(f'{s.pinned_ref}->{s.solve_ref}' for s in steps)}")

        # CURIEs known for each entity, growing as steps resolve them.
        known: Dict[str, List[str]] = {
            ref: list(curies) for ref, curies in resolutions.items() if curies
        }

        for step in steps:
            pinned = known.get(step.pinned_ref) or []
            if not pinned:
                execution.verdict = VERDICT_INCONCLUSIVE
                execution.failed_at_step = step.order
                execution.failed_at_hop = step.hop_index
                execution.coverage_complete = False
                execution.coverage_notes.append(
                    f"step {step.order} had no CURIEs for '{step.pinned_ref}'"
                )
                return

            # A closing hop verifies pairs between two entities that are both
            # already known, so both ends are pinned. Left open, the far end
            # binds to anything the relationship reaches, and an anchor such as
            # a disease is silently replaced by whatever disease came back.
            solve_ids: Optional[List[str]] = None
            if getattr(step, "closing", False):
                solve_ids = known.get(step.solve_ref) or []
                if not solve_ids:
                    execution.verdict = VERDICT_INCONCLUSIVE
                    execution.failed_at_step = step.order
                    execution.failed_at_hop = step.hop_index
                    execution.coverage_complete = False
                    execution.coverage_notes.append(
                        f"step {step.order} closes the path on "
                        f"'{step.solve_ref}', which has no CURIEs to pin; the "
                        f"hop was not run open-ended because that would let "
                        f"'{step.solve_ref}' bind to anything"
                    )
                    return

            result = self._run_step(
                step, entities, pinned, execution, solve_ids=solve_ids,
            )
            execution.steps.append(result)

            if result.outcome == OUTCOME_TIMEOUT:
                execution.verdict = VERDICT_INCONCLUSIVE
                execution.failure_kind = FAILURE_TIMEOUT
                execution.failed_at_step = step.order
                execution.failed_at_hop = step.hop_index
                execution.coverage_complete = False
                execution.coverage_notes.append(
                    f"step {step.order}: all {result.batches_total} batch(es) "
                    f"timed out; nothing is known about '{step.solve_ref}'"
                )
                execution.suggested_relaxations = suggest_relaxations(
                    step.hop, entities
                )
                self.log(f"{execution.path_id}: inconclusive at step "
                         f"{step.order} — every batch timed out")
                return

            if result.outcome == OUTCOME_ERROR:
                execution.verdict = VERDICT_ERROR
                execution.failure_kind = FAILURE_BACKEND
                execution.coverage_complete = False
                execution.failed_at_step = step.order
                execution.failed_at_hop = step.hop_index
                execution.error_message = "; ".join(result.errors[:3])
                return

            if not result.pairs:
                # Nothing found. Whether that is evidence depends entirely on
                # whether this step actually completed.
                execution.failed_at_step = step.order
                execution.failed_at_hop = step.hop_index
                execution.suggested_relaxations = suggest_relaxations(
                    step.hop, entities
                )

                if result.complete and result.outcome == OUTCOME_EMPTY:
                    execution.verdict = VERDICT_NO_ANSWER
                    execution.failure_kind = FAILURE_NO_DATA
                    self.log(
                        f"{execution.path_id}: no_answer at step {step.order} — "
                        f"{result.pinned_count} '{step.pinned_ref}' checked "
                        f"exhaustively, no '{step.solve_ref}' found"
                    )
                else:
                    execution.verdict = VERDICT_INCONCLUSIVE
                    execution.coverage_complete = False
                    execution.failure_kind = (
                        FAILURE_TRUNCATED if result.truncated else FAILURE_TIMEOUT
                    )
                    execution.coverage_notes.append(
                        f"step {step.order} returned nothing but did not complete "
                        f"({result.batches_failed} of {result.batches_total} "
                        f"batches failed"
                        + (", results truncated" if result.truncated else "")
                        + ") — absence here is not evidence"
                    )
                    self.log(f"{execution.path_id}: inconclusive at step {step.order}")
                return

            if not result.complete:
                execution.coverage_complete = False
                if result.batches_failed:
                    execution.coverage_notes.append(
                        f"step {step.order}: {result.batches_failed} of "
                        f"{result.batches_total} batches failed; some "
                        f"'{step.solve_ref}' may be missing"
                    )
                if result.truncated:
                    execution.coverage_notes.append(
                        f"step {step.order}: a batch hit the {self.max_results} "
                        f"result ceiling; some '{step.solve_ref}' may be missing"
                    )

            solved = result.solved_curies
            if self.max_intermediates and len(solved) > self.max_intermediates:
                execution.coverage_complete = False
                execution.coverage_notes.append(
                    f"step {step.order}: {len(solved)} intermediates capped at "
                    f"{self.max_intermediates}; downstream results are a subset"
                )
                solved = solved[: self.max_intermediates]

            # A closing hop verifies pairs already bound rather than adding
            # new CURIEs, so the existing set is left intact.
            if step.solve_ref not in known:
                known[step.solve_ref] = solved

            self.log(f"{execution.path_id}: step {step.order} -> "
                     f"{len(result.solved_curies)} '{step.solve_ref}' "
                     f"from {result.pinned_count} '{step.pinned_ref}'")

        execution.instances = self._chain_steps(execution.steps, resolutions)
        for sr in execution.steps:
            if sr.anchor_rejections:
                execution.warnings.append(
                    f"step {sr.step.order}: {sr.anchor_rejections} pair(s) "
                    f"dropped for binding anchor '{sr.step.solve_ref}' to a "
                    f"CURIE the resolver did not choose"
                )
        self._bind_return(execution)

        # Every hop returned bindings but no chain survived the join: the hops
        # do not share an intermediate. That is a different finding from a hop
        # having no data, and points at a different fix — the pair of
        # relationships does not meet in this graph, rather than one of them
        # being absent. Recorded before the verdict so it is not folded into a
        # generic no_answer.
        if not execution.instances and all(sr.pairs for sr in execution.steps):
            execution.join_failed = True
            execution.failure_kind = FAILURE_JOIN
            execution.coverage_notes.append(
                "every hop returned bindings, but they share no intermediate "
                "node, so no complete path exists: "
                + "; ".join(
                    f"step {sr.step.order} found {len(sr.solved_curies)} "
                    f"'{sr.step.solve_ref}' from {sr.pinned_count} "
                    f"'{sr.step.pinned_ref}'"
                    for sr in execution.steps
                )
            )

        if execution.direct_outcome in (OUTCOME_EMPTY, OUTCOME_SUCCESS) and (
            execution.mode == "direct_then_localized" and execution.instances
        ):
            # The decomposed form found paths the single query did not. That
            # is a fact about how the query was executed, not about the data,
            # and worth surfacing rather than quietly returning more results
            # than the direct attempt.
            execution.warnings.append(
                f"the direct multi-hop query returned nothing, but running the "
                f"hops separately found {len(execution.instances)} path(s). The "
                f"combined query, not the underlying data, was the limitation."
            )
        if execution.instances and not execution.candidates:
            execution.verdict = VERDICT_INCONCLUSIVE
            execution.coverage_complete = False
            execution.coverage_notes.append(
                f"{len(execution.instances)} path(s) assembled, but none bound "
                f"'{execution.return_entity_ref}'; no candidates to return"
            )
        else:
            execution.verdict = (
                VERDICT_SUCCESS if execution.instances
                else (VERDICT_NO_ANSWER if execution.coverage_complete
                      else VERDICT_INCONCLUSIVE)
            )

        if not execution.instances and execution.verdict == VERDICT_NO_ANSWER:
            # Each hop had results but no chain survives the join: the hops do
            # not share intermediates.
            execution.coverage_notes.append(
                "every hop returned results, but none share an intermediate "
                "node, so no complete path exists"
            )

        self.log(f"{execution.path_id}: decomposed -> {execution.verdict} "
                 f"({len(execution.candidates)} candidates, "
                 f"{len(execution.instances)} instances)")

    def _run_step(
        self,
        step: HopStep,
        entities: Dict[str, Any],
        pinned: Sequence[str],
        execution: PathExecution,
        solve_ids: Optional[Sequence[str]] = None,
    ) -> StepResult:
        """Run one hop across every batch of pinned intermediates.

        `solve_ids` pins the far end as well, for a closing hop. Only the
        pinned end is batched; the far end is the same short list each time.
        """
        start = time.time()
        batches = batch_ids(pinned, self.batch_size)
        result = StepResult(
            step=step, outcome=OUTCOME_EMPTY,
            batches_total=len(batches), pinned_count=len(set(pinned)),
            solve_pinned=bool(solve_ids),
        )

        any_success = False
        n_timeout = 0
        n_error = 0
        for i, batch in enumerate(batches):
            built = build_step_query(step, entities, batch, solve_ids=solve_ids)
            # Each step re-derives the same conditions for the entities it
            # touches; deduplicated so a filter is not counted once per batch.
            for pc in built.post_conditions:
                if not any(
                    existing.kind == pc.kind
                    and existing.target_key == pc.target_key
                    and existing.entity_ref == pc.entity_ref
                    for existing in execution.post_conditions
                ):
                    execution.post_conditions.append(pc)
            execution.submitted_queries.append({
                "stage": f"step{step.order}_batch{i}",
                "pinned_ref": step.pinned_ref, "solve_ref": step.solve_ref,
                "num_pinned": len(batch), "query_graph": built.query_graph,
            })
            resp = self.client.query(built, max_results=self.max_results)

            if resp.outcome in (OUTCOME_SUCCESS, OUTCOME_PARTIAL, OUTCOME_EMPTY):
                result.batches_ok += 1
                any_success = True
                if resp.truncated:
                    result.truncated = True
                if resp.outcome == OUTCOME_PARTIAL:
                    result.batches_failed += 1
                    result.errors.append(
                        f"batch {i}: partial results ({resp.error_message})"
                    )
                self._collect_pairs(resp, step, built, result, execution)
            else:
                result.batches_failed += 1
                if resp.outcome == OUTCOME_TIMEOUT:
                    n_timeout += 1
                else:
                    n_error += 1
                result.errors.append(f"batch {i}: {resp.outcome} — {resp.error_message}")
                self.log(f"    batch {i + 1}/{len(batches)} ({len(batch)} ids): "
                         f"{resp.outcome}")

        if result.pairs:
            result.outcome = OUTCOME_SUCCESS
        elif any_success:
            result.outcome = OUTCOME_EMPTY
        elif n_timeout and not n_error:
            # Every batch timed out. Nothing is broken and nothing is known —
            # reporting this as an error would send the planner to fix a query
            # that was never actually wrong.
            result.outcome = OUTCOME_TIMEOUT
        else:
            result.outcome = OUTCOME_ERROR

        result.elapsed_s = time.time() - start
        return result

    @staticmethod
    def _collect_pairs(
        resp: AraxResponse,
        step: HopStep,
        built: BuiltQuery,
        result: StepResult,
        execution: PathExecution,
    ) -> None:
        """Extract (pinned, solved, edge_id) triples and merge the KG fragment."""
        message = (resp.response or {}).get("message") or {}
        kg = message.get("knowledge_graph") or {}
        execution.kg_nodes.update(kg.get("nodes") or {})
        execution.kg_edges.update(kg.get("edges") or {})

        pinned_key = "n00" if step.pinned_is_subject else "n01"
        solve_key = "n01" if step.pinned_is_subject else "n00"

        for r in message.get("results") or []:
            nb = r.get("node_bindings") or {}
            pinned_binds, solve_binds = nb.get(pinned_key) or [], nb.get(solve_key) or []
            if not pinned_binds or not solve_binds:
                continue

            def first_id(binds):
                b = binds[0]
                return b.get("id") if isinstance(b, dict) else b

            p_curie, s_curie = first_id(pinned_binds), first_id(solve_binds)

            edge_id = ""
            for analysis in r.get("analyses") or []:
                for _, ebinds in (analysis.get("edge_bindings") or {}).items():
                    for eb in ebinds or []:
                        edge_id = eb.get("id") if isinstance(eb, dict) else eb
                        break
                    if edge_id:
                        break
                if edge_id:
                    break

            if p_curie and s_curie:
                result.pairs.append((p_curie, s_curie, edge_id))

    @staticmethod
    def _chain_steps(
        steps: List[StepResult],
        resolutions: Dict[str, Sequence[str]],
    ) -> List[PathInstance]:
        """Join per-hop results into whole paths on their shared intermediates.

        This is the merge the decomposition exists to make possible: hop 1
        found genes for the disease, hop 2 found drugs for those genes, and a
        path is only real when the same gene appears in both. An inner join,
        so a drug reached through a gene that hop 1 never returned is dropped.
        """
        if not steps:
            return []

        first = steps[0].step
        seeds = list(resolutions.get(first.pinned_ref) or [])
        partials: List[Tuple[Dict[str, str], List[str]]] = [
            ({first.pinned_ref: curie}, []) for curie in seeds
        ]
        if not partials:
            partials = [({}, [])]

        for sr in steps:
            step = sr.step
            by_pinned: Dict[str, List[Tuple[str, str]]] = {}
            for p_curie, s_curie, edge_id in sr.pairs:
                by_pinned.setdefault(p_curie, []).append((s_curie, edge_id))

            next_partials: List[Tuple[Dict[str, str], List[str]]] = []
            for bindings, edges in partials:
                pinned_curie = bindings.get(step.pinned_ref)
                matches = (
                    by_pinned.get(pinned_curie, [])
                    if pinned_curie is not None
                    else [pair for pairs in by_pinned.values() for pair in pairs]
                )
                for s_curie, edge_id in matches:
                    # A closing hop must agree with what is already bound,
                    # rather than introducing a second value for one entity.
                    existing = bindings.get(step.solve_ref)
                    if existing is not None and existing != s_curie:
                        continue
                    # An anchor binds only to what the resolver chose. When the
                    # query pinned this end the backend already enforced that
                    # (subclass matches included), so the check applies only
                    # to an end that was left open — which should never happen
                    # for an anchor, and is counted if it does.
                    anchor_ids = resolutions.get(step.solve_ref)
                    if (
                        anchor_ids
                        and not sr.solve_pinned
                        and s_curie not in anchor_ids
                    ):
                        sr.anchor_rejections += 1
                        continue
                    new_bindings = dict(bindings)
                    new_bindings[step.solve_ref] = s_curie
                    if pinned_curie is None:
                        new_bindings[step.pinned_ref] = next(
                            (p for p, prs in by_pinned.items()
                             if (s_curie, edge_id) in prs), ""
                        )
                    next_partials.append(
                        (new_bindings, edges + ([edge_id] if edge_id else []))
                    )
            partials = next_partials
            if not partials:
                break

        return [PathInstance(bindings=b, edge_ids=e) for b, e in partials]


# ---------------------------------------------------------------------------
# Plan-level driver
# ---------------------------------------------------------------------------


def execute_paths(
    executor: PathExecutor,
    paths: Iterable,
    entities: Dict[str, Any],
    resolutions: Dict[str, Sequence[str]],
    return_refs: Optional[Dict[str, str]] = None,
) -> Dict[str, PathExecution]:
    """Run every active path, keyed by path_id.

    Disabled paths are the caller's business to filter; a path that fails does
    not stop the others, since a plan's paths are alternative routes to the
    same answer and one failing is informative rather than fatal.
    """
    out: Dict[str, PathExecution] = {}
    for path in paths:
        execution = executor.execute_path(path, entities, resolutions)
        if return_refs:
            execution.return_entity_ref = return_refs.get(
                execution.path_id, execution.return_entity_ref
            )
        ref = execution.return_entity_ref
        if ref:
            for inst in execution.instances:
                inst.return_curie = inst.bindings.get(ref)
        out[execution.path_id] = execution
    return out
