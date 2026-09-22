"""
run_plan.py — Execute a query plan end to end.

    load & validate plan          plan_input.py
    resolve entities              resolver.py   (LLM required)
    execute paths                 executor.py   (direct, or decomposed)
    verify results are on-concept postfilter.py
    apply evidence policy         postfilter.py
    verify literature             evidence.py   (optional, LLM)
    rank candidates               rank.py       (optional LLM rerank)
    assemble output               aggregate.py

Ordering is not arbitrary. Each stage can stop the run before a later one
spends anything: an invalid plan never reaches ARAX, an unresolved anchor never
reaches a query, and a failed concept check is reported before its results are
ranked. The expensive stages sit last on purpose.

Failure policy differs by stage, deliberately
---------------------------------------------
Resolution halts the run when the LLM cannot be reached, because an unverified
anchor produces confident results about the wrong concept and nothing
downstream would reveal it. Reranking does not halt: its deterministic
fallback is the plan's own ranking criteria, which is a meaningful answer, and
a suboptimal order is visible in a way a wrong anchor is not.

Usage
-----
    python run_plan.py plan.json
    python run_plan.py plan.json --out runs/result.json
    python run_plan.py plan.json --check          # validate only, no queries
    python run_plan.py plan.json --mock           # offline, no ARAX or LLM
    python run_plan.py plan.json --no-verify-literature --no-rerank
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from . import config
from .aggregate import (
    build_ledger, build_result, controller_hints, result_summary_lines,
    write_result,
)
from .arax_client import AraxClient, MockArax
from .cache import QueryCache
from .evidence import EvidenceGatherer
from .executor import PathExecutor, execute_paths
from .explain import Explainer
from .llm import LLMAgent, LLMError, ScriptedAgent
from .meta_kg import MetaKnowledgeGraph
from .plan_input import Issue, SEVERITY_ERROR, load_plan_input
from .postfilter import (
    EvidencePolicy, EvidenceRecommendation, apply_to_execution,
    check_anchor_concept, check_result_category, distribution,
    resolve_concept_check, score_epc,
)
from .rank import CandidateRanker, RankingPlan, collect_candidates
from .resolver import EntityResolver, LLMUnavailableError


class Runner:
    """Orchestrates one execution of one plan."""

    def __init__(self, args: argparse.Namespace):
        self.args = args
        self.verbose = not args.quiet
        self.started_at = datetime.now(timezone.utc).isoformat()
        self.t0 = time.time()

        self.cache = QueryCache(
            path=args.cache, enabled=not args.no_cache,
        )
        self.llm: Any = None
        self.meta_checks: List[Dict[str, Any]] = []
        self.ranking_plan = RankingPlan()
        self.client: Optional[AraxClient] = None
        self.resolver: Optional[EntityResolver] = None
        self.evidence: Optional[EvidenceGatherer] = None

    def log(self, msg: str = "") -> None:
        if self.verbose:
            print(msg)

    def step(self, n: int, total: int, name: str) -> None:
        self.log(f"\n[{n}/{total}] {name}")

    # -- setup -------------------------------------------------------------

    def build_llm(self) -> Any:
        """Create the LLM agent, checking it responds before anything else runs.

        Checked up front because resolution halts the run on LLM failure;
        discovering the model is unreachable after several ARAX queries wastes
        minutes that a two-second probe prevents.
        """
        if self.args.mock:
            self.log("  mock mode: using a scripted agent (makes no judgements)")
            return ScriptedAgent(choose_index=0, same_concept_answer=True)

        agent = LLMAgent(
            model=self.args.model, url=self.args.ollama_url,
            timeout=self.args.llm_timeout, verbose=self.verbose,
        )
        health = agent.health_check()
        if not health["reachable"]:
            print(
                f"\nERROR: the LLM at {self.args.ollama_url} is not reachable "
                f"({health['error']}).\n"
                f"Entity resolution requires it — the name resolver's ranking "
                f"is not reliable enough to use unattended, and a wrong anchor "
                f"silently produces results about the wrong concept.\n"
                f"Start it with `ollama serve` and pull {self.args.model}, "
                f"then re-run. Cached work is not repeated.",
                file=sys.stderr,
            )
            raise SystemExit(2)
        self.log(f"  LLM reachable: {self.args.model}")
        return agent

    def build_client(self) -> AraxClient:
        mock = None
        if self.args.mock:
            # A mock that times out on multi-hop queries exercises the
            # decomposition path, which is the part most worth testing offline.
            mock = MockArax(timeout_when=lambda qg: len(qg.get("edges", {})) > 1)
            self.log("  mock mode: multi-hop queries will simulate a timeout")
        return AraxClient(
            url=self.args.arax_url, timeout=self.args.timeout,
            cache=self.cache, mock=mock, verbose=self.verbose,
        )

    # -- stages ------------------------------------------------------------

    def load(self):
        plan_input = load_plan_input(self.args.plan, strict=not self.args.lenient)
        self.log(plan_input.report())
        return plan_input

    @staticmethod
    def _mock_lookup(query: str, biolink_type: Optional[str]):
        """Synthetic name resolution, so --mock touches no network at all."""
        if not query:
            return []
        slug = "".join(ch for ch in query.upper() if ch.isalnum())[:12] or "X"
        prefix = {
            "biolink:Disease": "MONDO", "biolink:Gene": "NCBIGene",
            "biolink:Protein": "UniProtKB", "biolink:SmallMolecule": "CHEBI",
            "biolink:Drug": "CHEBI",
        }.get(biolink_type or "", "MOCK")
        return [{
            "curie": f"{prefix}:{abs(hash(slug)) % 900000 + 100000}",
            "label": query,
            "types": [biolink_type or "biolink:NamedThing"],
        }]

    def check_meta_kg(self, plan_input) -> bool:
        """Reject hops naming triples ARAX cannot answer.

        Runs before resolution because it needs no CURIEs — only categories and
        predicates, both of which the plan already states. A plan failing here
        has cost one cached fetch rather than a query per hop per path.

        Returns True when execution should continue.
        """
        meta = MetaKnowledgeGraph.load(
            self.client, cache=self.cache, vocab=self._biolink_vocab(),
        )
        if meta is None:
            self.log("  meta knowledge graph unavailable; skipping the check")
            return True

        self.log(f"  {meta.summary()['triples']} supported triple(s)")
        self.log("  a supported triple means ARAX can answer that shape of "
                 "hop, not that it holds data for these entities")
        checks = meta.check_plan(plan_input)
        self.meta_checks = [c.to_dict() for c in checks]

        blocked = False
        for check in checks:
            if check.supported:
                # Every hop is reported, matched exactly or not. A silent pass
                # is indistinguishable from a hop that was never examined, and
                # the difference matters when a plan later returns nothing.
                via = f" (as {check.matched_as})" if check.matched_as else ""
                self.log(f"  ok      {check.path_id} hop {check.hop_index}: "
                         f"{check.triple}{via}")
                continue

            blocked = True
            plan_input.blocked_paths[check.path_id] = check.message()
            plan_input.issues.append(Issue(
                SEVERITY_ERROR, "hop_unsupported_by_arax", check.message(),
                scope="path", target=check.path_id,
            ))
            self.log(f"  BLOCKED {check.path_id} hop {check.hop_index}: "
                     f"{check.triple}")
            self.log(f"          {check.message()}")

        if blocked and not plan_input.active_paths:
            self.log("\n  No path survives; nothing will be queried.")
            return False
        return True

    @staticmethod
    def _biolink_vocab():
        """The Biolink model, when plan_core can supply it.

        Used to expand categories and predicates to their descendants, matching
        how ARAX interprets a query. Without it the check falls back to a small
        equivalence table and is correspondingly blunter.
        """
        try:
            from plan_core import load_biolink_vocabulary
            return load_biolink_vocabulary()
        except Exception:
            return None

    def resolve(self, plan_input):
        # Only pass only_taxa when --taxon was given. Passing None would
        # override the resolver's human default with 'no filter', which is
        # the opposite of what omitting the flag should mean.
        taxa_kwargs = {}
        if self.args.taxon is not None:
            taxa_kwargs['only_taxa'] = [t for t in self.args.taxon if t]

        self.resolver = EntityResolver(
            cache=self.cache, disambiguator=self.llm, **taxa_kwargs,
            mock_lookup=self._mock_lookup if self.args.mock else None,
            confirm_single=not self.args.no_confirm_single,
            conflation=self.args.conflate or None,
            default_conflation=self.args.default_conflation,
            verbose=self.verbose,
        )
        try:
            resolutions = self.resolver.resolve_plan(
                plan_input, entities=plan_input.entities_by_ref,
            )
        except LLMUnavailableError as e:
            print(f"\nERROR: {e}", file=sys.stderr)
            raise SystemExit(2)

        plan_input.check_resolution(resolutions)
        return resolutions

    def execute(self, plan_input, resolutions) -> Dict[str, Any]:
        executor = PathExecutor(
            client=self.client,
            batch_size=self.args.batch_size,
            max_results=self.args.max_results,
            skip_direct_over_hops=self.args.skip_direct_over_hops,
            localize_empty=self.args.localize_empty,
            verbose=self.verbose,
        )
        curies = EntityResolver.curie_map(resolutions)
        executions = execute_paths(
            executor, plan_input.active_paths, plan_input.entities_by_ref,
            curies, return_refs=plan_input.return_refs,
        )
        # Instances carry their path so evidence can be grouped by route in
        # the output; two paths reaching one answer by different mechanisms is
        # stronger than one path reaching it twice.
        for pid, execution in executions.items():
            for inst in execution.instances:
                inst.path_id = pid
        return executions

    def check_concepts(self, plan_input, resolutions, executions) -> List[Any]:
        """Confirm results concern the concepts that were pinned.

        ARAX normalizes CURIEs internally and can merge concepts a plan meant
        to keep apart, so a pinned identifier is no guarantee. The structural
        comparison is free; the LLM is consulted only when it disagrees.
        """
        checks = []
        for pid, execution in executions.items():
            path = next(
                (p for p in plan_input.active_paths
                 if getattr(p, "path_id", None) == pid), None
            )
            for ref, res in resolutions.items():
                if getattr(res, "is_variable", False) or not getattr(res, "curies", None):
                    continue
                if not any(ref in (i.bindings or {}) for i in execution.instances):
                    continue
                check = check_anchor_concept(
                    execution, ref, res.curies[0], getattr(res, "label", None),
                )
                if check.needs_llm and self.llm is not None and not self.args.mock:
                    check = resolve_concept_check(check, plan_input.question, self.llm)
                checks.append(check)
                if not check.ok:
                    self.log(f"  concept check FAILED for {ref}: {check.detail}")

            expected = getattr(path, "expected_result_category", None) if path else None
            if expected and execution.return_entity_ref:
                cat = check_result_category(
                    execution, execution.return_entity_ref, expected,
                )
                if not cat.ok:
                    checks.append(cat)
                    self.log(f"  category check: {cat.detail}")
        return checks

    def filter(self, plan_input, executions):
        """Apply the evidence policy, recording the pre-filter distribution first.

        The distribution is taken before filtering because its purpose is to
        let a controlling agent choose a threshold; a post-filter histogram
        only shows what already passed.
        """
        distributions, filters, all_edges = {}, {}, {}
        for pid, execution in executions.items():
            distributions[pid] = distribution(execution.kg_edges)

            policy = EvidencePolicy.from_plan(plan_input.raw)
            post_conditions = getattr(execution, "post_conditions", None) or []
            result = apply_to_execution(
                execution, policy, post_conditions=post_conditions, mutate=True,
            )
            filters[pid] = result
            all_edges.update(execution.kg_edges)
            if self.verbose and (result.dropped_edge_ids or result.warnings):
                self.log(f"  {pid}:")
                for line in result.report().splitlines():
                    self.log(f"    {line}")
        return distributions, filters, all_edges

    def verify_literature(self, executions, all_edges, ranked_preview):
        """Check citations for the edges behind the leading candidates.

        Bounded to the top candidates because verification is the most
        expensive stage per unit of value: an edge supporting a candidate
        nobody will read does not need a warrant.
        """
        self.evidence = EvidenceGatherer(
            cache=self.cache, verifier=self.llm,
            max_read=self.args.max_abstracts, verbose=self.verbose,
        )
        target_edges = []
        for cand in ranked_preview[: self.args.verify_top_k]:
            target_edges.extend(sorted(cand.edge_ids))
        target_edges = list(dict.fromkeys(target_edges))[: self.args.max_verify_edges]

        if not target_edges:
            return {}, {}

        nodes: Dict[str, Any] = {}
        for execution in executions.values():
            nodes.update(execution.kg_nodes)

        self.log(f"  verifying {len(target_edges)} edge(s) behind the top "
                 f"{self.args.verify_top_k} candidate(s)")
        supports = self.evidence.verify_edges(target_edges, all_edges, nodes)

        dropped = [eid for eid, s in supports.items() if s.should_drop]
        if dropped:
            # Only text-mined edges reach this: the claim *is* the extraction,
            # so a citation that does not support it leaves nothing behind.
            self.log(f"  dropping {len(dropped)} text-mined edge(s) whose "
                     f"citations did not support them")
            for execution in executions.values():
                execution.kg_edges = {
                    k: v for k, v in execution.kg_edges.items() if k not in dropped
                }
                execution.instances = [
                    i for i in execution.instances
                    if not (set(getattr(i, "edge_ids", []) or []) & set(dropped))
                ]
        return {eid: s.to_dict() for eid, s in supports.items()}, \
            EvidenceGatherer.summarize(supports)

    def rank(self, plan_input, executions, all_edges, literature, use_reranker,
             preview=False, stage="discovery"):
        """Rank candidates for one stage of the plan's ranking block."""
        epc = {eid: score_epc(e)["epc_score"] for eid, e in all_edges.items()}
        candidates = collect_candidates(
            executions, path_priority=plan_input.path_priority,
            epc_by_edge=epc, literature_by_edge=literature or None,
        )
        ranker = CandidateRanker(
            self.ranking_plan.spec(stage),
            stage=stage,
            reranker=self.llm if use_reranker else None,
            rerank_top_k=self.args.rerank_top_k,
            quiet_warnings=preview,
            verbose=self.verbose,
        )
        return ranker.rank(candidates, question=plan_input.question, edges=all_edges)

    # -- main --------------------------------------------------------------

    def run(self) -> int:
        # Counted before the first step is printed, so the numbering does not
        # change part-way through. The plan is read twice, which costs nothing
        # against the queries that follow.
        total = 3 if self.args.check else 9
        if not self.args.check and self.args.explain:
            try:
                with open(self.args.plan) as f:
                    if json.load(f).get("explanation_queries"):
                        total = 10
            except Exception:
                pass

        self.step(1, total, "Loading and validating plan")
        plan_input = self.load()

        if plan_input.refused:
            self.log("\nThe planner refused this question; nothing to execute.")
            self._emit(build_result(
                plan_input=plan_input, executions={}, ranked=[],
                started_at=self.started_at, elapsed_s=time.time() - self.t0,
            ))
            return 0

        if not plan_input.can_execute:
            self.log("\nPlan cannot be executed. Nothing was queried.")
            self._emit(build_result(
                plan_input=plan_input, executions={}, ranked=[],
                started_at=self.started_at, elapsed_s=time.time() - self.t0,
            ))
            return 1

        if self.args.check:
            self.step(2, total, "Connectivity check")
            self.llm = self.build_llm()
            self.step(3, total, "Plan is executable")
            self.log("  --check requested; stopping before any query.")
            return 0

        self.step(2, total, "Connecting to the LLM")
        self.llm = self.build_llm()
        self.client = self.build_client()

        self.step(3, total, "Checking hops against the ARAX meta knowledge graph")
        if not self.args.meta_check:
            self.log("  skipped by request")
        elif not self.check_meta_kg(plan_input):
            self._emit(build_result(
                plan_input=plan_input, executions={}, ranked=[],
                started_at=self.started_at, elapsed_s=time.time() - self.t0,
            ))
            return 1

        self.step(4, total, "Resolving entities")
        resolutions = self.resolve(plan_input)
        if not plan_input.can_execute:
            self.log("\nNo path has a resolved anchor; nothing can be queried.")
            self._emit(build_result(
                plan_input=plan_input, executions={}, ranked=[],
                resolutions=resolutions, started_at=self.started_at,
                elapsed_s=time.time() - self.t0,
            ))
            return 1

        self.step(5, total, f"Executing {len(plan_input.active_paths)} path(s)")
        executions = self.execute(plan_input, resolutions)

        self.step(6, total, "Checking results are about the intended concepts")
        concept_checks = self.check_concepts(plan_input, resolutions, executions)

        self.step(7, total, "Applying evidence policy")
        distributions, filters, all_edges = self.filter(plan_input, executions)

        self.step(8, total, "Ranking candidates")
        self.ranking_plan = RankingPlan.from_plan(plan_input.raw.get("ranking"))
        for w in self.ranking_plan.warnings:
            self.log(f"  warning: {w}")
        self.log(f"  explanation_influence={self.ranking_plan.explanation_influence}")
        preview = self.rank(
            plan_input, executions, all_edges, None, use_reranker=False,
            preview=self.args.verify_literature and not self.args.mock,
        )

        literature, literature_summary = {}, {}
        if self.args.verify_literature and preview and not self.args.mock:
            self.log("  verifying literature support")
            literature, literature_summary = self.verify_literature(
                executions, all_edges, preview,
            )
            all_edges = {}
            for execution in executions.values():
                all_edges.update(execution.kg_edges)

        ranked = self.rank(
            plan_input, executions, all_edges, literature,
            use_reranker=self.args.rerank and not self.args.mock,
        )

        explanations: Dict[str, Any] = {}
        explanation_summaries: List[Dict[str, Any]] = []
        if plan_input.runnable_explanation_queries and self.args.explain:
            # Under `none` the explanation is fixed-endpoint context shared by
            # every candidate — it is still run and reported, but attaching it
            # per candidate would imply a distinction it cannot make.
            self.step(9, total, "Explaining candidates")
            explanations, explanation_summaries = Explainer(
                client=self.client, verbose=self.verbose,
            ).run_all(plan_input, resolutions, executions, ranked)

        self.step(total, total, "Assembling result")
        # `annotate_only` and `rerank` attach explanations to candidates;
        # `none` keeps them as standalone context, because a shared
        # fixed-endpoint route says the same thing about every candidate.
        attachable = (
            explanations if self.ranking_plan.attaches_explanations else {}
        )
        result = build_result(
            explanations=attachable,
            plan_input=plan_input, executions=executions, ranked=ranked,
            resolutions=resolutions, edges=all_edges, filter_results=filters,
            distributions=distributions, literature=literature,
            literature_summary=literature_summary, concept_checks=concept_checks,
            ledger=build_ledger(
                arax_client=self.client, cache=self.cache, resolver=self.resolver,
                evidence_gatherer=self.evidence, llm_agent=self.llm,
                executor_config={
                    "batch_size": self.args.batch_size,
                    "max_results": self.args.max_results,
                    "skip_direct_over_hops": self.args.skip_direct_over_hops,
                    "mock": self.args.mock,
                },
            ),
            started_at=self.started_at, elapsed_s=time.time() - self.t0,
        )
        if explanation_summaries or explanations:
            # Paths are attached to candidates when the keys line up, but an
            # explanation query whose endpoint is not itself a candidate — a
            # target gene, say, rather than a drug — produces routes that
            # belong to no result row. Recorded here so they are reported
            # rather than computed and discarded.
            attached = {
                r["curie"] for r in result.get("results", [])
                if r.get("explanation_paths")
            }
            unattached = {k: v for k, v in explanations.items() if k not in attached}
            result.setdefault("evidence", {})["explanations"] = {
                "summaries": explanation_summaries,
                "paths_by_endpoint": unattached,
            }
        self._emit(result)
        return 0 if result["verdict"] in ("success", "no_answer") else 1

    def _emit(self, result: Dict[str, Any]) -> None:
        path = write_result(result, self.args.out)
        self.log("\n" + "=" * 62)
        self.log("\n".join(result_summary_lines(result)))

        hints = controller_hints(result)
        if hints["actionable"]:
            self.log("\nNext steps:")
            for item in hints["actionable"]:
                self.log(f"  - {item}")

        if self.args.dump_queries:
            queries = {
                pid: path.get("submitted_queries", [])
                for pid, path in (result.get("paths") or {}).items()
            }
            write_result(queries, self.args.dump_queries)
            self.log(f"\nqueries -> {self.args.dump_queries}")

        if self.args.hints:
            write_result(hints, self.args.hints)
            self.log(f"\nhints  -> {self.args.hints}")
        self.log(f"result -> {path}")
        self.cache.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Execute a query plan against ARAX.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exit codes:\n"
            "  0  success, or a well-established no_answer\n"
            "  1  inconclusive, unexecutable, or an invalid plan\n"
            "  2  the LLM was unreachable (resolution cannot proceed)\n"
        ),
    )
    ap.add_argument("plan", help="Path to the plan JSON")
    ap.add_argument("--out", default="runs/result.json")
    ap.add_argument("--hints", help="Also write the controller hints here")
    ap.add_argument("--dump-queries", metavar="PATH",
                    help="Write the exact TRAPI query graphs submitted to ARAX, "
                         "for inspecting what was actually asked")
    ap.add_argument("-q", "--quiet", action="store_true")

    g = ap.add_argument_group("endpoints")
    g.add_argument("--arax-url", default=config.ARAX_QUERY_URL)
    g.add_argument("--ollama-url", default="http://localhost:11434/v1/chat/completions")
    g.add_argument("--model", default="gpt-oss:120b")

    g = ap.add_argument_group("execution")
    g.add_argument("--timeout", type=float, default=config.REQUEST_TIMEOUT,
                   help="seconds before a query is treated as timed out; "
                        "fractional values are accepted, which is mainly "
                        "useful for forcing the decomposition path in testing")
    g.add_argument("--batch-size", type=int, default=config.DEFAULT_BATCH_SIZE,
                   help="intermediates pinned per decomposed query")
    g.add_argument("--max-results", type=int, default=config.DEFAULT_MAX_RESULTS)
    g.add_argument("--no-meta-check", dest="meta_check", action="store_false",
                   help="do not check hops against ARAX's meta knowledge "
                        "graph first; a plan naming an unsupported triple will "
                        "then run and return nothing")
    g.add_argument("--no-localize-empty", dest="localize_empty",
                   action="store_false",
                   help="do not decompose a multi-hop query that returned "
                        "nothing; without this the report cannot say which "
                        "hop was empty")
    g.add_argument("--skip-direct-over-hops", type=int, default=None,
                   help="decompose immediately for paths with at least this "
                        "many hops, rather than waiting for a timeout")

    g = ap.add_argument_group("llm")
    g.add_argument("--llm-timeout", type=float, default=120)
    g.add_argument("--no-confirm-single", action="store_true",
                   help="skip LLM confirmation when only one candidate resolves")
    g.add_argument("--taxon", action="append", default=None,
                   metavar="NCBITaxon:9606",
                   help="restrict name resolution to these taxa "
                        "(default: human). Repeat for several, or pass "
                        "--taxon '' to allow any species.")
    g.add_argument("--conflate", action="append", default=[],
                   choices=["gene_protein", "drug_chemical"],
                   help="treat these category groups as interchangeable. "
                        "gene_protein is already applied to Gene/Protein "
                        "entities by default; see --no-default-conflation")
    g.add_argument("--no-default-conflation", dest="default_conflation",
                   action="store_false",
                   help="do not apply gene_protein conflation automatically "
                        "to Gene/Protein entities. A plan's category then "
                        "pins the lookup to one molecular form, which is how "
                        "this behaved before the default existed; use it to "
                        "reproduce an older run.")
    g.add_argument("--no-rerank", dest="rerank", action="store_false",
                   help="skip the LLM rerank; the plan's criteria decide order")
    g.add_argument("--no-explain", dest="explain", action="store_false",
                   help="skip the plan's explanation queries. Each runs one "
                        "ARAX connect() per candidate, which is the most "
                        "expensive stage here.")
    g.add_argument("--no-verify-literature", dest="verify_literature",
                   action="store_false", help="skip abstract verification")
    g.add_argument("--verify-top-k", type=int, default=20)
    g.add_argument("--rerank-top-k", type=int, default=20)
    g.add_argument("--max-abstracts", type=int, default=5,
                   help="abstracts read per edge")
    g.add_argument("--max-verify-edges", type=int, default=200)

    g = ap.add_argument_group("cache")
    g.add_argument("--cache", default=config.CACHE_PATH)
    g.add_argument("--no-cache", action="store_true")

    g = ap.add_argument_group("modes")
    g.add_argument("--check", action="store_true",
                   help="validate the plan and the LLM, then stop")
    g.add_argument("--mock", action="store_true",
                   help="run offline against synthetic data")
    g.add_argument("--lenient", action="store_true",
                   help="treat schema errors as warnings (Biolink validation "
                        "still runs)")

    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    try:
        return Runner(args).run()
    except KeyboardInterrupt:
        print("\ninterrupted; cached work is preserved for the next run",
              file=sys.stderr)
        return 130
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Could not load plan: {e}", file=sys.stderr)
        return 1
    except LLMError as e:
        print(f"LLM error: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
