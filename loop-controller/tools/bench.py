#!/usr/bin/env python3
"""
bench.py — Run a list of questions through the loop and summarise what happened.

The first real run against a live graph and a live model is the point where
most of the open questions get answered, and reading fifteen traces by hand is
how that turns into a whole afternoon. This runs the list, collects the numbers
that matter into one table, and flags the things worth looking at.

    python tools/bench.py questions.txt --out runs/bench \\
        --executor-pythonpath ../plan-core/src \\
        --executor-pythonpath ../plan-executor/src

Questions file: one per line. Blank lines and `#` comments are skipped. An
optional expected outcome after a `|` is compared against what happened:

    Which drugs treat dermatitis herpetiformis? | answered
    Which drugs reverse the TNBC signature?     | absence
    What should I take for my rash?             | refused

Deterministic by default (`--no-llm-controller`), because the first run should
tell you about the graph and the planner, not about a decision maker's mood. Add
`--llm-controller` for a second pass once the first is understood.

The summary answers the questions Tier 2 asks:

  * Did the planner ever return a plan that would query the same thing? That is
    the likeliest failure in the system, and `duplicate_revisions` counts it.
  * Did the cheap `--check` probe earn its place? `probe_caught` counts the runs
    where it saved a search.
  * Where does the model spend go? Broken down by component.
  * How long does a search really take? The timeout ladder was guessed; the
    elapsed column is what it should have been tuned against.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]      # loop-controller/
REPO = ROOT.parent                              # the checkout holding the siblings

# The four packages sit beside each other, and none of them has to be pip
# installed for this to work. Note the folder-vs-package mismatch that is easy
# to trip over: the directory is `planner`, the importable package inside it is
# `planner_agent`. The planner also imports `plan_core`, so that one is needed
# even though the bench never touches it directly.
for _src in (
    ROOT / "src",
    REPO / "plan-core" / "src",
    REPO / "planner" / "src",
    REPO / "plan-executor" / "src",
):
    if _src.is_dir() and str(_src) not in sys.path:
        sys.path.insert(0, str(_src))

from loop_controller.contracts import Budget                  # noqa: E402
from loop_controller.executor_cli import (                    # noqa: E402
    ExecutorSettings, ExecutorUnavailable, SubprocessExecutor,
)
from loop_controller.loop import ControllerConfig, LoopController  # noqa: E402


def parse_questions(path: Path) -> List[Tuple[str, Optional[str]]]:
    out: List[Tuple[str, Optional[str]]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            question, _, expected = line.partition("|")
            out.append((question.strip(), expected.strip() or None))
        else:
            out.append((line, None))
    return out


def build_planner(args: argparse.Namespace) -> Optional[Any]:
    """The real planner, or None with a clear reason.

    Without one the bench still runs plans you supply by hand, but it cannot
    answer the question it exists to answer — whether the planner produces a
    genuinely different plan when asked to revise.
    """
    if args.planner_src:
        for extra in args.planner_src:
            if extra not in sys.path:
                sys.path.insert(0, extra)

    try:
        from planner_agent import PlannerAgent
        from planner_agent.ollama_client import OllamaGPTOSSClient
        from loop_controller.planner_port import PlannerAgentPort
    except ModuleNotFoundError as exc:
        # Say what is missing and how to fix it. "No module named X" on its own
        # sends people looking for a bug in the wrong package.
        hint = {
            "planner_agent": (
                "the planner package was not found. Its folder is `planner` "
                "and the package inside is `planner_agent`, so the path to add "
                "is `planner/src`. This script looks for it beside "
                f"{REPO}; use --planner-src if your checkout is elsewhere."
            ),
            "plan_core": (
                "the shared plan contract was not found; expected "
                f"{REPO / 'plan-core' / 'src'}."
            ),
            "ollama": (
                "the planner's Ollama client needs the `ollama` package: "
                "pip install ollama"
            ),
        }.get(exc.name, str(exc))
        print(f"\nno planner available — {hint}\n", file=sys.stderr)
        return None
    except Exception as exc:
        print(f"\nno planner available: {exc}\n", file=sys.stderr)
        return None

    port = PlannerAgentPort(PlannerAgent(
        llm=OllamaGPTOSSClient(model=args.model, host=args.ollama_host),
    ))
    if not port.borrowed_helpers:
        print(
            "note: the planner's own parse/stamp/validate helpers were not "
            "found, so a local reimplementation is in use. Revisions are "
            "validated slightly more loosely than plans.",
            file=sys.stderr,
        )
    return port


def run_one(
    question: str,
    expected: Optional[str],
    index: int,
    args: argparse.Namespace,
    planner: Optional[Any],
) -> Dict[str, Any]:
    out_dir = Path(args.out) / f"q{index:02d}"

    executor = SubprocessExecutor(ExecutorSettings(
        pythonpath=list(args.executor_pythonpath),
        cwd=args.executor_cwd,
        runs_dir=str(out_dir),
        cache=args.cache,
        base_args=list(args.executor_arg),
        mock=args.dry_run,
        quiet=True,
        process_timeout_s=args.process_timeout,
    ))

    decision_maker = None
    if args.llm_controller:
        from loop_controller.decide import LLMDecisionMaker
        from planner_agent.ollama_client import OllamaGPTOSSClient
        decision_maker = LLMDecisionMaker(
            OllamaGPTOSSClient(model=args.model, host=args.ollama_host)
        )

    controller = LoopController(
        executor=executor,
        planner=planner,
        decision_maker=decision_maker,
        config=ControllerConfig(
            budget=Budget(
                max_iterations=args.max_iterations,
                max_planner_calls=args.max_planner_calls,
                max_wall_clock_s=args.max_wall_clock,
            ),
            runs_dir=str(out_dir),
            trace_path=str(out_dir / "trace.json"),
            answer_path=str(out_dir / "answer.json"),
            verbose=args.verbose,
        ),
    )

    # `--plan` fixes the *opening* plan only. The planner is still built and
    # still handles every revision, so a hand-written over-constrained plan is
    # a direct way to reach `relax_plan` — which natural-language questions
    # turn out to be very bad at arranging.
    fixed_plan = (
        json.loads(Path(args.plan).read_text(encoding="utf-8"))
        if args.plan else None
    )

    started = time.time()
    try:
        outcome = controller.run(question, plan=fixed_plan)
    except ExecutorUnavailable as exc:
        return {
            "index": index, "question": question, "expected": expected,
            "status": "executor_unavailable", "error": str(exc),
            "elapsed_s": round(time.time() - started, 1),
        }
    except Exception as exc:                       # noqa: BLE001 - a bench
        return {
            "index": index, "question": question, "expected": expected,
            "status": "crashed", "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": round(time.time() - started, 1),
        }

    return summarise_run(outcome, controller, question, expected, index, started)


def summarise_run(
    outcome: Any,
    controller: LoopController,
    question: str,
    expected: Optional[str],
    index: int,
    started: float,
) -> Dict[str, Any]:
    state = outcome.state
    answer = outcome.answer or {}
    grounding = answer.get("grounding") or {}
    notes = controller.trace.notes

    attempts = state.attempts if state else []
    actions = [a.decision.action for a in attempts if a.decision]

    # What happened to each revision that was asked for. "No duplicates" is
    # only good news if the revisions actually produced a usable plan, and the
    # planner's note is the only place that distinction lives.
    revisions = []
    for a in attempts:
        if not a.decision or a.decision.action not in ("relax_plan", "repair_plan"):
            continue
        note = a.planner_note or ""
        if "did not validate" in note:
            verdict = "invalid"
        elif "declined to revise" in note:
            verdict = "planner_refused"
        elif "differs in what it would query" in note:
            verdict = "duplicate"
        elif "budget spent" in note:
            verdict = "budget"
        else:
            verdict = "produced_a_plan"
        revisions.append({
            "action": a.decision.action, "verdict": verdict, "note": note,
        })

    return {
        "index": index,
        "question": question,
        "expected": expected,
        "status": outcome.status,
        "matched_expectation": (expected is None or expected == outcome.status),
        "reason": outcome.reason,
        "answer_kind": answer.get("answer_kind"),
        "results": len(answer.get("candidates") or []),
        "iterations": state.iteration if state else 0,
        "actions": actions,
        "revisions": revisions,
        # The number Tier 2 is really asking about.
        "duplicate_revisions": sum(
            1 for n in notes if "rejected duplicate revision" in n
        ),
        # The same question for the one-axis rule. `rejected` counts the
        # revisions the planner corrected after being told; `accepted` counts
        # the ones it would not correct, which are used anyway and cost the
        # answer its attribution claim. A run with many of the second is a run
        # whose relaxation prompt is not landing.
        "over_broad_rejected": sum(
            1 for n in notes if "rejected over-broad relaxation" in n
        ),
        "over_broad_accepted": len(state.over_relaxations) if state else 0,
        # What the model chose against what the policy would have chosen on the
        # same iteration. The loop evaluates the deterministic default every
        # time whether or not it uses it, precisely so this is free — and it is
        # the only thing that says whether the LLM decision maker is earning
        # its calls. Recorded per iteration so a disagreement can be read
        # rather than just counted.
        "decisions": [
            {
                "iteration": c["iteration"],
                "chosen": c["chosen"],
                "policy_would_choose": c["policy_would_choose"],
                "by": c["chosen_by"],
            }
            for c in controller.trace.counterfactual
        ],
        "disagreements": [
            c for c in controller.trace.counterfactual if not c["agreed"]
        ],
        # Empty results that were not reported as absences, because the query's
        # own anchor could not have matched. Each one is a confidently wrong
        # answer that did not get served.
        "anchor_mismatches": sum(
            len(a.diagnosis.anchor_category_mismatch)
            for a in attempts if a.diagnosis
        ),
        # Zero mismatches means "every anchor was sound" only if the check
        # actually ran. Without this, a run missing plan-core reports a
        # clean bill of health it never examined.
        "anchor_check_unavailable": next(
            (a.diagnosis.anchor_check_unavailable for a in attempts
             if a.diagnosis and a.diagnosis.anchor_check_unavailable), None,
        ),
        "probe_caught": sum(1 for n in notes if "--check probe found" in n),
        "planner_calls": state.planner_calls if state else 0,
        "arax_calls": state.arax_calls if state else 0,
        "llm_calls": state.llm_calls if state else 0,
        "llm_by_component": {
            "executor": state.executor_llm_calls if state else 0,
            "controller": state.controller_llm_calls if state else 0,
            "planner": state.planner_llm_calls if state else 0,
        },
        "grounded": grounding.get("ok"),
        "statements": grounding.get("checked"),
        "withheld": grounding.get("failed"),
        # The detail, not just the count. A withheld statement is a bug in the
        # composer or in the gate, and which one it is depends entirely on what
        # the violation says — so reading it should not need a second run.
        "violations": (grounding.get("violations") or [])[:20],
        "warnings": list(controller.trace.warnings),
        "elapsed_s": round(time.time() - started, 1),
    }


def _violation_samples(rows: List[Dict[str, Any]], per_kind: int = 3) -> Dict[str, List[str]]:
    """One or two examples of each violation kind.

    The kind alone says which check fired; the detail says which identifier or
    edge tripped it, and that is what actually locates the bug.
    """
    samples: Dict[str, List[str]] = {}
    for row in rows:
        for violation in row.get("violations") or []:
            kind = violation.get("kind", "?")
            bucket = samples.setdefault(kind, [])
            if len(bucket) < per_kind:
                bucket.append(violation.get("detail", ""))
    return samples


def _decision_maker_totals(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Whether the model is choosing differently from the policy, and where.

    An agreement rate of 1.0 is not a pass mark — it means the model cost a
    call per iteration and changed nothing, and the loop would behave
    identically with `--llm-controller` off. The interesting number is the
    disagreements, and they are listed rather than counted because each one is
    either the adaptivity the model was added for or a bug in the legal-action
    reporting, and only reading it tells you which.
    """
    decisions = [d for r in rows for d in (r.get("decisions") or [])]
    if not decisions:
        return {"decisions": 0, "note": "no decision was recorded"}

    by_llm = [d for d in decisions if d.get("by") == "llm"]
    disagreements = [
        {"q": r["question"], **d}
        for r in rows for d in (r.get("disagreements") or [])
    ]
    agreed = len(decisions) - len(disagreements)
    return {
        "decisions": len(decisions),
        "chosen_by_llm": len(by_llm),
        "agreed_with_policy": agreed,
        "agreement_rate": round(agreed / len(decisions), 3),
        "disagreements": disagreements,
    }


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    done = [r for r in rows if r.get("iterations")]
    elapsed = [r["elapsed_s"] for r in rows if r.get("elapsed_s")]
    return {
        "questions": len(rows),
        "status_counts": dict(Counter(r["status"] for r in rows)),
        "expectation_misses": [
            {"q": r["question"], "expected": r["expected"], "got": r["status"]}
            for r in rows
            if r.get("expected") and not r.get("matched_expectation")
        ],
        "duplicate_revisions_total": sum(
            r.get("duplicate_revisions", 0) for r in rows
        ),
        "questions_with_a_duplicate": sum(
            1 for r in rows if r.get("duplicate_revisions", 0)
        ),
        "probe_caught_total": sum(r.get("probe_caught", 0) for r in rows),
        "over_broad_relaxations": {
            "rejected_and_corrected": sum(
                r.get("over_broad_rejected", 0) for r in rows
            ),
            "accepted_with_attribution_withdrawn": sum(
                r.get("over_broad_accepted", 0) for r in rows
            ),
        },
        "anchor_mismatches_total": sum(r.get("anchor_mismatches", 0) for r in rows),
        "anchor_check_unavailable": next(
            (r["anchor_check_unavailable"] for r in rows
             if r.get("anchor_check_unavailable")), None,
        ),
        "decision_maker": _decision_maker_totals(rows),
        "ungrounded_answers": [
            r["question"] for r in rows if r.get("grounded") is False
        ],
        "withheld_statements_total": sum(r.get("withheld") or 0 for r in rows),
        "violation_kinds": dict(Counter(
            v.get("kind") for r in rows for v in (r.get("violations") or [])
        )),
        "violation_samples": _violation_samples(rows),
        "action_counts": dict(Counter(
            a for r in rows for a in r.get("actions", [])
        )),
        # A revision was asked for. Without any of these, the duplicate count
        # being zero says nothing at all.
        "revisions_attempted": sum(
            len(r.get("revisions") or []) for r in rows
        ),
        # The distinction the headline needs: asked-for is not the same as
        # worked.
        "revision_verdicts": dict(Counter(
            v["verdict"] for r in rows for v in (r.get("revisions") or [])
        )),
        "cost": {
            "arax_calls": sum(r.get("arax_calls", 0) for r in rows),
            "llm_calls": sum(r.get("llm_calls", 0) for r in rows),
            "llm_by_component": {
                key: sum((r.get("llm_by_component") or {}).get(key, 0) for r in rows)
                for key in ("executor", "controller", "planner")
            },
        },
        "elapsed_s": {
            "total": round(sum(elapsed), 1),
            "median": round(statistics.median(elapsed), 1) if elapsed else 0,
            "max": round(max(elapsed), 1) if elapsed else 0,
        },
        "iterations": {
            "median": statistics.median([r["iterations"] for r in done]) if done else 0,
            "max": max([r["iterations"] for r in done]) if done else 0,
        },
    }


def render(rows: List[Dict[str, Any]], totals: Dict[str, Any]) -> str:
    lines = ["# Bench run", ""]

    lines.append("| # | status | iters | actions | results | ARAX | LLM | s |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        flag = "" if r.get("matched_expectation", True) else " ⚠"
        lines.append(
            f"| {r['index']} | {r['status']}{flag} | {r.get('iterations', 0)} | "
            f"{' → '.join(r.get('actions') or []) or '—'} | "
            f"{r.get('results', 0)} | {r.get('arax_calls', 0)} | "
            f"{r.get('llm_calls', 0)} | {r.get('elapsed_s', 0)} |"
        )

    lines += ["", "## Totals", "", "```json",
              json.dumps(totals, indent=2), "```", ""]

    lines.append("## What to look at first")
    lines.append("")
    dupes = totals["questions_with_a_duplicate"]
    asked = totals["revisions_attempted"]
    if dupes:
        lines.append(
            f"- **{dupes} question(s) had a revision rejected as a duplicate.** "
            f"The planner returned a plan that would query the same thing. This "
            f"is the failure the loop was most likely to hit; the revision "
            f"prompt needs work."
        )
    elif asked:
        verdicts = totals.get("revision_verdicts") or {}
        worked = verdicts.get("produced_a_plan", 0)
        if worked == asked:
            lines.append(
                f"- All {asked} revision(s) produced a usable new plan. The "
                f"planner moved when told to, which was the biggest open "
                f"question."
            )
        else:
            failed = {k: v for k, v in verdicts.items() if k != "produced_a_plan"}
            lines.append(
                f"- **{asked} revision(s) asked for, {worked} produced a usable "
                f"plan.** No duplicates, but that is not the same as working: "
                f"{failed}. Read the `revisions` field on those rows — the "
                f"planner's note says what went wrong."
            )
    else:
        lines.append(
            "- **No revision was ever asked for**, so this run says nothing "
            "about whether the planner can revise. Every question either "
            "answered first time or failed in a way no new plan would fix. Add "
            "questions that should come back empty."
        )
    # Said before the over-relaxation numbers, because without any relaxation
    # those numbers are zero for the wrong reason and read as a pass.
    relaxed = (totals.get("action_counts") or {}).get("relax_plan", 0)
    if not relaxed:
        lines.append(
            "- **No question ever reached `relax_plan`**, so the one-axis rule "
            "is still completely unmeasured and the zeros below mean nothing. "
            "Relaxation needs a result that is empty *and* complete *and* has "
            "a loosenable axis; a plan the executor marks replannable goes to "
            "`repair_plan` instead and never gets there. Natural-language "
            "questions are a poor way to arrange that — use `--plan` with a "
            "deliberately over-tight plan."
        )

    over_broad = totals.get("over_broad_relaxations") or {}
    stuck = over_broad.get("accepted_with_attribution_withdrawn", 0)
    corrected = over_broad.get("rejected_and_corrected", 0)
    if stuck:
        lines.append(
            f"- **{stuck} relaxation(s) changed more than the one constraint "
            f"asked for, twice.** The plans were used — they were valid and "
            f"new — but those answers cannot be attributed to a single loosened "
            f"constraint, and say so in their caveats. Read `over_relaxations` "
            f"in the trace: the `off_axis` list names the fields the planner "
            f"also moved, which is what the relaxation prompt has to stop."
        )
    elif corrected:
        lines.append(
            f"- {corrected} relaxation(s) came back over-broad and were "
            f"corrected on the retry. The one-axis rule held; the prompt could "
            f"still be clearer about it."
        )
    if totals.get("anchor_check_unavailable"):
        lines.append(
            f"- **The anchor-category check did not run**, so the zero above is not a clean bill of health \u2014 nothing was examined, and absences were reported on anchors nobody checked: {totals['anchor_check_unavailable']}"
        )
    elif totals.get("anchor_mismatches_total"):
        lines.append(
            f"- **{totals['anchor_mismatches_total']} anchor(s) were pinned to "
            f"a category their resolved identifier does not satisfy.** Those "
            f"queries could not have matched anything, so no absence was "
            f"claimed from them. Each one is a planner bug worth reading: the "
            f"plan named a concept and then asked for the wrong kind of thing."
        )
    dm = totals.get("decision_maker") or {}
    if dm.get("chosen_by_llm"):
        rate = dm.get("agreement_rate")
        disagreements = dm.get("disagreements") or []
        if not disagreements:
            lines.append(
                f"- **The model agreed with the policy on all "
                f"{dm['decisions']} decision(s).** That is not a pass mark: it "
                f"means `--llm-controller` cost {dm['chosen_by_llm']} model "
                f"call(s) and changed nothing. Either the questions never "
                f"reached a genuinely ambiguous state, or the policy ordering "
                f"is already right and the model is not needed."
            )
        else:
            lines.append(
                f"- The model disagreed with the policy on "
                f"{len(disagreements)} of {dm['decisions']} decision(s) "
                f"(agreement {rate}). Each one is either the adaptivity the "
                f"model was added for or a gap in how the legal actions were "
                f"described to it:"
            )
            for d in disagreements[:6]:
                lines.append(
                    f"    - `{d.get('chosen')}` instead of "
                    f"`{d.get('policy_would_choose')}` — {d.get('q', '')[:60]}"
                )
    if totals["probe_caught_total"]:
        lines.append(
            f"- The `--check` probe caught {totals['probe_caught_total']} "
            f"broken plan(s) before any search ran."
        )
    if totals["ungrounded_answers"]:
        lines.append(
            f"- **{len(totals['ungrounded_answers'])} answer(s) failed the "
            f"grounding check**, withholding "
            f"{totals['withheld_statements_total']} statement(s). Kinds: "
            f"{totals['violation_kinds']}. Examples:"
        )
        for kind, details in (totals.get("violation_samples") or {}).items():
            for detail in details:
                lines.append(f"    - `{kind}` — {detail}")
    if totals["expectation_misses"]:
        lines.append(
            f"- {len(totals['expectation_misses'])} question(s) did not match "
            f"the outcome you expected."
        )
    lines.append(
        f"- Longest run {totals['elapsed_s']['max']}s, median "
        f"{totals['elapsed_s']['median']}s. The retry ladder is "
        f"120/300/600s — tune it against these."
    )
    return "\n".join(lines)


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("questions", help="file of questions, one per line")
    ap.add_argument("--out", default="runs/bench")
    ap.add_argument(
        "--plan",
        help="use this plan as the opening plan for every question instead of "
             "asking the planner for one. Revisions still go through the real "
             "planner, which is what makes a hand-written over-tight plan a "
             "way to exercise relaxation.",
    )
    ap.add_argument(
        "--no-planner", action="store_true",
        help="run without a planner at all. Repair and relax become illegal, "
             "so the loop can only accept, report an absence or abandon — for "
             "measuring the executor, not the loop.",
    )

    ap.add_argument("--model", default="gpt-oss:120b")
    ap.add_argument("--ollama-host", default=None)
    ap.add_argument(
        "--planner-src", action="append", default=[],
        help="extra path(s) holding the planner package; only needed when the "
             "checkout is not laid out as siblings. Repeatable.",
    )
    ap.add_argument(
        "--llm-controller", action="store_true",
        help="let a model choose the moves; off by default so the first run is "
             "reproducible",
    )

    ap.add_argument("--executor-pythonpath", action="append", default=[])
    ap.add_argument("--executor-arg", action="append", default=[])
    ap.add_argument("--executor-cwd", default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--process-timeout", type=float, default=1800.0)
    ap.add_argument("--dry-run", action="store_true",
                    help="executor mock mode: offline, synthetic data")

    ap.add_argument("--max-iterations", type=int, default=4)
    ap.add_argument("--max-planner-calls", type=int, default=3)
    ap.add_argument("--max-wall-clock", type=float, default=1800.0)
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    questions = parse_questions(Path(args.questions))
    if not questions:
        print("no questions found", file=sys.stderr)
        return 1

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # Built once. Constructing a PlannerAgent loads the Biolink model and
    # compiles the plan schema, which is slow enough that doing it per question
    # would dominate a short run.
    #
    # Built even with `--plan`, which fixes only the *opening* plan. This used
    # to skip the planner entirely, which quietly made `--plan` useless for the
    # one thing it is best at: handing the loop a deliberately over-constrained
    # plan and watching what the planner does when asked to loosen it. Without
    # a planner, `relax_plan` and `repair_plan` are both refused — "no planner
    # is configured to choose the replacement" — so such a run abandons on the
    # first empty result and reports nothing about relaxation at all. Three
    # probe runs were spent discovering that.
    planner = None if args.no_planner else build_planner(args)

    # Stop now rather than after fifteen identical failures. Without a planner
    # and without --plan there is nothing to run, and finding that out one
    # question at a time helps nobody.
    if planner is None and not args.plan:
        print(
            "nothing to run: no planner could be loaded, and no --plan was "
            "given. Fix the planner path above, or pass --plan to run one "
            "fixed plan for every question.",
            file=sys.stderr,
        )
        return 1

    rows: List[Dict[str, Any]] = []
    for index, (question, expected) in enumerate(questions, start=1):
        print(f"[{index}/{len(questions)}] {question[:70]}", flush=True)
        row = run_one(question, expected, index, args, planner)
        rows.append(row)
        print(
            f"    {row['status']} · {row.get('iterations', 0)} iter · "
            f"{row.get('elapsed_s', 0)}s",
            flush=True,
        )
        # Written after every question, so an interrupted run still has
        # everything up to that point.
        (out / "results.json").write_text(
            json.dumps(rows, indent=2, default=str), encoding="utf-8",
        )

    totals = aggregate(rows)
    (out / "summary.json").write_text(
        json.dumps(totals, indent=2, default=str), encoding="utf-8",
    )
    summary = render(rows, totals)
    (out / "summary.md").write_text(summary, encoding="utf-8")

    print("\n" + summary)
    print(f"\nwritten to {out}/summary.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
