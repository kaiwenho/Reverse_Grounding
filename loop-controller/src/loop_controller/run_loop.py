"""
run_loop.py — Command-line interface.

    python -m loop_controller.run_loop --question "..." --out runs/
    python -m loop_controller.run_loop --plan plan.json --no-llm-controller

Three things about the argument surface are deliberate.

The executor's options are not re-exposed. Anything this CLI does not name is
passed through with `--executor-arg`, so the two programs cannot drift into
disagreeing about what `--timeout` means, and a deployment's standing choices
live in one place.

`--no-llm-controller` runs the deterministic policy. It is the reproducible
mode: the same inputs give the same trace, which is what an evaluation run and
a bug report both need.

`--dry-run` uses the executor's own `--mock`, which answers offline with
synthetic data and simulates a timeout on multi-hop queries. That exercises the
loop's least-tested path — decomposition, then diagnosis of an unfinished run —
without a knowledge graph or a model anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .compose import ComposerSettings, render_text
from .contracts import Budget, LOOP_ABSENCE, LOOP_ANSWERED, LOOP_REFUSED
from .executor_cli import ExecutorSettings, SubprocessExecutor
from .loop import ControllerConfig, LoopController
from .decide import LLMDecisionMaker, PolicyDecisionMaker
from .trace import summary_lines


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="loop-controller",
        description="Run a question to a graph-grounded answer.",
    )

    source = ap.add_argument_group("input")
    source.add_argument("--question", help="natural-language question")
    source.add_argument("--plan", help="a plan JSON file, instead of planning one")
    source.add_argument(
        "--available-inputs", metavar="PATH",
        help="JSON file holding the authoritative external-input manifest",
    )

    out = ap.add_argument_group("output")
    out.add_argument("--out", default="runs", help="directory for run artefacts")
    out.add_argument("--answer", help="where to write answer.json")
    out.add_argument("--trace", help="where to write trace.json")
    out.add_argument("-q", "--quiet", action="store_true")

    budget = ap.add_argument_group("budget")
    budget.add_argument("--max-iterations", type=int, default=4)
    budget.add_argument("--max-planner-calls", type=int, default=3)
    budget.add_argument("--max-wall-clock", type=float, default=1800.0)
    budget.add_argument("--max-arax-calls", type=int, default=200)
    budget.add_argument("--max-llm-calls", type=int, default=400)

    controller = ap.add_argument_group("controller")
    controller.add_argument(
        "--no-llm-controller", dest="llm_controller", action="store_false",
        help="choose moves with the deterministic policy; fully reproducible",
    )
    controller.add_argument("--model", default="gpt-oss:120b")
    controller.add_argument("--ollama-host", default=None)

    executor = ap.add_argument_group("executor")
    executor.add_argument(
        "--executor-python", default=sys.executable,
        help="interpreter that can import plan_executor",
    )
    executor.add_argument(
        "--executor-pythonpath", action="append", default=[],
        help="prepended to PYTHONPATH for the executor; repeatable",
    )
    executor.add_argument(
        "--executor-arg", action="append", default=[], metavar="ARG",
        help="passed through to run_plan.py verbatim; repeatable",
    )
    executor.add_argument("--executor-cwd", default=None)
    executor.add_argument("--cache", default=None)
    executor.add_argument(
        "--process-timeout", type=float, default=1800.0,
        help="wall-clock ceiling for one executor process",
    )
    executor.add_argument(
        "--dry-run", action="store_true",
        help="run the executor in --mock mode: offline, synthetic data",
    )

    composer = ap.add_argument_group("answer")
    composer.add_argument("--top-k", type=int, default=20)
    composer.add_argument(
        "--no-quotes", dest="quotes", action="store_false",
        help="omit verified literature quotes from the answer",
    )

    return ap.parse_args(argv)


def build_planner(args: argparse.Namespace) -> Optional[Any]:
    """Wire up the planner when one can be imported and a question was given.

    Absent, this is not an error: with `--plan`, the controller executes,
    diagnoses and composes without ever needing to plan. The policy layer
    already reports repair and relax as unavailable in that case, which is the
    accurate account of what the loop can do.
    """
    if not args.question:
        return None
    try:
        from planner_agent import PlannerAgent  # type: ignore
        from planner_agent.ollama_client import OllamaGPTOSSClient  # type: ignore
    except Exception as exc:
        print(
            f"note: no planner is available ({exc}). A question cannot be "
            f"planned; supply --plan instead.",
            file=sys.stderr,
        )
        return None

    from .planner_port import PlannerAgentPort

    client = OllamaGPTOSSClient(model=args.model, host=args.ollama_host)
    return PlannerAgentPort(PlannerAgent(llm=client))


def build_decision_maker(args: argparse.Namespace) -> Any:
    if not args.llm_controller:
        return PolicyDecisionMaker()
    try:
        from planner_agent.ollama_client import OllamaGPTOSSClient  # type: ignore
    except Exception as exc:
        print(
            f"note: no model client is available for the controller ({exc}); "
            f"using the deterministic policy.",
            file=sys.stderr,
        )
        return PolicyDecisionMaker()
    return LLMDecisionMaker(
        OllamaGPTOSSClient(model=args.model, host=args.ollama_host)
    )


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not args.question and not args.plan:
        print("give --question or --plan", file=sys.stderr)
        return 1

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    plan: Optional[Dict[str, Any]] = None
    if args.plan:
        plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))

    available_inputs: List[Dict[str, Any]] = []
    if args.available_inputs:
        loaded = json.loads(Path(args.available_inputs).read_text(encoding="utf-8"))
        available_inputs = loaded if isinstance(loaded, list) else [loaded]

    executor = SubprocessExecutor(ExecutorSettings(
        python=args.executor_python,
        pythonpath=list(args.executor_pythonpath),
        cwd=args.executor_cwd,
        runs_dir=str(out_dir),
        cache=args.cache,
        base_args=list(args.executor_arg),
        mock=args.dry_run,
        quiet=True,
        process_timeout_s=args.process_timeout,
    ))

    config = ControllerConfig(
        budget=Budget(
            max_iterations=args.max_iterations,
            max_planner_calls=args.max_planner_calls,
            max_wall_clock_s=args.max_wall_clock,
            max_arax_calls=args.max_arax_calls,
            max_llm_calls=args.max_llm_calls,
        ),
        available_inputs=available_inputs,
        composer=ComposerSettings(top_k=args.top_k, include_quotes=args.quotes),
        runs_dir=str(out_dir),
        trace_path=args.trace or str(out_dir / "trace.json"),
        answer_path=args.answer or str(out_dir / "answer.json"),
        verbose=not args.quiet,
    )

    controller = LoopController(
        executor=executor,
        planner=build_planner(args),
        decision_maker=build_decision_maker(args),
        config=config,
    )

    outcome = controller.run(args.question, plan=plan)

    if not args.quiet:
        print()
        for line in summary_lines(outcome):
            print(line)
        if outcome.answer:
            print("\n" + "-" * 68)
            print(render_text(outcome.answer))

    if outcome.status in (LOOP_ANSWERED, LOOP_ABSENCE):
        return 0
    if outcome.status == LOOP_REFUSED:
        return 3
    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
