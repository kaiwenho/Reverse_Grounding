"""Command line entry point for the front door.

`QueryService` is where a question is supposed to enter the system: it holds
rate limiting, screening, topic scope, and the fixed table of things a caller
may be told. Until this module existed the only runnable entry point was
`loop-controller`, which starts one layer *below* the front door — so every
run anyone had done, including every benchmark in this repo, had skipped
intake entirely. A layer nothing can invoke is a layer nothing tests.

What this adds is a way in, not new behaviour. The request is built from an
untrusted payload exactly as an HTTP body would be, and everything the server
is allowed to know — who is calling, whether the raw question may be logged —
is supplied here and never read from the payload.

The executor, planner and decision maker are wired by `loop_controller.
run_loop`'s own helpers rather than reassembled here. Two wiring paths that
are meant to agree eventually stop agreeing, which this project has already
paid for once.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from loop_controller.executor_cli import ExecutorSettings, SubprocessExecutor
from loop_controller.run_loop import build_decision_maker, build_planner

from .models import PublicStatus, RequestContext, ReviewDepth
from .service import QueryService, ServiceConfig


#: Exit codes. A caller scripting this needs to tell "the system worked and
#: the graph had nothing" from "the system did not answer", and neither is a
#: crash. `no_data` is an established absence, which is a finding.
EXIT_OK = 0
EXIT_NOT_ANSWERED = 1
EXIT_UNAVAILABLE = 2

_OK_STATUSES = frozenset({PublicStatus.COMPLETED, PublicStatus.NO_DATA})


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="reverse-grounding",
        description="Ask a biomedical question through the intake layer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "exit codes:\n"
            f"  {EXIT_OK}  answered, or an established absence\n"
            f"  {EXIT_NOT_ANSWERED}  refused, needs clarification, invalid, "
            f"rate limited, or out of budget\n"
            f"  {EXIT_UNAVAILABLE}  the service could not run the request\n"
        ),
    )

    src = ap.add_argument_group("request")
    src.add_argument("--question", help="the question to ask")
    src.add_argument(
        "--payload", metavar="PATH",
        help="a JSON request body instead of --question; '-' reads stdin. "
             "Takes the same fields an HTTP caller would send, and rejects "
             "the same unknown ones.",
    )
    src.add_argument(
        "--review-depth", default=ReviewDepth.STANDARD.value,
        choices=[d.value for d in ReviewDepth],
        help="how much evidence review to spend; maps to fixed server-side "
             "limits (default: %(default)s)",
    )

    ctx = ap.add_argument_group("caller")
    ctx.add_argument(
        "--caller-id", default="cli",
        help="stands in for whatever would have authenticated the request. "
             "Rate limits are per caller. It is never read from the payload.",
    )
    ctx.add_argument(
        "--log-raw-question", action="store_true",
        help="write the question text into the trace. Off by default: a "
             "biomedical question can identify a person, and the digest is "
             "enough to correlate runs.",
    )

    out = ap.add_argument_group("output")
    out.add_argument("--out", default="runs", help="directory for run artefacts")
    out.add_argument("--result", metavar="PATH", help="also write the result JSON here")
    out.add_argument("-q", "--quiet", action="store_true")

    model = ap.add_argument_group("model")
    model.add_argument("--model", default="gpt-oss:120b")
    model.add_argument("--ollama-host", default=None)
    model.add_argument(
        "--llm-controller", action="store_true",
        help="let a model choose the loop's moves; off by default so a run "
             "is reproducible",
    )

    ex = ap.add_argument_group("executor")
    ex.add_argument("--executor-python", default=sys.executable)
    ex.add_argument("--executor-pythonpath", action="append", default=[])
    ex.add_argument("--executor-arg", action="append", default=[], metavar="ARG")
    ex.add_argument("--process-timeout", type=float, default=1800.0)
    ex.add_argument(
        "--dry-run", action="store_true",
        help="run the executor offline on synthetic data instead of querying "
             "the graph",
    )

    return ap.parse_args(argv)


def _payload_from(args: argparse.Namespace) -> Any:
    """The request body, from a file, stdin, or --question.

    Returned untyped and unvalidated on purpose. `UserRequest.from_payload`
    is what decides whether it is acceptable, and routing --question through
    the same door means the convenience flag cannot accidentally accept
    something an HTTP caller could not send.
    """
    if args.payload:
        text = (
            sys.stdin.read() if args.payload == "-"
            else Path(args.payload).read_text(encoding="utf-8")
        )
        return json.loads(text)
    return {"question": args.question, "review_depth": args.review_depth}


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)

    if not args.question and not args.payload:
        print("give --question or --payload", file=sys.stderr)
        return EXIT_NOT_ANSWERED

    try:
        payload = _payload_from(args)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"could not read the request body: {exc}", file=sys.stderr)
        return EXIT_NOT_ANSWERED

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    executor = SubprocessExecutor(ExecutorSettings(
        python=args.executor_python,
        pythonpath=list(args.executor_pythonpath),
        runs_dir=str(out_dir),
        base_args=list(args.executor_arg),
        mock=args.dry_run,
        quiet=True,
        process_timeout_s=args.process_timeout,
    ))

    # `build_planner` returns None when no question was given, which is not
    # the case here, and prints its own note when the planner cannot be
    # imported. Passing a Namespace it recognises keeps one wiring path.
    wiring = argparse.Namespace(
        question=payload.get("question") if isinstance(payload, dict) else None,
        model=args.model,
        ollama_host=args.ollama_host,
        llm_controller=args.llm_controller,
    )

    service = QueryService(
        executor=executor,
        planner=build_planner(wiring),
        decision_maker=build_decision_maker(wiring),
        config=ServiceConfig(runs_dir=str(out_dir), verbose=not args.quiet),
    )

    context = RequestContext(
        caller_id=args.caller_id,
        received_at=time.time(),
        log_raw_question=args.log_raw_question,
    )

    result = service.handle_payload(payload, context)
    body: Dict[str, Any] = result.to_dict()
    text = json.dumps(body, indent=2)

    if args.result:
        Path(args.result).write_text(text + "\n", encoding="utf-8")

    print(text)

    if result.status in _OK_STATUSES:
        return EXIT_OK
    if result.status is PublicStatus.UNAVAILABLE:
        return EXIT_UNAVAILABLE
    return EXIT_NOT_ANSWERED


if __name__ == "__main__":
    raise SystemExit(main())
