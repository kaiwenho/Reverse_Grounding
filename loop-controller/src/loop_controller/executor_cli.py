"""
executor_cli.py — Running plan-executor as a subprocess.

The executor is a command-line program with meaningful exit codes, a cache that
makes repeated runs cheap, and a habit of spending minutes inside a single ARAX
pathfinder call. Driving it as a process rather than importing it keeps those
properties intact: a run that hangs is killed by a timeout instead of blocking
the controller, a crash in a knowledge-provider client cannot take the loop
down with it, and the exact command is reproducible from the trace by hand.

The cost is that the coupling is no longer checked by the type system. Two
things pay it down. The outcome vocabulary is mirrored in `contracts.py` and
drift is reported rather than swallowed. And every run produces a result
document even when the process produced none — a crashed, killed or
never-started run is written up in the executor's own shape, so nothing
downstream needs a branch for "there is no result".

Exit codes, from the executor's README:

    0  results, or an established `no_answer`
    1  inconclusive, unexecutable, or an invalid plan
    2  the LLM was unavailable, so entity resolution could not continue

Code 2 is not a failure to route around. The executor stops there rather than
falling back to the name resolver's first guess, because an unreviewed anchor
produces confident results about the wrong concept. The controller treats it
the same way: it stops, and says why.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .contracts import (
    EXECUTION_KNOBS, OUTCOME_BACKEND, OUTCOME_TRUNCATED, VERDICT_INCONCLUSIVE,
)
from .ports import ExecutionRun, failure_result


class ExecutorUnavailable(RuntimeError):
    """The executor could not run, and re-running will not help.

    Raised for exit code 2 — the local model is not reachable — because that is
    an environment problem the loop cannot resolve by choosing a different
    move, and continuing would mean iterating over runs that all stop at the
    same place.
    """


@dataclass
class ExecutorSettings:
    """How to invoke the executor.

    ``base_args`` is where a deployment puts its standing choices: the stages
    that are always off, the ARAX endpoint, the model tag. Keeping them here
    rather than in the loop means the loop only ever adds the arguments that
    change between iterations, which is what makes a trace's command lines
    diffable.
    """

    python: str = sys.executable
    module: str = "plan_executor.run_plan"
    command: Optional[List[str]] = None
    cwd: Optional[str] = None
    pythonpath: List[str] = field(default_factory=list)
    runs_dir: str = "runs"
    cache: Optional[str] = None
    base_args: List[str] = field(default_factory=list)
    mock: bool = False
    quiet: bool = True
    process_timeout_s: float = 1800.0
    env: Dict[str, str] = field(default_factory=dict)

    def argv_prefix(self) -> List[str]:
        if self.command:
            return list(self.command)
        return [self.python, "-m", self.module]


class SubprocessExecutor:
    """Runs plan-executor and returns its two output documents."""

    def __init__(self, settings: Optional[ExecutorSettings] = None) -> None:
        self.settings = settings or ExecutorSettings()
        self.runs: List[Dict[str, Any]] = []
        Path(self.settings.runs_dir).mkdir(parents=True, exist_ok=True)

    # -- public ------------------------------------------------------------

    def run(
        self,
        plan: Dict[str, Any],
        *,
        overrides: Optional[Dict[str, Any]] = None,
        tag: str = "run",
    ) -> ExecutionRun:
        settings = self.settings
        runs = Path(settings.runs_dir)
        plan_path = runs / f"{tag}.plan.json"
        out_path = runs / f"{tag}.result.json"
        hints_path = runs / f"{tag}.hints.json"

        plan_path.write_text(json.dumps(plan, indent=2), encoding="utf-8")

        argv = settings.argv_prefix() + [
            str(plan_path),
            "--out", str(out_path),
            "--hints", str(hints_path),
        ]
        if settings.quiet:
            argv.append("--quiet")
        if settings.mock:
            argv.append("--mock")
        if settings.cache:
            argv += ["--cache", settings.cache]
        argv += list(settings.base_args)
        argv += _override_args(overrides or {})

        env = dict(os.environ)
        env.update(settings.env)
        if settings.pythonpath:
            existing = env.get("PYTHONPATH", "")
            joined = os.pathsep.join(settings.pythonpath + ([existing] if existing else []))
            env["PYTHONPATH"] = joined

        record: Dict[str, Any] = {"tag": tag, "argv": argv}
        self.runs.append(record)

        try:
            completed = subprocess.run(
                argv,
                cwd=settings.cwd,
                env=env,
                capture_output=True,
                text=True,
                timeout=settings.process_timeout_s,
            )
        except subprocess.TimeoutExpired:
            record["exit_code"] = None
            record["timed_out"] = True
            return ExecutionRun(
                result=failure_result(
                    plan, OUTCOME_TRUNCATED,
                    f"the executor process exceeded "
                    f"{settings.process_timeout_s:.0f}s and was stopped; "
                    f"absence of results is not established",
                    verdict=VERDICT_INCONCLUSIVE,
                ),
                exit_code=124,
                plan_path=str(plan_path),
                stderr_tail="process timeout",
            )
        except FileNotFoundError as exc:
            record["exit_code"] = None
            raise ExecutorUnavailable(
                f"could not start the executor ({' '.join(argv[:3])}): {exc}"
            ) from exc

        record["exit_code"] = completed.returncode
        stderr_tail = (completed.stderr or "")[-4000:]

        if completed.returncode == 2:
            raise ExecutorUnavailable(
                "the executor stopped because its LLM was unavailable; entity "
                "resolution requires it, and falling back to the resolver's "
                "first candidate would produce confident results about a "
                "possibly wrong concept.\n" + stderr_tail
            )

        result = _read_json(out_path)
        hints = _read_json(hints_path)

        if result is None and completed.returncode != 0:
            # A *failing* run that wrote nothing did not get far enough to have
            # an opinion. The executor's contract is that it always writes a
            # result document, including for its own failures — that is what
            # the whole outcome taxonomy is for — so a non-zero exit with no
            # document means a bad invocation, a missing dependency, or a crash
            # at import.
            #
            # This used to become a `backend_failure`, which the policy layer
            # reads as transient and therefore retryable. The loop then walked
            # the whole escalation ladder against a process that had never
            # started: three retries, zero model calls, zero graph calls, under
            # a second in total, and a final verdict of "exhausted" that
            # described the budget rather than the fault. A longer timeout
            # cannot help a program that exits before it opens a socket.
            #
            # The exit-code test is load-bearing, and the first version of this
            # left it out. `--check` writes no document on the clean path and
            # exits 0 — documented two methods below, and broken immediately by
            # treating every missing document as a crash.
            raise ExecutorUnavailable(_startup_failure_message(
                argv, completed.returncode, stderr_tail,
            ))

        if result is None:
            # Exit 0 and no document: the `--check` clean path. `check()`
            # branches on the exit code and never reads this, but `run()` must
            # still return something shaped like a result.
            result = failure_result(
                plan, OUTCOME_BACKEND,
                f"the executor exited 0 without writing a result document",
            )

        return ExecutionRun(
            result=result,
            hints=hints or {},
            exit_code=completed.returncode,
            plan_path=str(plan_path),
            result_path=str(out_path) if out_path.exists() else None,
            hints_path=str(hints_path) if hints_path.exists() else None,
            stderr_tail=stderr_tail,
        )

    # -- diagnostics -------------------------------------------------------

    def check(self, plan: Dict[str, Any], *, tag: str = "check") -> ExecutionRun:
        """Run the executor's own `--check`: validate, probe, query nothing.

        Seconds rather than minutes, and it catches the two failures that would
        otherwise cost a full search to discover — a plan that does not
        validate, and a hop the backend cannot answer.

        **Read the exit code, not the result.** `--check` writes a result
        document only when the plan cannot be executed; a plan that passes
        produces no file and exit code 0. On that clean path `run()` finds
        nothing to read and synthesises a backend-failure document, which is an
        artefact of this method reusing `run()` rather than a finding — callers
        must branch on `exit_code == 0` before looking at `result`.
        """
        saved = self.settings.base_args
        self.settings.base_args = list(saved) + ["--check"]
        try:
            return self.run(plan, tag=tag)
        finally:
            self.settings.base_args = saved


def _override_args(overrides: Dict[str, Any]) -> List[str]:
    """Turn execution knobs into command-line arguments.

    Only the knobs the decision contract allows. An unknown key is dropped
    rather than passed through, because an argument this package does not know
    is one the policy layer never checked for looseness.
    """
    argv: List[str] = []
    for knob, value in (overrides or {}).items():
        flag = EXECUTION_KNOBS.get(knob)
        if not flag:
            continue
        argv += [flag, str(value)]
    return argv


#: Import failures name the module that was missing, and the fix is almost
#: always a path rather than an install — the four packages sit beside each
#: other and none of them is pip installed.
_MISSING_MODULE_HINT: Dict[str, str] = {
    "plan_executor": (
        "pass --executor-pythonpath ../plan-executor/src (the executor runs as "
        "a subprocess and does not inherit this process's PYTHONPATH)"
    ),
    "plan_core": (
        "pass --executor-pythonpath ../plan-core/src; the executor validates "
        "plans against the shared contract"
    ),
    "ollama": "the executor's model client needs it: pip install ollama",
}

_MISSING_MODULE_RE = re.compile(r"No module named '([A-Za-z_][A-Za-z0-9_]*)'")


def _startup_failure_message(
    argv: List[str], returncode: Optional[int], stderr_tail: str,
) -> str:
    """Say what failed to start and, where the cause is legible, how to fix it.

    Worth the effort because this message is the whole diagnosis. The loop
    stops here, so whatever this says is what somebody reads at the end of a
    run that did nothing — and "the executor exited 1" sends people looking for
    a bug in the executor when the fault is in how it was invoked.
    """
    lines = [
        f"the executor exited {returncode} without writing a result document, "
        f"so it did not run: this is a fault in how it was invoked, not in the "
        f"backend, and retrying cannot help"
    ]

    missing = _MISSING_MODULE_RE.search(stderr_tail or "")
    if missing:
        module = missing.group(1)
        hint = _MISSING_MODULE_HINT.get(
            module, f"add whatever provides `{module}` to --executor-pythonpath"
        )
        lines.append(f"  missing module `{module}` — {hint}")

    lines.append(f"  command: {' '.join(argv[:4])} ...")
    if stderr_tail.strip():
        lines.append(f"  last stderr: {stderr_tail.strip()[-600:]}")
    return "\n".join(lines)


def _read_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        with path.open("r", encoding="utf-8") as handle:
            loaded = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return None
    return loaded if isinstance(loaded, dict) else None


# ---------------------------------------------------------------------------
# Test double
# ---------------------------------------------------------------------------


class ScriptedExecutor:
    """Replays recorded result documents in order.

    The loop's tests are built on this. Every stage that matters — diagnosis,
    legality, the decision maker, revision assembly, composition, the grounding
    gate — runs unchanged against a recorded run, so a scenario like "a timeout,
    then an empty result, then results" is a three-element list rather than a
    knowledge graph.
    """

    def __init__(
        self,
        results: List[Dict[str, Any]],
        hints: Optional[List[Dict[str, Any]]] = None,
        check_result: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.results = list(results)
        self.hints = list(hints or [])
        #: What `--check` finds. None mirrors the real executor's clean path:
        #: exit 0 and no document. A document here means the probe found a
        #: fault, and comes back with a non-zero exit like the real one.
        self.check_result = check_result
        self.calls: List[Dict[str, Any]] = []
        self.checks: List[Dict[str, Any]] = []

    def check(
        self, plan: Dict[str, Any], *, tag: str = "check",
    ) -> ExecutionRun:
        self.checks.append({"tag": tag, "plan": plan})
        if self.check_result is None:
            return ExecutionRun(
                result=failure_result(
                    plan, OUTCOME_BACKEND,
                    "no document: --check writes one only on failure",
                ),
                exit_code=0,
            )
        return ExecutionRun(result=self.check_result, exit_code=1)

    def run(
        self,
        plan: Dict[str, Any],
        *,
        overrides: Optional[Dict[str, Any]] = None,
        tag: str = "run",
    ) -> ExecutionRun:
        index = len(self.calls)
        self.calls.append({"tag": tag, "overrides": dict(overrides or {}), "plan": plan})
        if index < len(self.results):
            result = self.results[index]
        else:
            result = self.results[-1] if self.results else failure_result(
                plan, OUTCOME_BACKEND, "the scripted executor ran out of results",
            )
        hints = self.hints[index] if index < len(self.hints) else {}
        return ExecutionRun(result=result, hints=hints, exit_code=0, plan_path=None)
