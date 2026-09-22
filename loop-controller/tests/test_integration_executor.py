"""Against the real executor, in its offline mock mode.

Everything else in this suite replays recorded documents. These tests run
`plan_executor.run_plan` as a subprocess and read what it actually writes,
which is the only way to catch the failure mode a scripted executor cannot: a
mismatch between what this package assumes the executor's output and exit codes
look like and what they are.

Skipped unless the sibling checkouts are importable and the pinned Biolink
model has been downloaded, since none of that is guaranteed in a bare clone.
`--mock` means no ARAX and no local model are needed.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from loop_controller.contracts import (
    Budget, LOOP_ANSWERED, LOOP_EXHAUSTED, REPAIR_PLAN,
)
from loop_controller.executor_cli import ExecutorSettings, SubprocessExecutor
from loop_controller.loop import ControllerConfig, LoopController
from loop_controller.planner_port import ScriptedPlanner
from loop_controller.ports import PlanAttempt


ROOT = Path(__file__).resolve().parents[2]
PLAN_CORE = ROOT / "plan-core" / "src"
PLAN_EXECUTOR = ROOT / "plan-executor" / "src"
BIOLINK = ROOT / "plan-core" / "data" / "biolink-model.yaml"
EXAMPLE_PLAN = ROOT / "plan-executor" / "examples" / "gluten_plan.json"


def _executor_available() -> bool:
    if not (PLAN_CORE.exists() and PLAN_EXECUTOR.exists() and BIOLINK.exists()):
        return False
    probe = subprocess.run(
        [sys.executable, "-c", "import plan_executor.run_plan"],
        env={"PYTHONPATH": f"{PLAN_CORE}:{PLAN_EXECUTOR}", "PATH": "/usr/bin:/bin"},
        capture_output=True,
    )
    return probe.returncode == 0


AVAILABLE = _executor_available()
pytestmark = pytest.mark.skipif(
    not AVAILABLE,
    reason="plan-executor, plan-core or the pinned Biolink model is not present",
)


def settings(tmp_path: Path) -> ExecutorSettings:
    return ExecutorSettings(
        pythonpath=[str(PLAN_CORE), str(PLAN_EXECUTOR)],
        runs_dir=str(tmp_path),
        mock=True,
        quiet=True,
        process_timeout_s=300.0,
    )


def load_plan() -> dict:
    return json.loads(EXAMPLE_PLAN.read_text(encoding="utf-8"))


def test_a_real_run_produces_a_grounded_answer(tmp_path):
    executor = SubprocessExecutor(settings(tmp_path))
    outcome = LoopController(executor, config=ControllerConfig(verbose=False)).run(
        plan=load_plan(),
    )

    assert outcome.status == LOOP_ANSWERED
    assert outcome.answer["grounding"]["ok"] is True
    assert outcome.answer["candidates"]

    # The executor's own two documents were read, not guessed at.
    attempt = outcome.state.attempts[0]
    assert Path(attempt.result_path).exists()
    assert Path(attempt.hints_path).exists()


def test_hints_and_result_agree_about_the_outcome(tmp_path):
    """The coupling a subprocess boundary removes from the type system."""
    executor = SubprocessExecutor(settings(tmp_path))
    run = executor.run(load_plan(), tag="agreement")

    assert run.exit_code == 0
    assert run.hints["outcome"] == run.result["outcome"]["outcome"]
    assert run.hints["replannable"] == run.result["outcome"]["replannable"]
    assert run.hints["num_results"] == len(run.result["results"])


def test_the_diagnosis_reads_a_real_validation_failure(tmp_path):
    plan = load_plan()
    plan["paths"][0]["hops"][0]["predicate"] = "biolink:not_a_real_predicate"

    executor = SubprocessExecutor(settings(tmp_path))
    planner = ScriptedPlanner([PlanAttempt(ok=True, plan=load_plan())])
    outcome = LoopController(
        executor, planner,
        config=ControllerConfig(verbose=False, budget=Budget(max_iterations=3)),
    ).run(plan=plan)

    first = outcome.state.attempts[0]
    assert first.diagnosis.outcome == "invalid_plan"
    assert first.diagnosis.replannable is True
    assert any(
        "not a valid active Biolink predicate" in err
        for err in first.diagnosis.plan_validation_errors
    )
    assert first.decision.action == REPAIR_PLAN

    # The repair was executed and answered.
    assert outcome.status == LOOP_ANSWERED
    assert len(outcome.state.attempts) == 2


def test_an_unexecutable_plan_without_a_planner_stops_honestly(tmp_path):
    plan = load_plan()
    plan["paths"][0]["hops"][0]["predicate"] = "biolink:not_a_real_predicate"

    executor = SubprocessExecutor(settings(tmp_path))
    outcome = LoopController(executor, config=ControllerConfig(verbose=False)).run(
        plan=plan,
    )

    assert outcome.status == LOOP_EXHAUSTED
    # Not reported as an absence: nothing was established about the graph.
    assert outcome.answer["answer_kind"] == "inconclusive"


def test_the_check_mode_queries_nothing(tmp_path):
    executor = SubprocessExecutor(settings(tmp_path))
    run = executor.check(load_plan())
    assert run.exit_code == 0
    # `--check` is not restored into base_args afterwards.
    assert "--check" not in executor.settings.base_args


# ---------------------------------------------------------------------------
# The cheap probe, against the real executor
# ---------------------------------------------------------------------------


def test_check_reports_a_clean_plan_only_through_its_exit_code(tmp_path):
    """The behaviour the probe is built around, asserted against the real
    program rather than assumed: a plan that passes `--check` produces exit 0
    and no result document at all."""
    executor = SubprocessExecutor(settings(tmp_path))
    run = executor.check(load_plan(), tag="clean")

    assert run.exit_code == 0
    assert run.result_path is None
    assert run.hints_path is None


def test_check_writes_a_document_only_when_the_plan_is_broken(tmp_path):
    plan = load_plan()
    plan["paths"][0]["hops"][0]["predicate"] = "biolink:not_a_real_predicate"

    executor = SubprocessExecutor(settings(tmp_path))
    run = executor.check(plan, tag="broken")

    assert run.exit_code != 0
    assert run.result["outcome"]["outcome"] == "invalid_plan"
    assert run.result["outcome"]["replannable"] is True


def test_a_broken_plan_reaches_repair_without_a_single_graph_query(tmp_path):
    """End to end: the probe catches it, the loop repairs, and ARAX is never
    asked anything about the broken plan."""
    plan = load_plan()
    plan["paths"][0]["hops"][0]["predicate"] = "biolink:not_a_real_predicate"

    executor = SubprocessExecutor(settings(tmp_path))
    planner = ScriptedPlanner([PlanAttempt(ok=True, plan=load_plan())])
    outcome = LoopController(
        executor, planner,
        config=ControllerConfig(verbose=False, budget=Budget(max_iterations=3)),
    ).run(plan=plan)

    assert outcome.status == LOOP_ANSWERED
    assert outcome.state.attempts[0].decision.action == REPAIR_PLAN

    # One `--check` and one real run — the broken plan was never queried.
    commands = executor.runs
    checked = [c for c in commands if "--check" in c["argv"]]
    queried = [c for c in commands if "--check" not in c["argv"]]
    assert len(checked) == 1
    assert len(queried) == 1


def test_the_probe_is_counted_against_the_model_budget(tmp_path):
    """`--check` starts a model health check, so it is not free."""
    executor = SubprocessExecutor(settings(tmp_path))
    outcome = LoopController(
        executor, config=ControllerConfig(verbose=False),
    ).run(plan=load_plan())

    state = outcome.state
    assert state.llm_calls == (
        state.executor_llm_calls + state.controller_llm_calls
        + state.planner_llm_calls
    )
    assert state.executor_llm_calls > 0


# ---------------------------------------------------------------------------
# An executor that never started
# ---------------------------------------------------------------------------


def test_a_missing_executor_path_stops_the_loop_instead_of_retrying(tmp_path):
    """Exactly the live failure: `--executor-pythonpath` omitted.

    The executor is a subprocess and does not inherit this process's
    PYTHONPATH, so leaving the flag off makes it die at import. That used to
    become a `backend_failure`, which the policy layer reads as transient — so
    the loop climbed the whole escalation ladder against a program that had
    exited before opening a socket. Three retries, zero model calls, zero graph
    calls, 0.7 seconds, and a final verdict of `exhausted` that described the
    budget rather than the fault.

    The `-S` is load-bearing and must not be tidied away. Clearing PYTHONPATH
    alone reproduces the failure only while `plan_executor` is *not installed*,
    which was true when this test was written and stopped being true the moment
    the repo grew a `make install`. The subprocess then imported the package
    from site-packages, ran fine, and the test failed asserting that a working
    executor had failed. `-S` skips site initialisation, so neither
    site-packages nor an editable install's path file is on the subprocess's
    sys.path, and the import fails whatever is installed.
    """
    executor = SubprocessExecutor(ExecutorSettings(
        command=[sys.executable, "-S", "-m", "plan_executor.run_plan"],
        pythonpath=[],                       # the bug, reproduced
        runs_dir=str(tmp_path),
        mock=True,
        process_timeout_s=60.0,
        env={"PYTHONPATH": ""},
    ))
    outcome = LoopController(executor, config=ControllerConfig(verbose=False)).run(
        plan=load_plan(),
    )

    assert outcome.status == "failed"
    # One attempt, not four: the ladder is never climbed.
    assert len(executor.runs) == 1
    assert "retrying cannot help" in outcome.reason
    assert "plan_executor" in outcome.reason
    assert "--executor-pythonpath" in outcome.reason


def test_a_clean_check_still_writes_no_document_and_is_not_a_crash(tmp_path):
    """The exit-code test that the first version of the fix left out.

    `--check` writes a result document only when the plan is broken. Treating
    every missing document as a crash broke the probe on its clean path, which
    is the path it takes most often.
    """
    executor = SubprocessExecutor(settings(tmp_path))
    probe = executor.check(load_plan())

    assert probe.exit_code == 0
    assert probe.result is not None      # synthesised, and never read
