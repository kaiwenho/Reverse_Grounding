"""
loop.py — The state machine.

    plan ──► execute ──► diagnose ──► legal moves ──► decide ──┐
      ▲                                                        │
      └──────────── repair / relax ◄───────────────────────────┤
                    retry execution ◄───────────────────────────┤
                                                               ▼
                                          accept / absence / abandon ──► compose

The body is short because the hard parts live elsewhere: diagnosis reads the
result, policy decides what is permitted, the decision maker chooses, the
composer writes the answer. What remains here is sequencing and the two
guarantees the sequencing owns.

**A revision that would query the same thing is not executed.** The planner is
told the plan must differ in what it queries, and if it returns one that does
not, the loop rejects it before spending an execution, says so specifically,
and asks once more. Without this the loop's most common failure is invisible:
four iterations, four identical ARAX queries, one answer, and a trace that
looks like diligence.

**An iteration always advances something.** Every non-terminal move either
produces a new plan fingerprint, spends a relaxation axis, or climbs the
execution ladder. All three are finite, and the budget bounds them again from
outside, so the loop terminates whatever the decision maker does — including a
decision maker that would happily iterate forever.

**A relaxation loosens one constraint, and it is checked rather than asked
for.** The planner is told to loosen exactly the named axis and leave the rest
alone; a revision that moved something else is rejected with the offending
field named, once. If the second attempt is still over-broad the plan is used
anyway — it is valid and it is new, and discarding it would trade an answer for
none — but the loop records that it did, and the answer carries the withdrawal
of the claim that a single constraint was the obstacle.

The relaxation axis is marked spent when the relaxation is *requested*, not
when it succeeds. A relax that the planner could not carry out has still had
its turn; retrying it would be the one way the axis set could stop shrinking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .compose import ComposerSettings, compose
from .contracts import (
    ABANDON, ACCEPT, Attempt, Budget, Decision, Diagnosis, LOOP_ABSENCE,
    LOOP_ANSWERED, LOOP_EXHAUSTED, LOOP_FAILED, LOOP_REFUSED, LoopOutcome,
    LoopState, OUTCOME_REFUSED, PLANNER_ACTIONS, RELAX_PLAN, REPORT_ABSENCE,
    RETRY_EXECUTION, VERDICT_REFUSED,
)
from .decide import LLMDecisionMaker, PolicyDecisionMaker
from .diagnose import check_outcome_drift, diagnose, summary_line
from .executor_cli import ExecutorUnavailable
from .fingerprint import (
    describe_off_axis, diff_summary, fingerprint, relaxation_diff,
)
from .planner_port import NoPlanner
from .policy import default_action, legal_actions
from .ports import ExecutionRun, PlanAttempt, failure_result
from .revision import build_request
from .trace import Trace


@dataclass
class ControllerConfig:
    budget: Budget = field(default_factory=Budget)
    available_inputs: List[Dict[str, Any]] = field(default_factory=list)
    composer: ComposerSettings = field(default_factory=ComposerSettings)
    runs_dir: str = "runs"
    trace_path: Optional[str] = None
    answer_path: Optional[str] = None
    verbose: bool = True
    #: How many times a revision may be rejected as a duplicate before the loop
    #: stops asking. One retry, because the second rejection means the planner
    #: is not able to move rather than that it misread the instruction.
    duplicate_retries: int = 1


class LoopController:
    """Runs one question to an answer, or to a defensible refusal to answer."""

    def __init__(
        self,
        executor: Any,
        planner: Optional[Any] = None,
        decision_maker: Optional[Any] = None,
        config: Optional[ControllerConfig] = None,
    ) -> None:
        self.executor = executor
        self.planner = planner or NoPlanner()
        self.planner_available = planner is not None and not isinstance(planner, NoPlanner)
        self.decision_maker = decision_maker or PolicyDecisionMaker()
        self.config = config or ControllerConfig()
        self.trace = Trace()

    # -- logging -----------------------------------------------------------

    def log(self, message: str = "") -> None:
        if self.config.verbose:
            print(message)

    # -- entry point -------------------------------------------------------

    def run(
        self,
        question: Optional[str] = None,
        *,
        plan: Optional[Dict[str, Any]] = None,
    ) -> LoopOutcome:
        config = self.config
        state = LoopState(question=question or "", budget=config.budget)
        self.trace = Trace(
            question=question or "",
            started_at=datetime.now(timezone.utc).isoformat(),
        )

        current_plan, opening = self._opening_plan(question, plan, state)
        if opening is not None:
            return self._write_outputs(opening)
        assert current_plan is not None

        if not state.question:
            state.question = current_plan.get("question") or ""
            self.trace.question = state.question

        last_run: Optional[ExecutionRun] = None
        last_diag: Optional[Diagnosis] = None
        last_plan: Dict[str, Any] = current_plan

        while True:
            spent = state.budget_exhausted()
            if spent:
                self.trace.note(f"stopping: {spent}")
                return self._write_outputs(self._finalize_on_budget(
                    state, last_run, last_plan, last_diag, spent,
                ))

            iteration = state.iteration + 1
            self.log(f"\n[{iteration}] executing plan {fingerprint(current_plan)}")

            try:
                run, diag = self._execute(current_plan, state, iteration)
            except ExecutorUnavailable as exc:
                self.trace.warn(str(exc))
                return self._write_outputs(LoopOutcome(
                    status=LOOP_FAILED,
                    reason=str(exc),
                    final_plan=current_plan,
                    state=state,
                ))

            drift = check_outcome_drift(diag)
            if drift:
                self.trace.warn(drift)
                self.log(f"    WARNING: {drift}")
            self.log(f"    {summary_line(diag)}")

            attempt = Attempt(
                index=iteration,
                plan_fingerprint=fingerprint(current_plan),
                plan_path=run.plan_path,
                result_path=run.result_path,
                hints_path=run.hints_path,
                execution_overrides=dict(state.execution_overrides),
                diagnosis=diag,
            )

            legal = legal_actions(
                diag, state, planner_available=self.planner_available,
            )
            self.trace.record_legality(iteration, legal)

            # Evaluated on every iteration whether or not it is used: it is the
            # fallback, and recording it next to the actual choice turns "is
            # the model helping?" into a measurable question.
            policy_choice = default_action(diag, state, legal)

            decision = self.decision_maker.decide(diag, state, legal)
            attempt.decision = decision

            # One model call per exchange with the decision maker, including
            # the ones that were rejected and retried. Zero for the
            # deterministic maker, which keeps an empty attempt list.
            decision_attempts = getattr(self.decision_maker, "attempts", [])
            state.controller_llm_calls += len(decision_attempts)
            self.trace.record_decision_attempts(iteration, decision_attempts)
            self.trace.record_counterfactual(
                iteration, decision.action, policy_choice.action, decision.source,
            )

            state.record(attempt)
            last_run, last_diag, last_plan = run, diag, current_plan

            self.log(f"    -> {decision.action} [{decision.source}]")
            if decision.overridden_from:
                self.log(
                    f"       overrode '{decision.overridden_from}': "
                    f"{decision.override_reason}"
                )
            if decision.rationale:
                self.log(f"       {decision.rationale}")

            # -- terminal moves -------------------------------------------
            if decision.action == ACCEPT:
                return self._write_outputs(self._answer(
                    LOOP_ANSWERED, decision.rationale, run, current_plan, diag, state,
                ))

            if decision.action == REPORT_ABSENCE:
                return self._write_outputs(self._answer(
                    LOOP_ABSENCE, decision.rationale, run, current_plan, diag, state,
                ))

            if decision.action == ABANDON:
                return self._write_outputs(self._answer(
                    _stop_status(diag), decision.rationale,
                    run, current_plan, diag, state,
                ))

            # -- same plan, looser execution ------------------------------
            if decision.action == RETRY_EXECUTION:
                state.execution_overrides.update(decision.execution_overrides)
                self.trace.note(
                    f"iteration {iteration}: re-running the same plan with "
                    f"{decision.execution_overrides}"
                )
                continue

            # -- new plan --------------------------------------------------
            if decision.action in PLANNER_ACTIONS:
                revised, note = self._revise(
                    decision, diag, state, current_plan,
                )
                attempt.planner_note = note
                if note:
                    self.log(f"       planner: {note}")

                if revised is None:
                    return self._write_outputs(self._after_failed_revision(
                        run, current_plan, diag, state,
                        note or "the revision failed",
                    ))

                current_plan = revised
                continue

            # Every member of ACTIONS is handled above. Reaching here means a
            # move was added without a branch, and continuing would silently
            # re-execute the same plan — so it stops instead.
            raise AssertionError(
                f"the loop has no branch for action '{decision.action}'"
            )

    # -- execution ---------------------------------------------------------

    def _execute(
        self, plan: Dict[str, Any], state: LoopState, iteration: int,
    ) -> tuple[ExecutionRun, Diagnosis]:
        """Probe cheaply, then run — or skip the run if the probe found a fault.

        The executor's `--check` mode validates the plan and asks the meta
        knowledge graph whether the backend can answer each hop, without sending
        a single graph query. It takes seconds; a pathfinder search takes
        minutes. The two failures it catches — an invalid plan and an
        unsupported triple — are both repairable, so discovering them after a
        multi-minute search is pure waiting.

        Three conditions, each load-bearing.

        **Only on the first iteration.** Later iterations follow a repair, which
        already produced a validated plan, or a retry, where the plan passed.

        **Only when no execution overrides are in force.** Those exist only on a
        retry.

        **A clean probe is never mistaken for a result.** It means "nothing
        obviously wrong", not "this will work" — the backend supporting a hop
        shape says nothing about whether it holds data for these entities. So a
        clean probe falls through to the real run, and only a *replannable*
        verdict short-circuits it.

        One detail of the executor's behaviour drives the shape of this, and it
        is not obvious from its documentation: `--check` writes a result
        document **only when the plan cannot be executed**. A plan that passes
        produces no file and exit code 0. So the exit code is what says whether
        the probe found anything, and there is nothing to diagnose on the clean
        path — which is just as well, since a synthesised "everything is fine"
        document would be a claim the probe never made.
        """
        probe_first = (
            iteration == 1
            and not state.execution_overrides
            and hasattr(self.executor, "check")
        )

        if probe_first:
            probe = self.executor.check(plan, tag=f"iter{iteration}-check")

            if probe.exit_code == 0:
                self.log("    probe: no plan or capability fault")
            else:
                probe_diag = diagnose(probe.result, probe.hints, plan)
                state.executor_llm_calls += probe_diag.llm_calls

                if probe_diag.replannable:
                    self.log(
                        f"    probe: {probe_diag.outcome} — repairable, so the "
                        f"search was not run"
                    )
                    self.trace.note(
                        f"iteration {iteration}: the --check probe found "
                        f"'{probe_diag.outcome}'; no graph query was sent"
                    )
                    return probe, probe_diag

                # A non-zero exit the probe could not attribute to the plan.
                # Reported, then ignored: the real run is the authority on
                # whether the backend can be reached at all.
                self.trace.warn(
                    f"iteration {iteration}: the --check probe exited "
                    f"{probe.exit_code} with outcome '{probe_diag.outcome}', "
                    f"which is not repairable; running the query anyway"
                )

        run = self.executor.run(
            plan, overrides=state.execution_overrides, tag=f"iter{iteration}",
        )
        return run, diagnose(run.result, run.hints, plan)

    # -- opening -----------------------------------------------------------

    def _opening_plan(
        self,
        question: Optional[str],
        plan: Optional[Dict[str, Any]],
        state: LoopState,
    ) -> tuple[Optional[Dict[str, Any]], Optional[LoopOutcome]]:
        """Get the first plan, or the outcome that says why there isn't one.

        A plan supplied by the caller is used as given. That is how the
        controller is run without a planner at all — a hand-written plan still
        gets execution, diagnosis, and a composed, grounded answer, and the
        policy layer simply reports repair and relax as unavailable.
        """
        if plan is not None:
            return dict(plan), None

        if not question:
            return None, LoopOutcome(
                status=LOOP_FAILED,
                reason="neither a question nor a plan was given",
                state=state,
            )

        if not self.planner_available:
            return None, LoopOutcome(
                status=LOOP_FAILED,
                reason="a question was given but no planner is configured",
                state=state,
            )

        self.log(f"planning: {question}")
        attempt = self.planner.plan(
            question, available_inputs=self.config.available_inputs,
        )
        state.planner_calls += 1
        # `attempts` is the planner's own count: one, or two when its first
        # draft failed validation and it repaired.
        state.planner_llm_calls += max(1, int(attempt.attempts or 1))
        self.trace.planner_exchanges.append({"kind": "plan", **attempt.to_dict()})

        if attempt.refused:
            self.log(f"  the planner declined: {attempt.refusal_reason}")
            return None, self._refusal_outcome(attempt, question, state)

        if not attempt.ok or not attempt.plan:
            return None, LoopOutcome(
                status=LOOP_FAILED,
                reason="the planner did not produce a valid plan: "
                       + "; ".join(attempt.errors[:3]),
                state=state,
            )

        return attempt.plan, None

    def _refusal_outcome(
        self, attempt: PlanAttempt, question: str, state: LoopState,
    ) -> LoopOutcome:
        """A refusal is an answer, and gets composed like one.

        Rendered from the typed reason rather than the planner's message, which
        is model-written prose. The user is told what kind of question this is
        and that it was declined before any query ran.
        """
        result = failure_result(
            attempt.plan or {"question": question},
            OUTCOME_REFUSED,
            f"the planner declined: {attempt.refusal_reason}",
            verdict=VERDICT_REFUSED,
        )
        result["refusal"] = (attempt.plan or {}).get("refusal") or {
            "reason": attempt.refusal_reason,
        }
        answer = compose(
            result, None, question=question,
            settings=self.config.composer, iterations=0,
            plan=attempt.plan,
        )
        return LoopOutcome(
            status=LOOP_REFUSED,
            reason=f"the planner declined: {attempt.refusal_reason}",
            final_result=result,
            final_plan=attempt.plan,
            answer=answer,
            state=state,
        )

    # -- revision ----------------------------------------------------------

    def _revise(
        self,
        decision: Decision,
        diag: Diagnosis,
        state: LoopState,
        current_plan: Dict[str, Any],
    ) -> tuple[Optional[Dict[str, Any]], Optional[str]]:
        """Ask for a new plan, and refuse one that would query the same thing."""
        axis = None
        if decision.action == RELAX_PLAN and decision.relaxation_key:
            axis = diag.axis_by_key(decision.relaxation_key)
            # Spent on request, not on success: an axis that was tried and
            # could not be carried out has had its turn, and re-offering it is
            # the one way the axis set could stop shrinking.
            state.spent_axes.add(decision.relaxation_key)

        rejection: Optional[str] = None

        for round_index in range(self.config.duplicate_retries + 1):
            if state.planner_budget_exhausted():
                return None, (
                    f"planner call budget spent "
                    f"({state.budget.max_planner_calls})"
                )

            request = build_request(
                question=state.question,
                prior_plan=current_plan,
                prior_fingerprint=fingerprint(current_plan),
                action=decision.action,
                diagnosis=diag,
                relaxation=axis,
                repair_focus=decision.repair_focus,
                forbidden_fingerprints=sorted(state.tried_fingerprints),
                available_inputs=self.config.available_inputs,
                attempt_index=state.iteration,
                attempts_remaining=state.iterations_remaining(),
                previous_rejection=rejection,
            )

            attempt = self.planner.revise(request)
            state.planner_calls += 1
            state.planner_llm_calls += max(1, int(attempt.attempts or 1))
            self.trace.planner_exchanges.append({
                "kind": decision.action,
                "round": round_index + 1,
                "request": request.to_dict(),
                **attempt.to_dict(),
            })

            if attempt.refused:
                return None, (
                    f"the planner declined to revise: {attempt.refusal_reason}"
                )
            if not attempt.ok or not attempt.plan:
                return None, (
                    "the revision did not validate: "
                    + "; ".join(attempt.errors[:3])
                )

            new_fingerprint = fingerprint(attempt.plan)
            if new_fingerprint in state.tried_fingerprints:
                changed = diff_summary(current_plan, attempt.plan)
                rejection = (
                    f"the revised plan would send the same queries as one "
                    f"already executed ({new_fingerprint})"
                    + (
                        f"; the only differences were in {', '.join(changed)}"
                        if changed else
                        "; nothing in the executable content changed at all"
                    )
                )
                self.trace.note(f"rejected duplicate revision: {rejection}")
                continue

            if axis is None:
                return attempt.plan, attempt.note or None

            scope = relaxation_diff(current_plan, attempt.plan, axis)
            if scope.ok:
                return attempt.plan, attempt.note or None

            last_round = round_index >= self.config.duplicate_retries
            if not last_round:
                rejection = describe_off_axis(scope)
                self.trace.note(f"rejected over-broad relaxation: {rejection}")
                self.log(f"       rejected: {rejection}")
                continue

            # Asked once, told what was wrong, and it came back changing more
            # than one thing again. The plan is still valid and still new, and
            # throwing it away would trade a usable answer for none — so it is
            # used, and the claim it can no longer support is withdrawn rather
            # than quietly kept. A relaxation that moved several constraints
            # cannot tell anyone which one was the obstacle.
            record = {
                "iteration": state.iteration,
                "axis": axis.key,
                "diff": scope.to_dict(),
                "note": describe_off_axis(scope),
            }
            state.over_relaxations.append(record)
            self.trace.warn(
                f"iteration {state.iteration}: {record['note']}; the revision "
                f"is used, but results from it cannot be attributed to the "
                f"constraint that was loosened"
            )
            self.log(f"       WARNING: {record['note']}")
            return attempt.plan, attempt.note or None

        return None, (
            "the planner could not produce a plan that differs in what it "
            "would query"
        )

    # -- endings -----------------------------------------------------------

    def _answer(
        self,
        status: str,
        reason: str,
        run: ExecutionRun,
        plan: Dict[str, Any],
        diag: Diagnosis,
        state: LoopState,
    ) -> LoopOutcome:
        answer = compose(
            run.result, diag,
            question=state.question,
            settings=self.config.composer,
            iterations=state.iteration,
            plan=plan,
            extra_caveats=_run_caveats(state),
        )
        return LoopOutcome(
            status=status,
            reason=reason,
            final_result=run.result,
            final_plan=plan,
            answer=answer,
            state=state,
        )

    def _after_failed_revision(
        self,
        run: ExecutionRun,
        plan: Dict[str, Any],
        diag: Diagnosis,
        state: LoopState,
        note: str,
    ) -> LoopOutcome:
        """A revision could not be produced. Serve what is in hand, if anything.

        Results already retrieved do not stop being results because the next
        plan could not be written, and discarding them would trade a partial
        answer for none. What changes is the account of the run: the trace and
        the reason both record that the loop stopped early and why.
        """
        if diag.num_results:
            self.trace.note(f"serving results after a failed revision: {note}")
            return self._answer(
                LOOP_ANSWERED,
                f"results from the last successful execution; the loop stopped "
                f"because {note}",
                run, plan, diag, state,
            )

        if diag.unresolved_entities:
            self.trace.note(f"unresolved anchor; asking for clarification: {note}")
            return self._answer(
                _stop_status(diag),
                f"a concept in the question did not match any entry in the "
                f"graph: {', '.join(diag.unresolved_entities)}",
                run, plan, diag, state,
            )
        return self._answer(LOOP_EXHAUSTED, note, run, plan, diag, state)

    def _finalize_on_budget(
        self,
        state: LoopState,
        run: Optional[ExecutionRun],
        plan: Optional[Dict[str, Any]],
        diag: Optional[Diagnosis],
        spent: str,
    ) -> LoopOutcome:
        """The budget ran out between iterations."""
        if run is None or diag is None or plan is None:
            return LoopOutcome(
                status=LOOP_EXHAUSTED,
                reason=f"budget exhausted before anything ran: {spent}",
                state=state,
            )
        status = LOOP_ANSWERED if diag.num_results else LOOP_EXHAUSTED
        return self._answer(
            status,
            f"budget exhausted ({spent}); reporting the last completed run",
            run, plan, diag, state,
        )

    # -- output ------------------------------------------------------------

    def _write_outputs(self, outcome: LoopOutcome) -> LoopOutcome:
        config = self.config
        if config.trace_path:
            path = self.trace.write(config.trace_path, outcome)
            self.log(f"\ntrace: {path}")
        if config.answer_path and outcome.answer:
            import json

            target = Path(config.answer_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(
                json.dumps(outcome.answer, indent=2, default=str), encoding="utf-8",
            )
            self.log(f"answer: {target}")
        return outcome


def _stop_status(diag: Diagnosis) -> str:
    """The status for a run that stopped without an answer.

    Three endings look the same to the loop and mean different things to a
    reader. A plan the planner declined is a refusal. A name that matched
    nothing in the graph is also a refusal — the run established something
    specific, and the composer turns it into a question the user can act on.
    Everything else is exhaustion.

    Kept in one function because it was not, and the two paths drifted. The
    failed-revision ending learned about unresolved anchors and the `abandon`
    ending did not, so a live run that composed "here are twenty entries, which
    did you mean?" reported its status as `exhausted` — describing the loop's
    budget rather than what happened.
    """
    if diag.outcome == OUTCOME_REFUSED:
        return LOOP_REFUSED
    if diag.unresolved_entities:
        return LOOP_REFUSED
    return LOOP_EXHAUSTED


def _run_caveats(state: LoopState) -> List[str]:
    """What the shape of the run, rather than the result, obliges the answer to say.

    Only one entry so far, and it is the withdrawal of a claim the loop
    otherwise makes silently. Relaxation is one axis at a time so that an
    answer arriving after it can be attributed to the constraint that was
    loosened; when the planner moved several and would not take the correction,
    that attribution is gone, and the reader has to be told — the answer looks
    identical either way.
    """
    if not state.over_relaxations:
        return []

    axes = ", ".join(sorted({str(r.get("axis")) for r in state.over_relaxations}))
    return [
        f"Reaching this answer required loosening a constraint ({axes}), and "
        f"the revised plan changed more than the one constraint that was "
        f"requested. These results therefore cannot be attributed to any "
        f"single constraint having been the obstacle."
    ]


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def build_controller(
    executor: Any,
    planner: Optional[Any] = None,
    *,
    llm: Optional[Any] = None,
    config: Optional[ControllerConfig] = None,
) -> LoopController:
    """Assemble a controller, using an LLM decision maker when one is possible.

    Falls back to the deterministic policy when no model client is supplied.
    That fallback is not a degraded mode with a warning attached: the policy
    ordering is a complete controller, and a run that uses it produces the same
    kinds of answer with the same guarantees, less adaptively.
    """
    decision_maker = LLMDecisionMaker(llm) if llm is not None else PolicyDecisionMaker()
    return LoopController(
        executor=executor,
        planner=planner,
        decision_maker=decision_maker,
        config=config,
    )
