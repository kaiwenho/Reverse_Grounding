"""loop_controller — iterates the plan/execute loop and composes a grounded answer."""

__version__ = "0.1.0"

from .contracts import (
    ABANDON, ACCEPT, ACTIONS, Attempt, Budget, Decision, Diagnosis,
    LOOP_ABSENCE, LOOP_ANSWERED, LOOP_EXHAUSTED, LOOP_FAILED, LOOP_REFUSED,
    LoopOutcome, LoopState, RELAX_PLAN, REPAIR_PLAN, REPORT_ABSENCE,
    RETRY_EXECUTION, RelaxationAxis,
)
from .diagnose import diagnose, summary_line
from .fingerprint import fingerprint, locked_constraints, same_query
from .policy import LegalActions, default_action, legal_actions, validate_decision
from .decide import LLMDecisionMaker, PolicyDecisionMaker
from .revision import RevisionRequest, build_request, build_revision_message
from .ports import ExecutionRun, PlanAttempt
from .executor_cli import ExecutorSettings, ExecutorUnavailable, ScriptedExecutor, SubprocessExecutor
from .planner_port import NoPlanner, PlannerAgentPort, ScriptedPlanner
from .grounding import Lexicon, build_lexicon, check_answer, check_statement
from .compose import ComposerSettings, compose, render_text
from .loop import ControllerConfig, LoopController, build_controller
from .trace import Trace, summary_lines

__all__ = [
    "ABANDON", "ACCEPT", "ACTIONS", "RELAX_PLAN", "REPAIR_PLAN",
    "REPORT_ABSENCE", "RETRY_EXECUTION",
    "LOOP_ABSENCE", "LOOP_ANSWERED", "LOOP_EXHAUSTED", "LOOP_FAILED",
    "LOOP_REFUSED",
    "Attempt", "Budget", "Decision", "Diagnosis", "LoopOutcome", "LoopState",
    "RelaxationAxis",
    "diagnose", "summary_line",
    "fingerprint", "locked_constraints", "same_query",
    "LegalActions", "default_action", "legal_actions", "validate_decision",
    "LLMDecisionMaker", "PolicyDecisionMaker",
    "RevisionRequest", "build_request", "build_revision_message",
    "ExecutionRun", "PlanAttempt",
    "ExecutorSettings", "ExecutorUnavailable", "ScriptedExecutor",
    "SubprocessExecutor",
    "NoPlanner", "PlannerAgentPort", "ScriptedPlanner",
    "Lexicon", "build_lexicon", "check_answer", "check_statement",
    "ComposerSettings", "compose", "render_text",
    "ControllerConfig", "LoopController", "build_controller",
    "Trace", "summary_lines",
]
