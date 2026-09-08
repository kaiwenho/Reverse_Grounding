"""
plan_input.py — Load a plan, validate it, and decide whether it can run.

Two distinct questions, kept apart because they belong to different owners:

  * **Is this a valid plan?** Answered by `plan_core.validate_plan` — the same
    validator the planner uses. Schema conformance, entity references, Biolink
    predicate and category existence. Shared, so a rule added for the planner
    is enforced here without being written twice.

  * **Can *this executor* run it?** Answered here. An unanchored path is a
    perfectly valid plan; it is unexecutable only because pivot-first
    decomposition needs a concrete CURIE to start from. That is a fact about
    the execution strategy, not about the plan, and pushing it upstream would
    make the planner encode this executor's internals.

Failing early
-------------
Every check here is cheap and runs before the first ARAX call. An unanchored
path discovered at load time costs milliseconds; the same path discovered by
submitting it costs a 120-second timeout followed by a decomposition that
cannot start.

Two phases
----------
`load_plan_input` runs the structural checks. After `resolver.py` has run,
`check_resolution` runs the second phase: a path whose anchor did not resolve
is unexecutable for the same reason as one that never had an anchor, but that
cannot be known until resolution is attempted.

Standalone use
--------------
The executor runs without the *planner agent* — no LLM planning step, no
`planner_agent` package — but never without `plan_core`, which holds the
contract rather than the planning logic. A hand-written plan is exactly the
input that most needs checking: planner output at least came from a component
built against the schema, while a hand-edited file has had no such discipline
applied.

So a missing or broken `plan_core` is fatal, not a warning. Without it the
executability checks below still pass, since they only read hops and entity
refs — meaning a plan with `biolink:treatz` would reach ARAX, return nothing,
and be reported as an inconclusive result with relaxation suggestions for what
is actually a typo.

Usage
-----
    plan_input = load_plan_input("plan.json")
    if not plan_input.can_execute:
        print(plan_input.report())
        raise SystemExit(1)

    resolutions = resolver.resolve_plan(plan_input.plan)
    plan_input.check_resolution(resolutions)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

try:
    from plan_core import QueryPlan, validate_plan
    HAS_PLAN_CORE = True
except ImportError:  # pragma: no cover
    QueryPlan = None  # type: ignore
    validate_plan = None  # type: ignore
    HAS_PLAN_CORE = False


#: ARAX's connect() accepts max_path_length 1..5 (ARAX_connect.py). A plan
#: asking for more passes schema validation and is then rejected at execution,
#: so it is caught here instead.
#: Plan contract versions this executor has been checked against. An explicit
#: set, not a lower bound: accepting anything above a floor assumes the
#: contract only ever changes compatibly, and it was exactly that assumption
#: that let pre-0.8 plans run against executor defaults while appearing to
#: honour what they said. Adding a version here should follow reading its diff.
SUPPORTED_PLAN_VERSIONS = ("0.8.0", "0.10.0")

#: The version whose structures the executor was written against. Others in
#: the set are accepted because their executor-facing shape was checked to
#: match this one.
REFERENCE_PLAN_VERSION = "0.8.0"

ARAX_MAX_CONNECT_HOPS = 5

#: Paths longer than this are unlikely to complete as a single query. Not an
#: error — decomposition handles them — but worth skipping the direct attempt.
DIRECT_QUERY_HOP_WARNING = 3

SEVERITY_ERROR = "error"
SEVERITY_WARNING = "warning"


# ---------------------------------------------------------------------------
# Issues
# ---------------------------------------------------------------------------


@dataclass
class Issue:
    """One reason a plan may not run, or may not run well."""

    severity: str
    code: str
    message: str
    scope: str = "plan"          # plan | path | entity | explanation
    target: Optional[str] = None

    @property
    def blocking(self) -> bool:
        return self.severity == SEVERITY_ERROR

    def to_dict(self) -> Dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "scope": self.scope,
            "target": self.target,
            "message": self.message,
        }

    def __repr__(self) -> str:
        where = f" [{self.scope}:{self.target}]" if self.target else ""
        return f"<{self.severity.upper()}{where} {self.code}: {self.message}>"


# ---------------------------------------------------------------------------
# Loaded plan
# ---------------------------------------------------------------------------


@dataclass
class PlanInput:
    """A plan, validated and assessed for executability."""

    raw: Dict[str, Any]
    plan: Any = None
    refused: bool = False
    refusal: Optional[Dict[str, Any]] = None
    validation_errors: List[str] = field(default_factory=list)
    issues: List[Issue] = field(default_factory=list)
    blocked_paths: Dict[str, str] = field(default_factory=dict)
    schema_validated: bool = True

    # -- accessors ---------------------------------------------------------

    @property
    def question(self) -> str:
        return self.raw.get("question", "")

    @property
    def plan_mode(self) -> str:
        return self.raw.get("plan_mode", "")

    @property
    def entities_by_ref(self) -> Dict[str, Any]:
        """entity_ref -> Entity.

        The downstream modules index entities by ref; the plan stores them as a
        list. Converting once here keeps that detail out of every caller.
        """
        entities = getattr(self.plan, "entities", None)
        if isinstance(entities, dict):
            return entities
        source = entities if entities is not None else self.raw.get("entities", [])
        out: Dict[str, Any] = {}
        for e in source or []:
            ref = getattr(e, "entity_ref", None) or (
                e.get("entity_ref") if isinstance(e, dict) else None
            )
            if ref:
                out[ref] = _wrap(e) if isinstance(e, dict) else e
        return out

    @property
    def all_paths(self) -> List[Any]:
        paths = getattr(self.plan, "paths", None)
        if paths is None:
            paths = self.raw.get("paths", []) or []
        return [_wrap(p) if isinstance(p, dict) else p for p in paths]

    @property
    def active_paths(self) -> List[Any]:
        """Paths that are enabled and not blocked by an executability error."""
        out = []
        for p in self.all_paths:
            pid = _attr(p, "path_id")
            if _attr(p, "disabled", False):
                continue
            if pid in self.blocked_paths:
                continue
            out.append(p)
        return out

    @property
    def explanation_queries(self) -> List[Any]:
        eqs = getattr(self.plan, "explanation_queries", None)
        if eqs is None:
            eqs = self.raw.get("explanation_queries", []) or []
        return [_wrap(e) if isinstance(e, dict) else e for e in eqs]

    @property
    def path_priority(self) -> Dict[str, int]:
        """path_id -> position in the plan.

        Order in the plan expresses the planner's own preference, which is what
        `tie_breaker: path_priority_order` resolves ties by.
        """
        return {_attr(p, "path_id"): i for i, p in enumerate(self.all_paths)}

    @property
    def return_refs(self) -> Dict[str, str]:
        return {
            _attr(p, "path_id"): _attr(p, "return_entity_ref")
            for p in self.all_paths
        }

    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.blocking]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if not i.blocking]

    @property
    def runnable_explanation_queries(self) -> List[Any]:
        """Explanation queries that still have something to work on.

        One drawing its endpoints from a blocked discovery path has no
        candidates to explain, so it is not runnable work even though the
        query itself is well formed.
        """
        out = []
        for eq in self.explanation_queries:
            depends_on_blocked = False
            for slot in ("endpoint_a", "endpoint_b"):
                binding = _attr(eq, slot) or {}
                if _attr(binding, "binding_type") != "from_discovery":
                    continue
                # A binding draws on several paths; it still has work to do as
                # long as one of them ran.
                path_ids = list(_attr(binding, "from_path_ids") or [])
                if path_ids and all(p in self.blocked_paths for p in path_ids):
                    depends_on_blocked = True
            if not depends_on_blocked:
                out.append(eq)
        return out

    @property
    def fatal_issues(self) -> List[Issue]:
        """Blocking issues that disable the whole plan rather than one path.

        A path-scoped error only removes that path — the others may still
        answer the question. A plan- or entity-scoped error affects everything
        downstream, so it stops the run.
        """
        return [
            i for i in self.issues
            if i.blocking and i.scope in ("plan", "entity")
        ]

    @property
    def can_execute(self) -> bool:
        """True when there is work left that can actually be run.

        A refused plan is not executable and is not an error: the planner
        declined deliberately, and that decision is passed through rather than
        overridden.
        """
        if self.refused or self.validation_errors or self.fatal_issues:
            return False
        return bool(self.active_paths) or bool(self.runnable_explanation_queries)

    # -- second phase ------------------------------------------------------

    def check_resolution(self, resolutions: Dict[str, Any]) -> List[Issue]:
        """Assess executability once entity resolution has been attempted.

        A path whose anchors all failed to resolve has no pivot, exactly as if
        it had none to begin with — but that can only be known after the
        resolver runs, so it is a separate phase rather than part of loading.
        """
        new_issues: List[Issue] = []
        entities = self.entities_by_ref

        for ref, res in resolutions.items():
            if _attr(res, "is_variable", False):
                continue
            if _attr(res, "resolved", False):
                continue
            method = _attr(res, "method", "unresolved")
            new_issues.append(Issue(
                severity=SEVERITY_WARNING,
                code="entity_unresolved",
                scope="entity",
                target=ref,
                message=(
                    f"'{_attr(res, 'query', ref)}' did not resolve ({method}); "
                    f"any path anchored only on it cannot run"
                ),
            ))

        for path in self.all_paths:
            pid = _attr(path, "path_id")
            if _attr(path, "disabled", False) or pid in self.blocked_paths:
                continue

            refs = _path_entity_refs(path)
            anchors = [r for r in refs if not _attr(entities.get(r), "is_variable", False)]
            unresolved = [
                r for r in anchors
                if not _attr(resolutions.get(r), "resolved", False)
            ]
            # Every named entity must resolve, not merely one of them. A plan
            # naming a specific gene and a specific disease is asking about
            # that pair; running it with the gene left open by category asks a
            # different and much broader question, and the answer to that
            # question would be reported as if it answered this one.
            if unresolved:
                reason = (
                    f"named entities {unresolved} did not resolve; running the "
                    f"path would leave them open by category, which asks a "
                    f"broader question than the plan states"
                )
                self.blocked_paths[pid] = reason
                new_issues.append(Issue(
                    severity=SEVERITY_ERROR, code="path_no_resolved_anchor",
                    scope="path", target=pid, message=reason,
                ))

        self.issues.extend(new_issues)
        return new_issues

    # -- reporting ---------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        return {
            "question": self.question,
            "plan_mode": self.plan_mode,
            "refused": self.refused,
            "refusal": self.refusal,
            "schema_validated": self.schema_validated,
            "validation_errors": self.validation_errors,
            "can_execute": self.can_execute,
            "paths_total": len(self.all_paths),
            "paths_active": len(self.active_paths),
            "paths_blocked": self.blocked_paths,
            "explanation_queries": len(self.explanation_queries),
            "explanation_queries_runnable": len(self.runnable_explanation_queries),
            "issues": [i.to_dict() for i in self.issues],
        }

    def report(self) -> str:
        lines = [f"plan_mode={self.plan_mode}  question={self.question[:70]}"]

        if self.refused:
            lines.append(f"  REFUSED ({(self.refusal or {}).get('reason')})")
            lines.append(f"    {(self.refusal or {}).get('message', '')[:200]}")
            return "\n".join(lines)

        if not self.schema_validated:
            lines.append("  schema validation SKIPPED")
        for e in self.validation_errors[:10]:
            lines.append(f"  INVALID {e}")

        lines.append(
            f"  paths: {len(self.active_paths)} runnable of {len(self.all_paths)}"
            + (f", {len(self.runnable_explanation_queries)} of "
               f"{len(self.explanation_queries)} explanation quer(ies) runnable"
               if self.explanation_queries else "")
        )
        for issue in self.issues:
            lines.append(f"  {issue.severity.upper():7s} {issue.code}: {issue.message}")
        if not self.can_execute:
            lines.append("  -> nothing to execute")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Attribute helpers
#
# Plans arrive either as plan_core pydantic models or as plain dicts, and both
# must work so that a hand-written plan runs without the planner installed.
# ---------------------------------------------------------------------------


class _View:
    """Attribute access over a plain dict, recursively.

    Downstream modules reach into plans with attribute syntax
    (`hop.subject_ref`, `path.hops`) because that is what `plan_core` models
    provide. When a plan arrives as raw JSON — a hand-written file, or a model
    that would not construct — those accesses would fail on a dict. Wrapping
    here means every consumer works with either shape without carrying its own
    fallback.
    """

    __slots__ = ("_d",)

    def __init__(self, d: Dict[str, Any]):
        object.__setattr__(self, "_d", d)

    def __getattr__(self, name: str) -> Any:
        if name.startswith("__"):
            raise AttributeError(name)
        return _wrap(self._d.get(name))

    def __contains__(self, key: str) -> bool:
        return key in self._d

    def get(self, key: str, default: Any = None) -> Any:
        return _wrap(self._d.get(key, default))

    def to_dict(self) -> Dict[str, Any]:
        return self._d

    def __repr__(self) -> str:
        label = self._d.get("path_id") or self._d.get("entity_ref") or self._d.get("query_id")
        return f"<View {label or list(self._d)[:3]}>"


def _wrap(value: Any) -> Any:
    if isinstance(value, dict):
        return _View(value)
    if isinstance(value, list):
        return [_wrap(v) for v in value]
    return value


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    if isinstance(obj, _View):
        value = obj.get(name, default)
        return default if value is None else value
    return getattr(obj, name, default)


def _path_entity_refs(path: Any) -> List[str]:
    refs: List[str] = []
    seen = set()
    for hop in _attr(path, "hops", []) or []:
        for slot in ("subject_ref", "object_ref"):
            ref = _attr(hop, slot)
            if ref and ref not in seen:
                seen.add(ref)
                refs.append(ref)
    return refs


# ---------------------------------------------------------------------------
# Executability checks
# ---------------------------------------------------------------------------


def check_executability(plan_input: PlanInput) -> List[Issue]:
    """Structural checks this executor needs, beyond plan validity."""
    issues: List[Issue] = []
    entities = plan_input.entities_by_ref
    paths = plan_input.all_paths

    if not paths and not plan_input.explanation_queries:
        issues.append(Issue(
            SEVERITY_ERROR, "no_work", "plan has no paths and no explanation queries",
        ))

    enabled = [p for p in paths if not _attr(p, "disabled", False)]
    if paths and not enabled:
        issues.append(Issue(
            SEVERITY_ERROR, "all_paths_disabled",
            f"all {len(paths)} path(s) are disabled",
        ))

    for path in enabled:
        pid = _attr(path, "path_id", "?")
        hops = _attr(path, "hops", []) or []

        if not hops:
            plan_input.blocked_paths[pid] = "path has no hops"
            issues.append(Issue(
                SEVERITY_ERROR, "path_no_hops", "path declares no hops",
                scope="path", target=pid,
            ))
            continue

        # The anchor check: without a non-variable entity there is no CURIE to
        # pin, so neither the direct query nor decomposition has a starting
        # point. This is the single most valuable check here, because the
        # alternative is a slow timeout followed by a failure.
        refs = _path_entity_refs(path)
        anchors = [r for r in refs if not _attr(entities.get(r), "is_variable", False)]
        if not anchors:
            reason = (
                "every entity is variable, so there is no CURIE to pin; "
                "pivot-first decomposition has no pivot"
            )
            plan_input.blocked_paths[pid] = reason
            issues.append(Issue(
                SEVERITY_ERROR, "path_no_anchor", reason, scope="path", target=pid,
            ))

        missing = [r for r in refs if r not in entities]
        if missing:
            reason = f"hops reference undeclared entities {missing}"
            plan_input.blocked_paths[pid] = reason
            issues.append(Issue(
                SEVERITY_ERROR, "path_unknown_entity_ref", reason,
                scope="path", target=pid,
            ))

        if len(hops) >= DIRECT_QUERY_HOP_WARNING:
            issues.append(Issue(
                SEVERITY_WARNING, "path_long",
                f"{len(hops)} hops; the direct query will probably time out. "
                f"Consider skip_direct_over_hops={DIRECT_QUERY_HOP_WARNING} to "
                f"decompose immediately rather than waiting for it.",
                scope="path", target=pid,
            ))

        # In discovery mode the return entity is what the query solves for.
        # Pinning it to a CURIE asks a different question, and when the name is
        # a common noun ("drug", "gene") it asks nothing coherent — resolution
        # looks up the word itself and returns whatever shares that string.
        return_ref = _attr(path, "return_entity_ref")
        return_entity = entities.get(return_ref) if return_ref else None
        if (
            return_entity is not None
            and plan_input.plan_mode == "discovery"
            and not _attr(return_entity, "is_variable", False)
        ):
            reason = (
                f"return_entity_ref '{return_ref}' is not variable, but a "
                f"discovery path solves for it. Pinning the answer asks a "
                f"different question than the plan states."
            )
            plan_input.blocked_paths[pid] = reason
            issues.append(Issue(
                SEVERITY_ERROR, "path_return_not_variable", reason,
                scope="path", target=pid,
            ))

        if return_ref and return_ref not in refs:
            reason = f"return_entity_ref '{return_ref}' appears in no hop"
            plan_input.blocked_paths[pid] = reason
            issues.append(Issue(
                SEVERITY_ERROR, "path_return_not_in_hops", reason,
                scope="path", target=pid,
            ))

    for eq in plan_input.explanation_queries:
        qid = _attr(eq, "query_id", "?")
        max_hops = _attr(eq, "max_hops")

        if isinstance(max_hops, int) and max_hops > ARAX_MAX_CONNECT_HOPS:
            issues.append(Issue(
                SEVERITY_ERROR, "explanation_hops_exceed_arax",
                f"max_hops={max_hops} exceeds what ARAX connect() accepts "
                f"(1..{ARAX_MAX_CONNECT_HOPS}); the query would be rejected "
                f"at execution",
                scope="explanation", target=qid,
            ))

        for slot in ("endpoint_a", "endpoint_b"):
            binding = _attr(eq, slot) or {}
            if _attr(binding, "binding_type") != "from_discovery":
                continue
            blocked_sources = [
                p for p in (_attr(binding, "from_path_ids") or [])
                if p in plan_input.blocked_paths
            ]
            if blocked_sources:
                issues.append(Issue(
                    SEVERITY_WARNING, "explanation_depends_on_blocked_path",
                    f"{slot} draws candidates from {blocked_sources}, which "
                    f"cannot run; this explanation query will have fewer or no "
                    f"candidates to explain",
                    scope="explanation", target=qid,
                ))

    for ref, entity in entities.items():
        if _attr(entity, "is_variable", False):
            continue
        if not (_attr(entity, "name") or "").strip():
            issues.append(Issue(
                SEVERITY_ERROR, "entity_no_name",
                "non-variable entity has no name to resolve",
                scope="entity", target=ref,
            ))

    return issues


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_plan_input(
    source: Any,
    strict: bool = True,
) -> PlanInput:
    """Load and assess a plan.

    Args:
        source: path to a JSON file, a dict, or an already-built QueryPlan.
        strict: treat schema validation errors as blocking. With False they
            are recorded as warnings and execution proceeds, which is
            occasionally useful when debugging a planner. Biolink validation
            still runs either way.

    Returns:
        PlanInput. Check `.can_execute` and `.refused` before running anything.
    """
    if isinstance(source, str):
        with open(source) as f:
            raw = json.load(f)
    elif isinstance(source, dict):
        raw = source
    else:
        raw = getattr(source, "raw", None) or (
            source.model_dump(exclude_none=True) if hasattr(source, "model_dump") else {}
        )

    plan_input = PlanInput(raw=raw, plan=source if not isinstance(source, (str, dict)) else None)

    # A refusal is the planner's considered decision that the question cannot
    # be answered as asked. It is passed through untouched, and short-circuits
    # before any check that assumes a workflow exists.
    version = raw.get("plan_version")
    if version not in SUPPORTED_PLAN_VERSIONS:
        plan_input.schema_validated = False
        plan_input.issues.append(Issue(
            SEVERITY_ERROR, "unsupported_plan_version",
            f"plan_version is {version!r}; this executor accepts "
            f"{list(SUPPORTED_PLAN_VERSIONS)}.\n"
            f"  Earlier plans are refused rather than migrated. v0.8.0 "
            f"restructured ranking into candidate_ranking/explanation_ranking, "
            f"reserved evidence_policy for user-requested hard filters, and "
            f"replaced from_path_id with from_path_ids. Reading an older plan "
            f"under these rules would silently substitute executor defaults "
            f"for what the plan states.\n"
            f"  Re-generate the plan with a planner emitting one of "
            f"{list(SUPPORTED_PLAN_VERSIONS)}, or add this version to "
            f"SUPPORTED_PLAN_VERSIONS after checking its diff against "
            f"v{REFERENCE_PLAN_VERSION}.",
        ))
        return plan_input

    refusal = raw.get("refusal")
    if refusal:
        plan_input.refused = True
        plan_input.refusal = refusal
        return plan_input

    if not HAS_PLAN_CORE:
        plan_input.schema_validated = False
        plan_input.issues.append(Issue(
            SEVERITY_ERROR, "plan_core_missing",
            "plan_core is not importable, so the plan cannot be validated. "
            "Every plan must pass schema and Biolink validation before "
            "execution — an unchecked predicate or category reaches ARAX, "
            "returns nothing, and is indistinguishable from a genuine absence "
            "of data.\n"
            "  Install it alongside this package:  pip install -e ../plan-core",
        ))
        return plan_input
    else:
        try:
            result = validate_plan(raw)
            if not result.ok:
                errors = [
                    f"[{e.kind}] {e.location}: {e.message}" for e in result.errors
                ]
                plan_input.validation_errors = errors if strict else []
                for e in errors[:20]:
                    plan_input.issues.append(Issue(
                        SEVERITY_ERROR if strict else SEVERITY_WARNING,
                        "schema_invalid", e,
                    ))
        except Exception as e:
            # A validator that cannot run is not a validator that passed, so
            # this is fatal rather than a warning. The usual cause is the
            # Biolink model file being absent, which has a specific and easily
            # missed origin.
            plan_input.schema_validated = False
            plan_input.issues.append(Issue(
                SEVERITY_ERROR, "validation_unavailable",
                f"plan_core validation could not run: {e}\n"
                f"  Most likely data/biolink-model.yaml is missing or "
                f"unreadable inside plan-core.\n"
                f"  If plan-core was installed as a built wheel, its "
                f"pyproject package-data paths (../../schema, ../../data) do "
                f"not survive packaging — move schema/ and data/ inside "
                f"src/plan_core/ and read them with importlib.resources, or "
                f"reinstall editable:  pip install -e ../plan-core",
            ))
            return plan_input

    if plan_input.plan is None and HAS_PLAN_CORE:
        try:
            plan_input.plan = QueryPlan(**raw)
        except Exception:
            # Typed access is a convenience; every accessor here falls back to
            # the raw dict, so a model that will not construct is survivable.
            plan_input.plan = None

    plan_input.issues.extend(check_executability(plan_input))
    return plan_input


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Check whether a plan can be executed, without running it."
    )
    ap.add_argument("plan", help="Path to plan JSON")
    ap.add_argument("--lenient", action="store_true",
                    help="do not treat schema errors as blocking")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    try:
        plan_input = load_plan_input(args.plan, strict=not args.lenient)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Could not load plan: {e}")
        return 1

    if args.json:
        print(json.dumps(plan_input.to_dict(), indent=2))
    else:
        print(plan_input.report())

    if plan_input.refused:
        return 0
    return 0 if plan_input.can_execute else 1


if __name__ == "__main__":
    raise SystemExit(main())
