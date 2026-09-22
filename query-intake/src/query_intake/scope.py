"""
scope.py — Which concepts this service covers, decided on identifiers.

A keyword list fails on the first synonym. "Abortion", "pregnancy termination",
"TOP", the same word in another language, the MONDO id typed directly — all
different strings, one concept. Chasing that with text patterns is a losing
game, and the result is a list nobody can audit and everybody quietly distrusts.

The pipeline already collapses those spellings into one identifier. So the rule
goes on the identifier, and on the ontology subtree beneath it:

    block if a pinned anchor resolves to MONDO:0005240, or to anything under it

One rule, every phrasing, every subtype. And when someone asks why a question
was refused, the answer is a sentence you can defend in a review — *"the anchor
resolved to MONDO:0000123, a descendant of MONDO:0005240, which rule R-004 added
on 2026-03-14, owner K. Ho"* — rather than *"the classifier said no"*, which is
reproducible by nobody and changes the day the model is updated.

**Descendants are precomputed, not walked at request time.** MONDO's full
release is tens of megabytes and parsing it per request would be absurd; worse,
it would make the blocked set invisible. Instead `expand_closure` runs offline
against a pinned release and writes the descendant list to a file that a human
can open and read. The runtime check is a set lookup.

**Where this runs.** Wrapped around the planner, not in front of it. The planner
is the component that decides which entities a question is actually about — it
does the extraction that would otherwise need NER or another model. So: plan,
resolve the pinned anchors, check, and only then execute. A blocked question
costs one planner call and no graph queries. It also means the check sees what
the plan *pins*, not what the sentence says, so "replace eczema with X" is
decided on whatever ended up in the plan.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set

from .messages import RefusalReason


RULES_DIR = Path(__file__).resolve().parents[2] / "rules"
DEFAULT_RULES_PATH = RULES_DIR / "scope_rules.json"
DEFAULT_CLOSURE_PATH = RULES_DIR / "scope_closure.json"


# ---------------------------------------------------------------------------
# Rules
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScopeRule:
    """One blocked subtree, and who is answerable for it.

    The metadata is not bureaucracy. A scope list in a biomedical tool is a
    policy artefact: it decides which researchers the tool is useless for. The
    fields exist so that every entry can be traced to a named person and a
    stated requirement, and so that `stale_rules()` can surface entries nobody
    has looked at in a year.
    """

    rule_id: str
    curie: str
    label: str
    owner: str
    added: str
    justification: str
    review_by: Optional[str] = None

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "ScopeRule":
        missing = [
            k for k in ("rule_id", "curie", "label", "owner", "added", "justification")
            if not raw.get(k)
        ]
        if missing:
            raise ValueError(
                f"scope rule is missing required field(s) {missing}: {raw}"
            )
        return cls(
            rule_id=str(raw["rule_id"]),
            curie=str(raw["curie"]),
            label=str(raw["label"]),
            owner=str(raw["owner"]),
            added=str(raw["added"]),
            justification=str(raw["justification"]),
            review_by=str(raw["review_by"]) if raw.get("review_by") else None,
        )


@dataclass
class ScopeRules:
    """The loaded policy, plus the precomputed descendants of each root."""

    rules: List[ScopeRule] = field(default_factory=list)
    closure: Dict[str, Set[str]] = field(default_factory=dict)
    closure_source: Optional[str] = None

    @classmethod
    def empty(cls) -> "ScopeRules":
        """No rules. The default, and a legitimate configuration.

        A service with no scope list answers every biomedical question its
        planner can ground. That is the right starting point: entries should be
        added because something requires them, not because a file wanted
        filling.
        """
        return cls()

    @classmethod
    def load(
        cls,
        rules_path: Optional[Path] = None,
        closure_path: Optional[Path] = None,
    ) -> "ScopeRules":
        rules_path = Path(rules_path or DEFAULT_RULES_PATH)
        closure_path = Path(closure_path or DEFAULT_CLOSURE_PATH)

        if not rules_path.exists():
            return cls.empty()

        raw = json.loads(rules_path.read_text(encoding="utf-8"))
        rules = [ScopeRule.from_dict(r) for r in raw.get("rules", [])]

        closure: Dict[str, Set[str]] = {}
        source = None
        if closure_path.exists():
            blob = json.loads(closure_path.read_text(encoding="utf-8"))
            source = blob.get("ontology_version")
            closure = {
                root: set(descendants)
                for root, descendants in (blob.get("closure") or {}).items()
            }

        return cls(rules=rules, closure=closure, closure_source=source)

    def blocked_ids(self) -> Set[str]:
        """Every identifier the rules block: the roots plus their descendants."""
        out: Set[str] = set()
        for rule in self.rules:
            out.add(rule.curie)
            out |= self.closure.get(rule.curie, set())
        return out

    def rule_for(self, curie: str) -> Optional[ScopeRule]:
        for rule in self.rules:
            if curie == rule.curie or curie in self.closure.get(rule.curie, set()):
                return rule
        return None

    def unexpanded(self) -> List[str]:
        """Roots with no precomputed descendants.

        Not an error — a leaf term has none, and a deployment may deliberately
        block one exact concept. But a root that was *meant* to cover a subtree
        and was never expanded silently blocks only itself, which looks like a
        working rule and is not. Surfaced so it can be checked.
        """
        return [r.curie for r in self.rules if r.curie not in self.closure]

    def stale_rules(self, today: Optional[date] = None) -> List[ScopeRule]:
        """Rules past their review date."""
        today = today or date.today()
        stale: List[ScopeRule] = []
        for rule in self.rules:
            if not rule.review_by:
                continue
            try:
                if date.fromisoformat(rule.review_by) < today:
                    stale.append(rule)
            except ValueError:
                stale.append(rule)
        return stale


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


@dataclass
class ScopeDecision:
    allowed: bool
    reason: Optional[RefusalReason] = None
    rule: Optional[ScopeRule] = None
    matched_curie: Optional[str] = None
    checked: List[str] = field(default_factory=list)
    partial: bool = False
    detail: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "allowed": self.allowed,
            "reason": self.reason.value if self.reason else None,
            "rule_id": self.rule.rule_id if self.rule else None,
            "rule_owner": self.rule.owner if self.rule else None,
            "rule_added": self.rule.added if self.rule else None,
            "matched_curie": self.matched_curie,
            "checked": self.checked,
            "partial": self.partial,
            "detail": self.detail,
        }


class ScopeGate:
    """Checks resolved identifiers against the loaded rules."""

    def __init__(self, rules: Optional[ScopeRules] = None) -> None:
        self.rules = rules or ScopeRules.empty()

    def check(
        self, curies: Iterable[str], *, partial: bool = False,
    ) -> ScopeDecision:
        checked = [str(c) for c in curies if c]
        for curie in checked:
            rule = self.rules.rule_for(curie)
            if rule is not None:
                return ScopeDecision(
                    allowed=False,
                    reason=RefusalReason.OUT_OF_SCOPE,
                    rule=rule,
                    matched_curie=curie,
                    checked=checked,
                    partial=partial,
                    detail=(
                        f"{curie} is covered by rule {rule.rule_id} "
                        f"({rule.curie}, {rule.label}), added {rule.added} by "
                        f"{rule.owner}"
                    ),
                )
        return ScopeDecision(allowed=True, checked=checked, partial=partial)


# ---------------------------------------------------------------------------
# Offline expansion
# ---------------------------------------------------------------------------


def expand_closure(
    obographs_path: Path,
    roots: Sequence[str],
    *,
    predicates: Sequence[str] = ("is_a",),
) -> Dict[str, List[str]]:
    """Compute the descendants of each root from an ontology release.

    Run offline against a pinned release — the same convention `plan-core` uses
    for `biolink-model.yaml`. The output is a plain list of identifiers that a
    reviewer can read, which is the point: a blocked set you cannot enumerate is
    a blocked set you cannot review.

    Takes an OBO Graphs JSON file (MONDO publishes one). Edges run child to
    parent, so the descendant closure is computed over the reversed graph.
    """
    blob = json.loads(Path(obographs_path).read_text(encoding="utf-8"))

    children: Dict[str, Set[str]] = {}
    for graph in blob.get("graphs", []):
        for edge in graph.get("edges", []):
            if edge.get("pred") not in predicates:
                continue
            child = _curie_from_iri(edge.get("sub"))
            parent = _curie_from_iri(edge.get("obj"))
            if child and parent:
                children.setdefault(parent, set()).add(child)

    out: Dict[str, List[str]] = {}
    for root in roots:
        seen: Set[str] = set()
        stack = [root]
        while stack:
            node = stack.pop()
            for child in children.get(node, ()):
                if child not in seen:
                    seen.add(child)
                    stack.append(child)
        out[root] = sorted(seen)
    return out


def _curie_from_iri(iri: Optional[str]) -> Optional[str]:
    """`http://purl.obolibrary.org/obo/MONDO_0005240` -> `MONDO:0005240`."""
    if not iri:
        return None
    tail = str(iri).rsplit("/", 1)[-1]
    if "_" in tail:
        prefix, _, local = tail.partition("_")
        return f"{prefix}:{local}"
    return tail if ":" in tail else None


def write_closure(
    closure: Dict[str, List[str]],
    path: Path,
    *,
    ontology_version: str,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "ontology_version": ontology_version,
                "generated_note": (
                    "Generated by query_intake.scope.expand_closure. Do not "
                    "edit by hand; edit scope_rules.json and regenerate."
                ),
                "closure": closure,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return path


# ---------------------------------------------------------------------------
# Name resolution
# ---------------------------------------------------------------------------

#: Signature of a resolver: surface names in, identifiers out.
Resolver = Callable[[Sequence[str]], Dict[str, List[str]]]


class SriNameResolver:
    """The same name-resolution service the executor uses.

    Called here only to decide scope, so it asks for one candidate per name.
    Note the consequence: the identifier checked at intake is the resolver's
    first choice, while the executor later picks under model review and may
    choose differently. That gap is why the gate errs toward the resolver's top
    candidates rather than treating one lookup as authoritative.
    """

    def __init__(
        self,
        url: str = "https://name-resolution-sri.renci.org/lookup",
        limit: int = 3,
        timeout: float = 5.0,
    ) -> None:
        self.url = url
        self.limit = limit
        self.timeout = timeout

    def __call__(self, names: Sequence[str]) -> Dict[str, List[str]]:
        import requests  # imported here so the module loads without it

        out: Dict[str, List[str]] = {}
        for name in names:
            try:
                response = requests.post(
                    self.url,
                    params={"string": name, "limit": self.limit},
                    timeout=self.timeout,
                )
                response.raise_for_status()
                payload = response.json()
            except Exception:
                out[name] = []
                continue
            if isinstance(payload, dict):
                out[name] = list(payload.keys())[: self.limit]
            elif isinstance(payload, list):
                out[name] = [
                    item.get("curie") for item in payload[: self.limit]
                    if isinstance(item, dict) and item.get("curie")
                ]
            else:
                out[name] = []
        return out


def static_resolver(mapping: Dict[str, List[str]]) -> Resolver:
    """A resolver backed by a dict. For tests and offline runs."""

    def resolve(names: Sequence[str]) -> Dict[str, List[str]]:
        return {name: list(mapping.get(name, [])) for name in names}

    return resolve


# ---------------------------------------------------------------------------
# Planner decorator
# ---------------------------------------------------------------------------


class ScopeGuardedPlanner:
    """Wraps a planner and refuses plans that pin a blocked concept.

    A decorator rather than a step in the loop, for one reason: the controller
    already knows what to do with a planner refusal. It stops, composes an
    answer from the typed reason, and returns `refused`. Expressing a scope
    block as `out_of_scope` — a reason the plan contract and the composer both
    already understand — means this needs no changes to the controller at all,
    and the refusal reaches the caller through the same path as any other.

    Both `plan` and `revise` are wrapped. A revision can move the anchor, and a
    check that only ran on the first plan would be a gate with a door beside it.
    """

    def __init__(
        self,
        inner: Any,
        gate: ScopeGate,
        resolver: Optional[Resolver] = None,
        *,
        require_biomedical: bool = False,
    ) -> None:
        self.inner = inner
        self.gate = gate
        self.resolver = resolver
        #: Refuse when no pinned anchor resolves to anything at all.
        #:
        #: Off by default. It looks like a clean "is this biomedical?" test, but
        #: the controller already handles unresolved anchors better: the
        #: executor reports `unresolved_grounding`, hands back the candidates
        #: the resolver considered, and the loop asks the planner to try a
        #: different name. Refusing here pre-empts that with a worse answer.
        #: Enable it only where a fast refusal matters more than a repair.
        self.require_biomedical = require_biomedical
        self.decisions: List[ScopeDecision] = []

    # -- PlannerPort ------------------------------------------------------

    def plan(self, question: str, *, available_inputs=None):
        attempt = self.inner.plan(question, available_inputs=available_inputs)
        return self._guard(attempt)

    def revise(self, request):
        attempt = self.inner.revise(request)
        return self._guard(attempt)

    # -- internals --------------------------------------------------------

    def _guard(self, attempt: Any) -> Any:
        if not getattr(attempt, "ok", False) or not getattr(attempt, "plan", None):
            return attempt

        names, curies = _pinned_identifiers(attempt.plan)
        resolved: List[str] = list(curies)
        partial = self.resolver is None and bool(names)

        if self.resolver is not None and names:
            for found in self.resolver(names).values():
                resolved.extend(found)

        decision = self.gate.check(resolved, partial=partial)
        self.decisions.append(decision)

        if not decision.allowed:
            return _refusal(attempt, RefusalReason.OUT_OF_SCOPE, decision.detail)

        if self.require_biomedical and names and not resolved:
            decision = ScopeDecision(
                allowed=False,
                reason=RefusalReason.NOT_BIOMEDICAL,
                checked=[],
                detail="no pinned anchor resolved to a known identifier",
            )
            self.decisions[-1] = decision
            return _refusal(attempt, RefusalReason.NOT_BIOMEDICAL, decision.detail)

        return attempt


def _pinned_identifiers(plan: Dict[str, Any]) -> tuple[List[str], List[str]]:
    """Surface names and any explicit identifiers on the plan's anchors.

    Variable entities are skipped. They are the answer the query is looking for,
    not a concept the question is about — a scope rule on "any chemical the
    query might return" would block the tool rather than a topic.
    """
    names: List[str] = []
    curies: List[str] = []

    for entity in plan.get("entities") or []:
        if not isinstance(entity, dict) or entity.get("is_variable"):
            continue
        name = entity.get("name")
        if name:
            names.append(str(name))
        binding = entity.get("input_binding") or {}
        for curie in binding.get("identifiers") or binding.get("curies") or []:
            curies.append(str(curie))

    return names, curies


def _refusal(attempt: Any, reason: RefusalReason, detail: str) -> Any:
    """Rewrite an accepted plan into a refusal the controller understands."""
    from dataclasses import replace

    refusal_plan = dict(attempt.plan or {})
    refusal_plan["refusal"] = {"reason": reason.value}

    try:
        return replace(
            attempt, ok=False, plan=refusal_plan, refused=True,
            refusal_reason=reason.value,
            note=f"refused by the scope gate: {detail}",
        )
    except TypeError:  # pragma: no cover - a PlanAttempt-like without dataclass
        attempt.ok = False
        attempt.plan = refusal_plan
        attempt.refused = True
        attempt.refusal_reason = reason.value
        attempt.note = f"refused by the scope gate: {detail}"
        return attempt
