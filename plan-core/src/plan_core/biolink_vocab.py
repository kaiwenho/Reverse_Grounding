"""
Biolink Model vocabulary loader.

Reads the local Biolink Model YAML file and extracts the sets of valid
terms the Query Planner Agent needs to reference:

- Predicate CURIEs (biolink:foo_bar) — active, non-deprecated slots descending
  from `related to`.
- Category CURIEs (biolink:FooBar) — all classes descending from `named thing`,
  plus useful mixins (e.g. `ChemicalOrDrugOrTreatment`, `GeneOrGeneProduct`).
- Predicate domain, range, inverse, and symmetry metadata for direction checks.
- Qualifier enum values — permissible values from the qualifier-related enums
  (DirectionQualifierEnum, CausalMechanismQualifierEnum, aspect enums, etc.).
- Knowledge-level and agent-type enum values.

Downstream code should treat this as the single source of truth. If Biolink
is upgraded, drop in the new YAML — no code changes required.

Design notes
------------
The Biolink YAML uses space-separated names for slots and classes
("gene associated with condition", "small molecule"). Serialized form is
snake_case for predicates ("biolink:gene_associated_with_condition") and
CamelCase for classes ("biolink:SmallMolecule"). This module handles the
conversion consistently.

Mixins are included in the category set because the model uses them as
legitimate domain/range values (e.g. domain of `treats` is
`ChemicalOrDrugOrTreatment`, a mixin).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import yaml

DEFAULT_YAML_PATH = Path(__file__).resolve().parents[2] / "data" / "biolink-model.yaml"


# ------------------------------------------------------------------------
# Naming conversions between YAML form and Biolink CURIE form
# ------------------------------------------------------------------------

def slot_name_to_curie(name: str) -> str:
    """'gene associated with condition' -> 'biolink:gene_associated_with_condition'"""
    return f"biolink:{name.replace(' ', '_')}"


def class_name_to_curie(name: str) -> str:
    """'small molecule' -> 'biolink:SmallMolecule'"""
    # Titlecase each word and join. Preserve embedded capitalization like DNA if present.
    parts = name.split(" ")
    camel = "".join(p[:1].upper() + p[1:] for p in parts)
    return f"biolink:{camel}"


def class_name_to_category(name: str) -> str:
    """'small molecule' -> 'SmallMolecule' (unprefixed form used in Entity.biolink_category)."""
    parts = name.split(" ")
    return "".join(p[:1].upper() + p[1:] for p in parts)


def _snake_only(name: str) -> str:
    return name.replace(" ", "_")


# ------------------------------------------------------------------------
# Vocabulary bundle
# ------------------------------------------------------------------------

@dataclass(frozen=True)
class BiolinkVocabulary:
    """Immutable snapshot of the vocabulary extracted from a Biolink YAML file."""

    biolink_version: str

    # Predicates
    predicate_curies: frozenset[str]           # {'biolink:treats', 'biolink:affects', ...}
    predicate_names: frozenset[str]            # {'treats', 'affects', 'gene associated with condition', ...}
    deprecated_predicate_curies: frozenset[str]  # excluded from generation/validation
    predicate_parents: Dict[str, Optional[str]]  # child_curie -> parent_curie (or None)
    predicate_domains: Dict[str, Optional[str]]  # predicate_curie -> unprefixed category
    predicate_ranges: Dict[str, Optional[str]]   # predicate_curie -> unprefixed category
    predicate_inverses: Dict[str, str]           # predicate_curie -> inverse predicate_curie
    symmetric_predicates: frozenset[str]

    # Categories (used as Entity.biolink_category values — unprefixed CamelCase)
    categories: frozenset[str]                 # {'SmallMolecule', 'Gene', 'Disease', ...}
    category_curies: frozenset[str]            # {'biolink:SmallMolecule', ...}
    category_ancestors: Dict[str, frozenset[str]]
    mixin_categories: frozenset[str]
    abstract_categories: frozenset[str]
    deprecated_categories: frozenset[str]
    category_concrete_implementers: Dict[str, frozenset[str]]

    # Qualifier enums
    direction_qualifier_values: frozenset[str]
    aspect_qualifier_values: frozenset[str]
    causal_mechanism_qualifier_values: frozenset[str]
    knowledge_level_values: frozenset[str]
    agent_type_values: frozenset[str]

    # Raw enum map for anything else callers want
    all_enums: Dict[str, frozenset[str]]

    # ---- Convenience checks -------------------------------------------------

    def is_valid_predicate(self, curie: str) -> bool:
        return curie in self.predicate_curies

    def is_deprecated_predicate(self, curie: str) -> bool:
        """Return whether the loaded model explicitly marks this predicate deprecated."""
        return curie in self.deprecated_predicate_curies

    def is_valid_category(self, category: str) -> bool:
        """Accept either unprefixed ('SmallMolecule') or prefixed ('biolink:SmallMolecule')."""
        return category in self.categories or category in self.category_curies

    def is_valid_qualifier_value(self, enum_name: str, value: str) -> bool:
        return value in self.all_enums.get(enum_name, frozenset())

    def predicate_satisfies(self, predicate: str, ancestor: str) -> bool:
        """Return whether an active predicate is `ancestor` or its descendant."""
        if (
            predicate not in self.predicate_curies
            or ancestor not in self.predicate_curies
        ):
            return False

        current: Optional[str] = predicate
        seen: Set[str] = set()
        while current is not None and current not in seen:
            if current == ancestor:
                return True
            seen.add(current)
            current = self.predicate_parents.get(current)
        return False

    def qualifier_predicate_family(self, predicate: str) -> Optional[str]:
        """Return the supported qualifier-family root for an active predicate."""
        for family_root in QUALIFIER_PREDICATE_FAMILY_ROOTS:
            if self.predicate_satisfies(predicate, family_root):
                return family_root
        return None

    def category_satisfies(self, category: str, expected: str) -> bool:
        """Return whether ``category`` satisfies an expected Biolink category.

        Ordinary classes are checked through their transitive ``is_a`` and
        ``mixins`` ancestry.  When ``category`` is itself a mixin (for example
        ``GeneOrGeneProduct``), it satisfies a broad expected class only when
        all of its active concrete implementers satisfy that class.  This
        prevents mixin query categories from producing false direction errors
        against broad domains/ranges such as ``NamedThing``.
        """
        category = category.removeprefix("biolink:")
        expected = expected.removeprefix("biolink:")
        directly_satisfies = (
            category == expected
            or expected in self.category_ancestors.get(category, frozenset())
        )
        if directly_satisfies:
            return True

        if category not in self.mixin_categories:
            return False

        implementers = self.category_concrete_implementers.get(
            category, frozenset()
        )
        return bool(implementers) and all(
            implementer == expected
            or expected in self.category_ancestors.get(
                implementer, frozenset()
            )
            for implementer in implementers
        )

    def concrete_implementers(self, category: str) -> frozenset[str]:
        """Return active, instantiable implementations of a class or mixin.

        Results are derived from the bundled Biolink YAML by traversing both
        ``is_a`` and ``mixins`` relationships.  Mixin, abstract, and deprecated
        classes are excluded.  A concrete input category is included in its
        own result alongside any concrete descendants.
        """
        normalized = category.removeprefix("biolink:")
        return self.category_concrete_implementers.get(
            normalized, frozenset()
        )


# ------------------------------------------------------------------------
# Loader
# ------------------------------------------------------------------------

_PREDICATE_ROOT = "related to"
_CATEGORY_ROOT = "named thing"

# Conservative qualifier-bearing predicate families used by the plan contract.
# The roots are an explicit planner policy; membership below each root is always
# derived from the bundled Biolink predicate hierarchy.
QUALIFIER_PREDICATE_FAMILY_ROOTS = (
    "biolink:regulates",
    "biolink:affects",
    "biolink:interacts_with",
)


def _is_true_flag(definition: dict, key: str) -> bool:
    """Handle LinkML YAML booleans serialized as true or a true-like string."""
    value = definition.get(key)
    return value is True or (
        isinstance(value, str)
        and value.strip().lower() in {"true", "yes", "1"}
    )


def _is_deprecated(definition: dict) -> bool:
    return _is_true_flag(definition, "deprecated")


def _walk_is_a(name: str, table: Dict[str, dict]) -> List[str]:
    chain: List[str] = []
    cur: Optional[str] = name
    while cur and cur in table and cur not in chain:
        chain.append(cur)
        cur = table[cur].get("is_a")
    return chain


def _descends_from(name: str, root: str, table: Dict[str, dict]) -> bool:
    return root in _walk_is_a(name, table)


def _inherited_value(name: str, key: str, table: Dict[str, dict]):
    """Return the closest value for `key` on a slot or its is_a ancestors."""
    for ancestor in _walk_is_a(name, table):
        defn = table.get(ancestor, {})
        if key in defn:
            return defn[key]
    return None


def _class_ancestors(name: str, classes: Dict[str, dict]) -> Set[str]:
    """Collect transitive is_a parents and mixins, tolerating model cycles."""
    ancestors: Set[str] = set()
    pending = [name]
    while pending:
        current = pending.pop()
        defn = classes.get(current, {})
        related = []
        parent = defn.get("is_a")
        if isinstance(parent, str):
            related.append(parent)
        mixins = defn.get("mixins", [])
        if isinstance(mixins, str):
            related.append(mixins)
        elif isinstance(mixins, list):
            related.extend(m for m in mixins if isinstance(m, str))
        for related_name in related:
            if related_name != name and related_name not in ancestors:
                ancestors.add(related_name)
                pending.append(related_name)
    return ancestors


@lru_cache(maxsize=4)
def load_biolink_vocabulary(yaml_path: Optional[str] = None) -> BiolinkVocabulary:
    """
    Load and cache the vocabulary from a Biolink YAML file.

    :param yaml_path: Path to a Biolink Model YAML. Defaults to the bundled copy
                      at data/biolink-model.yaml.
    """
    path = Path(yaml_path) if yaml_path else DEFAULT_YAML_PATH
    if not path.exists():
        raise FileNotFoundError(
            f"Biolink YAML not found at {path}. "
            "Set BIOLINK_YAML env or pass yaml_path explicitly."
        )

    with path.open() as f:
        model = yaml.safe_load(f)

    version = str(model.get("version", "unknown"))
    slots = model.get("slots", {}) or {}
    classes = model.get("classes", {}) or {}
    enums = model.get("enums", {}) or {}

    # --- Predicates: active slots descending from "related to" ----------
    # Deprecated terms remain in the bundled model for provenance and clear
    # validation errors, but they are never exposed as available predicates.
    all_pred_names = {
        name
        for name in slots
        if name == _PREDICATE_ROOT
        or _descends_from(name, _PREDICATE_ROOT, slots)
    }
    deprecated_pred_names = {
        name for name in all_pred_names if _is_deprecated(slots.get(name, {}))
    }
    pred_names = all_pred_names - deprecated_pred_names

    def nearest_active_parent(name: str) -> Optional[str]:
        """Skip deprecated ancestors without breaking the active hierarchy."""
        parent = slots.get(name, {}).get("is_a")
        seen: Set[str] = set()
        while isinstance(parent, str) and parent not in seen:
            if parent in pred_names:
                return parent
            seen.add(parent)
            parent = slots.get(parent, {}).get("is_a")
        return None

    pred_parents: Dict[str, Optional[str]] = {}
    for name in pred_names:
        parent = nearest_active_parent(name)
        pred_parents[slot_name_to_curie(name)] = (
            slot_name_to_curie(parent) if parent else None
        )

    pred_curies = {slot_name_to_curie(n) for n in pred_names}
    deprecated_pred_curies = {
        slot_name_to_curie(n) for n in deprecated_pred_names
    }

    pred_domains: Dict[str, Optional[str]] = {}
    pred_ranges: Dict[str, Optional[str]] = {}
    pred_inverses: Dict[str, str] = {}
    symmetric_predicates: Set[str] = set()
    for name in pred_names:
        curie = slot_name_to_curie(name)
        domain = _inherited_value(name, "domain", slots)
        range_ = _inherited_value(name, "range", slots)
        pred_domains[curie] = (
            class_name_to_category(domain) if isinstance(domain, str) else None
        )
        pred_ranges[curie] = (
            class_name_to_category(range_) if isinstance(range_, str) else None
        )
        inverse = slots.get(name, {}).get("inverse")
        if isinstance(inverse, str) and inverse in pred_names:
            pred_inverses[curie] = slot_name_to_curie(inverse)
        # Do not inherit symmetry from broad parents such as `related to`:
        # descendants like `treats` and `affects` are directional.
        if slots.get(name, {}).get("symmetric") is True:
            symmetric_predicates.add(curie)

    # Biolink often declares the inverse on only one member of the pair.
    for predicate, inverse in list(pred_inverses.items()):
        pred_inverses.setdefault(inverse, predicate)

    # --- Categories: classes descending from "named thing" + mixins ----
    cat_names: Set[str] = set()
    for name, defn in classes.items():
        if name == _CATEGORY_ROOT or _descends_from(name, _CATEGORY_ROOT, classes):
            cat_names.add(name)
        elif _is_true_flag(defn, "mixin"):
            # Include mixins commonly used as domain/range (e.g. ChemicalOrDrugOrTreatment).
            # Filter to those referenced anywhere as a slot domain/range.
            cat_names.add(name)  # inclusive: filter later if needed

    mixin_cat_names = {
        name for name in cat_names
        if _is_true_flag(classes.get(name, {}), "mixin")
    }
    abstract_cat_names = {
        name for name in cat_names
        if _is_true_flag(classes.get(name, {}), "abstract")
    }
    deprecated_cat_names = {
        name for name in cat_names
        if _is_deprecated(classes.get(name, {}))
    }
    concrete_cat_names = (
        cat_names
        - mixin_cat_names
        - abstract_cat_names
        - deprecated_cat_names
    )

    raw_category_ancestors = {
        name: _class_ancestors(name, classes) for name in cat_names
    }
    categories = {class_name_to_category(n) for n in cat_names}
    category_curies = {class_name_to_curie(n) for n in cat_names}
    category_ancestors = {
        class_name_to_category(name): frozenset(
            class_name_to_category(ancestor)
            for ancestor in raw_category_ancestors[name]
        )
        for name in cat_names
    }
    category_concrete_implementers = {
        class_name_to_category(target): frozenset(
            class_name_to_category(candidate)
            for candidate in concrete_cat_names
            if candidate == target
            or target in raw_category_ancestors[candidate]
        )
        for target in cat_names
    }

    # --- Qualifier enums -------------------------------------------------
    def enum_values(enum_name: str) -> frozenset[str]:
        e = enums.get(enum_name, {})
        pvs = e.get("permissible_values", {}) or {}
        return frozenset(pvs.keys())

    direction_vals = enum_values("DirectionQualifierEnum")
    # Biolink models multiple aspect enums; the most general is the
    # gene-or-gene-product-or-chemical-entity-aspect enum. We union the
    # chemical-entity variant too if present, to be permissive.
    aspect_vals = enum_values("GeneOrGeneProductOrChemicalEntityAspectEnum")
    aspect_vals = aspect_vals | enum_values("ChemicalEntityDerivativeEnum")
    causal_vals = enum_values("CausalMechanismQualifierEnum")
    knowledge_vals = enum_values("KnowledgeLevelEnum")
    agent_vals = enum_values("AgentTypeEnum")

    # All enums frozen for downstream use
    all_enums = {
        name: enum_values(name) for name in enums
    }

    return BiolinkVocabulary(
        biolink_version=version,
        predicate_curies=frozenset(pred_curies),
        predicate_names=frozenset(pred_names),
        deprecated_predicate_curies=frozenset(deprecated_pred_curies),
        predicate_parents=dict(pred_parents),
        predicate_domains=dict(pred_domains),
        predicate_ranges=dict(pred_ranges),
        predicate_inverses=dict(pred_inverses),
        symmetric_predicates=frozenset(symmetric_predicates),
        categories=frozenset(categories),
        category_curies=frozenset(category_curies),
        category_ancestors=dict(category_ancestors),
        mixin_categories=frozenset(
            class_name_to_category(name) for name in mixin_cat_names
        ),
        abstract_categories=frozenset(
            class_name_to_category(name) for name in abstract_cat_names
        ),
        deprecated_categories=frozenset(
            class_name_to_category(name) for name in deprecated_cat_names
        ),
        category_concrete_implementers=dict(category_concrete_implementers),
        direction_qualifier_values=direction_vals,
        aspect_qualifier_values=aspect_vals,
        causal_mechanism_qualifier_values=causal_vals,
        knowledge_level_values=knowledge_vals,
        agent_type_values=agent_vals,
        all_enums={k: v for k, v in all_enums.items()},
    )


# ------------------------------------------------------------------------
# CLI / quick sanity when run directly
# ------------------------------------------------------------------------

if __name__ == "__main__":
    v = load_biolink_vocabulary()
    print(f"Biolink version: {v.biolink_version}")
    print(f"Predicates:      {len(v.predicate_curies)}")
    print(f"Deprecated:      {len(v.deprecated_predicate_curies)} (excluded)")
    print(f"Categories:      {len(v.categories)}")
    print(f"Direction vals:  {sorted(v.direction_qualifier_values)}")
    print(f"Knowledge lvls:  {sorted(v.knowledge_level_values)}")
    print(f"Agent types:     {sorted(v.agent_type_values)}")
    print(f"Aspect vals:     {len(v.aspect_qualifier_values)}")
    print(f"Causal mechs:    {len(v.causal_mechanism_qualifier_values)}")
    for p in ["biolink:treats", "biolink:affects", "biolink:not_a_thing"]:
        print(f"  is_valid_predicate({p!r}) = {v.is_valid_predicate(p)}")
    for c in ["SmallMolecule", "Gene", "NotACategory"]:
        print(f"  is_valid_category({c!r}) = {v.is_valid_category(c)}")
