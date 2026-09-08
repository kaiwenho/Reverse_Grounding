"""
Archetype catalog loader.

Loads the drug-repurposing question archetype definitions from
data/archetype_catalog.json and exposes:

- `load_archetype_catalog()` — the raw parsed dict, cached.
- `archetype_tags()` — the sorted list of tag strings.
- `archetype_by_tag(tag)` — the raw archetype entry for a tag.
- `archetype_summary_for_prompt(detail_level=...)` — a formatted Markdown
  block suitable for embedding in the LLM system prompt.

Detail levels for the prompt block:

- "slim"     — definition + signals + examples per archetype (~1-2K tokens).
- "standard" — slim + key predicates + qualifier semantics + gap flags +
               global predicate-hierarchy and qualifier notes (~3-4K tokens).
               This is the default; it's what the LLM actually needs to know
               to produce correct plans, and it's small enough not to bloat
               the prompt.
- "full"     — standard + every worked Biolink path per archetype (~7-9K
               tokens). Use for eval / when debugging plan quality.

The archetype catalog and the plan schema's `archetype_tag` enum are
kept in sync manually. A test in tests/ enforces they match.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Literal

DEFAULT_CATALOG_PATH = (
    Path(__file__).resolve().parents[2] / "data" / "archetype_catalog.json"
)

DetailLevel = Literal["slim", "standard", "full"]


@lru_cache(maxsize=2)
def load_archetype_catalog(path: str = "") -> Dict[str, Any]:
    p = Path(path) if path else DEFAULT_CATALOG_PATH
    with p.open() as f:
        return json.load(f)


def archetype_tags() -> List[str]:
    cat = load_archetype_catalog()
    return sorted(a["tag"] for a in cat["archetypes"])


def archetype_by_tag(tag: str) -> Dict[str, Any]:
    cat = load_archetype_catalog()
    for a in cat["archetypes"]:
        if a["tag"] == tag:
            return a
    raise KeyError(f"Unknown archetype tag: {tag}")


# ------------------------------------------------------------------------
# Prompt formatting
# ------------------------------------------------------------------------

def _format_global_notes(cat: Dict[str, Any]) -> List[str]:
    """Global principles that apply across archetypes. Emitted once."""
    lines: List[str] = []
    if note := cat.get("predicate_hierarchy_note"):
        lines.append("**Predicate hierarchy.** " + note)
        lines.append("")
    if note := cat.get("qualifier_semantics_note"):
        lines.append("**Qualifier semantics.** " + note)
        lines.append("")
    if useful := cat.get("useful_qualifier_values"):
        lines.append("**Useful qualifier values.** " + useful.get("note", ""))
        for qual_name, groups in useful.items():
            if qual_name == "note":
                continue
            lines.append(f"- `{qual_name}`:")
            for group_name, values in groups.items():
                vals = ", ".join(f"`{v}`" for v in values)
                lines.append(f"  - {group_name}: {vals}")
        lines.append("")
    if note := cat.get("path_syntax_notes"):
        lines.append("**Path syntax.** " + note)
        lines.append("")
    if priority := cat.get("cross_cutting_predicate_priority"):
        lines.append("**High-priority predicates (used across archetypes):**")
        for p in priority:
            used = ", ".join(p.get("used_by", []))
            lines.append(
                f"- `{p['predicate']}` — {p['note']} (used by: {used})"
            )
        lines.append("")
    return lines


def _format_archetype(
    a: Dict[str, Any], detail_level: DetailLevel
) -> List[str]:
    """Format a single archetype block per the requested detail level."""
    lines: List[str] = [f"### {a['tag']} — {a['name']}", a["definition"]]

    if a.get("typical_shape"):
        lines.append(f"**Typical shape:** {a['typical_shape']}")
    if a.get("linguistic_signals"):
        sig = ", ".join(f"`{s}`" for s in a["linguistic_signals"])
        lines.append(f"**Signals:** {sig}")
    if a.get("example_questions"):
        lines.append("**Examples:**")
        for eq in a["example_questions"]:
            lines.append(f"- {eq}")

    if detail_level == "slim":
        lines.append("")
        return lines

    # standard + full: key predicates, qualifier semantics, gap flags
    if kps := a.get("key_predicates"):
        lines.append("**Key Biolink predicates:** " + ", ".join(f"`{k}`" for k in kps))

    if qs := a.get("qualifier_semantics"):
        lines.append("**Qualifier guidance:**")
        if wd := qs.get("when_directional"):
            lines.append(f"- Directional case: {wd}")
        if wo := qs.get("when_omitted"):
            lines.append(f"- Risk if omitted: {wo}")

    if input_contract := a.get("input_contract"):
        lines.append("**Input contract:**")
        if inline := input_contract.get("inline_members"):
            lines.append(f"- Inline members: {inline}")
        if available := input_contract.get("when_available"):
            lines.append(f"- Available input: {available}")
        if missing := input_contract.get("when_missing"):
            lines.append(f"- Missing input: {missing}")
        if capability := input_contract.get("planner_boundary"):
            lines.append(f"- Planner boundary: {capability}")

    if variants := a.get("planning_variants"):
        lines.append("**Planning variants:**")
        for variant in variants:
            lines.append(f"- **{variant['when']}**")
            lines.append(f"  - Candidate query: {variant['candidate_query']}")
            lines.append(f"  - Explanation query: {variant['explanation_query']}")

    if psn := a.get("predicate_selection_note"):
        lines.append(f"**Predicate selection note:** {psn}")

    if gaps := a.get("gap_flags"):
        lines.append(
            "**Biolink gaps** (add to `gaps` in the plan when this archetype is used):"
        )
        for g in gaps:
            lines.append(f"- **{g['kind']}**: {g['description']}")
            if wa := g.get("workaround"):
                lines.append(f"  - Workaround: {wa}")

    if detail_level == "full" and (paths := a.get("biolink_paths")):
        lines.append("**Worked Biolink paths:**")
        for p in paths:
            hop_str = f"{p['hop_count']}-hop" if "hop_count" in p else ""
            variant = f" · {p['variant']}" if p.get("variant") else ""
            lines.append(f"- **{p['label']}** ({hop_str}{variant})")
            lines.append(f"  - Template: `{p['template']}`")
            lines.append(f"  - Rationale: {p['rationale']}")

    lines.append("")
    return lines


def archetype_summary_for_prompt(detail_level: DetailLevel = "standard") -> str:
    """
    Markdown block describing every archetype at the requested detail level.
    Formatted for inclusion in the LLM system prompt.

    :param detail_level: One of "slim", "standard", "full". See module docstring.
    """
    cat = load_archetype_catalog()
    lines: List[str] = []
    lines.append("## Drug-repurposing question archetypes\n")
    lines.append(
        "Assign each question to one or more of the tags below in "
        "`interpretation.archetypes` and `Path.archetype_tag` / "
        "`ExplanationQuery.archetype_tag`. Prefer a single tag; use "
        "multiple only when the question genuinely spans archetypes. "
        "Use `OTHER` sparingly — prefer to refuse if truly unclassifiable.\n"
    )

    if detail_level in ("standard", "full"):
        lines.extend(_format_global_notes(cat))

    for a in cat["archetypes"]:
        lines.extend(_format_archetype(a, detail_level))

    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    level: DetailLevel = "standard"
    if len(sys.argv) > 1 and sys.argv[1] in ("slim", "standard", "full"):
        level = sys.argv[1]  # type: ignore[assignment]
    print(archetype_summary_for_prompt(level))
