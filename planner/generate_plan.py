"""Generate one complete, validated JSON query plan.

Run from the ``planner`` directory:

    PYTHONPATH=../plan-core/src:src python generate_plan.py \
      "Which approved drugs inhibit human JAK2?" \
      --out ../plan-executor/plans/jak2_plan.json

If ``--out`` is omitted, the JSON plan is written to stdout. Diagnostics are
written to stderr so stdout can be redirected safely. ``PlannerAgent`` stamps
the authoritative plan and Biolink versions before validation; the LLM does
not choose those values.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

from planner_agent import LLMClient, PlannerAgent, PlannerResult


class PlanGenerationError(RuntimeError):
    """Raised when the planner cannot produce a valid plan after its retry."""

    def __init__(self, result: PlannerResult):
        self.result = result
        detail = result.error or result.validation.format()
        super().__init__(detail)


def generate_complete_plan(
    question: str,
    *,
    llm: LLMClient,
    available_inputs: Optional[Iterable[dict]] = None,
    include_full_schema: bool = False,
    archetype_detail: str = "standard",
) -> dict[str, Any]:
    """Return a validated, JSON-serializable plan including both versions."""
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")

    agent = PlannerAgent(
        llm=llm,
        include_full_schema=include_full_schema,
        archetype_detail=archetype_detail,
    )
    result = agent.plan(question, available_inputs=available_inputs)
    if not result.ok or result.plan is None:
        raise PlanGenerationError(result)
    return result.plan.to_json_dict()


def _load_available_inputs(path: Optional[str]) -> list[dict]:
    if path is None:
        return []
    value = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise ValueError("--available-inputs must contain a JSON array")
    return value


def _question_from_args(value: Optional[str], parser: argparse.ArgumentParser) -> str:
    if value is not None:
        question = value.strip()
    elif not sys.stdin.isatty():
        question = sys.stdin.read().strip()
    else:
        parser.error("provide a question argument or pipe one through stdin")
    if not question:
        parser.error("the biomedical question cannot be empty")
    return question


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Turn one natural-language biomedical question into a complete, "
            "validated JSON query plan."
        )
    )
    parser.add_argument(
        "question",
        nargs="?",
        help="Natural-language biomedical question; reads stdin when omitted.",
    )
    parser.add_argument(
        "--out",
        metavar="FILE",
        help="Write JSON to FILE. Omit this option, or use '-', for stdout.",
    )
    parser.add_argument(
        "--available-inputs",
        metavar="FILE",
        help="JSON array describing authoritative external input bindings.",
    )
    parser.add_argument("--model", default="gpt-oss:120b")
    parser.add_argument("--host", help="Ollama server URL.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--num-ctx", type=int, default=32768)
    parser.add_argument(
        "--archetype-detail",
        choices=("slim", "standard", "full"),
        default="standard",
    )
    parser.add_argument(
        "--full-schema",
        action="store_true",
        help="Include the complete compiled JSON Schema in the LLM prompt.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    question = _question_from_args(args.question, parser)

    try:
        available_inputs = _load_available_inputs(args.available_inputs)
        # Keep Ollama optional for code that imports generate_complete_plan and
        # supplies a different LLMClient implementation.
        from planner_agent.ollama_client import OllamaGPTOSSClient

        llm = OllamaGPTOSSClient(
            model=args.model,
            temperature=args.temperature,
            host=args.host,
            num_ctx=args.num_ctx,
        )
        plan = generate_complete_plan(
            question,
            llm=llm,
            available_inputs=available_inputs,
            include_full_schema=args.full_schema,
            archetype_detail=args.archetype_detail,
        )
    except PlanGenerationError as exc:
        print(
            f"Planner failed after {exc.result.attempts} attempt(s):\n{exc}",
            file=sys.stderr,
        )
        return 1
    except (ImportError, OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
        print(f"Cannot generate plan: {exc}", file=sys.stderr)
        return 2
    except Exception as exc:
        print(f"LLM or planner call failed: {exc}", file=sys.stderr)
        return 2

    payload = json.dumps(plan, indent=2, ensure_ascii=False) + "\n"
    if args.out is None or args.out == "-":
        sys.stdout.write(payload)
    else:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload, encoding="utf-8")
        print(f"Wrote validated plan to {output_path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
