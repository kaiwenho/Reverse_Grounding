"""
Smoke test: run the planner against a real gpt-oss:120b via Ollama.

Usage:
    pip install ollama
    ollama pull gpt-oss:120b
    ollama serve  # in another terminal
    PYTHONPATH=../plan-core/src:src python smoke_test.py
"""

import json
import re
from pathlib import Path

from planner_agent import PlannerAgent



QUESTIONS = [
    "Which drugs treat / are used for eczema from gluten allergy or celiac skin manifestations?",
    # "Which approved drugs inhibit human JAK2? Return the top 25 candidates, prioritizing stronger and better-sourced evidence.",
    # "How might metformin be connected to Alzheimer disease through genes, pathways, or biological processes? Show five diverse paths of at most four hops.",
    # "Which approved drugs inhibit human EGFR and might be repurposed for glioblastoma? Rank drugs by EGFR-inhibition evidence, and separately explain how EGFR may connect to glioblastoma.",
    # "Find approved drugs that may help idiopathic pulmonary fibrosis. Rank the candidates first, then attach a mechanism-of-action explanation to each without changing their order.",
    # "Find drugs that may treat rheumatoid arthritis, then rerank the candidates using the number and diversity of mechanistic explanation paths connecting each drug to the disease.",
    # "Using the supplied dataset asthma_signature, which contains a directional human gene signature, find compounds predicted to reverse it.",
    # "The human asthma signature contains the following up-regulated genes: IL4, IL5, and IL13. Which compounds may reverse it by decreasing their expression?",
    # "What prescription medicine should I personally start taking for chest pain?",
]

# Only inputs listed here are considered actually supplied. Leaving the map
# empty makes the asthma_signature question a needs_clarification boundary
# test even though its prose says "the supplied dataset".
AVAILABLE_INPUTS_BY_QUESTION = {}


def write_smoke_result(path: Path, question: str, result) -> None:
    """Persist every model response and its own validation outcome."""
    with path.open("w") as f:
        f.write(f"QUESTION: {question}\n\n")
        f.write(f"FINAL: ok={result.ok}  attempts={result.attempts}\n")
        if result.error:
            f.write(f"FINAL ERROR: {result.error}\n")

        for attempt in result.attempt_history:
            f.write(f"\n{'='*70}\nATTEMPT {attempt.number} RAW RESPONSE:\n{'='*70}\n")
            f.write(attempt.raw_response)
            if not attempt.raw_response.endswith("\n"):
                f.write("\n")
            if attempt.error:
                f.write(f"\nATTEMPT {attempt.number} ERROR:\n{attempt.error}\n")
            f.write(
                f"\n{'='*70}\nATTEMPT {attempt.number} VALIDATION:\n"
                f"{'='*70}\n"
            )
            if attempt.validation is None:
                f.write("NOT RUN (response could not be parsed as JSON)\n")
            else:
                f.write(attempt.validation.format())
                f.write("\n")

def main():
    # Keep the optional Ollama dependency out of module import so the result
    # writer can be tested and reused without installing the client package.
    from planner_agent.ollama_client import OllamaGPTOSSClient

    llm = OllamaGPTOSSClient(model="gpt-oss:120b")
    exemplars = [
        json.loads(
            Path("tests/fixtures/example_q3_signature_reversal.json").read_text()
        ),
        json.loads(
            Path("tests/fixtures/example_q7_explanation.json").read_text()
        ),
    ]

    # agent = PlannerAgent(
    #     llm=llm,
    #     exemplar_plans=exemplars,
    #     archetype_detail="standard",
    # )
    agent = PlannerAgent(llm=llm, archetype_detail="standard")

    for q in QUESTIONS:
        print(f"\n{'='*70}\nQ: {q}\n{'='*70}")
        result = agent.plan(
            q,
            available_inputs=AVAILABLE_INPUTS_BY_QUESTION.get(q, []),
        )
        Path("raw_responses").mkdir(exist_ok=True)
        slug = re.sub(r'\W+', '_', q)[:60].strip('_')
        write_smoke_result(Path(f"raw_responses/{slug}.txt"), q, result)
        print(f"ok={result.ok}  attempts={result.attempts}")
        if result.error:
            print(f"error: {result.error}")
        if result.plan:
            print(f"mode: {result.plan.plan_mode}")
            print(f"archetypes: {result.plan.interpretation.archetypes}")
            print(f"confidence: {result.plan.confidence.level}")
            if result.plan.refusal:
                print(f"refusal: {result.plan.refusal.reason} — {result.plan.refusal.message[:100]}")
            else:
                print(f"paths: {len(result.plan.paths or [])}, "
                      f"explanation_queries: {len(result.plan.explanation_queries or [])}")
                print(f"gaps: {len(result.plan.gaps or [])}")
        else:
            print("VALIDATION ERRORS:")
            print(result.validation.format()[:2000])
            print("\nRAW RESPONSE (first 1000 chars):")
            print(result.raw_response[:1000])

if __name__ == "__main__":
    main()
