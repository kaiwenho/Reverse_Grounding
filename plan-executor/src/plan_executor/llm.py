"""
llm.py — Ollama client implementing the four LLM roles the executor needs.

    Disambiguator      resolver.py     pick the CURIE an entity name means
    ConceptVerifier    postfilter.py   is what ARAX returned the same concept
    LiteratureVerifier evidence.py     does this abstract support this edge
    CandidateReranker  rank.py         reorder the top-K after reading evidence

One `LLMAgent` satisfies all four, so a single object can be handed to every
module and the model configuration lives in one place.

Design constraints carried through every prompt
-----------------------------------------------
**Ask, do not invite confirmation.** "Confirm that this abstract supports X"
gets agreement. "What relationship, if any, does this abstract describe" gets
an answer that can disagree. Comparison against the asserted predicate happens
in Python afterwards.

**Always supply an escape hatch.** Without an explicit "none of these" or
"insufficient text" option, a model asked to choose will choose. Every schema
here has one, and using it is described as a correct answer rather than a
failure.

**Require something checkable.** Literature verdicts must quote the abstract
verbatim; rerankings must cite edge ids. Both are verified mechanically by the
calling module, and unverifiable output is discarded rather than trusted. The
model's job is to point at evidence, not to generate it.

**Never invent identifiers.** Every prompt asks for a selection from supplied
options — an index, a CURIE from the list, an edge id from the evidence. No
prompt asks the model to produce a CURIE.

Failure handling
----------------
This module raises rather than guessing. Callers differ in what a failure
means: `resolver.py` halts the run, because an unverified anchor silently
corrupts every downstream result, while `rank.py` keeps its deterministic
order, which is meaningful on its own. Deciding here would take that choice
away from them.

Usage
-----
    agent = LLMAgent(model="gpt-oss:120b")
    resolver = EntityResolver(cache=cache, disambiguator=agent)
    gatherer = EvidenceGatherer(cache=cache, verifier=agent)
    ranker = CandidateRanker(plan.ranking, reranker=agent)
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

try:
    import requests
except ImportError:  # pragma: no cover
    requests = None

from .resolver import DisambiguationChoice


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

OLLAMA_URL = "http://localhost:11434/v1/chat/completions"
DEFAULT_MODEL = "gpt-oss:120b"

#: Low but not zero. These are selection tasks, so determinism matters more
#: than variety; zero can make some models degenerate into repetition.
DEFAULT_TEMPERATURE = 0.1
DEFAULT_TIMEOUT = 120
DEFAULT_RETRIES = 2

#: Abstracts are truncated before sending. Quote verification still runs
#: against the full abstract, and a span taken from the head of the text is
#: necessarily present in the whole of it.
MAX_ABSTRACT_CHARS = 4000

#: Reasoning models (gpt-oss, deepseek-r1, qwq) emit a thinking pass before
#: their answer, and Ollama counts those tokens against the same budget. A
#: limit sized for the answer alone gets consumed by the reasoning and the
#: response comes back empty, so these are set well above what the JSON
#: itself needs.
MAX_TOKENS = {
    "disambiguate": 2000,
    "same_concept": 2000,
    "judge_literature": 2500,
    "rerank": 6000,
}


class LLMError(RuntimeError):
    """The model could not be reached, or returned unusable output."""


@dataclass
class LLMCall:
    """One request, recorded for the ledger."""

    purpose: str
    attempts: int = 0
    elapsed_s: float = 0.0
    ok: bool = False
    error: Optional[str] = None
    prompt_chars: int = 0
    response_chars: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "purpose": self.purpose,
            "attempts": self.attempts,
            "elapsed_s": round(self.elapsed_s, 2),
            "ok": self.ok,
            "error": self.error,
            "prompt_chars": self.prompt_chars,
            "response_chars": self.response_chars,
        }


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_FENCE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


def extract_json(text: str) -> Dict[str, Any]:
    """Parse a JSON object out of a model response.

    Models wrap JSON in fences or prose even when told not to, so the raw
    parse is attempted first and the brace-span fallback second.
    """
    if not text or not text.strip():
        raise LLMError("empty response")

    cleaned = _FENCE.sub("", text.strip())
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict):
            return parsed
        if isinstance(parsed, list):
            return {"_list": parsed}
    except json.JSONDecodeError:
        pass

    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start != -1 and end > start:
        try:
            return json.loads(cleaned[start:end + 1])
        except json.JSONDecodeError:
            pass

    start, end = cleaned.find("["), cleaned.rfind("]")
    if start != -1 and end > start:
        try:
            return {"_list": json.loads(cleaned[start:end + 1])}
        except json.JSONDecodeError:
            pass

    raise LLMError(f"no JSON object in response: {cleaned[:200]}")


# ---------------------------------------------------------------------------
# Client
# ---------------------------------------------------------------------------


class OllamaClient:
    """Thin JSON-only wrapper over Ollama's OpenAI-compatible endpoint."""

    def __init__(
        self,
        url: str = OLLAMA_URL,
        model: str = DEFAULT_MODEL,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        verbose: bool = True,
    ):
        if requests is None:
            raise ImportError("llm needs `requests` (pip install requests)")
        self.url = url
        self.model = model
        self.temperature = temperature
        self.timeout = timeout
        self.retries = retries
        self.verbose = verbose
        self.calls: List[LLMCall] = []

    def log(self, msg: str) -> None:
        if self.verbose:
            print(f"  [llm] {msg}")

    def complete_json(
        self,
        system: str,
        user: str,
        purpose: str = "generic",
        max_tokens: int = 800,
    ) -> Dict[str, Any]:
        """Send a prompt and return parsed JSON.

        A parse failure is retried with the error fed back, which recovers most
        malformed output without a second design. Transport failures are
        retried plainly.

        Raises:
            LLMError: after retries are exhausted.
        """
        record = LLMCall(purpose=purpose, prompt_chars=len(system) + len(user))
        start = time.time()
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_error = "unknown"

        for attempt in range(self.retries + 1):
            record.attempts += 1
            try:
                resp = requests.post(
                    self.url,
                    json={
                        "model": self.model,
                        "messages": messages,
                        "temperature": self.temperature,
                        "max_tokens": max_tokens,
                        "response_format": {"type": "json_object"},
                    },
                    timeout=self.timeout,
                )
                resp.raise_for_status()
                message = resp.json()["choices"][0]["message"]
                content = message.get("content") or ""
                # Reasoning models sometimes return their answer only in
                # the thinking channel, leaving content empty. The JSON
                # is usually still in there, so it is worth looking
                # before declaring the call a failure.
                if not content.strip():
                    for alt_field in ("reasoning", "reasoning_content", "thinking"):
                        alt = message.get(alt_field)
                        if alt and alt.strip():
                            content = alt
                            self.log(
                                f"{purpose}: content was empty; read the "
                                f"answer from '{alt_field}' instead"
                            )
                            break
                if not content.strip():
                    finish = resp.json()["choices"][0].get("finish_reason")
                    last_error = (
                        f"empty response (finish_reason={finish}); if this "
                        f"is a reasoning model it may have spent the whole "
                        f"{max_tokens}-token budget thinking"
                    )
                    self.log(f"{purpose}: {last_error}")
                    continue
                record.response_chars = len(content)
            except requests.exceptions.Timeout:
                last_error = f"timeout after {self.timeout}s"
                self.log(f"{purpose}: {last_error}")
                continue
            except requests.exceptions.RequestException as e:
                last_error = f"transport error: {e}"
                self.log(f"{purpose}: {last_error}")
                continue
            except (KeyError, IndexError, ValueError) as e:
                last_error = f"unexpected response shape: {e}"
                self.log(f"{purpose}: {last_error}")
                continue

            try:
                parsed = extract_json(content)
            except LLMError as e:
                last_error = str(e)
                self.log(f"{purpose}: {last_error}")
                # Feeding the failure back usually fixes formatting without
                # needing a different prompt.
                messages = messages[:2] + [
                    {"role": "assistant", "content": content[:1000]},
                    {"role": "user", "content":
                        f"That was not valid JSON ({e}). Reply with the "
                        f"JSON object only — no prose, no code fences, no "
                        f"explanation before or after."},
                ]
                continue

            record.ok = True
            record.elapsed_s = time.time() - start
            self.calls.append(record)
            return parsed

        record.elapsed_s = time.time() - start
        record.error = last_error
        self.calls.append(record)
        raise LLMError(f"{purpose} failed after {record.attempts} attempt(s): {last_error}")

    def stats(self) -> Dict[str, Any]:
        by_purpose: Dict[str, Dict[str, Any]] = {}
        for c in self.calls:
            d = by_purpose.setdefault(c.purpose, {"calls": 0, "failed": 0, "seconds": 0.0})
            d["calls"] += 1
            d["failed"] += 0 if c.ok else 1
            d["seconds"] = round(d["seconds"] + c.elapsed_s, 2)
        return {
            "model": self.model,
            "total_calls": len(self.calls),
            "failed_calls": sum(1 for c in self.calls if not c.ok),
            "by_purpose": by_purpose,
        }

    def health_check(self) -> Dict[str, Any]:
        """Confirm the model responds and returns JSON, before a run starts.

        Worth doing up front: resolution halts the whole run on LLM failure, so
        discovering the model is unreachable after several ARAX queries wastes
        time that a two-second check prevents.
        """
        try:
            out = self.complete_json(
                system="Reply with JSON only.",
                user='Reply with exactly: {"ok": true}',
                purpose="health_check",
                max_tokens=50,
            )
            return {"reachable": True, "model": self.model, "response": out}
        except (LLMError, ImportError) as e:
            return {"reachable": False, "model": self.model, "error": str(e)}


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

_SYSTEM_BASE = (
    "You are a biomedical knowledge-graph analyst. You reply with a single "
    "JSON object and nothing else — no prose, no markdown, no code fences.\n"
    "You never invent identifiers. When asked to choose, you choose only from "
    "the options given. Saying that none of the options is correct, or that "
    "the text does not settle the question, is a correct and expected answer "
    "when true."
)


class LLMAgent:
    """Implements Disambiguator, ConceptVerifier, LiteratureVerifier and
    CandidateReranker over one Ollama client."""

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        model: str = DEFAULT_MODEL,
        url: str = OLLAMA_URL,
        temperature: float = DEFAULT_TEMPERATURE,
        timeout: int = DEFAULT_TIMEOUT,
        retries: int = DEFAULT_RETRIES,
        verbose: bool = True,
    ):
        self.client = client or OllamaClient(
            url=url, model=model, temperature=temperature,
            timeout=timeout, retries=retries, verbose=verbose,
        )

    def stats(self) -> Dict[str, Any]:
        return self.client.stats()

    def health_check(self) -> Dict[str, Any]:
        return self.client.health_check()

    # -- 1. Disambiguator --------------------------------------------------

    def choose(
        self,
        question: str,
        entity_name: str,
        expected_category: str,
        candidates: Sequence[Any],
        aliases: Sequence[str] = (),
        conflation: Sequence[str] = (),
    ) -> DisambiguationChoice:
        """Pick which candidate CURIE the entity name refers to.

        Candidates are numbered from 1 in the prompt because models handle
        1-based lists more reliably, and converted to 0-based indices on the
        way out. `resolver.py` range-checks the result regardless.
        """
        listing = []
        for i, c in enumerate(candidates, 1):
            types = ", ".join(getattr(c, "bare_types", [])[:3]) or "unknown"
            syns = getattr(c, "synonyms", None) or []
            line = f'{i}. curie={c.curie} label="{c.label}" types=[{types}]'
            taxa = getattr(c, "taxa", None) or []
            if taxa:
                line += f' taxa={taxa[:2]}'
            if syns:
                line += f' synonyms={syns[:3]}'
            listing.append(line)

        conflation_note = ""
        if conflation:
            conflation_note = (
                f"\nThe plan allows these category groups to be treated as "
                f"interchangeable: {list(conflation)}. A gene and its protein "
                f"product, or a drug and its active molecule, count as the "
                f"same concept under that setting."
            )

        user = f"""A query plan needs one identifier for an entity it refers to.

Research question: {question}
Entity as written in the plan: "{entity_name}"
Expected Biolink category: {expected_category or "unspecified"}
Known aliases: {list(aliases) or "none"}{conflation_note}

Candidates returned by a name-resolution service, in its own ranked order
(that order is unreliable — judge on content, not position):

{chr(10).join(listing)}

Which candidate is the concept this question is about?

Gene symbols are shared across species. Check the taxon and the label's
casing before choosing: KDR is human, Kdr is mouse, kdrl is zebrafish.
Unless the question says otherwise it concerns humans, and an ortholog
from another organism is the wrong answer even though it carries the
same symbol.

Prefer the specific disease, gene, or molecule the question names over a
broader parent or a narrower subtype. If the question concerns a distinct
disorder that merely shares wording with a candidate, that candidate is wrong.
If none of the candidates is the intended concept, answer with index null —
that is more useful than a wrong identifier, because every result downstream
depends on this choice.

Reply with JSON:
{{"index": <1-based number, or null if none fit>,
 "reason": "<one sentence: what distinguishes your choice>",
 "confidence": "high" | "medium" | "low"}}"""

        data = self.client.complete_json(
            _SYSTEM_BASE, user, purpose="disambiguate",
            max_tokens=MAX_TOKENS["disambiguate"],
        )

        # Only an explicit null means "none of these fit". A response that
        # omits the key entirely is malformed, and treating it as a
        # rejection would report a broken model as a finding about the
        # data — the exact confusion this module exists to prevent.
        if "index" not in data:
            raise LLMError(
                f"response has no 'index' field: {json.dumps(data)[:200]}"
            )

        raw_index = data.get("index")
        index = None
        if isinstance(raw_index, (int, float)) and not isinstance(raw_index, bool):
            index = int(raw_index) - 1
        elif isinstance(raw_index, str) and raw_index.strip().isdigit():
            index = int(raw_index.strip()) - 1
        elif raw_index is not None:
            raise LLMError(f"unusable index value: {raw_index!r}")

        return DisambiguationChoice(
            index=index,
            reason=str(data.get("reason", ""))[:400],
            confidence=str(data.get("confidence", "")).lower() or None,
        )

    # -- 2. ConceptVerifier ------------------------------------------------

    def same_concept(
        self,
        question: str,
        requested_curie: str,
        requested_label: str,
        observed_curie: str,
        observed_label: str,
        observed_synonyms: Sequence[str] = (),
    ) -> tuple:
        """Judge whether a returned concept is the one that was asked for.

        Called when ARAX binds a different identifier, or the same identifier
        under a different name, than was pinned. Its internal synonymizer can
        merge concepts the plan meant to keep apart, and only some of those
        merges matter for a given question.
        """
        user = f"""A knowledge graph was queried about one concept and answered about
another. Decide whether the difference matters for this question.

Research question: {question}

Asked about:  {requested_curie} — "{requested_label}"
Answered about: {observed_curie} — "{observed_label}"
Synonyms the knowledge graph holds for what it answered about:
{list(observed_synonyms)[:12] or "none available"}

Are these the same concept for the purposes of this question?

Answer true for a synonym, an alternative identifier for the same entity, or a
naming difference with no clinical consequence. Answer false when they are
distinct entities — a different disease, a related but separate disorder, a
parent category standing in for a specific subtype, or one subtype standing in
for the whole. Results built on the wrong concept look identical to correct
results, so a difference that changes what the answer is about should be
reported as false.

Reply with JSON:
{{"same_concept": true | false,
 "reason": "<one sentence>"}}"""

        data = self.client.complete_json(
            _SYSTEM_BASE, user, purpose="same_concept",
            max_tokens=MAX_TOKENS["same_concept"],
        )
        value = data.get("same_concept")
        is_same = value is True or str(value).strip().lower() == "true"
        return is_same, str(data.get("reason", ""))[:400]

    # -- 3. LiteratureVerifier --------------------------------------------

    def judge(
        self,
        subject_label: str,
        predicate: str,
        object_label: str,
        title: str,
        abstract: str,
        subject_synonyms: Sequence[str] = (),
        object_synonyms: Sequence[str] = (),
    ) -> Dict[str, str]:
        """Judge whether an abstract describes an edge's asserted relationship.

        Phrased as an open question about what the abstract says, not as a
        request to confirm the edge: asking for confirmation reliably produces
        it. The quote is checked against the abstract by `evidence.py`, so an
        unverifiable one costs the verdict entirely.
        """
        text = abstract[:MAX_ABSTRACT_CHARS]
        truncated = " [truncated]" if len(abstract) > MAX_ABSTRACT_CHARS else ""

        subj_names = [subject_label] + list(subject_synonyms)[:6]
        obj_names = [object_label] + list(object_synonyms)[:6]

        user = f"""Read this abstract and report what relationship, if any, it describes
between two concepts.

Concept A: "{subject_label}"  (also called: {subj_names[1:] or "no other names"})
Concept B: "{object_label}"  (also called: {obj_names[1:] or "no other names"})

A knowledge graph claims: A {predicate} B.
The abstract below is cited as evidence for that claim. Judge the abstract on
its own terms — do not assume the claim is correct.

Title: {title}
Abstract: {text}{truncated}

The abstract may name either concept by any of its other names, or by a brand
or gene symbol; treat those as the same concept.

Choose one verdict:
  supports_directly    reports a finding that establishes A {predicate} B
  supports_indirectly  consistent with it, e.g. a review or background mention
  refutes              reports evidence against it
  mentions_both_no_relation  names both, but describes no relationship between
                       them (co-occurrence only)
  unrelated            does not concern this pair
  insufficient_text    too little text to judge

If you choose supports_directly or supports_indirectly, you must copy a span
of at least 15 words from the abstract, exactly as written, that shows the
relationship. That span is checked against the abstract automatically; a span
that does not appear verbatim causes the verdict to be discarded. If no such
span exists, choose one of the other verdicts instead — that is the correct
answer, not a failure.

Reply with JSON:
{{"verdict": "<one of the six>",
 "quote": "<verbatim span from the abstract, or empty string>",
 "reason": "<one sentence>"}}"""

        data = self.client.complete_json(
            _SYSTEM_BASE, user, purpose="judge_literature",
            max_tokens=MAX_TOKENS["judge_literature"],
        )
        return {
            "verdict": str(data.get("verdict", "")).strip(),
            "quote": str(data.get("quote", "") or ""),
            "reason": str(data.get("reason", ""))[:400],
        }

    # -- 4. CandidateReranker ---------------------------------------------

    def rerank(
        self,
        question: str,
        candidates: Sequence[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Reorder top candidates after reading their evidence.

        Each candidate arrives with only the edges that actually support it, so
        a citation can be checked against that set. `rank.py` rejects any
        reordering whose citations fall outside it, which is what stops a
        plausible reason attaching to the wrong candidate.
        """
        blocks = []
        for c in candidates:
            lines = [
                f"CANDIDATE {c.get('curie')} — {c.get('label')}",
                f"  current rank: {c.get('deterministic_rank')}",
                f"  supported by paths: {c.get('supporting_paths')}",
                f"  metrics: {json.dumps(c.get('metrics', {}))}",
                "  evidence (cite these edge_ids only for this candidate):",
            ]
            for e in c.get("evidence", [])[:12]:
                lines.append(
                    f"    edge_id={e.get('edge_id')} "
                    f"{e.get('subject')} --[{(e.get('predicate') or '').replace('biolink:','')}]--> "
                    f"{e.get('object')} "
                    f"knowledge_level={e.get('knowledge_level')} "
                    f"source={e.get('primary_source')}"
                )
                status = e.get("literature_status")
                explanation = {
                    "supported": None,
                    "unsupported": "cited papers were read and none supported this",
                    "no_publications": "no papers are cited; the source asserts it directly",
                    "unverified": "papers not read (outside the checking budget)",
                    "not_checked": "papers not read (outside the checking budget)",
                }.get(status)
                if explanation:
                    lines.append(f"      literature: {explanation}")

                quote = e.get("verified_quote")
                if quote:
                    lines.append(
                        f'      verified quote from {e.get("verified_quote_pmid")}: '
                        f'"{quote}"'
                    )
            blocks.append("\n".join(lines))

        user = f"""These candidates answer the question below. They are currently ordered by
a numeric score over graph metrics. Reorder them using the evidence.

Research question: {question}

{chr(10).join(blocks)}

Rank the candidates from most to least convincing as an answer to this
question, judging by what the evidence above actually says.

For every candidate you rank, cite the edge_ids that justify your reasoning,
and make the reason refer to what those edges contain — the relationship they
assert, the qualifiers on it, or the quoted text where a quote is shown. Cite
only edge_ids listed under that same candidate.

Two things make a reason unusable, and both cause the candidate to keep its
original position:

  * Naming the source databases. "Supported by chembl and gtopdb" describes
    where the record came from, which is already scored numerically before you
    see it. It tells the reader nothing they did not have.
  * Stating what you already know about the drug. "Classic approved EGFR
    inhibitor" may be true, but it does not come from the evidence here, and a
    ranking built on recall rather than on these records is not a ranking of
    these records.

Write instead about what the edges say. If a verified quote is shown, draw on
it. If not, refer to the asserted relationship and its qualifiers.

Read the literature line carefully before treating a missing quote as weakness.
Only "cited papers were read and none supported this" is evidence against a
candidate. "No papers are cited" means a curator asserted the relationship
directly, which is often the strongest evidence there is for a drug and its
target — databases of this kind record established facts without attaching a
reference. "Papers not read" means only that the checking budget stopped
before reaching it, and says nothing at all.

Ranking a candidate lower because no quote was shown, when none was ever
sought, ranks it by which records happened to be checked rather than by what
is known about it.

If the current order is already right, return it unchanged, with reasons and
citations that meet the same standard.

Reply with JSON:
{{"ranking": [
  {{"curie": "<candidate curie>",
   "rank": <1 = best>,
   "reason": "<one or two sentences, referring only to the cited edges>",
   "cited_edge_ids": ["<edge_id>", ...]}}
]}}"""

        data = self.client.complete_json(
            _SYSTEM_BASE, user, purpose="rerank",
            max_tokens=MAX_TOKENS["rerank"],
        )

        raw = data.get("ranking") or data.get("_list") or []
        out: List[Dict[str, Any]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            cited = item.get("cited_edge_ids") or item.get("cited_edges") or []
            if isinstance(cited, str):
                cited = [cited]
            rank = item.get("rank")
            out.append({
                "curie": item.get("curie"),
                "rank": int(rank) if isinstance(rank, (int, float)) else None,
                "reason": str(item.get("reason", ""))[:600],
                "cited_edge_ids": [str(e) for e in cited],
            })
        return out


# ---------------------------------------------------------------------------
# Offline stand-in
# ---------------------------------------------------------------------------


class ScriptedAgent:
    """Deterministic stand-in for tests, implementing the same four protocols.

    Not a fallback for a missing LLM — it makes no judgements. It exists so the
    executor's plumbing can be exercised without a model running.
    """

    def __init__(
        self,
        choose_index: int = 0,
        same_concept_answer: bool = True,
        verdict: str = "insufficient_text",
        quote: str = "",
        rerank_result: Optional[List[Dict[str, Any]]] = None,
    ):
        self.choose_index = choose_index
        self.same_concept_answer = same_concept_answer
        self.verdict = verdict
        self.quote = quote
        self.rerank_result = rerank_result
        self.call_log: List[str] = []

    def choose(self, question, entity_name, expected_category, candidates,
               aliases=(), conflation=()) -> DisambiguationChoice:
        self.call_log.append(f"choose:{entity_name}")
        return DisambiguationChoice(
            index=self.choose_index, reason="scripted", confidence="low",
        )

    def same_concept(self, question, requested_curie, requested_label,
                     observed_curie, observed_label, observed_synonyms=()) -> tuple:
        self.call_log.append(f"same_concept:{requested_curie}->{observed_curie}")
        return self.same_concept_answer, "scripted"

    def judge(self, subject_label, predicate, object_label, title, abstract,
              subject_synonyms=(), object_synonyms=()) -> Dict[str, str]:
        self.call_log.append(f"judge:{subject_label}-{predicate}-{object_label}")
        return {"verdict": self.verdict, "quote": self.quote, "reason": "scripted"}

    def rerank(self, question, candidates) -> List[Dict[str, Any]]:
        self.call_log.append(f"rerank:{len(candidates)}")
        return self.rerank_result or []

    def stats(self) -> Dict[str, Any]:
        return {"model": "scripted", "total_calls": len(self.call_log)}


# ---------------------------------------------------------------------------
# CLI: verify a model is usable before a run
# ---------------------------------------------------------------------------


def main() -> int:
    import argparse

    ap = argparse.ArgumentParser(description="Check the LLM is reachable and JSON-clean.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--url", default=OLLAMA_URL)
    ap.add_argument("--full", action="store_true",
                    help="also exercise each of the four roles")
    args = ap.parse_args()

    agent = LLMAgent(model=args.model, url=args.url, verbose=True)
    health = agent.health_check()
    print(f"reachable: {health['reachable']}  model: {health['model']}")
    if not health["reachable"]:
        print(f"error: {health['error']}")
        print("Start Ollama with `ollama serve` and pull the model.")
        return 1

    if args.full:
        from .resolver import Candidate

        print("\n-- disambiguate --")
        choice = agent.choose(
            question="What drugs treat idiopathic pulmonary fibrosis?",
            entity_name="idiopathic pulmonary fibrosis",
            expected_category="Disease",
            candidates=[
                Candidate(curie="MONDO:0008345", label="idiopathic pulmonary fibrosis",
                          types=["biolink:Disease"]),
                Candidate(curie="MONDO:0044335", label="familial pulmonary fibrosis",
                          types=["biolink:Disease"]),
            ],
            aliases=["IPF"],
        )
        print(f"  index={choice.index} confidence={choice.confidence}")
        print(f"  reason: {choice.reason}")

        print("\n-- same_concept --")
        same, reason = agent.same_concept(
            question="What drugs treat idiopathic pulmonary fibrosis?",
            requested_curie="MONDO:0008345",
            requested_label="idiopathic pulmonary fibrosis",
            observed_curie="MONDO:0044335",
            observed_label="familial pulmonary fibrosis",
            observed_synonyms=["hereditary pulmonary fibrosis"],
        )
        print(f"  same_concept={same} (expected False)")
        print(f"  reason: {reason}")

        print("\n-- judge_literature --")
        verdict = agent.judge(
            subject_label="pirfenidone", predicate="treats",
            object_label="idiopathic pulmonary fibrosis",
            title="Pirfenidone in patients with idiopathic pulmonary fibrosis",
            abstract=(
                "In this randomised, double-blind, placebo-controlled trial, "
                "pirfenidone significantly reduced disease progression as "
                "measured by decline in forced vital capacity in patients with "
                "idiopathic pulmonary fibrosis over 52 weeks of treatment."
            ),
        )
        print(f"  verdict={verdict['verdict']}")
        print(f"  quote={verdict['quote'][:100]}")

    print(f"\n{json.dumps(agent.stats(), indent=2)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
