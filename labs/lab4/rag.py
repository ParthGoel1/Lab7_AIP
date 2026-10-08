#!/usr/bin/env python3
"""Lab 4 — your RAG pipeline.

Write this yourself. `aip/rag.py` is the reference implementation; look at it
after Part A, not before. Labs 5-7 build on whichever of the two you prefer,
but you must be able to explain every line of the one you use.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.guards import UNTRUSTED_SYSTEM_CLAUSE, delimit_untrusted  # noqa: E402
from aip.llm import chat  # noqa: E402
from aip.retrieval import Hit, Retriever, format_context  # noqa: E402
from aip.chunking import markdown_chunks
from aip import tracing

# The exact string the system must emit when it cannot answer. Exact, because
# downstream code detects refusal by matching it -- a paraphrase is a bug.
REFUSAL = "I don't have enough information in the provided sources to answer that."

# TODO A: write this before you read aip/rag.py::ANSWER_SYSTEM.
#ANSWER_SYSTEM = f"""\
#TODO. Six required elements, see the handout Part A. Include
#{UNTRUSTED_SYSTEM_CLAUSE!r} or your own equivalent -- Lab 6 will attack this.
#"""

'''
ANSWER_SYSTEM = f"""\
You answer questions using ONLY the numbered sources provided.

Rules, in priority order:
1. If the sources do not contain the answer, reply exactly:
   "I don't have enough information in the provided sources to answer that."
   Do not guess, and do not fall back on general knowledge.

2. If only part of the question can be answered from the sources, answer the
   supported part with citations and explicitly state that the unsupported part
   cannot be answered from the provided sources.

3. Every factual sentence must end with a citation of the source(s) that
   support it, in the form [1] or [2][5].

4. Never cite a number that was not given to you.

5. If sources disagree, say so and cite both.

6. Be concise. Two or three sentences unless the question needs more.

{UNTRUSTED_SYSTEM_CLAUSE}
'''

ANSWER_SYSTEM = f"""\
You answer questions using ONLY the numbered sources provided.

Rules, in priority order:
1. If the sources do not contain the answer, reply exactly:
   "I don't have enough information in the provided sources to answer that."
   Do not guess, and do not fall back on general knowledge.

2. If only part of the question can be answered from the sources, answer the
   supported part with citations and explicitly state that the unsupported part
   cannot be answered from the provided sources.

3. Every factual sentence must end with a citation of the source(s) that
   support it, in the form [1] or [2][5].

4. Never cite a number that was not given to you.

5. If sources disagree, say so and cite both.

6. For multi-part, comparison, or multi-hop questions:
   first identify all parts that must be answered,
   answer each part using the provided sources,
   and only then combine them into the final response.
   Do not omit a requested condition, comparison, or plan-specific detail.

7. Be concise. Two or three sentences unless the question needs more.

{UNTRUSTED_SYSTEM_CLAUSE}
"""

ANSWER_SYSTEM_STRICT = f"""\
You answer questions using ONLY the numbered sources provided.

Rules, in priority order:

1. Answer only when the sources clearly and directly support the requested fact.
   If support is incomplete, indirect, ambiguous, or requires inference, reply exactly:
   "I don't have enough information in the provided sources to answer that."
   Do not guess, infer missing details, combine weak clues, or use general knowledge.

2. If only part of the question is clearly and directly supported, answer only
   that supported part with citations and explicitly state that the unsupported
   part cannot be answered from the provided sources.

3. Every factual sentence must end with a citation of the source(s) that
   support it, in the form [1] or [2][5].

4. Never cite a number that was not given to you.

5. If sources disagree, say so and cite both.

6. Be concise. Two or three sentences unless the question needs more.

{UNTRUSTED_SYSTEM_CLAUSE}
"""



@dataclass
class Answer:
    question: str
    text: str
    hits: list[Hit] = field(default_factory=list)
    refused: bool = False
    citations_valid: bool = False
    invalid_citations: list[int] = field(default_factory=list)
    n_citations: int = 0
    truncated: bool = False


def validate_answer(text: str, n_sources: int, finish_reason: str | None = None) -> dict:

    refused = False
    truncated = False
    invalid_citations = []

    # check if the model refused
    if REFUSAL in text:
        refused = True

    # check if the answer was cut off
    if finish_reason == "length":
        truncated = True

    # find all citations like [1], [2], etc.
    citations = re.findall(r"\[(\d+)\]", text)
    citations = [int(x) for x in citations]

    # check if any citation is outside the source range
    for citation in citations:
        if citation < 1 or citation > n_sources:
            invalid_citations.append(citation)

    valid = True
    reason = "ok"

    if text.strip() == "":
        valid = False
        reason = "empty answer"

    elif truncated:
        valid = False
        reason = "answer was truncated"

    elif len(invalid_citations) > 0:
        valid = False
        reason = "invalid citation"

    elif text.strip() != REFUSAL and len(citations) == 0:
        valid = False
        reason = "no citation"

    return {
        "valid": valid,
        "refused": refused,
        "invalid_citations": invalid_citations,
        "n_citations": len(citations),
        "truncated": truncated,
        "reason": reason
    }


def answer_question(
    question: str,
    retriever: Retriever,
    hybrid_retriever=None,
    *,
    k: int = 12,
    final_k: int = 5,
    reranker=None,
    tier: str = "MAIN",
) -> Answer:

    # Check whether the question contains an exact-looking identifier
    has_identifier = bool(
        re.search(r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b", question)
    )

    # 1. Choose retriever
    if has_identifier and hybrid_retriever is not None:
        hits = hybrid_retriever.search(
            question,
            k=k,
        )
    else:
        hits = retriever.search(
            question,
            k=k,
        )

    # 2. Rerank if a reranker was supplied
    if reranker is not None:
        hits = reranker.rerank(
            question,
            hits,
        )

    # Keep only the final chunks
    hits = hits[:final_k]

    # 3. Convert retrieved chunks into numbered sources
    context = format_context(hits)

    # 4. Build the user prompt
    prompt = f"""\
Question:
{question}

Sources:
{context}
"""

    # 5. Generate the first answer
    response = chat(
        prompt,
        system=ANSWER_SYSTEM,
        tier=tier,
        return_full=True,
        reasoning_effort = 'low',
        max_tokens=1024
    )

    text = response["text"]
    finish_reason = response["finish_reason"]

    # --------------------------------------------------
    # 6. Validate the answer
    # --------------------------------------------------

    with tracing.trace(
        "rag.validate",
        n_sources=len(hits),
    ) as validation_span:

        check = validate_answer(
            text,
            n_sources=len(hits),
            finish_reason=finish_reason,
        )

        validation_span["valid"] = check["valid"]

    # --------------------------------------------------
    # 7. If invalid, retry once
    # --------------------------------------------------

    if not check["valid"]:

        repair_prompt = f"""\
Question:
{question}

Sources:
{context}

Your previous answer was invalid because: {check["reason"]}

Answer the question again.
Use only the numbered sources.
Use only citation numbers that exist.
Every factual answer must contain citations.
"""

        response = chat(
            repair_prompt,
            system=ANSWER_SYSTEM,
            tier=tier,
            return_full=True,
            reasoning_effort = 'low', 
            max_tokens=1024
        )

        text = response["text"]
        finish_reason = response["finish_reason"]

        # Validate repaired answer
        with tracing.trace(
            "rag.validate",
            n_sources=len(hits),
        ) as validation_span:

            check = validate_answer(
                text,
                n_sources=len(hits),
                finish_reason=finish_reason,
            )

            validation_span["valid"] = check["valid"]

    # --------------------------------------------------
    # 8. If repair also failed, safely refuse
    # --------------------------------------------------

    if not check["valid"]:

        text = REFUSAL

        # Validate final refusal too
        with tracing.trace(
            "rag.validate",
            n_sources=len(hits),
        ) as validation_span:

            check = validate_answer(
                text,
                n_sources=len(hits),
            )

            validation_span["valid"] = check["valid"]

    # --------------------------------------------------
    # 9. Return final answer
    # --------------------------------------------------

    return Answer(
        question=question,
        text=text,
        hits=hits,
        refused=check["refused"],
        citations_valid=check["valid"],
        invalid_citations=check["invalid_citations"],
        n_citations=check["n_citations"],
        truncated=check["truncated"],
    )


def answer_with_gold_context(question: str, gold_docs: list[str], *,
                             tier: str = "MAIN") -> Answer:
    """Use the gold documents directly instead of retrieving anything."""

    gold_chunks = []

    # Chunk each gold document
    for i, doc in enumerate(gold_docs):
        chunks = markdown_chunks(
            doc,
            doc_id=f"gold-{i+1}",
            size=400
        )

        gold_chunks.extend(chunks)

    # Build numbered context manually
    context_parts = []
    total_chars = 0

    for i, chunk in enumerate(gold_chunks, start=1):

        source = (
            f"[{i}] (source: {chunk.doc_id})\n"
            f"{chunk.text.strip()}\n"
        )

        # Keep the context within the same rough limit
        if total_chars + len(source) > 8000:
            break

        context_parts.append(source)
        total_chars += len(source)

    context = "\n".join(context_parts)

    # Ask the model using the same prompt style as answer_question()
    prompt = f"""\
                Question:
                {question}

                Sources:
                {context}
                """

    response = chat(
        prompt,
        system=ANSWER_SYSTEM,
        tier=tier,
        return_full=True,
    )

    text = response["text"]
    finish_reason = response["finish_reason"]

    # Validate the answer in the same way
    check = validate_answer(
        text,
        n_sources=len(context_parts),
        finish_reason=finish_reason
    )

    # Retry once if the answer is invalid
    if not check["valid"]:

        repair_prompt = f"""\
                            Question:
                            {question}

                            Sources:
                            {context}

                            Your previous answer was invalid because: {check["reason"]}

                            Answer again using only the numbered sources.
                            Use citations for factual claims.
                            Do not cite a source number that does not exist.
                            """

        response = chat(
            repair_prompt,
            system=ANSWER_SYSTEM,
            tier=tier,
            return_full=True
        )

        text = response["text"]
        finish_reason = response["finish_reason"]

        check = validate_answer(
            text,
            n_sources=len(context_parts),
            finish_reason=finish_reason
        )

    # If it still fails validation, safely refuse
    if not check["valid"]:

        text = REFUSAL

        check = validate_answer(
            text,
            n_sources=len(context_parts)
        )

    return Answer(
        question=question,
        text=text,
        hits=[],
        refused=check["refused"],
        citations_valid=check["valid"],
        invalid_citations=check["invalid_citations"],
        n_citations=check["n_citations"],
        truncated=check["truncated"]
    )