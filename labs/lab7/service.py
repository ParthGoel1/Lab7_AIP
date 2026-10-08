#!/usr/bin/env python3
"""Lab 7 — the service.

    uvicorn labs.lab7.service:app --reload --port 8000
    curl -s localhost:8000/ask -H 'content-type: application/json' \
         -d '{"question":"How long do I have to file a claim?"}' | jq
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip import cache, tracing  # noqa: E402
from aip.cost import BudgetExceeded, global_budget  # noqa: E402
from aip.retrieval import DenseRetriever, Bm25Retriever, HybridRetriever
from aip.guards import ToolGuard

from labs.lab3.search import load_corpus, build_chunks
from labs.lab4.rag import answer_question
from labs.lab6.agent import run_agent
from aip.llm import _is_retryable
from aip import cache, tracing, embed

import json
import re

from sse_starlette.sse import EventSourceResponse
from litellm import completion

import aip.llm as llm
from aip.retrieval import format_context

from labs.lab4.rag import (
    ANSWER_SYSTEM,
    REFUSAL,
    validate_answer,
)

app = FastAPI(title="Aurora Policy Assistant", version="1.0")

_PIPELINE = None
_STARTED = time.time()
_SEMANTIC_CACHE = []
SEMANTIC_THRESHOLD = 0.92

def pipeline():
    global _PIPELINE

    if _PIPELINE is None:

        documents = load_corpus()

        chunks = build_chunks(
            documents,
            strategy="markdown",
            size=400
        )

        dense = DenseRetriever(chunks)

        bm25 = Bm25Retriever(chunks)

        hybrid = HybridRetriever(
            [dense, bm25],
            rrf_k=10, 
            weights=[2.0, 1.0]
        )

        guard = ToolGuard(
            max_calls=3,
            allow={
                "search_policy",
                "get_policy_details",
                "compute_premium",
            }
        )

        _PIPELINE = {
            "retriever": dense,
            "hybrid_retriever": hybrid,
            "reranker": None,
            "guard": guard,
        }

    return _PIPELINE


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    mode: str = Field(default="rag", pattern="^(rag|tools)$")


class Citation(BaseModel):
    index: int
    doc_id: str
    excerpt: str


class AskResponse(BaseModel):
    answer: str
    refused: bool
    citations: list[Citation]
    sources: list[str]
    latency_ms: float
    cost_usd: float
    cached: bool
    trace_id: str

def normalise_question(question: str) -> str:
    return " ".join(question.lower().split())

def get_sources(citations: list[Citation]) -> list[str]:
    return list(dict.fromkeys(c.doc_id for c in citations))


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    """Return an answer with citations, cost, latency and trace information."""

    t0 = time.perf_counter()

    budget = global_budget()
    cost_before = budget.spent_usd

    try:

        with tracing.trace(
            "http.ask",
            question=req.question[:120],
        ) as span:

            # --------------------------------------------------
            # B1: exact response cache
            # --------------------------------------------------

            span["cached"] = False
            span["cache_type"] = "miss"

            cache_request = {
                "question": normalise_question(req.question),
                "top_k": req.top_k,
                "mode": req.mode,
            }

            cache_key = cache.make_key(
                "lab7-response",
                cache_request,
            )

            if req.mode == "rag":

                cached_response = cache.get(cache_key)

                if cached_response is not None:

                    span["cached"] = True
                    span["cache_type"] = "exact"

                    latency_ms = (
                        time.perf_counter() - t0
                    ) * 1000

                    cached_citations = [
                        Citation(**c)
                        for c in cached_response["citations"]
                    ]

                    return AskResponse(
                        answer=cached_response["answer"],
                        refused=cached_response["refused"],
                        citations=cached_citations,
                        sources=get_sources(cached_citations),
                        latency_ms=round(latency_ms, 1),
                        cost_usd=0.0,
                        cached=True,
                        trace_id=f"{span['run_id']}:{span['span_id']}",
                    )

            # --------------------------------------------------
            # B1: semantic response cache
            # --------------------------------------------------

            question_embedding = None

            if req.mode == "rag":

                question_embedding = embed(
                    req.question,
                    input_type="query",
                )

                best_match = None
                best_score = -1.0

                for entry in _SEMANTIC_CACHE:

                    # Do not mix responses created with different top_k.
                    if entry["top_k"] != req.top_k:
                        continue

                    score = float(
                        question_embedding
                        @ entry["embedding"]
                    )

                    if score > best_score:
                        best_score = score
                        best_match = entry

                if (
                    best_match is not None
                    and best_score >= SEMANTIC_THRESHOLD
                ):

                    latency_ms = (
                        time.perf_counter() - t0
                    ) * 1000

                    cost_after = budget.spent_usd

                    request_cost = (
                        cost_after - cost_before
                    )

                    response = best_match["response"]

                    span["cached"] = True
                    span["cache_type"] = "semantic"

                    semantic_citations = [
                        Citation(**c)
                        for c in response["citations"]
                    ]

                    return AskResponse(
                        answer=response["answer"],
                        refused=response["refused"],
                        citations=semantic_citations,
                        sources=get_sources(semantic_citations),
                        latency_ms=round(latency_ms, 1),
                        cost_usd=round(
                            request_cost,
                            6,
                        ),
                        cached=True,
                        trace_id=f"{span['run_id']}:{span['span_id']}",
                    )

            # --------------------------------------------------
            # No cache hit -> run normal pipeline
            # --------------------------------------------------

            p = pipeline()

            if req.mode == "rag":

                result = answer_question(
                    req.question,
                    retriever=p["retriever"],
                    hybrid_retriever=p["hybrid_retriever"],
                    reranker=p["reranker"],
                    final_k=req.top_k,
                )

                citations = [
                    Citation(
                        index=i,
                        doc_id=hit.doc_id,
                        excerpt=hit.text[:500],
                    )
                    for i, hit in enumerate(
                        result.hits,
                        start=1,
                    )
                ]

                answer_text = result.text
                refused = result.refused

            else:

                result = run_agent(
                    req.question,
                    guard=p["guard"],
                    layers={1, 2, 3, 5},
                )

                answer_text = result["answer"]
                refused = False
                citations = []

            # --------------------------------------------------
            # Latency + cost
            # --------------------------------------------------

            latency_ms = (
                time.perf_counter() - t0
            ) * 1000

            cost_after = budget.spent_usd

            request_cost = (
                cost_after - cost_before
            )

            # --------------------------------------------------
            # Save successful RAG response
            # --------------------------------------------------

            if req.mode == "rag":

                response_to_cache = {
                    "answer": answer_text,
                    "refused": refused,
                    "citations": [
                        c.model_dump()
                        for c in citations
                    ],
                }

                # Exact cache
                cache.put(
                    cache_key,
                    "lab7-response",
                    cache_request,
                    response_to_cache,
                )

                # Semantic cache
                _SEMANTIC_CACHE.append(
                    {
                        "question": normalise_question(
                            req.question
                        ),
                        "embedding": question_embedding,
                        "top_k": req.top_k,
                        "response": response_to_cache,
                    }
                )

        return AskResponse(
            answer=answer_text,
            refused=refused,
            citations=citations,
            sources=get_sources(citations),
            latency_ms=round(latency_ms, 1),
            cost_usd=round(request_cost, 6),
            cached=False,
            trace_id=f"{span['run_id']}:{span['span_id']}",
        )

    except BudgetExceeded as exc:

        raise HTTPException(
            status_code=429,
            detail=str(exc),
        ) from exc

    except NotImplementedError:
        raise

    except Exception as exc:  # noqa: BLE001

        if _is_retryable(exc):

            raise HTTPException(
                status_code=503,
                detail="upstream model unavailable",
                headers={
                    "Retry-After": "5"
                },
            ) from exc

        raise HTTPException(
            status_code=500,
            detail="internal server error",
        ) from exc


@app.get("/health")
def health() -> dict:

    p = pipeline()

    return {
        "status": "ok",
        "uptime_s": round(time.time() - _STARTED, 1),
        "index_size": len(p["retriever"].chunks),
        "model": llm.resolve_model("MAIN"),
        "cache": cache.stats(),
    }


def read_todays_traces() -> list[dict]:
    """Read traces from all Uvicorn runs created today."""

    today = time.strftime("%Y%m%d")

    trace_dir = ROOT / ".aip_traces"

    rows = []

    for path in trace_dir.glob(f"{today}-*.jsonl"):

        run_id = path.stem

        rows.extend(
            tracing.read_traces(run_id)
        )

    return rows


@app.get("/metrics")
def metrics() -> dict:
    """Return service metrics from the current trace run."""

    import numpy as np
    from collections import Counter

    rows = read_todays_traces()

    # --------------------------------------------------
    # Requests
    # --------------------------------------------------

    requests = [
        r for r in rows
        if r.get("name") == "http.ask"
    ]

    n_requests = len(requests)

    # --------------------------------------------------
    # Latency
    # --------------------------------------------------

    latencies = [
        r["duration_ms"]
        for r in requests
        if "duration_ms" in r
    ]

    if latencies:
        p50 = float(np.percentile(latencies, 50))
        p95 = float(np.percentile(latencies, 95))
        p99 = float(np.percentile(latencies, 99))
    else:
        p50 = p95 = p99 = 0.0

    # --------------------------------------------------
    # Cache
    # --------------------------------------------------

    cache_hits = sum(
        1
        for r in requests
        if r.get("cached") is True
    )

    exact_hits = sum(
        1
        for r in requests
        if r.get("cache_type") == "exact"
    )

    semantic_hits = sum(
        1
        for r in requests
        if r.get("cache_type") == "semantic"
    )

    cache_hit_rate = (
        cache_hits / n_requests
        if n_requests
        else 0.0
    )

    # --------------------------------------------------
    # Cost
    # --------------------------------------------------

    llm_calls = [
        r for r in rows
        if r.get("name") == "llm.call"
    ]

    total_cost = sum(
        float(r.get("cost_usd", 0.0))
        for r in llm_calls
    )

    cost_per_query = (
        total_cost / n_requests
        if n_requests
        else 0.0
    )

    # --------------------------------------------------
    # Errors
    # --------------------------------------------------

    failed_requests = [
        r for r in requests
        if r.get("status") == "error"
    ]

    errors_by_type = Counter()

    for r in failed_requests:
        error_text = r.get("error", "Unknown")
        error_type = error_text.split(":", 1)[0]

        errors_by_type[error_type] += 1

    error_rate = (
        len(failed_requests) / n_requests
        if n_requests
        else 0.0
    )

    # --------------------------------------------------
    # Tool calls
    # --------------------------------------------------

    tool_spans = [
        r for r in rows
        if r.get("name") == "tool.call"
    ]

    tool_counts = Counter(
        r.get("tool", "unknown")
        for r in tool_spans
    )

    # --------------------------------------------------
    # Final metrics
    # --------------------------------------------------

    return {
        "requests": n_requests,

        "total_cost_usd": round(
            total_cost,
            6,
        ),

        "cost_per_query_usd": round(
            cost_per_query,
            6,
        ),

        "cache": {
            "hits": cache_hits,
            "exact_hits": exact_hits,
            "semantic_hits": semantic_hits,
            "hit_rate": round(
                cache_hit_rate,
                3,
            ),
        },

        "latency_ms": {
            "p50": round(p50, 1),
            "p95": round(p95, 1),
            "p99": round(p99, 1),
        },

        "errors": {
            "count": len(failed_requests),
            "rate": round(
                error_rate,
                3,
            ),
            "by_type": dict(
                errors_by_type
            ),
        },

        "tool_calls": dict(
            tool_counts
        ),

        "llm_calls": len(
            llm_calls
        ),
    }


# TODO B2: POST /ask/stream with server-sent events.
#          Then solve B3: you cannot validate citations before you have sent
#          the answer. Pick one of the three strategies and defend it.
@app.post("/ask/stream")
def ask_stream(req: AskRequest):

    if req.mode != "rag":
        raise HTTPException(
            status_code=400,
            detail="streaming currently supports rag mode only",
        )

    def event_generator():

        t0 = time.perf_counter()

        p = pipeline()

        # ------------------------------------------
        # Same retrieval logic as answer_question()
        # ------------------------------------------

        has_identifier = bool(
            re.search(
                r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b",
                req.question,
            )
        )

        if (
            has_identifier
            and p["hybrid_retriever"] is not None
        ):
            hits = p["hybrid_retriever"].search(
                req.question,
                k=12,
            )

        else:
            hits = p["retriever"].search(
                req.question,
                k=12,
            )

        if p["reranker"] is not None:
            hits = p["reranker"].rerank(
                req.question,
                hits,
            )

        hits = hits[:req.top_k]

        context = format_context(hits)

        # Retrieval + context preparation is now complete.
        retrieval_done = time.perf_counter()

        prompt = f"""\
Question:
{req.question}

Sources:
{context}
"""

        # ------------------------------------------
        # Same model choice as aip.llm.chat()
        # ------------------------------------------

        model = llm.resolve_model("SMALL")

        messages = llm._normalise(
            prompt,
            ANSWER_SYSTEM,
        )

        # ------------------------------------------
        # Real streaming provider call
        # ------------------------------------------

        stream = completion(
            model=model,
            messages=messages,
            temperature=llm.settings.temperature,
            max_tokens=1024,
            timeout=llm.settings.timeout_s,
            stream=True,
        )

        full_text = ""
        first_token_time = None
        finish_reason = None

        # Initialise these so they always exist.
        retrieval_ms = (
            retrieval_done - t0
        ) * 1000

        model_ttft_ms = None

        for chunk in stream:

            choice = chunk.choices[0]

            delta = getattr(
                choice.delta,
                "content",
                None,
            )

            if delta:

                if first_token_time is None:
                    first_token_time = time.perf_counter()

                    # How long from context being ready
                    # until the model emitted its first text.
                    model_ttft_ms = (
                        first_token_time - retrieval_done
                    ) * 1000

                full_text += delta

                yield {
                    "event": "token",
                    "data": json.dumps({
                        "text": delta,
                    }),
                }

            chunk_finish = getattr(
                choice,
                "finish_reason",
                None,
            )

            if chunk_finish is not None:
                finish_reason = chunk_finish

        # ------------------------------------------
        # B3: validate only after stream completes
        # ------------------------------------------

        check = validate_answer(
            full_text,
            n_sources=len(hits),
            finish_reason=finish_reason,
        )

        total_ms = (
            time.perf_counter() - t0
        ) * 1000

        if first_token_time is None:

            # Defensive fallback in case the provider
            # returned no text chunks.
            ttft_ms = total_ms

            model_ttft_ms = (
                total_ms - retrieval_ms
            )

        else:

            ttft_ms = (
                first_token_time - t0
            ) * 1000

        yield {
            "event": "validation",
            "data": json.dumps({
                "valid": check["valid"],
                "refused": check["refused"],
                "reason": check["reason"],
            }),
        }

        yield {
            "event": "done",
            "data": json.dumps({
                "ttft_ms": round(ttft_ms, 1),
                "total_ms": round(total_ms, 1),

                # New timing breakdown
                "retrieval_ms": round(
                    retrieval_ms,
                    1,
                ),
                "model_ttft_ms": round(
                    model_ttft_ms,
                    1,
                ),

                "citations": [
                    {
                        "index": i,
                        "doc_id": hit.doc_id,
                        "excerpt": hit.text[:500],
                    }
                    for i, hit in enumerate(
                        hits,
                        start=1,
                    )
                ],
            }),
        }

    return EventSourceResponse(
        event_generator()
    )