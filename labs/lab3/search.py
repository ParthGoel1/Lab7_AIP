#!/usr/bin/env python3
"""Lab 3 — retrieval sweeps.

The scaffolding (corpus loading, metric computation, table printing) is
written for you. The sweeps are yours.

    python labs/lab3/search.py --baseline
    python labs/lab3/search.py --sweep chunking
    python labs/lab3/search.py --sweep retrieval
    python labs/lab3/search.py --sweep rerank
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import STRATEGIES, Chunk  # noqa: E402
from aip.evals import retrieval_metrics  # noqa: E402
from aip.retrieval import Bm25Retriever, DenseRetriever, HybridRetriever, Retriever, CrossEncoderReranker, LLMReranker  # noqa: E402
from aip.retrieval import ChromaRetriever

CORPUS_DIR = ROOT / "data/corpus"
GOLDEN = ROOT / "data/eval/rag_golden.jsonl"


# ---------------------------------------------------------------------------
# scaffolding (provided)
# ---------------------------------------------------------------------------
def load_corpus() -> dict[str, str]:
    return {p.stem: p.read_text(encoding="utf-8") for p in sorted(CORPUS_DIR.glob("*.md"))}


def load_questions(include_unanswerable: bool = False) -> list[dict]:
    rows = [json.loads(l) for l in GOLDEN.open(encoding="utf-8")]
    if include_unanswerable:
        return rows
    # THREE questions (Q36, Q38, Q39) have no relevant document, so recall and
    # nDCG are undefined for them -- you cannot rank correctly against an empty
    # relevant set. Dropping them leaves n = 42.
    #
    # Do not confuse that with the FIVE questions of kind 'unanswerable'
    # (Q36-Q40): two of those do keep relevant documents, because part of what
    # they ask is supported. All five are measured properly in Lab 4, as
    # refusal precision and recall.
    #
    # Excluding the three is correct -- but say so in your report rather than
    # letting an unexplained n = 42 pass for a stated 45.
    return [r for r in rows if r["relevant_docs"]]


def build_chunks(corpus: dict[str, str], strategy: str = "sliding",
                 size: int = 800, **kw) -> list[Chunk]:
    fn = STRATEGIES[strategy]
    out: list[Chunk] = []
    for doc_id, text in corpus.items():
        try:
            out.extend(fn(text, doc_id, size=size, **kw))
        except TypeError:                       # chunker without that kwarg
            out.extend(fn(text, doc_id, size=size))
    return out


def evaluate(retriever: Retriever, questions: list[dict], k: int = 10,
             reranker=None, final_k: int = 5) -> dict:
    """Run every question, return aggregate metrics + per-kind breakdown."""
    agg: dict[str, list[float]] = defaultdict(list)
    by_kind: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    latencies: list[float] = []
    per_q: dict[str, float] = {}
    per_q_mrr: dict[str, float] = {}

    for q in questions:
        t0 = time.perf_counter()
        hits = retriever.search(q["question"], k=k)
        if reranker is not None:
            hits = reranker.rerank(q["question"], hits, k=final_k)
        latencies.append((time.perf_counter() - t0) * 1000)

        # A document counts as retrieved at rank r if any of its chunks does.
        seen, ranked = set(), []
        for h in hits:
            if h.doc_id not in seen:
                seen.add(h.doc_id)
                ranked.append(h.doc_id)

        m = retrieval_metrics(ranked, q["relevant_docs"], ks=(1, 3, 5, 10))
        per_q[q["id"]] = m["hit_rate@5"]
        per_q_mrr[q["id"]] = m["mrr"]
        for key, val in m.items():
            agg[key].append(val)
            by_kind[q["kind"]][key].append(val)

    out = {k2: statistics.fmean(v) for k2, v in agg.items()}
    out["latency_p50_ms"] = statistics.median(latencies)
    out["latency_p95_ms"] = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    out["_by_kind"] = {kind: {k2: statistics.fmean(v) for k2, v in d.items()}
                       for kind, d in by_kind.items()}
    out["_per_question"] = per_q            # hit_rate@5 -- saturated, see kind_table
    out["_per_question_mrr"] = per_q_mrr    # use this one for Part B
    out["_kind_n"] = {kind: len(d["mrr"]) for kind, d in by_kind.items()}
    return out


def table(rows: dict[str, dict], cols: tuple[str, ...] =
          ("hit_rate@1", "hit_rate@5", "recall@5", "mrr", "ndcg@10",
           "latency_p95_ms")) -> str:
    name_w = max(len(n) for n in rows) + 2
    head = f"{'config':<{name_w}}" + "".join(f"{c:>15}" for c in cols)
    lines = [head, "-" * len(head)]
    for name, m in rows.items():
        lines.append(f"{name:<{name_w}}" + "".join(f"{m.get(c, 0):>15.4f}" for c in cols))
    return "\n".join(lines)


def kind_table(metrics: dict, col: str = "hit_rate@5") -> str:
    """Break a result down by question kind.

    NOTE the default column. `hit_rate@5` is saturated on this corpus -- every
    retriever scores 0.93-0.98 -- so this table will look flat and tell you
    nothing. Pass col='mrr' or col='ndcg@10' for Part B. The default is left
    saturated on purpose.
    """
    bk, counts = metrics["_by_kind"], metrics.get("_kind_n", {})
    w = max(len(k) for k in bk) + 2
    lines = [f"{'kind':<{w}}{col:>12}{'n':>6}", "-" * (w + 18)]
    for kind, m in sorted(bk.items()):
        lines.append(f"{kind:<{w}}{m.get(col, 0):>12.4f}{counts.get(kind, 0):>6}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# sweeps (yours)
# ---------------------------------------------------------------------------
def sweep_baseline() -> None:
    corpus, questions = load_corpus(), load_questions()
    chunks = build_chunks(corpus, "sliding", 800, overlap=150)
    print(f"corpus: {len(corpus)} docs -> {len(chunks)} chunks "
          f"(mean {statistics.fmean(len(c) for c in chunks):.0f} chars)")
    r = DenseRetriever(chunks)
    m = evaluate(r, questions)
    print(table({"baseline sliding-800 dense": m}))
    print()
    print(kind_table(m))
    print("\nWrite these numbers down before you change anything.")


def sweep_chunking() -> None:
    """Part A - compare chunking strategies and chunk sizes."""

    documents = load_corpus()
    questions = load_questions()

    # A1 - compare all chunking strategies at size 800
    a1_results = {}

    for strategy in STRATEGIES:
        chunks = build_chunks(documents, strategy=strategy, size=800)

        start = time.perf_counter()
        retriever = DenseRetriever(chunks)
        build_time = (time.perf_counter() - start) * 1000

        metrics = evaluate(retriever, questions)
        metrics["chunk_count"] = len(chunks)
        metrics["build_time_ms"] = build_time

        a1_results[strategy] = metrics

    print("\n--- A1: CHUNKING STRATEGIES ---")

    print(table(
        a1_results,
        cols=(
            "hit_rate@1",
            "recall@5",
            "mrr",
            "ndcg@10",
            "chunk_count",
            "build_time_ms"
        )
    ))

    # A2 - try different sizes using the best strategy
    a2_results = {}

    for size in [400, 800, 1600]:
        chunks = build_chunks(
            documents,
            strategy="markdown",
            size=size
        )

        start = time.perf_counter()
        retriever = DenseRetriever(chunks)
        build_time = (time.perf_counter() - start) * 1000

        metrics = evaluate(retriever, questions)
        metrics["chunk_count"] = len(chunks)
        metrics["build_time_ms"] = build_time

        a2_results[f"markdown-{size}"] = metrics

    print("\n--- A2: CHUNK SIZE ---")

    print(table(
        a2_results,
        cols=(
            "hit_rate@1",
            "recall@5",
            "mrr",
            "ndcg@10",
            "chunk_count",
            "build_time_ms"
        )
    ))

    # A3 - check whether markdown heading prefixes help
    chunks_with_headers = build_chunks(
        documents,
        strategy="markdown",
        size=400
    )

    chunks_without_headers = [
        Chunk(
            text=chunk.text.split("\n", 1)[1],
            doc_id=chunk.doc_id,
            chunk_id=chunk.chunk_id,
            meta=chunk.meta
        )
        for chunk in chunks_with_headers
    ]

    start = time.perf_counter()
    no_header_retriever = DenseRetriever(chunks_without_headers)
    build_time = (time.perf_counter() - start) * 1000

    no_header_metrics = evaluate(
        no_header_retriever,
        questions
    )

    no_header_metrics["chunk_count"] = len(chunks_without_headers)
    no_header_metrics["build_time_ms"] = build_time

    a3_results = {
        "with-header": a2_results["markdown-400"],
        "without-header": no_header_metrics
    }

    print("\n--- A3: MARKDOWN HEADERS ---")

    print(table(
        a3_results,
        cols=(
            "hit_rate@1",
            "recall@5",
            "mrr",
            "ndcg@10",
            "chunk_count",
            "build_time_ms"
        )
    ))

    # A4 - look for questions where overlap helped
    fixed_mrr = a1_results["fixed"]["_per_question_mrr"]
    sliding_mrr = a1_results["sliding"]["_per_question_mrr"]

    candidates = {}

    for qid in fixed_mrr:
        improvement = sliding_mrr[qid] - fixed_mrr[qid]

        if improvement > 0:
            candidates[qid] = improvement

    candidates = sorted(
        candidates.items(),
        key=lambda x: x[1],
        reverse=True
    )

    print("\n--- A4: POSSIBLE BOUNDARY FAILURES ---")

    for qid, improvement in candidates:
        question = next(
            q for q in questions
            if q["id"] == qid
        )

        print(
            qid,
            "| fixed:", fixed_mrr[qid],
            "| sliding:", sliding_mrr[qid],
            "| improvement:", improvement,
            "|", question["question"]
        )

    # Inspect Q45 manually
    question = next(
        q for q in questions
        if q["id"] == "Q45"
    )

    fixed_chunks = build_chunks(
        documents,
        strategy="fixed",
        size=800
    )

    fixed_retriever = DenseRetriever(fixed_chunks)
    hits = fixed_retriever.search(question["question"], k=5)

    print("\n--- A4 INSPECTION: Q45 ---")
    print("Question:", question["question"])
    print("Relevant docs:", question["relevant_docs"])

    for hit in hits:
        print(
            "\nRank:", hit.rank,
            "| Document:", hit.doc_id,
            "| Score:", round(hit.score, 4)
        )
        print(hit.text)

def sweep_retrieval() -> None:
    """Part B - compare dense, BM25 and hybrid retrieval."""

    documents = load_corpus()
    questions = load_questions()

    chunks = build_chunks(
        documents,
        strategy="markdown",
        size=400
    )

    dense = DenseRetriever(chunks)
    bm25 = Bm25Retriever(chunks)
    hybrid = HybridRetriever([dense, bm25])

    dense_metrics = evaluate(dense, questions)
    bm25_metrics = evaluate(bm25, questions)
    hybrid_metrics = evaluate(hybrid, questions)

    # B1
    print("\n--- B1: RETRIEVER COMPARISON ---")

    print(table(
        {
            "dense": dense_metrics,
            "bm25": bm25_metrics,
            "hybrid": hybrid_metrics
        },
        cols=(
            "hit_rate@1",
            "recall@5",
            "mrr",
            "ndcg@10"
        )
    ))

    # B2
    print("\n--- B2: RESULTS BY QUESTION TYPE ---")

    print("\nDense")
    print(kind_table(dense_metrics, col="mrr"))

    print("\nBM25")
    print(kind_table(bm25_metrics, col="mrr"))

    print("\nHybrid")
    print(kind_table(hybrid_metrics, col="mrr"))

    print("\nQ44 AND Q41")

    for qid in ["Q44", "Q41"]:
        print(
            qid,
            "| dense:", dense_metrics["_per_question_mrr"][qid],
            "| bm25:", bm25_metrics["_per_question_mrr"][qid],
            "| hybrid:", hybrid_metrics["_per_question_mrr"][qid]
        )

    # B3 - RRF k sweep
    rrf_results = {}

    for rrf_k in [10, 30, 60, 100]:
        retriever = HybridRetriever(
            [dense, bm25],
            rrf_k=rrf_k
        )

        rrf_results[f"rrf-{rrf_k}"] = evaluate(
            retriever,
            questions
        )

    print("\n--- B3: RRF K ---")

    print(table(
        rrf_results,
        cols=(
            "hit_rate@1",
            "recall@5",
            "mrr",
            "ndcg@10"
        )
    ))

    # B4 - try different dense/BM25 weights
    weight_configs = {
        "1:1": [1.0, 1.0],
        "2:1": [2.0, 1.0],
        "3:1": [3.0, 1.0],
        "1:2": [1.0, 2.0]
    }

    weight_results = {}

    for name, weights in weight_configs.items():
        retriever = HybridRetriever(
            [dense, bm25],
            rrf_k=10,
            weights=weights
        )

        weight_results[name] = evaluate(
            retriever,
            questions
        )

    print("\n--- B4: RRF WEIGHTS ---")

    print(table(
        weight_results,
        cols=(
            "hit_rate@1",
            "recall@5",
            "mrr",
            "ndcg@10"
        )
    ))

def sweep_rerank() -> None:
    """Part C - compare dense retrieval with two rerankers."""

    documents = load_corpus()
    questions = load_questions()

    chunks = build_chunks(
        documents,
        strategy="markdown",
        size=400
    )

    dense = DenseRetriever(chunks)

    # Baseline
    baseline = evaluate(
        dense,
        questions,
        k=30,
        final_k=5
    )

    # C1 - cross-encoder
    cross_encoder = CrossEncoderReranker()

    cross_metrics = evaluate(
        dense,
        questions,
        k=30,
        reranker=cross_encoder,
        final_k=5
    )

    print("\n--- C1: CROSS-ENCODER ---")

    print(table(
        {
            "dense-only": baseline,
            "cross-encoder": cross_metrics
        },
        cols=(
            "hit_rate@1",
            "recall@5",
            "mrr",
            "ndcg@5",
            "latency_p95_ms"
        )
    ))

    # C2 - LLM reranker
    llm_reranker = LLMReranker()

    llm_metrics = evaluate(
        dense,
        questions,
        k=30,
        reranker=llm_reranker,
        final_k=5
    )

    print("\n--- C2: LLM RERANKER ---")

    print(table(
        {
            "dense-only": baseline,
            "cross-encoder": cross_metrics,
            "llm-reranker": llm_metrics
        },
        cols=(
            "hit_rate@1",
            "recall@5",
            "mrr",
            "ndcg@5",
            "latency_p95_ms"
        )
    ))

    # C3 - same numbers in a smaller decision table
    print("\n--- C3: DECISION TABLE ---")

    print(table(
        {
            "dense-only": baseline,
            "cross-encoder": cross_metrics,
            "llm-reranker": llm_metrics
        },
        cols=(
            "ndcg@5",
            "hit_rate@1",
            "latency_p95_ms"
        )
    ))

    # C4 - find queries made worse by reranking
    baseline_mrr = baseline["_per_question_mrr"]
    cross_mrr = cross_metrics["_per_question_mrr"]
    llm_mrr = llm_metrics["_per_question_mrr"]

    cross_drops = []
    llm_drops = []

    for qid in baseline_mrr:

        cross_drop = baseline_mrr[qid] - cross_mrr[qid]
        llm_drop = baseline_mrr[qid] - llm_mrr[qid]

        if cross_drop > 0:
            cross_drops.append((qid, cross_drop))

        if llm_drop > 0:
            llm_drops.append((qid, llm_drop))

    cross_drops.sort(key=lambda x: x[1], reverse=True)
    llm_drops.sort(key=lambda x: x[1], reverse=True)

    print("\n--- C4: CROSS-ENCODER FAILURES ---")

    for qid, drop in cross_drops[:5]:
        question = next(
            q for q in questions
            if q["id"] == qid
        )

        print(
            qid,
            "| baseline:", baseline_mrr[qid],
            "| reranked:", cross_mrr[qid],
            "| drop:", round(drop, 4),
            "|", question["question"]
        )

    print("\n--- C4: LLM RERANKER FAILURES ---")

    for qid, drop in llm_drops[:5]:
        question = next(
            q for q in questions
            if q["id"] == qid
        )

        print(
            qid,
            "| baseline:", baseline_mrr[qid],
            "| reranked:", llm_mrr[qid],
            "| drop:", round(drop, 4),
            "|", question["question"]
        )

def sweep_index() -> None:
    """Part D - compare exact search with HNSW and test metadata filtering."""

    documents = load_corpus()
    questions = load_questions()

    chunks = build_chunks(
        documents,
        strategy="markdown",
        size=400
    )

    # D1 - exact search vs HNSW
    dense = DenseRetriever(chunks)
    chroma = ChromaRetriever(chunks, reset=True)

    dense_metrics = evaluate(dense, questions)
    chroma_metrics = evaluate(chroma, questions)

    print("\n--- D1: EXACT VS HNSW ---")

    print(table(
        {
            "exact-dense": dense_metrics,
            "chroma-hnsw": chroma_metrics
        },
        cols=(
            "hit_rate@1",
            "recall@5",
            "mrr",
            "ndcg@10",
            "latency_p95_ms"
        )
    ))

    # D2 - test latency as the index gets larger
    scaled_dir = ROOT / "data/corpus_scaled"

    scaled_documents = {
        p.stem: p.read_text(encoding="utf-8")
        for p in sorted(scaled_dir.glob("*.md"))
    }

    scaled_chunks = build_chunks(
        scaled_documents,
        strategy="markdown",
        size=400
    )

    sizes = [
        ("small", chunks),
        ("medium", chunks + scaled_chunks[:4000]),
        ("large", chunks + scaled_chunks)
    ]

    query = questions[0]["question"]

    print("\n--- D2: LATENCY VS CORPUS SIZE ---")

    for name, chunk_set in sizes:

        print(f"\n{name.upper()} corpus: {len(chunk_set)} chunks")

        print("Building exact index...")
        dense_test = DenseRetriever(
            chunk_set,
            show_progress=True
        )

        print("Building HNSW index...")
        chroma_test = ChromaRetriever(
            chunk_set,
            path=f".chroma_{name}",
            collection=f"d2_{name}",
            reset=True
        )

        # Warm-up
        dense_test.search(query, k=10)
        chroma_test.search(query, k=10)

        start = time.perf_counter()

        for _ in range(20):
            dense_test.search(query, k=10)

        dense_ms = (
            (time.perf_counter() - start)
            / 20
            * 1000
        )

        start = time.perf_counter()

        for _ in range(20):
            chroma_test.search(query, k=10)

        chroma_ms = (
            (time.perf_counter() - start)
            / 20
            * 1000
        )

        print(
            "Exact:",
            round(dense_ms, 3),
            "ms | HNSW:",
            round(chroma_ms, 3),
            "ms"
        )

    # D3 - filter archived documents using metadata
    for chunk in chunks:
        if "ARCHIVED" in chunk.doc_id:
            chunk.meta["status"] = "archived"
        else:
            chunk.meta["status"] = "current"

    filtered_chroma = ChromaRetriever(
        chunks,
        path=".chroma_d3",
        collection="d3_metadata",
        reset=True
    )

    trap_questions = [
        q for q in questions
        if q["id"] in ["Q29", "Q30", "Q31"]
    ]

    before_correct = 0
    after_correct = 0

    print("\n--- D3: METADATA FILTERING ---")

    for question in trap_questions:

        before = filtered_chroma.search(
            question["question"],
            k=5
        )

        after = filtered_chroma.search(
            question["question"],
            k=5,
            where={"status": "current"}
        )

        before_doc = before[0].doc_id
        after_doc = after[0].doc_id

        if before_doc in question["relevant_docs"]:
            before_correct += 1

        if after_doc in question["relevant_docs"]:
            after_correct += 1

        print(
            question["id"],
            "| before:", before_doc,
            "| after:", after_doc
        )

    before_hit_rate = before_correct / len(trap_questions)
    after_hit_rate = after_correct / len(trap_questions)

    print(
        "\nHit@1:",
        round(before_hit_rate, 4),
        "->",
        round(after_hit_rate, 4)
    )

    
SWEEPS = {
    "chunking": sweep_chunking,
    "retrieval": sweep_retrieval,
    "rerank": sweep_rerank,
    "index": sweep_index,
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", action="store_true")
    ap.add_argument("--sweep", choices=list(SWEEPS))
    args = ap.parse_args()
    if args.baseline or not args.sweep:
        sweep_baseline()
    if args.sweep:
        SWEEPS[args.sweep]()


if __name__ == "__main__":
    main()
