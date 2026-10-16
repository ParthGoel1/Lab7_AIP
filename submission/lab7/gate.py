#!/usr/bin/env python3
"""Lab 7 — the regression gate. Exits non-zero when a threshold is breached.

    python labs/lab7/gate.py --config labs/lab7/thresholds.yml
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import yaml


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from aip.retrieval import format_context
from labs.lab3.search import load_questions
from labs.lab4.evaluate import (
    build_retriever,
    judge_correctness,
    judge_faithfulness,
)
from labs.lab4.rag import answer_question
from aip.cost import global_budget


def measure() -> dict[str, float]:
    questions = load_questions(include_unanswerable=True)

    dense_retriever, hybrid_retriever = build_retriever()

    budget = global_budget()
    cost_before = budget.spent_usd

    rows = []
    latencies = []

    for q in questions:
        start = time.perf_counter()

        a = answer_question(
            q["question"],
            dense_retriever,
            hybrid_retriever=hybrid_retriever,
        )

        latency_ms = (
            time.perf_counter() - start
        ) * 1000

        latencies.append(latency_ms)

        context = format_context(a.hits)

        unanswerable = (
            not q["relevant_docs"]
            or q["kind"] == "unanswerable"
        )

        rows.append(
            {
                "unanswerable": unanswerable,
                "refused": a.refused,
                "citations_valid": a.citations_valid,

                "faithfulness": judge_faithfulness(
                    a.text,
                    context,
                ),

                "correctness": judge_correctness(
                    q["question"],
                    a.text,
                    q["gold_answer"],
                ),

                "retrieved": [
                    h.doc_id
                    for h in a.hits
                ],

                "relevant": q["relevant_docs"],
            }
        )

    answerable = [
        r
        for r in rows
        if not r["unanswerable"]
    ]

    cost_after = budget.spent_usd

    total_cost = cost_after - cost_before

    cost_per_query_usd = (
        (total_cost/len(questions))
        if questions
        else 0.0
    )

    unanswerable_rows = [
        r
        for r in rows
        if r["unanswerable"]
    ]

    refusals = [
        r
        for r in rows
        if r["refused"]
    ]

    correctness = (
        statistics.fmean(
            r["correctness"]
            for r in answerable
        )
        / 2
    )

    faithfulness = statistics.fmean(
        r["faithfulness"]
        for r in rows
    )

    citation_validity = statistics.fmean(
        r["citations_valid"]
        for r in rows
    )

    refusal_recall = (
        sum(
            1
            for r in unanswerable_rows
            if r["refused"]
        )
        / len(unanswerable_rows)
        if unanswerable_rows
        else 0.0
    )

    refusal_precision = (
        sum(
            1
            for r in refusals
            if r["unanswerable"]
        )
        / len(refusals)
        if refusals
        else 1.0
    )

    hit_scores = []

    for r in answerable:
        retrieved_top5 = set(
            r["retrieved"][:5]
        )

        relevant = set(
            r["relevant"]
        )

        hit_scores.append(
            1.0
            if retrieved_top5 & relevant
            else 0.0
        )

    hit_rate_at_5 = statistics.fmean(
        hit_scores
    )

    p95_latency_ms = statistics.quantiles(
        latencies,
        n=100,
    )[94]

    return {
        "correctness": correctness,
        "faithfulness": faithfulness,
        "citation_validity": citation_validity,
        "refusal_recall": refusal_recall,
        "refusal_precision": refusal_precision,
        "hit_rate_at_5": hit_rate_at_5,

        # We will wire real cost in next.
        "cost_per_query_usd": cost_per_query_usd,

        "p95_latency_ms": p95_latency_ms,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="labs/lab7/thresholds.yml")
    args = ap.parse_args()

    thresholds = yaml.safe_load((ROOT / args.config).read_text(encoding="utf-8"))
    metrics = measure()

    failures = []
    width = max(len(k) for k in thresholds)
    print(f"{'metric':<{width}}  {'value':>10}  {'gate':>14}  status")
    print("-" * (width + 40))
    for name, rule in thresholds.items():
        value = metrics.get(name)
        if value is None:
            failures.append(f"{name}: not measured")
            print(f"{name:<{width}}  {'—':>10}  {'':>14}  MISSING")
            continue
        ok, gate = True, ""
        if "min" in rule:
            gate, ok = f">= {rule['min']}", value >= rule["min"]
        if "max" in rule and ok:
            gate, ok = f"<= {rule['max']}", value <= rule["max"]
        if not ok:
            failures.append(f"{name}: {value} violates {gate}")
        print(f"{name:<{width}}  {value:>10.4f}  {gate:>14}  {'ok' if ok else 'FAIL'}")

    if failures:
        print("\nGATE FAILED:")
        for f in failures:
            print("  " + f)
        return 1
    print("\nGATE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
