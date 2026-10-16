#!/usr/bin/env python3
"""Reproduce the saved Lab 7 quality and live-benchmark numbers (stdlib only)."""
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"


def percentile_linear(values, p):
    """numpy-style default linear-interpolation percentile."""
    a = sorted(values)
    i = (len(a) - 1) * p / 100
    lo = math.floor(i)
    hi = math.ceil(i)
    return a[lo] + (a[hi] - a[lo]) * (i - lo)


def show_quality(filename, label):
    rows = json.loads((REPORTS / filename).read_text(encoding="utf-8"))
    answerable = [r for r in rows if not r["unanswerable"]]
    unanswerable = [r for r in rows if r["unanswerable"]]
    refused = [r for r in rows if r["refused"]]
    hit40 = sum(bool(set(r["retrieved"][:5]) & set(r["relevant"])) for r in answerable)
    with_relevant = [r for r in rows if r["relevant"]]
    hit42 = sum(bool(set(r["retrieved"][:5]) & set(r["relevant"])) for r in with_relevant)
    archive_ids = [r["id"] for r in rows if any("archiv" in d.lower() for d in r["retrieved"])]
    print(f"\n{label}: {len(rows)} golden cases; {len(answerable)} answerable")
    total_score = sum(r["correctness"] for r in answerable)
    print(f"  correctness = {total_score}/{2*len(answerable)} = {total_score/(2*len(answerable)):.4f}")
    print("  full/partial/incorrect = " + "/".join(str(sum(r["correctness"] == k for r in answerable)) for k in (2, 1, 0)))
    print(f"  faithfulness = {sum(r['faithfulness'] for r in rows)/len(rows):.4f}")
    print(f"  citation validity = {sum(r['citations_valid'] for r in rows)}/{len(rows)}")
    print(f"  refusal precision = {sum(r['unanswerable'] for r in refused)}/{len(refused)}")
    print(f"  refusal recall = {sum(r['refused'] for r in unanswerable)}/{len(unanswerable)}")
    print(f"  hit@5 = {hit40}/{len(answerable)} answerable; {hit42}/{len(with_relevant)} with relevant docs")
    print("  archive retrieval IDs =", ", ".join(archive_ids) or "none")
    print("  false refusals =", ", ".join(r["id"] for r in answerable if r["refused"]) or "none")


show_quality("lab4.json", "Medium-reasoning baseline")
show_quality("lab7_low_reasoning.json", "Final low reasoning")

batches = defaultdict(list)
with (REPORTS / "lab7_live_requests.csv").open(newline="", encoding="utf-8") as fh:
    for row in csv.DictReader(fh):
        if row["cached"].lower() == "false":
            batches[row["sample"]].append(row)

for label, rows in sorted(batches.items()):
    durations = [float(r["latency_ms"]) for r in rows]
    costs = [float(r["cost_usd"]) for r in rows]
    print(f"\nLive sample {label}: n={len(rows)}, all uncached")
    print(f"  p50={percentile_linear(durations, 50):.1f} ms, p95={percentile_linear(durations, 95):.1f} ms")
    print(f"  >6000 ms = {sum(x > 6000 for x in durations)}/{len(rows)}")
    print(f"  average cost from rounded API values = ${sum(costs)/len(costs):.6f}")
