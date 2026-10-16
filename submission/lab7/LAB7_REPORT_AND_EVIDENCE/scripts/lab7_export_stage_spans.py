#!/usr/bin/env python3
"""Export the stage spans of one Lab 7 run, without question/answer text."""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

from aip import tracing

STAGES = ("embed.batch", "retrieve.dense", "llm.call", "rag.validate", "http.ask")


def percentile(values: list[float], p: float) -> float:
    values = sorted(values)
    x = (len(values) - 1) * p / 100
    i, j = math.floor(x), math.ceil(x)
    return values[i] + (values[j] - values[i]) * (x - i)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default="20261008-154047")
    parser.add_argument("--output", default="reports/lab7_stage_spans.csv")
    args = parser.parse_args()

    rows = [r for r in tracing.read_traces(args.run_id)
            if r.get("name") in STAGES and r.get("duration_ms") is not None]
    if not rows:
        raise SystemExit(f"No matching stage spans found for run {args.run_id!r}")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=("run_id", "stage", "duration_ms", "cached"))
        writer.writeheader()
        for row in rows:
            writer.writerow({
                "run_id": args.run_id,
                "stage": row["name"],
                "duration_ms": row["duration_ms"],
                "cached": row.get("cached", ""),
            })

    print(f"Exported {len(rows)} sanitized stage spans to {out}")
    print(f"{'stage':<20} {'count':>6} {'p50':>10} {'p95':>10}")
    for stage in STAGES:
        values = [float(r["duration_ms"]) for r in rows if r["name"] == stage]
        if values:
            print(f"{stage:<20} {len(values):>6} {percentile(values, 50):>10.1f} {percentile(values, 95):>10.1f}")


if __name__ == "__main__":
    main()
