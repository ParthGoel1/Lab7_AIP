#!/usr/bin/env python3
"""Lab 7 — the observability dashboard, read from local traces.

    streamlit run labs/lab7/dashboard.py

`aip.tracing` writes one JSONL file per run to .aip_traces/. This page reads
them back. It is a teaching-scale stand-in for Langfuse / LangSmith / Phoenix;
the concept -- structured spans with a run id and a parent id -- is identical.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
import streamlit as st
import statistics
import time

from aip.retrieval import format_context
from labs.lab3.search import load_questions
from labs.lab4.evaluate import (
    build_retriever,
    judge_correctness,
    judge_faithfulness,
)
from labs.lab4.rag import answer_question


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.config import settings  # noqa: E402


# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Aurora Assistant — Ops",
    layout="wide",
)

st.title("Aurora Policy Assistant — operations")


# ---------------------------------------------------------------------------
# Load trace files
# ---------------------------------------------------------------------------

runs = sorted(
    settings.trace_dir.glob("*.jsonl"),
    reverse=True,
)

if not runs:
    st.info(
        f"No traces yet in {settings.trace_dir}. "
        "Run some queries first."
    )
    st.stop()


# Default to all runs created today.
today = pd.Timestamp.now().strftime("%Y%m%d")

today_runs = [
    p.stem
    for p in runs
    if p.stem.startswith(today)
]


st.sidebar.header("Trace selection")

mode = st.sidebar.radio(
    "Data",
    [
        "Today",
        "Latest run",
        "Custom",
    ],
)

if mode == "Today":
    chosen = today_runs

elif mode == "Latest run":
    chosen = [runs[0].stem]

else:
    chosen = st.sidebar.multiselect(
        "Select runs",
        [p.stem for p in runs],
        default=today_runs,
    )

rows = [
    json.loads(line)
    for path in runs
    if path.stem in chosen
    for line in path.open(encoding="utf-8")
    if line.strip()
]


if not rows:
    st.info("No trace rows found for the selected runs.")
    st.stop()


# ---------------------------------------------------------------------------
# Build dataframe
# ---------------------------------------------------------------------------

df = pd.DataFrame(rows)

df["ts"] = pd.to_datetime(
    df["ts"],
    unit="s",
)


# ---------------------------------------------------------------------------
# Separate important span types
# ---------------------------------------------------------------------------

requests = df[
    df["name"] == "http.ask"
].copy()

llm = df[
    df["name"] == "llm.call"
].copy()


# ---------------------------------------------------------------------------
# Main summary metrics
# ---------------------------------------------------------------------------

cache_hit_rate = 0.0

if len(requests):

    cache_hit_rate = (
        requests
        .get(
            "cached",
            pd.Series(
                False,
                index=requests.index,
            ),
        )
        .fillna(False)
        .mean()
    )


request_errors = (
    (requests["status"] == "error").sum()
    if len(requests)
    else 0
)


error_rate = (
    request_errors / len(requests)
    if len(requests)
    else 0.0
)


total_cost = (
    df
    .get(
        "cost_usd",
        pd.Series([0]),
    )
    .fillna(0)
    .sum()
)


c = st.columns(5)

c[0].metric(
    "requests",
    len(requests),
)

c[1].metric(
    "total cost",
    f"${total_cost:.4f}",
)

c[2].metric(
    "model calls",
    len(llm),
)

c[3].metric(
    "cache hit rate",
    f"{cache_hit_rate:.1%}",
)

c[4].metric(
    "error rate",
    f"{error_rate:.1%}",
)


# ---------------------------------------------------------------------------
# C3 — Latency by stage
# ---------------------------------------------------------------------------

st.subheader("Latency by stage")

stage = (
    df
    .dropna(
        subset=["duration_ms"]
    )
    .groupby("name")["duration_ms"]
    .agg(
        n="count",
        p50="median",
        p95=lambda s: s.quantile(0.95),
        total="sum",
    )
    .sort_values(
        "total",
        ascending=False,
    )
)

st.dataframe(
    stage,
    use_container_width=True,
)


# ---------------------------------------------------------------------------
# C3 — Latency over time
# ---------------------------------------------------------------------------

st.subheader("Latency over time")

latency_df = df[
    df["name"].isin(
        [
            "http.ask",
            "embed.batch",
            "retrieve.dense",
            "retrieve.hybrid",
            "rerank.cross_encoder",
            "rerank.llm",
            "llm.call",
            "rag.validate",
            "tool.call",
        ]
    )
][
    [
        "ts",
        "name",
        "duration_ms",
    ]
].copy()


if len(latency_df):

    latency_pivot = (
        latency_df
        .pivot_table(
            index="ts",
            columns="name",
            values="duration_ms",
            aggfunc="mean",
        )
        .sort_index()
    )

    st.line_chart(
        latency_pivot
    )

else:

    st.info(
        "No latency data available yet."
    )


# ---------------------------------------------------------------------------
# C3 — Cumulative cost
# ---------------------------------------------------------------------------

st.subheader("Cumulative cost")

if "cost_usd" in df.columns:

    cost_df = (
        df[
            [
                "ts",
                "cost_usd",
            ]
        ]
        .copy()
        .sort_values("ts")
    )

    cost_df["cost_usd"] = (
        cost_df["cost_usd"]
        .fillna(0)
    )

    cost_df["cumulative_cost_usd"] = (
        cost_df["cost_usd"]
        .cumsum()
    )

    st.line_chart(
        cost_df.set_index(
            "ts"
        )[
            "cumulative_cost_usd"
        ]
    )

else:

    st.info(
        "No cost data available yet."
    )


# ---------------------------------------------------------------------------
# C3 — Request health
# ---------------------------------------------------------------------------

st.subheader("Request health")

if len(requests):

    h1, h2, h3 = st.columns(3)

    h1.metric(
        "requests",
        len(requests),
    )

    h2.metric(
        "error rate",
        f"{error_rate:.1%}",
    )

    h3.metric(
        "cache hit rate",
        f"{cache_hit_rate:.1%}",
    )

else:

    st.info(
        "No request traces available yet."
    )


# ---------------------------------------------------------------------------
# C3 — Cache breakdown
# ---------------------------------------------------------------------------

st.subheader("Cache breakdown")

if len(requests):

    if "cache_type" in requests.columns:

        cache_counts = (
            requests["cache_type"]
            .fillna("unknown")
            .value_counts()
            .rename_axis("cache_type")
            .reset_index(name="count")
        )

        st.dataframe(
            cache_counts,
            use_container_width=True,
        )

    else:

        st.info(
            "No cache type information recorded."
        )

else:

    st.info(
        "No request traces available yet."
    )


# ---------------------------------------------------------------------------
# C3 — Tool-call counts
# ---------------------------------------------------------------------------

st.subheader("Tool calls")

tool_df = df[
    df["name"] == "tool.call"
].copy()


if len(tool_df):

    if "tool" in tool_df.columns:

        tool_counts = (
            tool_df["tool"]
            .fillna("unknown")
            .value_counts()
            .rename_axis("tool")
            .reset_index(name="count")
        )

        st.dataframe(
            tool_counts,
            use_container_width=True,
        )

    else:

        st.info(
            "Tool spans exist but do not contain tool names."
        )

else:

    st.info(
        "No tool calls recorded."
    )


# ---------------------------------------------------------------------------
# C3 — Errors
# ---------------------------------------------------------------------------

st.subheader("Errors")

if "status" in df.columns:

    errs = df[
        df["status"] == "error"
    ].copy()

else:

    errs = pd.DataFrame()


if len(errs):

    error_columns = [
        col
        for col in [
            "ts",
            "name",
            "error",
        ]
        if col in errs.columns
    ]

    st.dataframe(
        errs[error_columns],
        use_container_width=True,
    )

else:

    st.info(
        "No errors recorded."
    )


# ---------------------------------------------------------------------------
# C4 — Alerting
# ---------------------------------------------------------------------------
# TODO C4:
# Add one alert condition and explain what action should be taken when it fires.
#
# Candidates:
# - p95 latency above the SLO for 5 minutes
# - cost per hour above budget
# - refusal rate doubling
#
# The lab suggests refusal-rate doubling as a useful signal that retrieval
# or the underlying index may have broken.
# ---------------------------------------------------------------------------
# C4 — Alerting
# ---------------------------------------------------------------------------

st.subheader("Alert status")

if len(requests):

    request_p95 = requests["duration_ms"].quantile(0.95)

    if request_p95 > 6000:

        st.error(
            f"Latency alert: p95 request latency is "
            f"{request_p95:.0f} ms, above the 6000 ms SLO."
        )

        st.write(
            "Action: inspect the slow request traces and identify which "
            "stage is responsible. If llm.call dominates, investigate "
            "provider/model latency or move suitable requests to a faster "
            "model tier."
        )

    else:

        st.success(
            f"Latency healthy: p95 is "
            f"{request_p95:.0f} ms."
        )

else:

    st.info(
        "No request data available for alerting."
    )