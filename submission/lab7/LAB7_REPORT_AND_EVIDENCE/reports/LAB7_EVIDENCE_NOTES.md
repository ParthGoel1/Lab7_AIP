# Lab 7 evaluation evidence — provenance and reproduction

These files support `EVALUATION_REPORT.md` (root):

- `reports/lab4.json` — the earlier medium-reasoning golden-set baseline; included in the bundle for self-contained verification (and already tracked in the repository).
- `reports/lab7_low_reasoning.json` — 45 saved per-question rows from the final low-reasoning evaluation.
- `reports/lab7_live_requests.csv` — 20 uncached API requests copied from terminal output (two 10-question runs). The first run ID was not saved in the provided output; the second was `20261008-154047`. The second batch did not print individual refusal flags, so they are left blank. The API-response p50/p95 recomputed from the rounded individual values differ by ~1 ms from the trace-span p50/p95 shown in the report; the latter are used for the stage table. `cost_usd` values are rounded as displayed by the service; the reported average costs come from benchmark output.
- `reports/lab7_latency_summary.csv` — summary p50/p95 from separately executed model/token experiments. Those earlier raw response logs are not included here; these are recorded measurements, not independently replayable numbers.
- `reports/lab7_stage_latency.csv` — recorded p50/p95 by trace name for the second low-reasoning sample. Embedding spans count 21 across 10 requests; these are mixed cached/uncached individual spans. No raw trace JSONL is included (local `.aip_traces/` is ignored by Git).
- `scripts/lab7_evidence_summary.py` — recomputes quality and the two live sample latency/cost figures using the committed JSON/CSV evidence.
- `scripts/lab7_export_stage_spans.py` — optional one-command export of the actual stage spans from the local trace file, if that trace is still present. This enables independent recomputation of stage percentiles without committing the full trace (which could include question text).

## Reproduce the regression gate (offline; no key needed)

```bash
AIP_OFFLINE=1 AIP_PROFILE=gemini python labs/lab7/gate.py --config labs/lab7/thresholds.yml
```

The committed `.aip_cache/calls.sqlite3` supplies cached model/judge calls, so gate latency and cost represent cache replay, **not uncached API service performance**. The gate's retrieval hit@5 uses 40 answerable questions. The distinct 41/42 result includes all cases with relevant-document labels.

## Recompute the saved experiment metrics

```bash
python scripts/lab7_evidence_summary.py
```

This script needs only Python 3.11+ and the files in this bundle including `reports/lab4.json`. To regenerate *live* latency measurements rather than recomputing saved data, run the API online with valid provider credentials, use new uncached questions, and keep model/prompt/reasoning parameters unchanged. Latency is provider- and workload-dependent.

## Optional: make the stage breakdown fully recomputable

If you still have the trace recorded locally for `20261008-154047`, run from the repository root:

```bash
python scripts/lab7_export_stage_spans.py --run-id 20261008-154047
```

This creates `reports/lab7_stage_spans.csv` with only stage names, durations, run ID and cache flag; it excludes prompts/answers. Commit that CSV with the other evidence files. If the local trace is gone, do not invent spans: keep the measured aggregate `lab7_stage_latency.csv` and say the underlying trace was not committed.
