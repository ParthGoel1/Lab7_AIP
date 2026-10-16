# Aurora Insurance Policy Assistant

**AI in Practice — Lab 7 | End-to-end RAG service**

This project turns the retrieval, RAG, evaluation and tool-calling work from Labs 3–6 into a usable insurance-policy assistant. A user can ask about Aurora plans, benefits, exclusions or claim procedures; the service retrieves relevant policy text and returns an answer with numbered citations and source excerpts. When the documents do not support an answer, the assistant is designed to acknowledge the gap instead of inventing a policy rule.

The project includes a FastAPI backend, a Streamlit interface, caching, server-sent-event streaming, request tracing, an operations dashboard and a golden-set regression gate in GitHub Actions. It is a **course demonstration using sample policy/customer data**, not a claims adjudication or payment system.

## Quick start (from a clean clone)

**Requirements:** Python **3.11–3.14**, Git, internet access for the first dependency installation and live requests, and a Gemini API key for online use. Run commands from the **repository root**.

```bash
git clone https://github.com/ParthGoel1/Lab7_AIP.git
cd Lab7_AIP
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Create a `.env` file in the root (it is Git-ignored):

```dotenv
AIP_PROFILE=gemini
GEMINI_API_KEY=YOUR_GEMINI_API_KEY
AIP_BUDGET_USD=2
```

Start the API:

```bash
python -m uvicorn labs.lab7.service:app --host 127.0.0.1 --port 8000
```

Open **http://127.0.0.1:8000/docs** for the interactive API reference, or verify that the service is up in another terminal:

```bash
curl -s http://127.0.0.1:8000/health | python -m json.tool
```

`/health` reports service status, uptime, index size, active MAIN model and cache statistics. First startup may take longer while the retrieval index is initialized. The initial dependency installation can also exceed five minutes on a fresh machine, depending on network speed.

## Use the assistant

**Grounded RAG question:**

```bash
curl -s http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"How long do I have to file a reimbursement claim?","mode":"rag","top_k":5}' \
  | python -m json.tool
```

The JSON includes `answer`, `refused`, `citations` (with document IDs and excerpts), `sources`, `latency_ms`, `cost_usd`, `cached` and `trace_id`. Ask the same question a second time to observe the exact-response cache (`cached: true`).

**Guarded tools mode:**

```bash
curl -s http://127.0.0.1:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"Give me the details for policy AUR-1234567","mode":"tools"}' \
  | python -m json.tool
```

Tools mode uses the Lab 6 agent with an allowlisted tool guard. The policy ID above refers to **synthetic customer data** bundled with the course.

**Streaming (RAG only):**

```bash
curl -N http://127.0.0.1:8000/ask/stream \
  -H 'Content-Type: application/json' \
  -d '{"question":"What is the room-rent limit on the Silver plan?","mode":"rag"}'
```

This endpoint emits `token` events followed by `validation` and `done` events (including latency/TTFT information). Streaming uses the **SMALL** model tier; the standard `/ask` RAG path uses **MAIN**. Because validation occurs after tokens are sent, a streaming client must pay attention to the final validation event.

## User interface and monitoring

Keep the API running. In **separate terminals**, activate `.venv` and run:

```bash
# End-user interface: expandable citation/source excerpts
python -m streamlit run labs/lab7/ui.py --server.port 8501

# Operations dashboard: traces, cost, cache, errors and stage latency
python -m streamlit run labs/lab7/dashboard.py --server.port 8502
```

Open **http://localhost:8501** for the assistant and **http://localhost:8502** for the dashboard. The dashboard is populated from trace files, so send a few `/ask` requests first. You can also inspect JSON metrics directly:

```bash
curl -s http://127.0.0.1:8000/metrics | python -m json.tool
```

Metrics include request counts, cost per query, cache hits by type, p50/p95/p99 latency, errors and tool-call counts. The dashboard flags request p95 above the 6,000 ms SLO and suggests inspecting the slowest traces.

## Run the regression gate (no API key required)

The repository includes a committed SQLite model-response cache in `.aip_cache/calls.sqlite3`. To replay the golden-set evaluation **offline**, from the repository root run:

```bash
AIP_OFFLINE=1 AIP_PROFILE=gemini \
  python labs/lab7/gate.py --config labs/lab7/thresholds.yml
```

It evaluates **45 golden questions**, checking correctness, faithfulness, citation validity, refusal precision/recall, retrieval hit rate, evaluation cost and latency against `labs/lab7/thresholds.yml`. It exits with a non-zero status if any threshold fails. The same command runs on every push and pull request in `.github/workflows/eval.yml`.

**Note:** Offline replay uses cached model calls. Its near-zero reported cost and very low latency are **not live API performance measurements**. The live service requires a key and may incur provider costs. Changes to prompts or model settings can also cause offline cache misses until the cache is repopulated.

## Implementation at a glance

| Component | File / behavior |
|---|---|
| API and caching | `labs/lab7/service.py` — FastAPI, exact-response cache, semantic cache (cosine threshold **0.92**) |
| Retrieval | `labs/lab3/search.py` — markdown chunking, dense retrieval, BM25/hybrid fallback for identifier-like questions |
| Grounded generation | `labs/lab4/rag.py` — **Gemini 3.7 Flash (MAIN), low reasoning effort, 1,024-token limit**, answer/citation validation |
| Guarded agent | `labs/lab6/agent.py` — tool orchestration plus allowlisted actions |
| User-facing UI | `labs/lab7/ui.py` — answers and expandable sources |
| Monitoring | `labs/lab7/dashboard.py`, `/metrics`, `.aip_traces/` |
| Regression evaluation | `labs/lab7/gate.py`, `labs/lab7/thresholds.yml`, GitHub Actions |

`aip/` is the provided course toolkit. The project builds on its caching, tracing, model and retrieval interfaces rather than modifying them.

## Evaluation snapshot

The final low-reasoning configuration passed the offline regression gate locally and in GitHub Actions:

| Metric | Result |
|---|---:|
| Normalized correctness (40 answerable questions) | **0.8375** (67/80 points) |
| Faithfulness (45 questions) | **0.9333** |
| Citation validity | **1.0000** |
| Refusal recall | **5/5** |
| Refusal precision | **5/7** |
| Gate hit@5 (answerable questions) | **40/40** |
| Mean live cost per uncached query (first 10-question sample) | **$0.001015** |
| Live uncached p95, first 10-question sample | **6,101.6 ms** |
| Live uncached p95, second 10-question sample | **7,378.2 ms** |

The uncached latency target is **6,000 ms p95**, which was **not consistently met**. Traces show model generation dominates the slow requests; reducing reasoning effort from medium to low improved the initial observed p95 while slightly reducing golden-set correctness. Some retrieval results also included archived policy documents, which the assistant generally distinguished but which should be deprioritized for current-policy questions. These benchmarks are small exploratory samples, not production latency guarantees.

For the methodology, stage-by-stage latency, costs, failure analysis, trade-offs and next steps, see **[EVALUATION_REPORT.md](EVALUATION_REPORT.md)**.

## Troubleshooting

| Symptom | What to check |
|---|---|
| `ModuleNotFoundError` | Activate `.venv`; run `python -m pip install -r requirements.txt` from the root |
| Model or embedding request fails | Check `.env`, `GEMINI_API_KEY`, connectivity, account quota and `AIP_PROFILE=gemini` |
| HTTP 422 | Check that `question` is present (3–1,000 characters); `mode` is `rag` or `tools` |
| HTTP 429 | Per-process `AIP_BUDGET_USD` limit or provider rate limiting; inspect logs |
| HTTP 503 | Temporary upstream/model problem; retry after the response's `Retry-After` interval |
| `/metrics` or dashboard has no requests | Send `/ask` queries first; look for trace files under `.aip_traces/` |
| Offline gate reports `CacheMiss` | The model/prompt request differs from the committed cache; run online to populate it before offline replay |

**Safety boundary:** this is a policy-information prototype. The final policy wording, effective date, coverage decisions and any actual financial actions require human verification.
