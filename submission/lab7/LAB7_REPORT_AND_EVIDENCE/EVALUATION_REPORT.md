# Lab 7 — Evaluation Report

**System:** Aurora Insurance Policy Assistant  
**Final setup:** Gemini 3.7 Flash (`MAIN`), `reasoning_effort="low"`, 1,024 output tokens, top five context chunks. **Evaluation:** 45 golden questions (40 answerable, five unanswerable), two ten-query live samples, and service-level experiments.

## 1. What it does

I turned the RAG pipeline from the earlier labs into an API-based policy assistant. A user can ask about Aurora plans, benefits or claims and receive an answer grounded in retrieved policy documents, with numbered citations and expandable excerpts. The FastAPI service also supports guarded policy-lookup tools, caching, server-sent-event streaming, health/metrics endpoints, tracing and a Streamlit dashboard. The goal is to explain policy wording, not make claims decisions.

## 2. How well it works — experiments and quality

I tested the components separately, rather than relying only on the final evaluation score:

| Experiment | What I observed and decided |
|---|---|
| **API and guards** | Tested grounded `/ask` responses, citations, error handling, costs and trace IDs. Synthetic policy lookups worked, while the guard blocked an unauthorized refund action. |
| **Exact caching** | A reimbursement-deadline answer fell from **7,168.8 ms / $0.002541** to **~0.5 ms / $0** when repeated. |
| **Semantic caching** | Chose cosine threshold **0.92** after testing paraphrases (~0.945–0.973 similarity) against risky cross-plan matches (~0.852–0.891). Avoiding a wrong reused answer mattered more than maximizing hits. |
| **Streaming** | Observed time-to-first-token (TTFT) of **~650, 1,427 and 2,015 ms**. Tokens are sent immediately, with a final validation event; streamed text is provisional until then. |
| **Observability and CI** | Used traces and the dashboard to isolate slow stages and monitor cost, cache hits and errors. Deliberately made the regression gate fail by tightening a threshold, restored it, and verified a green offline run and GitHub Actions build. |

| Golden-set metric | Medium (earlier) | **Low (final)** | Gate |
|---|---:|---:|---:|
| Correctness, 40 answerable | 69/80 = 0.8625 | **67/80 = 0.8375** | ≥0.75 |
| Faithfulness, 45 | 0.9778 | **0.9333** | ≥0.90 |
| Citation validity | 45/45 | **45/45** | ≥0.98 |
| Refusal recall | 5/5 | **5/5** | ≥0.80 |
| Refusal precision | 5/6 | **5/7** | ≥0.70 |
| Retrieval hit@5, 40 answerable | 40/40 | **40/40** | ≥0.85 |

**All six final quality checks passed.** The broader retrieval calculation was **41/42** cases with labeled relevant documents (a different denominator from the 40-answerable-question gate). The full per-question records are in `reports/lab4.json` and `reports/lab7_low_reasoning.json`; the latter and the live benchmark tables accompany this report. The offline gate can be rerun with `AIP_OFFLINE=1 AIP_PROFILE=gemini python labs/lab7/gate.py --config labs/lab7/thresholds.yml`. Its near-zero cost and latency reflect **cache replay**, not live service speed.

## 3. Where it falls short — quantified

The clearest operational limitation is **inconsistent uncached latency**. In the first low-reasoning sample, **1/10** requests exceeded six seconds (p95 **6.10 s**); in the next, **3/10** exceeded it (p95 **7.38 s**). Retrieval also returned `claims-timelines-2024-ARCHIVED` alongside current documents for **3/45** golden questions (Q01, Q30 and Q31). The answers generally identified the older rules, but retrieval should prioritize current versions by default.

Of the **40 answerable** questions in the final run, **29** were fully correct, **nine** partly correct and **two** unnecessarily refused (Q23 and Q44). Citation validity stayed at 45/45 and all five genuinely unanswerable cases were refused. Q05 also received different judge scores for an identical answer across runs, so small score changes need interpretation rather than automatic attribution to the model setting.

## 4. What it costs

The first ten uncached low-reasoning API queries averaged **$0.001015/query** (second sample: ~$0.001023). At the first sample's rate, that is **$1.015 per 1,000 queries**, or **about $3,705/year** at 10,000 queries/day. This is a *serving-only extrapolation* for similar uncached queries: it excludes hosting, evaluation judges, provider price changes and traffic differences. Exact-cache hits can avoid new generation cost, but I did not assume a cache-hit rate in the projection.

## 5. How fast it is — model experiments and stages

I started with `MAIN` at its default **medium** reasoning level. The smaller `SMALL` model did not reliably improve p95; reducing output to 512 tokens caused truncation and occasional repairs. Traces indicated that the model call, rather than retrieval, dominated slow requests. I therefore kept `MAIN` and 1,024 tokens but changed reasoning effort to **low**.

| Uncached configuration (10 questions per sample) | p50 | p95 |
|---|---:|---:|
| MAIN / medium / 1,024 tokens | ~4.59 s | **9.86 s** |
| SMALL / default / 1,024 tokens | 5.93 s | **10.02 s** |
| MAIN / medium / 512 tokens | 5.91 s | **14.30 s** |
| MAIN / low / 1,024 — first sample | **4.63 s** | **6.10 s** |
| MAIN / low / 1,024 — second sample | **3.61 s** | **7.38 s** |

**Stage breakdown — second low-reasoning sample:**

| Recorded stage | Count | p50 (ms) | p95 (ms) |
|---|---:|---:|---:|
| Embedding* | 21 spans | 0.0 | 950.6 |
| Dense retrieval | 10 | 1.1 | 1.5 |
| LLM generation | 10 | 2,818.1 | 6,428.0 |
| Validation | 10 | 0.0 | 0.1 |
| **Total request** | **10** | **3,609.1** | **7,378.2** |

*Embedding uses individual spans (including cache hits), not per-request sums; stage percentiles should not be added together.* Lower reasoning gave a promising initial p95 improvement, but samples used different questions, so this is **not a controlled A/B speedup**. Correctness also changed from **69/80 to 67/80**, although the final gates passed. Neither uncached p95 ≤6 s nor TTFT ≤1.5 s was consistently demonstrated; a single ~0.5 ms exact-cache hit is not a cached-p95 measurement.

## 6. What it is not safe for

I would use this system to **explain** policy information, not approve reimbursements, determine binding coverage or issue payments. A human should check the customer's actual policy, its effective date and the cited passage before acting. The guard restricts sensitive tool actions, but citations do not guarantee that the newest document was retrieved. In streaming mode, clients must also wait for the final validation event before treating text as verified.

## 7. Next three improvements, ranked by expected value

1. **Make latency predictable — highest expected value.** Benchmark at least 30 *identical* questions per configuration, inspect generation outliers and tune the serving path. **Success:** repeatably achieve uncached p95 ≤6 s; measure cached p95 and TTFT separately.
2. **Prioritize current policy versions — high expected value.** Add effective-date/version metadata and demote archived chunks unless the question asks about historical rules. **Success:** reduce archived results on current-policy golden queries from **3/45 to 0/45** without losing historical coverage.
3. **Improve completeness and evaluation stability — moderate expected value.** Add targeted regression cases for Q23/Q44, recheck judge disagreements such as Q05, and version response-cache keys with model/prompt settings. **Success:** reduce unnecessary refusals from **2/40 to 0/40** while retaining **5/5** correct refusals.
