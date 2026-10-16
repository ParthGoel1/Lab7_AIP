# Lab 7 Concept Note — From AI Prototype to Working System

**Project:** Aurora Insurance Policy Assistant  
**Course:** AI in Practice, Lab 7  
**Focus:** RAG and tool integration, API, caching, streaming, safeguards, observability, and regression testing

## 1. Problem

Earlier labs gave us individual pieces of an AI assistant: retrieval, grounded generation, evaluation, and tool use. Lab 7 asked a different question: **how do we turn those pieces into a working application that is usable, measurable, reasonably fast, cost-conscious, and safe?** A correct answer in a notebook is not sufficient if an API fails, repeated questions waste tokens, a tool performs an unauthorized action, or a new optimization reduces answer quality. The objective was to integrate the components while defining how the system behaves under normal use, mistakes, and failures.

## 2. Architecture and Request Flow

```text
User / Streamlit UI
        |
        v
FastAPI: input validation + request routing
        |
        +---- RAG mode -----------------------+---- Tools mode ---------------------+
        |                                     |                                     |
  Exact/semantic cache                  LLM selects a tool                       |
        |                                     |                                     |
  Hybrid retrieval                   ToolGuard + Pydantic schemas                |
        |                                     |                                     |
  Grounded LLM generation             Authorized tool execution                  |
        |                                     |                                     |
  Citation checks / repair / refusal   Structured output + filtering             |
        +-------------------------------------+-------------------------------------+
                                              |
                                  Answer + metadata to user
                                              |
                        Metrics, traces, cost tracking, regression tests
```

**API and UI.** Streamlit provides the interface; FastAPI exposes `POST /ask` for complete RAG/tool answers, `POST /ask/stream` for RAG streaming, `GET /health` for service status, and `GET /metrics` for operations data. Pydantic validates inputs before expensive model calls.

**RAG route.** The service checks a cache, retrieves supporting policy chunks through hybrid search, generates an evidence-based answer, and checks citation structure. If validation fails, it can attempt repair or return a refusal. Citation validity confirms that references point to supplied excerpts; it does **not** independently prove every factual claim.

**Tools route.** The model requests functions such as `get_policy_details`, `compute_premium`, or `search_policy`. Python performs data lookup and deterministic calculation; `ToolGuard` enforces permitted tools and argument checks. Refund actions are not authorized by the Lab 7 allowlist. The tool assistant also uses structured-output Layer 3, with the smaller model tier for the final conversion step.

## 3. Core Concepts and Engineering Decisions

| Concept | What we implemented / learned |
|---|---|
| **Exact caching** | Reuse responses to identical requests, avoiding repeated generation costs. |
| **Semantic caching** | Reuse answers to sufficiently similar questions, while checking plan names and numeric details to reduce wrong cross-plan matches. |
| **Streaming (SSE)** | Send `token`, `validation`, and `done` events so users receive text before the complete response. **TTFT** measures waiting time for the first token; total latency measures completion time. |
| **Guardrails** | Untrusted-context delimiters, injection detection, structured schemas, tool allowlists, and output filtering. Important controls operate in code rather than relying solely on prompts. |
| **Observability** | Track request counts, tool calls, cache activity, errors, latency, cost, and trace identifiers to make failures diagnosable. |
| **Performance and budgets** | Measure p95 rather than only average latency; use model tiers and caching to manage speed/cost. Application spend limits are distinct from provider quotas. |
| **Regression gates** | Re-run quality checks after changes so an optimization does not silently damage correctness, faithfulness, citations, or refusals. |

### Evaluation results and qualifications

The final **functional** test suite passed **30/30 checks**, including both streaming cases, cache behavior, permitted tool execution, blocked refund attempts, and API validation. The final streaming run measured TTFT at approximately **1.44 s** for a supported policy and **1.06 s** for a refusal. Thirty repeated exact-cache requests achieved about **2.2 ms p95**.

The separately run **offline regression gate passed all eight thresholds**. Its key quality scores were correctness **0.8375**, faithfulness **0.9333**, citation validity **1.000**, refusal recall **1.000**, refusal precision **0.7143**, and retrieval Hit@5 **1.000**. However, the offline gate uses cached/replayed calls: its near-zero latency and cost do **not** measure uncached serving performance. Earlier live samples recorded approximately **6.3 s RAG p95** and **7.65 s tools p95**; later tool measurements improved, but model-level caching complicates a direct comparison.

## 4. How the Labs Connected

- **Lab 3:** Chunking, hybrid retrieval, and finding evidence.
- **Lab 4:** Grounded answers, citations, and refusal behavior.
- **Lab 5:** Evaluations, failure analysis, and quality baselines.
- **Lab 6:** Tool calling, argument validation, and security layers.
- **Lab 7:** Packaging these capabilities as a service with UI, streaming, caches, metrics, budgets, and automated checks.

## 5. Personal Takeaways

**An AI model is only one part of an AI product.** I initially focused on whether the answer looked correct. Here, I had to think about input validation, what happens during retrieval, whether a tool is authorized, response time, and what evidence exists when something fails.

**Optimization is a system-level trade-off.** Switching Layer 3 structured conversion to a smaller model was a useful experiment: relatively simple formatting may not need the same model as tool selection. But speed gains must be evaluated alongside correctness, and cache hits must not be confused with fresh model inference.

**The testing system can have bugs of its own.** Our automated streaming checks failed even though `curl` showed correct SSE events. The fault was eventually traced to the test harness's event parsing. This taught me to isolate the server, client, provider, and evaluator before changing code.

**Safety is stronger when enforced by software boundaries.** Tool allowlists and Pydantic constraints make certain actions impossible through the exposed interface, even if an LLM requests them. This matters especially in financial or insurance workflows.

**A green test suite is meaningful but limited.** The 30 passing functional tests and eight regression gates support readiness for a course demo. Production readiness would still require broader uncached load tests, concurrency checks, authentication, auditability, data-access controls, and evaluation against real operating conditions.

## 6. Conclusion

Lab 7 was where retrieval, generation, evaluation, and agentic tool use became **one integrated AI service**. The core lesson was that moving from prototype to application means designing not just for good answers, but also for **measurability, reliability, security, cost, and the ability to debug and improve the system**.
