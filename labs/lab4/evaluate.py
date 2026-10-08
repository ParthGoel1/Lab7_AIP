#!/usr/bin/env python3
"""Lab 4 evaluation. Scaffolding provided; the judges are yours.

    python labs/lab4/evaluate.py --full --save reports/lab4.json
    python labs/lab4/evaluate.py --gold-context
    python labs/lab4/evaluate.py --calibrate      # writes the hand-label sheet
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aip.chunking import markdown_chunks  # noqa: E402
from aip.cost import Budget  # noqa: E402
from aip.evals import (  # noqa: E402
    JUDGE_RUBRIC_CORRECTNESS,
    JUDGE_RUBRIC_FAITHFULNESS,
    judge_agreement,
    llm_judge,
)
from aip.retrieval import DenseRetriever, Bm25Retriever, HybridRetriever, format_context  # noqa: E402
from labs.lab3.search import load_corpus, load_questions  # noqa: E402
from labs.lab4.rag import REFUSAL, answer_question, answer_with_gold_context  # noqa: E402

GOLDEN = ROOT / "data/eval/rag_golden.jsonl"
LABEL_SHEET = ROOT / "labs/lab4/calibration_labels.jsonl"

LAB4_JUDGE_RUBRIC_FAITHFULNESS = """\
You are grading whether an ANSWER is fully supported by the provided CONTEXT.

Judge ONLY faithfulness to the context.

Rules:
- Score 1 only if every factual claim in the answer is supported by the context.
- Score 0 if any factual claim is unsupported, stronger than what the context
  states, or depends on outside knowledge.
- A citation does not make a claim supported if the cited context does not
  actually support that claim.
- A paraphrase is supported only if it preserves the meaning and strength of
  the source.
- If the answer correctly answers one supported part of the question and
  refuses another unsupported part, it may still score 1.
- A full refusal counts as supported only when the context genuinely does not
  contain enough information to answer.
- Do NOT judge helpfulness, style, correctness against a reference answer,
  or whether the answer is true in the real world.

CONTEXT:
{context}

ANSWER:
{answer}

Reply as JSON:
{{"score": 0 or 1,
  "unsupported_claims": [],
  "reason": "one sentence"}}
"""

LAB4_JUDGE_RUBRIC_CORRECTNESS = """\
You are grading whether a CANDIDATE answer substantively matches the REFERENCE
answer for the same QUESTION.

Judge ONLY substantive correctness.

Scoring:
- Score 2: The candidate matches the essential substantive content of the
  reference. Exact wording is not required.
- Score 1: The candidate is partially correct but omits an important part,
  contains a materially incorrect addition, or gives only part of the reference.
- Score 0: The candidate is wrong, contradicts the reference, or refuses when
  the reference provides an answer.

Refusal rules:
- If the reference indicates that the question cannot be answered from the
  available information, a correct refusal scores 2.
- If the reference gives a partial answer and refuses only the unsupported part,
  the candidate must do the same substantively to score 2.
- Refusing the entire question when the reference contains answerable
  information scores 0.
- Do NOT judge citation formatting, writing style, verbosity, or whether the
  candidate is supported by the retrieved context.

QUESTION:
{question}

REFERENCE:
{reference}

CANDIDATE:
{candidate}

Reply as JSON:
{{"score": 0 or 1 or 2,
  "reason": "one sentence"}}
"""


def build_retriever():
    corpus = load_corpus()

    chunks = [
        c
        for doc_id, text in corpus.items()
        for c in markdown_chunks(text, doc_id, size=400)
    ]

    dense = DenseRetriever(chunks)
    bm25 = Bm25Retriever(chunks)
    hybrid = HybridRetriever(
        [dense, bm25],
        rrf_k=10,
        weights=[2.0, 1.0]
    )

    return dense, hybrid


# ---------------------------------------------------------------------------
# judges (yours)
# ---------------------------------------------------------------------------
def judge_faithfulness(answer_text: str, context: str) -> int:
    """TODO D1: improve JUDGE_RUBRIC_FAITHFULNESS and return 0 or 1.

    Things the shipped rubric does not yet handle well:
      - a partial refusal (answers part, refuses part)
      - an answer that cites correctly but paraphrases into a stronger claim
      - an answer that is right about the world and wrong about the context
    """
    verdict = llm_judge(LAB4_JUDGE_RUBRIC_FAITHFULNESS.format(
        context=context[:8000], answer=answer_text), tier="LARGE")
    return int(verdict.get("score", 0))


def judge_correctness(question: str, candidate: str, reference: str) -> int:
    """TODO D1: returns 0, 1 or 2. Handle refusal cases explicitly --
    a correct refusal on an unanswerable question must score 2, and the
    shipped rubric does not say so."""
    verdict = llm_judge(LAB4_JUDGE_RUBRIC_CORRECTNESS.format(
        question=question, reference=reference, candidate=candidate), tier="LARGE")
    return int(verdict.get("score", 0))



# ---------------------------------------------------------------------------
def run_full(save: str = "") -> None:
    questions = load_questions(include_unanswerable=True)
    dense_retreiver, hyrbid_retreiver = build_retriever()
    rows = []

    with Budget(limit_usd=1.00, label="lab4-full") as b:
        for q in questions:
            a = answer_question(q["question"], dense_retreiver, hybrid_retriever=hyrbid_retreiver)
            ctx = format_context(a.hits)
            unanswerable = not q["relevant_docs"] or q["kind"] == "unanswerable"
            rows.append({
                "id": q["id"], "kind": q["kind"], "unanswerable": unanswerable,
                "answer": a.text, "refused": a.refused,
                "citations_valid": a.citations_valid,
                "invalid_citations": a.invalid_citations,
                "faithfulness": judge_faithfulness(a.text, ctx),
                "correctness": judge_correctness(q["question"], a.text, q["gold_answer"]),
                "retrieved": [h.doc_id for h in a.hits],
                "relevant": q["relevant_docs"],
            })

    ans = [r for r in rows if not r["unanswerable"]]
    una = [r for r in rows if r["unanswerable"]]
    refusals = [r for r in rows if r["refused"]]

    print(f"\nn = {len(rows)}  ({len(ans)} answerable, {len(una)} unanswerable)")
    print(f"citation validity   {statistics.fmean(r['citations_valid'] for r in rows):.3f}"
          "   (target 1.000)")
    print(f"faithfulness        {statistics.fmean(r['faithfulness'] for r in rows):.3f}")
    print(f"correctness (0-2)   {statistics.fmean(r['correctness'] for r in ans):.3f}"
          f"  normalised {statistics.fmean(r['correctness'] for r in ans) / 2:.3f}")
    rec = (sum(1 for r in una if r["refused"]) / len(una)) if una else 0.0
    prec = (sum(1 for r in refusals if r["unanswerable"]) / len(refusals)) if refusals else 1.0
    print(f"refusal recall      {rec:.3f}   ({sum(1 for r in una if r['refused'])}/{len(una)})")
    print(f"refusal precision   {prec:.3f}   ({len(refusals)} refusals total)")
    print("\n" + b.report())

    print("\nby question kind (mean correctness / 2):")
    kinds = sorted({r["kind"] for r in ans})
    for kind in kinds:
        sub = [r for r in ans if r["kind"] == kind]
        print(f"  {kind:<16} {statistics.fmean(r['correctness'] for r in sub)/2:.3f}"
              f"  n={len(sub)}")

    if save:
        p = ROOT / save
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nsaved -> {p}   (Lab 5 reads this file)")


def run_gold_context() -> None:
    """E2: the decomposition. This is the highest-value 10 minutes in the lab."""
    questions = [q for q in load_questions() if q["relevant_docs"]]
    dense_retreiver, hybrid_retreiver = build_retriever()
    corpus = load_corpus()

    retrieved_scores, gold_scores = [], []
    with Budget(limit_usd=1.00, label="lab4-decomposition"):
        for q in questions:
            a = answer_question(q["question"], dense_retreiver, hybrid_retriever=hybrid_retreiver)
            retrieved_scores.append(
                judge_correctness(q["question"], a.text, q["gold_answer"]) / 2)
            g = answer_with_gold_context(
                q["question"], [corpus[d] for d in q["relevant_docs"] if d in corpus])
            gold_scores.append(
                judge_correctness(q["question"], g.text, q["gold_answer"]) / 2)

    A, B = statistics.fmean(gold_scores), statistics.fmean(retrieved_scores)
    print(f"\ncorrectness with GOLD context       A = {A:.3f}   <- generation ceiling")
    print(f"correctness with RETRIEVED context  B = {B:.3f}   <- your system")
    print(f"retrieval-attributable loss   A - B = {A - B:.3f}")
    print(f"generation-attributable loss  1 - A = {1 - A:.3f}")
    print("\nWhichever is larger is where Lab 5 goes.")


def make_calibration_sheet() -> None:
    """D2: writes 20 answers for you to hand-label BEFORE seeing the judge."""
    rows = json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))
    sample = rows[:20]
    LABEL_SHEET.write_text("\n".join(json.dumps({
        "id": r["id"], "answer": r["answer"],
        "human_faithfulness": None, "human_correctness": None,
    }, ensure_ascii=False) for r in sample) + "\n", encoding="utf-8")
    print(f"wrote {LABEL_SHEET}")
    print("Fill in human_faithfulness (0/1) and human_correctness (0/1/2), then:")
    print("  python labs/lab4/evaluate.py --kappa")

def make_detailed_calibration_sheet():
    questions = load_questions(include_unanswerable=True)
    dense_retriever, hybrid_retriever = build_retriever()

    rows = []

    for q in questions[:20]:

        a = answer_question(
            q["question"],
            dense_retriever,
            hybrid_retriever=hybrid_retriever
        )

        context = format_context(a.hits)

        rows.append({
            "id": q["id"],
            "question": q["question"],
            "gold_answer": q["gold_answer"],
            "context": context,
            "answer": a.text,
            "human_faithfulness": None,
            "human_correctness": None
        })

    path = ROOT / "labs/lab4/calibration_detailed.jsonl"

    path.write_text(
        "\n".join(
            json.dumps(r, ensure_ascii=False)
            for r in rows
        ) + "\n",
        encoding="utf-8"
    )

    print("wrote", path)

def report_kappa() -> None:
    human = [json.loads(l) for l in LABEL_SHEET.open(encoding="utf-8")]
    machine = {r["id"]: r for r in
               json.loads((ROOT / "reports/lab4.json").read_text(encoding="utf-8"))}
    for field in ("faithfulness", "correctness"):
        h = [r[f"human_{field}"] for r in human if r[f"human_{field}"] is not None]
        m = [machine[r["id"]][field] for r in human if r[f"human_{field}"] is not None]
        if not h:
            print(f"{field}: no human labels yet")
            continue
        print(f"{field}: {judge_agreement(m, h)}")
    print("\nkappa < 0.4 -> fix the rubric, not the model. Read your disagreements.")

def test_refusal_metrics():
    questions = load_questions(include_unanswerable=True)

    dense_retriever, hybrid_retriever = build_retriever()

    total_refused = 0
    correctly_refused = 0
    wrongful_refusals = 0
    should_have_refused = 0

    for q in questions:
        a = answer_question(
            q["question"],
            dense_retriever,
            hybrid_retriever=hybrid_retriever
        )

        unanswerable = (
            not q["relevant_docs"]
            or q["kind"] == "unanswerable"
        )

        if unanswerable:
            should_have_refused += 1

        if a.refused:
            total_refused += 1

            if unanswerable:
                correctly_refused += 1

            else:
                wrongful_refusals += 1

                print("\nWrongful refusal:", q["id"])
                print("Question:", q["question"])
                print("Answer:", a.text)

    refusal_recall = (
        correctly_refused / should_have_refused
        if should_have_refused > 0
        else 0
    )

    refusal_precision = (
        correctly_refused / total_refused
        if total_refused > 0
        else 1
    )

    print("\n--- REFUSAL METRICS ---")
    print("Should have refused:", should_have_refused)
    print("Correct refusals:", correctly_refused)
    print("Wrongful refusals:", wrongful_refusals)
    print("Total refusals:", total_refused)

    print(
        f"Refusal recall: {correctly_refused}/{should_have_refused} "
        f"= {refusal_recall:.3f}"
    )

    print(
        f"Refusal precision: {correctly_refused}/{total_refused} "
        f"= {refusal_precision:.3f}"
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--full", action="store_true")
    ap.add_argument("--gold-context", action="store_true")
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--kappa", action="store_true")
    ap.add_argument("--save", default="")
    a = ap.parse_args()
    if a.full:
        run_full(a.save)
    if a.gold_context:
        run_gold_context()
    if a.calibrate:
        make_calibration_sheet()
    if a.kappa:
        report_kappa()
    if not any([a.full, a.gold_context, a.calibrate, a.kappa]):
        ap.print_help()
