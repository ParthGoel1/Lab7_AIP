#!/usr/bin/env python3
"""Lab 5 — the failure classifier.

    python labs/lab5/diagnose.py --input reports/lab4.json
    python labs/lab5/diagnose.py --input reports/lab4.json --pareto

Implements the T4 §5 diagnostic tree.
Mode 2 still needs manual inspection.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from aip.chunking import markdown_chunks  # noqa: E402
from labs.lab3.search import load_corpus, load_questions  # noqa: E402
from labs.lab4.rag import answer_with_gold_context  # noqa: E402
from labs.lab4.evaluate import judge_correctness, build_retriever  # noqa: E402


MODES = {
    1: "missing_content",
    2: "chunk_boundary",
    3: "embedding_mismatch",
    4: "ranking",
    5: "reranker",
    6: "generation",
    7: "presentation",
}


# ---------------------------------------------------------------------------
# Mode 1
# ---------------------------------------------------------------------------

def answer_in_corpus(
    gold_answer: str,
    corpus: dict[str, str],
    relevant_docs: list[str]
) -> bool:
    """Check whether the gold answer seems to exist in the relevant documents."""

    text = " ".join(
        corpus.get(doc, "")
        for doc in relevant_docs
    ).lower()

    if not text:
        return False

    cleaned_answer = (
        gold_answer.lower()
        .replace(",", "")
        .replace(".", "")
        .replace(";", "")
        .replace("(", "")
        .replace(")", "")
    )

    cleaned_text = (
        text.replace(",", "")
        .replace(".", "")
        .replace(";", "")
        .replace("(", "")
        .replace(")", "")
    )

    words = [
        word
        for word in cleaned_answer.split()
        if len(word) > 4
    ]

    if not words:
        return cleaned_answer in cleaned_text

    matches = 0

    for word in words:
        if word in cleaned_text:
            matches += 1

    overlap = matches / len(words)

    return overlap >= 0.5


# ---------------------------------------------------------------------------
# Retrieval helpers
# ---------------------------------------------------------------------------

def choose_retriever(
    question: str,
    dense_retriever,
    hybrid_retriever
):
    """Use the same routing logic as Lab 4."""

    has_identifier = bool(
        re.search(
            r"\b[A-Z]{2,}(?:-[A-Z0-9]+)+\b",
            question
        )
    )

    if has_identifier:
        return hybrid_retriever

    return dense_retriever


def clean_words(text: str) -> set[str]:
    """Small helper used only to locate the likely gold chunk."""

    words = re.findall(
        r"\b\w+\b",
        text.lower()
    )

    return {
        word
        for word in words
        if len(word) > 4
    }


def find_best_gold_chunk(
    q: dict,
    corpus: dict[str, str]
):
    """Find the chunk in the relevant docs that overlaps most with the gold answer.

    This is used only for the Mode 3 self-retrieval test.
    """

    gold_words = clean_words(
        q["gold_answer"]
    )

    best_chunk = None
    best_score = -1

    for doc_id in q["relevant_docs"]:

        if doc_id not in corpus:
            continue

        chunks = markdown_chunks(
            corpus[doc_id],
            doc_id,
            size=400
        )

        for chunk in chunks:

            chunk_words = clean_words(
                chunk.text
            )

            score = len(
                gold_words.intersection(
                    chunk_words
                )
            )

            if score > best_score:
                best_score = score
                best_chunk = chunk

    return best_chunk


def gold_chunk_retrieves_itself(
    q: dict,
    corpus: dict[str, str],
    dense_retriever
) -> bool:
    """Mode 3 test.

    If the best gold chunk can retrieve its own document when queried with
    its own text, the chunk is searchable and the original query is likely
    the mismatch.
    """

    gold_chunk = find_best_gold_chunk(
        q,
        corpus
    )

    if gold_chunk is None:
        return False

    hits = dense_retriever.search(
        gold_chunk.text,
        k=5
    )

    return any(
        hit.doc_id == gold_chunk.doc_id
        for hit in hits
    )


# ---------------------------------------------------------------------------
# Diagnostic tree
# ---------------------------------------------------------------------------

def classify(
    row: dict,
    q: dict,
    corpus: dict[str, str],
    *,
    gold_context_fixes_it: bool | None = None,
    in_top_30: bool | None = None,
    in_final_k: bool | None = None,
    gold_chunk_self_retrieves: bool | None = None,
    dropped_by_reranker: bool | None = None
) -> tuple[int, str]:
    """Walk the Lab 5 diagnostic tree."""

    # -------------------------------------------------
    # Mode 7 — presentation
    # -------------------------------------------------

    if (
        row.get("correctness", 0) >= 2
        and not row.get("citations_valid", True)
    ):
        return 7, (
            f"correct answer, invalid citations "
            f"{row.get('invalid_citations')}"
        )


    # -------------------------------------------------
    # Mode 1 — missing content
    # -------------------------------------------------

    if not answer_in_corpus(
        q["gold_answer"],
        corpus,
        q["relevant_docs"]
    ):
        return 1, (
            "gold answer content not found "
            "in the relevant documents"
        )


    # -------------------------------------------------
    # Mode 6 — generation
    # -------------------------------------------------

    if gold_context_fixes_it is False:
        return 6, (
            "answer still wrong even with gold context"
        )


    # -------------------------------------------------
    # Mode 5 — reranker
    # -------------------------------------------------

    if (
        in_top_30 is True
        and in_final_k is False
        and dropped_by_reranker is True
    ):
        return 5, (
            "relevant document was found "
            "but dropped by reranker"
        )


    # -------------------------------------------------
    # Mode 4 — ranking
    # -------------------------------------------------

    if (
        in_top_30 is True
        and in_final_k is False
    ):
        return 4, (
            "relevant document found in top 30 "
            "but not in final context"
        )


    # -------------------------------------------------
    # Relevant document reached final context
    # -------------------------------------------------

    if in_final_k is True:
        return 2, (
            "needs_human_check: relevant document reached final context; "
            "inspect the actual chunks and distractors"
        )


    # -------------------------------------------------
    # Mode 3 — embedding mismatch
    # -------------------------------------------------

    if (
        in_top_30 is False
        and gold_chunk_self_retrieves is True
    ):
        return 3, (
            "gold chunk retrieves itself, "
            "so the original query is an embedding mismatch"
        )


    # -------------------------------------------------
    # Mode 2 — chunk boundary
    # -------------------------------------------------

    return 2, (
        "needs_human_check: inspect the chunks "
        "around the gold answer"
    )


# ---------------------------------------------------------------------------
# Pareto
# ---------------------------------------------------------------------------

def pareto(tally: Counter) -> str:

    total = sum(
        tally.values()
    ) or 1

    lines = [
        "failure mode          n    share   cumulative"
    ]

    cumulative = 0

    for mode, n in tally.most_common():

        cumulative += n

        bar = "█" * round(
            30 * n / total
        )

        lines.append(
            f"{MODES[mode]:<20} "
            f"{n:>3}   "
            f"{n / total:>5.1%}   "
            f"{cumulative / total:>5.1%}  "
            f"{bar}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:

    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--input",
        default="reports/lab4.json"
    )

    ap.add_argument(
        "--pareto",
        action="store_true"
    )

    ap.add_argument(
        "--save",
        default="reports/lab5_diagnosis.json"
    )

    args = ap.parse_args()


    # -------------------------------------------------
    # Load Lab 4 results
    # -------------------------------------------------

    rows = json.loads(
        (ROOT / args.input).read_text(
            encoding="utf-8"
        )
    )


    # -------------------------------------------------
    # Load questions
    # -------------------------------------------------

    questions = {
        q["id"]: q
        for q in load_questions(
            include_unanswerable=True
        )
    }


    # -------------------------------------------------
    # Load corpus
    # -------------------------------------------------

    corpus = load_corpus()


    # -------------------------------------------------
    # Build retrievers once
    # -------------------------------------------------

    dense_retriever, hybrid_retriever = (
        build_retriever()
    )


    # -------------------------------------------------
    # Keep only Lab 4 failures
    # -------------------------------------------------

    failures = [
        row
        for row in rows
        if (
            row.get("correctness", 2) < 2
            or not row.get(
                "citations_valid",
                True
            )
        )
    ]


    print(
        f"{len(failures)} failures "
        f"out of {len(rows)}\n"
    )


    out = []
    tally = Counter()


    # -------------------------------------------------
    # Diagnose each failure
    # -------------------------------------------------

    for row in failures:

        q = questions[
            row["id"]
        ]


        # =================================================
        # 1. Gold-context test
        # =================================================

        gold_context_fixes_it = None

        if q["relevant_docs"]:

            gold_docs = [
                corpus[doc_id]
                for doc_id in q["relevant_docs"]
                if doc_id in corpus
            ]


            gold_answer = answer_with_gold_context(
                q["question"],
                gold_docs
            )


            gold_score = judge_correctness(
                q["question"],
                gold_answer.text,
                q["gold_answer"]
            )


            gold_context_fixes_it = (
                gold_score == 2
            )


        # =================================================
        # 2. Retrieval checks
        # =================================================

        in_top_30 = None
        in_final_k = None


        if gold_context_fixes_it is True:

            retriever = choose_retriever(
                q["question"],
                dense_retriever,
                hybrid_retriever
            )


            # Broad retrieval
            hits_30 = retriever.search(
                q["question"],
                k=30
            )


            # Recreate normal Lab 4 retrieval
            hits_12 = retriever.search(
                q["question"],
                k=12
            )

            final_hits = hits_12[:5]


            top_30_docs = {
                hit.doc_id
                for hit in hits_30
            }


            final_docs = {
                hit.doc_id
                for hit in final_hits
            }


            relevant_docs = set(
                q["relevant_docs"]
            )


            # At least one labelled relevant document
            # appeared in broad retrieval
            in_top_30 = bool(
                relevant_docs.intersection(
                    top_30_docs
                )
            )


            # At least one relevant document
            # reached final context
            in_final_k = bool(
                relevant_docs.intersection(
                    final_docs
                )
            )


        # =================================================
        # 3. Mode 3 self-retrieval test
        # =================================================

        gold_chunk_self_retrieves = None


        if (
            gold_context_fixes_it is True
            and in_top_30 is False
        ):

            gold_chunk_self_retrieves = (
                gold_chunk_retrieves_itself(
                    q,
                    corpus,
                    dense_retriever
                )
            )


        # =================================================
        # 4. Classify
        # =================================================

        mode, evidence = classify(
            row,
            q,
            corpus,
            gold_context_fixes_it=(
                gold_context_fixes_it
            ),
            in_top_30=in_top_30,
            in_final_k=in_final_k,
            gold_chunk_self_retrieves=(
                gold_chunk_self_retrieves
            ),

            # No reranker was active in your Lab 4 pipeline
            dropped_by_reranker=False
        )


        # =================================================
        # 5. Record result
        # =================================================

        tally[
            mode
        ] += 1


        out.append({
            "id": row["id"],
            "kind": q["kind"],
            "mode": mode,
            "mode_name": MODES[mode],
            "evidence": evidence,
            "question": q["question"],
            "answer": row["answer"][:300]
        })


        print(
            f"  {row['id']:<5} "
            f"{MODES[mode]:<20} "
            f"{evidence}"
        )


    # -------------------------------------------------
    # Pareto summary
    # -------------------------------------------------

    print(
        "\n" + pareto(
            tally
        )
    )


    print(
        "\nCases marked needs_human_check "
        "are Part A2. Open them."
    )


    # -------------------------------------------------
    # Save diagnosis
    # -------------------------------------------------

    path = ROOT / args.save


    path.parent.mkdir(
        parents=True,
        exist_ok=True
    )


    path.write_text(
        json.dumps(
            out,
            indent=2,
            ensure_ascii=False
        ),
        encoding="utf-8"
    )


    print(
        f"\nsaved -> {path}"
    )


if __name__ == "__main__":
    main()