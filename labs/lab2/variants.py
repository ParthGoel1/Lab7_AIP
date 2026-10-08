#!/usr/bin/env python3
"""Lab 2 — the configurations under test.

Each variant is a callable str -> dict. grid.py runs them all through the
same harness, so the only thing that differs between rows is the thing
we intended to differ.
"""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


from aip.llm import structured

from labs.lab1.extract import (
    SYSTEM_PROMPT,
    TicketRecordC,
    apply_business_rules,
    extract_deterministic,
)


# ---------------------------------------------------------------------------
# Conservative API pacing
# ---------------------------------------------------------------------------

# Keep at least 1.5 seconds between API calls.
# This gives a theoretical maximum of about 40 requests/minute.
#
# The lock also makes this safe if grid.py is accidentally run with
# multiple workers.
API_DELAY_SECONDS = 1.5

_api_lock = threading.Lock()
_last_api_call_time = 0.0


def pace_api() -> None:
    """Space API requests so we do not unnecessarily bombard the provider."""

    global _last_api_call_time

    with _api_lock:
        now = time.monotonic()
        elapsed = now - _last_api_call_time

        if _last_api_call_time > 0 and elapsed < API_DELAY_SECONDS:
            wait = API_DELAY_SECONDS - elapsed
            time.sleep(wait)

        _last_api_call_time = time.monotonic()


# ---------------------------------------------------------------------------
# A1 — six selected few-shot examples
# ---------------------------------------------------------------------------

FEW_SHOT_IDS: list[str] = [
    "T0097",  # billing vs complaint
    "T0048",  # no explicit policy number -> null
    "T0200",  # Hinglish
    "T0201",  # sentiment and urgency are independent
    "T0020",  # forwarded / quoted-message structure
    "T0054",  # one I got wrong in Lab 1
]


def load_examples(ids: list[str]) -> list[dict]:
    """Load selected few-shot examples from the dev set."""

    path = ROOT / "data/eval/extraction_dev.jsonl"

    rows = [
        json.loads(line)
        for line in path.open(encoding="utf-8")
    ]

    by_id = {
        row["id"]: row
        for row in rows
    }

    missing = [
        ticket_id
        for ticket_id in ids
        if ticket_id not in by_id
    ]

    if missing:
        raise KeyError(
            f"Unknown few-shot example IDs: {missing}"
        )

    return [
        by_id[ticket_id]
        for ticket_id in ids
    ]


def few_shot_block(ids: list[str]) -> str:
    """Render selected examples into the prompt."""

    rows = load_examples(ids)

    examples = []

    for row in rows:
        gold = row["expected"]

        # These are the labelled fields available in the gold data.
        # Evidence is not included here because it is not supplied as
        # a labelled gold field in these examples.
        model_output = {
            "category": gold["category"],
            "urgency": gold["urgency"],
            "sentiment": gold["sentiment"],
            "product": gold["product"],
            "language": gold["language"],
        }

        example = (
            "Example input:\n"
            f"{row['input']}\n\n"
            "Example output:\n"
            f"{json.dumps(model_output, ensure_ascii=False)}"
        )

        examples.append(example)

    return "\n\n".join(examples)


# ---------------------------------------------------------------------------
# Zero-shot
# ---------------------------------------------------------------------------

def zero_shot(
    ticket: str,
    tier: str = "SMALL",
) -> dict:
    """Lab 1 Part C baseline: no few-shot examples."""

    pace_api()

    result = structured(
        ticket,
        schema=TicketRecordC,
        system=SYSTEM_PROMPT,
        tier=tier,
    )

    rec = result.model_dump()

    rec.update(
        extract_deterministic(ticket)
    )

    rec = apply_business_rules(
        rec,
        ticket,
    )

    return rec


def zero_shot_main(ticket: str) -> dict:
    """Zero-shot using the MAIN model."""

    return zero_shot(
        ticket,
        tier="MAIN",
    )


# ---------------------------------------------------------------------------
# Few-shot
# ---------------------------------------------------------------------------

def few_shot(
    ticket: str,
    tier: str = "SMALL",
    temperature: float = 0.0,
) -> dict:
    """Zero-shot prompt plus six selected examples."""

    examples = few_shot_block(
        FEW_SHOT_IDS
    )

    system_with_examples = (
        SYSTEM_PROMPT
        + "\n\n"
        + examples
    )

    pace_api()

    result = structured(
        ticket,
        schema=TicketRecordC,
        system=system_with_examples,
        tier=tier,
        temperature=temperature,
    )

    rec = result.model_dump()

    rec.update(
        extract_deterministic(ticket)
    )

    rec = apply_business_rules(
        rec,
        ticket,
    )

    return rec


def few_shot_main(ticket: str) -> dict:
    """Few-shot using the MAIN model."""

    return few_shot(
        ticket,
        tier="MAIN",
        temperature=0.0,
    )


# ---------------------------------------------------------------------------
# Few-shot + reasoning
# ---------------------------------------------------------------------------

class TicketRecordReasoned(BaseModel):
    """TicketRecordC-style schema with reasoning placed first."""

    reasoning: str = Field(
        description=(
            "Briefly reason through the relevant facts in the ticket "
            "before producing the classification fields."
        )
    )

    evidence: str = Field(
        max_length=200,
        description=(
            "The span of the ticket that determined the category, "
            "quoted verbatim. One sentence at most."
        ),
    )

    category: Literal[
        "billing",
        "claims",
        "policy_change",
        "technical",
        "complaint",
        "information",
    ] = Field(
        description=(
            "Classify the main subject of the ticket. "
            "billing = premiums, payments, refunds, duplicate charges, "
            "or other money paid to Aurora. "
            "claims = filing, processing, approving, denying, reimbursing, "
            "or asking about an insurance claim. "
            "policy_change = changing policy details, coverage, members, "
            "nominees, or the insurance contract. "
            "technical = an app, website, portal, login, upload, "
            "or other technical feature is not working. "
            "complaint = the main issue is Aurora's service, conduct, "
            "repeated mishandling, or treatment of the customer, rather "
            "than the underlying billing/claim/policy task itself. "
            "information = a general question or request for information "
            "with no pending customer-specific transaction."
        ),
    )

    urgency: int = Field(
        ge=1,
        le=5,
        description=(
            "Choose the highest urgency level whose condition applies. "
            "1 = general information or self-service; "
            "no customer record lookup needed. "
            "2 = customer-specific lookup, account action, defect fix, "
            "or transaction in progress, but nothing has yet failed "
            "or become stuck. "
            "3 = something has already failed, gone wrong, or is stuck "
            "and the customer is waiting. "
            "4 = repeated unresolved failure, money or access is at risk now, "
            "or the customer threatens escalation. "
            "5 = emergency in progress, formal denial requiring immediate "
            "reversal, or the customer states they are actively escalating "
            "to the Ombudsman. "
            "After choosing the base level, add 1 for a same-day or "
            "next-morning deadline, capped at 5. "
            "Tone, shouting, politeness, and message length do not "
            "increase urgency."
        ),
    )

    sentiment: Literal[
        "angry",
        "frustrated",
        "neutral",
        "satisfied",
    ] = Field(
        description=(
            "Customer tone only, independent of urgency. "
            "angry = hostile, shouting, insulting, or threatening. "
            "frustrated = unhappy or tired of trying and references "
            "a prior failure, such as a repeated attempt, unanswered "
            "request, delay, or something not working. "
            "neutral = matter-of-fact or a first-time request, even if terse. "
            "satisfied = expresses thanks, praise, or a positive experience."
        ),
    )

    product: Literal[
        "bronze",
        "silver",
        "gold",
        "platinum",
        "unknown",
    ] = Field(
        description=(
            "Insurance product named in the customer's message: "
            "bronze, silver, gold, or platinum. "
            "Use 'unknown' if the message does not explicitly name the plan. "
            "Do not infer the product from premium amount, sum insured, "
            "policy number, customer details, or other context."
        ),
    )

    language: Literal[
        "en",
        "hi-en",
    ]


def few_shot_reasoned(
    ticket: str,
    tier: str = "SMALL",
    temperature: float = 0.0,
) -> dict:
    """Few-shot variant with reasoning first in the schema."""

    examples = few_shot_block(
        FEW_SHOT_IDS
    )

    system_with_examples = (
        SYSTEM_PROMPT
        + "\n\n"
        + examples
    )

    pace_api()

    result = structured(
        ticket,
        schema=TicketRecordReasoned,
        system=system_with_examples,
        tier=tier,
        temperature=temperature,
    )

    rec = result.model_dump()

    rec.update(
        extract_deterministic(ticket)
    )

    rec = apply_business_rules(
        rec,
        ticket,
    )

    return rec


def few_shot_reasoned_main(ticket: str) -> dict:
    """Few-shot + reasoning using the MAIN model."""

    return few_shot_reasoned(
        ticket,
        tier="MAIN",
        temperature=0.0,
    )


# ---------------------------------------------------------------------------
# Cascade
# ---------------------------------------------------------------------------

def cascade(ticket: str) -> dict:
    """Use SMALL first; escalate to MAIN when two SMALL samples disagree."""

    try:
        first = few_shot(
            ticket,
            tier="SMALL",
            temperature=0.0,
        )

        second = few_shot(
            ticket,
            tier="SMALL",
            temperature=0.7,
        )

        fields = [
            "category",
            "urgency",
            "sentiment",
            "product",
            "language",
        ]

        agree = all(
            first.get(field) == second.get(field)
            for field in fields
        )

        evidence_ok = bool(
            first.get("evidence", "").strip()
        )

        if agree and evidence_ok:
            first["_path"] = "small"
            return first

    except Exception:
        pass

    rec = few_shot(
        ticket,
        tier="MAIN",
        temperature=0.0,
    )

    rec["_path"] = "large"
    return rec


# ---------------------------------------------------------------------------
# Grid configuration
# ---------------------------------------------------------------------------

VARIANTS = {
    "zero_shot": lambda t: zero_shot(
        t,
        tier="SMALL",
    ),

    "zero_shot_main": zero_shot_main,

    "few_shot": lambda t: few_shot(
        t,
        tier="SMALL",
        temperature=0.0,
    ),

    "few_shot_main": few_shot_main,

    "few_shot_reasoned": lambda t: few_shot_reasoned(
        t,
        tier="SMALL",
        temperature=0.0,
    ),

    "few_shot_reasoned_main": few_shot_reasoned_main,

    "cascade": cascade,
}