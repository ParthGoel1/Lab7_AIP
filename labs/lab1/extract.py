#!/usr/bin/env python3
"""Lab 1, Parts B and C — the extractor you actually ship.

Complete the TODOs. `run_eval.py` imports `extract_b` and `extract_c` from
here, so keep those two function names.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from aip.guards import _PII_PATTERNS  # noqa: E402
from aip.llm import StructuredOutputError, structured  # noqa: E402

CATEGORIES = Literal["billing", "claims", "policy_change",
                     "technical", "complaint", "information"]

SENTIMENTS = Literal["angry","frustrated","neutral","satisfied"]

PRODUCTS = Literal["bronze","silver","gold","platinum","unknown"]

LANGUAGES = Literal["en","hi-en"]

# ===========================================================================
# PART B — the schema
# ===========================================================================
class TicketRecord(BaseModel):
    """The contract. Everything the model is allowed to say, and nothing else.

    Remember from T2 §3.2: field `description`s are shipped to the model as
    part of the JSON Schema. They are the highest-leverage place to put an
    instruction, because they sit next to the thing they govern. Write them as
    instructions to the model, not as documentation for a human.
    """

    # TODO B1a: Should `evidence` be declared here, BEFORE the fields it
    #           justifies, or after them? T2 §3.3. Decide, move it, and leave
    #           a one-line comment saying which effect you chose and why.



    # Putting it before, since there could be wrong classification, so evidence first then decide the category

    evidence : str = Field(max_length=200, description="The span of the ticket that determined the category, "
                            "quoted verbatim. One sentence at most.")

    category: CATEGORIES = Field(
        description=(
        "Classify the main subject of the ticket. "
        "billing = premiums, payments, refunds, duplicate charges, or other money paid to Aurora. "
        "claims = filing, processing, approving, denying, reimbursing, or asking about an insurance claim. "
        "policy_change = changing policy details, coverage, members, nominees, or the insurance contract. "
        "technical = an app, website, portal, login, upload, or other technical feature is not working. "
        "complaint = the main issue is Aurora's service, conduct, repeated mishandling, or treatment of the customer, "
        "rather than the underlying billing/claim/policy task itself. "
        "information = a general question or request for information with no pending customer-specific transaction."
    )
    )

    urgency: int = Field(
        ge=1, le=5,
        description=(
        "Choose the highest urgency level whose condition applies. "
        "1 = general information or self-service; no customer record lookup needed. "
        "2 = customer-specific lookup, account action, defect fix, or transaction in progress, "
        "but nothing has yet failed or become stuck. "
        "3 = something has already failed, gone wrong, or is stuck and the customer is waiting. "
        "4 = repeated unresolved failure, money or access is at risk now, "
        "or the customer threatens escalation. "
        "5 = emergency in progress, formal denial requiring immediate reversal, "
        "or the customer states they are actively escalating to the Ombudsman. "
        "After choosing the base level, add 1 for a same-day or next-morning deadline, capped at 5. "
        "Tone, shouting, politeness, and message length do not increase urgency."
    )
    )

    # TODO B1d: sentiment  -> Literal["angry","frustrated","neutral","satisfied"]
    sentiment : SENTIMENTS = Field(description=(
        "Customer tone only, independent of urgency. "
        "angry = hostile, shouting, insulting, or threatening. "
        "frustrated = unhappy or tired of trying and references a prior failure, "
        "such as a repeated attempt, unanswered request, delay, or something not working. "
        "neutral = matter-of-fact or a first-time request, even if terse. "
        "satisfied = expresses thanks, praise, or a positive experience."
    ))
    # TODO B1e: product    -> Literal["bronze","silver","gold","platinum","unknown"]
    #           Note "unknown" is a legal value. Say explicitly when to use it.
    product : PRODUCTS = Field(description=(
        "Insurance product named in the customer's message: bronze, silver, gold, "
        "or platinum. Use 'unknown' if the message does not explicitly name the plan. "
        "Do not infer the product from premium amount, sum insured, policy number, "
        "customer details, or other context."
    ))
    # TODO B1f: language   -> Literal["en","hi-en"]
    language : LANGUAGES
    # TODO B1g: evidence   -> str, max_length=200, "the span of the ticket that
    #           determined the category, quoted verbatim"

    # Part B only: the model decides these. In Part C you will delete them
    # from this schema and compute them in code instead.
    policy_number: str | None = Field(
        default=None,
        description=(
        "Policy number from the current live message only. "
        "It must have the exact format 'AUR-' followed by exactly 7 digits "
        "and must be copied verbatim without inventing, correcting, or reformatting it. "
        "Ignore policy numbers found only in quoted reply lines beginning with '>' "
        "or in signature blocks. "
        "Use null if no valid policy number appears in the live message."
    )
    )
    contains_pii: bool = Field(
        default=False,
        description=(
        "True if the message contains a phone number or an email address, "
        "except Aurora's published addresses support@aurorahealth.example and "
        "grievance@aurorahealth.example. "
        "A personal name alone does not count as PII for this field. "
        "False otherwise."
    )
    )

    # Set by our code, never by the model.
    needs_human_review: bool = False
    review_reason: str = ""

    @field_validator("policy_number")
    @classmethod
    def _policy_format(cls, v: str | None) -> str | None:
        # TODO B1j: reject anything that is not exactly AUR-<7 digits>.
        #           Return None rather than raising if the model returned an
        #           empty string or the literal "null" -- decide which of those
        #           two behaviours you want and defend it in your report.
        if v is None: 
            return None

        if v.strip().lower() in {"", "null"}:
            return None
        if not re.fullmatch(r"AUR-\d{7}", v):
            raise ValueError("policy_number must match AUR- followed by exactly 7 digits")

        return v



SYSTEM_PROMPT = """\
You are a support-ticket classifier for Aurora Health Insurance.

Your task is to classify each customer support ticket according to the
provided TicketRecord schema. Follow the field descriptions exactly;
they define the labeling policy and take precedence over assumptions.

Classify only from information present in the current customer message.
Do not invent missing details or infer values that the schema rules say
must be explicit. Treat each field independently where required.

Pay particular attention to boundary cases described in the schema,
including urgency, sentiment, product, policy number, and PII rules.

Return only a valid TicketRecord-compatible structured response.
Do not include explanations, commentary, markdown, or extra fields.

The customer ticket follows:
"""



def extract_b(ticket: str) -> TicketRecord:
    """Part B: the model decides everything."""
    # TODO B3: call aip.llm.structured with TicketRecord.
    try:
        result = structured(
            ticket, 
            schema=TicketRecord, 
            system= SYSTEM_PROMPT
            )
        return result
    
    except StructuredOutputError:
        fallback = TicketRecord(
            evidence="Structured extraction failed",
            category="information",
            urgency=3,
            sentiment="neutral",
            product="unknown",
            language="en",
            policy_number=None,
            contains_pii=False,
            needs_human_review=True,
            review_reason="call failed"
            )
        return fallback

    # TODO B4: catch StructuredOutputError and return a record with
    #          needs_human_review=True. This function must never raise.


# ===========================================================================
# PART C — move the deterministic work out of the model
# ===========================================================================
POLICY_RE = re.compile(r"\bAUR-\d{7}\b")

# The quoted-reply marker. Everything after this is history, not the current
# message. Part C3 asks you to decide what that means for policy extraction.
QUOTE_MARKER = re.compile(r"^\s*>", re.MULTILINE)

aurora_mails = [
    'support@aurorahealth.example',
    'grievance@aurorahealth.example'
]


def extract_deterministic(ticket: str) -> dict:
    """TODO C1: return {'policy_number', 'contains_pii'} without a model call.

    policy_number:
        Find AUR-<7 digits>.

    TODO C3 -- the trap. Some tickets contain TWO policy-number-shaped strings:
        one in the live body, and one in a quoted reply below a '>' line from
        an earlier thread. They are not always the same number.

        Decide a rule. Write it down in a comment right here. Implement it.
        Then ask yourself whether it generalises or whether you have fitted it
        to this dataset -- the honest answer is worth marks.

    contains_pii:
        True if the ticket contains a phone number or an email address.
        aip.guards._PII_PATTERNS has the patterns. Note that a *name* alone
        does not count for this dataset's labels -- check the gold data and
        say in your report whether you think that definition is right.
    """
    quote_match = QUOTE_MARKER.search(ticket)

    contains_pii = False 
    if quote_match:
        live_text = ticket[:quote_match.start()]
    else:
        live_text = ticket

    policy_match = POLICY_RE.search(live_text)

    if policy_match:
        policy_number = policy_match.group()
    else:
        policy_number = None

    email_matches = _PII_PATTERNS["EMAIL"].findall(ticket)
    phone_match = _PII_PATTERNS["PHONE_IN"].search(ticket)

    if phone_match:
        contains_pii = True
    else:
        for email in email_matches:
            if email not in aurora_mails:
                contains_pii = True
                break

    return {
        'policy_number' : policy_number, 
        'contains_pii' : contains_pii
    }

    raise NotImplementedError


def apply_business_rules(rec_fields: dict, ticket: str) -> dict:
    """TODO C1b: compute `escalate` in code.

        escalate = urgency >= 4 or 'ombudsman' appears in the ticket

    This is a business rule. It belongs in code where it can be read by a
    compliance officer, changed without touching a prompt, and unit-tested.
    Write the unit test in tests/ while you are here.
    """
    updated = rec_fields.copy()
    high_urgency = updated['urgency'] >= 4
    mentions_ombuds = "ombudsman" in ticket.lower()
    updated['escalate'] = high_urgency or mentions_ombuds
    return updated
    raise NotImplementedError


class TicketRecordC(BaseModel):

     # Evidence first so the model grounds its classification before deciding.
     """TODO C2: the reduced schema the model sees in Part C.
    
        Copy TicketRecord and delete the fields you now compute in code. Fewer
        fields means a shorter prompt, fewer output tokens, and three fields at
        100% accuracy. Measure all three effects.
        """
     evidence: str = Field(
        max_length=200,
        description=(
            "The span of the ticket that determined the category, "
            "quoted verbatim. One sentence at most."
        )
    )
     category: CATEGORIES = Field(
        description=(
            "Classify the main subject of the ticket. "
            "billing = premiums, payments, refunds, duplicate charges, or other money paid to Aurora. "
            "claims = filing, processing, approving, denying, reimbursing, or asking about an insurance claim. "
            "policy_change = changing policy details, coverage, members, nominees, or the insurance contract. "
            "technical = an app, website, portal, login, upload, or other technical feature is not working. "
            "complaint = the main issue is Aurora's service, conduct, repeated mishandling, or treatment of the customer, "
            "rather than the underlying billing/claim/policy task itself. "
            "information = a general question or request for information with no pending customer-specific transaction."
        )
    )
     urgency: int = Field(
        ge=1,
        le=5,
        description=(
            "Choose the highest urgency level whose condition applies. "
            "1 = general information or self-service; no customer record lookup needed. "
            "2 = customer-specific lookup, account action, defect fix, or transaction in progress, "
            "but nothing has yet failed or become stuck. "
            "3 = something has already failed, gone wrong, or is stuck and the customer is waiting. "
            "4 = repeated unresolved failure, money or access is at risk now, "
            "or the customer threatens escalation. "
            "5 = emergency in progress, formal denial requiring immediate reversal, "
            "or the customer states they are actively escalating to the Ombudsman. "
            "After choosing the base level, add 1 for a same-day or next-morning deadline, capped at 5. "
            "Tone, shouting, politeness, and message length do not increase urgency."
        )
    )
     sentiment: SENTIMENTS = Field(
        description=(
            "Customer tone only, independent of urgency. "
            "angry = hostile, shouting, insulting, or threatening. "
            "frustrated = unhappy or tired of trying and references a prior failure, "
            "such as a repeated attempt, unanswered request, delay, or something not working. "
            "neutral = matter-of-fact or a first-time request, even if terse. "
            "satisfied = expresses thanks, praise, or a positive experience."
        )
    )
     product: PRODUCTS = Field(
        description=(
            "Insurance product named in the customer's message: bronze, silver, gold, "
            "or platinum. Use 'unknown' if the message does not explicitly name the plan. "
            "Do not infer the product from premium amount, sum insured, policy number, "
            "customer details, or other context."
        )
    )
     language: LANGUAGES


def extract_c(ticket: str) -> dict:
    """Part C: model for judgement, code for everything else.

    Returns a plain dict (model fields + deterministic fields + business rules)
    so that run_eval.py can score it against the gold labels directly.
    """

    model_result = structured(
        ticket,
        schema=TicketRecordC,
        system=SYSTEM_PROMPT
    )

    rec_fields = model_result.model_dump()

    deterministic_fields = extract_deterministic(ticket)
    rec_fields.update(deterministic_fields)

    rec_fields = apply_business_rules(rec_fields, ticket)

    return rec_fields


if __name__ == "__main__":
    import json

    root = Path(__file__).resolve().parents[2]
    sample = json.loads(
        (root / "data/eval/extraction_dev.jsonl").open(encoding="utf-8").readline()
    )
    print("--- ticket ---")
    print(sample["input"][:600])
    print("\n--- gold ---")
    print(sample["expected"])
    print("\n--- yours ---")
    print(extract_c(sample["input"]))
