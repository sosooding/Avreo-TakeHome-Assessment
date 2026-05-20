"""
Northwind Triage Agent — powered by Google Gemini.

Architecture decision: single-shot structured output via Gemini's JSON mode.
All policy documents (SOP, Service Catalogue, Tone & Style Guide) are embedded
directly in the system prompt (~4k tokens). No RAG, no multi-agent graph — the
task is bounded and the reference corpus fits comfortably in context.

Key design choices:
  - Explicit 8-step decision procedure to guide reasoning
  - Structured JSON schema enforcement for reliable output
  - Python-side out-of-hours detection injected into input
  - Gemini's response_mime_type + response_schema for guaranteed structure
"""

import json
import os
import random
import re
import time
from datetime import datetime, timezone, timedelta
from typing import Optional

from google import genai
from google.genai import types
from google.genai import errors as genai_errors
from dotenv import load_dotenv

load_dotenv()

# ─────────────────────────────────────────────
# Policy documents embedded in the system prompt
# ─────────────────────────────────────────────

SOP = """
STANDARD OPERATING PROCEDURE — Northwind Home Services (v3.2)

SECTION 2 — CATEGORIES
Every inbound message must be classified into exactly one:
  BOOKING      — Customer wants to schedule or is asking about availability for a known service.
  QUOTE        — Customer is asking for a price/estimate for work not yet agreed.
  COMPLAINT    — Customer is unhappy with completed work, tradesperson conduct, billing accuracy, or service delivery.
  EMERGENCY    — Active risk to property or safety: water leak in progress, no hot water in winter, electrical sparking, gas smell.
  BILLING      — Customer asking about invoice, payment, refund, or account statement.
  OUT_OF_SCOPE — Request is for something we don't offer, or is not actionable (spam, wrong number, garbled).

Classification notes:
  - If a message contains both a complaint AND a new request → classify as COMPLAINT, note secondary request in reasoning.
  - A reschedule request is BOOKING, not COMPLAINT, unless the customer explicitly expresses dissatisfaction.
  - If unsure between QUOTE and BOOKING → default to QUOTE (safer to confirm scope before dispatching).

SECTION 3 — PRIORITY LEVELS
  P1 — Active safety or property risk. First response SLA: within 1 hour, 24/7.
  P2 — Loss of essential function (heating, hot water, working toilet) but no immediate damage. Or any complaint over $1,000. First response: within 4 business hours.
  P3 — Standard enquiry, quote request, non-urgent booking. First response: within 1 business day.
  EMERGENCY messages are ALWAYS P1. Never downgrade even if the customer's tone is calm.

SECTION 4 — ROUTING
  Dispatch     — All BOOKING and EMERGENCY messages.
  Sales        — All QUOTE messages.
  Accounts     — All BILLING messages.
  Customer Care — All COMPLAINT messages. Also fallback for anything that doesn't fit cleanly elsewhere.
  Special rules:
  - OUT_OF_SCOPE → Customer Care (polite decline).
  - COMPLAINT involving billing dispute over $500 → cc BOTH Accounts AND Customer Care (route_to: "Customer Care + Accounts").
  - Emergency plumbing AND emergency electrical → Dispatch (on-call tradies for both).
  - We have NO on-call HVAC. After-hours HVAC issues are P2 and route to Dispatch for next-business-day allocation.

SECTION 5 — FIRST-RESPONSE DRAFTING
  Every message gets a draft reply at triage time. The draft must:
  - Acknowledge the customer's specific situation in the first sentence.
  - State what happens next and roughly when (use SLA windows, not specific times).
  - Match the Tone & Style Guide.
  - NEVER quote a price unless the catalogue lists a FIXED price for that exact service.
  - NEVER commit to a specific tradesperson by name.

SECTION 6 — WHEN TO FLAG FOR HUMAN REVIEW (needs_human_review = true)
  Flag if ANY of the following apply:
  - Customer is angry, distressed, or threatens legal action / online review.
  - Request involves a quote over $5,000 or a refund over $500.
  - Message is in a language other than English, or appears garbled/incoherent.
  - Cannot confidently classify the message after re-reading.
  - Customer mentions a previous complaint or escalation.
  - Request is borderline outside the service catalogue.
  Flagging is cheap. Missing an escalation is expensive. When in doubt, flag.

SECTION 7 — OUT-OF-HOURS HANDLING
  Outside 07:00–18:00 weekdays, only P1 EMERGENCY messages are actioned live.
  P2 and P3 are queued for the next business morning.
  Draft replies for queued messages must acknowledge the wait ("we'll be in touch first thing tomorrow").
"""

SERVICE_CATALOGUE = """
SERVICE CATALOGUE — Northwind Home Services (v2024.1)

Prices marked FIXED can be quoted directly. Prices marked 'from' are estimates — NEVER quote these to customers.

PLUMBING:
  Tap washer replacement: $120 FIXED (single tap)
  Hot water system repair: from $180/hr (gas, electric, heat-pump)
  Hot water system replacement: from $1,800 (site assessment required)
  Burst pipe repair: from $220/hr (ALWAYS P1 if water actively flowing)
  Blocked drain clearing: $280 FIXED (standard drain)
  Toilet repair/replacement: from $150
  Bathroom renovation plumbing: from $4,500 (site visit + written quote ALWAYS required)

ELECTRICAL:
  Power point installation: $190 FIXED per outlet (standard; surcharge for upper floors)
  Light fitting installation: $150 FIXED (customer supplies fitting)
  Switchboard upgrade: from $2,200 (site assessment required)
  Safety switch installation: $320 FIXED per circuit
  Electrical fault diagnosis: from $180/hr (sparking, tripping, smell of burning = P1)
  EV charger installation: from $1,400 (single-phase; min. 12-month service age)
  Solar panel installation: NOT OFFERED — refer to SunPath Energy

HVAC:
  Split-system service/clean: $220 FIXED per unit
  Split-system installation: from $1,600 (standard install up to 5m pipe run)
  Ducted system service: from $380
  Ducted system installation: from $9,500 (site assessment ALWAYS required; lead time 2-4 weeks)
  Gas heater service: $280 FIXED (includes safety/carbon monoxide check)
  Evaporative cooler service: $240 FIXED

THINGS WE DO NOT DO (decline politely):
  - Roofing, gutter cleaning, anything requiring full roof access
  - Solar panel installation or repair (refer: SunPath Energy)
  - Pool plumbing or equipment (refer: AquaCorp Pools)
  - Appliance repair (washing machines, dishwashers, ovens) — we INSTALL but do NOT REPAIR
  - Anything in commercial premises larger than 200m² (residential only)
  - Locksmithing, glazing, pest control

SERVICE AREA: Within 40km of Sydney CBD. Outside radius → flag for human review.

BILLING:
  - Payment due within 14 days of invoice.
  - Jobs over $2,000 require 30% deposit before work.
  - Refunds for completed work handled case-by-case through Customer Care.
"""

TONE_GUIDE = """
TONE & STYLE GUIDE — Northwind Home Services

HOW WE SOUND: Like a competent neighbour, not a call centre.
Three rules: (1) Plain, not formal. (2) Specific, not generic. (3) Honest, not performative.

LENGTH: 2-4 sentences. Short. Direct.

GREETINGS & SIGN-OFFS:
  - Open with customer's first name if known. If not, go straight in.
  - Sign off with "— The Northwind team" for first responses.
  - NEVER use "Dear", "Kind regards", or "Yours sincerely".

AVOID → USE INSTEAD:
  "At your earliest convenience" → "Today / tomorrow / by Friday"
  "We will endeavour to" → "We'll"
  "Apologies for any inconvenience" → "Sorry about that — here's what we'll do"
  "Please rest assured" → (delete it)
  "Service representative" → "tradie / plumber / electrician / someone from dispatch"
  "Thank you for contacting Northwind" → (skip opener; acknowledge the actual issue)

HARD RULES:
  - NEVER quote a 'from' price. Only quote fixed prices.
  - NEVER name a specific tradesperson.
  - NEVER promise an exact time — give a window or SLA.
  - NEVER use exclamation marks.
  - NEVER use emoji.

EXAMPLES:
  Booking ✓: "Hi Sarah — got your message about the dripping tap in the ensuite. We'll have someone call you back within the day to lock in a time. — The Northwind team"
  Emergency ✓: "Hi Tom — water leak is a priority for us. Someone from dispatch will call you within the hour. In the meantime, if you can shut off the mains at the meter, that'll buy us time. — The Northwind team"
  Out of scope ✓: "Hi Mei — gutter cleaning isn't something we cover, sorry. Have a look at Allroof Services in your area. — The Northwind team"
  Complaint ✓: "Hi Dan — that's a frustrating experience and not what we want for our customers. I've passed this to our Customer Care lead, who'll call you back today. — The Northwind team"
"""

SYSTEM_PROMPT = f"""You are the first-pass triage agent for Northwind Home Services, a residential trades business in Sydney, Australia. You receive customer messages and produce a structured triage decision for human review.

The three documents below are your ONLY source of truth. Do not invent rules that are not in them.

{'='*60}
{SOP}
{'='*60}
{SERVICE_CATALOGUE}
{'='*60}
{TONE_GUIDE}
{'='*60}

DECISION PROCEDURE — follow these steps in order for every message:

1. Read the full message. Note: sender, channel, time context, what they're asking, tone, language, coherence.

2. Check OUT_OF_SCOPE first. Does the request match anything in "Things we do not do"?
   - Gutter cleaning, roof access → OUT_OF_SCOPE
   - Solar panels → OUT_OF_SCOPE (refer SunPath Energy)
   - Appliance REPAIR (dishwashers, washing machines, ovens) → OUT_OF_SCOPE (we install only)
   - Pool plumbing/equipment → OUT_OF_SCOPE
   - Commercial premises >200m² → OUT_OF_SCOPE

3. Classify category per SOP Section 2. Evaluate in order: EMERGENCY → COMPLAINT → BILLING → QUOTE → BOOKING.
   - EMERGENCY = ACTIVE risk happening NOW: water actively flowing, sparking, gas smell, no hot water in winter.
   - Heater stopped working / HVAC failure is NOT EMERGENCY — it's a BOOKING (we have no on-call HVAC).
   - Reschedule of existing booking = BOOKING, not COMPLAINT.
   - Refund queries (even slightly frustrated) = BILLING, not COMPLAINT.
   - Conduct complaints about tradesperson = COMPLAINT.

4. Set priority per SOP Section 3.
   - EMERGENCY is always P1.
   - Loss of essential function (no heating in winter, no hot water, no working toilet) = P2 if not EMERGENCY.
   - Complaint over $1,000 OR over the $500 refund/billing threshold = P2.
   - Refund delays where customer is following up = P2 (financial impact).
   - HVAC after-hours = P2 (no on-call), routes to Dispatch for next business day.
   - Everything else = P3.

5. Route per SOP Section 4.
   - BOOKING / EMERGENCY → Dispatch
   - QUOTE → Sales
   - BILLING → Accounts
   - COMPLAINT → Customer Care
   - OUT_OF_SCOPE → Customer Care
   - COMPLAINT with billing dispute over $500 → "Customer Care + Accounts"

6. **HUMAN REVIEW CHECKLIST** — go through EVERY item. If ANY is yes, set needs_human_review = true:

   a. Does the request involve a quote/job in the catalogue listed at "from $X" where X ≥ $5,000?
      → Examples: ducted aircon installation (from $9,500) → YES.
      → Bathroom renovation plumbing (from $4,500) is borderline — flag if customer mentions full reno scope.

   b. Does the message mention a refund OR a paid deposit over $500?
      → Any refund query over $500 → YES.

   c. Is the message in a language OTHER than English (Spanish, Mandarin, French, etc.)?
      → Detect by looking at the body text. If you see "necesito", "hola", non-English words → YES.

   d. Is the message garbled, incoherent, or appears to be spam/test?
      → "asdf asdf", random characters, no real content → YES.

   e. Is the customer angry, distressed, OR threatening legal action / online review / social media?
      → Phrases like "considering leaving a review", "we are not happy", "this is unacceptable" → YES.

   f. Is the request borderline outside the service catalogue (e.g. appliance repair where we only install)?
      → Dishwasher repair, washing machine repair → YES (borderline, customer needs referral).

   g. Is the address outside the 40km Sydney CBD service area?
      → YES if mentioned outside (most messages won't trigger this).

   h. Does the message involve multi-unit / strata / commercial-adjacent work?
      → Strata block servicing, body corporate work → YES (borderline residential).

   i. Does the customer mention a previous complaint, escalation, or recommendation context that implies higher expectations?
      → YES.

   j. Cannot you confidently classify the message even after re-reading?
      → YES.

   Flagging is cheap. Missing an escalation is expensive. When in doubt → flag.

7. Draft reply: 2–4 sentences, matches tone guide, no 'from' prices, no tradesperson names, no exact times.

8. Write reasoning: cite specific rules (e.g. "Section 6 trigger: ducted aircon from $9,500 > $5k threshold").
   Be explicit about which human-review trigger(s) applied, or that NONE applied.

CRITICAL:
- Output valid JSON only. No markdown, no preamble.
- category must be exactly: BOOKING, QUOTE, COMPLAINT, EMERGENCY, BILLING, or OUT_OF_SCOPE
- priority must be exactly: P1, P2, or P3
- route_to must be exactly: Dispatch, Sales, Accounts, Customer Care, or "Customer Care + Accounts"
- For garbled/spam messages, write a brief honest note that you couldn't understand the message
"""

# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

SYDNEY_TZ = timezone(timedelta(hours=10))  # AEST


def _out_of_hours_context(received_at: Optional[str]) -> str:
    """
    Deterministic out-of-hours detection. LLMs are unreliable at parsing
    ISO timestamps into day-of-week so we compute it in Python and inject
    the result as plain text.
    """
    if not received_at:
        return ""
    try:
        dt = datetime.fromisoformat(received_at)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=SYDNEY_TZ)
        local = dt.astimezone(SYDNEY_TZ)
        weekday = local.strftime("%A")
        time_str = local.strftime("%H:%M")
        is_weekend = local.weekday() >= 5
        hour = local.hour
        is_outside = is_weekend or hour < 7 or hour >= 18
        if is_outside:
            return (
                f"\n⚠ OUT-OF-HOURS: Message received {weekday} at {time_str} AEST "
                f"(outside 07:00–18:00 weekdays). Only P1 EMERGENCY is actioned live. "
                f"P2/P3 → queue for next business morning."
            )
        return f"\n✓ IN-HOURS: Message received {weekday} at {time_str} AEST."
    except Exception:
        return ""


def _format_message(msg: dict) -> str:
    """Format a message dict into a structured prompt."""
    ooh = _out_of_hours_context(msg.get("received_at"))
    return f"""<inbound_message>
ID: {msg.get('id', 'N/A')}
Channel: {msg.get('channel', 'unknown')}
Received: {msg.get('received_at', 'unknown')}{ooh}
From: {msg.get('sender_name', 'unknown')} <{msg.get('sender_address', '')}>
Subject: {msg.get('subject') or '(none)'}

{msg.get('body', '')}
</inbound_message>

Triage this message. Return valid JSON only."""


# ─────────────────────────────────────────────
# Response schema for Gemini's JSON mode
# ─────────────────────────────────────────────

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "category": {
            "type": "string",
            "enum": ["BOOKING", "QUOTE", "COMPLAINT", "EMERGENCY", "BILLING", "OUT_OF_SCOPE"]
        },
        "priority": {
            "type": "string",
            "enum": ["P1", "P2", "P3"]
        },
        "route_to": {
            "type": "string"
        },
        "draft_reply": {
            "type": "string"
        },
        "needs_human_review": {
            "type": "boolean"
        },
        "reasoning": {
            "type": "string"
        }
    },
    "required": ["category", "priority", "route_to", "draft_reply", "needs_human_review", "reasoning"]
}


# ─────────────────────────────────────────────
# TriageAgent
# ─────────────────────────────────────────────

class RateLimitError(Exception):
    """Raised after exhausting retries on a 429."""
    pass


# Model: gemini-3.1-flash-lite.
#
# This is a rule-based classification task with structured JSON output. flash-lite
# is the right tier: fast, high-throughput, and reliable for structured output.
# Override via the GEMINI_MODEL environment variable if needed.
DEFAULT_MODEL = "gemini-3.1-flash-lite"

# Retry config — Gemini free tier is 10–15 RPM, so 429s on bursts are normal.
# We retry with exponential backoff up to ~180s total, respecting any retry-after hint.
# Daily quota errors come back with 40+ second retryDelay hints, so we honor those too.
_MAX_RETRIES = 5
_BASE_DELAY_S = 2.0
_MAX_DELAY_S = 90.0  # Daily-quota errors typically suggest 40s+ waits


def _is_daily_quota_error(err: Exception) -> bool:
    """
    Detect a daily quota exhaustion (vs per-minute rate limit). Daily quota
    means the agent should stop entirely — retrying won't help until midnight PT.
    """
    msg = str(err).lower()
    return (
        "perday" in msg
        or "per-day" in msg
        or "rpd" in msg
        or "generaterequestsperday" in msg
    )


def _parse_retry_after(err: Exception) -> Optional[float]:
    """
    Pull a retry-after hint from a Gemini APIError if present.
    Google returns this as a 'retryDelay' field inside the error details (e.g. '23s').
    """
    msg = str(err)
    # Look for "retryDelay": "Ns" or "retry after Ns"
    m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', msg)
    if m:
        return float(m.group(1))
    m = re.search(r'retry[_\s-]?after[:\s]+(\d+(?:\.\d+)?)', msg, re.IGNORECASE)
    if m:
        return float(m.group(1))
    return None


def _is_rate_limit_error(err: Exception) -> bool:
    """Detect a 429 / RESOURCE_EXHAUSTED error from Gemini."""
    if isinstance(err, genai_errors.ClientError):
        # The new SDK exposes .code as the HTTP status
        code = getattr(err, "code", None) or getattr(err, "status_code", None)
        if code == 429:
            return True
    msg = str(err).lower()
    return "429" in msg or "resource_exhausted" in msg or "rate limit" in msg or "quota" in msg


def _is_transient_server_error(err: Exception) -> bool:
    """
    Detect a transient server-side error worth retrying: 503 UNAVAILABLE
    ("model experiencing high demand") or 500 INTERNAL. These are temporary
    and usually clear on a retry — unlike a 400/permission error.
    """
    code = getattr(err, "code", None) or getattr(err, "status_code", None)
    if code in (500, 503):
        return True
    msg = str(err).lower()
    return ("503" in msg or "500" in msg or "unavailable" in msg
            or "high demand" in msg or "internal" in msg or "overloaded" in msg)


def _is_retryable_error(err: Exception) -> bool:
    """Either a rate-limit (429) or a transient server error (500/503) — both worth retrying."""
    return _is_rate_limit_error(err) or _is_transient_server_error(err)


class TriageAgent:
    """
    Single-shot Gemini triage agent using the google-genai SDK.

    Design: one well-prompted call with the full reference corpus in context.
    Low temperature for rule-following consistency. JSON mode for structured output.

    Rate-limit handling: the .triage() method retries 429s with exponential backoff
    and respects the server's retryDelay hint when present. It also retries once on a
    truncated (MAX_TOKENS) response with a larger output-token budget.
    """

    def __init__(self, model: Optional[str] = None):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError(
                "GEMINI_API_KEY not set. Add it to your .env file.\n"
                "Get a free key at: https://aistudio.google.com/"
            )
        self._model_name = model or os.getenv("GEMINI_MODEL", DEFAULT_MODEL)
        self._client = genai.Client(api_key=api_key)

        # Gemini models have "thinking" enabled by default, which spends part of the
        # output-token budget on internal reasoning before producing the response.
        # For a fast, deterministic classification task we set thinking to its lowest
        # setting so the full budget goes to the structured JSON response, and we give
        # generous output headroom as a safety net.
        self._thinking_config = None
        try:
            self._thinking_config = types.ThinkingConfig(thinking_level="low")
        except Exception:
            self._thinking_config = None

        self._max_tokens_initial = 4096
        self._max_tokens_retry = 8192
        self._config = self._build_config(self._max_tokens_initial)

        print(f"[agent] Using model: {self._model_name} "
              f"(max_output_tokens: {self._max_tokens_initial})")

    def _build_config(self, max_output_tokens: int) -> "types.GenerateContentConfig":
        """Build a GenerateContentConfig with the given output-token budget."""
        return types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            response_mime_type="application/json",
            response_schema=RESPONSE_SCHEMA,
            temperature=0.1,
            max_output_tokens=max_output_tokens,
            thinking_config=self._thinking_config,
        )

    def _generate_with_retry(self, prompt: str, config=None):
        """Call Gemini with exponential backoff on 429 rate limits and transient 5xx errors."""
        if config is None:
            config = self._config
        last_err: Optional[Exception] = None
        for attempt in range(_MAX_RETRIES + 1):
            try:
                return self._client.models.generate_content(
                    model=self._model_name,
                    contents=prompt,
                    config=config,
                )
            except Exception as e:
                last_err = e
                # Only retry rate limits (429) and transient server errors (500/503).
                # Anything else (bad request, auth, etc.) is not retryable — raise it.
                if not _is_retryable_error(e):
                    raise
                # Daily quota errors can't be retried away — surface a clear message
                if _is_daily_quota_error(e):
                    raise RateLimitError(
                        f"Daily quota exhausted for {self._model_name} on the free tier "
                        f"(typically 200–1000 requests/day). The quota resets at midnight "
                        f"Pacific time. To raise this limit immediately, enable billing on "
                        f"your Google Cloud project for an instant Tier-1 upgrade. "
                        f"Original error: {e}"
                    ) from e
                if attempt == _MAX_RETRIES:
                    raise
                # Prefer server-provided retry-after; otherwise exponential backoff with jitter
                hinted = _parse_retry_after(e)
                if hinted is not None:
                    delay = min(hinted + random.uniform(0.2, 0.8), _MAX_DELAY_S)
                else:
                    delay = min(_BASE_DELAY_S * (2 ** attempt) + random.uniform(0, 1.0), _MAX_DELAY_S)
                kind = "503 server-overload" if _is_transient_server_error(e) else "429 rate-limited"
                print(f"[agent] {kind} — backing off {delay:.1f}s (attempt {attempt + 1}/{_MAX_RETRIES})")
                time.sleep(delay)
        # If we exhausted retries
        raise RateLimitError(f"Exhausted {_MAX_RETRIES} retries on retryable error: {last_err}")

    def _call(self, prompt: str, config) -> tuple:
        """Call Gemini, returning (raw_text, finish_reason)."""
        response = self._generate_with_retry(prompt, config)
        finish_reason = None
        try:
            if response.candidates:
                finish_reason = response.candidates[0].finish_reason
        except Exception:
            pass
        raw = (response.text or "").strip()
        # Strip any accidental markdown fences
        raw = re.sub(r'^```(?:json)?\s*', '', raw)
        raw = re.sub(r'\s*```$', '', raw)
        return raw, finish_reason

    def triage(self, message: dict) -> dict:
        """Triage a single inbound message. Returns a structured dict."""
        prompt = _format_message(message)

        # First attempt with the standard token budget
        raw, finish_reason = self._call(prompt, self._config)

        # If the response was truncated (MAX_TOKENS) or empty, retry once with
        # a much larger token budget. This handles cases where the model spends
        # part of its budget on internal thinking before producing the response.
        reason_str = str(finish_reason) if finish_reason else "unknown"
        if not raw or "MAX_TOKENS" in reason_str:
            print(f"[agent] First attempt truncated/empty (finish_reason={reason_str}). "
                  f"Retrying with max_output_tokens={self._max_tokens_retry}…")
            bigger_config = self._build_config(self._max_tokens_retry)
            raw, finish_reason = self._call(prompt, bigger_config)
            reason_str = str(finish_reason) if finish_reason else "unknown"

        if not raw:
            raise ValueError(
                f"Gemini returned an empty response (finish_reason={reason_str}) "
                f"even after retry with {self._max_tokens_retry} tokens. "
                "This usually means a safety filter blocked the response. "
                "Check the message content."
            )

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            # Log the raw response so the user can see what actually came back.
            print(f"[agent] JSON parse failed. finish_reason={reason_str}. Raw response:")
            print(f"[agent] {'-'*60}")
            print(raw)
            print(f"[agent] {'-'*60}")

            hint = ""
            if "MAX_TOKENS" in reason_str:
                hint = " The response was truncated by the token limit even after retry."
            elif "SAFETY" in reason_str or "PROHIBITED" in reason_str or "BLOCKLIST" in reason_str:
                hint = f" The response was blocked by a content filter ({reason_str})."

            raise ValueError(
                f"Gemini returned invalid JSON (finish_reason={reason_str}).{hint} "
                f"Raw: {raw[:300]}…"
            ) from e

        # Validate required fields
        required = ["category", "priority", "route_to", "draft_reply", "needs_human_review", "reasoning"]
        for field in required:
            if field not in result:
                raise ValueError(f"Missing field in Gemini response: {field}")

        return result
