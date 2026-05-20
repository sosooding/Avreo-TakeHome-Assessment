"""
Qualitative checks for the Northwind triage agent's draft replies and reasoning.

The evaluation rubric (07_Evaluation_Rubric.pdf) defines three qualitative checks
that are "not scored, but reported":
  - Draft reply quality — hits the points in `draft_reply_must_include`, avoids the
    items in `draft_reply_must_not_include`, and sounds like the tone guide.
  - Reasoning quality — does the reasoning weigh the right rules?
  - Tone — does the draft sound like Northwind, or like a generic LLM?

This module provides two layers:

  1. DETERMINISTIC CHECKS (free, no API calls)
     Objective, mechanical tone-guide rules that can be verified with string logic:
     exclamation marks, banned phrases, sign-off presence, length, and 'from' prices.

  2. LLM-AS-JUDGE (optional, one extra API call per message)
     A second Gemini call that grades the semantic checks the benchmark describes in
     natural language ("specific reference to the dripping tap") plus overall tone and
     reasoning quality on a 1–5 scale. Opt-in, because it doubles API usage.
"""

import json
import os
import re
from typing import Optional

# Tone-guide banned phrases (from 04_Tone_and_Style_Guide.pdf, "Words and phrases to avoid")
BANNED_PHRASES = [
    "at your earliest convenience",
    "we will endeavour",
    "apologies for any inconvenience",
    "please rest assured",
    "rest assured",
    "service representative",
    "kindly",
    "reach out",
    "thank you for contacting",
    "dear ",
    "kind regards",
    "yours sincerely",
]

# "from $X" services in the catalogue — quoting these exact prices violates the tone guide.
# We look for a dollar amount that matches a known 'from' price.
FROM_PRICES = ["180", "1,800", "1800", "220", "150", "4,500", "4500",
               "2,200", "2200", "1,400", "1400", "1,600", "1600",
               "380", "9,500", "9500"]


# ─────────────────────────────────────────────
# Layer 1: Deterministic checks (free)
# ─────────────────────────────────────────────

def deterministic_checks(draft_reply: str) -> dict:
    """
    Run objective tone-guide checks on a draft reply. No API calls.

    Returns a dict with each check's pass/fail and a list of any issues found.
    """
    issues = []
    text = draft_reply or ""
    lower = text.lower()

    # 1. No exclamation marks (tone guide hard rule)
    has_exclamation = "!" in text
    if has_exclamation:
        issues.append("Contains an exclamation mark (tone guide forbids these).")

    # 2. No emoji (tone guide hard rule)
    has_emoji = bool(re.search(
        r"[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F0FF]", text))
    if has_emoji:
        issues.append("Contains an emoji (tone guide forbids these).")

    # 3. No banned phrases
    found_banned = [p for p in BANNED_PHRASES if p in lower]
    if found_banned:
        issues.append(f"Uses banned phrase(s): {', '.join(repr(p) for p in found_banned)}.")

    # 4. Has the correct sign-off (first responses end with "The Northwind team")
    has_signoff = "northwind team" in lower
    if not has_signoff:
        issues.append("Missing the '— The Northwind team' sign-off.")

    # 5. Length: 2–4 sentences (tone guide). Count sentence-ending punctuation,
    #    excluding the sign-off line.
    body_for_count = re.sub(r"—\s*the northwind team\.?\s*$", "", text.strip(), flags=re.IGNORECASE)
    sentence_count = len(re.findall(r"[.?]+(?:\s|$)", body_for_count))
    length_ok = 2 <= sentence_count <= 5  # allow a little slack (4 + sign-off fragment)
    if not length_ok:
        issues.append(f"Length looks off (~{sentence_count} sentences; tone guide wants 2–4).")

    # 6. No 'from' price quoted. Flag only if a dollar sign appears near a 'from' price.
    quoted_from_price = None
    if "$" in text:
        for price in FROM_PRICES:
            # match e.g. "$180", "$1,800", "$9,500"
            if re.search(r"\$\s*" + re.escape(price), text):
                quoted_from_price = price
                break
    if quoted_from_price:
        issues.append(f"Quotes a 'from' (estimate) price (${quoted_from_price}); tone guide says only fixed prices may be quoted.")

    checks = {
        "no_exclamation": not has_exclamation,
        "no_emoji": not has_emoji,
        "no_banned_phrases": not found_banned,
        "has_signoff": has_signoff,
        "length_ok": length_ok,
        "no_from_price": quoted_from_price is None,
    }
    passed = sum(1 for v in checks.values() if v)
    total = len(checks)

    return {
        "checks": checks,
        "passed": passed,
        "total": total,
        "score_pct": round(passed / total * 100, 1),
        "issues": issues,
    }


# ─────────────────────────────────────────────
# Layer 2: LLM-as-judge (optional)
# ─────────────────────────────────────────────

JUDGE_SYSTEM_PROMPT = """You are a strict quality reviewer for a customer-service triage agent at Northwind Home Services, a residential trades business. You assess whether a drafted reply and the agent's reasoning meet the company's standards. You are fair but not generous — a 5 means genuinely excellent, a 3 means acceptable-with-flaws.

Northwind's tone: like a competent neighbour, not a call centre. Plain not formal, specific not generic, honest not performative. Replies are 2–4 sentences, open with the customer's first name (if known) or go straight in, and sign off "— The Northwind team". They never quote estimate prices, never name a tradesperson, never promise an exact time, never use exclamation marks or emoji.

You will be given the original customer message, the agent's draft reply, the agent's reasoning, and two checklists: things the reply MUST include and things it MUST NOT include. Grade against these.

Return ONLY valid JSON, no markdown, in exactly this shape:
{
  "must_include_met": <integer count of must-include items the reply satisfies>,
  "must_include_total": <integer total must-include items>,
  "must_not_include_violations": <integer count of must-not-include items the reply violates>,
  "tone_score": <1-5 integer: how well it matches Northwind's voice>,
  "reasoning_score": <1-5 integer: did the reasoning weigh the right rules, or bluff past ambiguity>,
  "comment": "<one sentence on the most important strength or weakness>"
}"""


def _build_judge_prompt(message: dict, decision: dict, benchmark: dict) -> str:
    must_include = benchmark.get("draft_reply_must_include", [])
    must_not = benchmark.get("draft_reply_must_not_include", [])
    return f"""ORIGINAL CUSTOMER MESSAGE:
{message.get('body', '')}

AGENT'S DRAFT REPLY:
{decision.get('draft_reply', '')}

AGENT'S REASONING:
{decision.get('reasoning', '')}

MUST INCLUDE (the reply should satisfy each of these):
{json.dumps(must_include, indent=2)}

MUST NOT INCLUDE (the reply must avoid each of these):
{json.dumps(must_not, indent=2)}

Grade the draft and reasoning. Return only the JSON object."""


def llm_judge(client, model_name: str, message: dict, decision: dict, benchmark: dict) -> Optional[dict]:
    """
    Use a second Gemini call to grade the draft reply and reasoning.

    `client` is a google.genai Client (passed in from the agent so we reuse its key).
    Returns the parsed judge result, or None on failure (judging is best-effort).
    """
    try:
        from google.genai import types
    except Exception:
        return None

    prompt = _build_judge_prompt(message, decision, benchmark)
    try:
        config = types.GenerateContentConfig(
            system_instruction=JUDGE_SYSTEM_PROMPT,
            response_mime_type="application/json",
            temperature=0.0,
            max_output_tokens=2048,
        )
        try:
            config.thinking_config = types.ThinkingConfig(thinking_level="low")
        except Exception:
            pass

        response = client.models.generate_content(
            model=model_name,
            contents=prompt,
            config=config,
        )
        raw = (response.text or "").strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)
        return json.loads(raw)
    except Exception as e:
        return {"error": str(e)}


# ─────────────────────────────────────────────
# Aggregation helpers
# ─────────────────────────────────────────────

def summarise_qualitative(results: list) -> dict:
    """
    Aggregate per-message qualitative results into a summary.
    `results` is the list of per-message dicts that each contain a 'qualitative' key.
    """
    det_scores = []
    tone_scores = []
    reasoning_scores = []
    must_include_met = 0
    must_include_total = 0
    must_not_violations = 0
    judged = 0

    for r in results:
        q = r.get("qualitative")
        if not q:
            continue
        det = q.get("deterministic")
        if det:
            det_scores.append(det["score_pct"])
        judge = q.get("llm_judge")
        if judge and "error" not in judge:
            judged += 1
            if isinstance(judge.get("tone_score"), (int, float)):
                tone_scores.append(judge["tone_score"])
            if isinstance(judge.get("reasoning_score"), (int, float)):
                reasoning_scores.append(judge["reasoning_score"])
            must_include_met += judge.get("must_include_met", 0)
            must_include_total += judge.get("must_include_total", 0)
            must_not_violations += judge.get("must_not_include_violations", 0)

    def _avg(xs):
        return round(sum(xs) / len(xs), 2) if xs else None

    summary = {
        "deterministic_avg_pct": _avg(det_scores),
        "messages_judged_by_llm": judged,
    }
    if judged:
        summary.update({
            "avg_tone_score": _avg(tone_scores),
            "avg_reasoning_score": _avg(reasoning_scores),
            "must_include_coverage_pct": (
                round(must_include_met / must_include_total * 100, 1)
                if must_include_total else None
            ),
            "total_must_not_violations": must_not_violations,
        })
    return summary
