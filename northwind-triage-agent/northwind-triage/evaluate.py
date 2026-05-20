"""
Batch evaluator for the Northwind triage agent.

Runs the agent against all 20 messages in 05_Inbound_Messages.json and
scores against 06_Benchmark.json per the rubric in 07_Evaluation_Rubric.pdf.

Scoring:
  - category:           exact match (1) or no match (0)
  - priority:           exact match (1) or no match (0)
  - route_to:           exact match (1); partial credit (0.5) if primary team correct but cc missed
  - needs_human_review: exact match (1) or no match (0)
  - strict:             1 only if all four hard fields match
"""

import json
import os
import time
from typing import Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def load_messages() -> list:
    path = os.path.join(BASE_DIR, "05_Inbound_Messages.json")
    # Force UTF-8: Python's default `open()` uses the locale codepage on Windows
    # (cp1252), which mangles the UTF-8 characters in the message bodies
    # (e.g. "12m²" → "12mÂ²", "—" → "â€”").
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return data["messages"]


def load_benchmark() -> dict:
    path = os.path.join(BASE_DIR, "06_Benchmark.json")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {d["id"]: d for d in data["decisions"]}


def _score_route(agent_route: str, benchmark_route: str) -> float:
    """
    Route scoring with partial credit.
    "Customer Care + Accounts" vs "Customer Care" → 0.5 (primary right, cc missed).
    """
    a = agent_route.strip().lower()
    b = benchmark_route.strip().lower()

    if a == b:
        return 1.0

    # Partial credit: primary team correct but compound routing missed
    b_primary = b.split("+")[0].strip()
    a_primary = a.split("+")[0].strip()

    if a_primary == b_primary:
        return 0.5  # primary team right, cc missed

    return 0.0


def run_evaluation(agent, progress_callback=None) -> dict:
    """
    Run agent across all 20 messages. Returns a full evaluation report.

    Rate limiting:
      Gemini free tier is 10 RPM (2.5-flash) or 15 RPM (2.5-flash-lite). The
      agent itself retries 429s with exponential backoff, but proactive spacing
      between calls is cheaper than letting the server reject us.

      Default: 7s between calls (~8.5 RPM, comfortably under the 10 RPM limit).
      Override with GEMINI_REQUEST_DELAY_S env var. Set to 4.5 for paid tier
      or flash-lite (15 RPM), or 0 to send back-to-back and rely on retries.
    """
    messages = load_messages()
    benchmark = load_benchmark()

    delay_s = float(os.getenv("GEMINI_REQUEST_DELAY_S", "4.5"))
    print(f"[eval] Running {len(messages)} messages with {delay_s}s spacing between calls "
          f"(set GEMINI_REQUEST_DELAY_S to override)")

    results = []
    field_scores = {"category": [], "priority": [], "route_to": [], "needs_human_review": []}
    strict_matches = 0
    daily_quota_hit = False

    for i, msg in enumerate(messages):
        msg_id = msg["id"]
        gold = benchmark.get(msg_id, {})

        # Short-circuit on daily quota: no point spending 90s per message getting rejected
        if daily_quota_hit:
            print(f"[eval] [{msg_id}] SKIPPED — daily quota already exhausted")
            results.append({
                "id": msg_id,
                "error": "Skipped — daily quota exhausted on earlier message",
                "agent": {},
                "benchmark": gold,
                "scores": {"category": 0, "priority": 0, "route_to": 0, "needs_human_review": 0, "strict": 0},
            })
            for k in field_scores:
                field_scores[k].append(0)
            continue

        print(f"[eval] [{i + 1}/{len(messages)}] Triaging {msg_id}…")
        if progress_callback:
            progress_callback(i + 1, len(messages), msg_id)

        # Call the agent (which has its own retry/backoff for 429s)
        try:
            decision = agent.triage(msg)
            error = None
        except Exception as e:
            decision = {}
            error = str(e)
            print(f"[eval] [{msg_id}] ERROR: {error}")
            # Detect daily quota exhaustion → stop trying more messages
            err_lower = error.lower()
            if "daily quota" in err_lower or "perday" in err_lower or "per-day" in err_lower:
                daily_quota_hit = True
                print("[eval] Daily quota exhausted. Skipping remaining messages. "
                      "Re-run after midnight Pacific time or enable billing for an instant tier upgrade.")

        if error:
            result = {
                "id": msg_id,
                "error": error,
                "agent": {},
                "benchmark": gold,
                "scores": {"category": 0, "priority": 0, "route_to": 0, "needs_human_review": 0, "strict": 0},
            }
            results.append(result)
            for k in field_scores:
                field_scores[k].append(0)
            # Even on error, space the next call out (unless quota is hit, in which case continue immediately)
            if i < len(messages) - 1 and not daily_quota_hit:
                time.sleep(delay_s)
            continue

        # Score each field
        cat_score = 1 if decision.get("category") == gold.get("category") else 0
        pri_score = 1 if decision.get("priority") == gold.get("priority") else 0
        route_score = _score_route(decision.get("route_to", ""), gold.get("route_to", ""))
        nhr_score = 1 if decision.get("needs_human_review") == gold.get("needs_human_review") else 0

        strict = 1 if (cat_score == 1 and pri_score == 1 and route_score == 1 and nhr_score == 1) else 0
        if strict:
            strict_matches += 1

        field_scores["category"].append(cat_score)
        field_scores["priority"].append(pri_score)
        field_scores["route_to"].append(route_score)
        field_scores["needs_human_review"].append(nhr_score)

        results.append({
            "id": msg_id,
            "error": None,
            "agent": decision,
            "benchmark": gold,
            "scores": {
                "category": cat_score,
                "priority": pri_score,
                "route_to": route_score,
                "needs_human_review": nhr_score,
                "strict": strict,
            },
        })

        # Rate limiting: stay under Gemini free-tier RPM
        if i < len(messages) - 1:
            time.sleep(delay_s)

    n = len(messages)
    per_field = {k: round(sum(v) / n * 100, 1) for k, v in field_scores.items()}

    return {
        "total_messages": n,
        "strict_accuracy": round(strict_matches / n * 100, 1),
        "strict_count": strict_matches,
        "per_field_accuracy": per_field,
        "results": results,
    }


def print_report(report: dict):
    """Pretty-print the evaluation report to stdout."""
    print("\n" + "="*60)
    print("NORTHWIND TRIAGE AGENT — EVALUATION REPORT")
    print("="*60)
    print(f"Strict accuracy (all 4 fields): {report['strict_accuracy']}% ({report['strict_count']}/{report['total_messages']})")
    print("\nPer-field accuracy:")
    for field, pct in report["per_field_accuracy"].items():
        bar = "█" * int(pct / 5) + "░" * (20 - int(pct / 5))
        print(f"  {field:<22} {bar} {pct}%")

    print("\nPer-message breakdown:")
    print(f"  {'ID':<12} {'Cat':>4} {'Pri':>4} {'Route':>6} {'NHR':>4} {'Strict':>7}  Agent vs Benchmark")
    print("  " + "-"*80)
    for r in report["results"]:
        s = r["scores"]
        a = r["agent"]
        b = r["benchmark"]
        strict_icon = "✓" if s["strict"] else "✗"
        if r["error"]:
            print(f"  {r['id']:<12} ERROR: {r['error']}")
            continue
        mismatch = []
        if s["category"] == 0:
            mismatch.append(f"cat:{a.get('category','?')}≠{b.get('category','?')}")
        if s["priority"] == 0:
            mismatch.append(f"pri:{a.get('priority','?')}≠{b.get('priority','?')}")
        if s["route_to"] < 1:
            mismatch.append(f"route:{a.get('route_to','?')}≠{b.get('route_to','?')}")
        if s["needs_human_review"] == 0:
            mismatch.append(f"nhr:{a.get('needs_human_review','?')}≠{b.get('needs_human_review','?')}")
        note = " | " + ", ".join(mismatch) if mismatch else ""
        print(f"  {r['id']:<12} {s['category']:>4} {s['priority']:>4} {s['route_to']:>6.1f} {s['needs_human_review']:>4} {strict_icon:>7}{note}")

    print("="*60 + "\n")


if __name__ == "__main__":
    from agent import TriageAgent
    agent = TriageAgent()
    report = run_evaluation(agent)
    print_report(report)

    os.makedirs(os.path.join(BASE_DIR, "outputs"), exist_ok=True)
    out_path = os.path.join(BASE_DIR, "outputs", "evaluation_report.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str, ensure_ascii=False)
    print(f"Full report saved to {out_path}")
