# Northwind Home Services — Triage Agent

An AI-powered triage agent that reads inbound customer messages for Northwind Home
Services (a residential trades business) and produces a structured first-pass decision:
what the message is about, how urgent it is, which team should handle it, a drafted
reply in the company's voice, and whether a human needs to look at it.

Built with **Google Gemini (`gemini-3.1-flash-lite`)**, a **FastAPI** backend, and a
single-page **HTML/CSS/JS** frontend.

---

## Table of contents

1. [What it does](#what-it-does)
2. [Project structure](#project-structure)
3. [Quick start](#quick-start)
4. [Environment variables](#environment-variables)
5. [How the agent works](#how-the-agent-works)
6. [API reference](#api-reference)
7. [Running the batch evaluation](#running-the-batch-evaluation)
8. [Results](#results)
9. [Notes on the benchmark](#notes-on-the-benchmark)

---

## What it does

For each inbound customer message (email, SMS, or web form), the agent returns six fields:

| Field                | What it is                                                              |
|----------------------|-------------------------------------------------------------------------|
| `category`           | One of: `BOOKING`, `QUOTE`, `COMPLAINT`, `EMERGENCY`, `BILLING`, `OUT_OF_SCOPE` |
| `priority`           | `P1` (same-day), `P2` (within 48h), `P3` (within 5 business days)        |
| `route_to`           | `Dispatch`, `Sales`, `Accounts`, `Customer Care`, or `Customer Care + Accounts` |
| `draft_reply`        | A short, customer-facing reply written in Northwind's tone              |
| `needs_human_review` | `true` / `false` — flags anything ambiguous or outside policy           |
| `reasoning`          | 2–4 sentences explaining the decision                                   |

The decision logic is driven entirely by three reference documents (the SOP, the
service catalogue, and the tone guide), which are embedded into the agent's prompt.

---

## Project structure

```
northwind-triage/
├── backend/
│   ├── agent.py                 # The triage agent: prompt, decision logic, Gemini calls
│   ├── app.py                   # FastAPI server (the API + serves the frontend)
│   ├── evaluate.py              # Batch evaluation against the benchmark
│   ├── qualitative_checks.py    # Tone checks + optional LLM-as-judge
│   ├── requirements.txt         # Python dependencies
│   ├── 05_Inbound_Messages.json # 20 test messages (input)
│   └── 06_Benchmark.json        # Gold-standard answers (for scoring)
├── frontend/
│   └── index.html               # The web UI (self-contained, no build step)
├── .env.example                 # Copy to .env and add your API key
├── .gitignore
└── README.md
```

The backend serves the frontend automatically, so you only run **one** command to get
the whole app running.

---

## Quick start

You need **Python 3.10 or newer** and a **Google Gemini API key**
(free from [aistudio.google.com](https://aistudio.google.com/)).

```bash
# 1. Install the backend dependencies
cd backend
pip install -r requirements.txt
cd ..

# 2. Add your API key
cp env.example .env
#    then open .env and set GEMINI_API_KEY=your-key-here
#
#    (There is also a hidden ".env.example" — same contents. Use whichever
#     your file manager shows. On Mac/Linux you can reveal hidden files with
#     Cmd+Shift+. or `ls -a`.)

# 3. Start the server (from the backend folder)
cd backend
python app.py
```

Now open **http://localhost:8001** in your browser.

> **Where does the `.env` go?** Keep it at the project root (next to `.env.example`).
> The server loads it automatically when it starts.

That's it. The page lets you either paste in a custom message or pick one of the 20
test messages, then shows the full triage decision. A second tab runs the batch
evaluation across all 20 messages and scores the agent against the benchmark.

---

## Environment variables

Set these in your `.env` file (only the first is required):

| Variable                  | Required | Default                  | What it does                                                |
|---------------------------|----------|--------------------------|-------------------------------------------------------------|
| `GEMINI_API_KEY`          | **Yes**  | —                        | Your Google Gemini API key.                                 |
| `GEMINI_MODEL`            | No       | `gemini-3.1-flash-lite`  | The Gemini model to use.                                    |
| `GEMINI_REQUEST_DELAY_S`  | No       | `4.5`                    | Seconds between calls during batch eval (free-tier pacing). |
| `PORT`                    | No       | `8001`                   | Port the backend listens on.                                |

---

## How the agent works

**One model, one call, per message.** The agent makes a single Gemini call for each
message and asks for a structured JSON object back. There is no multi-agent pipeline,
no retrieval database, and no orchestration framework — and that's a deliberate choice,
not a shortcut.

**Why a single agent?** The three reference documents (SOP, catalogue, tone guide)
total only a few thousand tokens, so they fit comfortably inside the model's context
window on every call. Splitting the work across multiple steps (classify → prioritise →
route → draft) would mean re-sending that same context several times, adding latency
and cost while creating more places for information to get dropped between steps. For a
task this size, a single well-prompted call is both simpler and more accurate.

**What's in the prompt.** The system prompt contains the full text of all three
documents, followed by an explicit 8-step decision procedure that walks the model
through the same checks a human dispatcher would make — including a detailed checklist
for the "does this need human review?" decision, which is the hardest field to get right.

**A few engineering details that matter:**

- **Structured output is enforced.** The call uses Gemini's JSON mode with a strict
  response schema, so `category` and `priority` can only ever be one of their allowed
  values. There's no fragile text parsing.
- **Time-of-day is computed in code, not by the model.** Language models are unreliable
  at turning a timestamp into "this arrived on a Saturday night." The backend works that
  out deterministically and tells the model whether the message is in or out of business
  hours, which feeds the out-of-hours rules in the SOP.
- **Low temperature (0.1).** This is a rule-following task, not a creative one, so the
  model is set to be as consistent as possible.
- **Robust to API hiccups.** The agent automatically retries on rate-limit errors with
  exponential backoff, retries once with a larger token budget if a response gets cut
  off, and gives a clear message if the daily quota is exhausted.

## Running the batch evaluation

The evaluation runs the agent across all 20 messages in `05_Inbound_Messages.json`,
compares each result to `06_Benchmark.json`, and reports accuracy.

**From the UI:** open the app and click the **Batch Evaluation** tab, then **Run All 20 Messages**.

**From the command line:**

```bash
cd backend
python evaluate.py
```

This prints a per-message breakdown and saves a full report to
`backend/outputs/evaluation_report.json`.

> **A note on timing and quota.** On the free tier, calls are spaced ~4.5 seconds apart
> to stay under the per-minute request limit, so a full run takes about 1.5 minutes.
> The free tier also has a daily request cap — if you hit it, the agent stops with a
> clear message and you can resume the next day, or enable billing on your Google Cloud
> project for higher limits.

**Scoring:**

- **Strict accuracy** — the percentage of messages where the agent matched the benchmark
  on *all four* hard fields (`category`, `priority`, `route_to`, `needs_human_review`).
- **Per-field accuracy** — the percentage correct on each field individually.
- `route_to` awards half credit when the primary team is right but a secondary "cc"
  team was missed.

### Qualitative checks (draft reply & reasoning quality)

The rubric also asks for *qualitative* checks that aren't part of the strict score but
are reported alongside it: does the draft reply hit the points it should, avoid the things
it shouldn't, and sound like Northwind rather than a generic chatbot? The evaluation covers
this in two layers.

**Deterministic tone checks (always on, free).** Every batch evaluation checks each draft
reply against the objective, mechanical rules from the tone guide — no API calls, no
subjectivity:

- no exclamation marks or emoji
- no banned phrases ("at your earliest convenience", "please rest assured", "kindly", etc.)
- ends with the "— The Northwind team" sign-off
- length is 2–4 sentences
- doesn't quote an estimate ("from") price

**LLM-as-judge (opt-in, one extra API call per message).** For the checks that need actual
understanding — "does the reply specifically reference the dripping tap?", "does it sound
like the tone guide?", "did the reasoning weigh the right rules?" — a second Gemini call
grades each draft. It scores tone fidelity and reasoning quality on a 1–5 scale and counts
how many of the benchmark's `must_include` / `must_not_include` items the draft satisfies.
It's off by default because it doubles API usage; enable it with the checkbox in the UI or
the `--judge` flag on the command line:

```bash
cd backend
python evaluate.py            # scorecard + free deterministic tone checks (default)
python evaluate.py --judge    # also run the LLM judge
```

This split is deliberate: objective rules are checked for free and deterministically, while
the genuinely subjective judgments are a clearly-labelled, optional, model-graded layer —
rather than pretending a string match can measure tone.

---

## Results

Latest run: **90% strict accuracy** (18 of 20 messages matched on all four hard fields),
with all 20 calls completing cleanly (no API errors).

| Field                | Accuracy |
|----------------------|----------|
| Category             | 100%     |
| Priority             | 95%      |
| Route                | 100%     |
| Needs human review   | 95%      |
| **Strict (all four)**| **90%**  |

Category and routing were perfect, and the two strict misses are both single-field edge
cases on messages the benchmark itself flags as debatable:

- **MSG-006 — `needs_human_review` over-flag.** The agent correctly classified this as
  `EMERGENCY` / `P1` / `Dispatch`, but *also* set `needs_human_review = true` where the
  benchmark says `false`. Every other field was right. This is the agent's one genuinely
  questionable call (discussed in the failures section).
- **MSG-017 — priority `P3` vs `P2`.** A conduct complaint on a $280 job. The strict SOP
  reading (priority by dollar amount) gives `P3`, which the agent chose; the benchmark goes
  `P2` on "upset customer." The benchmark's own notes concede "P3 is also defensible." Every
  other field was right.

Both misses are off by a single field, and in both cases the field involves a *soft* signal
(when to flag, how much weight to give an upset tone) where the source documents don't give a
clean numeric rule. The clear-cut cases — category and routing across all 20 messages — were
handled without error.

### Qualitative checks (with LLM judge enabled, all 20 messages)

| Metric                                 | Score |
|----------------------------------------|-------|
| Deterministic tone rules (pass rate)   | 100%  |
| Tone fidelity (LLM judge, 1–5)         | 4.75  |
| Reasoning quality (LLM judge, 1–5)     | 4.6   |
| `must_include` coverage (LLM judge)    | 91.1% |
| `must_not_include` violations          | 0     |

Every draft reply passed **all** the deterministic tone rules — no exclamation marks, no banned
phrases, correct sign-off, right length, no estimate ("from") prices quoted — and there were
**zero** `must_not_include` violations. The LLM judge rated tone (4.75/5) and reasoning (4.6/5)
highly. The ~9% gap in `must_include` coverage is the only soft spot worth a manual look, but
overall the draft voice held to the tone guide rather than drifting into a generic-chatbot
register.

---

## Notes on the benchmark

The exercise specifically asks us to flag cases where the benchmark is debatable rather than
quietly tuning the agent to match it. Three cases are worth calling out — and two of them are
flagged as debatable in the benchmark's *own* notes:

**MSG-017 — conduct complaint, P2 vs P3 (one of our two strict misses).**
A customer complains about a plumber's conduct on a $280 job. The SOP's priority table defines
P2 by dollar amount (complaints over $1,000), and $280 is well under that — which makes `P3`
the strict reading, and that's what the agent chose. The benchmark goes `P2` on the grounds
that the tone guide says to treat upset customers with urgency. The benchmark's notes explicitly
concede "P3 is also defensible by strict reading of SOP priority rules." This is a genuine
tension between two documents (the SOP's numeric table vs the tone guide's "acknowledge upset
customers directly"), not a clear agent error. I'd resolve it by adding an explicit SOP rule:
"conduct complaints where the customer is upset → P2."

**MSG-004 — billing dispute routing (agent got this right this run).**
The SOP says to cc Accounts on billing disputes *over $500*. The disputed amount here is $150,
below that threshold, so by the strict letter of the SOP the cc to Accounts shouldn't trigger.
The benchmark routes to "Customer Care + Accounts" anyway — defensible, since the customer also
threatens an online review, which is itself a flag trigger. The agent matched the benchmark
here, but the underlying rule is genuinely ambiguous: the $500 cc-threshold doesn't strictly
fire, and the benchmark's own notes say "either is defensible."

**MSG-008 — bathroom renovation, whether to flag for human review (agent got this right this run).**
The human-review rule triggers on quotes over $5,000. The catalogue lists bathroom reno plumbing
at "from $4,500" — below the line. A strict reading says don't flag; a practical read says these
jobs routinely exceed $5,000, so flag. The benchmark flags it and the agent agreed, but the rule
and the catalogue genuinely pull in different directions: the catalogue *floor* is under the
threshold while the realistic *final cost* is over it.

The broader takeaway: the SOP, catalogue, and tone guide were written by different people for
different purposes, and they disagree at the edges — especially around when a soft signal (an
upset customer, a job likely to exceed a threshold) should override a hard numeric rule. The
agent handles the clear-cut cases reliably; the remaining disagreements are exactly the cases a
human reviewer should see, which is what `needs_human_review` is for.

## Where the agent actually failed

Setting aside the benchmark disagreements above, there's one miss that is genuinely the agent's
own:

**MSG-006 — over-flagging a clear emergency.**
"No hot water, two kids, freezing" in a Sydney June is a textbook EMERGENCY / P1, and the agent
got category, priority, and routing all right. But it also set `needs_human_review = true`, where
the benchmark says `false`. The human-review checklist in the prompt is deliberately cautious
("when in doubt, flag"), and that caution occasionally fires on cases that are urgent but not
*ambiguous*. The fix is a prompt clarification: a clean EMERGENCY that matches one of the SOP's
named examples is not, by itself, a reason to flag — flagging is for genuine ambiguity and policy
edges, not for urgency that the priority field already captures.

### Future Improvements

I'd focus on turning the agent from a strong demo system into something that could operate reliably in a real support workflow. The biggest step would be adding a lightweight feedback loop: tracking which triage decisions staff corrected, which drafted replies customers responded well to, and which cases consistently escalated to humans. That would make it possible to evaluate the agent against real operational outcomes instead of only a fixed benchmark.

I'd also tighten the needs_human_review logic around edge cases like `MSG-006`. Right now the prompt intentionally errs on the side of caution, but that sometimes causes the agent to flag messages that are urgent yet completely unambiguous. Refining those rules — especially around “clear emergency vs genuine ambiguity” — would improve precision without reducing safety.
