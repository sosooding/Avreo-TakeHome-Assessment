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
cp .env.example .env
#    then open .env and set GEMINI_API_KEY=your-key-here

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

---

## API reference

The backend exposes these endpoints (interactive docs are available at
`http://localhost:8001/docs` once the server is running):

| Method & path   | What it does                                                       |
|-----------------|--------------------------------------------------------------------|
| `POST /triage`  | Triage a single message. Send the message in the body, get the decision back as JSON. |
| `POST /evaluate`| Run the agent across all 20 benchmark messages and return the scored report. |
| `GET /messages` | Return the 20 test messages (used by the UI's test panel).         |
| `GET /benchmark`| Return the gold-standard benchmark decisions.                      |
| `GET /health`   | Health check — confirms the agent initialised correctly.           |
| `GET /`         | Serves the frontend UI.                                            |

**Example — triage one message:**

```bash
curl -X POST http://localhost:8001/triage \
  -H "Content-Type: application/json" \
  -d '{
    "channel": "email",
    "sender_name": "Sarah Patel",
    "subject": "Dripping tap",
    "body": "The cold tap in our ensuite has been dripping for a week. Can you book someone? We are in Mosman."
  }'
```

Response:

```json
{
  "category": "BOOKING",
  "priority": "P3",
  "route_to": "Dispatch",
  "draft_reply": "Hi Sarah — got your message about the dripping tap in the ensuite. We'll have someone call you back within the day to lock in a time. — The Northwind team",
  "needs_human_review": false,
  "reasoning": "Customer is asking to book a known service (tap repair). No safety risk, so P3. Routes to Dispatch per SOP. Nothing triggers a human-review flag."
}
```

---

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

---

## Results

Latest run: **95% strict accuracy** (19 of 20 messages matched on all four hard fields).

| Field                | Accuracy |
|----------------------|----------|
| Category             | 100%     |
| Priority             | 95%      |
| Route                | 100%     |
| Needs human review   | 100%     |
| **Strict (all four)**| **95%**  |

The single strict miss is **MSG-017**, where the agent chose priority `P3` and the
benchmark expects `P2` — and this is one of the benchmark's own deliberately debatable
cases (see below). Every other field on that message was correct.

---

## Notes on the benchmark

The exercise specifically asks us to flag cases where the benchmark is debatable rather
than quietly tuning the agent to match it. A few worth calling out:

**MSG-017 — conduct complaint, P2 vs P3 (our one strict miss).**
A customer complains about a plumber's conduct on a $280 job. The SOP's priority table
defines P2 by dollar amount (complaints over $1,000), and $280 is well under that — which
makes `P3` the strict reading, and that's what the agent chose. The benchmark goes `P2`
on the grounds that the tone guide says to treat upset customers with urgency. Both are
defensible; the benchmark's own notes acknowledge "P3 is also defensible by strict reading
of SOP priority rules." This is a genuine tension between two documents, not an agent error.

**MSG-008 — bathroom renovation, whether to flag for human review.**
The human-review rule triggers on quotes over $5,000. The catalogue lists bathroom reno
plumbing at "from $4,500" — below the line. A strict reading says don't flag; a practical
read says these jobs usually blow past $5,000, so flag. The benchmark flags it, and the
agent now agrees, but the rule and the catalogue genuinely pull in different directions here.

**MSG-004 — billing dispute routing.**
The SOP says to cc Accounts on billing disputes over $500. The disputed amount here is
$150, below that threshold, yet the benchmark routes to "Customer Care + Accounts" anyway.
The agent matches the benchmark, but strictly speaking the $500 cc-rule doesn't trigger.

The broader takeaway: the SOP, catalogue, and tone guide were written by different people
for different purposes, and they occasionally disagree at the edges — especially around
when a soft signal (an upset customer, a job likely to exceed a threshold) should override
a hard numeric rule. The agent handles the clear-cut cases reliably; the remaining
disagreements are exactly the cases a human reviewer should see, which is what the
`needs_human_review` flag is for.

### If I had another day

I'd close the loop on the drafted replies. Right now the agent writes a reply but nothing
happens to it. I'd add an endpoint that sends an approved draft (via an email provider),
captures the customer's response, and records whether the thread resolved without a human
having to step in. After a few hundred real messages, that gives you genuine production
data on which draft styles actually work — far more useful for improving the agent than
a 20-message benchmark.
