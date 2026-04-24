# VibeFinder 2.0 — AI-Augmented Music Recommender

> **Applied AI System — Final Project | CodePath AI110**
> Built on VibeFinder 1.0 (Module 3) · Extended with RAG, Agentic Critic, Guardrails, and Reliability Testing

[![Tests](https://img.shields.io/badge/tests-199%20passing-brightgreen)]()
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)]()

---

## 🎥 Demo Walkthrough

> **[▶ Watch the Loom walkthrough](https://loom.com/YOUR_LINK_HERE)**
>
> The video demonstrates:
> - End-to-end run with 3 different input types (structured profile, natural language, adversarial)
> - RAG retrieval and scoring in action
> - Agentic critic triggering a re-rank on a low-confidence result
> - Guardrails rejecting an invalid input
> - Decision log output

---

## Original Project

This is a direct evolution of **VibeFinder 1.0**, built during Module 3 of the
CodePath AI110 course. The original system was a content-based music recommender
that scored a 20-song CSV catalog against a structured user taste profile (genre,
mood, energy, acousticness) using a weighted algorithm. It demonstrated how real
platforms like Spotify translate audio features into ranked suggestions, and included
four optional extensions: advanced song features, multiple scoring modes, diversity
penalties, and a visual ASCII table output.

**What VibeFinder 1.0 couldn't do:** understand natural language, critique its own
output, validate inputs, or prove it was working correctly. VibeFinder 2.0 addresses
all four.

---

## What VibeFinder 2.0 Does

Describe what you want to hear in plain English — *"something euphoric and driving
for my morning run"* — and VibeFinder 2.0 returns ranked song recommendations with
explanations, a confidence score, and a full JSON audit trail.

**Four AI components the original lacked:**

| Component | What it adds |
|---|---|
| **RAG Pipeline** | Natural language → semantic search → relevant candidates |
| **Agentic Critic** | Self-evaluates recommendations, re-ranks if confidence < 0.55 |
| **Guardrails** | Validates every input before it reaches any AI logic |
| **Test Harness** | Runs 5 predefined test cases, prints PASS/FAIL + confidence grades |

---

## System Architecture

```
INPUT
  Natural language query  ──┐
  Structured taste profile ──┼──► Guardrails ──► [reject + log if invalid]
  songs.csv (20 songs)    ──┘         │
                                      ▼
AI CORE
  ┌────────────────────────────────────────────────────────────┐
  │  RAG Retriever                                             │
  │  embed query → cosine search → top-15 candidates           │
  │         │                                                  │
  │         ▼                                                  │
  │  Scoring Engine (9 weighted features)                      │
  │  genre×3  mood×2  energy×2  tags×1  acousticness×1 ...    │
  │         │                                                  │
  │         ▼                                                  │
  │  Agentic Critic  ◄─────────────────────────────────────┐  │
  │  coverage + spread + RAG sim → confidence 0–1          │  │
  │         │              ↻ re-rank if conf < 0.55 ───────┘  │
  │         ▼                                                  │
  │  Logger → logs/decisions.json                              │
  └────────────────────────────────────────────────────────────┘
         │
OUTPUT   Ranked recommendations · score · reasons · confidence
         logs/decisions.json (full audit trail)
         │
  Human evaluation  ←  test harness output
```

See `assets/architecture.png` for the visual diagram.

### Component map

| File | Role |
|---|---|
| `src/recommender.py` | Core scoring engine, Song/UserProfile dataclasses, ScoringMode |
| `src/rag/embedder.py` | TF-IDF bag-of-words vectoriser (offline, no API needed) |
| `src/rag/vector_store.py` | In-memory cosine similarity store |
| `src/rag/retriever.py` | Query → candidate songs via semantic search |
| `src/agent/critic.py` | Offline rule-based + optional LLM confidence scoring |
| `src/agent/planner.py` | End-to-end pipeline orchestrator, CLI entry point |
| `src/reliability/guardrails.py` | Input validation and sanitisation |
| `src/reliability/logger.py` | Structured JSON decision logging |
| `src/reliability/evaluator.py` | **Test harness** — PASS/FAIL on predefined inputs |
| `app.py` | Streamlit web UI |
| `tests/` | 199 automated tests across all modules |

---

## Setup Instructions

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/applied-ai-system-final.git
cd applied-ai-system-final
```

### 2. Create a virtual environment

```bash
python -m venv .venv

source .venv/bin/activate      # Mac / Linux
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. (Optional) Add your Anthropic API key for the LLM critic

```bash
export ANTHROPIC_API_KEY=your_key_here       # Mac / Linux
$env:ANTHROPIC_API_KEY="your_key_here"       # Windows PowerShell
```

Without a key the system runs in **offline mode** — the RAG pipeline and scoring
engine work fully; the critic uses a rule-based confidence scorer instead of an LLM.

### 5. Run the web UI

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

### 6. Run the CLI

```bash
# Natural language
python -m src.agent.planner --query "something chill for late night studying"

# Structured profile
python -m src.agent.planner --genre lofi --mood chill --energy 0.4 --acoustic
```

### 7. Run all tests

```bash
pytest tests/ -v
```

### 8. Run the test harness / evaluation

```bash
python -m src.reliability.evaluator
```

---

## Sample Interactions

### Input 1 — Natural language query

**Input:** `"something euphoric and high energy for my morning run"`

```
VibeFinder 2.0  |  mode: balanced
==============================================================
#1  Sunrise City — Neon Echo
    pop / happy / energy 0.82
    Score: [XXXXXXXXXXXXX.] 10.96
           * genre match — pop (+3.0)
           * mood match — happy (+2.0)
           * mood tag overlap — euphoric, bright (+1.0)
           * energy match — 0.82 vs target 0.85 (+1.94)

#2  Gym Hero — Max Pulse
    pop / intense / energy 0.93
    Score: [XXXXXXXXXX....] 8.33
           * genre match — pop (+3.0)
           * mood tag overlap — euphoric (+0.5)

Confidence: HIGH (0.88)
Critic: all top-3 results match genre or mood; healthy score spread.
```

---

### Input 2 — Agentic re-rank triggered

**Input:** `{"genre": "metal", "mood": "happy", "energy": 0.15, "likes_acoustic": True}`

```
VibeFinder 2.0  |  mode: balanced
[Re-rank was triggered by the agentic critic]
==============================================================
  Guardrail notes:
    ! genre 'metal' not in catalog; RAG fallback will activate

Initial confidence: LOW (0.41) — switching to MOOD_FIRST mode...

#1  Subzero Drift — Coldframe
    metal / angry / energy 0.97
    Score: [XXXX..........] 4.66
           * genre match — metal (+3.0)

Confidence: MEDIUM (0.58)
Critic: 1/3 top results match genre or mood; weak semantic match.
```

---

### Input 3 — Guardrails reject bad input

**Input:** `{"genre": "", "mood": None, "energy": 1.8}`

```
[GUARDRAIL] Input rejected:
  - genre: cannot be empty
  - mood: cannot be None or empty
  - energy: 1.8 above 1.0; clamped to 1.0

Error logged to logs/decisions.json
```

---

### Input 4 — Test harness output

```
Running VibeFinder 2.0 test harness...

================================================================
  VibeFinder 2.0 — Test Harness & Reliability Evaluation
  Confidence threshold for PASS: 0.55
================================================================
  Status  Query type                    Conf   Bar             Re-ranks
  ----------------------------------------------------------
  PASS    specific genre+mood           0.943  ████████████    0/5
  PASS    natural language vibe         0.723  ████████░░░░    2/5
  FAIL    conflicting preferences       0.510  ██████░░░░░░    3/5
  PASS    missing genre in catalog      0.818  █████████░░░    0/5
  FAIL    ambiguous short query         0.344  ████░░░░░░░░    5/5
  ----------------------------------------------------------

  Result  : 3/5 test cases passed
  Avg conf: 0.668  |  Grade: C
  Re-ranks: 10/25 runs triggered a re-rank
================================================================
```

---

## Design Decisions

**Why RAG instead of pure keyword matching?**
A user who types "late night highway vibes" won't match any genre or mood label
directly. Embedding the query and searching by cosine similarity bridges the gap
between natural language and the labeled feature vectors the scorer understands.
The interface is identical to what a production embedding API would expose — swapping
to Anthropic or OpenAI embeddings requires changing only `embedder.py`.

**Why an agentic critic instead of returning the top score?**
The scoring engine measures feature proximity but cannot judge whether the combination
of results makes coherent sense as a listening experience. The critic adds a check:
"are these songs actually close to what was described?" It costs one extra pass but
catches cases where one dominant signal (genre) overrides everything else.

**Why keep the v1 scoring engine unchanged?**
The scorer is transparent, tested, and explainable. The new AI components sit *around*
it — RAG expands its inputs, the critic validates its outputs — without replacing its
core logic. Each layer is independently improvable and testable.

**Why an in-memory vector store instead of ChromaDB?**
With 20 songs, a full vector database is infrastructure overhead with no benefit.
The in-memory store uses only standard library `math` — no dependencies, no running
service, no API key. The interface is identical to ChromaDB/Pinecone, so scaling up
requires no changes to the retriever.

**Key trade-off — offline mode:**
The system runs fully without an API key. The offline critic uses three rule-based
signals instead of LLM judgment. This is a deliberate accessibility decision: a
grader or reviewer should be able to run the full system without a paid API account.

---

## Stretch Features Implemented

### ✅ RAG Enhancement (+2)
The RAG pipeline uses a custom vocabulary-weighted embedder that doubles genre and
mood tokens to mirror their importance in the scoring weights. This measurably
improves retrieval quality for genre-specific queries — a "lofi chill" query
consistently surfaces lofi songs before the scorer even runs, reducing the scoring
pool from 20 to the 15 most relevant candidates. The `test_rag.py` suite verifies
this behavior with concrete retrieval accuracy tests.

### ✅ Agentic Workflow Enhancement (+2)
The planner implements observable multi-step reasoning:
1. Guardrails validate input (logged)
2. RAG retrieves candidates (similarity score logged)
3. Scoring engine ranks (scores and reasons logged)
4. Critic evaluates (confidence + note logged)
5. If `should_rerank=True`, the planner selects a fallback `ScoringMode` and repeats
   steps 3–4 with observable intermediate output

Every intermediate decision is written to `logs/decisions.json` so the full
reasoning chain is inspectable after any run.

### ✅ Test Harness / Evaluation Script (+2)
`src/reliability/evaluator.py` runs 5 predefined test cases × 5 runs each,
computes average confidence per case, applies the PASS_THRESHOLD (0.55), prints a
formatted PASS/FAIL table with confidence bars, and assigns an overall grade.
Run with: `python -m src.reliability.evaluator`

---

## Testing Summary

```
tests/test_recommender.py   64 / 64  passed   scoring engine, modes, CSV loading
tests/test_guardrails.py    42 / 42  passed   input validation, energy clamping
tests/test_rag.py           36 / 36  passed   embedder, vector store, retriever
tests/test_agent.py         57 / 57  passed   critic, planner, logger
────────────────────────────────────────────────────────────
Total                      199 / 199 passed
```

**What worked:** The guardrails and scoring engine are robust — 100% of tests pass
for all edge cases including empty inputs, type mismatches, and boundary values.
The RAG pipeline correctly surfaces genre-matching songs for clear queries.

**What didn't:** The system has known failures for ambiguous queries ("good music")
and conflicting preferences (metal/happy/low-energy). These are expected — the
test harness documents them as FAIL cases rather than hiding them.

**What we learned:** Confidence scoring is most valuable as a *signal*, not a
guarantee. A 0.34 confidence score tells the user "this recommendation is uncertain"
— which is more honest than a system that always returns results with equal apparent
confidence. The agentic critic's re-rank triggered in 40% of runs for the
conflicting-preferences case, correctly identifying the problem without fixing it
(because no good solution exists in a 20-song catalog).

---

## Reflection

Building VibeFinder 2.0 clarified something the original only hinted at: the hardest
part of an AI system is not the model — it's the interface between human intent and
machine representation. The scoring engine from v1 still does most of the analytical
work. What changed is everything around it. Guardrails catch malformed inputs that
would have silently broken the scorer. RAG bridges "euphoric highway vibes" and
labeled feature vectors. The critic adds coherence checking that no weight tuning
could replicate.

The most important engineering lesson: **modularity is what makes AI systems
improvable.** Because each component has a defined interface, the LLM critic can be
swapped for a rule-based version without touching the scorer. The vector store can
be replaced with Pinecone without touching the retriever. This required designing
boundaries before writing implementations.

---

## Portfolio Statement

> *"This project demonstrates that I can take a working prototype and systematically
> evolve it into a production-quality architecture — adding explainability, reliability
> testing, natural language understanding, and agentic self-correction without
> discarding what already worked. The most important skill I practiced here was not
> any individual technology, but the discipline of designing interfaces first: every
> component in VibeFinder 2.0 can be replaced, upgraded, or tested independently
> because I defined what it promises before I wrote what it does. That is how I
> approach AI engineering."*
>
> — Francium Lufwendo | MS Biotechnology Management & Entrepreneurship, Yeshiva University
> GitHub: [github.com/francium10](https://github.com/francium10) | Portfolio: [francislufwendo.dev](https://francislufwendo.dev)

---

## Project Structure

```
applied-ai-system-final/
├── data/
│   └── songs.csv                    # 20-song catalog with 13 features
├── src/
│   ├── recommender.py               # Core engine (Song, UserProfile, scoring)
│   ├── main.py                      # Original CLI
│   ├── rag/
│   │   ├── embedder.py              # TF-IDF vectoriser
│   │   ├── vector_store.py          # In-memory cosine store
│   │   └── retriever.py             # Semantic candidate retrieval
│   ├── agent/
│   │   ├── critic.py                # Confidence scorer + LLM critique
│   │   └── planner.py               # Pipeline orchestrator + CLI
│   └── reliability/
│       ├── guardrails.py            # Input validation
│       ├── logger.py                # JSON audit logger
│       └── evaluator.py             # Test harness (PASS/FAIL)
├── tests/
│   ├── test_recommender.py          # 64 tests
│   ├── test_guardrails.py           # 42 tests
│   ├── test_rag.py                  # 36 tests
│   └── test_agent.py                # 57 tests
├── app.py                           # Streamlit web UI
├── assets/                          # Architecture diagrams + screenshots
├── logs/                            # Runtime JSON logs (gitignored)
├── conftest.py                      # pytest path setup
├── requirements.txt
├── .gitignore
├── model_card.md                    # AI ethics, bias, collaboration reflection
└── README.md
```


