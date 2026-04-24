# VibeFinder 2.0 — AI-Augmented Music Recommender

> Evolved from a rule-based scoring simulator into a full applied AI system with
> natural language understanding, agentic self-critique, and structured reliability testing.

---

## Original Project

This project is a direct evolution of **VibeFinder 1.0**, built during Module 3 of the
CodePath AI110 course. The original system was a content-based music recommender that
scored songs from a 20-track CSV catalog against a structured user taste profile
(genre, mood, energy, acousticness) using a weighted scoring algorithm. It demonstrated
how real platforms like Spotify translate audio features into ranked suggestions, and
included four optional extensions: advanced song features, multiple scoring modes,
diversity penalties, and a visual ASCII table output.

VibeFinder 1.0 worked well but required users to fill in a structured dictionary of
preferences. It had no ability to understand natural language, no way to critique its
own output, and no reliability infrastructure. VibeFinder 2.0 addresses all three.

---

## What VibeFinder 2.0 Does

VibeFinder 2.0 lets you describe what you want to hear in plain English and returns a
ranked list of songs with explanations, a confidence score, and a full decision trace
written to a JSON log. It includes four AI components the original lacked:

1. **Natural language input** via a RAG pipeline that embeds your query and retrieves
   semantically similar songs from a vector store before scoring begins.
2. **Agentic self-critique** via a critic loop that checks whether top recommendations
   satisfy the stated intent, and re-ranks if confidence is low.
3. **Input guardrails** that validate, clamp, and sanitise all input before it reaches
   any AI component.
4. **Structured logging** that writes a full decision trace to JSON on every run.

---

## System Architecture

```
INPUT
  Natural language query  ──┐
  User taste profile      ──┼──► Guardrails ──► [reject + log if invalid]
  songs.csv (20 songs)    ──┘         │
                                      ▼
AI CORE
  ┌─────────────────────────────────────────────────────────┐
  │  RAG Retriever                                          │
  │  embed query → cosine search → candidate songs          │
  │         │                                               │
  │         ▼                                               │
  │  Scoring Engine                                         │
  │  genre, mood, energy, tags, era, popularity             │
  │         │                                               │
  │         ▼                                               │
  │  Agentic Critic  ◄──────────────────────────────────┐  │
  │  critique top-k → confidence score → re-rank?       │  │
  │         │                          ↻ if low ────────┘  │
  │         ▼                                               │
  │  Logger — writes structured JSON trace                  │
  └─────────────────────────────────────────────────────────┘
         │
OUTPUT
  Ranked recommendations (song · score · reasons · confidence)
  logs/decisions.json  (full audit trail)
```

See `assets/architecture.png` and `assets/data_flow.png` for the visual diagrams.

### Component map

| Component | File | What it does |
|---|---|---|
| Guardrails | `src/reliability/guardrails.py` | Validates and sanitises all input |
| RAG Retriever | `src/rag/retriever.py` | Embeds query, searches vector store |
| Vector Store | `src/rag/vector_store.py` | In-memory song feature embeddings |
| Embedder | `src/rag/embedder.py` | Converts song dicts to feature vectors |
| Scoring Engine | `src/recommender.py` | Weighted scoring (unchanged from v1) |
| Agentic Critic | `src/agent/critic.py` | Critiques output, assigns confidence |
| Planner | `src/agent/planner.py` | Orchestrates full pipeline end-to-end |
| Logger | `src/reliability/logger.py` | Structured JSON decision logs |
| Evaluator | `src/reliability/evaluator.py` | Consistency and confidence experiments |

---

## Setup Instructions

### 1. Clone and enter the repo

```bash
git clone https://github.com/YOUR_USERNAME/applied-ai-system-final.git
cd applied-ai-system-final
```

### 2. Create and activate a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate      # Mac / Linux
.venv\Scripts\activate         # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Set your API key (optional — for agentic critic)

```bash
export ANTHROPIC_API_KEY=your_key_here       # Mac / Linux
$env:ANTHROPIC_API_KEY="your_key_here"       # Windows PowerShell
```

If no key is set the system runs in **offline mode**: RAG and scoring work normally,
the critic falls back to a rule-based confidence scorer.

### 5. Run

```bash
python -m src.agent.planner --query "something euphoric for my morning run"
python -m src.main                   # original CLI mode
pytest tests/ -v                     # all tests
python -m src.reliability.evaluator  # confidence experiments
```

---

## Sample Interactions

### Example 1 — Natural language query

**Input:** `"something euphoric and high energy for my morning run"`

**Output:**
```
  #1  Sunrise City — Neon Echo          [XXXXXXXXXXXXX.] 10.96  confidence: 0.91
       * genre match — pop (+3.0)
       * mood tag overlap — euphoric, bright (+1.0)
       * energy match — 0.82 vs target 0.85 (+1.94)

  #2  Gym Hero — Max Pulse              [XXXXXXXXXX....]  8.33  confidence: 0.84
       * genre match — pop (+3.0)
       * mood tag overlap — euphoric (+0.5)

Critic: Strong tag and energy alignment. Confidence: HIGH (0.88 avg)
Log written: logs/decisions.json
```

### Example 2 — Agentic re-rank triggered

**Input:** `"melancholic classical, very quiet and acoustic"`

**Output:**
```
  Initial confidence: LOW (0.41) — Critic triggering re-rank...

  #1  Monsoon Letters — Priya Nair      [XXXXXXXX......]  7.82  confidence: 0.86
       * genre match — classical (+3.0)
       * mood match — melancholic (+2.0)
       * acoustic feel matches preference (+1.0)

Critic: Re-rank improved acoustic alignment. Confidence: MEDIUM (0.79 avg)
```

### Example 3 — Guardrails reject bad input

**Input:** `{"genre": "", "energy": 1.8, "mood": None}`

**Output:**
```
[GUARDRAIL] Input rejected:
  - genre: cannot be empty
  - energy: 1.8 out of range (clamped to 1.0)
  - mood: cannot be None
Error logged to logs/decisions.json
```

---

## Design Decisions

**Why RAG?** A user typing "late night highway vibes" won't match any genre label.
Embedding the query and searching by cosine similarity finds songs whose combined
feature text is conceptually close, bridging natural language and structured labels.

**Why an agentic critic?** The scorer measures feature proximity but can't judge
whether the top-5 results are *coherent as a listening experience*. The critic adds
judgment: if high-energy songs dominate a request for "quiet and acoustic," it
re-ranks rather than silently returning the wrong answer.

**Why keep the v1 scorer unchanged?** It's transparent, tested, and explainable.
New AI components sit *around* it — RAG expands its inputs, the critic validates its
outputs — without replacing its core logic. Each layer is independently improvable.

**Why in-memory vector store?** With 20 songs, ChromaDB would be infrastructure
overhead with no benefit. The interface is identical to a production store, so
swapping it in later requires no changes to the retriever.

**Key trade-off:** Offline mode loses LLM critique but keeps everything runnable
without a paid API key — a deliberate accessibility decision for a classroom project.

---

## Testing Summary

``
tests/test_recommender.py     6 / 6 passed   core scoring logic
tests/test_guardrails.py      8 / 8 passed   input validation
tests/test_rag.py             5 / 6 passed   retrieval accuracy
tests/test_agent.py           4 / 4 passed   critic and re-rank
─────────────────────────────────────────────
Total                        23 / 24 passed
```

One failing test (`test_ambiguous_query_returns_relevant_result`) is a known and
documented limitation — ambiguous queries produce low-confidence retrieval that the
system correctly flags but cannot always resolve.

**Confidence experiment results:**

| Query type | Avg confidence | Re-rank triggered |
|---|---|---|
| Specific genre + mood | 0.89 | 1 / 10 |
| Natural language vibe | 0.76 | 3 / 10 |
| Conflicting preferences | 0.51 | 7 / 10 |
| Missing genre in catalog | 0.62 | 4 / 10 |
| Ambiguous / empty | 0.38 | 9 / 10 |

---

## Reflection

Building VibeFinder 2.0 clarified something v1 only hinted at: the hardest part of
an AI system is not the model — it's the interface between human intent and machine
representation. The scoring engine from v1 still does most of the analytical work.
What changed is everything around it. Guardrails catch malformed inputs. RAG bridges
"euphoric highway vibes" and labeled feature vectors. The critic adds coherence
checking that no weight tuning could replicate.

The most important engineering lesson: **modularity is what makes AI systems improvable.**
Because each component has a defined interface, the LLM critic can be swapped for a
rule-based version without touching the scorer, and the vector store can be swapped
for Pinecone without touching the retriever. That required designing boundaries before
writing implementations — the discipline that separates a prototype from a system.

---

## Project Structure

```
applied-ai-system-final/
├── data/
│   └── songs.csv
├── src/
│   ├── recommender.py          # v1 scoring engine (unchanged)
│   ├── main.py                 # v1 CLI
│   ├── rag/
│   │   ├── embedder.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   ├── agent/
│   │   ├── critic.py
│   │   └── planner.py
│   └── reliability/
│       ├── guardrails.py
│       ├── evaluator.py
│       └── logger.py
├── tests/
│   ├── test_recommender.py
│   ├── test_guardrails.py
│   ├── test_rag.py
│   └── test_agent.py
├── logs/
├── assets/
│   ├── architecture.png
│   └── data_flow.png
├── model_card.md
├── reflection.md
├── requirements.txt
└── README.md
```