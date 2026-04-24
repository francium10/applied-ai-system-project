# 🎧 Model Card: VibeFinder 2.0


## 1. Model Name

**VibeFinder 2.0** — AI-Augmented Music Recommender

An evolution of VibeFinder 1.0 (Module 3), extended with a RAG retrieval pipeline,
an agentic self-critique loop, structured input guardrails, and a full reliability
test harness.

---

## 2. Goal / Task

VibeFinder 2.0 answers one question: *"Given how a listener describes what they want
to hear — in plain English or as a structured profile — which songs best match their
taste right now?"*

It does this through a five-stage pipeline:
1. Validate input through guardrails
2. Retrieve semantically relevant songs via RAG
3. Rank candidates with a weighted scoring engine
4. Critique output quality with an agentic self-check
5. Log the full decision trace for auditability

The system is not a prediction model — it does not predict whether you will enjoy a
song. It measures how closely a song's features align with a described taste. That
distinction matters: predictions require feedback loops (skips, replays, saves) that
this classroom system intentionally omits.

---

## 3. Data Used

**Catalog:** 20 songs in `data/songs.csv`

| Feature | Type | What it captures |
|---|---|---|
| genre | string | Broad category (pop, lofi, rock, etc.) |
| mood | string | Emotional tone (happy, chill, intense, etc.) |
| energy | float 0–1 | Track intensity/drive |
| tempo_bpm | float | Pace in beats per minute |
| valence | float 0–1 | Musical positiveness |
| danceability | float 0–1 | Rhythmic suitability |
| acousticness | float 0–1 | Acoustic vs. electronic ratio |
| popularity | float 0–100 | Chart performance score |
| release_decade | string | Era (e.g. "2020s") |
| mood_tags | string | Pipe-separated fine-grained descriptors |

**Genres covered:** pop, lofi, rock, indie pop, ambient, jazz, synthwave, electronic,
r&b, classical, funk, folk, gospel, metal, latin, dream pop (16 total)

**Known data limits:**
- 20 songs is a toy catalog. A real platform has millions.
- Genre representation is uneven (lofi: 3 songs, metal/classical: 1 each).
- No lyrics, language, cultural context, or artist identity signals.
- The curator's taste skews toward contemporary electronic and Western pop.

---

## 4. Algorithm Summary

**Stage 1 — Guardrails:** All input is validated before any AI component runs.
Empty fields are rejected. Out-of-range energy values are clamped. Unknown types
are refused with a clear error.

**Stage 2 — RAG Retrieval:** The query is converted to a TF-IDF bag-of-words vector
over a 70-token vocabulary of genres, moods, and tags. The vector store finds the 15
most semantically similar songs by cosine similarity. Genre and mood are double-weighted
in the feature string to mirror their importance in scoring.

**Stage 3 — Scoring Engine:** Nine weighted features score each candidate:

| Feature | Weight | Method |
|---|---|---|
| Genre match | 3.0 | Binary categorical |
| Mood match | 2.0 | Binary categorical |
| Energy proximity | 2.0 | `(1 - abs(target - actual))` |
| Mood tag overlap | 1.0 | Overlap ratio |
| Acousticness | 1.0 | Boolean alignment |
| Valence | 1.0 | Emotional clarity proximity |
| Danceability | 0.5 | Proportional |
| Popularity | 0.5 | Normalised 0–1 |
| Era match | 0.5 | Binary categorical |

**Stage 4 — Agentic Critic:** Three signals produce a confidence score (0–1):
genre/mood coverage in top-3, score spread between #1 and #2, and RAG similarity.
If confidence < 0.55, a re-rank triggers using a different ScoringMode preset.

**Stage 5 — Logger:** Every run writes a JSON audit record.

---

## 5. Observed Behavior and Biases

**What works well:** Clean profiles with well-represented genres (lofi, pop, rock)
consistently return results that feel intuitively correct. The agentic critic
correctly detects ambiguous queries and flags them as low-confidence rather than
silently returning poor results.

**Genre dominance bias:** Genre carries 3× the weight of any other feature.
A genre-matching song with the wrong mood will outscore a wrong-genre song with
perfect mood and energy. A "happy pop" user can receive an "intense pop" song at #2
because the genre label overrides the mood mismatch entirely.

**Filter bubble:** Because genre dominates, users almost never see cross-genre
recommendations even when a song in a different genre would be emotionally closer
to what they described.

**Single-song genre trap:** With only one metal song in the catalog, any metal user
receives that song regardless of mood or energy mismatch. A user requesting
"metal / happy / low energy" receives the only metal track — which is angry and
high-energy — because the 3.0 genre bonus cannot be overcome.

**Catalog skew:** Any genre with one song gives users no real diversity. Genres
absent from the catalog (e.g., bossa nova) silently fall back to mood and energy
matching without informing the user.

---

## 6. Evaluation Process and Testing Results

**Test harness results** (run with `python -m src.reliability.evaluator`):

| Status | Query type | Avg confidence | Re-ranks |
|---|---|---|---|
| PASS | specific genre+mood | 0.943 | 0/5 |
| PASS | natural language vibe | 0.723 | 2/5 |
| FAIL | conflicting preferences | 0.510 | 3/5 |
| PASS | missing genre in catalog | 0.818 | 0/5 |
| FAIL | ambiguous short query | 0.344 | 5/5 |

**Summary:** 3/5 test cases passed. Average confidence 0.668. Grade: C.
The system handles clear, well-specified queries reliably (confidence > 0.9)
but struggles with conflicting preferences and ambiguous language — which is
expected and documented behavior, not a silent failure.

**Automated tests:** 199/199 passing across four test files covering the scoring
engine, guardrails, RAG pipeline, and agentic components.

**What surprised me:** The "missing genre in catalog" case (bossa nova) scored
higher than the "conflicting preferences" case (metal/happy). A completely absent
genre forced the system to fall back gracefully on mood and energy, while a
*partially matching* genre produced a worse result by letting one signal dominate.
One bad match is worse than no match.

---

## 7. Intended Use and Non-Intended Use

**This system IS designed for:**
- Learning how content-based filtering and RAG pipelines work
- Classroom exploration of algorithmic bias, filter bubbles, and confidence scoring
- Portfolio demonstration of modular AI system design

**This system IS NOT designed for:**
- Serving real listeners on a production platform
- Catalogs larger than a few hundred songs (the scoring loop is O(n))
- Adapting to user feedback — it has no memory of skips or replays
- Representing diverse global musical culture

---

## 8. Ethics: Misuse, Limitations, and Responsible Design

**Could this system be misused?**

Yes, in two ways:

1. **Bias amplification at scale:** If deployed with real users, the genre
   dominance weight would systematically under-serve listeners with cross-genre
   tastes or niche preferences (metal, classical, folk). At scale, this would
   reduce discovery and reinforce cultural homogeneity in what gets recommended.
   Prevention: replace hardcoded weights with learned weights from user feedback
   (skip rates, replay rates), and add diversity injection to the re-ranking step.

2. **Profile inference:** The system accepts structured taste profiles that could
   theoretically be used to build listener profiles without consent. Even though
   this version has no user accounts or storage, the logging system writes every
   query to disk. In a real deployment this would require explicit data consent,
   retention limits, and anonymisation.
   Prevention: the current logger is opt-in offline only. A production version
   would need a privacy layer.

**What would make this system more responsible:**
- Explicit diversity requirements in the ranking step
- A "why not" explanation showing what songs were *not* recommended and why
- Confidence score displayed to users so they know when results are uncertain
- Periodic catalog audits for demographic representation gaps

---

## 9. Personal Reflection and AI Collaboration

**Biggest learning moment:**
The weight-shift experiment was the most clarifying moment. Changing genre weight
from 3.0 to 1.5 changed 4 of 5 recommendations for the same user. The algorithm
didn't change. The data didn't change. One editorial decision changed, and the
output looked meaningfully different. In a weighted scoring system, the weights
*are* the designer's values encoded as numbers.

**What surprised me about simple algorithms feeling like recommendations:**
This system has no understanding of music whatsoever. It knows labels and floats.
And yet when labels and floats align, the output feels like a thoughtful suggestion.
The conviction comes from pattern alignment; the brittleness comes from having no
model of what the patterns mean. That gap is why real platforms add collaborative
signals — to escape the brittleness without losing the conviction.

**AI collaboration — one helpful suggestion:**
When designing the energy scoring formula, the AI suggested using a proximity
approach `(1 - abs(target - actual))` instead of a binary threshold. This was
exactly right — it makes the system forgiving of near-matches and creates a
continuous scoring surface instead of a cliff edge. I would not have thought of
this formulation as quickly on my own.

**AI collaboration — one flawed suggestion:**
Early in the project, the AI suggested implementing the vector store using the
`chromadb` library. This was technically correct but wrong for the context — a
20-song catalog with no API key requirement doesn't benefit from a full vector
database. It would have added a heavy dependency, required a running service,
and made the project harder to set up for reviewers. The right call was to build
a lightweight in-memory store using standard library math. The AI defaulted to
"use the production tool" when the teaching context called for "build it from
scratch so you understand it." That distinction — production tool vs. pedagogical
clarity — required human judgment.

**How this changed how I think about AI:**
Every recommendation system is making editorial decisions. The question is not
whether bias exists — it always does — but whether it is visible, documented,
and subject to challenge. VibeFinder makes its bias explicit in code (the weight
constants), in the model card (this document), and in the test harness output
(the FAIL results for conflicting inputs). That transparency is the difference
between a responsible system and a black box.