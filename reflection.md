# Reflection: VibeFinder 2.0 — Ethics, Bias, and AI Collaboration

> **Rubric requirement addressed:** This document answers all five required reflection
> questions from the Applied AI System specification.

---

## Question 1: What are the limitations or biases in your system?

**Genre dominance bias** is the most significant. Genre carries 3× the weight of any
other feature. A song in the right genre with the wrong mood will consistently outscore
a song in the wrong genre with a perfectly matching mood and energy. This means users
who want "happy music" can receive "intense music" simply because both are labeled
"pop." The system optimizes for *genre loyalty* over *emotional experience*.

**Filter bubble** follows directly from genre dominance. Users almost never see
cross-genre recommendations, even when a funk song with identical mood and energy
would serve them better than a mediocre pop song. The system rewards familiarity
over discovery.

**Catalog skew** compounds both problems. Genres with only one song give users no
real choice within that genre. Genres absent from the catalog cause silent fallback
without informing the user their preference wasn't found.

**The boolean acousticness threshold** is too coarse. Songs with acousticness 0.59
and 0.05 are treated identically as "electronic," even though they sound very
different to a human listener. A continuous proximity score would be more accurate.

**No feedback loop** means the system cannot learn from mistakes. If a user skips
every "intense pop" song, the weights never adjust. Real platforms escape this
through implicit feedback (skips, replays, playlist additions) — signals this
system intentionally omits.

---

## Question 2: Could your AI be misused, and how would you prevent it?

**Misuse vector 1 — Bias amplification at scale:**
If deployed with real users, the genre dominance weight would systematically
under-serve listeners whose tastes cross genre boundaries or favor underrepresented
genres (metal, classical, folk). At scale this is not just a bad user experience —
it's an equity issue. Listeners from musical traditions poorly represented in the
20-song catalog would consistently receive worse recommendations than listeners
whose tastes align with the catalog's Western pop bias.

*Prevention:* Replace hardcoded weights with weights learned from diverse user
feedback. Add a diversity injection step that guarantees at least 2–3 different
genres appear in every top-5. Expand the catalog with deliberate attention to
underrepresented musical traditions.

**Misuse vector 2 — Profile inference without consent:**
The logging system writes every query and parsed profile to `logs/decisions.json`.
In a multi-user deployment, this creates a persistent record of listener preferences
that could be used for profiling without explicit user consent.

*Prevention:* The current system writes logs only locally and only in single-user
mode. A production version would require explicit consent, retention limits (e.g.,
30-day auto-deletion), and anonymisation before any log could be inspected.

**Misuse vector 3 — False confidence:**
The confidence score (0–1) could be misread as a measure of recommendation *quality*
rather than *internal consistency*. A 0.9 confidence score means the system's signals
are aligned — not that the user will enjoy the song. Presenting this to users without
explanation could mislead them into trusting poor recommendations.

*Prevention:* The UI labels confidence clearly ("How consistent our signals are, not
how much you'll like it") and always shows the critic's plain-language note.

---

## Question 3: What surprised you while testing your AI's reliability?

Two things genuinely surprised me:

**The "missing genre gracefully degrades better than one bad match" finding:**
I expected the "missing genre in catalog" test case (bossa nova) to score lower than
the "conflicting preferences" case (metal/happy/low-energy). The opposite happened.
When bossa nova was entirely absent, the system fell back to mood and energy signals
and produced reasonable results. When metal had one song — but the wrong mood and
energy — the 3.0 genre bonus pushed that one song to #1 regardless. One partial match
is more harmful than no match at all.

**The re-rank rate for conflicting preferences:**
The agentic critic triggered a re-rank on 3 out of 5 runs for the metal/happy profile.
This felt surprising at first — the critic was doing its job correctly by flagging the
conflict. But the re-rank didn't fix the problem because no song in the catalog
satisfies "metal + happy + acoustic." The critic correctly identified uncertainty,
but couldn't produce a better answer from the same limited catalog. Confidence scoring
is most useful as a transparency tool, not as a fix.

---

## Question 4: Describe your AI collaboration — one helpful and one flawed suggestion

**One genuinely helpful AI suggestion:**

When designing the energy scoring formula, the AI suggested using proximity scoring
`(1 - abs(target - actual))` instead of a binary threshold. This was exactly right.
It makes the system reward "close enough" matches rather than cutting off at an
arbitrary boundary. A song 0.05 away from the target scores nearly as well as a
perfect match; a song 0.5 away scores much lower. This continuous formulation made
the entire scoring system more nuanced and realistic, and I would not have arrived
at it as quickly without the suggestion.

**One flawed AI suggestion:**

Early in the project, the AI recommended implementing the vector store using the
`chromadb` library — a production-grade vector database. This was technically
correct but contextually wrong. A 20-song catalog doesn't benefit from a full vector
database. ChromaDB would have added a heavy dependency requiring a separate service,
made the project harder to set up for reviewers without Docker, and obscured the
underlying math (cosine similarity) that the project is meant to demonstrate.

The right call — building an in-memory store using standard library `math` — required
overriding the AI's suggestion. The AI defaulted to "use the production tool." The
teaching context called for "build it from scratch so you understand it." That
distinction requires human judgment about the purpose of the project, which no AI
tool could fully appreciate from the prompt alone.

---

## Question 5: What does this project say about me as an AI engineer?

I can take a working prototype and systematically evolve it into a production-quality
architecture — adding explainability, reliability testing, natural language
understanding, and agentic self-correction without discarding what already worked.

The most important skill I practiced here was not any individual technology, but the
discipline of designing interfaces first. Every component in VibeFinder 2.0 can be
replaced, upgraded, or tested independently because I defined what it promises before
I wrote what it does. The embedder, the vector store, the critic, and the logger all
have narrow, well-defined interfaces. Swapping the in-memory store for Pinecone
requires changing one file. Swapping the rule-based critic for a Claude-powered one
requires setting one environment variable.

That is how I approach AI engineering: build the seams between components as carefully
as the components themselves.

---

## Pair 1: pop_fan vs chill_studier

**pop_fan** (pop / happy / energy 0.8) → top result: *Sunrise City* (10.96)
**chill_studier** (lofi / chill / energy 0.4 / acoustic) → top result: *Midnight Coding* (9.58)

These profiles are mirror images. The system correctly routes each to a completely
different catalog segment with no overlap in the top 3. This is the system working
as intended — four consistent signals (genre + mood + energy + acoustic) clearly
separate listener types.

## Pair 2: gym_warrior vs late_night_driver

**gym_warrior** (rock / intense / energy 0.95) → top: *Storm Runner* (9.63)
**late_night_driver** (synthwave / moody / energy 0.7) → top: *Night Drive Loop* (9.66)

Both share "electronic" preference and moderate-high energy, but mood and genre
pull them apart. After #1, the late_night_driver's scores drop sharply (9.66 → 4.41
at #2) because synthwave has only one song. The system finds the right #1 but has
nowhere to go for diversity. This is the catalog skew problem in action.

## Pair 3: conflicted vs genre_ghost (adversarial)

**conflicted** (metal / happy / energy 0.15 / acoustic) → top: *Subzero Drift* (4.66, metal/angry)
**genre_ghost** (bossa nova / relaxed / energy 0.5 / acoustic) → top: *Coffee Shop Stories* (6.14)

Conflicted exposed the single-song genre trap. Genre_ghost showed graceful
degradation. One partial bad match (metal with wrong mood) is worse than no match
at all (bossa nova absent from catalog).

## Pair 4: Original weights vs experimental

Halving genre weight (3.0→1.5) and doubling energy weight (2.0→4.0) changed 4 of 5
rankings for the pop_fan. The new #2–4 matched the "happy" mood better. Genre weight
is the single most consequential design decision — it encodes whether the system
optimizes for genre loyalty or emotional experience. A real recommender would learn
this weight from user behavior rather than hardcode it.