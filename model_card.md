# 🎧 Model Card: Music Recommender Simulation

## 1. Model Name

**VibeFinder 1.0** — A content-based music recommendation simulator built to
explore how weighted feature matching turns song data and listener preferences
into personalized suggestions.

---

## 2. Intended Use

VibeFinder 1.0 is designed to **simulate** how a real-world content-based music
recommender works. It takes a listener's taste profile (preferred genre, mood,
energy level, and acoustic preference) and scores every song in a 20-song catalog
against those preferences, returning the top-k most relevant results with a plain-
language explanation of why each song was selected.

- **What it recommends:** Songs from a curated 20-track CSV catalog, ranked by weighted relevance score.
- **Assumptions about the user:** Each user is represented by a single static profile. The system assumes taste is fixed — it does not learn or adapt over time.
- **Who it is for:** This is a classroom simulation, not a production recommender. It is designed to make the mechanics of content-based filtering visible and understandable, not to serve real listeners at scale.

---

## 3. How the Model Works

Imagine you are a music store clerk who knows a customer loves upbeat pop. Every
time a new album arrives, you mentally check: "Is this pop? Is it happy? Does the
energy feel right?" You award more importance to genre than to tempo, because genre
is the strongest signal of whether someone will even consider the song.

VibeFinder works the same way, just with numbers instead of intuition:

1. **Each song gets a score.** The system checks six features: genre, mood, energy,
   acoustic character, emotional positiveness (valence), and danceability. For each
   feature, it awards points based on how closely the song matches the user's preferences.

2. **Weights decide importance.** Genre is worth 3 points if it matches (most
   important). Mood is worth 2 points. Energy uses a "proximity" approach — a song
   with energy 0.82 scores almost as well as 0.80 because it is very close, rather
   than failing a hard pass/fail test. Acousticness, valence, and danceability add
   smaller supporting signals.

3. **All songs are ranked.** Once every song has a score, they are sorted from
   highest to lowest. The top 5 are returned with a plain-language explanation of
   what contributed to each score.

The key design choice: **energy is scored by closeness, not by category.** A song
0.02 away from the target scores nearly as well as a perfect match, while a song
0.5 away scores much lower. This makes the system forgiving and realistic.

---

## 4. Data

The catalog contains **20 songs** stored in `data/songs.csv`. Each song has 10 features:
id, title, artist, genre, mood, energy (0-1), tempo in BPM, valence (0-1), danceability (0-1),
and acousticness (0-1).

**Genres represented:** pop, lofi, rock, indie pop, ambient, jazz, synthwave,
electronic, r&b, classical, funk, folk, gospel, metal, latin, dream pop (16 total)

**Moods represented:** happy, chill, intense, relaxed, moody, focused,
melancholic, nostalgic, uplifting, angry (10 total)

**Data added:** 10 songs were added to the original 10-song starter set to improve
genre and mood diversity. The original set was heavily weighted toward lofi and pop.

**What is missing:**
- No song lyrics, language, or cultural context
- No release year or era
- No artist popularity or social signals
- Genre distribution is uneven: lofi has 3 entries, metal and classical have 1 each

---

## 5. Strengths

The system performs well when the user's preferences are specific and well-represented:

- The `chill_studier` profile (lofi / chill / energy 0.4) returns two near-perfect
  matches scoring 9.58 and 9.44 — both intuitively correct for a late-night study session.
- The `gym_warrior` profile correctly surfaces Storm Runner at 9.63 without being
  confused by other high-energy songs in different genres.
- The system is **transparent**: every score includes a plain-language breakdown of
  exactly which features contributed and by how much.
- The energy proximity formula handles gradual mismatches gracefully — it does not
  hard-cut songs that are 0.05 away from the target.

---

## 6. Limitations and Bias

**Filter bubble:** Because genre carries 3x the weight of any other feature, users
are almost always served songs from their stated genre in the top results. A pop fan
who might enjoy a funk song with identical mood and energy will never see it ranked
above a mediocre pop song. The system optimizes for "safe" matches rather than discovery.

**The "Gym Hero" Problem:** Gym Hero (pop / intense / energy 0.93) appears in results
for the `pop_fan` (who wants *happy* pop) because the genre match at weight 3.0
overrides the mood mismatch. A user who asked for happy music gets an intense song at
#2. In a real product this would be a frustrating and confusing recommendation.

**Catalog skew:** Genres with only one song (metal, classical, gospel) will always show
that same song to any matching user. There is no within-genre diversity possible.

**Adversarial profile findings:**
- `conflicted` (metal / happy / energy 0.15): The system returned Subzero Drift
  (metal/angry/high-energy) — the emotional opposite of what was requested. A 3-point
  genre bonus overwhelmed mood and energy signals entirely.
- `genre_ghost` (bossa nova): No bossa nova exists in the catalog. The system fell
  back gracefully to mood + energy matches, but gave no signal that the preferred genre
  was unrepresented.
- `middle_of_road` (pop / chill / energy 0.5): Genre match dominated, returning
  high-energy pop songs even though the user wanted chill pop.

**Boolean acousticness is too coarse:** Songs above 0.6 are "acoustic," below is
"electronic." A song at 0.55 and one at 0.05 are treated identically. This threshold
creates an artificial boundary that does not reflect real listener perception.

---

## 7. Evaluation

**Profiles tested:** pop_fan, chill_studier, gym_warrior, late_night_driver,
conflicted, genre_ghost, middle_of_road (7 total, including 3 adversarial edge cases)

**What matched intuition:**
The three "clean" profiles all returned highly intuitive top results: Midnight Coding
for a study session (9.58), Storm Runner for a workout (9.63), Night Drive Loop for
a late-night drive (9.66). These feel exactly right.

**What surprised me:**
1. The `conflicted` profile (metal / happy / low energy) surfaced the angriest,
   loudest song in the catalog. A 3.0 genre weight with only one metal song in the
   catalog meant genre dominance overrode every other preference completely.
2. The `genre_ghost` (bossa nova) gracefully returned jazz and r&b as fallbacks —
   better behavior than expected for a missing genre.
3. `middle_of_road` showed that when genre and mood conflict in direction, genre wins
   every time, even if that produces a mood mismatch the user would notice immediately.

**Weight-shift experiment (genre x1.5, energy x4.0):**
Halving genre weight and doubling energy weight caused 4 of 5 rankings to change
for the pop_fan profile. Gym Hero fell from #2 to #5 because its energy (0.93) was
now penalized more relative to Havana Daydream (energy 0.79, very close to target 0.80).
The experimental ranking *felt* more accurate — the top 4 results all had moods and
energies that matched the "happy pop" intent more closely.

---

## 8. Future Work

- **Diversity injection:** Re-rank results to ensure top-5 spans at least 2-3 genres,
  breaking the filter bubble without eliminating relevance.
- **Continuous acousticness:** Replace the boolean preference with a float target and
  proximity scoring, eliminating the hard 0.6 threshold.
- **Fallback signaling:** When the preferred genre is absent from the catalog, tell the
  user explicitly rather than silently substituting.
- **Multi-mood profiles:** Allow a primary and secondary mood preference so partial
  mood matches are rewarded instead of binary match/miss.
- **Collaborative signal layer:** Add a "users who liked X also liked Y" signal on top
  of content-based scoring to improve discovery for exploratory listeners.

---

## 9. Personal Reflection

Building VibeFinder made one thing immediately clear: every weight is an editorial
decision with real consequences. Setting genre to 3.0 is not a neutral technical
choice — it is a statement that genre identity matters more than emotional state, more
than energy, more than acoustic texture combined. That decision produces a system that
feels safe for mainstream users but fails listeners with cross-genre tastes.

The most unexpected discovery came from the adversarial profiles. The "conflicted"
user wanted metal but also happy and calm — and the system returned the angriest,
loudest song in the catalog. No human curator would do that. It revealed that the
system optimizes for the label of genre match rather than the experience of listening.

This project changed how I interpret recommendation interfaces. When Spotify puts a
song in Discover Weekly, I now think about which features it matched, what weight
schema was used, and what I might never hear because the algorithm decided genre
identity outweighs mood preference that day. The filter bubble is not a bug — it is
baked into the math.