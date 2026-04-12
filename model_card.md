# 🎧 Model Card: VibeFinder 1.0

---

## 1. Model Name

**VibeFinder 1.0**

A content-based music recommendation simulator. The name reflects the core idea:
matching songs to a listener's "vibe" using measurable audio features rather than
guessing from other users' behavior.

---

## 2. Goal / Task

VibeFinder tries to answer one question: *"Given what I know about a listener's
taste, which songs in the catalog are most worth suggesting?"*

It does this by taking a listener's stated preferences — favorite genre, preferred
mood, target energy level, and whether they like acoustic or electronic sounds —
and scoring every song in the catalog against those preferences. The songs with the
highest scores get recommended, along with a plain-language explanation of why.

This is **not** a prediction model. It does not predict whether you will like a song.
It measures how closely a song matches a description of your taste. The difference
matters: a prediction requires feedback data (skips, replays, saves); a matching
system only needs the song's features and your stated preferences.

---

## 3. Data Used

**Catalog size:** 20 songs in `data/songs.csv`

**Features per song:**

| Feature | Type | What it measures |
|---|---|---|
| genre | text | Broad style category (pop, lofi, rock, etc.) |
| mood | text | Emotional tone (happy, chill, intense, etc.) |
| energy | 0–1 float | How driving or intense the track feels |
| tempo_bpm | float | Beats per minute |
| valence | 0–1 float | Musical positiveness (high = cheerful) |
| danceability | 0–1 float | How suitable the track is for dancing |
| acousticness | 0–1 float | How acoustic vs. electronic the track sounds |

**Genres in catalog:** pop, lofi, rock, indie pop, ambient, jazz, synthwave,
electronic, r&b, classical, funk, folk, gospel, metal, latin, dream pop

**Moods in catalog:** happy, chill, intense, relaxed, moody, focused,
melancholic, nostalgic, uplifting, angry

**Known data limits:**
- 20 songs is a toy catalog. A real platform has millions.
- Genre distribution is uneven: lofi has 3 songs, metal and classical have 1 each.
- No lyrics, language, cultural context, or release era is captured.
- The data reflects one curator's taste — it skews toward contemporary electronic
  and indie sounds and underrepresents global music traditions.

---

## 4. Algorithm Summary

The scoring works in three steps:

**Step 1 — Score each song individually.**
For every song in the catalog, the system checks six features and awards points:

- Genre match → +3.0 points if it matches the user's favorite genre, 0 if not
- Mood match → +2.0 points if it matches the user's preferred mood, 0 if not
- Energy proximity → up to +2.0 points, scaled by how *close* the song's energy
  is to the user's target. A perfect match earns the full 2.0. A song that is 0.5
  away earns 1.0. This is the key insight: closeness is rewarded, not just exact match.
- Acoustic alignment → +1.0 if the song's acoustic character matches the user's preference
- Valence → up to +1.0 for emotionally expressive songs (very cheerful or very melancholic)
- Danceability → up to +0.5 as a mild supporting signal

Maximum possible score: **9.5 points**

**Step 2 — Rank all songs.**
All 20 songs are sorted from highest score to lowest. This is the Ranking Rule —
it turns 20 individual scores into a single ordered list.

**Step 3 — Return the top k.**
The top 5 songs (by default) are returned with their scores and reason lists.

The design is intentionally transparent. Every number the system uses is visible
in the code, and every recommendation comes with an explanation a non-programmer
can read.

---

## 5. Observed Behavior and Biases

**What works well:**
When a user's preferences match a well-represented genre and the catalog has multiple
songs in that space, the results feel genuinely right. A lofi/chill listener gets
Midnight Coding (9.58) and Library Rain (9.44) — both are objectively good study
session picks. A rock/intense listener gets Storm Runner first, every time.

**The genre dominance problem:**
Genre is worth 3.0 points — more than mood and energy combined. This means a song
in the right genre with the wrong mood can outscore a song in the wrong genre with
a perfect mood and energy match. In testing, a user who asked for *happy pop* got
an *intense pop* song at #2, because the genre label matched even though the mood
was the opposite of what they wanted. This is the biggest bias in the current system.

**The filter bubble:**
Because genre is the dominant signal, users almost never see recommendations outside
their stated genre. A pop listener who might enjoy a funk song with the exact same
mood and energy will never see it in their top 5. The system optimizes for familiarity
over discovery.

**The single-song genre trap:**
When only one song in the catalog matches the user's genre, that song ranks #1
regardless of whether it matches anything else. In the "conflicted" adversarial test
(metal / happy / low energy), the system returned the only metal song — which was
*angry* and *high energy* — because the 3.0 genre bonus outweighed all other mismatches.
No human music curator would make that recommendation.

**The catalog skew:**
Any genre with one song gives that user no real choice. Any genre not in the catalog
at all causes the system to silently fall back to mood and energy matching without
telling the user their preference wasn't found.

---

## 6. Evaluation Process

**Standard profiles tested:**
Four listener types were run: pop fan, chill studier, gym warrior, late night driver.
Results for all four felt intuitively correct — each profile's top song was an obvious
match, and the ranking order made sense.

**Adversarial profiles tested:**
Three edge cases were designed to stress-test the logic:

1. `conflicted` — metal genre but happy mood and very low energy. Exposed the genre
   dominance failure: the system returned an angry high-energy metal song.
2. `genre_ghost` — bossa nova preference with nothing in the catalog. The system
   gracefully fell back to mood and energy matching, surfacing jazz/relaxed as a
   reasonable substitute.
3. `middle_of_road` — pop genre but chill mood. Genre won over mood, returning
   energetic pop songs to someone who asked for calm ones.

**Weight-shift experiment:**
Genre weight was halved (3.0 → 1.5) and energy weight was doubled (2.0 → 4.0).
For the pop fan profile, 4 of 5 rankings changed. The new #2–4 results matched the
"happy" mood better than the original, suggesting the default genre weight may be
calibrated too aggressively for mood-sensitive users.

**What surprised me:**
The "conflicted" result was the biggest surprise. I expected the system to partially
satisfy the mood preference when genre was a mismatch on other dimensions. Instead,
one genre point acted like a veto that overrode everything else. It revealed that
the system has no concept of *internal preference conflict* — it treats each feature
independently, with no awareness that metal and happy/low-energy are contradictory
as a combined request.

---

## 7. Intended Use and Non-Intended Use

**This system IS designed for:**
- Learning how content-based filtering works by reading transparent, explainable output
- Classroom exploration of algorithmic bias and filter bubbles
- Experimenting with how weight changes affect recommendation behavior
- Building intuition for how real platforms structure their recommendation logic

**This system IS NOT designed for:**
- Serving real listeners on a real platform
- Handling more than a few dozen songs (the scoring loop is O(n) and not optimized)
- Adapting to user feedback — it has no memory of what was skipped or replayed
- Representing diverse global music taste — the catalog and genre labels reflect
  a narrow cultural slice
- Making decisions about what music is "good" — it only measures feature proximity,
  not quality, cultural significance, or emotional depth

If you are building a real music app, use a platform with collaborative filtering,
a large catalog, and feedback loops. VibeFinder 1.0 is a teaching tool, not a product.

---

## 8. Ideas for Improvement

**1. Add a diversity re-ranking step.**
After scoring, penalize consecutive songs from the same genre in the top-5. This
would break the filter bubble without changing the underlying scores — it would just
ensure variety in what gets surfaced. Real platforms call this "diversity injection."

**2. Replace the boolean acoustic preference with a continuous target.**
Instead of asking "do you like acoustic music: yes or no," ask for a target acousticness
value (0.0–1.0) and score it the same way energy is scored — by proximity. This would
eliminate the artificial 0.6 threshold that currently treats very different songs identically.

**3. Add a "no match found" signal for missing genres.**
When zero songs in the catalog share the user's genre, tell them explicitly: "No [genre]
songs in catalog — showing closest matches by mood and energy." Right now the fallback
is silent, which could confuse a user who doesn't know their genre isn't represented.

**Bonus if time allowed:**
Layer a simple collaborative signal on top — "listeners who enjoyed Midnight Coding
also saved Focus Flow" — to enable discovery beyond what the feature vectors alone
can surface. This is how Spotify escapes the content-based filter bubble.

---

## 9. Personal Reflection

**Biggest learning moment:**
The weight-shift experiment was the most clarifying moment of the whole project.
Changing one number — genre weight from 3.0 to 1.5 — changed 4 of 5 recommendations
for the same user. That made the abstract idea of "weights encode assumptions" feel
completely concrete. The algorithm did not change. The data did not change. One
design decision changed, and the output looked meaningfully different. That is the
most important thing I learned: in a weighted scoring system, the weights *are* the
editorial voice of the system's designer.

**How AI tools helped, and when I needed to double-check:**
AI tools were most useful for scaffolding — generating the initial dataclass structure,
suggesting the proximity formula for energy scoring, and drafting the CSV expansion.
The moments that required the most careful human review were the weight decisions and
the adversarial test design. No tool told me to test a "metal / happy / low energy"
user — that required thinking about what would expose a failure mode, not what would
demonstrate success. AI is good at generating plausible output; humans are better at
designing tests that probe for implausible failure.

**What surprised me about simple algorithms feeling like recommendations:**
The system has no understanding of music whatsoever. It does not know that metal is
loud, that lofi is associated with studying, or that "happy" and "angry" are
opposites. It knows numbers and labels. And yet when the labels and numbers align
well with a user's stated preferences, the output genuinely feels like a thoughtful
suggestion. That gap — between "matching labels and numbers" and "understanding
taste" — is what makes this simulation both convincing and brittle at the same time.
The conviction comes from pattern alignment; the brittleness comes from having no
model of what the patterns actually mean.

**What I would try next:**
The most interesting extension would be a feedback loop. After the user sees the top
5 recommendations, let them mark each one as "loved," "okay," or "skip." Use those
signals to adjust the weights automatically — genre weight goes down if the user keeps
skipping genre-matched songs, energy weight goes up if they consistently prefer songs
close to their target. That would turn VibeFinder from a static matcher into something
that actually learns, which is the core of what makes Spotify's Discover Weekly feel
personal after a few weeks of use.