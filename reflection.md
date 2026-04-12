# Reflection: Comparing User Profiles and Outputs

This file documents what I observed when running different listener profiles through
VibeFinder 1.0, and what those observations tell us about how the scoring logic behaves.

---

## Pair 1: pop_fan vs chill_studier

**pop_fan** (pop / happy / energy 0.8 / electronic) → top result: *Sunrise City* (9.46)
**chill_studier** (lofi / chill / energy 0.4 / acoustic) → top result: *Midnight Coding* (9.58)

These two profiles are almost mirror images of each other — different genre, different
mood, opposite energy levels, and opposite acoustic preference. The results made
complete sense: the system correctly routed each user to a completely different part
of the catalog with no overlap in the top 3.

What this tells us: **when the user's preferences are internally consistent and the
catalog has good coverage of their genre, the system works exactly as intended.** The
four-signal combination (genre + mood + energy + acoustic) is specific enough to clearly
separate listener types.

---

## Pair 2: gym_warrior vs late_night_driver

**gym_warrior** (rock / intense / energy 0.95 / electronic) → top result: *Storm Runner* (9.63)
**late_night_driver** (synthwave / moody / energy 0.7 / electronic) → top result: *Night Drive Loop* (9.66)

Both profiles share the "electronic" preference and have moderate-to-high energy, but
the mood and genre pull them to completely different songs. Storm Runner is aggressive
and driving; Night Drive Loop is dark and atmospheric. The system got this right.

The interesting finding: after #1, the late_night_driver's top-5 drops sharply in score
(from 9.66 to 4.41 for #2). That 5-point cliff exists because synthwave is the rarest
genre in the catalog — once the only synthwave song is matched, everything else is just
energy proximity. The gym_warrior's cliff is smaller (9.63 → 6.63) because "intense"
mood gives partial credit to several electronic songs.

**What this tells us:** A rare genre preference does not degrade the #1 result, but it
dramatically weakens the quality of recommendations #2 through #5. Real platforms solve
this with catalog expansion and collaborative filtering.

---

## Pair 3: conflicted vs genre_ghost (adversarial)

**conflicted** (metal / happy / energy 0.15 / acoustic) → top result: *Subzero Drift* (4.66, metal/angry)
**genre_ghost** (bossa nova / relaxed / energy 0.5 / acoustic) → top result: *Coffee Shop Stories* (6.14, jazz/relaxed)

These were designed to find the edges of the system, and they revealed two very different
failure modes:

**Conflicted** exposed the genre dominance problem. The user wanted metal music but at
a low, happy, acoustic vibe — essentially asking for something like acoustic folk metal.
The system had no such song, but instead of defaulting to mood + energy (which would
have suggested acoustic/chill songs), it grabbed the only metal song in the catalog even
though it was the emotional opposite of everything else the user wanted. Genre at weight
3.0 is so strong that it overrode every other preference when only one song matched.

**Genre_ghost** behaved much better. Since no bossa nova exists in the catalog, genre
gave zero points to every song. That forced the system to fall back on mood, energy, and
acoustic preference — and the results were reasonable. Coffee Shop Stories (jazz/relaxed)
is a plausible recommendation for someone who wanted a bossa nova feel.

The lesson: **the system degrades better when the genre is entirely absent than when
only one song partially matches.** One matching song with wrong mood/energy is worse
than no matches at all, because the genre bonus pushes it to #1 regardless.

---

## Pair 4: Original weights vs experimental weights (pop_fan)

**Original** (genre×3.0, energy×2.0): #2 = Gym Hero (pop/intense, score 7.33)
**Experimental** (genre×1.5, energy×4.0): #2 = Havana Daydream (latin/happy, score 8.01)

The pop_fan asked for *happy* pop music. Under original weights, Gym Hero ranked #2
purely because it shares the "pop" genre label — even though its "intense" mood is the
opposite of "happy." Gym Hero kept showing up like a person who wears the right team
jersey but plays the wrong sport entirely.

When energy weight doubled, songs with energy levels very close to the user's 0.80 target
rose in the rankings. Havana Daydream has energy 0.79 — almost a perfect energy match —
and it carries the "happy" mood the user wanted. So in the experimental version, #2
through #4 are all non-pop songs that are energetically and emotionally closer to what the
user described.

**What this tells us:** Genre weight acts as a shortcut that can mask mood mismatches.
Reducing it forces the system to pay more attention to how the song *feels* rather than
just what *category* it belongs to. Whether that is better depends on the user — some
people genuinely care about genre loyalty, others care about vibe.

---

## Summary: What the Comparisons Reveal

| Comparison | Key Finding |
|---|---|
| pop_fan vs chill_studier | Clean separation when catalog coverage is good |
| gym_warrior vs late_night_driver | Rare genres cause a sharp quality drop after #1 |
| conflicted (adversarial) | One bad genre match is worse than zero genre matches |
| genre_ghost (adversarial) | Missing genre gracefully degrades to mood + energy |
| Weight experiment | Lowering genre weight surfaces better mood alignment |

The consistent theme: **the genre weight is the most consequential design decision in
this system.** Everything else adjusts around it. A real recommender would learn this
weight from user behavior — skip rates, replay rates, playlist additions — rather than
hardcoding it at 3.0.
