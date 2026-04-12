"""
Music Recommender Simulation — CLI Runner
=========================================
Demonstrates all four optional extensions:

  Challenge 1 — Extended features: popularity, release_decade, mood_tags
  Challenge 2 — Scoring modes: BALANCED, MOOD_FIRST, ENERGY_FOCUS, GENRE_FIRST
  Challenge 3 — Diversity penalty: prevents same artist dominating top-k
  Challenge 4 — Visual ASCII table with score bars and reasons

Usage
-----
    python -m src.main
"""

from src.recommender import (
    load_songs,
    recommend_songs,
    ScoringMode,
    WEIGHTS,
    MAX_POSSIBLE_SCORE,
)

# ── Display constants ─────────────────────────────────────────────────────────
W_RANK = 3
W_TITLE = 24
W_ARTIST = 16
W_GENRE = 10
W_MOOD = 9
W_SCORE = 18   # score bar column
TABLE_WIDTH = W_RANK + W_TITLE + W_ARTIST + W_GENRE + W_MOOD + W_SCORE + 14

BAR_FULL = 14   # characters in score bar
PENALTY = 0.6

# ── User profiles ─────────────────────────────────────────────────────────────
USER_PROFILES = {
    "pop_fan": {
        "genre": "pop",       "mood": "happy",   "energy": 0.80,
        "likes_acoustic": False, "preferred_decade": "2020s",
        "preferred_mood_tags": ["euphoric", "bright"],
    },
    "chill_studier": {
        "genre": "lofi",      "mood": "chill",   "energy": 0.40,
        "likes_acoustic": True,  "preferred_decade": "",
        "preferred_mood_tags": ["focused", "mellow"],
    },
    "gym_warrior": {
        "genre": "rock",      "mood": "intense", "energy": 0.95,
        "likes_acoustic": False, "preferred_decade": "",
        "preferred_mood_tags": ["aggressive", "driving"],
    },
    "late_night_driver": {
        "genre": "synthwave", "mood": "moody",   "energy": 0.70,
        "likes_acoustic": False, "preferred_decade": "2020s",
        "preferred_mood_tags": ["nostalgic", "cinematic"],
    },
}


# ── Formatting helpers ────────────────────────────────────────────────────────

def score_bar(score: float, max_score: float = MAX_POSSIBLE_SCORE) -> str:
    """Render a fixed-width ASCII bar: [XXXXXXXX......] 7.50"""
    filled = round((score / max_score) * BAR_FULL)
    empty = BAR_FULL - filled
    return f"[{'X'*filled}{'.'*empty}] {score:5.2f}"


def cell(text: str, width: int) -> str:
    """Left-align text in a fixed-width cell, truncate with ellipsis if needed."""
    text = str(text)
    if len(text) > width:
        text = text[:width - 1] + "…"
    return text.ljust(width)


def table_divider(char: str = "-") -> str:
    return char * TABLE_WIDTH


def table_header() -> str:
    return (
        f"  {cell('#',   W_RANK)}"
        f"  {cell('Title',  W_TITLE)}"
        f"  {cell('Artist', W_ARTIST)}"
        f"  {cell('Genre',  W_GENRE)}"
        f"  {cell('Mood',   W_MOOD)}"
        f"  {'Score bar'}"
    )


def table_row(rank: int, song: dict, score: float) -> str:
    return (
        f"  {cell(str(rank), W_RANK)}"
        f"  {cell(song['title'],  W_TITLE)}"
        f"  {cell(song['artist'], W_ARTIST)}"
        f"  {cell(song['genre'],  W_GENRE)}"
        f"  {cell(song['mood'],   W_MOOD)}"
        f"  {score_bar(score)}"
    )


def reason_rows(explanation: str) -> list:
    """Split explanation string into one indented line per reason."""
    indent = " " * (W_RANK + W_TITLE + W_ARTIST + W_GENRE + W_MOOD + 12)
    return [f"{indent}  * {r}" for r in explanation.split("; ") if r]


def print_section(title: str, subtitle: str = "") -> None:
    print(f"\n{'='*TABLE_WIDTH}")
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print(f"{'='*TABLE_WIDTH}")


# ── Section printers ──────────────────────────────────────────────────────────

def print_standard_table(profile_name: str, prefs: dict, songs: list) -> None:
    """Challenge 4 — visual table for a single profile in BALANCED mode."""
    results = recommend_songs(prefs, songs, k=5)
    acoustic_label = "acoustic" if prefs.get(
        "likes_acoustic") else "electronic"
    tags = ", ".join(prefs.get("preferred_mood_tags", [])) or "none"
    decade = prefs.get("preferred_decade") or "any"

    print_section(
        f"Profile: {profile_name.upper()}",
        f"genre={prefs['genre']} | mood={prefs['mood']} | "
        f"energy={prefs['energy']} | {acoustic_label} | "
        f"era={decade} | tags=[{tags}]"
    )
    print(table_header())
    print(table_divider())
    for rank, (song, score, explanation) in enumerate(results, 1):
        print(table_row(rank, song, score))
        for line in reason_rows(explanation):
            print(line)
        if rank < len(results):
            print(table_divider("."))
    print(table_divider())


def print_mode_comparison(prefs: dict, songs: list) -> None:
    """Challenge 2 — show how top-3 changes across scoring modes."""
    print_section(
        "Scoring Mode Comparison  [Challenge 2]",
        f"Same profile (pop_fan) — four different ranking strategies"
    )
    header = f"  {'Mode':<14}" + \
        "".join(f"  #{i+1} {'Song':<22}" for i in range(3))
    print(header)
    print(table_divider())

    for mode in ScoringMode:
        results = recommend_songs(prefs, songs, k=3, mode=mode)
        row = f"  {mode.value:<14}"
        for song, score, _ in results:
            row += f"  {cell(song['title'], 22)} "
        print(row)
    print(table_divider())
    print("  Note: MOOD_FIRST and ENERGY_FOCUS surface cross-genre songs.")
    print("        GENRE_FIRST amplifies the filter bubble effect.")


def print_diversity_comparison(prefs: dict, songs: list) -> None:
    """Challenge 3 — before/after diversity penalty for a lofi user."""
    print_section(
        "Diversity Penalty  [Challenge 3]",
        "lofi profile — LoRoom has 2 songs that would otherwise dominate"
    )
    without = recommend_songs(prefs, songs, k=5, apply_diversity_penalty=False)
    with_p = recommend_songs(prefs, songs, k=5, apply_diversity_penalty=True)

    header = f"  {'#':<3}  {'WITHOUT penalty':<28}  {'WITH penalty':<28}"
    print(header)
    print(table_divider())
    for i, ((s1, sc1, _), (s2, sc2, _)) in enumerate(zip(without, with_p), 1):
        col1 = f"{s1['title'][:22]:<22} {sc1:5.2f}"
        col2 = f"{s2['title'][:22]:<22} {sc2:5.2f}"
        flag = "  <- swapped" if s1["title"] != s2["title"] else ""
        print(f"  {i:<3}  {col1:<28}  {col2:<28}{flag}")
    print(table_divider())
    print(f"  Penalty: repeated artist score x{PENALTY} (40% reduction)")
    print("  Result:  second LoRoom song drops, more artist variety surfaces.")


def print_extended_features(prefs: dict, songs: list) -> None:
    """Challenge 1 — show mood tag + era scoring contribution."""
    print_section(
        "Extended Features  [Challenge 1]",
        f"pop_fan with mood_tags=[euphoric, bright] | preferred_decade=2020s"
    )
    results = recommend_songs(prefs, songs, k=5)
    header = f"  {'#':<3}  {'Title':<24}  {'Tags on song':<30}  {'Era':<6}  {'Pop':>3}  Score"
    print(header)
    print(table_divider())
    for rank, (song, score, explanation) in enumerate(results, 1):
        tags = song.get("mood_tags", "")[:28]
        era = song.get("release_decade", "?")
        pop = int(song.get("popularity", 0))
        print(
            f"  {rank:<3}  {cell(song['title'], 24)}  {tags:<30}  {era:<6}  {pop:>3}  {score:.2f}")
    print(table_divider())
    print("  Popularity bonus:  up to +0.50 (normalised 0-100 → 0-1)")
    print("  Era match bonus:   +0.50 when release_decade == preferred_decade")
    print("  Mood tag bonus:    up to +1.00 based on overlap ratio")


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    """Load catalog and run all four challenge demonstrations."""
    songs = load_songs("data/songs.csv")

    print()
    print(f"{'='*TABLE_WIDTH}")
    print(f"  VIBEFINDER 1.0 — Music Recommender Simulation")
    print(
        f"  Catalog: {len(songs)} songs  |  Max score (BALANCED): {MAX_POSSIBLE_SCORE:.1f}")
    print(f"  Weights: genre x{WEIGHTS['genre']} | mood x{WEIGHTS['mood']} | "
          f"energy x{WEIGHTS['energy']} | mood_tags x{WEIGHTS['mood_tags']} | "
          f"popularity x{WEIGHTS['popularity']}")
    print(f"{'='*TABLE_WIDTH}")

    # Challenge 4 — table output for every standard profile
    for name, prefs in USER_PROFILES.items():
        print_standard_table(name, prefs, songs)

    # Challenge 2 — mode comparison
    print_mode_comparison(USER_PROFILES["pop_fan"], songs)

    # Challenge 3 — diversity penalty
    print_diversity_comparison(USER_PROFILES["chill_studier"], songs)

    # Challenge 1 — extended feature breakdown
    print_extended_features(USER_PROFILES["pop_fan"], songs)

    print()


if __name__ == "__main__":
    main()
