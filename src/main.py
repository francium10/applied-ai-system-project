"""
Music Recommender Simulation — CLI Runner
=========================================
Run with:  python -m src.main

Demonstrates:
  - Phase 4 : standard + adversarial user profiles
  - Phase 4 : weight experiment (ENERGY_FOCUS vs BALANCED)
  - Challenge 2 : ScoringMode strategy switching
  - Challenge 3 : diversity penalty
  - Challenge 4 : tabulate formatted output
"""

from tabulate import tabulate

from src.recommender import (
    MAX_POSSIBLE_SCORE,
    MODE_WEIGHTS,
    WEIGHTS,
    ScoringMode,
    load_songs,
    recommend_songs,
)

WIDTH = 65
BAR_MAX = 20


# ---------------------------------------------------------------------------
# User profiles — standard + adversarial edge cases (Phase 4 Step 1)
# ---------------------------------------------------------------------------

STANDARD_PROFILES = {
    "pop_fan": {
        "genre": "pop", "mood": "happy", "energy": 0.8,
        "likes_acoustic": False, "preferred_decade": "2020s",
        "preferred_mood_tags": ["euphoric", "upbeat"],
    },
    "chill_studier": {
        "genre": "lofi", "mood": "chill", "energy": 0.4,
        "likes_acoustic": True, "preferred_decade": "2020s",
        "preferred_mood_tags": ["focused", "dreamy"],
    },
    "gym_warrior": {
        "genre": "rock", "mood": "intense", "energy": 0.95,
        "likes_acoustic": False, "preferred_decade": "2010s",
        "preferred_mood_tags": ["aggressive", "driving"],
    },
    "late_night_driver": {
        "genre": "synthwave", "mood": "moody", "energy": 0.7,
        "likes_acoustic": False, "preferred_decade": "2020s",
        "preferred_mood_tags": ["nostalgic", "cinematic"],
    },
}

# Adversarial profiles: designed to expose edge cases and bias (Phase 4 Step 1)
ADVERSARIAL_PROFILES = {
    "genre_orphan": {
        # Genre that doesn't exist in the catalog — tests graceful degradation
        "genre": "bluegrass", "mood": "happy", "energy": 0.6,
        "likes_acoustic": True, "preferred_decade": "", "preferred_mood_tags": [],
    },
    "conflicted_listener": {
        # High energy + melancholic mood — internally contradictory preference
        "genre": "pop", "mood": "melancholic", "energy": 0.9,
        "likes_acoustic": False, "preferred_decade": "2000s",
        "preferred_mood_tags": ["sad", "aggressive"],
    },
    "perfectly_average": {
        # All preferences at midpoint — tests what "neutral" looks like
        "genre": "jazz", "mood": "relaxed", "energy": 0.5,
        "likes_acoustic": False, "preferred_decade": "", "preferred_mood_tags": [],
    },
    "niche_completionist": {
        # Highly specific niche preferences with many mood tags
        "genre": "classical", "mood": "melancholic", "energy": 0.2,
        "likes_acoustic": True, "preferred_decade": "2000s",
        "preferred_mood_tags": ["nostalgic", "tender", "sad"],
    },
}


# ---------------------------------------------------------------------------
# Formatting helpers (Challenge 4 — tabulate output)
# ---------------------------------------------------------------------------


def score_bar(score: float, max_score: float = MAX_POSSIBLE_SCORE) -> str:
    """Render an ASCII progress bar: [XXXX....] score/max."""
    filled = round((score / max_score) * BAR_MAX)
    bar = "X" * filled + "." * (BAR_MAX - filled)
    return f"[{bar}] {score:.2f}"


def build_table(results: list, max_score: float = MAX_POSSIBLE_SCORE) -> str:
    """
    Build a tabulate table from recommend_songs() results.

    Challenge 4: uses tabulate with 'rounded_outline' format.
    Each row shows rank, title, artist, genre, score bar, and top reason.
    """
    rows = []
    for rank, (song, score, explanation) in enumerate(results, start=1):
        top_reason = explanation.split(";")[0].strip()
        rows.append([
            f"#{rank}",
            song["title"],
            song["artist"],
            f"{song['genre']} / {song['mood']}",
            score_bar(score, max_score),
            top_reason,
        ])
    return tabulate(
        rows,
        headers=["Rank", "Title", "Artist",
                 "Genre / Mood", "Score", "Top reason"],
        tablefmt="rounded_outline",
    )


def print_section(title: str, subtitle: str = "") -> None:
    """Print a section header."""
    print()
    print("=" * WIDTH)
    print(f"  {title}")
    if subtitle:
        print(f"  {subtitle}")
    print("=" * WIDTH)


# ---------------------------------------------------------------------------
# Experiment: BALANCED vs ENERGY_FOCUS (Phase 4 Step 3)
# ---------------------------------------------------------------------------


def run_weight_experiment(songs: list, profile_name: str, prefs: dict) -> None:
    """
    Run the same profile under two scoring modes and print side-by-side.

    Phase 4 Step 3 — Weight Shift experiment:
    BALANCED  : genre×3, mood×2, energy×2
    ENERGY_FOCUS: genre×1, mood×1, energy×4
    Doubling energy weight shows how much genre dominance relaxes.
    """
    print_section(
        "WEIGHT EXPERIMENT",
        f"Profile: {profile_name} | BALANCED vs ENERGY_FOCUS mode"
    )

    balanced = recommend_songs(prefs, songs, k=5, mode=ScoringMode.BALANCED)
    energy_mode = recommend_songs(
        prefs, songs, k=5, mode=ScoringMode.ENERGY_FOCUS)

    max_b = sum(MODE_WEIGHTS[ScoringMode.BALANCED].values())
    max_e = sum(MODE_WEIGHTS[ScoringMode.ENERGY_FOCUS].values())

    print("\n  BALANCED (genre x3.0 | mood x2.0 | energy x2.0)")
    print(build_table(balanced, max_b))

    print("\n  ENERGY_FOCUS (genre x1.0 | mood x1.0 | energy x4.0)")
    print(build_table(energy_mode, max_e))

    b_titles = [s["title"] for s, _, _ in balanced]
    e_titles = [s["title"] for s, _, _ in energy_mode]
    changed = [t for t in e_titles if t not in b_titles]
    if changed:
        print(f"\n  >> New entries under ENERGY_FOCUS: {', '.join(changed)}")
        print("     (genre dominance relaxed — energy proximity now leads)")
    else:
        print("\n  >> Top-5 unchanged — energy and genre signals aligned for this profile.")


# ---------------------------------------------------------------------------
# Diversity penalty demo (Challenge 3)
# ---------------------------------------------------------------------------


def run_diversity_demo(songs: list, prefs: dict) -> None:
    """Show recommendations before and after applying the diversity penalty."""
    print_section(
        "CHALLENGE 3 — DIVERSITY PENALTY",
        "Penalises repeated artists by 40%"
    )

    before = recommend_songs(prefs, songs, k=5, apply_diversity_penalty=False)
    after = recommend_songs(prefs, songs, k=5, apply_diversity_penalty=True)

    print("\n  Without diversity penalty:")
    print(build_table(before))
    print("\n  With diversity penalty (40% score reduction on repeated artist):")
    print(build_table(after))


# ---------------------------------------------------------------------------
# Scoring mode showcase (Challenge 2)
# ---------------------------------------------------------------------------


def run_mode_showcase(songs: list, prefs: dict, profile_name: str) -> None:
    """Run all four scoring modes for one profile and show the differences."""
    print_section(
        "CHALLENGE 2 — SCORING MODES",
        f"Profile: {profile_name} across all four strategies"
    )
    for mode in ScoringMode:
        max_score = sum(MODE_WEIGHTS[mode].values())
        results = recommend_songs(prefs, songs, k=3, mode=mode)
        print(f"\n  Mode: {mode.value.upper()}")
        print(build_table(results, max_score))


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def main() -> None:
    songs = load_songs("data/songs.csv")

    print_section(
        "MUSIC RECOMMENDER SIMULATION",
        f"Catalog: {len(songs)} songs  |  Max score (BALANCED): {MAX_POSSIBLE_SCORE:.1f}"
    )

    # ── Standard profiles (Phase 4 + Challenge 4 tabulate) ──────────────────
    print_section("STANDARD PROFILES")
    for name, prefs in STANDARD_PROFILES.items():
        results = recommend_songs(prefs, songs, k=5, mode=ScoringMode.BALANCED)
        print(f"\n  Profile: {name.upper()}")
        print(
            f"  genre={prefs['genre']} | mood={prefs['mood']} | energy={prefs['energy']}")
        print(build_table(results))

    # ── Adversarial / edge-case profiles (Phase 4 Step 1) ───────────────────
    print_section(
        "ADVERSARIAL PROFILES",
        "Edge cases designed to expose bias and graceful degradation"
    )
    for name, prefs in ADVERSARIAL_PROFILES.items():
        results = recommend_songs(prefs, songs, k=5, mode=ScoringMode.BALANCED)
        print(f"\n  Profile: {name.upper()}")
        print(
            f"  genre={prefs['genre']} | mood={prefs['mood']} | energy={prefs['energy']}")
        print(build_table(results))

    # ── Weight experiment (Phase 4 Step 3) ──────────────────────────────────
    run_weight_experiment(songs, "gym_warrior",
                          STANDARD_PROFILES["gym_warrior"])

    # ── Challenge 2: all scoring modes ──────────────────────────────────────
    run_mode_showcase(songs, STANDARD_PROFILES["pop_fan"], "pop_fan")

    # ── Challenge 3: diversity penalty ──────────────────────────────────────
    run_diversity_demo(songs, STANDARD_PROFILES["chill_studier"])

    print()
    print("=" * WIDTH)
    print()


if __name__ == "__main__":
    main()
