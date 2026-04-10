"""
Music Recommender Simulation — CLI Runner
=========================================
Entry point for running the recommender from the terminal.

Usage
-----
    python -m src.main

Edit USER_PROFILES below to experiment with different listener types.
Each profile is a dict with keys: genre, mood, energy, likes_acoustic.
"""

from src.recommender import load_songs, recommend_songs, WEIGHTS, MAX_POSSIBLE_SCORE

# ---------------------------------------------------------------------------
# Terminal display constants
# ---------------------------------------------------------------------------

WIDTH = 60
BAR_WIDTH = 20  # Width of the ASCII score bar


# ---------------------------------------------------------------------------
# Sample user profiles for CLI experimentation
# ---------------------------------------------------------------------------

USER_PROFILES = {
    "pop_fan": {
        "genre": "pop",
        "mood": "happy",
        "energy": 0.8,
        "likes_acoustic": False,
    },
    "chill_studier": {
        "genre": "lofi",
        "mood": "chill",
        "energy": 0.4,
        "likes_acoustic": True,
    },
    "gym_warrior": {
        "genre": "rock",
        "mood": "intense",
        "energy": 0.95,
        "likes_acoustic": False,
    },
    "late_night_driver": {
        "genre": "synthwave",
        "mood": "moody",
        "energy": 0.7,
        "likes_acoustic": False,
    },
}


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def score_bar(score: float, max_score: float = MAX_POSSIBLE_SCORE) -> str:
    """Render a compact ASCII progress bar showing score vs maximum possible."""
    filled = round((score / max_score) * BAR_WIDTH)
    empty = BAR_WIDTH - filled
    bar = "X" * filled + "." * empty
    return f"[{bar}]  {score:.2f} / {max_score:.2f}"


def print_divider(char: str = "-", width: int = WIDTH) -> None:
    """Print a full-width horizontal rule."""
    print(char * width)


def print_profile_header(name: str, prefs: dict) -> None:
    """Print a formatted header block for a user profile section."""
    print()
    print_divider("=")
    print(f"  >> {name.upper()}")
    acoustic_label = "acoustic" if prefs.get(
        "likes_acoustic") else "electronic"
    print(
        f"  Prefers: {prefs['genre']} | {prefs['mood']} | "
        f"energy {prefs['energy']} | {acoustic_label}"
    )
    print_divider("=")


def print_recommendation(rank: int, song: dict, score: float, explanation: str) -> None:
    """Print one formatted recommendation entry with score bar and reasons."""
    print(f"\n  #{rank}  {song['title']}")
    print(f"       Artist : {song['artist']}")
    print(
        f"       Genre  : {song['genre']}  |  Mood: {song['mood']}  |  Energy: {song['energy']:.2f}")
    print(f"       Score  : {score_bar(score)}")
    reasons = explanation.split("; ")
    print(f"       Why    :")
    for reason in reasons:
        print(f"                * {reason}")


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------


def main() -> None:
    """Load catalog, run all profiles, and print ranked recommendations."""
    songs = load_songs("data/songs.csv")

    print()
    print_divider("=")
    print("  MUSIC RECOMMENDER SIMULATION")
    print(
        f"  Catalog: {len(songs)} songs  |  Max possible score: {MAX_POSSIBLE_SCORE:.1f}")
    print(f"  Weights: genre x{WEIGHTS['genre']} | mood x{WEIGHTS['mood']} | "
          f"energy x{WEIGHTS['energy']} | acousticness x{WEIGHTS['acousticness']}")
    print_divider("=")

    for profile_name, user_prefs in USER_PROFILES.items():
        print_profile_header(profile_name, user_prefs)

        results = recommend_songs(user_prefs, songs, k=5)

        for rank, (song, score, explanation) in enumerate(results, start=1):
            print_recommendation(rank, song, score, explanation)

        print()
        print_divider()

    print()


if __name__ == "__main__":
    main()
