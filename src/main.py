"""
Music Recommender Simulation — CLI Runner
=========================================
Run this file to see your recommender in action from the terminal.

Usage
-----
    python -m src.main

Edit the USER_PROFILES dict below to experiment with different listeners.
"""

from src.recommender import load_songs, recommend_songs

# ---------------------------------------------------------------------------
# Sample user profiles for experimentation
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


def print_recommendations(profile_name: str, user_prefs: dict, songs: list) -> None:
    """Print formatted recommendations for a given user profile."""
    print(f"\n{'=' * 55}")
    print(f"  🎧 Recommendations for: {profile_name.upper()}")
    print(
        f"  Prefers: {user_prefs['genre']} | {user_prefs['mood']} | energy {user_prefs['energy']}")
    print(f"{'=' * 55}")

    results = recommend_songs(user_prefs, songs, k=5)

    for rank, (song, score, explanation) in enumerate(results, start=1):
        print(f"\n  #{rank}  {song['title']} — {song['artist']}")
        print(f"       Score : {score:.2f}")
        print(f"       Why   : {explanation}")


def main() -> None:
    songs = load_songs("data/songs.csv")
    print(f"\n✅ Loaded {len(songs)} songs from catalog.")

    for name, prefs in USER_PROFILES.items():
        print_recommendations(name, prefs, songs)

    print(f"\n{'=' * 55}\n")


if __name__ == "__main__":
    main()
