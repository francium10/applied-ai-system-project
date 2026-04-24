"""
tests/test_recommender.py
==========================
Comprehensive tests for the core scoring engine.

Coverage map:
  Song / UserProfile dataclasses       — construction, field access, defaults
  Recommender OOP class                — sorting, k edge cases, explain
  score_song()                         — every scoring branch individually
  score_song_with_weights()            — all four ScoringMode presets
  recommend_songs()                    — modes, diversity penalty, edge cases
  load_songs()                         — CSV parsing, types, field validation
  ScoringMode / MODE_WEIGHTS           — enum completeness, weight relationships

Run with:  pytest tests/test_recommender.py -v
"""

import pytest
from src.recommender import (
    Song, UserProfile, Recommender, ScoringMode, MODE_WEIGHTS,
    WEIGHTS, MAX_POSSIBLE_SCORE, load_songs, recommend_songs,
    score_song, score_song_with_weights, _song_to_dict, _profile_to_dict,
)

CATALOG = "data/songs.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def songs():
    return load_songs(CATALOG)

@pytest.fixture
def two_rec():
    return Recommender([
        Song(id=1, title="Pop Track", artist="Artist A", genre="pop",
             mood="happy", energy=0.8, tempo_bpm=120, valence=0.9,
             danceability=0.8, acousticness=0.2),
        Song(id=2, title="Lofi Loop", artist="Artist B", genre="lofi",
             mood="chill", energy=0.4, tempo_bpm=80, valence=0.6,
             danceability=0.5, acousticness=0.9),
    ])

@pytest.fixture
def pop_user():
    return UserProfile(favorite_genre="pop", favorite_mood="happy",
                       target_energy=0.8, likes_acoustic=False)

@pytest.fixture
def lofi_user():
    return UserProfile(favorite_genre="lofi", favorite_mood="chill",
                       target_energy=0.4, likes_acoustic=True)

@pytest.fixture
def base_prefs():
    return {"genre": "pop", "mood": "happy", "energy": 0.8,
            "likes_acoustic": False, "preferred_mood_tags": [], "preferred_decade": ""}

@pytest.fixture
def base_song():
    return {"genre": "pop", "mood": "happy", "energy": 0.82,
            "valence": 0.84, "danceability": 0.79, "acousticness": 0.18,
            "mood_tags": "euphoric|bright|upbeat", "release_decade": "2020s", "popularity": 88}

# ---------------------------------------------------------------------------
# 1. Dataclasses
# ---------------------------------------------------------------------------

class TestDataclasses:
    def test_song_core_fields(self):
        s = Song(id=1, title="T", artist="A", genre="pop", mood="happy",
                 energy=0.5, tempo_bpm=100, valence=0.5, danceability=0.5, acousticness=0.3)
        assert s.genre == "pop" and s.energy == 0.5

    def test_song_extended_defaults(self):
        s = Song(id=1, title="T", artist="A", genre="pop", mood="happy",
                 energy=0.5, tempo_bpm=100, valence=0.5, danceability=0.5, acousticness=0.3)
        assert isinstance(s.popularity, float)
        assert s.mood_tags == ""

    def test_user_profile_extended_defaults(self):
        u = UserProfile(favorite_genre="pop", favorite_mood="happy",
                        target_energy=0.7, likes_acoustic=False)
        assert u.preferred_decade == "" and u.preferred_mood_tags == []

    def test_song_to_dict_has_all_keys(self):
        s = Song(id=1, title="T", artist="A", genre="pop", mood="happy",
                 energy=0.5, tempo_bpm=100, valence=0.5, danceability=0.5, acousticness=0.3)
        d = _song_to_dict(s)
        assert {"id","title","artist","genre","mood","energy","tempo_bpm",
                "valence","danceability","acousticness","popularity",
                "release_decade","mood_tags"}.issubset(d.keys())

    def test_profile_to_dict_has_all_keys(self):
        u = UserProfile(favorite_genre="pop", favorite_mood="happy",
                        target_energy=0.8, likes_acoustic=False)
        d = _profile_to_dict(u)
        assert {"genre","mood","energy","likes_acoustic",
                "preferred_decade","preferred_mood_tags"}.issubset(d.keys())

# ---------------------------------------------------------------------------
# 2. Recommender OOP
# ---------------------------------------------------------------------------

class TestRecommender:
    def test_pop_ranks_first_for_pop_user(self, two_rec, pop_user):
        r = two_rec.recommend(pop_user, k=2)
        assert r[0].genre == "pop" and r[0].mood == "happy"

    def test_lofi_ranks_first_for_lofi_user(self, two_rec, lofi_user):
        assert two_rec.recommend(lofi_user, k=1)[0].genre == "lofi"

    def test_returns_exact_k(self, two_rec, pop_user):
        assert len(two_rec.recommend(pop_user, k=2)) == 2

    def test_k_larger_than_catalog(self, two_rec, pop_user):
        assert len(two_rec.recommend(pop_user, k=999)) == len(two_rec.songs)

    def test_k_zero_returns_empty(self, two_rec, pop_user):
        assert two_rec.recommend(pop_user, k=0) == []

    def test_explain_non_empty_string(self, two_rec, pop_user):
        e = two_rec.explain_recommendation(pop_user, two_rec.songs[0])
        assert isinstance(e, str) and e.strip() != ""

    def test_explain_mismatch_still_string(self, two_rec, lofi_user):
        e = two_rec.explain_recommendation(lofi_user, two_rec.songs[0])
        assert isinstance(e, str)

    def test_different_users_different_top(self, two_rec, pop_user, lofi_user):
        assert (two_rec.recommend(pop_user, k=1)[0].genre !=
                two_rec.recommend(lofi_user, k=1)[0].genre)

# ---------------------------------------------------------------------------
# 3. score_song — every feature branch
# ---------------------------------------------------------------------------

class TestScoreSong:
    def test_non_negative_on_total_mismatch(self):
        p = {"genre": "classical", "mood": "sad", "energy": 0.1,
             "likes_acoustic": True, "preferred_mood_tags": [], "preferred_decade": ""}
        s = {"genre": "metal", "mood": "angry", "energy": 0.97, "valence": 0.2,
             "danceability": 0.5, "acousticness": 0.02, "mood_tags": "",
             "popularity": 5, "release_decade": "2010s"}
        sc, _ = score_song(p, s)
        assert sc >= 0.0

    def test_genre_match_adds_points(self, base_prefs, base_song):
        sc_m, _ = score_song(base_prefs, base_song)
        sc_n, _ = score_song(base_prefs, {**base_song, "genre": "country"})
        assert sc_m > sc_n

    def test_mood_match_adds_points(self, base_prefs, base_song):
        sc_m, _ = score_song(base_prefs, base_song)
        sc_n, _ = score_song(base_prefs, {**base_song, "mood": "angry"})
        assert sc_m > sc_n

    def test_energy_closer_scores_higher(self, base_prefs):
        close = {"genre":"x","mood":"x","energy":0.80,"valence":0.5,
                 "danceability":0.5,"acousticness":0.1,"mood_tags":"","popularity":50,"release_decade":""}
        far   = {**close, "energy": 0.10}
        sc_c, _ = score_song(base_prefs, close)
        sc_f, _ = score_song(base_prefs, far)
        assert sc_c > sc_f

    def test_energy_proximity_continuous(self, base_prefs):
        """Scores decrease monotonically as energy drifts from 0.8."""
        energies = [0.80, 0.70, 0.55, 0.40, 0.20]
        scores = []
        for e in energies:
            s = {"genre":"x","mood":"x","energy":e,"valence":0.5,"danceability":0.5,
                 "acousticness":0.1,"mood_tags":"","popularity":50,"release_decade":""}
            sc, _ = score_song(base_prefs, s)
            scores.append(sc)
        assert scores == sorted(scores, reverse=True)

    def test_acoustic_alignment(self):
        p = {"genre":"folk","mood":"relaxed","energy":0.35,"likes_acoustic":True,
             "preferred_mood_tags":[],"preferred_decade":""}
        ac = {"genre":"folk","mood":"relaxed","energy":0.35,"valence":0.7,
              "danceability":0.4,"acousticness":0.88,"mood_tags":"","popularity":60,"release_decade":""}
        el = {**ac, "acousticness": 0.10}
        sc_ac, _ = score_song(p, ac)
        sc_el, _ = score_song(p, el)
        assert sc_ac > sc_el

    def test_popularity_bonus(self, base_prefs):
        popular = {"genre":"pop","mood":"happy","energy":0.8,"valence":0.8,
                   "danceability":0.7,"acousticness":0.2,"mood_tags":"","release_decade":"","popularity":98}
        obscure = {**popular, "popularity": 2}
        sc_p, _ = score_song(base_prefs, popular)
        sc_o, _ = score_song(base_prefs, obscure)
        assert sc_p > sc_o

    def test_era_match_bonus(self):
        p = {"genre":"pop","mood":"happy","energy":0.8,"likes_acoustic":False,
             "preferred_mood_tags":[],"preferred_decade":"2020s"}
        matched = {"genre":"pop","mood":"happy","energy":0.8,"valence":0.8,
                   "danceability":0.7,"acousticness":0.2,"mood_tags":"","popularity":50,"release_decade":"2020s"}
        wrong   = {**matched, "release_decade": "1990s"}
        sc_m, _ = score_song(p, matched)
        sc_w, _ = score_song(p, wrong)
        assert sc_m > sc_w

    def test_no_era_pref_ignored(self, base_prefs, base_song):
        prefs_no_era = {**base_prefs, "preferred_decade": ""}
        sc_no, _ = score_song(prefs_no_era, base_song)
        prefs_era = {**base_prefs, "preferred_decade": "2020s"}
        sc_era, _ = score_song(prefs_era, base_song)
        assert sc_era >= sc_no

    def test_mood_tag_full_overlap(self):
        p = {"genre":"x","mood":"x","energy":0.5,"likes_acoustic":False,
             "preferred_mood_tags":["euphoric","bright"],"preferred_decade":""}
        full    = {"genre":"x","mood":"x","energy":0.5,"valence":0.5,"danceability":0.5,
                   "acousticness":0.3,"mood_tags":"euphoric|bright","popularity":50,"release_decade":""}
        no_tags = {**full, "mood_tags": ""}
        sc_f, _ = score_song(p, full)
        sc_n, _ = score_song(p, no_tags)
        assert sc_f > sc_n

    def test_partial_tag_overlap_less_than_full(self):
        p = {"genre":"x","mood":"x","energy":0.5,"likes_acoustic":False,
             "preferred_mood_tags":["euphoric","bright"],"preferred_decade":""}
        full    = {"genre":"x","mood":"x","energy":0.5,"valence":0.5,"danceability":0.5,
                   "acousticness":0.3,"mood_tags":"euphoric|bright","popularity":50,"release_decade":""}
        partial = {**full, "mood_tags": "euphoric"}
        sc_f, _ = score_song(p, full)
        sc_p, _ = score_song(p, partial)
        assert sc_f > sc_p

    def test_reasons_is_list(self, base_prefs, base_song):
        _, reasons = score_song(base_prefs, base_song)
        assert isinstance(reasons, list)

    def test_genre_reason_present_on_match(self, base_prefs, base_song):
        _, reasons = score_song(base_prefs, base_song)
        assert any("genre" in r.lower() for r in reasons)

    def test_score_is_float(self, base_prefs, base_song):
        sc, _ = score_song(base_prefs, base_song)
        assert isinstance(sc, float)

    def test_full_match_beats_partial(self, base_prefs, base_song):
        sc_full,    _ = score_song(base_prefs, base_song)
        sc_partial, _ = score_song(base_prefs, {**base_song, "mood": "angry"})
        assert sc_full > sc_partial

    def test_custom_weights_override(self, base_prefs, base_song):
        heavy = {**WEIGHTS, "genre": 10.0}
        light = {**WEIGHTS, "genre": 0.1}
        sc_h, _ = score_song(base_prefs, base_song, weights=heavy)
        sc_l, _ = score_song(base_prefs, base_song, weights=light)
        assert sc_h > sc_l

# ---------------------------------------------------------------------------
# 4. ScoringMode and MODE_WEIGHTS
# ---------------------------------------------------------------------------

class TestScoringModes:
    def test_all_four_modes_exist(self):
        values = [m.value for m in ScoringMode]
        for v in ["balanced","mood_first","energy_focus","genre_first"]:
            assert v in values

    def test_all_modes_non_negative(self, base_prefs, base_song):
        for mode in ScoringMode:
            sc, _ = score_song_with_weights(base_prefs, base_song, mode)
            assert sc >= 0.0

    def test_genre_first_highest_genre_weight(self):
        gw = {m: MODE_WEIGHTS[m]["genre"] for m in ScoringMode}
        assert gw[ScoringMode.GENRE_FIRST] == max(gw.values())

    def test_mood_first_highest_mood_weight(self):
        mw = {m: MODE_WEIGHTS[m]["mood"] for m in ScoringMode}
        assert mw[ScoringMode.MOOD_FIRST] == max(mw.values())

    def test_energy_focus_highest_energy_weight(self):
        ew = {m: MODE_WEIGHTS[m]["energy"] for m in ScoringMode}
        assert ew[ScoringMode.ENERGY_FOCUS] == max(ew.values())

    def test_all_weights_positive(self):
        for mode, weights in MODE_WEIGHTS.items():
            for feat, w in weights.items():
                assert w > 0, f"{mode.value}:{feat} non-positive"

    def test_max_score_matches_balanced_sum(self):
        assert MAX_POSSIBLE_SCORE == sum(WEIGHTS.values())

    def test_genre_first_genre_contribution_exceeds_balanced(self, base_prefs, base_song):
        """GENRE_FIRST should award more points specifically for genre, even if total differs."""
        genre_first_genre_weight  = MODE_WEIGHTS[ScoringMode.GENRE_FIRST]["genre"]
        balanced_genre_weight     = MODE_WEIGHTS[ScoringMode.BALANCED]["genre"]
        assert genre_first_genre_weight > balanced_genre_weight

    def test_mood_first_mood_contribution_exceeds_balanced(self, base_prefs, base_song):
        """MOOD_FIRST should award more points specifically for mood, even if total differs."""
        mood_first_mood_weight = MODE_WEIGHTS[ScoringMode.MOOD_FIRST]["mood"]
        balanced_mood_weight   = MODE_WEIGHTS[ScoringMode.BALANCED]["mood"]
        assert mood_first_mood_weight > balanced_mood_weight

# ---------------------------------------------------------------------------
# 5. recommend_songs functional API
# ---------------------------------------------------------------------------

class TestRecommendSongs:
    def test_returns_correct_k(self, songs, base_prefs):
        assert len(recommend_songs(base_prefs, songs, k=5)) == 5

    def test_k_larger_than_catalog(self, songs, base_prefs):
        assert len(recommend_songs(base_prefs, songs, k=999)) == len(songs)

    def test_k_zero_empty(self, songs, base_prefs):
        assert recommend_songs(base_prefs, songs, k=0) == []

    def test_sorted_descending(self, songs, base_prefs):
        scores = [sc for _, sc, _ in recommend_songs(base_prefs, songs, k=10)]
        assert scores == sorted(scores, reverse=True)

    def test_each_result_three_tuple(self, songs, base_prefs):
        for song, score, explanation in recommend_songs(base_prefs, songs, k=3):
            assert isinstance(song, dict)
            assert isinstance(score, float)
            assert isinstance(explanation, str)

    def test_top_matches_genre(self, songs):
        prefs = {"genre":"lofi","mood":"chill","energy":0.4,"likes_acoustic":True,
                 "preferred_mood_tags":[],"preferred_decade":""}
        top, _, _ = recommend_songs(prefs, songs, k=1)[0]
        assert top["genre"] == "lofi"

    def test_all_modes_produce_results(self, songs, base_prefs):
        for mode in ScoringMode:
            assert len(recommend_songs(base_prefs, songs, k=3, mode=mode)) == 3

    def test_diversity_no_artist_repeat_top3(self, songs):
        prefs = {"genre":"lofi","mood":"chill","energy":0.4,"likes_acoustic":True,
                 "preferred_mood_tags":[],"preferred_decade":""}
        results = recommend_songs(prefs, songs, k=5, apply_diversity_penalty=True)
        artists = [s["artist"] for s, _, _ in results[:3]]
        assert len(artists) == len(set(artists))

    def test_diversity_changes_order(self, songs):
        prefs = {"genre":"lofi","mood":"chill","energy":0.4,"likes_acoustic":True,
                 "preferred_mood_tags":[],"preferred_decade":""}
        without = [s["title"] for s,_,_ in recommend_songs(prefs, songs, k=5, apply_diversity_penalty=False)]
        with_p  = [s["title"] for s,_,_ in recommend_songs(prefs, songs, k=5, apply_diversity_penalty=True)]
        assert without != with_p

    def test_diversity_label_in_explanation(self, songs):
        prefs = {"genre":"lofi","mood":"chill","energy":0.4,"likes_acoustic":True,
                 "preferred_mood_tags":[],"preferred_decade":""}
        results = recommend_songs(prefs, songs, k=5, apply_diversity_penalty=True)
        exps = [e for _,_,e in results]
        assert any("diversity" in e.lower() for e in exps)

    def test_explanation_non_empty(self, songs, base_prefs):
        _, _, exp = recommend_songs(base_prefs, songs, k=1)[0]
        assert exp.strip() != ""

    def test_single_song_catalog(self):
        one = [{"id":99,"title":"Only","artist":"Solo","genre":"pop","mood":"happy",
                "energy":0.8,"valence":0.8,"danceability":0.7,"acousticness":0.2,
                "mood_tags":"","release_decade":"2020s","popularity":50}]
        prefs = {"genre":"pop","mood":"happy","energy":0.8,"likes_acoustic":False,
                 "preferred_mood_tags":[],"preferred_decade":""}
        results = recommend_songs(prefs, one, k=5)
        assert len(results) == 1 and results[0][0]["title"] == "Only"

# ---------------------------------------------------------------------------
# 6. load_songs CSV
# ---------------------------------------------------------------------------

class TestLoadSongs:
    def test_returns_list(self, songs):
        assert isinstance(songs, list)

    def test_20_songs(self, songs):
        assert len(songs) == 20

    def test_each_song_is_dict(self, songs):
        for s in songs:
            assert isinstance(s, dict)

    def test_core_fields_present(self, songs):
        required = {"id","title","artist","genre","mood","energy",
                    "tempo_bpm","valence","danceability","acousticness"}
        for s in songs:
            assert required.issubset(s.keys())

    def test_extended_fields_present(self, songs):
        for s in songs:
            assert "popularity" in s and "release_decade" in s and "mood_tags" in s

    def test_energy_float_in_range(self, songs):
        for s in songs:
            assert isinstance(s["energy"], float)
            assert 0.0 <= s["energy"] <= 1.0

    def test_valence_in_range(self, songs):
        for s in songs:
            assert 0.0 <= s["valence"] <= 1.0

    def test_danceability_in_range(self, songs):
        for s in songs:
            assert 0.0 <= s["danceability"] <= 1.0

    def test_acousticness_in_range(self, songs):
        for s in songs:
            assert 0.0 <= s["acousticness"] <= 1.0

    def test_id_is_int(self, songs):
        for s in songs:
            assert isinstance(s["id"], int)

    def test_ids_unique(self, songs):
        ids = [s["id"] for s in songs]
        assert len(ids) == len(set(ids))

    def test_genre_lowercase(self, songs):
        for s in songs:
            assert s["genre"] == s["genre"].lower()

    def test_mood_lowercase(self, songs):
        for s in songs:
            assert s["mood"] == s["mood"].lower()

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_songs("data/does_not_exist.csv")

    def test_popularity_in_range(self, songs):
        for s in songs:
            assert 0 <= s["popularity"] <= 100
