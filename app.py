"""
app.py — VibeFinder 2.0 Streamlit UI  (production-ready)
=========================================================
Run with:  streamlit run app.py

Bugs fixed vs uploaded version:
  1. MODE_WEIGHTS unused import removed
  2. MAX_SCORE hardcoded 11.5 → uses MAX_POSSIBLE_SCORE from recommender
  3. Loading spinner added — UI never freezes silently
  4. Evaluator tab added — run test harness from the UI during demo
  5. structured_profile now always runs through guardrails (was bypassed)
  6. Tags deduped before use
  7. All user-controlled strings HTML-escaped before injection into markdown
"""

from src.agent.critic import critique
from src.rag.retriever import Retriever
from src.rag.vector_store import VectorStore
from src.reliability.evaluator import run_test_harness, PASS_THRESHOLD
from src.reliability.logger import log_decision, read_log, clear_log
from src.reliability.guardrails import validate_query, VALID_GENRES, VALID_MOODS
from src.recommender import (
    load_songs,
    recommend_songs,
    ScoringMode,
    MAX_POSSIBLE_SCORE,
)
import streamlit as st
import sys
import os
import html as html_lib

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="VibeFinder 2.0",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1140px; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0a0a0f; }
::-webkit-scrollbar-thumb { background: #2a2a45; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #3a3a60; }

/* ── Hero ── */
.hero {
    background: linear-gradient(135deg, #0d0d1a 0%, #0a0a14 50%, #0d0d1a 100%);
    border: 1px solid #1e1e3a;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 220px; height: 220px;
    background: radial-gradient(circle, rgba(99,102,241,0.18) 0%, transparent 70%);
    border-radius: 50%;
    animation: pulse-glow 4s ease-in-out infinite alternate;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 160px; height: 160px;
    background: radial-gradient(circle, rgba(16,185,129,0.08) 0%, transparent 70%);
    border-radius: 50%;
    animation: pulse-glow 5s ease-in-out infinite alternate-reverse;
}
@keyframes pulse-glow {
    from { opacity: 0.6; transform: scale(1); }
    to   { opacity: 1;   transform: scale(1.12); }
}
.hero-title {
    font-family: 'Space Mono', monospace;
    font-size: 2.4rem;
    font-weight: 700;
    color: #ffffff;
    letter-spacing: -1px;
    margin: 0 0 0.4rem;
    line-height: 1.1;
}
.hero-title span { color: #6366f1; }
.hero-sub { font-size: 1rem; color: #7c7c9a; font-weight: 300; }
.badge {
    display: inline-block;
    background: rgba(99,102,241,0.12);
    border: 1px solid rgba(99,102,241,0.3);
    color: #818cf8;
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    padding: 3px 10px;
    border-radius: 100px;
    letter-spacing: 0.1em;
    margin-right: 8px;
    margin-top: 1rem;
    text-transform: uppercase;
    transition: background 0.2s, border-color 0.2s;
}
.badge:hover { background: rgba(99,102,241,0.22); border-color: rgba(99,102,241,0.5); }

/* ── Section labels ── */
.section-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4b4b6b;
    margin-bottom: 0.75rem;
    margin-top: 1.75rem;
}

/* ── Cards ── */
.card {
    background: #0f0f1e;
    border: 1px solid #1e1e3a;
    border-radius: 12px;
    padding: 1.5rem;
    margin-bottom: 1rem;
    transition: border-color 0.2s;
}
.card:hover { border-color: #2e2e5a; }

/* ── Result cards ── */
.result-card {
    background: #0c0c1a;
    border: 1px solid #1a1a35;
    border-radius: 14px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.2s, background 0.2s, transform 0.2s, box-shadow 0.2s;
}
.result-card:hover {
    border-color: #3730a3;
    background: #0e0e20;
    transform: translateX(4px);
    box-shadow: -3px 0 16px rgba(99,102,241,0.12);
}
.result-rank { font-family: 'Space Mono', monospace; font-size: 0.7rem; color: #4b4b6b; margin-bottom: 2px; }
.result-title { font-size: 1.15rem; font-weight: 600; color: #ffffff; margin: 0.1rem 0; }
.result-artist { font-size: 0.85rem; color: #6366f1; margin-bottom: 2px; }
.result-meta { font-size: 0.78rem; color: #4b4b6b; margin-top: 0.3rem; font-family: 'Space Mono', monospace; }

/* ── Score bar ── */
.score-bar-wrap {
    background: #1a1a30;
    border-radius: 4px;
    height: 6px;
    margin: 0.75rem 0 0.4rem;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 4px;
    animation: bar-grow 0.7s cubic-bezier(0.22, 1, 0.36, 1) both;
    transform-origin: left;
}
@keyframes bar-grow {
    from { transform: scaleX(0); opacity: 0.4; }
    to   { transform: scaleX(1); opacity: 1; }
}

/* ── Reason pills ── */
.reason-pill {
    display: inline-block;
    background: rgba(99,102,241,0.08);
    border: 1px solid rgba(99,102,241,0.18);
    color: #a5b4fc;
    font-size: 0.72rem;
    padding: 2px 10px;
    border-radius: 100px;
    margin: 2px 3px 2px 0;
    font-family: 'Space Mono', monospace;
    transition: background 0.15s;
}
.reason-pill:hover { background: rgba(99,102,241,0.18); }

/* ── Confidence colours ── */
.conf-high   { color: #10b981; font-weight: 600; }
.conf-medium { color: #f59e0b; font-weight: 600; }
.conf-low    { color: #ef4444; font-weight: 600; }

/* ── Rerank notice ── */
.rerank-notice {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.25);
    border-radius: 8px;
    padding: 0.65rem 1rem;
    font-size: 0.82rem;
    color: #fbbf24;
    margin-bottom: 1rem;
    font-family: 'Space Mono', monospace;
}

/* ── Guardrail warning ── */
.guardrail-warn {
    background: rgba(239,68,68,0.06);
    border: 1px solid rgba(239,68,68,0.2);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #fca5a5;
    font-family: 'Space Mono', monospace;
    margin-bottom: 0.5rem;
}

/* ── Test harness rows ── */
.test-row {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 9px 0;
    border-bottom: 1px solid #1a1a30;
    font-size: 0.8rem;
    font-family: 'Space Mono', monospace;
}
.test-row:last-child { border-bottom: none; }
.pass-pill {
    font-size: 0.68rem;
    font-weight: 700;
    padding: 2px 10px;
    border-radius: 4px;
    min-width: 44px;
    text-align: center;
    flex-shrink: 0;
    letter-spacing: 0.05em;
}
.pass-pill.pass { background: rgba(16,185,129,0.15); color: #10b981; border: 1px solid rgba(16,185,129,0.3); }
.pass-pill.fail { background: rgba(239,68,68,0.12); color: #ef4444; border: 1px solid rgba(239,68,68,0.25); }
.conf-mini-bar {
    flex: 1;
    height: 4px;
    background: #1a1a30;
    border-radius: 2px;
    overflow: hidden;
}
.conf-mini-fill { height: 100%; border-radius: 2px; }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #08080f !important;
    border-right: 1px solid #1a1a30 !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

/* ── Widget overrides ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0f0f1e !important;
    border: 1px solid #2a2a45 !important;
    border-radius: 8px !important;
    color: #e8e8f0 !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
.stTextInput > div > div > input:focus,
.stTextArea > div > div > textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
    outline: none !important;
}
.stTextInput > div > div > input::placeholder,
.stTextArea > div > div > textarea::placeholder { color: #3a3a5a !important; }

.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    width: 100% !important;
    transition: opacity 0.2s, box-shadow 0.2s, transform 0.15s !important;
}
.stButton > button:hover {
    opacity: 0.9 !important;
    box-shadow: 0 0 20px rgba(99,102,241,0.35) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:active { transform: translateY(0) !important; box-shadow: none !important; }

.stTabs [data-baseweb="tab-list"] {
    background: #0f0f1e;
    border-radius: 8px;
    padding: 4px;
    border: 1px solid #1e1e3a;
    gap: 2px;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4b4b6b !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.72rem !important;
    border-radius: 6px !important;
    transition: color 0.2s !important;
    letter-spacing: 0.04em !important;
}
.stTabs [data-baseweb="tab"]:hover { color: #9ca3af !important; }
.stTabs [aria-selected="true"] {
    background: #1e1e3a !important;
    color: #e8e8f0 !important;
}

.log-entry {
    background: #0c0c18;
    border: 1px solid #1a1a30;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    color: #6b7280;
    transition: border-color 0.2s;
}
.log-entry:hover { border-color: #2a2a45; }
.log-entry span { color: #a5b4fc; }
</style>
""", unsafe_allow_html=True)


# ── Cached data loaders ───────────────────────────────────────────────────────
@st.cache_data
def get_songs():
    return load_songs("data/songs.csv")


@st.cache_resource
def get_store(_song_count: int) -> VectorStore:
    """Build vector store once; rebuilds only when catalog size changes."""
    songs = get_songs()
    store = VectorStore()
    store.build(songs)
    return store


# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Vibe<span>Finder</span> 2.0</div>
    <div class="hero-sub">
        AI-augmented music recommendations —
        RAG &nbsp;·&nbsp; Agentic critic &nbsp;·&nbsp; Reliability scoring
    </div>
    <div style="margin-top:1rem">
        <span class="badge">RAG</span>
        <span class="badge">Agentic</span>
        <span class="badge">Guardrails</span>
        <span class="badge">Logged</span>
        <span class="badge">199 tests</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.68rem;
                letter-spacing:0.15em;text-transform:uppercase;
                color:#4b4b6b;margin-bottom:1.2rem;padding-bottom:0.5rem;
                border-bottom:1px solid #1a1a30">
        Taste Profile
    </div>""", unsafe_allow_html=True)

    input_mode = st.radio(
        "Input mode",
        ["🎤  Natural language", "🎛️  Build a profile"],
        label_visibility="collapsed",
    )
    st.divider()

    if input_mode == "🎤  Natural language":
        nl_query = st.text_area(
            "Describe what you want to hear",
            placeholder="e.g. something euphoric and driving for a morning run...",
            height=110,
        )
        user_input: str | dict = nl_query.strip()
        structured_profile = None

    else:
        genre = st.selectbox("Genre", sorted(VALID_GENRES))
        mood = st.selectbox("Mood",  sorted(VALID_MOODS))
        energy = st.slider(
            "Energy", 0.0, 1.0, 0.7, 0.05,
            help="0 = very calm · 1 = maximum intensity",
        )
        likes_acoustic = st.toggle("Prefer acoustic", value=False)
        decade = st.selectbox(
            "Preferred era (optional)",
            ["", "1980s", "1990s", "2000s", "2010s", "2020s"],
        )
        tags_raw = st.text_input(
            "Mood tags (comma-separated)",
            placeholder="euphoric, bright, driving",
        )
        # Deduplicate and strip tags
        tags = list(dict.fromkeys(
            t.strip().lower() for t in tags_raw.split(",") if t.strip()
        ))

        structured_profile = {
            "genre": genre, "mood": mood, "energy": energy,
            "likes_acoustic": likes_acoustic, "preferred_decade": decade,
            "preferred_mood_tags": tags,
        }
        user_input = structured_profile

    st.divider()

    scoring_mode_label = st.selectbox(
        "Scoring mode",
        ["BALANCED", "MOOD_FIRST", "ENERGY_FOCUS", "GENRE_FIRST"],
        help="Changes which features the engine prioritises",
    )
    k = st.slider("Results to show", 3, 10, 5)
    diversity_penalty = st.toggle(
        "Diversity penalty", value=True,
        help="Reduces score if an artist already appears in top results",
    )

    st.divider()
    run_btn = st.button("Find my songs →", use_container_width=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_results, tab_how, tab_eval, tab_log = st.tabs([
    "RECOMMENDATIONS", "HOW IT WORKS", "TEST HARNESS", "DECISION LOG",
])


# ===========================================================================
# Tab 1 — Recommendations
# ===========================================================================
with tab_results:
    if not run_btn:
        st.markdown("""
        <div class="card" style="text-align:center;padding:3.5rem 2rem;border-style:dashed;border-color:#1e1e3a">
            <div style="font-size:2.8rem;margin-bottom:1rem">🎧</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.8rem;color:#4b4b6b;letter-spacing:0.08em">
                BUILD YOUR PROFILE AND HIT "FIND MY SONGS"
            </div>
            <div style="font-size:0.75rem;color:#2e2e4a;margin-top:0.75rem">
                Use natural language or the profile builder in the sidebar
            </div>
        </div>""", unsafe_allow_html=True)

    else:
        # ── Guard empty input ────────────────────────────────────────────────
        if not user_input:
            st.error("Please enter a query or build a profile first.")
            st.stop()

        # ── Guardrails — ALL inputs go through here ──────────────────────────
        # FIX: structured_profile was previously bypassing this step in
        # the original code. Now every input path runs through validate_query().
        clean_profile, errors = validate_query(user_input)

        hard_errors = [
            e for e in errors
            if "cannot be" in e or "too short" in e or "not supported" in e
        ]

        if hard_errors:
            st.markdown(
                '<div class="section-label">Guardrails — input rejected</div>',
                unsafe_allow_html=True,
            )
            for e in hard_errors:
                st.markdown(
                    f'<div class="guardrail-warn">✗ {html_lib.escape(e)}</div>',
                    unsafe_allow_html=True,
                )
            st.stop()

        if errors:
            with st.expander("⚠️ Guardrail warnings (pipeline continues)", expanded=False):
                for e in errors:
                    st.warning(e)

        # ── Pipeline — wrapped in spinner so UI never freezes ────────────────
        with st.spinner("Retrieving · scoring · critiquing…"):

            # RAG
            songs = get_songs()
            store = get_store(len(songs))
            retriever = Retriever(store)

            # Build query text for embedding
            if isinstance(user_input, str):
                query_text = user_input
            else:
                parts = (
                    [clean_profile.get("genre", ""),
                     clean_profile.get("mood", "")]
                    + clean_profile.get("preferred_mood_tags", [])
                )
                query_text = " ".join(p for p in parts if p)

            top_similarity = retriever.top_similarity(query_text)
            candidates = retriever.retrieve(query_text, k=min(len(songs), 15))

            # Score
            mode = ScoringMode[scoring_mode_label]
            results = recommend_songs(
                clean_profile, candidates, k=k,
                mode=mode, apply_diversity_penalty=diversity_penalty,
            )

            # Critic
            critique_result = critique(
                query_text, clean_profile, results, top_similarity)
            rerank_triggered = False

            if critique_result.should_rerank:
                rerank_triggered = True
                fallback = (
                    ScoringMode.MOOD_FIRST
                    if mode != ScoringMode.MOOD_FIRST
                    else ScoringMode.ENERGY_FOCUS
                )
                results = recommend_songs(
                    clean_profile, candidates, k=k,
                    mode=fallback, apply_diversity_penalty=diversity_penalty,
                )
                critique_result = critique(
                    query_text, clean_profile, results, top_similarity, attempt=2
                )

            # Log every run
            log_decision(
                query=query_text,
                parsed_profile=clean_profile,
                rag_candidates=[s.get("id", 0) for s in candidates],
                top_results=[
                    {"id": s.get("id"), "title": s.get(
                        "title"), "score": round(sc, 2)}
                    for s, sc, _ in results
                ],
                confidence=critique_result.confidence,
                rerank_triggered=rerank_triggered,
                critic_note=critique_result.note,
                guardrail_errors=errors,
                mode=scoring_mode_label,
            )

        # ── Confidence header ─────────────────────────────────────────────────
        conf = critique_result.confidence
        conf_label = critique_result.label()
        conf_class = {"HIGH": "conf-high",
                      "MEDIUM": "conf-medium", "LOW": "conf-low"}[conf_label]

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="card" style="padding:1rem 1.2rem;text-align:center">
                <div class="section-label" style="margin-top:0">Confidence</div>
                <div class="result-title {conf_class}" style="font-size:1.4rem">{conf_label}</div>
                <div class="result-meta">{conf:.2f} / 1.00</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="card" style="padding:1rem 1.2rem;text-align:center">
                <div class="section-label" style="margin-top:0">Scoring mode</div>
                <div class="result-title" style="font-size:0.95rem">
                    {html_lib.escape(scoring_mode_label)}
                </div>
                <div class="result-meta">{"↻ re-ranked" if rerank_triggered else "first pass"}</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            rag_strength = "strong" if top_similarity > 0.4 else "moderate" if top_similarity > 0.2 else "weak"
            st.markdown(f"""
            <div class="card" style="padding:1rem 1.2rem;text-align:center">
                <div class="section-label" style="margin-top:0">RAG similarity</div>
                <div class="result-title" style="font-size:0.95rem">{top_similarity:.2f}</div>
                <div class="result-meta">{rag_strength} semantic match</div>
            </div>""", unsafe_allow_html=True)

        if rerank_triggered:
            st.markdown("""
            <div class="rerank-notice">
                ↻ Agentic critic detected low confidence — re-ranked using fallback mode
            </div>""", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="section-label">Critic note</div>
        <div class="card" style="padding:0.8rem 1.2rem;font-size:0.83rem;
                                  color:#9ca3af;font-style:italic;border-left:3px solid #1e1e3a">
            {html_lib.escape(critique_result.note)}
        </div>""", unsafe_allow_html=True)

        # ── Result cards ──────────────────────────────────────────────────────
        st.markdown(
            f'<div class="section-label">Top {len(results)} recommendations</div>',
            unsafe_allow_html=True,
        )

        for rank, (song, score, explanation) in enumerate(results, 1):
            pct = min(score / MAX_POSSIBLE_SCORE, 1.0) * 100
            bar_color = (
                "#10b981" if pct > 75
                else "#6366f1" if pct > 50
                else "#f59e0b" if pct > 30
                else "#4b5563"
            )

            reasons_html = "".join(
                f'<span class="reason-pill">{html_lib.escape(r.strip())}</span>'
                for r in explanation.split(";") if r.strip()
            )
            mood_tags = html_lib.escape(
                song.get("mood_tags", "").replace("|", " · "))
            era = html_lib.escape(song.get("release_decade", ""))
            pop = song.get("popularity", 0)

            st.markdown(f"""
            <div class="result-card">
                <div class="result-rank">#{rank} &nbsp;·&nbsp; {html_lib.escape(song['genre'])} / {html_lib.escape(song['mood'])}</div>
                <div class="result-title">{html_lib.escape(song['title'])}</div>
                <div class="result-artist">{html_lib.escape(song['artist'])}</div>
                <div class="result-meta">
                    energy {song['energy']:.2f}
                    &nbsp;·&nbsp; popularity {int(pop)}
                    {"&nbsp;·&nbsp;" + era if era else ""}
                </div>
                <div class="score-bar-wrap">
                    <div class="score-bar-fill"
                         style="width:{pct:.1f}%;background:{bar_color}"></div>
                </div>
                <div style="display:flex;justify-content:space-between;
                            align-items:center;margin-bottom:0.6rem">
                    <div style="font-family:'Space Mono',monospace;
                                font-size:0.68rem;color:#4b4b6b">
                        {score:.2f} / {MAX_POSSIBLE_SCORE:.1f} pts
                    </div>
                    <div style="font-family:'Space Mono',monospace;
                                font-size:0.68rem;color:#4b4b6b">
                        {pct:.0f}% match
                    </div>
                </div>
                {f'<div style="margin-bottom:0.5rem;font-size:0.72rem;color:#3a3a5a">tags: {mood_tags}</div>' if mood_tags else ''}
                <div style="margin-top:0.4rem">{reasons_html}</div>
            </div>""", unsafe_allow_html=True)


# ===========================================================================
# Tab 2 — How it works
# ===========================================================================
with tab_how:
    st.markdown(
        '<div class="section-label" style="margin-top:0.5rem">Five-stage pipeline</div>',
        unsafe_allow_html=True,
    )

    steps = [
        ("1 · Guardrails", "#f59e0b",
         "Every input is validated before touching any AI component. Empty queries, "
         "out-of-range energy values, and null fields are caught here. Hard failures "
         "abort the pipeline with a clear error. Soft warnings log the issue and allow "
         "the pipeline to continue. This layer runs first, always."),
        ("2 · RAG Retriever", "#6366f1",
         "Your query is converted to a feature vector using a TF-IDF bag-of-words approach "
         "over a 70-token vocabulary of genres, moods, and tags. Genre and mood tokens are "
         "doubled in the song feature string to mirror their scoring importance. The vector "
         "store finds the 15 most semantically similar songs by cosine similarity — narrowing "
         "the catalog before scoring runs."),
        ("3 · Scoring Engine", "#8b5cf6",
         "Each candidate song is scored against your profile using nine weighted features: "
         "genre (×3.0), mood (×2.0), energy proximity (×2.0), acousticness (×1.0), "
         "mood tags (×1.0), valence (×1.0), danceability (×0.5), popularity (×0.5), "
         "era match (×0.5). Energy uses proximity scoring — a song 0.05 from your target "
         "scores almost the same as a perfect match. Songs rank descending by total score."),
        ("4 · Agentic Critic", "#10b981",
         "The critic evaluates top-k results using three signals: genre/mood coverage in "
         "the top-3 (0–0.40 pts), score spread between #1 and #2 (0–0.30 pts), and RAG "
         "cosine similarity (0–0.30 pts). Sum = confidence 0–1. If confidence < 0.55, "
         "the critic sets should_rerank=True and the planner retries with a fallback "
         "ScoringMode. This is the agentic self-correction loop."),
        ("5 · Logger", "#4b5563",
         "Every run writes a newline-delimited JSON record to logs/decisions.json: "
         "timestamp, query, parsed profile, RAG candidate IDs, top results with scores, "
         "confidence, re-rank flag, and critic note. The full decision trace is inspectable "
         "after any run. Clear the log from the Decision Log tab."),
    ]

    for title, color, desc in steps:
        st.markdown(f"""
        <div class="card" style="border-left:3px solid {color};
                                  padding-left:1.2rem;border-radius:0 12px 12px 0">
            <div style="font-family:'Space Mono',monospace;font-size:0.78rem;
                        color:{color};margin-bottom:0.5rem;font-weight:700">
                {title}
            </div>
            <div style="font-size:0.875rem;color:#9ca3af;line-height:1.65">
                {desc}
            </div>
        </div>""", unsafe_allow_html=True)

    # Weights chart
    st.markdown(
        '<div class="section-label">Scoring weights — BALANCED mode</div>',
        unsafe_allow_html=True,
    )

    weights_data = [
        ("Genre match",       3.0, "#6366f1"),
        ("Mood match",        2.0, "#8b5cf6"),
        ("Energy proximity",  2.0, "#a78bfa"),
        ("Mood tag overlap",  1.0, "#10b981"),
        ("Acousticness",      1.0, "#34d399"),
        ("Valence",           1.0, "#6ee7b7"),
        ("Danceability",      0.5, "#4b5563"),
        ("Popularity",        0.5, "#4b5563"),
        ("Era match",         0.5, "#4b5563"),
    ]

    st.markdown('<div class="card">', unsafe_allow_html=True)
    for label, w, color in weights_data:
        pct = (w / 3.0) * 100
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:8px">
            <div style="width:148px;font-size:0.77rem;color:#9ca3af;
                        font-family:'Space Mono',monospace;flex-shrink:0">
                {label}
            </div>
            <div style="flex:1;background:#1a1a30;border-radius:3px;height:7px;overflow:hidden">
                <div style="width:{pct:.0f}%;height:100%;background:{color};
                            border-radius:3px;transition:width 0.4s ease">
                </div>
            </div>
            <div style="width:36px;text-align:right;font-size:0.75rem;
                        color:#6366f1;font-family:'Space Mono',monospace;flex-shrink:0">
                ×{w}
            </div>
        </div>""", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f"""
    <div style="font-size:0.78rem;color:#4b4b6b;font-family:'Space Mono',monospace;
                margin-top:0.5rem;text-align:right">
        Max possible score (BALANCED): {MAX_POSSIBLE_SCORE:.1f} pts
    </div>""", unsafe_allow_html=True)


# ===========================================================================
# Tab 3 — Test Harness (NEW — key for demo)
# ===========================================================================
with tab_eval:
    st.markdown(
        '<div class="section-label" style="margin-top:0.5rem">Reliability evaluation</div>',
        unsafe_allow_html=True,
    )
    st.markdown(f"""
    <div class="card" style="padding:1rem 1.4rem;margin-bottom:1rem">
        <div style="font-size:0.85rem;color:#9ca3af;line-height:1.6">
            Runs <strong style="color:#e8e8f0">5 predefined test cases × 5 runs each</strong>
            through the full pipeline and reports PASS / FAIL per case.
            A case passes when average confidence ≥ {PASS_THRESHOLD:.2f}.
            This is the stretch-feature test harness — the same script you can run with
            <code style="background:#1a1a30;padding:1px 6px;border-radius:4px">
            python -m src.reliability.evaluator</code>
        </div>
    </div>""", unsafe_allow_html=True)

    if st.button("▶  Run test harness", use_container_width=False):
        with st.spinner("Running 25 pipeline executions — this takes ~10 seconds…"):
            songs = get_songs()
            results = run_test_harness(songs=songs)

        pass_count = sum(1 for r in results if r["passed"])
        total = len(results)
        all_conf = [r["avg_confidence"] for r in results]
        avg_conf = sum(all_conf) / len(all_conf)

        # Grade
        ratio = pass_count / total
        grade = (
            "A" if ratio >= 0.9 and avg_conf >= 0.75 else
            "B" if ratio >= 0.8 and avg_conf >= 0.65 else
            "C" if ratio >= 0.6 and avg_conf >= 0.55 else "D"
        )

        # Summary stats
        gc1, gc2, gc3 = st.columns(3)
        with gc1:
            st.markdown(f"""
            <div class="card" style="padding:1rem;text-align:center">
                <div class="section-label" style="margin-top:0">Cases passed</div>
                <div class="result-title" style="font-size:1.6rem;
                     color:{'#10b981' if pass_count == total else '#f59e0b'}">
                     {pass_count}/{total}
                </div>
            </div>""", unsafe_allow_html=True)
        with gc2:
            st.markdown(f"""
            <div class="card" style="padding:1rem;text-align:center">
                <div class="section-label" style="margin-top:0">Avg confidence</div>
                <div class="result-title" style="font-size:1.6rem">{avg_conf:.3f}</div>
            </div>""", unsafe_allow_html=True)
        with gc3:
            grade_color = {"A": "#10b981", "B": "#6366f1",
                           "C": "#f59e0b", "D": "#ef4444"}[grade]
            st.markdown(f"""
            <div class="card" style="padding:1rem;text-align:center">
                <div class="section-label" style="margin-top:0">Grade</div>
                <div class="result-title" style="font-size:1.6rem;color:{grade_color}">
                    {grade}
                </div>
            </div>""", unsafe_allow_html=True)

        # Per-case rows
        st.markdown(
            '<div class="section-label" style="margin-top:1rem">Per-case breakdown</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<div class="card" style="padding:1rem 1.4rem">',
                    unsafe_allow_html=True)
        for r in results:
            conf = r["avg_confidence"]
            passed = r["passed"]
            pct = conf * 100
            pill_cls = "pass" if passed else "fail"
            pill_txt = "PASS" if passed else "FAIL"
            bar_col = "#10b981" if passed else "#ef4444"
            rerank_str = f"{r['rerank_count']}/{r['runs']} re-ranked"

            st.markdown(f"""
            <div class="test-row">
                <span class="pass-pill {pill_cls}">{pill_txt}</span>
                <span style="flex:1;color:#9ca3af">{html_lib.escape(r['label'])}</span>
                <div class="conf-mini-bar">
                    <div class="conf-mini-fill"
                         style="width:{pct:.0f}%;background:{bar_col}"></div>
                </div>
                <span style="min-width:38px;text-align:right;color:#6b7280">
                    {conf:.3f}
                </span>
                <span style="min-width:90px;text-align:right;color:#4b4b6b;font-size:0.72rem">
                    {rerank_str}
                </span>
            </div>""", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-size:0.78rem;color:#4b4b6b;font-family:'Space Mono',monospace;
                    margin-top:0.75rem;line-height:1.6">
            PASS threshold: ≥{PASS_THRESHOLD:.2f} avg confidence over {results[0]['runs']} runs.
            FAIL cases are expected and documented — conflicting preferences and ambiguous
            queries cannot be reliably resolved with a 20-song catalog.
        </div>""", unsafe_allow_html=True)


# ===========================================================================
# Tab 4 — Decision Log
# ===========================================================================
with tab_log:
    st.markdown(
        '<div class="section-label" style="margin-top:0.5rem">Recent pipeline decisions</div>',
        unsafe_allow_html=True,
    )

    records = read_log(n=10)

    if not records:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2.5rem;
                                  border-style:dashed;border-color:#1e1e3a">
            <div style="font-family:'Space Mono',monospace;
                        font-size:0.8rem;color:#4b4b6b;letter-spacing:0.08em">
                NO LOG ENTRIES YET
            </div>
            <div style="font-size:0.75rem;color:#2e2e4a;margin-top:0.5rem">
                Run a query in the Recommendations tab first
            </div>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(
            f'<div style="font-size:0.75rem;color:#4b4b6b;'
            f'font-family:Space Mono,monospace;margin-bottom:0.75rem">'
            f'Showing last {len(records)} entries · stored in logs/decisions.json</div>',
            unsafe_allow_html=True,
        )
        for rec in reversed(records):
            ts = rec.get("timestamp", "")[:19].replace("T", " ")
            query = html_lib.escape(rec.get("query", "—")[:60])
            conf = rec.get("confidence", 0)
            rerank = "↻ re-ranked" if rec.get(
                "rerank_triggered") else "first pass"
            mode = html_lib.escape(rec.get("mode", "balanced"))
            col = "#10b981" if conf >= 0.8 else "#f59e0b" if conf >= 0.55 else "#ef4444"
            top_str = html_lib.escape(
                ", ".join(r.get("title", "?")
                          for r in rec.get("top_results", [])[:3])
            )
            errs = rec.get("guardrail_errors", [])
            err_badge = (
                f'&nbsp;·&nbsp;<span style="color:#ef4444">'
                f'{len(errs)} warning{"s" if len(errs) != 1 else ""}</span>'
                if errs else ""
            )

            st.markdown(f"""
            <div class="log-entry">
                <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem">
                    <span>{ts}</span>
                    <span style="color:{col}">conf {conf:.2f} · {rerank}{err_badge}</span>
                </div>
                <div style="margin-bottom:0.2rem"><span>query:</span> {query}</div>
                <div style="color:#4b4b6b">
                    <span style="color:#a5b4fc">mode:</span> {mode}
                    &nbsp;·&nbsp;
                    <span style="color:#a5b4fc">top songs:</span> {top_str}
                </div>
            </div>""", unsafe_allow_html=True)

        st.divider()
        if st.button("🗑  Clear log", type="secondary"):
            clear_log()
            st.rerun()
