"""
app.py — VibeFinder 2.0 Streamlit UI
=====================================
Run with:  streamlit run app.py
"""

from src.agent.critic import critique
from src.rag.retriever import Retriever
from src.rag.vector_store import VectorStore
from src.reliability.logger import log_decision, read_log
from src.reliability.guardrails import validate_query, VALID_GENRES, VALID_MOODS
from src.recommender import load_songs, recommend_songs, ScoringMode, MODE_WEIGHTS
import streamlit as st
import sys
import os
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

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 2rem 2.5rem 4rem; max-width: 1100px; }

/* ── Hero header ── */
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
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -40px; left: 30%;
    width: 300px; height: 150px;
    background: radial-gradient(ellipse, rgba(16,185,129,0.08) 0%, transparent 70%);
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
.hero-sub {
    font-size: 1rem;
    color: #7c7c9a;
    font-weight: 300;
    letter-spacing: 0.02em;
}
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
}

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
    position: relative;
    transition: all 0.2s ease;
}
.result-card:hover {
    border-color: #3730a3;
    background: #0e0e20;
    transform: translateX(3px);
}
.result-rank {
    font-family: 'Space Mono', monospace;
    font-size: 0.7rem;
    color: #4b4b6b;
    letter-spacing: 0.1em;
}
.result-title {
    font-size: 1.15rem;
    font-weight: 600;
    color: #ffffff;
    margin: 0.15rem 0;
}
.result-artist {
    font-size: 0.85rem;
    color: #6366f1;
    font-weight: 400;
}
.result-meta {
    font-size: 0.78rem;
    color: #4b4b6b;
    margin-top: 0.3rem;
    font-family: 'Space Mono', monospace;
}
.score-bar-wrap {
    background: #1a1a30;
    border-radius: 4px;
    height: 6px;
    margin: 0.75rem 0 0.5rem;
    overflow: hidden;
}
.score-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #4f46e5, #10b981);
    transition: width 0.6s ease;
}
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
}

/* ── Confidence badge ── */
.conf-high   { color: #10b981; font-weight: 600; }
.conf-medium { color: #f59e0b; font-weight: 600; }
.conf-low    { color: #ef4444; font-weight: 600; }

/* ── Rerank notice ── */
.rerank-notice {
    background: rgba(245,158,11,0.08);
    border: 1px solid rgba(245,158,11,0.2);
    border-radius: 8px;
    padding: 0.6rem 1rem;
    font-size: 0.82rem;
    color: #fbbf24;
    margin-bottom: 1rem;
    font-family: 'Space Mono', monospace;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #08080f !important;
    border-right: 1px solid #1a1a30 !important;
}
section[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

/* ── Streamlit widgets ── */
.stSelectbox > div > div,
.stTextInput > div > div > input,
.stTextArea > div > div > textarea {
    background: #0f0f1e !important;
    border: 1px solid #2a2a45 !important;
    border-radius: 8px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
}
.stSlider > div > div > div { color: #6366f1 !important; }
.stButton > button {
    background: linear-gradient(135deg, #4f46e5, #6366f1) !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important;
    font-size: 0.95rem !important;
    width: 100% !important;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.85 !important; }

/* ── Tab ── */
.stTabs [data-baseweb="tab-list"] {
    background: #0f0f1e;
    border-radius: 8px;
    padding: 4px;
    gap: 2px;
    border: 1px solid #1e1e3a;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #4b4b6b !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.05em !important;
    border-radius: 6px !important;
}
.stTabs [aria-selected="true"] {
    background: #1e1e3a !important;
    color: #e8e8f0 !important;
}

/* ── Log table ── */
.log-entry {
    background: #0c0c18;
    border: 1px solid #1a1a30;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    font-size: 0.78rem;
    font-family: 'Space Mono', monospace;
    color: #6b7280;
}
.log-entry span { color: #a5b4fc; }
</style>
""", unsafe_allow_html=True)


# ── Data loading (cached so it doesn't reload on every interaction) ───────────
@st.cache_data
def get_songs():
    return load_songs("data/songs.csv")


@st.cache_resource
def get_store(song_count):
    songs = get_songs()
    store = VectorStore()
    store.build(songs)
    return store


# ── Hero header ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Vibe<span>Finder</span> 2.0</div>
    <div class="hero-sub">AI-augmented music recommendations — RAG · Agentic critic · Reliability scoring</div>
    <div style="margin-top:1rem">
        <span class="badge">RAG</span>
        <span class="badge">Agentic</span>
        <span class="badge">Guardrails</span>
        <span class="badge">Logged</span>
    </div>
</div>
""", unsafe_allow_html=True)


# ── Sidebar — user profile builder ───────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="font-family:'Space Mono',monospace;font-size:0.7rem;
                letter-spacing:0.15em;text-transform:uppercase;
                color:#4b4b6b;margin-bottom:1.2rem">
        Taste Profile
    </div>
    """, unsafe_allow_html=True)

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
            height=100,
        )
        user_input = nl_query.strip()
        structured_profile = None

    else:
        genre = st.selectbox("Genre", sorted(VALID_GENRES))
        mood = st.selectbox("Mood",  sorted(VALID_MOODS))
        energy = st.slider("Energy", 0.0, 1.0, 0.7, 0.05,
                           help="0 = very calm, 1 = maximum intensity")
        likes_acoustic = st.toggle("Prefer acoustic", value=False)

        decade = st.selectbox(
            "Preferred era (optional)",
            ["", "1980s", "1990s", "2000s", "2010s", "2020s"],
        )
        tags_input = st.text_input(
            "Mood tags (comma-separated, optional)",
            placeholder="euphoric, bright, driving",
        )
        tags = [t.strip() for t in tags_input.split(",") if t.strip()]

        structured_profile = {
            "genre": genre,
            "mood": mood,
            "energy": energy,
            "likes_acoustic": likes_acoustic,
            "preferred_decade": decade,
            "preferred_mood_tags": tags,
        }
        user_input = structured_profile

    st.divider()
    scoring_mode_label = st.selectbox(
        "Scoring mode",
        ["BALANCED", "MOOD_FIRST", "ENERGY_FOCUS", "GENRE_FIRST"],
        help="Changes which features the engine prioritises"
    )
    k = st.slider("Results to show", 3, 10, 5)
    diversity_penalty = st.toggle("Diversity penalty", value=True,
                                  help="Reduces score if artist already appears in top results")

    st.divider()
    run_btn = st.button("Find my songs →", use_container_width=True)


# ── Main content ──────────────────────────────────────────────────────────────
tab_results, tab_how, tab_log = st.tabs(
    ["RECOMMENDATIONS", "HOW IT WORKS", "DECISION LOG"])


# ── Tab 1: Results ────────────────────────────────────────────────────────────
with tab_results:
    if not run_btn:
        st.markdown("""
        <div class="card" style="text-align:center;padding:3rem 2rem;border-style:dashed">
            <div style="font-size:2.5rem;margin-bottom:1rem">🎧</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.8rem;
                        color:#4b4b6b;letter-spacing:0.1em">
                BUILD YOUR PROFILE AND HIT "FIND MY SONGS"
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Guardrails ────────────────────────────────────────────────────────
        clean_profile, errors = validate_query(
            user_input if user_input else "")
        hard_errors = [e for e in errors if "cannot be" in e or "too short" in e
                       or "not supported" in e]

        if hard_errors:
            st.error("**Input rejected by guardrails:**\n" +
                     "\n".join(f"• {e}" for e in hard_errors))
        else:
            if errors:
                with st.expander("⚠️ Guardrail warnings", expanded=False):
                    for e in errors:
                        st.warning(e)

            # ── RAG retrieval ─────────────────────────────────────────────────
            songs = get_songs()
            store = get_store(len(songs))
            retriever = Retriever(store)

            if isinstance(user_input, str) and user_input:
                query_text = user_input
            elif structured_profile:
                parts = [structured_profile.get("genre", ""),
                         structured_profile.get("mood", "")]
                parts += structured_profile.get("preferred_mood_tags", [])
                query_text = " ".join(p for p in parts if p)
                clean_profile = structured_profile
            else:
                query_text = ""

            top_similarity = retriever.top_similarity(
                query_text) if query_text else 0.5
            candidates = retriever.retrieve(query_text, k=min(
                len(songs), 15)) if query_text else songs

            # ── Scoring ───────────────────────────────────────────────────────
            mode = ScoringMode[scoring_mode_label]
            results = recommend_songs(
                clean_profile, candidates, k=k,
                mode=mode,
                apply_diversity_penalty=diversity_penalty,
            )

            # ── Agentic critic ────────────────────────────────────────────────
            critique_result = critique(
                query_text, clean_profile, results, top_similarity
            )

            rerank_triggered = False
            if critique_result.should_rerank:
                rerank_triggered = True
                fallback_mode = ScoringMode.MOOD_FIRST if mode != ScoringMode.MOOD_FIRST else ScoringMode.ENERGY_FOCUS
                results = recommend_songs(
                    clean_profile, candidates, k=k,
                    mode=fallback_mode,
                    apply_diversity_penalty=diversity_penalty,
                )
                critique_result = critique(
                    query_text, clean_profile, results, top_similarity, attempt=2)

            # ── Logger ────────────────────────────────────────────────────────
            log_decision(
                query=query_text,
                parsed_profile=clean_profile,
                rag_candidates=[s.get("id", 0) for s in candidates],
                top_results=[{"id": s.get("id"), "title": s.get("title"), "score": round(sc, 2)}
                             for s, sc, _ in results],
                confidence=critique_result.confidence,
                rerank_triggered=rerank_triggered,
                critic_note=critique_result.note,
                guardrail_errors=errors,
                mode=scoring_mode_label,
            )

            # ── Confidence header ─────────────────────────────────────────────
            conf = critique_result.confidence
            conf_label = critique_result.label()
            conf_class = {"HIGH": "conf-high",
                          "MEDIUM": "conf-medium", "LOW": "conf-low"}[conf_label]

            col_conf, col_mode, col_rag = st.columns(3)
            with col_conf:
                st.markdown(f"""
                <div class="card" style="padding:1rem;text-align:center">
                    <div class="section-label" style="margin-top:0">Confidence</div>
                    <div class="result-title {conf_class}">{conf_label}</div>
                    <div class="result-meta">{conf:.2f} / 1.00</div>
                </div>""", unsafe_allow_html=True)
            with col_mode:
                st.markdown(f"""
                <div class="card" style="padding:1rem;text-align:center">
                    <div class="section-label" style="margin-top:0">Scoring mode</div>
                    <div class="result-title" style="font-size:0.95rem">{scoring_mode_label}</div>
                    <div class="result-meta">{"re-ranked" if rerank_triggered else "first pass"}</div>
                </div>""", unsafe_allow_html=True)
            with col_rag:
                st.markdown(f"""
                <div class="card" style="padding:1rem;text-align:center">
                    <div class="section-label" style="margin-top:0">RAG similarity</div>
                    <div class="result-title" style="font-size:0.95rem">{top_similarity:.2f}</div>
                    <div class="result-meta">{"strong match" if top_similarity > 0.4 else "weak match"}</div>
                </div>""", unsafe_allow_html=True)

            if rerank_triggered:
                st.markdown("""
                <div class="rerank-notice">
                    ↻ Agentic critic detected low confidence — re-ranked using fallback mode
                </div>""", unsafe_allow_html=True)

            # ── Critic note ───────────────────────────────────────────────────
            st.markdown(f"""
            <div class="section-label">Critic note</div>
            <div class="card" style="padding:0.8rem 1.2rem;font-size:0.83rem;color:#9ca3af;font-style:italic">
                {critique_result.note}
            </div>""", unsafe_allow_html=True)

            # ── Recommendation cards ──────────────────────────────────────────
            st.markdown('<div class="section-label">Top results</div>',
                        unsafe_allow_html=True)

            MAX_SCORE = 11.5
            for rank, (song, score, explanation) in enumerate(results, 1):
                pct = min(score / MAX_SCORE, 1.0) * 100
                bar_color = "#10b981" if pct > 75 else "#6366f1" if pct > 45 else "#4b5563"

                reasons_html = "".join(
                    f'<span class="reason-pill">{r.strip()}</span>'
                    for r in explanation.split(";") if r.strip()
                )

                mood_tags = song.get("mood_tags", "").replace("|", " · ")
                era = song.get("release_decade", "")

                st.markdown(f"""
                <div class="result-card">
                    <div class="result-rank">#{rank}</div>
                    <div class="result-title">{song['title']}</div>
                    <div class="result-artist">{song['artist']}</div>
                    <div class="result-meta">
                        {song['genre']} &nbsp;·&nbsp; {song['mood']} &nbsp;·&nbsp;
                        energy {song['energy']:.2f}
                        {"&nbsp;·&nbsp;" + era if era else ""}
                    </div>
                    <div class="score-bar-wrap">
                        <div class="score-bar-fill" style="width:{pct:.1f}%;background:{bar_color}"></div>
                    </div>
                    <div style="display:flex;justify-content:space-between;
                                align-items:center;margin-bottom:0.5rem">
                        <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#4b4b6b">
                            score {score:.2f}
                        </div>
                        <div style="font-family:'Space Mono',monospace;font-size:0.7rem;color:#4b4b6b">
                            {pct:.0f}% match
                        </div>
                    </div>
                    {f'<div style="margin-top:0.4rem;font-size:0.72rem;color:#4b4b6b">tags: {mood_tags}</div>' if mood_tags else ''}
                    <div style="margin-top:0.6rem">{reasons_html}</div>
                </div>
                """, unsafe_allow_html=True)


# ── Tab 2: How it works ───────────────────────────────────────────────────────
with tab_how:
    st.markdown("""
    <div class="section-label" style="margin-top:0.5rem">System pipeline</div>
    """, unsafe_allow_html=True)

    steps = [
        ("1 · Guardrails", "#4b5563",
         "Every input is validated before touching any AI component. Empty queries, out-of-range energy values, and null fields are caught here. Hard failures abort the pipeline; soft warnings are logged and the pipeline continues."),
        ("2 · RAG Retriever", "#6366f1",
         "Your query is converted into a feature vector using a TF-IDF bag-of-words approach over genre, mood, and tag tokens. The vector store finds the 15 most semantically similar songs by cosine similarity. This is the Retrieval-Augmented Generation step — finding candidates before scoring."),
        ("3 · Scoring Engine", "#8b5cf6",
         "Each candidate song is scored against your taste profile using six weighted features: genre (×3.0), mood (×2.0), energy proximity (×2.0), acousticness alignment (×1.0), mood tags (×1.0), valence (×1.0), popularity and era bonuses. Songs are ranked descending by total score."),
        ("4 · Agentic Critic", "#10b981",
         "The critic evaluates the top-k results and assigns a confidence score (0–1) based on three signals: genre/mood coverage, score spread, and RAG similarity. If confidence < 0.55, it triggers a re-rank using a different scoring mode. This is the agentic loop — the system checks its own work."),
        ("5 · Logger", "#f59e0b",
         "Every run writes a structured JSON record to logs/decisions.json: timestamp, query, parsed profile, RAG candidates, top results, confidence, re-rank flag, and critic note. This is the full audit trail."),
    ]

    for title, color, desc in steps:
        st.markdown(f"""
        <div class="card" style="border-left:3px solid {color};padding-left:1.2rem">
            <div style="font-family:'Space Mono',monospace;font-size:0.8rem;
                        color:{color};margin-bottom:0.4rem">{title}</div>
            <div style="font-size:0.88rem;color:#9ca3af;line-height:1.6">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="section-label">Scoring weights (BALANCED mode)</div>
    <div class="card">
    """, unsafe_allow_html=True)

    weights_data = {
        "Genre match": (3.0, "#6366f1"),
        "Mood match": (2.0, "#8b5cf6"),
        "Energy proximity": (2.0, "#a78bfa"),
        "Mood tag overlap": (1.0, "#10b981"),
        "Acousticness": (1.0, "#34d399"),
        "Valence": (1.0, "#6ee7b7"),
        "Danceability": (0.5, "#4b5563"),
        "Popularity bonus": (0.5, "#4b5563"),
        "Era match": (0.5, "#4b5563"),
    }
    max_w = 3.0
    for label, (w, color) in weights_data.items():
        pct = (w / max_w) * 100
        st.markdown(f"""
        <div style="display:flex;align-items:center;gap:1rem;margin-bottom:0.5rem">
            <div style="width:140px;font-size:0.78rem;color:#9ca3af;
                        font-family:'Space Mono',monospace">{label}</div>
            <div style="flex:1;background:#1a1a30;border-radius:4px;height:8px;overflow:hidden">
                <div style="width:{pct:.0f}%;height:100%;background:{color};border-radius:4px"></div>
            </div>
            <div style="width:30px;text-align:right;font-size:0.75rem;
                        color:#6366f1;font-family:'Space Mono',monospace">×{w}</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ── Tab 3: Decision log ───────────────────────────────────────────────────────
with tab_log:
    st.markdown('<div class="section-label" style="margin-top:0.5rem">Recent decisions</div>',
                unsafe_allow_html=True)

    records = read_log(n=10)
    if not records:
        st.markdown("""
        <div class="card" style="text-align:center;padding:2rem;border-style:dashed">
            <div style="font-family:'Space Mono',monospace;font-size:0.8rem;color:#4b4b6b">
                NO LOG ENTRIES YET — RUN A QUERY FIRST
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for rec in reversed(records):
            ts = rec.get("timestamp", "")[:19].replace("T", " ")
            query = rec.get("query", "—")[:60]
            conf = rec.get("confidence", 0)
            rerank = "↻ re-ranked" if rec.get(
                "rerank_triggered") else "first pass"
            mode = rec.get("mode", "balanced")
            conf_col = "#10b981" if conf >= 0.8 else "#f59e0b" if conf >= 0.55 else "#ef4444"
            top = rec.get("top_results", [])
            top_str = ", ".join(r.get("title", "?") for r in top[:3])

            st.markdown(f"""
            <div class="log-entry">
                <div style="display:flex;justify-content:space-between;margin-bottom:0.4rem">
                    <span>{ts}</span>
                    <span style="color:{conf_col}">conf {conf:.2f} · {rerank}</span>
                </div>
                <div><span>query:</span> {query}</div>
                <div><span>mode:</span> {mode} &nbsp;·&nbsp; <span>top songs:</span> {top_str}</div>
            </div>
            """, unsafe_allow_html=True)

        if st.button("Clear log", type="secondary"):
            from src.reliability.logger import clear_log
            clear_log()
            st.rerun()
