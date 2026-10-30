import asyncio
import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt

import core

st.set_page_config(page_title="TrustRec", page_icon="🎬", layout="wide")

PROFILE_LIMIT = 10


def _apply_theme():
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

        :root {
            --ink: #f6f7fb;
            --muted: #a7aec4;
            --panel: rgba(25, 30, 52, .78);
            --panel-strong: #1d2340;
            --line: rgba(255,255,255,.10);
            --purple: #8b7cff;
            --teal: #37d5c1;
            --gold: #ffc857;
        }

        .stApp {
            background:
                radial-gradient(circle at 85% 4%, rgba(108, 99, 255, .20), transparent 28rem),
                radial-gradient(circle at 8% 30%, rgba(34, 193, 181, .09), transparent 24rem),
                #0c1020;
            color: var(--ink);
            font-family: 'DM Sans', sans-serif;
        }
        [data-testid="stHeader"] { background: transparent; }
        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #151a32 0%, #101427 100%);
            border-right: 1px solid var(--line);
        }
        [data-testid="stSidebar"] > div:first-child { padding-top: 2rem; }
        h1, h2, h3, h4, [data-testid="stMetricValue"] {
            font-family: 'Space Grotesk', sans-serif;
            letter-spacing: -.03em;
        }
        h1 { font-size: clamp(2.4rem, 5vw, 4.5rem) !important; line-height: .98 !important; }
        h2, h3 { color: var(--ink); }
        p, label, .stCaption { color: var(--muted); }
        .hero {
            padding: 2.4rem 2.6rem 2.2rem;
            margin: .4rem 0 1.6rem;
            border: 1px solid var(--line);
            border-radius: 28px;
            background: linear-gradient(115deg, rgba(33, 39, 73, .94), rgba(20, 25, 47, .68));
            box-shadow: 0 24px 70px rgba(0,0,0,.24);
            overflow: hidden;
            position: relative;
        }
        .hero:after {
            content: '✦';
            position: absolute;
            right: 7%; top: -18px;
            color: rgba(139,124,255,.25);
            font-size: 13rem;
            line-height: 1;
        }
        .eyebrow {
            color: var(--teal);
            font-size: .75rem;
            font-weight: 700;
            letter-spacing: .16em;
            text-transform: uppercase;
        }
        .hero h1 { margin: .45rem 0 .7rem; position: relative; z-index: 1; }
        .hero p { max-width: 650px; font-size: 1.08rem; position: relative; z-index: 1; }
        .status-row { display: flex; gap: .6rem; flex-wrap: wrap; margin-top: 1.35rem; position: relative; z-index: 1; }
        .status-pill {
            display: inline-flex; align-items: center; gap: .45rem;
            padding: .5rem .75rem; border-radius: 999px;
            color: #dce1f6; background: rgba(255,255,255,.07);
            border: 1px solid var(--line); font-size: .82rem;
        }
        .status-pill strong { color: var(--ink); }
        .section-kicker {
            color: var(--purple); font-size: .72rem; font-weight: 700;
            letter-spacing: .14em; text-transform: uppercase; margin-bottom: -.3rem;
        }
        [data-baseweb="tab-list"] { gap: .45rem; border-bottom: 1px solid var(--line); }
        [data-baseweb="tab"] {
            height: 3.2rem; padding: 0 1.1rem; color: var(--muted);
            border-radius: 12px 12px 0 0; font-weight: 600;
        }
        [aria-selected="true"] { color: var(--ink) !important; background: rgba(139,124,255,.13); }
        [data-baseweb="tab-highlight"] { background: linear-gradient(90deg, var(--purple), var(--teal)); height: 3px; }
        [data-testid="stMetric"] {
            padding: 1rem 1.1rem; border: 1px solid var(--line);
            border-radius: 16px; background: var(--panel);
        }
        [data-testid="stMetricLabel"] { color: var(--muted); }
        [data-testid="stMetricValue"] { color: var(--ink); }
        .stButton > button {
            border: 1px solid rgba(139,124,255,.45);
            border-radius: 12px; color: white; font-weight: 700;
            background: linear-gradient(135deg, #7568ed, #5149bb);
            transition: transform .18s ease, box-shadow .18s ease;
        }
        .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 10px 24px rgba(108,99,255,.28); }
        [data-testid="stDataFrame"], [data-testid="stExpander"] {
            border: 1px solid var(--line); border-radius: 16px; overflow: hidden;
        }
        [data-testid="stExpander"] summary { background: rgba(255,255,255,.03); }
        div[data-testid="stAlert"] { border-radius: 14px; }
        hr { border-color: var(--line); }
        </style>
        """,
        unsafe_allow_html=True,
    )


_apply_theme()


def _load_local_env():
    """Load simple KEY=VALUE entries without overwriting existing environment values."""
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if not os.path.exists(env_path):
        return
    with open(env_path, encoding="utf-8") as env_file:
        for raw_line in env_file:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip("\"'")
            if key and key.upper() not in os.environ:
                os.environ[key.upper()] = value


_load_local_env()
AI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.8-flash")
AI_EXECUTOR = ThreadPoolExecutor(max_workers=1)


@st.cache_resource(show_spinner="Loading data and training models (first run only)...")
def setup():
    R, titles, G, is_real = core.load_data()
    Rtr, Rte = core.split(R)
    return {"titles": titles, "G": G, "real": is_real, "Rtr": Rtr, "Rte": Rte,
            "model": core.fit(Rtr), "detector": core.train_detector(Rtr)}


async def _create_ai_overviews(api_key, recommendations, anchors):
    """Create movie explanations without blocking Streamlit's recommendation flow."""
    from google import genai

    client = genai.Client(api_key=api_key)
    prompt = """You are CineCompass AI, a concise movie recommendation explainer.
For each recommended movie below, explain in one friendly sentence why it may appeal
to this user based on their highest-rated movies. Do not invent plot details,
cast, ratings, or genres. Return exactly one line per movie in the format:
MOVIE TITLE: explanation
Keep each explanation under 28 words.

User's highest-rated movies:
{anchors}

Recommended movies:
{recommendations}
""".format(
        anchors=", ".join(anchors),
        recommendations="\n".join(f"- {title}" for title in recommendations),
    )
    response = await client.aio.models.generate_content(model=AI_MODEL, contents=prompt)
    return response.text.strip()


def _run_ai_overviews(api_key, recommendations, anchors):
    return asyncio.run(_create_ai_overviews(api_key, recommendations, anchors))


def _start_ai_overviews(recommendations, anchors):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set. Add it to your environment and try again.")
    return AI_EXECUTOR.submit(_run_ai_overviews, api_key, recommendations, anchors)


def _format_ai_error(error):
    """Keep transient Gemini capacity errors friendly and actionable."""
    message = str(error).lower()
    high_traffic_markers = (
        "high demand",
        "resource exhausted",
        "rate limit",
        "429",
        "503",
        "overloaded",
        "unavailable",
    )
    if any(marker in message for marker in high_traffic_markers):
        return "please wait for a moment experiencing high traffic"
    return f"CineCompass AI could not respond: {error}"


D = setup()
titles, G, Rtr, Rte = D["titles"], D["G"], D["Rtr"], D["Rte"]

st.markdown(
    f"""
    <div class="hero">
        <div class="eyebrow">Personalized discovery · responsible AI</div>
        <h1>Find your next<br><span style="color:#8b7cff">favorite story.</span></h1>
        <p>TrustRec blends taste, context, and security to make recommendations you can actually trust.</p>
        <div class="status-row">
            <span class="status-pill">🎬 <strong>{len(titles):,}</strong>&nbsp; movies</span>
            <span class="status-pill">👥 <strong>{len(Rtr):,}</strong>&nbsp; user profiles</span>
            <span class="status-pill">🛡️ Hybrid model + attack defense</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
if not D["real"]:
    st.warning("Could not download MovieLens 100K, so synthetic demo data is in use. "
               "Place the real dataset at data/ml-100k/ for real results (see README).")

with st.sidebar:
    st.markdown("## 🎬 TrustRec")
    st.caption("Tune the recommendation engine")
    st.markdown('<div class="section-kicker">Recommendation engine</div>', unsafe_allow_html=True)
    alpha = st.slider("Hybrid weight (CF vs content)", 0.0, 1.0, 0.7, 0.1,
                      help="1.0 = pure item-based CF, 0.0 = pure content-based")
    st.divider()
    st.markdown('<div class="section-kicker">Safety laboratory</div>', unsafe_allow_html=True)
    st.subheader("Attack simulator")
    kind = st.selectbox("Attack type", core.ATTACKS)
    frac = st.slider("Fake users (% of real users)", 1, 10, 5) / 100
    pop = (Rtr > 0).sum(0)
    unpopular = np.argsort(pop)[100:400]  # obscure-but-rated movies make good targets
    target = st.selectbox("Target movie to push", unpopular,
                          format_func=lambda i: titles[i])
    seed = st.number_input("Random seed", 0, 9999, 1)

tab1, profile_tab, tab2, tab3 = st.tabs(
    ["1. Recommend", "2. My profile", "3. Attack and defend", "About"]
)

# ---------------------------------------------------------------- recommend
with tab1:
    user = st.number_input("User ID", 1, len(Rtr), 1) - 1
    profile_anchors = [i for i in np.argsort(-Rtr[user]) if Rtr[user, i] > 0][:5]

    scores, Pn = core.score(Rtr, D["model"], G, alpha)
    top = core.top_n(scores, Rtr, 10)[user]
    evidence = core.explain_recommendations(Rtr[user], D["model"], G, top, alpha)

    st.subheader(f"Top-10 for user {user + 1}")
    rows = []
    for item in evidence:
        i, j, g = item["item"], item["anchor"], item["genre"]
        why = (f"Similar to '{titles[j]}' (you rated {int(item['anchor_rating'])})"
               if j >= 0 else "Popular among similar items")
        rows.append({"Movie": titles[i], "Score": round(item["score"], 3),
                     "Why": why, "Best genre match": core.GENRES[g] if g >= 0 else "-"})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.markdown("#### Why these movies?")
    st.caption("Each bar decomposes the recommendation score into evidence from your similar-item "
               "taste profile and your genre profile. Hover for the exact values.")
    chart_rows = []
    movie_order = []
    for rank, item in enumerate(evidence, 1):
        movie = titles[item["item"]]
        movie_order.append(movie)
        for label, value in (("Similar items", item["cf_component"]),
                             ("Genre profile", item["content_component"])):
            chart_rows.append({"Movie": movie, "Evidence": label,
                               "Contribution": value, "Score": item["score"],
                               "Rank": rank})
    chart_data = pd.DataFrame(chart_rows)
    bars = alt.Chart(chart_data).mark_bar(cornerRadius=5).encode(
        x=alt.X("Contribution:Q", stack="zero", title="Contribution to hybrid score",
                scale=alt.Scale(domain=[0, max(chart_data["Score"].max() * 1.16, 0.1)])),
        y=alt.Y("Movie:N", sort=movie_order, title=None),
        color=alt.Color("Evidence:N", title="Evidence",
                        scale=alt.Scale(range=["#6C63FF", "#22C1B5"])),
        order=alt.Order("Evidence:N", sort="ascending"),
        tooltip=[alt.Tooltip("Movie:N", title="Movie"),
                 alt.Tooltip("Evidence:N", title="Evidence"),
                 alt.Tooltip("Contribution:Q", format=".3f"),
                 alt.Tooltip("Score:Q", format=".3f", title="Final score")],
    )
    labels = alt.Chart(chart_data.drop_duplicates("Movie")).mark_text(
        align="left", dx=5, color="#666"
    ).encode(
        x=alt.X("Score:Q"),
        y=alt.Y("Movie:N", sort=movie_order),
        text=alt.Text("Score:Q", format=".3f"),
    )
    st.altair_chart((bars + labels).properties(height=360), use_container_width=True)

    st.markdown("#### 🧭 CineCompass AI")
    st.caption("An optional Gemini-powered overview of why each recommendation matches "
               "your highest-rated movies. It runs separately and never changes the ranking.")
    ai_recommendations = [titles[item["item"]] for item in evidence]
    ai_anchors = [titles[i] for i in profile_anchors]
    if not ai_anchors:
        st.info("Rate at least one movie above to enable CineCompass AI.")
    elif st.button("Generate AI overviews", key="generate_ai"):
        try:
            st.session_state["ai_future"] = _start_ai_overviews(ai_recommendations, ai_anchors)
            st.session_state.pop("ai_result", None)
            st.session_state.pop("ai_error", None)
        except RuntimeError as error:
            st.session_state["ai_error"] = str(error)

    ai_future = st.session_state.get("ai_future")
    if ai_future is not None:
        if ai_future.done():
            try:
                st.session_state["ai_result"] = ai_future.result()
                st.session_state.pop("ai_error", None)
            except Exception as error:
                st.session_state["ai_error"] = _format_ai_error(error)
            st.session_state.pop("ai_future", None)
        else:
            st.info("CineCompass AI is preparing your movie map. Your recommendations remain available.")
            st.button("Refresh AI status", key="refresh_ai")
            st.stop()
    if st.session_state.get("ai_error"):
        st.error(st.session_state["ai_error"])
    elif st.session_state.get("ai_result"):
        st.markdown(st.session_state["ai_result"])

    with st.expander("Movies this user rated highest"):
        best = np.argsort(-Rtr[user])[:8]
        st.write(", ".join(f"{titles[b]} ({int(Rtr[user, b])})" for b in best if Rtr[user, b] > 0))

# -------------------------------------------------------------- custom profile
with profile_tab:
    st.subheader("✨ Create your own recommendation profile")
    st.caption(f"Select up to {PROFILE_LIMIT} movies and rate each from 0 to 5 in 0.5-point steps. "
               "These ratings are separate from the MovieLens User IDs.")
    profile_movies = st.multiselect(
        "Movies in your profile",
        options=list(range(len(titles))),
        format_func=lambda i: titles[i],
        max_selections=PROFILE_LIMIT,
        key="custom_profile_movies",
    )
    profile_ratings = {
        movie: st.slider(
            f"Your rating: {titles[movie]}",
            min_value=0.0,
            max_value=5.0,
            value=5.0,
            step=0.5,
            key=f"custom_profile_rating_{movie}",
        )
        for movie in profile_movies
    }
    recommendation_count = st.slider(
        "Number of recommendations", 1, 10, 10, key="custom_profile_count"
    )

    if not profile_movies:
        st.info("Select at least one movie to build your personal profile.")
    else:
        custom_profile = np.zeros(Rtr.shape[1])
        for movie, rating in profile_ratings.items():
            custom_profile[movie] = rating
        custom_profile = custom_profile.reshape(1, -1)
        custom_anchors = [movie for movie, _ in sorted(
            profile_ratings.items(), key=lambda pair: pair[1], reverse=True
        )]
        custom_scores, _ = core.score(custom_profile, D["model"], G, alpha)
        custom_scores[0, profile_movies] = -np.inf
        custom_top = np.argsort(-custom_scores[0])[:recommendation_count]
        custom_evidence = core.explain_recommendations(
            custom_profile[0], D["model"], G, custom_top, alpha
        )

        st.subheader(f"Top-{len(custom_top)} for your profile")
        custom_rows = []
        for item in custom_evidence:
            i, j, g = item["item"], item["anchor"], item["genre"]
            custom_rows.append({
                "Movie": titles[i],
                "Score": round(item["score"], 3),
                "Why": (f"Similar to '{titles[j]}' (you rated {item['anchor_rating']:.1f})"
                        if j >= 0 else "Popular among similar items"),
                "Best genre match": core.GENRES[g] if g >= 0 else "-",
            })
        st.dataframe(pd.DataFrame(custom_rows), use_container_width=True, hide_index=True)

        st.markdown("#### 🧭 CineCompass AI")
        st.caption("Generate optional Gemini explanations independently of the recommender.")
        if st.button("Generate AI overviews", key="generate_custom_ai"):
            try:
                st.session_state["custom_ai_future"] = _start_ai_overviews(
                    [titles[item["item"]] for item in custom_evidence],
                    [titles[i] for i in custom_anchors],
                )
                st.session_state.pop("custom_ai_result", None)
                st.session_state.pop("custom_ai_error", None)
            except RuntimeError as error:
                st.session_state["custom_ai_error"] = str(error)

        custom_future = st.session_state.get("custom_ai_future")
        if custom_future is not None:
            if custom_future.done():
                try:
                    st.session_state["custom_ai_result"] = custom_future.result()
                    st.session_state.pop("custom_ai_error", None)
                except Exception as error:
                    st.session_state["custom_ai_error"] = _format_ai_error(error)
                st.session_state.pop("custom_ai_future", None)
            else:
                st.info("CineCompass AI is preparing your movie map. Your recommendations remain available.")
                st.button("Refresh AI status", key="refresh_custom_ai")
        if st.session_state.get("custom_ai_error"):
            st.error(st.session_state["custom_ai_error"])
        elif st.session_state.get("custom_ai_result"):
            st.markdown(st.session_state["custom_ai_result"])

# ------------------------------------------------------------ attack/defend
with tab2:
    st.write(f"**Target:** {titles[target]}  |  **Attack:** {kind}  |  "
             f"**Fake users:** {int(len(Rtr) * frac)}")
    if st.button("Run attack and defense", type="primary"):
        with st.spinner("Simulating..."):
            m, sc, det = core.run_scenarios(Rtr, Rte, G, D["model"], D["detector"],
                                            kind, frac, target, alpha, int(seed))
            tops = {n: core.top_n(s, Rtr, 10) for n, s in sc.items()}
        st.session_state["res"] = (m, det, tops, target)

    if "res" in st.session_state:
        m, det, tops, tgt = st.session_state["res"]
        df = pd.DataFrame(m).T
        st.subheader("Impact on the system")
        c = st.columns(3)
        for col, name in zip(c, ["Clean", "Attacked", "Defended"]):
            col.metric(f"{name}: target hit ratio@10", f"{m[name]['Target hit ratio@10']:.1%}")
        st.dataframe(df.style.format({"Precision@10": "{:.4f}", "Target hit ratio@10": "{:.1%}",
                                      "Target avg rank": "{:.0f}"}), use_container_width=True)
        st.bar_chart(df[["Target hit ratio@10"]])

        st.subheader("Detector performance")
        d = st.columns(5)
        d[0].metric("Fake injected", det["Fake profiles injected"])
        d[1].metric("Fake caught", det["Fake caught"])
        d[2].metric("Real wrongly flagged", det["Real users wrongly flagged"])
        d[3].metric("Recall", f"{det['Recall']:.0%}")
        d[4].metric("Precision", f"{det['Precision']:.0%}")

        st.subheader("Same user, three worlds")
        u = st.number_input("User ID to compare", 1, len(Rtr), 1, key="cmp") - 1
        cols = st.columns(3)
        for col, name in zip(cols, ["Clean", "Attacked", "Defended"]):
            col.markdown(f"**{name}**")
            for r, i in enumerate(tops[name][u], 1):
                col.write(f"{r}. {'🎯 ' if i == tgt else ''}{titles[i]}")
    else:
        st.info("Choose attack settings in the sidebar and click the button.")

# -------------------------------------------------------------------- about
with tab3:
    st.markdown("""
**Pipeline:** item-based CF + content-based (genre profiles) are blended into a hybrid recommender.
A shilling attack injects fake profiles that rate a target movie 5 and mimic real users on filler items.
A Random Forest detector (features: rating count, mean, std, RDMA, popularity share, share of 5s) flags
suspicious profiles, which are removed before the model is retrained.

**Metrics:** Precision@10 on a 10% held-out split (accuracy) and target hit ratio@10 (the share of users
who are shown the target in their top-10, i.e. attack success).

**Course mapping:** content-based (CO2), item CF and metrics (CO3), attacks and detection (CO4),
hybridization (CO5).
""")
