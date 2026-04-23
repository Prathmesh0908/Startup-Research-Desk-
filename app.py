import json
import os
import re

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from agents import run_research_pipeline
from api_service import MODEL

load_dotenv()

st.set_page_config(
    page_title="Startup Research Desk",
    page_icon="🔍",
    layout="wide",
)

st.title("🔍 Startup Research Desk")
st.caption("AI-powered VC diligence workspace for polished investment briefs")

st.markdown(
    """
    <style>
    .detail-card {
        background: linear-gradient(180deg, rgba(27,34,51,.95), rgba(18,22,34,.98));
        border: 1px solid rgba(118, 135, 172, 0.22);
        border-radius: 18px;
        padding: 1rem 1rem 0.8rem 1rem;
        margin-bottom: 0.9rem;
        box-shadow: 0 14px 30px rgba(0, 0, 0, 0.18);
    }
    .detail-label {
        color: #9fb3d9;
        font-size: 0.78rem;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.45rem;
    }
    .detail-value {
        color: #f5f7fb;
        font-size: 1.05rem;
        line-height: 1.55;
    }
    .score-chip {
        display: inline-block;
        background: rgba(67, 97, 238, 0.18);
        color: #dfe8ff;
        border: 1px solid rgba(95, 129, 255, 0.26);
        border-radius: 999px;
        padding: 0.2rem 0.65rem;
        font-size: 0.84rem;
        margin-bottom: 0.6rem;
    }
    .subtle-card {
        background: linear-gradient(180deg, rgba(20,24,38,.98), rgba(14,18,29,.98));
        border: 1px solid rgba(118, 135, 172, 0.18);
        border-radius: 20px;
        padding: 1.1rem 1.1rem 0.9rem 1.1rem;
        margin-bottom: 1rem;
    }
    .section-kicker {
        color: #8fb0ff;
        font-size: 0.75rem;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }
    .section-title {
        color: #f4f7ff;
        font-size: 1.25rem;
        font-weight: 600;
        margin-bottom: 0.35rem;
    }
    .section-copy {
        color: #c7d3eb;
        line-height: 1.65;
        font-size: 0.98rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def _extract_json_blob(text):
    match = re.search(r"\{[\s\S]*\}", text)
    return match.group(0) if match else None


def _parse_analysis_content(content):
    if isinstance(content, (dict, list)):
        return content, None
    if not isinstance(content, str):
        return None, str(content)

    blob = _extract_json_blob(content)
    if blob:
        try:
            return json.loads(blob), None
        except json.JSONDecodeError:
            pass
    return None, content.replace("```json", "").replace("```", "").strip()


def _render_section_intro(kicker, title, copy):
    st.markdown(
        f"""
        <div class="subtle-card">
            <div class="section-kicker">{kicker}</div>
            <div class="section-title">{title}</div>
            <div class="section-copy">{copy}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _render_market_view(parsed, raw_text):
    _render_section_intro(
        "Market Signal",
        "Market Opportunity Snapshot",
        "A synthesized view of category size, momentum, trendlines, and recent movement around the company.",
    )
    if isinstance(parsed, dict):
        top = st.columns(3)
        top[0].metric("Market Size", parsed.get("market_size", "N/A"))
        top[1].metric("Growth Rate", parsed.get("growth_rate", "N/A"))
        top[2].metric("Confidence", f"{parsed.get('confidence_score', 'N/A')}/5")

        trends, news = st.columns(2)
        with trends:
            st.markdown("#### Key Trends")
            for item in parsed.get("key_trends", []) or ["No trends captured."]:
                st.markdown(f"- {item}")
        with news:
            st.markdown("#### Recent News")
            for item in parsed.get("recent_news", []) or ["No recent news captured."]:
                st.markdown(f"- {item}")
    elif raw_text:
        st.markdown("#### Research Notes")
        st.write(raw_text)
    else:
        st.info("No market details were generated for this run.")


def _render_founders_view(parsed, raw_text):
    _render_section_intro(
        "Team Read",
        "Founder Assessment",
        "A structured look at founder background, domain fit, exits, and execution risks surfaced during research.",
    )
    if isinstance(parsed, dict):
        headline = st.columns(3)
        headline[0].metric("Team Score", f"{parsed.get('team_score', 'N/A')}/5")
        headline[1].metric("Domain Expertise", parsed.get("domain_expertise", "N/A"))
        headline[2].metric("Notable Exits", parsed.get("notable_exits", "None noted"))

        founders = parsed.get("founders", [])
        if founders:
            for founder in founders:
                name = founder.get("name", "Unknown Founder")
                role = founder.get("role", "Role unavailable")
                background = founder.get("background", [])
                st.markdown(
                    f"""
                    <div class="detail-card">
                        <div class="detail-label">{role}</div>
                        <div class="detail-value">{name}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if isinstance(background, list) and background:
                    for item in background:
                        st.markdown(f"- {item}")

        st.markdown("#### Red Flags")
        for item in parsed.get("red_flags", []) or ["No material red flags surfaced."]:
            st.markdown(f"- {item}")
    elif raw_text:
        st.markdown("#### Research Notes")
        st.write(raw_text)
    else:
        st.info("No founder details were generated for this run.")


def _render_competition_view(parsed, raw_text):
    _render_section_intro(
        "Competitive Edge",
        "Landscape and Moat Review",
        "An at-a-glance view of competitors, differentiation, strengths, weaknesses, and defensibility.",
    )
    if isinstance(parsed, dict):
        overview = st.columns(3)
        overview[0].metric("Competitive Score", f"{parsed.get('competitive_score', 'N/A')}/5")
        overview[1].metric("Differentiation", parsed.get("differentiation", "N/A"))
        overview[2].metric("Moat", parsed.get("moat", "N/A"))

        competitors = parsed.get("competitors", [])
        if competitors:
            for competitor in competitors:
                st.markdown(
                    f"""
                    <div class="detail-card">
                        <div class="detail-label">{competitor.get("name", "Competitor")}</div>
                        <div class="detail-value">{competitor.get("positioning", "No positioning captured.")}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                cols = st.columns(2)
                with cols[0]:
                    st.markdown("**Strengths**")
                    for item in competitor.get("strengths", []) or ["No strengths captured."]:
                        st.markdown(f"- {item}")
                with cols[1]:
                    st.markdown("**Weaknesses**")
                    for item in competitor.get("weaknesses", []) or ["No weaknesses captured."]:
                        st.markdown(f"- {item}")
    elif raw_text:
        st.markdown("#### Research Notes")
        st.write(raw_text)
    else:
        st.info("No competition details were generated for this run.")

hero_left, hero_right = st.columns([1.3, 0.7])

with hero_left:
    startup_name = st.text_input(
        "Startup Name",
        placeholder="e.g. Notion",
    )
    domain = st.text_input(
        "Domain (optional)",
        placeholder="e.g. notion.so",
    )

with hero_right:
    st.markdown("### What You Get")
    st.markdown(
        "- Market sizing and trend scan\n"
        "- Founder and team assessment\n"
        "- Competitor mapping and moat view\n"
        "- VC-style investment brief with QA scoring"
    )

run = st.button("Generate Investment Brief", type="primary", use_container_width=True)

if run:
    if not startup_name:
        st.error("Please enter startup name.")
        st.stop()

    api_key = os.getenv("GROQ_API_KEY", "")
    tavily_key = os.getenv("TAVILY_API_KEY", "")

    if not api_key or not tavily_key:
        st.error("Missing backend configuration. Please set `GROQ_API_KEY` and `TAVILY_API_KEY` in your environment.")
        st.stop()

    client = Groq(api_key=api_key)
    status_box = st.empty()
    progress = st.progress(0)
    status_steps = []

    def log_step(msg):
        status_steps.append(msg)
        progress.progress(min(100, max(10, len(status_steps) * 8)))
        status_box.info("\n".join(status_steps[-4:]))

    try:
        with st.spinner("Running backend research pipeline..."):
            result = run_research_pipeline(
                client,
                startup_name=startup_name,
                domain=domain,
                log_step=log_step,
            )

        progress.progress(100)

        judge = result.get("judge", {}) or {}
        overall_score = judge.get("overall_score", "N/A")
        if isinstance(overall_score, float) and overall_score.is_integer():
            overall_score = int(overall_score)
        top_strength = judge.get("top_strength", "Pending")
        top_improvement = judge.get("top_improvement", "Pending")

        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric("Research Score", f"{overall_score}/5")
        metric_2.metric("Top Strength", top_strength)
        metric_3.metric("Key Improvement", top_improvement)

        st.markdown("## Investment Brief")
        st.markdown(result["report"])

        st.download_button(
            "Download Report",
            result["report"],
            file_name=f"{startup_name}_report.md",
            use_container_width=True,
        )

        review_tab, research_tab = st.tabs(["Quality Review", "Internal Research Details"])

        with review_tab:
            summary = judge.get("summary")
            if summary:
                st.markdown(
                    f"""
                    <div class="detail-card">
                        <div class="detail-label">Executive Readout</div>
                        <div class="detail-value">{summary}</div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            scores = judge.get("scores", {}) or {}
            if scores:
                score_items = list(scores.items())
                for index in range(0, len(score_items), 3):
                    score_columns = st.columns(min(3, len(score_items) - index))
                    for column, (label, item) in zip(score_columns, score_items[index:index + 3]):
                        score_value = item.get("score", "N/A") if isinstance(item, dict) else "N/A"
                        reasoning = item.get("reasoning", "") if isinstance(item, dict) else ""
                        title = label.replace("_", " ").title()
                        with column:
                            st.markdown(
                                f"""
                                <div class="detail-card">
                                    <div class="detail-label">{title}</div>
                                    <div class="score-chip">{score_value}/5</div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )
                            if reasoning:
                                st.caption(reasoning)
            else:
                st.json(judge)

        with research_tab:
            analysis = result.get("analysis", {}) or {}
            market_panel, founders_panel, competition_panel = st.tabs(
                ["Market", "Founders", "Competition"]
            )

            market_parsed, market_raw = _parse_analysis_content(analysis.get("market"))
            founders_parsed, founders_raw = _parse_analysis_content(analysis.get("founders"))
            competition_parsed, competition_raw = _parse_analysis_content(analysis.get("competition"))

            with market_panel:
                _render_market_view(market_parsed, market_raw)

            with founders_panel:
                _render_founders_view(founders_parsed, founders_raw)

            with competition_panel:
                _render_competition_view(competition_parsed, competition_raw)

        st.success("Investment brief generated successfully.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        status_box.empty()
