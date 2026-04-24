import os
import textwrap

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
    .entity-card {
        background: linear-gradient(180deg, rgba(23,29,44,.98), rgba(16,20,31,.98));
        border: 1px solid rgba(118, 135, 172, 0.16);
        border-radius: 18px;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    .entity-name {
        color: #f5f7fb;
        font-size: 1.05rem;
        font-weight: 600;
        margin-bottom: 0.3rem;
    }
    .entity-subtext {
        color: #b9c8e8;
        line-height: 1.6;
        margin-bottom: 0.65rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)
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


def _format_inline_value(value):
    if value is None or value == "":
        return "N/A"
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(cleaned) if cleaned else "N/A"
    if isinstance(value, dict):
        cleaned = [f"{key}: {val}" for key, val in value.items() if val not in (None, "", [], {})]
        return ", ".join(cleaned) if cleaned else "N/A"
    return str(value)


def _format_score_display(value, default="3/5"):
    text = _format_inline_value(value)
    if text in ("N/A", "N/A/5", "Pending"):
        return default
    if text.endswith("/5"):
        return text
    return f"{text}/5"


def _render_bullets(items, empty_message):
    cleaned = []
    for item in items or []:
        text = _format_inline_value(item)
        if text != "N/A":
            cleaned.append(text)

    for item in cleaned or [empty_message]:
        st.markdown(f"- {item}")


def _render_raw_notes(raw_text, label="Research Notes"):
    st.markdown(f"#### {label}")
    pretty = raw_text.strip()
    wrapped = "\n".join(textwrap.wrap(pretty, width=110, replace_whitespace=False)) if pretty else "No notes captured."
    st.code(wrapped, language="text")


def _render_section_view(section):
    view = section.get("view", {}) if isinstance(section, dict) else {}
    _render_section_intro(
        view.get("kicker", "Research"),
        view.get("title", "Section Overview"),
        view.get("copy", ""),
    )

    metrics = view.get("metrics", []) or []
    if metrics:
        cols = st.columns(len(metrics))
        for col, metric in zip(cols, metrics):
            col.metric(metric.get("label", "Metric"), _format_inline_value(metric.get("value", "N/A")))

    bullet_groups = view.get("bullet_groups", []) or []
    if bullet_groups:
        cols = st.columns(len(bullet_groups))
        for col, group in zip(cols, bullet_groups):
            with col:
                st.markdown(f"#### {group.get('title', 'Highlights')}")
                _render_bullets(group.get("items", []), "No details captured.")

    entities = view.get("entities", []) or []
    if entities:
        for entity in entities:
            st.markdown(
                f"""
                <div class="entity-card">
                    <div class="entity-name">{entity.get("name", "Entry")}</div>
                    <div class="entity-subtext">{entity.get("subtitle", "No summary captured.")}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            columns = entity.get("columns", []) or []
            bullets = entity.get("bullets", []) or []
            if columns:
                cols = st.columns(len(columns))
                for col, block in zip(cols, columns):
                    with col:
                        st.markdown(f"**{block.get('title', 'Details')}**")
                        _render_bullets(block.get("items", []), "No details captured.")
            elif bullets:
                _render_bullets(bullets, "No details captured.")

    raw_fallback = view.get("raw_fallback") or (section.get("raw") if isinstance(section, dict) else None)
    if not metrics and not bullet_groups and not entities:
        if raw_fallback:
            _render_raw_notes(raw_fallback, label="Structured Output")
        else:
            st.info("No details were generated for this run.")

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
        overall_score = judge.get("overall_score", 3)
        if isinstance(overall_score, float) and overall_score.is_integer():
            overall_score = int(overall_score)
        top_strength = judge.get("top_strength") or "Recovered quality review"
        top_improvement = judge.get("top_improvement") or "Improve evidence consistency"

        metric_1, metric_2, metric_3 = st.columns(3)
        metric_1.metric("Research Score", _format_score_display(overall_score))
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
                                    <div class="score-chip">{_format_score_display(score_value)}</div>
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

            market_data = analysis.get("market", {}) or {}
            founders_data = analysis.get("founders", {}) or {}
            competition_data = analysis.get("competition", {}) or {}

            with market_panel:
                _render_section_view(market_data)

            with founders_panel:
                _render_section_view(founders_data)

            with competition_panel:
                _render_section_view(competition_data)

        st.success("Investment brief generated successfully.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        status_box.empty()
