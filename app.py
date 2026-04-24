import os
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from groq import Groq

from agents import run_research_pipeline, similar_startup_explorer, startup_validator_pro

load_dotenv()

st.set_page_config(
    page_title="Startup Research Desk AI",
    page_icon="🚀",
    layout="wide",
)


def load_css():
    css_path = Path(__file__).with_name("styles.css")
    if css_path.exists():
        st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


load_css()


def get_client():
    api_key = os.getenv("GROQ_API_KEY", "")
    tavily_key = os.getenv("TAVILY_API_KEY", "")
    if not api_key or not tavily_key:
        st.error("Missing backend configuration. Please set `GROQ_API_KEY` and `TAVILY_API_KEY` in your environment.")
        st.stop()
    return Groq(api_key=api_key)


def format_value(value, default="N/A"):
    if value in (None, "", [], {}):
        return default
    if isinstance(value, list):
        cleaned = [str(item).strip() for item in value if str(item).strip()]
        return ", ".join(cleaned) if cleaned else default
    if isinstance(value, dict):
        cleaned = [f"{key}: {val}" for key, val in value.items() if val not in (None, "", [], {})]
        return ", ".join(cleaned) if cleaned else default
    return str(value)


def format_score(value, default=5, max_score=10):
    if isinstance(value, (int, float)):
        score = float(value)
    else:
        text = format_value(value, default="")
        match = None
        if text:
            import re
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)", text)
        score = float(match.group(1)) if match else float(default)
    return max(0.0, min(score, float(max_score)))


def render_page_header(title, subtitle, badge):
    st.markdown(
        f"""
        <section class="hero-shell">
            <div class="hero-badge">{badge}</div>
            <h1 class="hero-title">{title}</h1>
            <p class="hero-subtitle">{subtitle}</p>
        </section>
        """,
        unsafe_allow_html=True,
    )


def render_glance_cards(items):
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        with column:
            st.markdown(
                f"""
                <div class="glass-card">
                    <div class="card-kicker">{item['title']}</div>
                    <div class="card-copy">{item['copy']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_metric_row(items):
    columns = st.columns(len(items))
    for column, item in zip(columns, items):
        with column:
            st.markdown(
                f"""
                <div class="metric-tile">
                    <div class="metric-label">{item['label']}</div>
                    <div class="metric-value">{item['value']}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_text_card(title, body, tone="default"):
    tone_class = f"card-tone-{tone}"
    st.markdown(
        f"""
        <div class="content-card {tone_class}">
            <div class="card-title">{title}</div>
            <div class="card-body">{body}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_bullet_card(title, items, tone="default"):
    bullets = items or ["No details captured."]
    bullet_html = "".join(f"<li>{format_value(item)}</li>" for item in bullets)
    tone_class = f"card-tone-{tone}"
    st.markdown(
        f"""
        <div class="content-card {tone_class}">
            <div class="card-title">{title}</div>
            <ul class="bullet-list">{bullet_html}</ul>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_progress_panel(title, score, max_score, caption):
    numeric_score = format_score(score, default=max_score / 2, max_score=max_score)
    st.markdown(f"#### {title}")
    st.progress(numeric_score / max_score)
    st.markdown(
        f"""
        <div class="score-line">{numeric_score:.1f}/{max_score}</div>
        <div class="micro-copy">{caption}</div>
        """,
        unsafe_allow_html=True,
    )


def render_badge(text, tone="neutral"):
    st.markdown(f"<span class='status-pill status-{tone}'>{text}</span>", unsafe_allow_html=True)


def render_research_details(analysis):
    market_panel, founders_panel, competition_panel = st.tabs(["Market", "Founders", "Competition"])
    sections = {
        "Market": analysis.get("market", {}) or {},
        "Founders": analysis.get("founders", {}) or {},
        "Competition": analysis.get("competition", {}) or {},
    }

    for panel, section in zip([market_panel, founders_panel, competition_panel], sections.values()):
        with panel:
            view = section.get("view", {}) if isinstance(section, dict) else {}
            render_text_card(view.get("title", "Section Overview"), view.get("copy", "Structured analysis from the backend."), tone="subtle")

            metrics = view.get("metrics", []) or []
            meaningful_metrics = [metric for metric in metrics if format_value(metric.get("value")) != "N/A"]
            if meaningful_metrics:
                render_metric_row(
                    [{"label": metric.get("label", "Metric"), "value": format_value(metric.get("value"))} for metric in meaningful_metrics]
                )

            bullet_groups = view.get("bullet_groups", []) or []
            if bullet_groups:
                cols = st.columns(len(bullet_groups))
                for col, group in zip(cols, bullet_groups):
                    with col:
                        render_bullet_card(group.get("title", "Highlights"), group.get("items", []))

            entities = view.get("entities", []) or []
            for entity in entities:
                render_text_card(entity.get("name", "Entry"), entity.get("subtitle", "No summary captured."))
                columns = entity.get("columns", []) or []
                bullets = entity.get("bullets", []) or []
                if columns:
                    cols = st.columns(len(columns))
                    for col, block in zip(cols, columns):
                        with col:
                            render_bullet_card(block.get("title", "Details"), block.get("items", []), tone="subtle")
                elif bullets:
                    render_bullet_card("Details", bullets, tone="subtle")

            if not meaningful_metrics and not bullet_groups and not entities:
                raw = view.get("raw_fallback") or section.get("raw")
                if raw:
                    st.code(raw, language="text")
                else:
                    st.info("No structured analysis available for this section.")


def render_quality_review(judge):
    summary = judge.get("summary")
    if summary:
        render_text_card("Executive Readout", summary, tone="subtle")

    scores = judge.get("scores", {}) or {}
    if scores:
        items = []
        for label, item in scores.items():
            items.append(
                {
                    "label": label.replace("_", " ").title(),
                    "value": f"{format_score(item.get('score'), default=3, max_score=5):.1f}/5",
                }
            )
        render_metric_row(items[:3])
        if len(items) > 3:
            render_metric_row(items[3:])

        for label, item in scores.items():
            reasoning = item.get("reasoning")
            if reasoning:
                st.caption(f"{label.replace('_', ' ').title()}: {reasoning}")
    else:
        st.json(judge)


def render_home_page():
    render_page_header(
        "Startup Research Desk AI",
        "AI-powered startup validation, founder intelligence, and portfolio-grade investment briefing for modern operators.",
        "Portfolio-Level Startup Intelligence",
    )

    render_glance_cards(
        [
            {"title": "Founder Intelligence", "copy": "Team quality, market signals, and investor-style reasoning in one workflow."},
            {"title": "Validation Engine", "copy": "Idea flaws, saturation, monetization risk, and pivot guidance before you build."},
            {"title": "VC Briefing", "copy": "Premium research briefs with quality review and structured internal notes."},
        ]
    )

    st.markdown("### Why This Project Is Unique")
    render_glance_cards(
        [
            {"title": "Flaw Detector", "copy": "Surfaces hidden risks, execution drag, business model gaps, and failure patterns."},
            {"title": "Similarity Mapping", "copy": "Finds copycats, global analogs, saturation levels, and whitespace opportunities."},
            {"title": "Investor Lens", "copy": "Turns raw research into founder readiness and fundability signals recruiters will remember."},
        ]
    )

    st.markdown("### Generate Investment Brief")
    left, right = st.columns([1.2, 0.8])
    with left:
        startup_name = st.text_input("Startup Name", placeholder="e.g. Sourcy AI", key="home_startup_name")
        domain = st.text_input("Domain (optional)", placeholder="e.g. sourcy.ai", key="home_domain")
    with right:
        render_text_card(
            "What This Delivers",
            "A polished investment brief, internal research details, and quality-reviewed analysis that feels like a real VC workflow.",
            tone="subtle",
        )

    if st.button("Generate Investment Brief", type="primary", use_container_width=True, key="home_generate"):
        if not startup_name:
            st.error("Please enter a startup name.")
            st.stop()

        client = get_client()
        with st.spinner("Building premium startup brief..."):
            st.session_state["home_result"] = run_research_pipeline(
                client,
                startup_name=startup_name,
                domain=domain,
            )

    result = st.session_state.get("home_result")
    if result:
        judge = result.get("judge", {}) or {}
        render_metric_row(
            [
                {"label": "Research Score", "value": f"{format_score(judge.get('overall_score'), default=3, max_score=5):.1f}/5"},
                {"label": "Top Strength", "value": format_value(judge.get("top_strength"), "Recovered quality review")},
                {"label": "Key Improvement", "value": format_value(judge.get("top_improvement"), "Improve evidence consistency")},
            ]
        )

        st.markdown("## Investment Brief")
        st.markdown(result["report"])
        st.download_button(
            "Download Report",
            result["report"],
            file_name=f"{result.get('startup_name', 'startup')}_report.md",
            use_container_width=True,
        )

        review_tab, research_tab = st.tabs(["Quality Review", "Internal Research Details"])
        with review_tab:
            render_quality_review(judge)
        with research_tab:
            render_research_details(result.get("analysis", {}) or {})


def render_validator_page():
    render_page_header(
        "Startup Validator Pro",
        "Pressure-test startup ideas like a founder, operator, and investor in one premium dashboard.",
        "Founder Intelligence Dashboard",
    )

    idea = st.text_area(
        "Enter your startup idea",
        placeholder="AI-based grocery delivery for hostel students",
        height=140,
        key="validator_idea",
    )

    if st.button("Analyze Idea", type="primary", use_container_width=True, key="validator_button"):
        if not idea.strip():
            st.error("Please enter your startup idea.")
            st.stop()

        client = get_client()
        with st.spinner("Running idea validation engine..."):
            st.session_state["validator_result"] = startup_validator_pro(client, idea)

    result = st.session_state.get("validator_result")
    if not result:
        return

    investor = result.get("investor_lens", {}) or {}
    verdict = format_value(investor.get("verdict"), "Maybe")
    verdict_tone = "good" if verdict.lower() == "yes" else "warn" if verdict.lower() == "maybe" else "bad"

    render_text_card("Executive Summary", format_value(result.get("executive_summary")), tone="subtle")
    render_badge(f"Investor Lens: {verdict}", tone=verdict_tone)

    render_metric_row(
        [
            {"label": "Competition Level", "value": format_value(result.get("competition_level"), "Moderate")},
            {"label": "Launch Difficulty", "value": format_value(result.get("launch_difficulty_level"), "Medium")},
            {"label": "Revenue Model", "value": format_value(result.get("revenue_model"), "Needs validation")},
        ]
    )

    score_left, score_right = st.columns(2)
    with score_left:
        render_progress_panel(
            "Founder Readiness Score",
            result.get("founder_readiness_score", 5),
            10,
            format_value(result.get("founder_readiness_reasoning"), "Needs stronger validation signals."),
        )
    with score_right:
        render_progress_panel(
            "Investor Interest Score",
            result.get("investor_interest_score", 5),
            10,
            format_value(investor.get("why"), "Investor interest depends on sharper traction and positioning."),
        )

    info_left, info_right = st.columns(2)
    with info_left:
        render_text_card("Problem Solved", format_value(result.get("problem_solved")), tone="subtle")
        render_text_card("Market Opportunity", format_value(result.get("market_opportunity")), tone="subtle")
    with info_right:
        render_bullet_card("Target Audience", result.get("target_audience", []))
        render_text_card("Investor Verdict", format_value(investor.get("why")), tone="subtle")

    grid_left, grid_right = st.columns(2)
    with grid_left:
        render_bullet_card("Hidden Flaws", result.get("hidden_flaws", []), tone="bad")
        render_bullet_card("Risks & Red Flags", result.get("risks_red_flags", []), tone="bad")
    with grid_right:
        render_bullet_card("Suggested Improvements", result.get("suggested_improvements", []), tone="good")
        render_bullet_card("Pivot Ideas", result.get("pivot_ideas", []), tone="warn")

    render_bullet_card("Recommended Next Steps", result.get("recommended_next_steps", []), tone="subtle")


def render_explorer_page():
    render_page_header(
        "Similar Startup Explorer",
        "Map the market around your idea across competitors, global analogs, Indian variants, and whitespace.",
        "Similarity Discovery Engine",
    )

    idea = st.text_area(
        "Describe your startup idea",
        placeholder="AI co-pilot for SMB founders to validate startup ideas before launch",
        height=140,
        key="explorer_idea",
    )

    if st.button("Search Similar Ideas", type="primary", use_container_width=True, key="explorer_button"):
        if not idea.strip():
            st.error("Please describe your startup idea.")
            st.stop()

        client = get_client()
        with st.spinner("Exploring startup similarity landscape..."):
            st.session_state["explorer_result"] = similar_startup_explorer(client, idea)

    result = st.session_state.get("explorer_result")
    if not result:
        return

    render_metric_row(
        [
            {"label": "Saturation Score", "value": f"{format_score(result.get('saturation_score'), default=5, max_score=10):.1f}/10"},
            {"label": "Better Launch Angle", "value": format_value(result.get("better_launch_angle"), "Find a narrower wedge.")},
        ]
    )

    left, right = st.columns(2)
    with left:
        render_bullet_card("Related Competitors", result.get("related_competitors", []))
        render_bullet_card("White Space Opportunities", result.get("white_space_opportunities", []), tone="good")
        render_bullet_card("Untapped Audience Ideas", result.get("untapped_audiences", []), tone="good")
    with right:
        render_bullet_card("Global Versions", result.get("global_versions", []))
        render_bullet_card("Indian Market Versions", result.get("indian_market_versions", []))
        render_bullet_card("What Is Missing In Current Market", result.get("market_gaps", []), tone="warn")

    st.markdown("### Similar Existing Startups")
    startups = result.get("similar_existing_startups", []) or []
    if startups:
        for startup in startups:
            render_text_card(
                format_value(startup.get("name"), "Comparable Startup"),
                f"{format_value(startup.get('description'), 'Comparable play in this space.')}<br><span class='mini-tag'>{format_value(startup.get('region'), 'Global')}</span>",
                tone="subtle",
            )
    else:
        st.info("No similar startups captured for this run.")

    render_bullet_card("Top 5 Niche Variations", result.get("niche_variations", []), tone="subtle")


def render_about_page():
    render_page_header(
        "About The AI Model",
        "A premium AI startup intelligence model designed to impress recruiters, incubators, founders, and analysts.",
        "Model Story",
    )

    render_text_card(
        "Startup Research Desk AI",
        "This AI model combines startup validation, founder intelligence, competitor mapping, and investor-style judgment inside one modern Streamlit product shell.",
        tone="subtle",
    )

    render_glance_cards(
        [
            {"title": "LLM Research Layer", "copy": "LLM-backed research synthesis with structured outputs and safe fallbacks."},
            {"title": "Search Intelligence", "copy": "Tavily-powered web context for markets, competitors, and whitespace discovery."},
            {"title": "Product-Ready UX", "copy": "Premium SaaS styling, multiple interfaces, and dashboard-quality rendering."},
        ]
    )


def render_unique_page():
    render_page_header(
        "Why This Project Is Unique",
        "Most student projects stop at report generation. This one goes further into flaw detection, saturation analysis, founder readiness, and investor-style decisioning.",
        "Differentiation",
    )

    render_glance_cards(
        [
            {"title": "Startup Flaw Detector", "copy": "Exposes hidden risks, weak monetization logic, scalability drag, privacy risks, and adoption friction."},
            {"title": "Similar Startup Discovery Engine", "copy": "Maps existing startups, copycat risks, market saturation, and better launch angles."},
            {"title": "Founder Readiness + Investor Lens", "copy": "Adds decision-grade scoring that feels closer to an incubator or VC workflow than a chatbot."},
        ]
    )

    render_text_card(
        "Why Recruiters Notice It",
        "It demonstrates full-stack product thinking: multi-flow AI UX, structured backend outputs, strong visual design, and domain-specific reasoning for startups and venture workflows.",
        tone="good",
    )


st.sidebar.markdown("<div class='sidebar-brand'>Startup Research Desk AI</div>", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    <div class="sidebar-hero">
        <div class="sidebar-badge">AI Startup Intelligence</div>
        <div class="sidebar-copy">
            Founder intelligence, startup validation, and investor-style analysis in one premium AI model.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)
page = st.sidebar.radio(
    "Navigation",
    ["Home", "Startup Validator Pro", "Similar Startup Explorer", "About AI Model", "Why Unique"],
)

if page == "Home":
    render_home_page()
elif page == "Startup Validator Pro":
    render_validator_page()
elif page == "Similar Startup Explorer":
    render_explorer_page()
elif page == "About AI Model":
    render_about_page()
else:
    render_unique_page()
