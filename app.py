import os

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

        with st.expander("Quality Review", expanded=False):
            summary = judge.get("summary")
            if summary:
                st.write(summary)
            st.json(judge)

        with st.expander("Internal Research Details", expanded=False):
            st.json(result["analysis"])

        st.success("Investment brief generated successfully.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
    finally:
        status_box.empty()
