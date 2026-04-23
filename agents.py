import json
import re
import os

from tavily import TavilyClient

from api_service import simple_generate, safe_generate


def get_tavily():
    return TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


def run_search(query):
    try:
        tavily = get_tavily()
        result = tavily.search(query=query, max_results=5)
        snippets = [r["content"] for r in result.get("results", [])]
        return "\n\n".join(snippets) if snippets else "No results found."
    except Exception as e:
        return f"Search failed: {e}"


def market_researcher(client, startup_name, domain, log_step=None):
    if log_step:
        log_step("Market Researcher: gathering market context...")

    searches = [
        f"{startup_name} market size industry trends",
        f"{startup_name} funding news market growth",
        f"{startup_name} industry outlook competitors",
    ]

    if domain:
        searches.append(f"{startup_name} {domain} market positioning")

    context = ""
    for query in searches:
        if log_step:
            log_step(f"Market Researcher: searching -> {query}")
        context += run_search(query) + "\n\n"

    prompt = (
        f"Research the market for startup: {startup_name} ({domain}).\n\n"
        f"Search context:\n{context}\n\n"
        "Return a JSON summary with keys: market_size, growth_rate, "
        "key_trends (list), recent_news (list), confidence_score (1-5)."
    )

    return simple_generate(
        client,
        prompt,
        system="You are a market research analyst focused on startup market opportunity.",
        log_step=log_step,
    )


def founder_analyst(client, startup_name, log_step=None):
    if log_step:
        log_step("Founder Analyst: researching team...")

    searches = [
        f"{startup_name} founder CEO background",
        f"{startup_name} founding team education previous experience",
        f"{startup_name} founders previous startup exits",
    ]

    context = ""
    for q in searches:
        if log_step:
            log_step(f"Founder Analyst: searching -> {q}")
        context += run_search(q) + "\n\n"

    prompt = (
        f"Based on this research, analyse the founders of {startup_name}.\n\n"
        f"{context}\n\n"
        "Return a JSON with keys: founders (list of name, role, background), "
        "notable_exits, domain_expertise, team_score (1-5), red_flags (list)."
    )

    return simple_generate(
        client,
        prompt,
        system="You are a VC analyst evaluating founding teams.",
        log_step=log_step,
    )


def competitive_intel(client, startup_name, market_summary, log_step=None):
    if log_step:
        log_step("Competitive Intel: mapping competitors...")

    searches = [
        f"{startup_name} competitors alternatives",
        f"{startup_name} vs competitors comparison",
        f"best {startup_name} alternatives 2024",
    ]

    context = ""
    for q in searches:
        if log_step:
            log_step(f"Competitive Intel: searching -> {q}")
        context += run_search(q) + "\n\n"

    prompt = (
        f"Map the competitive landscape for {startup_name}.\n\n"
        f"Market context: {market_summary}\n\nSearch results:\n{context}\n\n"
        "Return JSON with keys: competitors (list of name, positioning, strengths, weaknesses), "
        "differentiation, moat, competitive_score (1-5)."
    )

    return simple_generate(
        client,
        prompt,
        system="You are a competitive intelligence analyst.",
        log_step=log_step,
    )


def report_writer(client, startup_name, market_data, founder_data, competitor_data, log_step=None):
    if log_step:
        log_step("Report Writer: composing VC brief...")

    prompt = (
        f"Write a VC-style investment brief for {startup_name}.\n\n"
        f"## Market Research\n{market_data}\n\n"
        f"## Founder Analysis\n{founder_data}\n\n"
        f"## Competitive Intel\n{competitor_data}\n\n"
        "Structure the report with sections: Executive Summary, Market Opportunity, "
        "Team Assessment, Competitive Landscape, Investment Thesis (Invest / Pass / Watch + rationale). "
        "Use professional VC language. Output clean Markdown."
    )

    return simple_generate(
        client,
        prompt,
        system="You are a senior VC analyst writing an investment committee brief.",
        log_step=log_step,
    )


JUDGE_SYSTEM = """You are a senior investment analyst evaluating AI-generated startup research briefs.
Score each criterion 1-5. Be strict. A 3 = meets expectations. A 5 = exceptional.
Return ONLY valid JSON. No markdown, no preamble."""

JUDGE_PROMPT = """Evaluate this startup research brief:

=== REPORT ===
{report}

=== RUBRIC ===
1. MARKET_ACCURACY (1-5): Is market size/growth specific and sourced?
   1=vague/made-up, 3=general TAM cited, 5=specific TAM/SAM with recent data
2. FOUNDER_CREDIBILITY (1-5): Are founders identified with background details?
   1=no founder info, 3=names and roles, 5=past exits, domain expertise verified
3. COMPETITIVE_DEPTH (1-5): How thorough is the competitor analysis?
   1=1 competitor, 3=3 named with positioning, 5=4-5 with full matrix
4. REPORT_CLARITY (1-5): Is the report executive-ready?
   1=disjointed, 3=readable but informal, 5=structured, investment-committee-ready
5. ACTIONABILITY (1-5): Is the investment recommendation clear?
   1=no recommendation, 3=weak signals noted, 5=clear Invest/Pass/Watch with rationale

=== REQUIRED OUTPUT FORMAT ===
{{
  "scores": {{
    "market_accuracy": {{"score": N, "reasoning": "..."}} ,
    "founder_credibility": {{"score": N, "reasoning": "..."}} ,
    "competitive_depth": {{"score": N, "reasoning": "..."}} ,
    "report_clarity": {{"score": N, "reasoning": "..."}} ,
    "actionability": {{"score": N, "reasoning": "..."}}
  }},
  "overall_score": N,
  "summary": "One paragraph assessment",
  "top_strength": "Best aspect",
  "top_improvement": "Most important improvement"
}}"""


def judge_agent(client, report, log_step=None):
    if log_step:
        log_step("Judge: evaluating report against rubric...")

    prompt = JUDGE_PROMPT.format(report=report)
    response = safe_generate(
        client,
        messages=[{"role": "user", "content": prompt}],
        system_instruction=JUDGE_SYSTEM,
        log_step=log_step,
        response_format={"type": "json_object"},
    )

    text = (response.choices[0].message.content or "").strip()
    text = re.sub(r"```json|```", "", text).strip()

    try:
        result = json.loads(text)
        if log_step:
            log_step(f"Judge: overall score = {result.get('overall_score', '?')}/5")
        return result
    except json.JSONDecodeError:
        return {"error": "Judge failed to return valid JSON", "raw": text}


def run_research_pipeline(client, startup_name, domain="", log_step=None):
    if log_step:
        log_step("Backend pipeline: running multi-agent analysis...")

    market_data = market_researcher(
        client,
        startup_name,
        domain,
        log_step=log_step,
    )
    founder_data = founder_analyst(
        client,
        startup_name,
        log_step=log_step,
    )
    competitor_data = competitive_intel(
        client,
        startup_name,
        market_data,
        log_step=log_step,
    )
    report = report_writer(
        client,
        startup_name,
        market_data,
        founder_data,
        competitor_data,
        log_step=log_step,
    )
    judge = judge_agent(
        client,
        report,
        log_step=log_step,
    )

    return {
        "startup_name": startup_name,
        "domain": domain,
        "report": report,
        "judge": judge,
        "analysis": {
            "market": market_data,
            "founders": founder_data,
            "competition": competitor_data,
        },
    }
