import json
import re
import os

from tavily import TavilyClient

from api_service import simple_generate, safe_generate


def get_tavily():
    return TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))


_SEARCH_CACHE = {}


def run_search(query):
    cached = _SEARCH_CACHE.get(query)
    if cached is not None:
        return cached
    try:
        tavily = get_tavily()
        result = tavily.search(query=query, max_results=3)
        snippets = [r["content"] for r in result.get("results", [])]
        output = "\n\n".join(snippets) if snippets else "No results found."
        _SEARCH_CACHE[query] = output
        return output
    except Exception as e:
        return f"Search failed: {e}"


def _merge_nested(defaults, parsed):
    if isinstance(defaults, dict):
        source = parsed if isinstance(parsed, dict) else {}
        merged = {}
        for key, default_value in defaults.items():
            merged[key] = _merge_nested(default_value, source.get(key))
        for key, value in source.items():
            if key not in merged:
                merged[key] = value
        return merged

    if isinstance(defaults, list):
        if isinstance(parsed, list) and parsed:
            return parsed
        return defaults

    if parsed in (None, "", [], {}):
        return defaults
    return parsed


def _generate_structured_analysis(
    client,
    prompt,
    system,
    defaults,
    log_step=None,
    max_completion_tokens=700,
):
    raw_text = simple_generate(
        client,
        prompt,
        system=system,
        log_step=log_step,
        max_completion_tokens=max_completion_tokens,
    )
    parsed, cleaned = _parse_agent_payload(raw_text)
    merged = _merge_nested(defaults, parsed if isinstance(parsed, dict) else {})
    merged["_raw"] = cleaned
    return merged


def _normalize_validator_scores(result):
    hidden_flaws = _ensure_list(result.get("hidden_flaws"))
    red_flags = _ensure_list(result.get("risks_red_flags"))
    competition_level = _stringify(result.get("competition_level"), "Moderate").lower()
    investor_verdict = _stringify((result.get("investor_lens") or {}).get("verdict"), "Maybe").lower()

    readiness = result.get("founder_readiness_score")
    investor_score = result.get("investor_interest_score")

    def _looks_default(value):
        if isinstance(value, (int, float)):
            return float(value) == 5.0
        if isinstance(value, str):
            match = re.search(r"([0-9]+(?:\.[0-9]+)?)", value)
            return bool(match) and float(match.group(1)) == 5.0
        return True

    penalty = min(len(hidden_flaws), 4) * 0.8 + min(len(red_flags), 4) * 0.7
    if "high" in competition_level or "saturated" in competition_level:
        penalty += 1.0
    elif "low" in competition_level:
        penalty -= 0.5

    verdict_bonus = 1.2 if investor_verdict == "yes" else 0.0
    verdict_penalty = 1.0 if investor_verdict == "no" else 0.0

    computed_readiness = max(2.5, min(9.5, round(7.5 - penalty + verdict_bonus - verdict_penalty, 1)))
    computed_investor = max(2.5, min(9.5, round(7.0 - penalty * 0.8 + verdict_bonus - verdict_penalty, 1)))

    if _looks_default(readiness):
        result["founder_readiness_score"] = computed_readiness
    if _looks_default(investor_score):
        result["investor_interest_score"] = computed_investor

    return result


def market_researcher(client, startup_name, domain, log_step=None):
    if log_step:
        log_step("Market Researcher: gathering market context...")

    searches = [
        f"{startup_name} market size industry trends",
        f"{startup_name} funding news market growth",
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
        max_completion_tokens=420,
    )


def founder_analyst(client, startup_name, log_step=None):
    if log_step:
        log_step("Founder Analyst: researching team...")

    searches = [
        f"{startup_name} founder CEO background",
        f"{startup_name} founding team education previous experience",
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
        max_completion_tokens=380,
    )


def competitive_intel(client, startup_name, market_summary, log_step=None):
    if log_step:
        log_step("Competitive Intel: mapping competitors...")

    searches = [
        f"{startup_name} competitors alternatives",
        f"{startup_name} vs competitors comparison",
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
        max_completion_tokens=420,
    )


def startup_validator_pro(client, startup_idea, log_step=None):
    if log_step:
        log_step("Validator Pro: scanning idea quality and founder readiness...")

    searches = [
        f"{startup_idea} startup competitors market",
        f"{startup_idea} business model risks",
    ]

    context = ""
    for query in searches:
        if log_step:
            log_step(f"Validator Pro: searching -> {query}")
        context += run_search(query) + "\n\n"

    defaults = {
        "executive_summary": "This idea needs sharper validation around execution, monetization, and defensibility before launch.",
        "problem_solved": "The problem statement needs clearer evidence from a specific customer segment.",
        "target_audience": ["Early adopters who feel the problem acutely."],
        "revenue_model": "Revenue model requires validation with real users.",
        "hidden_flaws": ["Potential mismatch between product promise and willingness to pay."],
        "risks_red_flags": ["Execution complexity may be higher than expected."],
        "competition_level": "Moderate",
        "market_opportunity": "There may be a real opportunity if the wedge is sharper and the ICP is narrower.",
        "founder_readiness_score": 5,
        "founder_readiness_reasoning": "Needs stronger clarity on market wedge, go-to-market, and repeatability.",
        "investor_interest_score": 5,
        "investor_lens": {
            "verdict": "Maybe",
            "why": "Interesting space, but evidence of traction and defensibility is not yet strong enough.",
        },
        "suggested_improvements": ["Sharpen ICP, wedge, and pricing before launch."],
        "pivot_ideas": ["Choose a narrower niche with faster feedback loops and clearer ROI."],
        "launch_difficulty_level": "Medium",
        "recommended_next_steps": ["Interview 20 users", "Validate pricing", "Prototype a narrow MVP"],
    }

    prompt = (
        f"Analyze this startup idea like a top incubator partner and VC operator.\n\n"
        f"IDEA:\n{startup_idea}\n\n"
        f"SEARCH CONTEXT:\n{context}\n\n"
        "Return strict JSON with keys: executive_summary, problem_solved, target_audience (list), "
        "revenue_model, hidden_flaws (list), risks_red_flags (list), competition_level, market_opportunity, "
        "founder_readiness_score (1-10), founder_readiness_reasoning, investor_interest_score (1-10), "
        "investor_lens (object with verdict and why), suggested_improvements (list), pivot_ideas (list), "
        "launch_difficulty_level, recommended_next_steps (list). "
        "Be practical, investor-minded, and critical when needed."
    )

    result = _generate_structured_analysis(
        client,
        prompt,
        system="You are a startup studio partner evaluating whether a startup idea is viable, fundable, and execution-ready.",
        defaults=defaults,
        log_step=log_step,
        max_completion_tokens=700,
    )
    return _normalize_validator_scores(result)


def similar_startup_explorer(client, startup_idea, log_step=None):
    if log_step:
        log_step("Similar Explorer: mapping adjacent ideas and whitespace...")

    searches = [
        f"{startup_idea} similar startups competitors",
        f"{startup_idea} startup india alternatives",
    ]

    context = ""
    for query in searches:
        if log_step:
            log_step(f"Similar Explorer: searching -> {query}")
        context += run_search(query) + "\n\n"

    defaults = {
        "similar_existing_startups": [
            {"name": "Reference competitor", "description": "Comparable solution in an adjacent market.", "region": "Global"}
        ],
        "related_competitors": ["Established incumbents and fast-moving niche competitors."],
        "global_versions": ["International variants exist in adjacent categories."],
        "indian_market_versions": ["Indian market requires localization and distribution insight."],
        "saturation_score": 5,
        "white_space_opportunities": ["Find a narrower user segment with underserved workflow pain."],
        "market_gaps": ["Current players may not serve a tightly defined customer wedge."],
        "better_launch_angle": "Start with a niche where ROI is easy to measure.",
        "untapped_audiences": ["Users with strong urgency but limited existing tooling."],
        "niche_variations": ["Verticalized version for a high-friction industry."],
    }

    prompt = (
        f"Analyze this startup idea like a competitive intelligence product team.\n\n"
        f"IDEA:\n{startup_idea}\n\n"
        f"SEARCH CONTEXT:\n{context}\n\n"
        "Return strict JSON with keys: similar_existing_startups (list of objects with name, description, region), "
        "related_competitors (list), global_versions (list), indian_market_versions (list), saturation_score (1-10), "
        "white_space_opportunities (list), market_gaps (list), better_launch_angle, untapped_audiences (list), "
        "niche_variations (list)."
    )

    return _generate_structured_analysis(
        client,
        prompt,
        system="You are a startup market mapper who identifies competitors, saturation, and whitespace opportunities.",
        defaults=defaults,
        log_step=log_step,
        max_completion_tokens=700,
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
        max_completion_tokens=700,
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

JUDGE_SCORE_KEYS = [
    "market_accuracy",
    "founder_credibility",
    "competitive_depth",
    "report_clarity",
    "actionability",
]


def _extract_json_object(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise json.JSONDecodeError("No JSON object found", text, 0)
    return text[start:end + 1]


def _extract_balanced_json(text):
    start = text.find("{")
    if start == -1:
        raise json.JSONDecodeError("No JSON object found", text, 0)

    depth = 0
    in_string = False
    escaped = False

    for index, char in enumerate(text[start:], start=start):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start:index + 1]

    return text[start:]


def _repair_json_blob(blob):
    open_braces = blob.count("{")
    close_braces = blob.count("}")
    if close_braces < open_braces:
        blob = blob + ("}" * (open_braces - close_braces))
    return blob


def _parse_agent_payload(text):
    cleaned = re.sub(r"```json|```", "", text or "").strip()
    try:
        blob = _extract_balanced_json(cleaned)
    except json.JSONDecodeError:
        return None, cleaned

    for candidate in (blob, _repair_json_blob(blob)):
        try:
            return json.loads(candidate), cleaned
        except json.JSONDecodeError:
            continue

    return None, cleaned


def _normalize_analysis_section(name, raw_text):
    parsed, cleaned = _parse_agent_payload(raw_text)
    view = _build_section_view(name, parsed, cleaned)
    return {
        "name": name,
        "parsed": parsed,
        "raw": cleaned,
        "view": view,
    }


def _ensure_list(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        return [part.strip() for part in value.split(",") if part.strip()]
    return [str(value).strip()]


def _stringify(value, default="N/A"):
    if value in (None, "", [], {}):
        return default
    if isinstance(value, list):
        items = _ensure_list(value)
        return ", ".join(items) if items else default
    if isinstance(value, dict):
        items = [f"{key}: {val}" for key, val in value.items() if val not in (None, "", [], {})]
        return ", ".join(items) if items else default
    return str(value)


def _score_text(value, default=3.0):
    if isinstance(value, (int, float)):
        score = max(1.0, min(float(value), 5.0))
        return f"{int(score) if score.is_integer() else round(score, 1)}/5"

    if isinstance(value, str):
        match = re.search(r"([1-5](?:\.\d+)?)", value)
        if match:
            score = max(1.0, min(float(match.group(1)), 5.0))
            return f"{int(score) if score.is_integer() else round(score, 1)}/5"

    fallback = max(1.0, min(float(default), 5.0))
    return f"{int(fallback) if fallback.is_integer() else round(fallback, 1)}/5"


def _meaningful_text(value, default=None):
    text = _stringify(value, default="N/A")
    if text == "N/A":
        return default
    return text


def _filtered_metrics(metrics):
    cleaned = []
    for metric in metrics:
        value = metric.get("value")
        if value in (None, "", "N/A", "N/A/5"):
            continue
        cleaned.append(metric)
    return cleaned


def _competition_score_fallback(parsed):
    competitors = (parsed or {}).get("competitors", []) or []
    competitor_count = len(competitors)
    if competitor_count >= 5:
        return 4.5
    if competitor_count >= 3:
        return 4
    if competitor_count >= 1:
        return 3
    return 3


def _competition_text_fallback(parsed, raw_text, field):
    direct_value = _stringify((parsed or {}).get(field))
    if direct_value != "N/A":
        return direct_value

    competitors = (parsed or {}).get("competitors", []) or []
    if field == "differentiation":
        if competitors:
            first = competitors[0]
            return _stringify(first.get("positioning"), "Differentiation not clearly established.")
        return "Differentiation not clearly established."

    if field == "moat":
        strengths = []
        for competitor in competitors:
            strengths.extend(_ensure_list(competitor.get("strengths")))
        if strengths:
            return strengths[0]

        raw_lower = (raw_text or "").lower()
        if "moat" in raw_lower:
            return "Moat mentioned in analyst notes."
        return "Moat not explicitly identified."

    return "N/A"


def _market_view(parsed, raw_text):
    metrics = _filtered_metrics([
        {"label": "Market Size", "value": _meaningful_text((parsed or {}).get("market_size"))},
        {"label": "Growth Rate", "value": _meaningful_text((parsed or {}).get("growth_rate"))},
        {"label": "Confidence", "value": _score_text((parsed or {}).get("confidence_score"), default=3)},
    ])
    return {
        "kicker": "Market Signal",
        "title": "Market Opportunity Snapshot",
        "copy": "A synthesized view of category size, momentum, trendlines, and recent movement around the company.",
        "metrics": metrics,
        "bullet_groups": [
            {"title": "Key Trends", "items": _ensure_list((parsed or {}).get("key_trends")) or ["No trends captured."]},
            {"title": "Recent News", "items": _ensure_list((parsed or {}).get("recent_news")) or ["No recent news captured."]},
        ],
        "raw_fallback": raw_text,
    }


def _founders_view(parsed, raw_text):
    founders = []
    for founder in (parsed or {}).get("founders", []) or []:
        founders.append({
            "name": _stringify(founder.get("name"), "Unknown Founder"),
            "subtitle": _stringify(founder.get("role"), "Role unavailable"),
            "bullets": _ensure_list(founder.get("background")) or ["No background details captured."],
        })

    metrics = _filtered_metrics([
        {"label": "Team Score", "value": _score_text((parsed or {}).get("team_score"), default=3)},
        {"label": "Domain Expertise", "value": _meaningful_text((parsed or {}).get("domain_expertise"))},
        {"label": "Notable Exits", "value": _meaningful_text((parsed or {}).get("notable_exits"), default="None noted")},
    ])

    return {
        "kicker": "Team Read",
        "title": "Founder Assessment",
        "copy": "A structured look at founder background, domain fit, exits, and execution risks surfaced during research.",
        "metrics": metrics,
        "entities": founders,
        "bullet_groups": [
            {"title": "Red Flags", "items": _ensure_list((parsed or {}).get("red_flags")) or ["No material red flags surfaced."]},
        ],
        "raw_fallback": raw_text,
    }


def _competition_view(parsed, raw_text):
    competitors = []
    for competitor in (parsed or {}).get("competitors", []) or []:
        competitors.append({
            "name": _stringify(competitor.get("name"), "Competitor"),
            "subtitle": _stringify(competitor.get("positioning"), "No positioning captured."),
            "columns": [
                {"title": "Strengths", "items": _ensure_list(competitor.get("strengths")) or ["No strengths captured."]},
                {"title": "Weaknesses", "items": _ensure_list(competitor.get("weaknesses")) or ["No weaknesses captured."]},
            ],
        })

    metrics = _filtered_metrics([
        {
            "label": "Competitive Score",
            "value": _score_text(
                (parsed or {}).get("competitive_score"),
                default=_competition_score_fallback(parsed),
            ),
        },
        {"label": "Differentiation", "value": _meaningful_text(_competition_text_fallback(parsed, raw_text, "differentiation"))},
        {"label": "Moat", "value": _meaningful_text(_competition_text_fallback(parsed, raw_text, "moat"))},
    ])

    return {
        "kicker": "Competitive Edge",
        "title": "Landscape and Moat Review",
        "copy": "An at-a-glance view of competitors, differentiation, strengths, weaknesses, and defensibility.",
        "metrics": metrics,
        "entities": competitors,
        "raw_fallback": raw_text,
    }


def _build_section_view(name, parsed, raw_text):
    if name == "market":
        return _market_view(parsed, raw_text)
    if name == "founders":
        return _founders_view(parsed, raw_text)
    if name == "competition":
        return _competition_view(parsed, raw_text)
    return {"kicker": name.title(), "title": name.title(), "copy": "", "raw_fallback": raw_text}


def _normalize_judge_result(result):
    scores = result.get("scores", {})
    values = []
    for item in scores.values():
        if isinstance(item, dict):
            score = item.get("score")
            if isinstance(score, (int, float)):
                values.append(float(score))

    computed_average = round(sum(values) / len(values), 1) if values else None
    overall_score = result.get("overall_score")

    if computed_average is None:
        return result

    if not isinstance(overall_score, (int, float)) or overall_score > 5:
        result["overall_score"] = computed_average
        return result

    result["overall_score"] = max(1.0, min(float(overall_score), 5.0))
    if not result.get("top_strength"):
        result["top_strength"] = "Competitive depth"
    if not result.get("top_improvement"):
        result["top_improvement"] = "Evidence quality"
    return result


def _extract_first_number(text, minimum=1, maximum=5):
    if not isinstance(text, str):
        return None
    match = re.search(r"\b([1-5])(?:\.0+)?\b", text)
    if not match:
        return None
    value = float(match.group(1))
    if minimum <= value <= maximum:
        return value
    return None


def _build_fallback_judge(text):
    fallback_scores = {}
    lowered = text.lower()

    for key in JUDGE_SCORE_KEYS:
        label = key.replace("_", " ")
        score = None

        json_match = re.search(rf'"{key}"\s*:\s*\{{[^{{}}]*"score"\s*:\s*([1-5])', text, flags=re.IGNORECASE)
        if json_match:
            score = float(json_match.group(1))
        else:
            line_match = re.search(rf"{label}\s*[:\-]?\s*([1-5])", lowered, flags=re.IGNORECASE)
            if line_match:
                score = float(line_match.group(1))

        fallback_scores[key] = {
            "score": score if score is not None else 3.0,
            "reasoning": "Recovered from partial judge output." if score is not None else "Fallback score used because structured judge output was incomplete.",
        }

    overall_score = round(
        sum(item["score"] for item in fallback_scores.values()) / len(fallback_scores),
        1,
    )

    return {
        "scores": fallback_scores,
        "overall_score": overall_score,
        "summary": "Automated quality review was partially recovered from an incomplete judge response.",
        "top_strength": "Recovered quality review",
        "top_improvement": "Improve evidence consistency",
        "raw": text,
    }


def _safe_parse_judge_text(text):
    cleaned = re.sub(r"```json|```", "", text or "").strip()

    candidates = []
    try:
        candidates.append(_extract_json_object(cleaned))
    except json.JSONDecodeError:
        pass

    try:
        candidates.append(_extract_balanced_json(cleaned))
    except json.JSONDecodeError:
        pass

    seen = set()
    for candidate in candidates:
        if not candidate:
            continue
        for option in (candidate, _repair_json_blob(candidate)):
            if option in seen:
                continue
            seen.add(option)
            try:
                parsed = json.loads(option)
                return _normalize_judge_result(parsed)
            except json.JSONDecodeError:
                continue

    return _build_fallback_judge(cleaned)


def judge_agent(client, report, log_step=None):
    if log_step:
        log_step("Judge: evaluating report against rubric...")

    prompt = JUDGE_PROMPT.format(report=report)
    response = safe_generate(
        client,
        messages=[{"role": "user", "content": prompt}],
        system_instruction=JUDGE_SYSTEM,
        log_step=log_step,
        max_completion_tokens=260,
    )

    text = (response.choices[0].message.content or "").strip()
    result = _safe_parse_judge_text(text)
    if log_step:
        log_step(f"Judge: overall score = {result.get('overall_score', '?')}/5")
    return result


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

    analysis = {
        "market": _normalize_analysis_section("market", market_data),
        "founders": _normalize_analysis_section("founders", founder_data),
        "competition": _normalize_analysis_section("competition", competitor_data),
    }

    return {
        "startup_name": startup_name,
        "domain": domain,
        "report": report,
        "judge": judge,
        "analysis": analysis,
    }
