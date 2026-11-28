import os
import re
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

import httpx
from langchain_core.tools import tool

from app.agent.tools.scraper import (
    clean_html_text as _clean_html_text,
    scrape_url,
    scrape_urls,
)
from app.config import SCRAPE_MAX_CHARS, SCRAPE_MAX_PAGES


def _search_duckduckgo_pkg(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Attempt search using duckduckgo_search package."""
    from duckduckgo_search import DDGS

    with DDGS() as ddgs:
        # First try dedicated news tab
        results: List[Dict[str, Any]] = []
        try:
            results = list(ddgs.news(query, max_results=max_results))
        except Exception:
            results = []

        # Fallback to general text search if news returned nothing
        if not results:
            results = list(ddgs.text(query, max_results=max_results))

        parsed: List[Dict[str, str]] = []
        for r in results:
            title = r.get("title", "Untitled")
            url = r.get("url") or r.get("href", "")
            snippet = r.get("body") or r.get("snippet", "")
            date = r.get("date", "Recent")
            source = r.get("source", "Web")
            if url:
                parsed.append({
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "date": date,
                    "source": source,
                })
        return parsed


def _search_httpx_fallback(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Fallback search using httpx to scrape DuckDuckGo HTML directly without extra dependencies."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept-Language": "en-US,en;q=0.9",
    }
    url = "https://html.duckduckgo.com/html/"
    parsed: List[Dict[str, str]] = []

    with httpx.Client(timeout=10.0, follow_redirects=True, headers=headers) as client:
        resp = client.post(url, data={"q": query})
        if resp.status_code != 200:
            return parsed

        content = resp.text
        # Extract results matching result__body
        # Regex to capture link, title, and snippet in DDG HTML
        pattern = re.compile(
            r'<a[^>]*class="[^"]*result__url[^"]*"[^>]*href="(?P<url>[^"]+)"[^>]*>.*?'
            r'<a[^>]*class="[^"]*result__title[^"]*"[^>]*>(?P<title>.*?)</a>.*?'
            r'<a[^>]*class="[^"]*result__snippet[^"]*"[^>]*>(?P<snippet>.*?)</a>',
            re.DOTALL | re.IGNORECASE,
        )

        matches = list(pattern.finditer(content))
        if not matches:
            # Broader pattern if specific class nesting changes
            link_pattern = re.compile(
                r'<a\s+class="result__snippet"[^>]*href="(?P<url>[^"]+)".*?>(?P<snippet>.*?)</a>',
                re.DOTALL | re.IGNORECASE,
            )
            title_pattern = re.compile(
                r'<h2\s+class="result__title"[^>]*>\s*<a[^>]*href="(?P<url>[^"]+)"[^>]*>(?P<title>.*?)</a>',
                re.DOTALL | re.IGNORECASE,
            )
            titles = title_pattern.findall(content)
            snippets = link_pattern.findall(content)
            for idx, (t_url, t_text) in enumerate(titles[:max_results]):
                actual_url = t_url
                if "uddg=" in actual_url:
                    actual_url = unquote(actual_url.split("uddg=")[-1].split("&")[0])
                snip_text = snippets[idx][1] if idx < len(snippets) else ""
                parsed.append({
                    "title": _clean_html_text(t_text),
                    "url": actual_url,
                    "snippet": _clean_html_text(snip_text),
                    "date": "Recent",
                    "source": "Web Search",
                })
            return parsed

        for m in matches[:max_results]:
            raw_url = m.group("url")
            # DuckDuckGo HTML often redirects through /l/?uddg=<encoded_url>
            if "uddg=" in raw_url:
                actual_url = unquote(raw_url.split("uddg=")[-1].split("&")[0])
            else:
                actual_url = raw_url

            parsed.append({
                "title": _clean_html_text(m.group("title")),
                "url": actual_url,
                "snippet": _clean_html_text(m.group("snippet")),
                "date": "Recent",
                "source": "Web Search",
            })

    return parsed


def _search_tavily(query: str, max_results: int = 5) -> List[Dict[str, str]]:
    """Attempt search using Tavily API if key is present."""
    from tavily import TavilyClient

    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError("TAVILY_API_KEY not set")

    client = TavilyClient(api_key=api_key)
    res = client.search(query=query, topic="news", max_results=max_results)
    parsed: List[Dict[str, str]] = []
    for r in res.get("results", []):
        parsed.append({
            "title": r.get("title", "Untitled"),
            "url": r.get("url", ""),
            "snippet": r.get("content", ""),
            "date": r.get("published_date", "Recent"),
            "source": "Tavily News",
        })
    return parsed


def _run_web_search(query: str, max_results: int = 5) -> Tuple[List[Dict[str, str]], List[str]]:
    """Run the search waterfall (Tavily -> DDGS -> direct httpx) and return (articles, error_notes)."""
    articles: List[Dict[str, str]] = []
    error_notes: List[str] = []

    if os.getenv("TAVILY_API_KEY"):
        try:
            articles = _search_tavily(query, max_results=max_results)
        except Exception as e:
            error_notes.append(f"Tavily search notice: {e}")

    if not articles:
        try:
            articles = _search_duckduckgo_pkg(query, max_results=max_results)
        except Exception as e:
            error_notes.append(f"DDGS search notice: {e}")

    if not articles:
        try:
            articles = _search_httpx_fallback(query, max_results=max_results)
        except Exception as e:
            error_notes.append(f"HTTP search fallback notice: {e}")

    return articles, error_notes


@tool
def search_latest_news(query: str) -> str:
    """Search for the latest news, breaking industry updates, recent company announcements,
    salary trends, and real-time information outside the knowledge base and LLM training cutoff.
    Returns findings with direct clickable Markdown resource links.
    """
    if not query or not query.strip():
        return "Search query cannot be empty."

    from app.agent.guardrails import is_career_web_query

    if not is_career_web_query(query):
        return (
            "Tool Guardrail Refusal: 'search_latest_news' is strictly reserved for career topics, "
            "job market developments, hiring/layoff news, salary benchmarks, company career intelligence, "
            "and tech industry workforce updates. Out-of-scope searches (sports, entertainment, recipes, "
            "general trivia, or unrelated coding) are blocked."
        )

    articles, error_notes = _run_web_search(query.strip(), max_results=5)

    if not articles:
        details = f" (Details: {'; '.join(error_notes)})" if error_notes else ""
        return f"No recent news or web results found for '{query}'{details}."

    formatted_entries = []
    resource_links = []

    for idx, item in enumerate(articles, start=1):
        title = item["title"] or f"Resource {idx}"
        url = item["url"]
        snippet = item["snippet"] or "No detailed description available."
        date = item["date"]
        source = item["source"]

        formatted_entries.append(
            f"{idx}. **{title}**\n"
            f"   - **Source/Date**: {source} ({date})\n"
            f"   - **Summary**: {snippet}\n"
            f"   - **Resource Link**: [{title}]({url})"
        )
        resource_links.append(f"- [{title}]({url}) — {source} ({date})")

    results_text = "\n\n".join(formatted_entries)
    links_section = "\n".join(resource_links)

    return (
        f"### Latest Web Search Results for '{query}':\n\n"
        f"{results_text}\n\n"
        f"### Verified Resource Links:\n"
        f"{links_section}"
    )


@tool
def browse_webpage(url: str, focus_query: Optional[str] = None) -> str:
    """Visit and scrape the clean main content of a specific webpage URL (e.g., job postings,
    company press releases, news articles). Optionally pass focus_query to prioritize relevant sections.
    Returns clean Markdown text with verified resource links.
    """
    result = scrape_url(url, focus_query=focus_query, max_chars=SCRAPE_MAX_CHARS)
    if not result.ok:
        return result.error or f"Failed to retrieve webpage at {url}."

    title_line = f"**Title**: {result.title}\n" if result.title else ""
    more_notice = "\n\n*(Content truncated for length)*" if result.truncated else ""

    return (
        f"### Web Page Content\n"
        f"{title_line}"
        f"**Resource Link**: [{result.url}]({result.url})\n"
        f"**Extraction Method**: `{result.method}`\n\n"
        f"{result.content}{more_notice}\n\n"
        f"---\n"
        f"**Original Source Link**: [{result.url}]({result.url})"
    )


@tool
def scrape_web_content(query: str, max_pages: int = SCRAPE_MAX_PAGES) -> str:
    """Search the web for the user's career/job query and scrape the full clean content of the top
    matching pages (using local HTTP extraction with Tavily Extract and Jina Reader fallbacks).
    Use this when search snippets are not enough and detailed page content is needed.
    """
    if not query or not query.strip():
        return "Search query cannot be empty."

    from app.agent.guardrails import is_career_web_query

    if not is_career_web_query(query):
        return (
            "Tool Guardrail Refusal: 'scrape_web_content' is strictly reserved for career topics, "
            "job descriptions, interview prep, salary benchmarks, company engineering/hiring intelligence, "
            "and tech industry workforce updates."
        )

    capped_pages = max(1, min(int(max_pages or SCRAPE_MAX_PAGES), 5))
    articles, error_notes = _run_web_search(query.strip(), max_results=capped_pages)
    if not articles:
        details = f" (Details: {'; '.join(error_notes)})" if error_notes else ""
        return f"No web pages found to scrape for '{query}'{details}."

    urls = [a["url"] for a in articles if a.get("url")]
    if not urls:
        return f"Search returned no valid URLs to scrape for '{query}'."

    per_page_budget = max(1800, SCRAPE_MAX_CHARS // max(len(urls), 1))
    scraped = scrape_urls(urls, focus_query=query.strip(), max_chars_per_page=per_page_budget)
    scraped_by_url = {r.url: r for r in scraped}

    sections: List[str] = []
    resource_links: List[str] = []

    for idx, item in enumerate(articles, start=1):
        u = item["url"]
        search_title = item.get("title") or f"Resource {idx}"
        source = item.get("source") or "Web"
        date = item.get("date") or "Recent"
        snippet = item.get("snippet") or ""

        res = scraped_by_url.get(u)
        if res and res.ok:
            page_title = res.title or search_title
            trunc_note = "\n*(Content truncated for length)*" if res.truncated else ""
            body = f"{res.content}{trunc_note}"
            method = res.method
        else:
            page_title = search_title
            fallback_reason = res.error if res and res.error else "Scrape unavailable"
            body = f"{snippet}\n*(Scraped fallback to search snippet: {fallback_reason})*"
            method = "snippet_fallback"

        sections.append(
            f"#### {idx}. {page_title}\n"
            f"- **Source**: [{page_title}]({u}) — {source} ({date}) | **Method**: `{method}`\n\n"
            f"{body}"
        )
        resource_links.append(f"- [{page_title}]({u}) — {source} ({date})")

    return (
        f"### Scraped Web Content for '{query}':\n\n"
        + "\n\n---\n\n".join(sections)
        + "\n\n### Verified Resource Links:\n"
        + "\n".join(resource_links)
    )


@tool
def evaluate_github_profile(github_url_or_username: str) -> str:
    """Analyze and evaluate a candidate's GitHub profile or repository to extract technical skills,
    profile README information, top repositories, primary programming languages, and project complexity.
    Provides actionable portfolio insights, resume alignment recommendations, and verified resource links.
    """
    if not github_url_or_username or not github_url_or_username.strip():
        return "Please provide a valid GitHub username or repository URL (e.g. 'the007prime' or 'https://github.com/the007prime')."

    raw_input = github_url_or_username.strip()

    # Parse username and repository
    username = ""
    repo_name = None

    if "github.com" in raw_input:
        # Match github.com/owner/repo or github.com/owner
        clean_url = raw_input.split("?")[0].split("#")[0].rstrip("/")
        match_repo = re.search(r"github\.com/([a-zA-Z0-9_\-]+)/([a-zA-Z0-9_\-\.]+)", clean_url)
        match_user = re.search(r"github\.com/([a-zA-Z0-9_\-]+)", clean_url)
        if match_repo:
            username = match_repo.group(1)
            repo_name = match_repo.group(2)
        elif match_user:
            username = match_user.group(1)
    else:
        parts = raw_input.lstrip("@").strip("/").split("/")
        username = parts[0]
        if len(parts) > 1 and parts[1]:
            repo_name = parts[1]

    if not username:
        return f"Could not determine a valid GitHub username from '{raw_input}'. Please provide a link like 'https://github.com/username'."

    headers = {
        "User-Agent": "HirePulse-CareerAgent/1.0",
        "Accept": "application/vnd.github.v3+json",
    }

    client = httpx.Client(timeout=12.0, follow_redirects=True, headers=headers)
    try:
        # 1. Fetch user profile
        user_info = {}
        try:
            u_resp = client.get(f"https://api.github.com/users/{username}")
            if u_resp.status_code == 200:
                user_info = u_resp.json()
        except Exception:
            pass

        # 2. Fetch Profile README.md (the special username/username repository)
        profile_readme = ""
        for branch in ["main", "master"]:
            try:
                r_resp = client.get(f"https://raw.githubusercontent.com/{username}/{username}/{branch}/README.md")
                if r_resp.status_code == 200 and r_resp.text.strip():
                    profile_readme = r_resp.text.strip()
                    break
            except Exception:
                pass

        # 3. Fetch Top Repositories
        repos_data = []
        try:
            repos_resp = client.get(f"https://api.github.com/users/{username}/repos?sort=updated&per_page=12")
            if repos_resp.status_code == 200:
                repos_data = repos_resp.json()
        except Exception:
            pass

        # 4. If a specific repo was requested, fetch its README
        specific_repo_readme = ""
        specific_repo_info = {}
        if repo_name:
            try:
                repo_resp = client.get(f"https://api.github.com/repos/{username}/{repo_name}")
                if repo_resp.status_code == 200:
                    specific_repo_info = repo_resp.json()
            except Exception:
                pass

            for branch in ["main", "master"]:
                try:
                    s_resp = client.get(f"https://raw.githubusercontent.com/{username}/{repo_name}/{branch}/README.md")
                    if s_resp.status_code == 200 and s_resp.text.strip():
                        specific_repo_readme = s_resp.text.strip()
                        break
                except Exception:
                    pass

        # If user not found and no README found
        if not user_info and not profile_readme and not repos_data and not specific_repo_readme:
            return (
                f"Could not retrieve public GitHub information for '{username}'. "
                f"Please verify the username or ensure the profile/repository is public."
            )

        # Build Structured Candidate Dossier
        profile_url = f"https://github.com/{username}"
        name = user_info.get("name") or username
        bio = user_info.get("bio") or "No public bio provided."
        company = user_info.get("company") or "Independent"
        location = user_info.get("location") or "Not specified"
        website = user_info.get("blog") or ""
        public_repos_count = user_info.get("public_repos", len(repos_data))
        followers = user_info.get("followers", 0)

        sections = []
        sections.append(f"## 👤 GitHub Profile: {name} (@{username})")
        sections.append(f"- **Profile URL**: [{profile_url}]({profile_url})")
        sections.append(f"- **Bio / Headline**: {bio}")
        sections.append(f"- **Company / Org**: {company} | **Location**: {location}")
        if website:
            sections.append(f"- **Personal Website / Portfolio**: [{website}]({website})")
        sections.append(f"- **Public Repositories**: {public_repos_count} | **Followers**: {followers}")

        # Profile README section
        if profile_readme:
            # Truncate if extremely long to respect token budget
            max_readme_chars = 2500
            truncated_readme = profile_readme[:max_readme_chars]
            if len(profile_readme) > max_readme_chars:
                truncated_readme += "\n\n*(Profile README truncated for length)*"
            sections.append(f"\n### 📄 Profile README (Candidate Self-Description & Portfolio)\n```markdown\n{truncated_readme}\n```")

        # Top Repositories & Tech Stack Aggregation
        languages = set()
        topics_all = set()
        repo_cards = []

        # Sort repos: prefer non-forks, highest stars first
        valid_repos = [r for r in repos_data if isinstance(r, dict)]
        valid_repos.sort(key=lambda r: (not r.get("fork", False), r.get("stargazers_count", 0)), reverse=True)

        for r in valid_repos[:8]:
            r_name = r.get("name", "Unnamed")
            r_url = r.get("html_url") or f"https://github.com/{username}/{r_name}"
            r_desc = r.get("description") or "No description provided."
            r_lang = r.get("language")
            if r_lang:
                languages.add(r_lang)
            r_stars = r.get("stargazers_count", 0)
            r_forks = r.get("forks_count", 0)
            r_topics = r.get("topics") or []
            topics_all.update(r_topics)

            meta_details = []
            if r_lang:
                meta_details.append(f"**Language**: {r_lang}")
            if r_stars > 0:
                meta_details.append(f"⭐ {r_stars}")
            if r_forks > 0:
                meta_details.append(f"🍴 {r_forks}")
            if r_topics:
                meta_details.append(f"Tags: `{'`, `'.join(r_topics[:4])}`")

            details_str = " | ".join(meta_details) if meta_details else "Public repo"
            repo_cards.append(f"- **[{r_name}]({r_url})**: {r_desc}\n  *{details_str}*")

        if languages:
            sections.append(f"\n### 💻 Discovered Programming Languages & Tech Stack\n- **Languages**: {', '.join(sorted(languages))}")
        if topics_all:
            sections.append(f"- **Key Topics / Technologies**: {', '.join(sorted(list(topics_all)[:12]))}")

        if repo_cards:
            sections.append("\n### 🚀 Featured & Recent Repositories\n" + "\n".join(repo_cards))

        # Specific Repo README if requested
        if repo_name and specific_repo_readme:
            max_s_chars = 2500
            truncated_s = specific_repo_readme[:max_s_chars]
            if len(specific_repo_readme) > max_s_chars:
                truncated_s += "\n\n*(Repository README truncated for length)*"
            sections.append(f"\n### 📁 Repository Deep-Dive: `{repo_name}` README\n```markdown\n{truncated_s}\n```")

        # Verified Resource Links Section
        verified_links = [f"- [{name}'s GitHub Profile]({profile_url}) — GitHub"]
        for r in valid_repos[:5]:
            r_name = r.get("name", "Repository")
            r_url = r.get("html_url") or f"https://github.com/{username}/{r_name}"
            verified_links.append(f"- [{r_name}]({r_url}) — GitHub Repository")

        sections.append("\n### Verified Resource Links\n" + "\n".join(verified_links))

        return "\n\n".join(sections)
    finally:
        client.close()
