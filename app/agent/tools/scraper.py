from __future__ import annotations

import html
import ipaddress
import logging
import re
import socket
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlparse

import httpx
from bs4 import BeautifulSoup
import trafilatura

from app.config import (
    JINA_API_KEY,
    SCRAPE_MAX_CHARS,
    SCRAPE_MIN_CONTENT_CHARS,
    SCRAPE_TIMEOUT_SECONDS,
    TAVILY_API_KEY,
)

logger = logging.getLogger(__name__)

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

MAX_RAW_HTML_BYTES = 2 * 1024 * 1024  # 2 MB cap before parsing

JS_CHALLENGE_MARKERS = (
    "enable javascript",
    "you need to enable javascript",
    "please enable cookies",
    "checking your browser",
    "just a moment...",
    "ddos protection by",
    "cf-browser-verification",
)

BLOCKED_HOSTNAMES = {
    "localhost",
    "localhost.localdomain",
    "metadata.google.internal",
    "169.254.169.254",
}

TEXT_CONTENT_TYPES = (
    "text/html",
    "application/xhtml+xml",
    "text/plain",
    "text/markdown",
    "application/json",
    "application/xml",
    "text/xml",
)


@dataclass
class ScrapeResult:
    """Structured result from scraping a single URL."""

    url: str
    title: str = ""
    content: str = ""
    method: str = "none"  # tier1_httpx | tier2_tavily | tier2_jina | error
    error: Optional[str] = None
    truncated: bool = False

    @property
    def ok(self) -> bool:
        return bool(self.content.strip()) and self.error is None


def clean_html_text(raw_html: str) -> str:
    """Strip HTML tags and normalize whitespace (lightweight fallback)."""
    cleaned = re.sub(
        r"<(script|style)[^>]*>.*?</\1>",
        " ",
        raw_html,
        flags=re.DOTALL | re.IGNORECASE,
    )
    cleaned = re.sub(r"<[^>]+>", " ", cleaned)
    cleaned = re.sub(r"\s+([.,!?;:])", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return html.unescape(cleaned)


def _is_disallowed_ip(ip_str: str) -> bool:
    """Return True if an IP address is private, loopback, link-local, or reserved."""
    try:
        ip_obj = ipaddress.ip_address(ip_str)
    except ValueError:
        return True
    return (
        ip_obj.is_private
        or ip_obj.is_loopback
        or ip_obj.is_link_local
        or ip_obj.is_multicast
        or ip_obj.is_reserved
        or ip_obj.is_unspecified
    )


def validate_public_url(url: str) -> Tuple[bool, str]:
    """Validate URL format and block SSRF targets (private/loopback/metadata IPs)."""
    if not url or not isinstance(url, str):
        return False, f"Invalid URL: '{url}'. Please provide a valid HTTP/HTTPS web link."

    candidate = url.strip()
    if not candidate.startswith(("http://", "https://")):
        return False, f"Invalid URL: '{url}'. Please provide a valid HTTP/HTTPS web link."

    parsed = urlparse(candidate)
    if not parsed.hostname:
        return False, f"Invalid URL: '{url}'. Missing hostname."

    if parsed.username or parsed.password:
        return False, f"Unsafe URL rejected: embedded credentials are not allowed."

    hostname = parsed.hostname.lower().rstrip(".")
    if hostname in BLOCKED_HOSTNAMES or hostname.endswith((".local", ".internal", ".localhost")):
        return False, f"Blocked URL '{url}': local or internal hostnames are not permitted."

    # Check if hostname is a literal IP address
    try:
        ipaddress.ip_address(hostname)
        if _is_disallowed_ip(hostname):
            return False, f"Blocked URL '{url}': private or loopback IP addresses are not permitted."
        return True, candidate
    except ValueError:
        pass

    # Resolve DNS if available; block if any resolved IP is private/loopback
    try:
        addr_info = socket.getaddrinfo(hostname, parsed.port or (443 if parsed.scheme == "https" else 80))
        for entry in addr_info:
            resolved_ip = entry[4][0]
            if _is_disallowed_ip(resolved_ip):
                return False, f"Blocked URL '{url}': resolves to a private or internal IP ({resolved_ip})."
    except socket.gaierror:
        # Allow fetch layer to report network/DNS errors (keeps offline mocked tests working)
        pass

    return True, candidate


def extract_clean_content(raw_html: str) -> Tuple[str, str]:
    """Extract (title, clean_markdown_text) from HTML using Trafilatura -> BS4 -> regex."""
    if not raw_html or not raw_html.strip():
        return "", ""

    bounded_html = raw_html[:MAX_RAW_HTML_BYTES]
    title = ""

    # 1. Try Trafilatura (best main-content & boilerplate removal)
    try:
        meta = trafilatura.extract_metadata(bounded_html)
        if meta and meta.title:
            title = meta.title.strip()

        extracted = trafilatura.extract(
            bounded_html,
            output_format="markdown",
            include_links=False,
            include_tables=True,
            favor_precision=False,
        )
        if extracted and extracted.strip():
            return title, extracted.strip()
    except Exception as exc:
        logger.debug("Trafilatura extraction failed, falling back to BS4: %s", exc)

    # 2. Fallback to BeautifulSoup DOM cleaning
    try:
        soup = BeautifulSoup(bounded_html, "lxml")
        if not title and soup.title and soup.title.string:
            title = soup.title.string.strip()

        for tag in soup(["script", "style", "noscript", "svg", "nav", "footer", "aside"]):
            tag.decompose()

        lines: List[str] = []
        for elem in soup.find_all(["h1", "h2", "h3", "h4", "p", "li", "pre", "blockquote"]):
            text = re.sub(r"\s+", " ", elem.get_text(" ", strip=True))
            if not text:
                continue
            if elem.name in ("h1", "h2", "h3", "h4"):
                level = "#" * int(elem.name[1])
                lines.append(f"{level} {text}")
            elif elem.name == "li":
                lines.append(f"- {text}")
            else:
                lines.append(text)

        if lines:
            # Deduplicate consecutive identical lines
            deduped: List[str] = []
            for line in lines:
                if not deduped or deduped[-1] != line:
                    deduped.append(line)
            return title, "\n\n".join(deduped).strip()

        body_text = re.sub(r"\s+", " ", soup.get_text(" ", strip=True)).strip()
        if body_text:
            return title, body_text
    except Exception as exc:
        logger.debug("BeautifulSoup extraction failed, falling back to regex: %s", exc)

    # 3. Last-resort regex strip
    return title, clean_html_text(bounded_html)


def _needs_js_fallback(text: str, status_code: Optional[int] = None) -> bool:
    """Detect whether Tier 1 output is likely a JS-rendered shell or soft bot block."""
    if status_code in (401, 403, 429, 503):
        return True
    stripped = (text or "").strip()
    if len(stripped) < SCRAPE_MIN_CONTENT_CHARS:
        return True
    lower = stripped.lower()
    return any(marker in lower for marker in JS_CHALLENGE_MARKERS)


def focus_and_truncate(
    content: str,
    focus_query: Optional[str] = None,
    max_chars: int = SCRAPE_MAX_CHARS,
) -> Tuple[str, bool]:
    """Keep content within max_chars, prioritizing sections relevant to focus_query."""
    if len(content) <= max_chars:
        return content, False

    if not focus_query or not focus_query.strip():
        return content[:max_chars], True

    keywords = {
        w.lower()
        for w in re.findall(r"[a-zA-Z0-9_+#.-]{3,}", focus_query)
        if w.lower() not in {"the", "and", "for", "with", "from", "that", "this", "what", "how"}
    }
    if not keywords:
        return content[:max_chars], True

    paragraphs = [p.strip() for p in re.split(r"\n{2,}", content) if p.strip()]
    if len(paragraphs) <= 2:
        return content[:max_chars], True

    # Always preserve the lead/intro paragraph for context, then rank remaining by relevance
    lead = paragraphs[0]
    scored: List[Tuple[int, int, str]] = []
    for idx, para in enumerate(paragraphs[1:], start=1):
        lower_p = para.lower()
        hits = sum(1 for kw in keywords if kw in lower_p)
        scored.append((hits, -idx, para))

    # If nothing matched keywords, fall back to sequential head truncation
    if not any(item[0] > 0 for item in scored):
        return content[:max_chars], True

    scored.sort(reverse=True)
    selected_indices = {0}
    used_chars = len(lead)

    for hits, neg_idx, para in scored:
        if hits == 0 and used_chars > max_chars * 0.6:
            break
        cost = len(para) + 2
        if used_chars + cost <= max_chars:
            selected_indices.add(-neg_idx)
            used_chars += cost

    ordered = [paragraphs[i] for i in sorted(selected_indices)]
    joined = "\n\n".join(ordered)
    if len(joined) > max_chars:
        joined = joined[:max_chars]
    return joined, True


def _fetch_tier1_httpx(url: str, timeout: float = SCRAPE_TIMEOUT_SECONDS) -> Tuple[ScrapeResult, bool]:
    """Tier 1: Fast local HTTP fetch + Trafilatura/BS4 extraction.

    Returns (ScrapeResult, should_try_tier2).
    """
    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=DEFAULT_HEADERS) as client:
            resp = client.get(url)
            resp.raise_for_status()

        headers = getattr(resp, "headers", None)
        if isinstance(headers, dict) or hasattr(headers, "get"):
            ctype = headers.get("content-type", "")
            if isinstance(ctype, str) and ctype.strip():
                mime = ctype.split(";")[0].strip().lower()
                if mime and not any(mime.startswith(allowed) for allowed in TEXT_CONTENT_TYPES):
                    return (
                        ScrapeResult(
                            url=url,
                            method="tier1_httpx",
                            error=f"Unsupported content type '{mime}' (non-HTML/binary resource).",
                        ),
                        False,
                    )

        title, text = extract_clean_content(resp.text)
        needs_t2 = _needs_js_fallback(text, status_code=getattr(resp, "status_code", 200))
        return ScrapeResult(url=url, title=title, content=text, method="tier1_httpx"), needs_t2

    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code if exc.response is not None else None
        should_retry_t2 = status in (401, 403, 429, 503)
        return (
            ScrapeResult(
                url=url,
                method="tier1_httpx",
                error=f"HTTP error occurred while visiting {url}: {status} ({exc})",
            ),
            should_retry_t2,
        )
    except Exception as exc:
        return (
            ScrapeResult(
                url=url,
                method="tier1_httpx",
                error=f"Failed to retrieve webpage at {url}: {exc}",
            ),
            True,
        )


def _fetch_tier2_tavily_extract(
    urls: List[str],
    extract_depth: str = "basic",
) -> Dict[str, ScrapeResult]:
    """Tier 2a: Batch JS-aware extraction via Tavily Extract API."""
    results: Dict[str, ScrapeResult] = {}
    if not TAVILY_API_KEY or not urls:
        return results

    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.extract(urls=urls[:20], extract_depth=extract_depth)
        for item in response.get("results", []):
            item_url = item.get("url", "")
            raw_content = (item.get("raw_content") or "").strip()
            if item_url and raw_content:
                results[item_url] = ScrapeResult(
                    url=item_url,
                    title=item.get("title", ""),
                    content=raw_content,
                    method="tier2_tavily",
                )
    except Exception as exc:
        logger.warning("Tavily extract failed: %s", exc)

    return results


def _fetch_tier2_jina(url: str, timeout: float = SCRAPE_TIMEOUT_SECONDS) -> ScrapeResult:
    """Tier 2b: Zero-config JS-rendered Markdown extraction via Jina Reader."""
    jina_url = f"https://r.jina.ai/{url}"
    headers = {
        "Accept": "text/markdown",
        "User-Agent": DEFAULT_HEADERS["User-Agent"],
    }
    if JINA_API_KEY:
        headers["Authorization"] = f"Bearer {JINA_API_KEY}"

    try:
        with httpx.Client(timeout=timeout, follow_redirects=True, headers=headers) as client:
            resp = client.get(jina_url)
            resp.raise_for_status()

        text = resp.text.strip()
        if not text:
            return ScrapeResult(url=url, method="tier2_jina", error="Jina Reader returned empty content.")

        title = ""
        for line in text.splitlines()[:5]:
            if line.lower().startswith("title:"):
                title = line.split(":", 1)[1].strip()
                break

        return ScrapeResult(url=url, title=title, content=text, method="tier2_jina")
    except Exception as exc:
        return ScrapeResult(url=url, method="tier2_jina", error=f"Jina Reader fallback failed: {exc}")


def scrape_url(
    url: str,
    focus_query: Optional[str] = None,
    max_chars: int = SCRAPE_MAX_CHARS,
) -> ScrapeResult:
    """Scrape a single URL using Tier 1 (httpx+trafilatura) -> Tier 2a (Tavily) -> Tier 2b (Jina)."""
    is_valid, msg_or_url = validate_public_url(url)
    if not is_valid:
        return ScrapeResult(url=url, method="error", error=msg_or_url)

    clean_url = msg_or_url
    t1_res, try_t2 = _fetch_tier1_httpx(clean_url)

    best_res = t1_res
    if try_t2:
        # Tier 2a: Tavily Extract (if key configured)
        if TAVILY_API_KEY:
            t2_map = _fetch_tier2_tavily_extract([clean_url])
            t2_res = t2_map.get(clean_url)
            if t2_res and t2_res.ok and len(t2_res.content) > len(t1_res.content):
                best_res = t2_res

        # Tier 2b: Jina Reader if we still don't have adequate content
        if not best_res.ok or len(best_res.content) < SCRAPE_MIN_CONTENT_CHARS:
            jina_res = _fetch_tier2_jina(clean_url)
            if jina_res.ok and len(jina_res.content) > len(best_res.content):
                best_res = jina_res

    if best_res.ok:
        focused, was_truncated = focus_and_truncate(
            best_res.content,
            focus_query=focus_query,
            max_chars=max_chars,
        )
        best_res.content = focused
        best_res.truncated = was_truncated
        return best_res

    return best_res


def scrape_urls(
    urls: List[str],
    focus_query: Optional[str] = None,
    max_chars_per_page: int = SCRAPE_MAX_CHARS,
) -> List[ScrapeResult]:
    """Concurrently scrape multiple URLs with batched Tavily Extract & Jina fallbacks."""
    valid_urls: List[str] = []
    results_by_url: Dict[str, ScrapeResult] = {}

    for u in urls:
        ok, msg_or_url = validate_public_url(u)
        if not ok:
            results_by_url[u] = ScrapeResult(url=u, method="error", error=msg_or_url)
        elif msg_or_url not in results_by_url:
            valid_urls.append(msg_or_url)

    if not valid_urls:
        return list(results_by_url.values())

    # 1. Run Tier 1 concurrently
    tier2_candidates: List[str] = []
    with ThreadPoolExecutor(max_workers=min(len(valid_urls), 4)) as executor:
        future_map = {executor.submit(_fetch_tier1_httpx, u): u for u in valid_urls}
        for future in as_completed(future_map):
            u = future_map[future]
            try:
                t1_res, try_t2 = future.result()
            except Exception as exc:
                t1_res = ScrapeResult(url=u, method="tier1_httpx", error=str(exc))
                try_t2 = True

            results_by_url[u] = t1_res
            if try_t2:
                tier2_candidates.append(u)

    # 2. Batch Tier 2a (Tavily Extract) for any URLs that needed JS/bot fallback
    remaining_for_jina: List[str] = []
    if tier2_candidates and TAVILY_API_KEY:
        t2_map = _fetch_tier2_tavily_extract(tier2_candidates)
        for u in tier2_candidates:
            t2_res = t2_map.get(u)
            if t2_res and t2_res.ok and len(t2_res.content) > len(results_by_url[u].content):
                results_by_url[u] = t2_res
            else:
                remaining_for_jina.append(u)
    else:
        remaining_for_jina = tier2_candidates

    # 3. Tier 2b (Jina Reader) concurrently for any remaining failed/sparse URLs
    if remaining_for_jina:
        with ThreadPoolExecutor(max_workers=min(len(remaining_for_jina), 3)) as executor:
            future_map = {executor.submit(_fetch_tier2_jina, u): u for u in remaining_for_jina}
            for future in as_completed(future_map):
                u = future_map[future]
                try:
                    jina_res = future.result()
                    if jina_res.ok and len(jina_res.content) > len(results_by_url[u].content):
                        results_by_url[u] = jina_res
                except Exception:
                    pass

    # 4. Focus & truncate in original URL order
    ordered_results: List[ScrapeResult] = []
    for u in valid_urls:
        res = results_by_url[u]
        if res.ok:
            focused, was_truncated = focus_and_truncate(
                res.content,
                focus_query=focus_query,
                max_chars=max_chars_per_page,
            )
            res.content = focused
            res.truncated = was_truncated
        ordered_results.append(res)

    return ordered_results
