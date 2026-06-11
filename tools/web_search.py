"""
Operon Web Tools.

  1. duckduckgo_search  — real-time DuckDuckGo search results (no API key)
  2. web_scrape         — fetch and extract clean text from any URL
  3. tavily_search      — Tavily Search API (requires TAVILY_API_KEY)
"""

import os
import re
import urllib.parse
import requests
from typing import Optional

try:
    from tavily import TavilyClient
    _TAVILY = True
except ImportError:
    _TAVILY = False

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:
    _BS4 = False

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}

_TIMEOUT = 12


def duckduckgo_search(query: str, max_results: int = 6) -> dict:
    """
    Search DuckDuckGo and return structured results.
    Uses the public HTML endpoint — no API key required.

    Returns:
        {"success": bool, "results": [{"title", "url", "snippet"}, ...], "error": str}
    """
    url = "https://html.duckduckgo.com/html/"
    params = {"q": query, "kl": "us-en"}

    try:
        resp = requests.post(url, data=params, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        return {"success": False, "results": [], "error": str(e)}

    if not _BS4:
        return {
            "success": False,
            "results": [],
            "error": "beautifulsoup4 not installed. Run: pip install beautifulsoup4",
        }

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    for r in soup.select(".result.results_links_deep")[:max_results]:
        title_el   = r.select_one(".result__title a")
        snippet_el = r.select_one(".result__snippet")
        url_el     = r.select_one(".result__url")

        if not title_el:
            continue

        title   = title_el.get_text(strip=True)
        snippet = snippet_el.get_text(strip=True) if snippet_el else ""
        href    = title_el.get("href", "")
        # DuckDuckGo redirect links — extract the real URL
        real_url = _extract_ddg_url(href) or (url_el.get_text(strip=True) if url_el else href)

        results.append({"title": title, "url": real_url, "snippet": snippet})

    if not results:
        # Fallback: try simpler selector
        for r in soup.select(".result")[:max_results]:
            title_el   = r.select_one("a.result__a")
            snippet_el = r.select_one(".result__snippet")
            if not title_el:
                continue
            results.append({
                "title":   title_el.get_text(strip=True),
                "url":     title_el.get("href", ""),
                "snippet": snippet_el.get_text(strip=True) if snippet_el else "",
            })

    return {"success": True, "results": results, "error": ""}


def x_search(query: str, max_results: int = 8) -> dict:
    """
    Search X / Twitter content.

    Strategy (tried in order):
      1. Nitter public instance scrape  — richest results, no API key needed
      2. DuckDuckGo site:x.com fallback — always available

    Returns:
        {"success": bool, "results": [{"text","user","url","date"},...], "error": str}
    """
    _NITTER = [
        "https://nitter.net",
        "https://nitter.privacydev.net",
        "https://nitter.poast.org",
        "https://nitter.catsarch.com",
    ]
    for base in _NITTER:
        try:
            resp = requests.get(
                f"{base}/search",
                params={"q": query, "f": "tweets"},
                headers=_HEADERS,
                timeout=8,
            )
            if not resp.ok:
                continue
            if not _BS4:
                break
            soup   = BeautifulSoup(resp.text, "html.parser")
            tweets = soup.select(".timeline-item")
            if not tweets:
                continue
            results = []
            for t in tweets[:max_results]:
                text_el = t.select_one(".tweet-content")
                user_el = t.select_one(".username")
                date_el = t.select_one(".tweet-date a")
                if not text_el:
                    continue
                user   = user_el.get_text(strip=True) if user_el else "?"
                date   = date_el.get("title", "") if date_el else ""
                href   = date_el.get("href", "") if date_el else ""
                tw_url = f"https://x.com{href}" if href.startswith("/") else href
                results.append({
                    "text": text_el.get_text(strip=True),
                    "user": user,
                    "url":  tw_url,
                    "date": date,
                })
            if results:
                return {"success": True, "results": results,
                        "source": base, "error": ""}
        except Exception:
            continue

    # Fallback: DuckDuckGo site:x.com
    ddg = duckduckgo_search(f"site:x.com {query}", max_results=max_results)
    if ddg["success"] and ddg["results"]:
        return {
            "success": True,
            "source":  "duckduckgo",
            "results": [
                {
                    "text": r["snippet"],
                    "user": (r["url"].split("/")[3]
                             if len(r["url"].split("/")) > 3 else "?"),
                    "url":  r["url"],
                    "date": "",
                }
                for r in ddg["results"]
            ],
            "error": "",
        }
    return {"success": False, "results": [],
            "error": "No X/Twitter results found. Nitter may be down."}


def _extract_ddg_url(href: str) -> Optional[str]:
    """Pull the real destination URL out of a DuckDuckGo redirect href."""
    if not href:
        return None
    parsed = urllib.parse.urlparse(href)
    qs = urllib.parse.parse_qs(parsed.query)
    return qs.get("uddg", [None])[0] or qs.get("u", [None])[0]


def web_scrape(url: str, max_chars: int = 8000) -> dict:
    """
    Fetch a URL and return its visible text content.
    Strips nav, scripts, styles, and ads as best as possible.

    Returns:
        {"success": bool, "url": str, "title": str, "content": str, "error": str}
    """
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        return {"success": False, "url": url, "title": "", "content": "", "error": str(e)}

    if not _BS4:
        # Naive strip
        clean = re.sub(r"<[^>]+>", " ", resp.text)
        clean = re.sub(r"\s+", " ", clean).strip()
        return {"success": True, "url": url, "title": "", "content": clean[:max_chars], "error": ""}

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noise
    for tag in soup(["script", "style", "nav", "footer", "header",
                     "aside", "iframe", "noscript", "form"]):
        tag.decompose()

    title = soup.title.get_text(strip=True) if soup.title else ""

    # Prefer <main> or <article> if available
    body = soup.find("main") or soup.find("article") or soup.find("body") or soup

    text = body.get_text(separator="\n", strip=True)
    # Collapse excessive blank lines
    text = re.sub(r"\n{3,}", "\n\n", text).strip()

    return {
        "success": True,
        "url":     url,
        "title":   title,
        "content": text[:max_chars],
        "error":   "",
    }


def tavily_search(query: str, max_results: int = 6) -> dict:
    """
    Search the web via the Tavily Search API and return structured results.
    Requires TAVILY_API_KEY environment variable.

    Returns:
        {"success": bool, "results": [{"title", "url", "snippet"}, ...], "error": str}
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        return {
            "success": False,
            "results": [],
            "error": "TAVILY_API_KEY environment variable not set.",
        }
    if not _TAVILY:
        return {
            "success": False,
            "results": [],
            "error": "tavily-python not installed. Run: pip install tavily-python",
        }

    try:
        client = TavilyClient(api_key=api_key)
        response = client.search(query=query, max_results=max_results)
        results = [
            {
                "title":   r.get("title", ""),
                "url":     r.get("url", ""),
                "snippet": r.get("content", ""),
            }
            for r in response.get("results", [])
        ]
        return {"success": True, "results": results, "error": ""}
    except Exception as e:
        return {"success": False, "results": [], "error": str(e)}
