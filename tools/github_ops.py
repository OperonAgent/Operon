"""tools/github_ops.py — GitHub API integration for Operon.

All functions return {success, output, error}.
Requires GITHUB_TOKEN env var (or personal access token via params).
No extra packages beyond stdlib + requests (already in requirements).
"""

from __future__ import annotations

import os
import json
import time
from typing import Any, Dict, List, Optional

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _gh_token(token: str = "") -> str:
    return token or os.environ.get("GITHUB_TOKEN", os.environ.get("GH_TOKEN", ""))


def _headers(token: str = "") -> Dict[str, str]:
    tok = _gh_token(token)
    h = {
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if tok:
        h["Authorization"] = f"Bearer {tok}"
    return h


def _get(path: str, params: Dict = None, token: str = "") -> Dict:
    """Perform a GitHub API GET and return parsed JSON or error dict."""
    import urllib.request
    import urllib.parse
    import urllib.error

    base = "https://api.github.com"
    url  = f"{base}{path}"
    if params:
        url += "?" + urllib.parse.urlencode(params)

    req = urllib.request.Request(url, headers=_headers(token))
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode())
            return {"ok": True, "data": data, "status": resp.status}
    except urllib.error.HTTPError as e:
        try:
            body = json.loads(e.read().decode())
            msg  = body.get("message", str(e))
        except Exception:
            msg = str(e)
        return {"ok": False, "data": None, "status": e.code, "error": msg}
    except Exception as exc:
        return {"ok": False, "data": None, "status": 0, "error": str(exc)}


def _post(path: str, body: Dict, method: str = "POST", token: str = "") -> Dict:
    """Perform a GitHub API write (POST/PATCH/PUT/DELETE)."""
    import urllib.request
    import urllib.error

    base = "https://api.github.com"
    url  = f"{base}{path}"
    data = json.dumps(body).encode()

    req = urllib.request.Request(
        url,
        data=data,
        headers={**_headers(token), "Content-Type": "application/json"},
        method=method,
    )
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            raw  = resp.read()
            data = json.loads(raw) if raw else {}
            return {"ok": True, "data": data, "status": resp.status}
    except urllib.error.HTTPError as e:
        try:
            body_err = json.loads(e.read().decode())
            msg      = body_err.get("message", str(e))
        except Exception:
            msg = str(e)
        return {"ok": False, "data": None, "status": e.code, "error": msg}
    except Exception as exc:
        return {"ok": False, "data": None, "status": 0, "error": str(exc)}


def _ok(output: Any) -> Dict:
    return {"success": True, "output": output, "error": ""}


def _err(msg: str) -> Dict:
    return {"success": False, "output": None, "error": msg}


def _no_token() -> Dict:
    return _err(
        "No GitHub token found. Set GITHUB_TOKEN env var or pass token= param. "
        "Create one at: https://github.com/settings/tokens"
    )


# ---------------------------------------------------------------------------
# Public tool functions
# ---------------------------------------------------------------------------

def github_repo_info(owner: str, repo: str, token: str = "") -> Dict:
    """Return metadata about a GitHub repository.

    Returns name, description, stars, forks, language, open issues,
    default branch, visibility, and license.
    """
    if not owner or not repo:
        return _err("owner and repo are required.")
    r = _get(f"/repos/{owner}/{repo}", token=token)
    if not r["ok"]:
        return _err(f"GitHub API error ({r['status']}): {r['error']}")
    d = r["data"]
    return _ok({
        "full_name":      d.get("full_name", ""),
        "description":    d.get("description", ""),
        "stars":          d.get("stargazers_count", 0),
        "forks":          d.get("forks_count", 0),
        "open_issues":    d.get("open_issues_count", 0),
        "language":       d.get("language", ""),
        "default_branch": d.get("default_branch", "main"),
        "visibility":     d.get("visibility", ""),
        "license":        (d.get("license") or {}).get("spdx_id", ""),
        "html_url":       d.get("html_url", ""),
        "created_at":     d.get("created_at", ""),
        "updated_at":     d.get("updated_at", ""),
        "topics":         d.get("topics", []),
        "size_kb":        d.get("size", 0),
    })


def github_list_repos(
    username: str = "",
    org: str = "",
    sort: str = "updated",
    limit: int = 20,
    token: str = "",
) -> Dict:
    """List repositories for a user or organisation.

    Provide username OR org (not both).
    sort: 'created' | 'updated' | 'pushed' | 'full_name'
    """
    if not username and not org:
        return _err("Provide username or org.")
    path   = f"/users/{username}/repos" if username else f"/orgs/{org}/repos"
    params = {"sort": sort, "per_page": min(limit, 100)}
    r = _get(path, params=params, token=token)
    if not r["ok"]:
        return _err(f"GitHub API error ({r['status']}): {r['error']}")
    repos = [
        {
            "name":        repo.get("name", ""),
            "full_name":   repo.get("full_name", ""),
            "description": repo.get("description", ""),
            "stars":       repo.get("stargazers_count", 0),
            "language":    repo.get("language", ""),
            "private":     repo.get("private", False),
            "html_url":    repo.get("html_url", ""),
            "updated_at":  repo.get("updated_at", ""),
        }
        for repo in (r["data"] or [])[:limit]
    ]
    return _ok({"repos": repos, "count": len(repos)})


def github_list_issues(
    owner: str,
    repo: str,
    state: str = "open",
    labels: str = "",
    limit: int = 20,
    token: str = "",
) -> Dict:
    """List issues for a repository.

    state: 'open' | 'closed' | 'all'
    labels: comma-separated label names to filter by.
    """
    if not owner or not repo:
        return _err("owner and repo are required.")
    params: Dict[str, Any] = {
        "state":    state,
        "per_page": min(limit, 100),
    }
    if labels:
        params["labels"] = labels
    r = _get(f"/repos/{owner}/{repo}/issues", params=params, token=token)
    if not r["ok"]:
        return _err(f"GitHub API error ({r['status']}): {r['error']}")
    issues = [
        {
            "number":     i.get("number"),
            "title":      i.get("title", ""),
            "state":      i.get("state", ""),
            "labels":     [lb.get("name", "") for lb in i.get("labels", [])],
            "author":     (i.get("user") or {}).get("login", ""),
            "created_at": i.get("created_at", ""),
            "html_url":   i.get("html_url", ""),
            "comments":   i.get("comments", 0),
            "body_preview": (i.get("body") or "")[:200],
        }
        for i in (r["data"] or [])[:limit]
        if "pull_request" not in i   # skip PRs that appear as issues
    ]
    return _ok({"issues": issues, "count": len(issues), "state": state})


def github_create_issue(
    owner: str,
    repo: str,
    title: str,
    body: str = "",
    labels: List[str] = None,
    assignees: List[str] = None,
    token: str = "",
) -> Dict:
    """Create a new GitHub issue.  Requires GITHUB_TOKEN with repo scope."""
    if not _gh_token(token):
        return _no_token()
    if not owner or not repo or not title:
        return _err("owner, repo, and title are required.")
    payload: Dict[str, Any] = {"title": title}
    if body:
        payload["body"] = body
    if labels:
        payload["labels"] = labels
    if assignees:
        payload["assignees"] = assignees
    r = _post(f"/repos/{owner}/{repo}/issues", payload, token=token)
    if not r["ok"]:
        return _err(f"GitHub API error ({r['status']}): {r['error']}")
    d = r["data"]
    return _ok({
        "number":   d.get("number"),
        "title":    d.get("title", ""),
        "html_url": d.get("html_url", ""),
        "state":    d.get("state", "open"),
    })


def github_list_prs(
    owner: str,
    repo: str,
    state: str = "open",
    limit: int = 20,
    token: str = "",
) -> Dict:
    """List pull requests for a repository.

    state: 'open' | 'closed' | 'all'
    """
    if not owner or not repo:
        return _err("owner and repo are required.")
    params = {"state": state, "per_page": min(limit, 100)}
    r = _get(f"/repos/{owner}/{repo}/pulls", params=params, token=token)
    if not r["ok"]:
        return _err(f"GitHub API error ({r['status']}): {r['error']}")
    prs = [
        {
            "number":      pr.get("number"),
            "title":       pr.get("title", ""),
            "state":       pr.get("state", ""),
            "author":      (pr.get("user") or {}).get("login", ""),
            "head":        (pr.get("head") or {}).get("ref", ""),
            "base":        (pr.get("base") or {}).get("ref", ""),
            "created_at":  pr.get("created_at", ""),
            "html_url":    pr.get("html_url", ""),
            "draft":       pr.get("draft", False),
            "reviews":     pr.get("requested_reviewers", []),
            "body_preview": (pr.get("body") or "")[:200],
        }
        for pr in (r["data"] or [])[:limit]
    ]
    return _ok({"pull_requests": prs, "count": len(prs), "state": state})


def github_search_code(
    query: str,
    language: str = "",
    owner: str = "",
    repo: str = "",
    limit: int = 10,
    token: str = "",
) -> Dict:
    """Search code across GitHub repositories.

    Constructs a GitHub code search query from the natural language query.
    Optionally filter by language, owner, or specific repo.
    Rate-limited to ~10 req/min without auth.
    """
    if not query:
        return _err("query is required.")
    q = query
    if language:
        q += f" language:{language}"
    if owner and repo:
        q += f" repo:{owner}/{repo}"
    elif owner:
        q += f" user:{owner}"
    params = {"q": q, "per_page": min(limit, 30)}
    r = _get("/search/code", params=params, token=token)
    if not r["ok"]:
        if r["status"] == 403:
            return _err(
                "GitHub rate limit hit or authentication required. "
                "Set GITHUB_TOKEN to increase limits."
            )
        return _err(f"GitHub API error ({r['status']}): {r['error']}")
    items = r["data"].get("items", [])
    results = [
        {
            "name":       item.get("name", ""),
            "path":       item.get("path", ""),
            "repository": item.get("repository", {}).get("full_name", ""),
            "html_url":   item.get("html_url", ""),
            "sha":        item.get("sha", ""),
        }
        for item in items[:limit]
    ]
    return _ok({
        "query":        q,
        "total_count":  r["data"].get("total_count", 0),
        "results":      results,
        "count":        len(results),
    })


def github_search_repos(
    query: str,
    language: str = "",
    sort: str = "stars",
    limit: int = 10,
    token: str = "",
) -> Dict:
    """Search GitHub repositories by keyword.

    sort: 'stars' | 'forks' | 'help-wanted-issues' | 'updated'
    """
    if not query:
        return _err("query is required.")
    q = query
    if language:
        q += f" language:{language}"
    params = {"q": q, "sort": sort, "per_page": min(limit, 30)}
    r = _get("/search/repositories", params=params, token=token)
    if not r["ok"]:
        return _err(f"GitHub API error ({r['status']}): {r['error']}")
    items = r["data"].get("items", [])
    results = [
        {
            "full_name":   item.get("full_name", ""),
            "description": item.get("description", ""),
            "stars":       item.get("stargazers_count", 0),
            "forks":       item.get("forks_count", 0),
            "language":    item.get("language", ""),
            "html_url":    item.get("html_url", ""),
            "topics":      item.get("topics", []),
        }
        for item in items[:limit]
    ]
    return _ok({
        "query":       q,
        "total_count": r["data"].get("total_count", 0),
        "results":     results,
        "count":       len(results),
    })


def github_get_file(
    owner: str,
    repo: str,
    path: str,
    ref: str = "",
    token: str = "",
) -> Dict:
    """Fetch the contents of a file from a GitHub repository.

    Returns the decoded text content (base64-decoded automatically).
    ref: branch name, tag, or commit SHA (default: repo default branch).
    """
    if not owner or not repo or not path:
        return _err("owner, repo, and path are required.")
    import base64
    params = {}
    if ref:
        params["ref"] = ref
    r = _get(f"/repos/{owner}/{repo}/contents/{path.lstrip('/')}", params=params, token=token)
    if not r["ok"]:
        return _err(f"GitHub API error ({r['status']}): {r['error']}")
    d = r["data"]
    if isinstance(d, list):
        # path is a directory
        return _ok({
            "type":  "directory",
            "path":  path,
            "items": [{"name": f.get("name"), "type": f.get("type"), "path": f.get("path")} for f in d],
        })
    enc = d.get("encoding", "")
    raw = d.get("content", "")
    if enc == "base64":
        try:
            content = base64.b64decode(raw.replace("\n", "")).decode("utf-8", errors="replace")
        except Exception:
            content = raw
    else:
        content = raw
    return _ok({
        "type":      "file",
        "path":      d.get("path", path),
        "name":      d.get("name", ""),
        "size":      d.get("size", 0),
        "sha":       d.get("sha", ""),
        "html_url":  d.get("html_url", ""),
        "content":   content,
    })


def github_create_gist(
    description: str,
    files: Dict[str, str],
    public: bool = False,
    token: str = "",
) -> Dict:
    """Create a GitHub Gist.

    files: dict of {filename: content}.
    Returns the gist URL and ID.
    Requires GITHUB_TOKEN with gist scope.
    """
    if not _gh_token(token):
        return _no_token()
    if not files:
        return _err("files dict is required (filename → content).")
    payload = {
        "description": description or "",
        "public":      public,
        "files":       {name: {"content": content} for name, content in files.items()},
    }
    r = _post("/gists", payload, token=token)
    if not r["ok"]:
        return _err(f"GitHub API error ({r['status']}): {r['error']}")
    d = r["data"]
    return _ok({
        "id":          d.get("id", ""),
        "html_url":    d.get("html_url", ""),
        "description": d.get("description", ""),
        "public":      d.get("public", False),
        "files":       list(d.get("files", {}).keys()),
    })


def github_user_info(
    username: str = "",
    token: str = "",
) -> Dict:
    """Return public profile information for a GitHub user.

    If username is omitted and a token is provided, returns the authenticated user's info.
    """
    path = f"/users/{username}" if username else "/user"
    if not username and not _gh_token(token):
        return _no_token()
    r = _get(path, token=token)
    if not r["ok"]:
        return _err(f"GitHub API error ({r['status']}): {r['error']}")
    d = r["data"]
    return _ok({
        "login":       d.get("login", ""),
        "name":        d.get("name", ""),
        "bio":         d.get("bio", ""),
        "company":     d.get("company", ""),
        "location":    d.get("location", ""),
        "email":       d.get("email", ""),
        "public_repos": d.get("public_repos", 0),
        "followers":   d.get("followers", 0),
        "following":   d.get("following", 0),
        "html_url":    d.get("html_url", ""),
        "created_at":  d.get("created_at", ""),
        "avatar_url":  d.get("avatar_url", ""),
    })


def github_list_commits(
    owner: str,
    repo: str,
    branch: str = "",
    author: str = "",
    limit: int = 20,
    token: str = "",
) -> Dict:
    """List recent commits for a repository branch.

    branch: branch name (default: repo default branch).
    author: filter by GitHub username or email.
    """
    if not owner or not repo:
        return _err("owner and repo are required.")
    params: Dict[str, Any] = {"per_page": min(limit, 100)}
    if branch:
        params["sha"] = branch
    if author:
        params["author"] = author
    r = _get(f"/repos/{owner}/{repo}/commits", params=params, token=token)
    if not r["ok"]:
        return _err(f"GitHub API error ({r['status']}): {r['error']}")
    commits = [
        {
            "sha":     c.get("sha", "")[:7],
            "message": (c.get("commit", {}).get("message", "") or "").split("\n")[0],
            "author":  c.get("commit", {}).get("author", {}).get("name", ""),
            "date":    c.get("commit", {}).get("author", {}).get("date", ""),
            "html_url": c.get("html_url", ""),
        }
        for c in (r["data"] or [])[:limit]
    ]
    return _ok({"commits": commits, "count": len(commits), "branch": branch or "default"})
