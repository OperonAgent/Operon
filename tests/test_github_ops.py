"""Tests for tools/github_ops.py"""
import json
import pytest
from unittest.mock import patch, MagicMock
from io import BytesIO
from urllib.error import HTTPError

from tools.github_ops import (
    github_repo_info, github_list_repos, github_list_issues,
    github_create_issue, github_list_prs, github_search_code,
    github_search_repos, github_get_file, github_create_gist,
    github_user_info, github_list_commits,
    _gh_token, _headers, _get, _post, _ok, _err, _no_token,
)


# ── Fixtures / helpers ────────────────────────────────────────────────────────

def make_response(data: dict, status: int = 200):
    """Return a mock urllib response object."""
    m = MagicMock()
    m.status = status
    m.read.return_value = json.dumps(data).encode()
    m.__enter__ = lambda s: s
    m.__exit__ = MagicMock(return_value=False)
    return m


def make_http_error(status: int, message: str = "Not found"):
    """Return a mock HTTPError."""
    body = json.dumps({"message": message}).encode()
    err = HTTPError(
        url="https://api.github.com/test",
        code=status,
        msg=message,
        hdrs={},  # type: ignore
        fp=BytesIO(body),
    )
    return err


# ── _gh_token ─────────────────────────────────────────────────────────────────

class TestGhToken:
    def test_explicit_token_returned(self):
        assert _gh_token("mytoken") == "mytoken"

    def test_env_var_fallback(self, monkeypatch):
        monkeypatch.setenv("GITHUB_TOKEN", "envtoken")
        assert _gh_token() == "envtoken"

    def test_gh_token_env_fallback(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.setenv("GH_TOKEN", "ghtoken")
        assert _gh_token() == "ghtoken"

    def test_empty_when_no_token(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        assert _gh_token() == ""


# ── _headers ──────────────────────────────────────────────────────────────────

class TestHeaders:
    def test_auth_header_present_with_token(self):
        h = _headers("tok123")
        assert "Authorization" in h
        assert "Bearer tok123" in h["Authorization"]

    def test_no_auth_without_token(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        h = _headers()
        assert "Authorization" not in h

    def test_accept_header_always_present(self):
        h = _headers()
        assert "Accept" in h
        assert "github" in h["Accept"]


# ── _ok / _err helpers ────────────────────────────────────────────────────────

class TestHelpers:
    def test_ok_structure(self):
        r = _ok({"key": "value"})
        assert r["success"] is True
        assert r["output"] == {"key": "value"}
        assert r["error"] == ""

    def test_err_structure(self):
        r = _err("something went wrong")
        assert r["success"] is False
        assert r["output"] is None
        assert "something went wrong" in r["error"]

    def test_no_token_is_error(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        r = _no_token()
        assert not r["success"]
        assert "GITHUB_TOKEN" in r["error"]


# ── github_repo_info ──────────────────────────────────────────────────────────

class TestRepoInfo:
    SAMPLE = {
        "full_name": "octocat/Hello-World",
        "description": "My first repository",
        "stargazers_count": 100,
        "forks_count": 50,
        "open_issues_count": 5,
        "language": "Python",
        "default_branch": "main",
        "visibility": "public",
        "license": {"spdx_id": "MIT"},
        "html_url": "https://github.com/octocat/Hello-World",
        "created_at": "2011-01-26T19:01:12Z",
        "updated_at": "2024-01-01T00:00:00Z",
        "topics": ["python", "ai"],
        "size": 1024,
    }

    def test_missing_owner(self):
        r = github_repo_info("", "Hello-World")
        assert not r["success"]
        assert "required" in r["error"].lower()

    def test_missing_repo(self):
        r = github_repo_info("octocat", "")
        assert not r["success"]

    def test_success(self):
        with patch("urllib.request.urlopen", return_value=make_response(self.SAMPLE)):
            r = github_repo_info("octocat", "Hello-World", token="tok")
        assert r["success"]
        assert r["output"]["stars"] == 100
        assert r["output"]["language"] == "Python"
        assert r["output"]["license"] == "MIT"

    def test_http_error_404(self):
        with patch("urllib.request.urlopen", side_effect=make_http_error(404, "Not Found")):
            r = github_repo_info("octocat", "nonexistent")
        assert not r["success"]
        assert "404" in r["error"] or "Not Found" in r["error"]


# ── github_list_repos ─────────────────────────────────────────────────────────

class TestListRepos:
    SAMPLE = [
        {
            "name": "repo1", "full_name": "user/repo1", "description": "Desc",
            "stargazers_count": 10, "language": "Python",
            "private": False, "html_url": "https://github.com/user/repo1",
            "updated_at": "2024-01-01T00:00:00Z",
        }
    ]

    def test_requires_username_or_org(self):
        r = github_list_repos()
        assert not r["success"]

    def test_list_by_username(self):
        with patch("urllib.request.urlopen", return_value=make_response(self.SAMPLE)):
            r = github_list_repos(username="user")
        assert r["success"]
        assert r["output"]["count"] == 1
        assert r["output"]["repos"][0]["name"] == "repo1"

    def test_list_by_org(self):
        with patch("urllib.request.urlopen", return_value=make_response(self.SAMPLE)):
            r = github_list_repos(org="myorg")
        assert r["success"]

    def test_limit_respected(self):
        big_list = self.SAMPLE * 30   # 30 repos
        with patch("urllib.request.urlopen", return_value=make_response(big_list)):
            r = github_list_repos(username="user", limit=5)
        assert r["success"]
        assert r["output"]["count"] <= 5


# ── github_list_issues ────────────────────────────────────────────────────────

class TestListIssues:
    SAMPLE = [
        {
            "number": 1, "title": "Bug report", "state": "open",
            "labels": [{"name": "bug"}], "user": {"login": "alice"},
            "created_at": "2024-01-01T00:00:00Z",
            "html_url": "https://github.com/o/r/issues/1",
            "comments": 2, "body": "Bug details here",
        },
        # this one should be excluded (has pull_request key)
        {
            "number": 2, "title": "PR", "state": "open",
            "pull_request": {"url": "..."},
            "labels": [], "user": {"login": "bob"},
            "created_at": "2024-01-02T00:00:00Z",
            "html_url": "https://github.com/o/r/pull/2",
            "comments": 0, "body": "",
        },
    ]

    def test_requires_owner_and_repo(self):
        assert not github_list_issues("", "repo")["success"]
        assert not github_list_issues("owner", "")["success"]

    def test_excludes_prs(self):
        with patch("urllib.request.urlopen", return_value=make_response(self.SAMPLE)):
            r = github_list_issues("o", "r")
        assert r["success"]
        # Only 1 real issue (PR filtered out)
        assert r["output"]["count"] == 1
        assert r["output"]["issues"][0]["number"] == 1

    def test_label_in_output(self):
        with patch("urllib.request.urlopen", return_value=make_response(self.SAMPLE)):
            r = github_list_issues("o", "r")
        assert "bug" in r["output"]["issues"][0]["labels"]


# ── github_create_issue ───────────────────────────────────────────────────────

class TestCreateIssue:
    RESPONSE = {
        "number": 42, "title": "New Issue",
        "html_url": "https://github.com/o/r/issues/42",
        "state": "open",
    }

    def test_no_token_returns_error(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        r = github_create_issue("o", "r", "Title")
        assert not r["success"]
        assert "token" in r["error"].lower()

    def test_missing_title(self):
        r = github_create_issue("o", "r", "", token="tok")
        assert not r["success"]

    def test_success(self):
        with patch("urllib.request.urlopen", return_value=make_response(self.RESPONSE, 201)):
            r = github_create_issue("o", "r", "New Issue",
                                    body="Details", labels=["bug"], token="tok")
        assert r["success"]
        assert r["output"]["number"] == 42
        assert r["output"]["html_url"] != ""


# ── github_list_prs ───────────────────────────────────────────────────────────

class TestListPRs:
    SAMPLE = [
        {
            "number": 5, "title": "Add feature",
            "state": "open", "user": {"login": "dev"},
            "head": {"ref": "feature/x"}, "base": {"ref": "main"},
            "created_at": "2024-01-01T00:00:00Z",
            "html_url": "https://github.com/o/r/pull/5",
            "draft": False, "requested_reviewers": [],
            "body": "PR description",
        }
    ]

    def test_requires_owner_and_repo(self):
        assert not github_list_prs("", "r")["success"]

    def test_success(self):
        with patch("urllib.request.urlopen", return_value=make_response(self.SAMPLE)):
            r = github_list_prs("o", "r")
        assert r["success"]
        assert r["output"]["count"] == 1
        pr = r["output"]["pull_requests"][0]
        assert pr["number"] == 5
        assert pr["head"] == "feature/x"


# ── github_search_code ────────────────────────────────────────────────────────

class TestSearchCode:
    SAMPLE = {
        "total_count": 1,
        "items": [
            {
                "name": "main.py",
                "path": "src/main.py",
                "repository": {"full_name": "o/r"},
                "html_url": "https://github.com/o/r/blob/main/src/main.py",
                "sha": "abc123",
            }
        ],
    }

    def test_requires_query(self):
        assert not github_search_code("")["success"]

    def test_success(self):
        with patch("urllib.request.urlopen", return_value=make_response(self.SAMPLE)):
            r = github_search_code("def run_agent", language="python")
        assert r["success"]
        assert r["output"]["total_count"] == 1
        assert r["output"]["results"][0]["path"] == "src/main.py"

    def test_rate_limit_error(self):
        with patch("urllib.request.urlopen",
                   side_effect=make_http_error(403, "rate limit exceeded")):
            r = github_search_code("myquery")
        assert not r["success"]
        assert "rate limit" in r["error"].lower() or "authentication" in r["error"].lower()


# ── github_search_repos ───────────────────────────────────────────────────────

class TestSearchRepos:
    SAMPLE = {
        "total_count": 1,
        "items": [
            {
                "full_name": "awesome/repo",
                "description": "An awesome repo",
                "stargazers_count": 5000,
                "forks_count": 1000,
                "language": "Python",
                "html_url": "https://github.com/awesome/repo",
                "topics": ["ai"],
            }
        ],
    }

    def test_requires_query(self):
        assert not github_search_repos("")["success"]

    def test_success(self):
        with patch("urllib.request.urlopen", return_value=make_response(self.SAMPLE)):
            r = github_search_repos("agent framework", language="python")
        assert r["success"]
        assert r["output"]["results"][0]["stars"] == 5000

    def test_language_appended_to_query(self):
        captured = []
        def fake_urlopen(req, timeout=20):
            captured.append(req.full_url)
            return make_response({"total_count": 0, "items": []})
        with patch("urllib.request.urlopen", side_effect=fake_urlopen):
            github_search_repos("test", language="rust")
        assert any("language" in u for u in captured)


# ── github_get_file ───────────────────────────────────────────────────────────

class TestGetFile:
    FILE_RESPONSE = {
        "type": "file",
        "path": "README.md",
        "name": "README.md",
        "size": 123,
        "sha": "abc123",
        "html_url": "https://github.com/o/r/blob/main/README.md",
        "encoding": "base64",
        "content": "SGVsbG8gV29ybGQ=\n",  # "Hello World"
    }
    DIR_RESPONSE = [
        {"name": "src", "type": "dir", "path": "src"},
        {"name": "README.md", "type": "file", "path": "README.md"},
    ]

    def test_requires_owner_repo_path(self):
        assert not github_get_file("", "r", "README.md")["success"]
        assert not github_get_file("o", "", "README.md")["success"]
        assert not github_get_file("o", "r", "")["success"]

    def test_decodes_base64(self):
        with patch("urllib.request.urlopen", return_value=make_response(self.FILE_RESPONSE)):
            r = github_get_file("o", "r", "README.md")
        assert r["success"]
        assert "Hello World" in r["output"]["content"]
        assert r["output"]["type"] == "file"

    def test_directory_response(self):
        with patch("urllib.request.urlopen", return_value=make_response(self.DIR_RESPONSE)):
            r = github_get_file("o", "r", ".")
        assert r["success"]
        assert r["output"]["type"] == "directory"
        assert len(r["output"]["items"]) == 2


# ── github_create_gist ────────────────────────────────────────────────────────

class TestCreateGist:
    RESPONSE = {
        "id": "abc123gist",
        "html_url": "https://gist.github.com/abc123gist",
        "description": "My gist",
        "public": False,
        "files": {"hello.py": {"filename": "hello.py"}},
    }

    def test_no_token_error(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        r = github_create_gist("test", {"hello.py": "print('hi')"})
        assert not r["success"]

    def test_empty_files_error(self):
        r = github_create_gist("test", {}, token="tok")
        assert not r["success"]

    def test_success(self):
        with patch("urllib.request.urlopen", return_value=make_response(self.RESPONSE, 201)):
            r = github_create_gist(
                "My gist",
                {"hello.py": "print('hello')"},
                token="tok",
            )
        assert r["success"]
        assert "html_url" in r["output"]
        assert r["output"]["id"] == "abc123gist"


# ── github_user_info ──────────────────────────────────────────────────────────

class TestUserInfo:
    SAMPLE = {
        "login": "octocat",
        "name": "The Octocat",
        "bio": "GitHub mascot",
        "company": "GitHub",
        "location": "San Francisco",
        "email": "octocat@github.com",
        "public_repos": 42,
        "followers": 10000,
        "following": 10,
        "html_url": "https://github.com/octocat",
        "created_at": "2011-01-25T18:44:36Z",
        "avatar_url": "https://avatars.githubusercontent.com/u/583231",
    }

    def test_success_with_username(self):
        with patch("urllib.request.urlopen", return_value=make_response(self.SAMPLE)):
            r = github_user_info("octocat")
        assert r["success"]
        assert r["output"]["login"] == "octocat"
        assert r["output"]["public_repos"] == 42

    def test_no_token_no_username_error(self, monkeypatch):
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        monkeypatch.delenv("GH_TOKEN", raising=False)
        r = github_user_info()
        assert not r["success"]

    def test_auth_user_with_token(self):
        with patch("urllib.request.urlopen", return_value=make_response(self.SAMPLE)):
            r = github_user_info(token="mytoken")
        assert r["success"]


# ── github_list_commits ───────────────────────────────────────────────────────

class TestListCommits:
    SAMPLE = [
        {
            "sha": "abc1234567890",
            "commit": {
                "message": "Fix bug\n\nLong description",
                "author": {"name": "Alice", "date": "2024-01-01T00:00:00Z"},
            },
            "html_url": "https://github.com/o/r/commit/abc1234",
        }
    ]

    def test_requires_owner_and_repo(self):
        assert not github_list_commits("", "r")["success"]
        assert not github_list_commits("o", "")["success"]

    def test_success(self):
        with patch("urllib.request.urlopen", return_value=make_response(self.SAMPLE)):
            r = github_list_commits("o", "r")
        assert r["success"]
        assert r["output"]["count"] == 1
        commit = r["output"]["commits"][0]
        assert commit["sha"] == "abc1234"   # truncated to 7 chars
        assert commit["message"] == "Fix bug"   # first line only
        assert commit["author"] == "Alice"

    def test_branch_filter(self):
        with patch("urllib.request.urlopen", return_value=make_response(self.SAMPLE)):
            r = github_list_commits("o", "r", branch="develop")
        assert r["success"]
        assert r["output"]["branch"] == "develop"
