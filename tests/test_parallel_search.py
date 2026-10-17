"""Real registry, HTTP MCP transport and scripted agent-loop coverage (no live LLM)."""

import json
from unittest.mock import MagicMock

import pytest
import requests

from core import mcp
from core.iteration_budget import IterationBudget
from core.version import __version__
from tools import registry
from tools.parallel_search import _ENDPOINT


SOURCE = "https://www.python.org/downloads/release/python-3130/"
PAYLOAD = {"results": [{"url": SOURCE, "title": "Python 3.13.0",
                         "excerpts": ["Python 3.13 introduces a new interactive interpreter."]}]}
SEARCH = {"objective": "Find Python 3.13 highlights", "search_queries": ["Python 3.13 release"]}


def response(body, content_type="application/json", status=200):
    resp = requests.Response()
    resp.status_code = status
    resp.headers["Content-Type"] = content_type
    resp._content = body.encode("utf-8")
    resp._content_consumed = True
    return resp


@pytest.fixture
def wire(monkeypatch):
    calls = []
    monkeypatch.delenv("PARALLEL_API_KEY", raising=False)
    monkeypatch.delenv("PARALLEL_SEARCH_API_KEY", raising=False)

    def post(url, **kwargs):
        calls.append((url, kwargs))
        message = kwargs["json"]
        if "id" not in message:
            return response("", status=202)
        method = message["method"]
        if method == "initialize":
            result = {"protocolVersion": "2024-11-05", "capabilities": {}}
        elif method == "tools/list":
            result = {"tools": [{"name": "web_search"}, {"name": "web_fetch"}]}
        else:
            assert method == "tools/call"
            result = {"content": [{"type": "text", "text": json.dumps(PAYLOAD)}],
                      "structuredContent": PAYLOAD}
        return response(json.dumps({"jsonrpc": "2.0", "id": message["id"], "result": result}))

    monkeypatch.setattr(mcp._requests, "post", post)
    return calls


def test_registry_search_fetch_no_key_and_preserved_choices(wire, monkeypatch, tmp_path):
    # The optional tools never load, overwrite or auto-connect saved MCP servers.
    saved = tmp_path / "mcp_servers.json"
    saved.write_text('{"chosen": {"transport": "http", "url": "https://example.org/mcp"}}')
    before = saved.read_bytes()
    monkeypatch.setattr(mcp, "MCP_CONFIG_PATH", saved)
    monkeypatch.setenv("PARALLEL_API_KEY", "unrelated-configured-key")
    ddg = registry._DISPATCH["duckduckgo_search"]
    reg = registry.ToolRegistry()
    assert reg.execute("parallel_search", SEARCH) == {"success": True, "output": PAYLOAD, "error": ""}
    fetched = reg.execute("parallel_fetch", {"urls": [SOURCE], "objective": "Release highlights"})
    assert fetched["success"] and fetched["output"]["results"][0]["url"] == SOURCE
    assert saved.read_bytes() == before
    assert registry._DISPATCH["duckduckgo_search"] is ddg
    assert "parallel_search" in reg.get_descriptions()
    assert "parallel_fetch" in reg.get_compact_descriptions()
    assert [c[1]["json"]["method"] for c in wire] == [
        "initialize", "notifications/initialized", "tools/list", "tools/call"] * 2
    for url, kwargs in wire:
        assert url == _ENDPOINT
        assert kwargs["headers"]["User-Agent"] == f"Operon/{__version__} (+https://github.com/OperonAgent/Operon)"
        assert kwargs["headers"]["Accept"] == "application/json, text/event-stream"
        assert "Authorization" not in kwargs["headers"]
        assert kwargs["timeout"] == (30.0 if kwargs["json"]["method"] == "tools/call" else 15.0)
        prepared = requests.Request("POST", url, auth=kwargs["auth"],
                                    headers={"Authorization": "unrelated"}).prepare()
        assert "Authorization" not in prepared.headers
    arguments = [kw["json"]["params"]["arguments"] for _, kw in wire
                 if kw["json"]["method"] == "tools/call"]
    assert arguments[0]["search_queries"] == SEARCH["search_queries"]
    assert arguments[1]["urls"] == [SOURCE]
    assert arguments[0]["session_id"] == arguments[1]["session_id"]
    assert len(arguments[0]["session_id"]) == 32


def test_web_toolset_disable_enable_and_read_only_budget(wire):
    reg = registry.ToolRegistry()
    try:
        registry.disable_toolset("web")
        for name in ("parallel_search", "parallel_fetch"):
            assert not reg.execute(name, SEARCH)["success"]
        assert not wire
    finally:
        registry.enable_toolset("web")
    assert reg.execute("parallel_search", SEARCH)["success"]
    budget = IterationBudget(1)
    assert budget.consume()
    assert budget.refund_if_read_only("parallel_search")
    assert budget.refund_if_read_only("parallel_fetch")


@pytest.mark.parametrize("name,params", [
    ("parallel_search", {"objective": "", "search_queries": ["query"]}),
    ("parallel_search", {"objective": "find", "search_queries": "query"}),
    ("parallel_search", {"objective": "find", "search_queries": []}),
    ("parallel_fetch", {"urls": []}),
    ("parallel_fetch", {"urls": ["file:///etc/passwd"]}),
    ("parallel_fetch", {"urls": ["https://user:secret@example.org"]}),
    ("parallel_fetch", {"urls": [SOURCE] * 21}),
    ("parallel_fetch", {"urls": [SOURCE], "objective": "x" * 201}),
    ("parallel_fetch", {"urls": [SOURCE], "full_content": "true"}),
])
def test_invalid_input_never_requests(wire, name, params):
    assert not registry.ToolRegistry().execute(name, params)["success"]
    assert not wire


@pytest.mark.parametrize("failure", ["timeout", "http", "rpc", "tool", "malformed", "id", "discovery"])
def test_failure_returns_error_without_retry_or_provider_fallback(wire, monkeypatch, failure):
    good_post = mcp._requests.post

    def post(url, **kwargs):
        method = kwargs["json"]["method"]
        target = "tools/list" if failure == "discovery" else "tools/call"
        if method != target:
            return good_post(url, **kwargs)
        wire.append((url, kwargs))
        if failure == "timeout":
            raise requests.Timeout("request timed out")
        if failure == "http":
            return response("limited", status=429)
        if failure == "malformed":
            return response("not json")
        reply = {"jsonrpc": "2.0", "id": kwargs["json"]["id"]}
        if failure == "rpc":
            reply["error"] = {"code": -32602, "message": "invalid arguments"}
        else:
            reply["result"] = {"isError": True, "content": [{"type": "text", "text": "service error"}]}
        if failure == "id":
            reply["id"] = "wrong"
        return response(json.dumps(reply))

    monkeypatch.setattr(mcp._requests, "post", post)
    result = registry.ToolRegistry().execute("parallel_search", SEARCH)
    assert result["success"] is False and result["error"]
    assert len(wire) == (3 if failure == "discovery" else 4)
    assert all(url == _ENDPOINT for url, _ in wire)


def test_streamable_sse_utf8_and_text_only_result(wire, monkeypatch):
    good_post = mcp._requests.post

    def post(url, **kwargs):
        if kwargs["json"]["method"] != "tools/call":
            return good_post(url, **kwargs)
        wire.append((url, kwargs))
        result = {"content": [{"type": "text", "text": "Résumé " + SOURCE}]}
        event = {"jsonrpc": "2.0", "id": kwargs["json"]["id"], "result": result}
        return response(': ping\n\nevent: message\ndata: ' + json.dumps(event, ensure_ascii=False) + '\n\n',
                        "text/event-stream")

    monkeypatch.setattr(mcp._requests, "post", post)
    result = registry.ToolRegistry().execute("parallel_fetch", {"urls": [SOURCE], "full_content": True})
    assert result["success"] and result["output"] == "Résumé " + SOURCE
    assert wire[-1][1]["json"]["params"]["arguments"]["full_content"] is True


@pytest.mark.parametrize("stage", ["initialize", "tools/list", "tools/call"])
def test_keyboard_interrupt_propagates_without_retry(wire, monkeypatch, stage):
    good_post = mcp._requests.post
    def interrupted(url, **kwargs):
        if kwargs["json"]["method"] == stage:
            raise KeyboardInterrupt()
        return good_post(url, **kwargs)
    monkeypatch.setattr(mcp._requests, "post", interrupted)
    with pytest.raises(KeyboardInterrupt):
        registry.ToolRegistry().execute("parallel_search", SEARCH)


def test_explicit_duckduckgo_selection_does_not_use_parallel(wire, monkeypatch):
    selected = MagicMock(return_value={"success": True, "results": [], "error": ""})
    monkeypatch.setitem(registry._DISPATCH, "duckduckgo_search", selected)
    assert registry.ToolRegistry().execute("duckduckgo_search", {"query": "Python"})["success"]
    selected.assert_called_once_with(query="Python")
    assert not wire


def run_scripted_loop(monkeypatch, approval=False, denied=False):
    import main
    from core.session import SessionManager

    session = SessionManager()
    session.add_message("user", "Use Parallel to find Python 3.13 highlights and read the release page.")
    config = MagicMock()
    values = {"max_tool_iters": 4, "streaming_enabled": False, "memory_enabled": False}
    config.get.side_effect = lambda key, default=None: values.get(key, default)
    config.resolve_model.return_value = {"provider": "openai"}
    router = MagicMock()
    actions = [
        {"action": {"type": "tool", "tool_name": "parallel_search", "params": SEARCH}},
        {"action": {"type": "tool", "tool_name": "parallel_fetch", "params": {"urls": [SOURCE]}}},
        {"action": {"type": "response", "content": "Python 3.13 introduces a new interactive interpreter. Source: " + SOURCE}},
    ]
    router.complete.side_effect = [json.dumps(a) for a in actions]
    # Use the production parser; only model inference and unrelated background work are stubbed.
    router.parse_response.side_effect = main.ModelRouter.parse_response
    monkeypatch.setattr(main, "ThinkingSpinner", MagicMock())
    monkeypatch.setattr(main, "_APPROVAL_MODE", approval)
    monkeypatch.setattr(main, "_VECTOR_MEMORY_AVAILABLE", False)
    monkeypatch.setattr(main, "_OBSIDIAN_AVAILABLE", False)
    monkeypatch.setattr(main, "_SKILL_SYNTH_AVAILABLE", False)
    monkeypatch.setattr(main, "_retry_mgr", None)
    monkeypatch.setattr(main, "log_tool_call", lambda *args: None)
    monkeypatch.setattr("builtins.input", lambda: "n" if denied else "y")
    monkeypatch.setattr("core.background_review.BackgroundReviewer.maybe_spawn", lambda *args, **kwargs: None)
    main.run_agent_loop(session, router, MagicMock(), registry.ToolRegistry(),
                        MagicMock(), config, MagicMock())
    return session, router


def test_scripted_agent_loop_consumes_search_and_fetch(wire, monkeypatch):
    session, router = run_scripted_loop(monkeypatch)
    transcript = session.get_messages_for_api()
    for name in ("parallel_search", "parallel_fetch"):
        assert any(f"[TOOL_RESULT: {name}]" in m["content"] and SOURCE in m["content"]
                   and "interactive interpreter" in m["content"] for m in transcript)
    assert any(SOURCE in m["content"] for m in router.complete.call_args.kwargs["messages"])
    assert len(wire) == 8


def test_scripted_agent_loop_honors_denied_approval(wire, monkeypatch):
    session, _ = run_scripted_loop(monkeypatch, approval=True, denied=True)
    assert not wire
    assert any("[TOOL_BLOCKED: parallel_search]" in m["content"]
               for m in session.get_messages_for_api())
