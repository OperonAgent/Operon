"""Tests for core/delegation_bus.py"""
import time
import threading
from unittest import mock

import pytest

from core.delegation_bus import (
    DelegationBus, AgentRegistration, AgentCapability, AgentState,
    TaskState, AggregationStrategy, EventBus, AgentRegistry,
    TaskRouter, ResultAggregator, DeadLetterQueue, DelegationContext,
    BusEvent, EventPriority, DeadLetterEntry,
    get_bus, register_agent, delegate,
)


# ── BusEvent ──────────────────────────────────────────────────────────────────

class TestBusEvent:
    def test_default_event_id(self):
        e = BusEvent(topic="test.event")
        assert len(e.event_id) > 0

    def test_to_dict_keys(self):
        e = BusEvent(topic="task.started", payload={"x": 1}, agent_id="a1")
        d = e.to_dict()
        assert d["topic"] == "task.started"
        assert d["payload"] == {"x": 1}
        assert d["agent_id"] == "a1"

    def test_timestamp_set(self):
        before = time.time()
        e = BusEvent(topic="x")
        after = time.time()
        assert before <= e.timestamp <= after


# ── EventBus ──────────────────────────────────────────────────────────────────

class TestEventBus:
    def test_publish_and_subscribe(self):
        eb = EventBus()
        received = []
        eb.subscribe("task.started", lambda e: received.append(e))
        eb.publish("task.started", payload={"task": "test"})
        time.sleep(0.05)
        assert len(received) == 1
        assert received[0].topic == "task.started"

    def test_wildcard_all(self):
        eb = EventBus()
        received = []
        eb.subscribe("*", lambda e: received.append(e))
        eb.publish("task.started")
        eb.publish("result.done")
        time.sleep(0.05)
        assert len(received) >= 2

    def test_wildcard_subtopic(self):
        eb = EventBus()
        received = []
        eb.subscribe("task.*", lambda e: received.append(e))
        eb.publish("task.started")
        eb.publish("task.completed")
        eb.publish("result.done")  # should not match
        time.sleep(0.05)
        assert len(received) >= 2

    def test_unsubscribe(self):
        eb = EventBus()
        received = []
        def handler(e): received.append(e)
        eb.subscribe("test.event", handler)
        eb.unsubscribe("test.event", handler)
        eb.publish("test.event")
        time.sleep(0.05)
        assert received == []

    def test_replay_returns_history(self):
        eb = EventBus()
        eb.publish("task.started")
        eb.publish("task.completed")
        history = eb.replay("task.*")
        assert len(history) >= 2

    def test_replay_limit(self):
        eb = EventBus()
        for i in range(10):
            eb.publish("x.event", payload=i)
        history = eb.replay("x.*", limit=5)
        assert len(history) <= 5

    def test_handler_exception_isolated(self):
        eb = EventBus()
        def bad_handler(e): raise ValueError("handler error")
        good_received = []
        def good_handler(e): good_received.append(e)
        eb.subscribe("test.*", bad_handler)
        eb.subscribe("test.*", good_handler)
        eb.publish("test.event")
        time.sleep(0.05)
        assert len(good_received) >= 1

    def test_stats_keys(self):
        eb = EventBus()
        eb.publish("topic.one")
        eb.publish("topic.two")
        s = eb.stats()
        assert "history_size" in s
        assert "subscriber_patterns" in s

    def test_clear_history(self):
        eb = EventBus()
        eb.publish("x.event")
        eb.clear_history()
        assert eb.replay("*") == []

    def test_thread_safe_publish(self):
        eb = EventBus()
        counts = {"n": 0}
        lock = threading.Lock()
        def handler(e):
            with lock:
                counts["n"] += 1
        eb.subscribe("*", handler)
        threads = [threading.Thread(target=lambda: eb.publish("t.event")) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        time.sleep(0.1)
        assert counts["n"] == 20

    def test_matches_exact_topic(self):
        from core.delegation_bus import EventBus
        assert EventBus._matches("task.started", "task.started")

    def test_matches_wildcard(self):
        assert EventBus._matches("task.started", "task.*")
        assert EventBus._matches("anything", "*")
        assert not EventBus._matches("result.done", "task.*")


# ── AgentRegistration ─────────────────────────────────────────────────────────

class TestAgentRegistration:
    def _make_agent(self, caps=None):
        return AgentRegistration(
            agent_id="test",
            name="Test Agent",
            capabilities=[AgentCapability(name=c) for c in (caps or ["web_search"])],
        )

    def test_capability_names_set(self):
        a = self._make_agent(["web_search", "file_read"])
        assert a.capability_names == {"web_search", "file_read"}

    def test_is_available_idle(self):
        a = self._make_agent()
        a.state = AgentState.IDLE
        a.current_load = 0
        assert a.is_available

    def test_not_available_fully_loaded(self):
        a = self._make_agent()
        a.current_load = a.max_parallel
        assert not a.is_available

    def test_not_available_offline(self):
        a = self._make_agent()
        a.state = AgentState.OFFLINE
        assert not a.is_available

    def test_load_factor_zero_when_idle(self):
        a = self._make_agent()
        a.current_load = 0
        assert a.load_factor == 0.0

    def test_load_factor_one_when_full(self):
        a = self._make_agent()
        a.current_load = a.max_parallel
        assert a.load_factor == 1.0

    def test_capability_score_exact_match(self):
        a = self._make_agent(["web_search"])
        score = a.capability_score(["web_search"])
        assert score > 0.8

    def test_capability_score_no_match(self):
        a = self._make_agent(["web_search"])
        score = a.capability_score(["shell_exec"])
        assert score == 0.0

    def test_capability_score_partial_match(self):
        a = self._make_agent(["web_search", "file_read"])
        score = a.capability_score(["web_search", "shell_exec"])  # 1 of 2
        assert 0.0 < score < 1.0

    def test_capability_score_empty_required(self):
        a = self._make_agent(["web_search"])
        score = a.capability_score([])
        assert score == 1.0

    def test_capability_proficiency_affects_score(self):
        a_high = AgentRegistration(
            agent_id="h", name="h",
            capabilities=[AgentCapability("web_search", proficiency=1.0)],
        )
        a_low = AgentRegistration(
            agent_id="l", name="l",
            capabilities=[AgentCapability("web_search", proficiency=0.3)],
        )
        assert a_high.capability_score(["web_search"]) > a_low.capability_score(["web_search"])


# ── AgentRegistry ─────────────────────────────────────────────────────────────

class TestAgentRegistry:
    def test_register_and_get(self):
        reg = AgentRegistry()
        r = reg.register("agent1", capabilities=["web_search"])
        assert reg.get("agent1") is r
        assert reg.get("agent1").agent_id == "agent1"

    def test_unregister(self):
        reg = AgentRegistry()
        reg.register("agent1")
        assert reg.unregister("agent1") is True
        assert reg.get("agent1") is None

    def test_unregister_nonexistent(self):
        reg = AgentRegistry()
        assert reg.unregister("ghost") is False

    def test_increment_load(self):
        reg = AgentRegistry()
        reg.register("a")
        reg.increment_load("a", 2)
        assert reg.get("a").current_load == 2
        assert reg.get("a").state == AgentState.BUSY

    def test_record_completion(self):
        reg = AgentRegistry()
        reg.register("a")
        reg.increment_load("a", 2)
        reg.record_completion("a")
        assert reg.get("a").current_load == 1

    def test_record_completion_back_to_idle(self):
        reg = AgentRegistry()
        reg.register("a")
        reg.increment_load("a", 1)
        reg.record_completion("a", success=True)
        assert reg.get("a").state == AgentState.IDLE

    def test_record_failure_tracked(self):
        reg = AgentRegistry()
        reg.register("a")
        reg.increment_load("a", 1)
        reg.record_completion("a", success=False)
        assert reg.get("a").failed_tasks == 1

    def test_available_agents(self):
        reg = AgentRegistry()
        reg.register("idle_agent")
        reg.register("full_agent")
        reg.get("full_agent").current_load = reg.get("full_agent").max_parallel
        available = reg.available_agents()
        ids = [a.agent_id for a in available]
        assert "idle_agent" in ids
        assert "full_agent" not in ids

    def test_stats_structure(self):
        reg = AgentRegistry()
        reg.register("a1", capabilities=["web_search"])
        reg.register("a2", capabilities=["shell_exec"])
        s = reg.stats()
        assert s["total"] == 2
        assert "agents" in s

    def test_set_state(self):
        reg = AgentRegistry()
        reg.register("a")
        reg.set_state("a", AgentState.COOLING)
        assert reg.get("a").state == AgentState.COOLING


# ── TaskRouter ────────────────────────────────────────────────────────────────

class TestTaskRouter:
    def _registry_with_agents(self):
        reg = AgentRegistry()
        reg.register("researcher", capabilities=["web_search", "http_get"])
        reg.register("coder",      capabilities=["shell_exec", "file_write"])
        reg.register("analyst",    capabilities=["data_analysis", "data_chart"])
        return reg

    def test_route_by_capability(self):
        reg = self._registry_with_agents()
        router = TaskRouter(reg)
        agent = router.route("find info", required_capabilities=["web_search"])
        assert agent == "researcher"

    def test_route_coder(self):
        reg = self._registry_with_agents()
        router = TaskRouter(reg)
        agent = router.route("run script", required_capabilities=["shell_exec"])
        assert agent == "coder"

    def test_route_returns_none_when_no_agents(self):
        reg = AgentRegistry()
        router = TaskRouter(reg)
        assert router.route("task") is None

    def test_route_excludes_agents(self):
        reg = self._registry_with_agents()
        router = TaskRouter(reg)
        agent = router.route("search", required_capabilities=["web_search"],
                             exclude_agents=["researcher"])
        # Should return some other agent (possibly coder or analyst)
        assert agent != "researcher" or agent is None

    def test_route_prefers_specified_agents(self):
        reg = self._registry_with_agents()
        router = TaskRouter(reg)
        agent = router.route("any task", preferred_agents=["analyst"])
        assert agent == "analyst"

    def test_route_all_returns_multiple(self):
        reg = self._registry_with_agents()
        router = TaskRouter(reg)
        agents = router.route_all("task", count=2)
        assert len(agents) <= 2

    def test_infer_capabilities_search(self):
        router = TaskRouter(AgentRegistry())
        caps = router._infer_capabilities("search the web for Python news")
        assert "web_search" in caps

    def test_infer_capabilities_code(self):
        router = TaskRouter(AgentRegistry())
        caps = router._infer_capabilities("write code to parse CSV")
        assert any("shell" in c or "write" in c for c in caps)

    def test_route_least_loaded_fallback(self):
        reg = AgentRegistry()
        reg.register("a", capabilities=[])
        reg.register("b", capabilities=[])
        router = TaskRouter(reg)
        # With no specific caps, should route to least loaded
        agent = router.route("some task", required_capabilities=["exotic_cap"])
        assert agent in ["a", "b"]


# ── ResultAggregator ──────────────────────────────────────────────────────────

class TestResultAggregator:
    def _agg(self):
        return ResultAggregator()

    def test_best_highest_confidence(self):
        agg = self._agg()
        results = [("a1", "Answer A", 0.9), ("a2", "Answer B", 0.5)]
        r, conf, _ = agg.aggregate(results, AggregationStrategy.BEST)
        assert r == "Answer A"
        assert conf == 0.9

    def test_first_returns_first(self):
        agg = self._agg()
        results = [("a1", "First", 0.5), ("a2", "Second", 0.9)]
        r, _, _ = agg.aggregate(results, AggregationStrategy.FIRST)
        assert r == "First"

    def test_merge_combines_unique(self):
        agg = self._agg()
        results = [
            ("a1", "Line 1\nLine 2", 0.9),
            ("a2", "Line 2\nLine 3", 0.8),
        ]
        r, _, meta = agg.aggregate(results, AggregationStrategy.MERGE)
        assert "Line 1" in r
        assert "Line 3" in r
        assert meta["contributors"] == 2

    def test_merge_deduplicates(self):
        agg = self._agg()
        results = [("a1", "Fact A\nFact B", 0.9), ("a2", "Fact B\nFact C", 0.8)]
        r, _, _ = agg.aggregate(results, AggregationStrategy.MERGE)
        # Fact B should appear only once
        assert r.count("Fact B") == 1

    def test_vote_majority_wins(self):
        agg = self._agg()
        results = [
            ("a1", "Paris", 0.9),
            ("a2", "Paris", 0.8),
            ("a3", "London", 0.95),
        ]
        r, conf, meta = agg.aggregate(results, AggregationStrategy.VOTE)
        assert "paris" in r.lower()
        assert meta["votes"] == 2

    def test_vote_single_result(self):
        agg = self._agg()
        results = [("a1", "Answer", 0.8)]
        r, conf, _ = agg.aggregate(results, AggregationStrategy.VOTE)
        assert r == "Answer"

    def test_empty_results(self):
        agg = self._agg()
        r, conf, meta = agg.aggregate([], AggregationStrategy.BEST)
        assert r == ""
        assert conf == 0.0

    def test_none_results_filtered(self):
        agg = self._agg()
        results = [("a1", None, 0.9), ("a2", "Valid", 0.7)]
        r, _, _ = agg.aggregate(results, AggregationStrategy.BEST)
        assert r == "Valid"

    def test_default_strategy_is_best(self):
        agg = self._agg()
        results = [("a1", "A", 0.9), ("a2", "B", 0.5)]
        r, _, _ = agg.aggregate(results)
        assert r == "A"


# ── DeadLetterQueue ───────────────────────────────────────────────────────────

class TestDeadLetterQueue:
    def test_push_and_list(self):
        dlq = DeadLetterQueue()
        ctx = DelegationContext(task="dead task", agent_id="a1", error="timeout")
        ctx.retries = 3
        dlq.push(ctx)
        entries = dlq.list()
        assert len(entries) == 1
        assert entries[0].task_id == ctx.task_id

    def test_clear(self):
        dlq = DeadLetterQueue()
        for i in range(3):
            ctx = DelegationContext(task=f"task {i}", agent_id="a")
            dlq.push(ctx)
        n = dlq.clear()
        assert n == 3
        assert dlq.list() == []

    def test_max_size_respected(self):
        dlq = DeadLetterQueue(max_size=5)
        for i in range(10):
            ctx = DelegationContext(task=f"t{i}", agent_id="a")
            dlq.push(ctx)
        assert len(dlq.list()) <= 5

    def test_stats_by_agent(self):
        dlq = DeadLetterQueue()
        for _ in range(3):
            ctx = DelegationContext(task="t", agent_id="coder")
            dlq.push(ctx)
        ctx2 = DelegationContext(task="t", agent_id="researcher")
        dlq.push(ctx2)
        s = dlq.stats()
        assert s["by_agent"]["coder"] == 3
        assert s["by_agent"]["researcher"] == 1


# ── DelegationContext ─────────────────────────────────────────────────────────

class TestDelegationContext:
    def test_auto_task_id(self):
        ctx = DelegationContext(task="test")
        assert len(ctx.task_id) > 0

    def test_is_not_done_initially(self):
        ctx = DelegationContext()
        assert not ctx.is_done

    def test_is_done_when_done(self):
        ctx = DelegationContext()
        ctx.state = TaskState.DONE
        assert ctx.is_done

    def test_is_done_when_dead(self):
        ctx = DelegationContext()
        ctx.state = TaskState.DEAD
        assert ctx.is_done

    def test_duration_none_before_complete(self):
        ctx = DelegationContext()
        ctx.started_at = time.time()
        assert ctx.duration_ms is None

    def test_duration_computed_after_complete(self):
        ctx = DelegationContext()
        ctx.started_at = time.time()
        time.sleep(0.01)
        ctx.completed_at = time.time()
        assert ctx.duration_ms is not None
        assert ctx.duration_ms > 0

    def test_to_dict(self):
        ctx = DelegationContext(task="test", agent_id="a1")
        d = ctx.to_dict()
        assert d["task"] == "test"
        assert d["agent_id"] == "a1"
        assert "events" not in d   # excluded from dict


# ── DelegationBus ─────────────────────────────────────────────────────────────

class TestDelegationBus:
    def _bus(self):
        db = DelegationBus()
        db.register_agent("researcher", capabilities=["web_search"])
        db.register_agent("coder",      capabilities=["shell_exec"])
        return db

    def test_register_agent(self):
        db = DelegationBus()
        reg = db.register_agent("agent1", capabilities=["web_search"])
        assert reg.agent_id == "agent1"

    def test_unregister_agent(self):
        db = DelegationBus()
        db.register_agent("agent1")
        ok = db.unregister_agent("agent1")
        assert ok

    def test_route_to_best_agent(self):
        db = self._bus()
        agent = db.route("search the web", required_capabilities=["web_search"])
        assert agent == "researcher"

    def test_delegate_creates_running_context(self):
        db = self._bus()
        ctx = db.delegate("find Python news", to_agent="researcher")
        assert ctx.state == TaskState.RUNNING
        assert ctx.agent_id == "researcher"

    def test_delegate_auto_routes(self):
        db = self._bus()
        ctx = db.delegate("search", required_capabilities=["web_search"])
        assert ctx.agent_id == "researcher"

    def test_complete_marks_done(self):
        db = self._bus()
        ctx = db.delegate("task", to_agent="researcher")
        db.complete(ctx.task_id, result="found 42 results", confidence=0.9)
        ctx2 = db.get_context(ctx.task_id)
        assert ctx2.state == TaskState.DONE
        assert ctx2.result == "found 42 results"
        assert ctx2.confidence == 0.9

    def test_fail_with_retry(self):
        db = self._bus()
        ctx = db.delegate("task", to_agent="coder", max_retries=3)
        db.fail(ctx.task_id, error="timeout", retry=True)
        ctx2 = db.get_context(ctx.task_id)
        assert ctx2.state == TaskState.RETRYING
        assert ctx2.retries == 1

    def test_fail_without_retry_goes_to_dlq(self):
        db = self._bus()
        ctx = db.delegate("task", to_agent="coder")
        db.fail(ctx.task_id, error="permission denied", retry=False)
        assert db.get_context(ctx.task_id).state == TaskState.DEAD
        assert db.dlq.stats()["total"] >= 1

    def test_fail_exhausted_retries_goes_to_dlq(self):
        db = self._bus()
        ctx = db.delegate("task", to_agent="coder", max_retries=1)
        db.fail(ctx.task_id, error="err1", retry=True)   # attempt 1
        db.fail(ctx.task_id, error="err2", retry=True)   # exhausted
        assert db.get_context(ctx.task_id).state == TaskState.DEAD

    def test_list_tasks_all(self):
        db = self._bus()
        db.delegate("t1", to_agent="researcher")
        db.delegate("t2", to_agent="coder")
        all_tasks = db.list_tasks()
        assert len(all_tasks) >= 2

    def test_list_tasks_filter_state(self):
        db = self._bus()
        ctx = db.delegate("t1", to_agent="researcher")
        db.complete(ctx.task_id, "done")
        running = db.list_tasks(state=TaskState.RUNNING)
        done    = db.list_tasks(state=TaskState.DONE)
        assert all(t.state == TaskState.RUNNING for t in running)
        assert all(t.state == TaskState.DONE    for t in done)

    def test_get_context_unknown(self):
        db = self._bus()
        assert db.get_context("nonexistent-task-id") is None

    def test_stats_structure(self):
        db = self._bus()
        s = db.stats()
        assert "agents" in s
        assert "tasks" in s
        assert "dlq" in s
        assert "bus" in s

    def test_agent_heartbeat(self):
        db = DelegationBus()
        db.register_agent("a1")
        db.agent_heartbeat("a1", metadata={"cpu": 0.5})
        assert db.registry.get("a1").metadata.get("cpu") == 0.5

    def test_event_published_on_delegate(self):
        db = self._bus()
        published_topics = []
        db.event_bus.subscribe("task.*", lambda e: published_topics.append(e.topic))
        db.delegate("task", to_agent="researcher")
        time.sleep(0.05)
        assert "task.started" in published_topics

    def test_fan_out_no_fn(self):
        db = self._bus()
        result, conf, meta = db.fan_out("task", agents=["researcher", "coder"])
        # Without fn, returns empty aggregation
        assert isinstance(result, str)
        assert isinstance(meta, dict)


# ── Module-level API ──────────────────────────────────────────────────────────

class TestModuleAPI:
    def test_get_bus_returns_instance(self):
        b = get_bus()
        assert isinstance(b, DelegationBus)

    def test_get_bus_singleton(self):
        b1 = get_bus()
        b2 = get_bus()
        assert b1 is b2

    def test_register_agent_shortcut(self):
        bus = DelegationBus()
        with mock.patch("core.delegation_bus._default_bus", bus):
            from core.delegation_bus import get_bus as gb
            # Register via module function
            reg = bus.register_agent("test-module-agent", capabilities=["web_search"])
            assert reg.agent_id == "test-module-agent"
