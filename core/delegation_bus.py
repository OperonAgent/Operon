"""
Operon Delegation Bus — Production multi-agent coordination layer.

Matches Hermes team_runner.py + OpenClaw lobster engine depth.

Architecture
============
┌─────────────────────────────────────────────────────────────────────────┐
│  DelegationBus                                                          │
│  ┌──────────────────────┐  ┌───────────────────────────────────────┐   │
│  │   AgentRegistry      │  │  EventBus (pub/sub)                   │   │
│  │   (lifecycle mgmt)   │  │  Topics: task.*, result.*, error.*    │   │
│  └──────────────────────┘  └───────────────────────────────────────┘   │
│  ┌──────────────────────────────────────────────────────────────────┐   │
│  │  TaskRouter — match tasks to capable agents                       │   │
│  └──────────────────────────────────────────────────────────────────┘   │
│  ┌─────────────────────┐  ┌─────────────────────┐                      │
│  │ ResultAggregator    │  │ DeadLetterQueue      │                      │
│  │ (voting, merge)     │  │ (retry/poison queue) │                      │
│  └─────────────────────┘  └─────────────────────┘                      │
└─────────────────────────────────────────────────────────────────────────┘

Features
--------
• **EventBus** — pub/sub with topic wildcards (task.*, result.*).
  Thread-safe, supports async subscribers, replay last N events per topic.

• **AgentRegistry** — track agent capabilities, health, load.
  Supports capability-based routing: "I need an agent that can do X".

• **TaskRouter** — match pending tasks to available agents using
  capability scoring (overlap score + load factor).

• **DelegationContext** — per-task context tracking: who did what,
  tool calls made, outputs produced, timing.

• **ResultAggregator** — collect results from parallel agents:
  - Voting: pick the result that majority agrees on
  - Merge: concatenate and deduplicate text results
  - Best: pick result with highest confidence score

• **DeadLetterQueue** — track tasks that failed all retries.
  Supports manual re-queue, poison detection (max retries exceeded).

• **DelegationBus** — unified facade tying all components together.

Usage
-----
    from core.delegation_bus import DelegationBus, AgentCapability

    bus = DelegationBus()

    # Register agents
    bus.register_agent("researcher", capabilities=["web_search", "read"])
    bus.register_agent("coder",      capabilities=["shell_exec", "write", "git"])
    bus.register_agent("analyst",    capabilities=["data_analysis", "charts"])

    # Route a task
    agent = bus.route("I need to search the web for Python trends")
    # → "researcher"

    # Delegate with full tracking
    ctx = bus.delegate(
        task="Search for Python 3.13 new features",
        to_agent="researcher",
        tools=["web_search"],
    )
    bus.complete(ctx.task_id, result="Python 3.13 adds free-threading...")

    # Parallel fan-out
    results = bus.fan_out(
        task="Analyse this codebase",
        agents=["coder", "analyst"],
        aggregation="merge",
    )
"""

from __future__ import annotations

import fnmatch
import json
import logging
import re
import threading
import time
import uuid
from collections import defaultdict, deque
from concurrent.futures import Future, ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

log = logging.getLogger("operon.delegation_bus")

# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------

class AgentState(str, Enum):
    IDLE      = "idle"
    BUSY      = "busy"
    COOLING   = "cooling"
    OFFLINE   = "offline"
    ERROR     = "error"


class TaskState(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    DONE      = "done"
    FAILED    = "failed"
    RETRYING  = "retrying"
    DEAD      = "dead"   # exhausted all retries → DLQ


class AggregationStrategy(str, Enum):
    VOTE    = "vote"    # majority wins
    MERGE   = "merge"   # concatenate outputs
    BEST    = "best"    # highest confidence
    FIRST   = "first"   # first non-empty result


class EventPriority(int, Enum):
    LOW    = 0
    NORMAL = 1
    HIGH   = 2
    URGENT = 3


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class AgentCapability:
    """A capability an agent supports, with optional proficiency score."""
    name:        str
    proficiency: float = 1.0   # 0.0 → 1.0
    description: str   = ""


@dataclass
class AgentRegistration:
    """An agent registered in the bus."""
    agent_id:     str
    name:         str
    capabilities: List[AgentCapability]
    state:        AgentState = AgentState.IDLE
    max_parallel: int        = 3
    current_load: int        = 0
    total_tasks:  int        = 0
    failed_tasks: int        = 0
    registered_at: float     = field(default_factory=time.time)
    last_active:  float      = field(default_factory=time.time)
    metadata:     Dict[str, Any] = field(default_factory=dict)

    @property
    def capability_names(self) -> Set[str]:
        return {c.name for c in self.capabilities}

    @property
    def load_factor(self) -> float:
        """0.0 = idle, 1.0 = fully loaded."""
        return self.current_load / self.max_parallel if self.max_parallel > 0 else 1.0

    @property
    def is_available(self) -> bool:
        return self.state in (AgentState.IDLE, AgentState.BUSY) and \
               self.current_load < self.max_parallel

    def capability_score(self, required: List[str]) -> float:
        """
        Score how well this agent matches required capabilities.
        Returns overlap ratio adjusted by proficiency.
        """
        if not required:
            return 1.0
        my_caps = {c.name: c.proficiency for c in self.capabilities}
        total_prof = 0.0
        matched    = 0
        for req in required:
            # Exact match first, then glob
            if req in my_caps:
                total_prof += my_caps[req]
                matched += 1
            else:
                # Wildcard glob match (e.g. "file_*" matches "file_read")
                for cap_name, prof in my_caps.items():
                    if fnmatch.fnmatch(cap_name, req):
                        total_prof += prof
                        matched += 1
                        break
        if matched == 0:
            return 0.0
        # Score = (matched/required) * avg_proficiency * (1 - load_factor * 0.3)
        coverage   = matched / len(required)
        avg_prof   = total_prof / matched
        load_pen   = 1.0 - self.load_factor * 0.3
        return coverage * avg_prof * load_pen


@dataclass
class BusEvent:
    """An event emitted on the delegation bus."""
    event_id:  str    = field(default_factory=lambda: str(uuid.uuid4())[:8])
    topic:     str    = ""
    payload:   Any    = None
    agent_id:  str    = ""
    task_id:   str    = ""
    timestamp: float  = field(default_factory=time.time)
    priority:  int    = EventPriority.NORMAL

    def to_dict(self) -> Dict[str, Any]:
        return {
            "event_id": self.event_id,
            "topic":    self.topic,
            "payload":  self.payload,
            "agent_id": self.agent_id,
            "task_id":  self.task_id,
            "timestamp": self.timestamp,
            "priority": self.priority,
        }


@dataclass
class DelegationContext:
    """Per-task delegation context: tracks everything that happened."""
    task_id:     str   = field(default_factory=lambda: str(uuid.uuid4())[:12])
    task:        str   = ""
    agent_id:    str   = ""
    state:       TaskState = TaskState.PENDING
    created_at:  float = field(default_factory=time.time)
    started_at:  Optional[float] = None
    completed_at: Optional[float] = None
    tools_used:  List[str] = field(default_factory=list)
    result:      Optional[str] = None
    error:       Optional[str] = None
    retries:     int   = 0
    max_retries: int   = 3
    confidence:  float = 1.0
    metadata:    Dict[str, Any] = field(default_factory=dict)
    events:      List[BusEvent] = field(default_factory=list, repr=False)

    @property
    def duration_ms(self) -> Optional[float]:
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at) * 1000
        return None

    @property
    def is_done(self) -> bool:
        return self.state in (TaskState.DONE, TaskState.FAILED, TaskState.DEAD)

    def add_event(self, event: BusEvent) -> None:
        self.events.append(event)

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["state"] = self.state.value
        d.pop("events", None)
        return d


@dataclass
class DeadLetterEntry:
    """A task that exhausted all retries."""
    task_id:   str
    task:      str
    agent_id:  str
    error:     str
    retries:   int
    created_at: float
    died_at:   float = field(default_factory=time.time)
    requeued:  bool  = False


# ---------------------------------------------------------------------------
# EventBus
# ---------------------------------------------------------------------------

class EventBus:
    """
    Thread-safe pub/sub event bus with topic wildcards and event replay.

    Topics use dot notation: "task.started", "result.done", "error.timeout"
    Subscribers can use wildcards: "task.*", "*.done", "*"
    """

    def __init__(self, max_history: int = 500) -> None:
        self._subscribers: Dict[str, List[Callable[[BusEvent], None]]] = defaultdict(list)
        self._history: deque[BusEvent] = deque(maxlen=max_history)
        self._lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="bus")

    def publish(self, topic: str, payload: Any = None,
                agent_id: str = "", task_id: str = "",
                priority: int = EventPriority.NORMAL) -> BusEvent:
        """Publish an event. Notifies all matching subscribers asynchronously."""
        event = BusEvent(topic=topic, payload=payload, agent_id=agent_id,
                         task_id=task_id, priority=priority)
        with self._lock:
            self._history.append(event)
            subscribers = list(self._get_matching_subscribers(topic))

        for handler in subscribers:
            self._executor.submit(self._safe_call, handler, event)

        log.debug("bus.publish: %s [%s]", topic, event.event_id)
        return event

    def subscribe(self, topic_pattern: str, handler: Callable[[BusEvent], None]) -> None:
        """
        Subscribe to events matching topic_pattern.
        Supports: "task.started", "task.*", "*", "*.error"
        """
        with self._lock:
            self._subscribers[topic_pattern].append(handler)

    def unsubscribe(self, topic_pattern: str, handler: Callable[[BusEvent], None]) -> None:
        with self._lock:
            self._subscribers[topic_pattern] = [
                h for h in self._subscribers[topic_pattern] if h is not handler
            ]

    def replay(self, topic_pattern: str = "*", limit: int = 50) -> List[BusEvent]:
        """Return recent events matching topic_pattern (newest first)."""
        with self._lock:
            all_events = list(self._history)
        matching = [e for e in all_events if self._matches(e.topic, topic_pattern)]
        return list(reversed(matching))[:limit]

    def clear_history(self) -> None:
        with self._lock:
            self._history.clear()

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            topics = defaultdict(int)
            for e in self._history:
                topics[e.topic] += 1
            return {
                "history_size": len(self._history),
                "subscriber_patterns": list(self._subscribers.keys()),
                "topic_counts": dict(sorted(topics.items(), key=lambda x: -x[1])[:10]),
            }

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)

    def _get_matching_subscribers(self, topic: str) -> List[Callable]:
        handlers = []
        for pattern, subs in self._subscribers.items():
            if self._matches(topic, pattern):
                handlers.extend(subs)
        return handlers

    @staticmethod
    def _matches(topic: str, pattern: str) -> bool:
        """Check if topic matches pattern (wildcard: * matches any segment)."""
        if pattern == "*":
            return True
        # Convert dot-notation pattern to fnmatch glob
        return fnmatch.fnmatch(topic, pattern)

    @staticmethod
    def _safe_call(handler: Callable, event: BusEvent) -> None:
        try:
            handler(event)
        except Exception as exc:
            log.warning("bus handler error: %s", exc)


# ---------------------------------------------------------------------------
# AgentRegistry
# ---------------------------------------------------------------------------

class AgentRegistry:
    """Registry of all agents, their capabilities, and health status."""

    def __init__(self) -> None:
        self._agents: Dict[str, AgentRegistration] = {}
        self._lock   = threading.Lock()

    def register(
        self,
        agent_id: str,
        name: str = "",
        capabilities: Optional[List[str]] = None,
        cap_objects: Optional[List[AgentCapability]] = None,
        max_parallel: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentRegistration:
        """Register an agent. Returns the registration object."""
        caps: List[AgentCapability] = []
        if cap_objects:
            caps = cap_objects
        elif capabilities:
            caps = [AgentCapability(name=c) for c in capabilities]

        reg = AgentRegistration(
            agent_id=agent_id,
            name=name or agent_id,
            capabilities=caps,
            max_parallel=max_parallel,
            metadata=metadata or {},
        )
        with self._lock:
            self._agents[agent_id] = reg
        log.info("registered agent: %s (caps: %s)", agent_id, [c.name for c in caps])
        return reg

    def unregister(self, agent_id: str) -> bool:
        with self._lock:
            return bool(self._agents.pop(agent_id, None))

    def get(self, agent_id: str) -> Optional[AgentRegistration]:
        with self._lock:
            return self._agents.get(agent_id)

    def set_state(self, agent_id: str, state: AgentState) -> None:
        with self._lock:
            if agent_id in self._agents:
                self._agents[agent_id].state = state
                self._agents[agent_id].last_active = time.time()

    def increment_load(self, agent_id: str, delta: int = 1) -> None:
        with self._lock:
            if agent_id in self._agents:
                a = self._agents[agent_id]
                a.current_load = max(0, a.current_load + delta)
                if a.current_load > 0:
                    a.state = AgentState.BUSY
                else:
                    a.state = AgentState.IDLE

    def record_completion(self, agent_id: str, success: bool = True) -> None:
        with self._lock:
            if agent_id in self._agents:
                a = self._agents[agent_id]
                a.total_tasks += 1
                if not success:
                    a.failed_tasks += 1
                a.current_load = max(0, a.current_load - 1)
                a.state = AgentState.IDLE if a.current_load == 0 else AgentState.BUSY
                a.last_active = time.time()

    def available_agents(self) -> List[AgentRegistration]:
        with self._lock:
            return [a for a in self._agents.values() if a.is_available]

    def all_agents(self) -> List[AgentRegistration]:
        with self._lock:
            return list(self._agents.values())

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            agents = list(self._agents.values())
        return {
            "total":     len(agents),
            "available": sum(1 for a in agents if a.is_available),
            "agents": [
                {
                    "id":    a.agent_id,
                    "state": a.state.value,
                    "load":  a.current_load,
                    "caps":  [c.name for c in a.capabilities],
                    "tasks": a.total_tasks,
                }
                for a in agents
            ],
        }


# ---------------------------------------------------------------------------
# TaskRouter
# ---------------------------------------------------------------------------

class TaskRouter:
    """
    Routes tasks to the best-available agent based on capability matching.
    """

    def __init__(self, registry: AgentRegistry) -> None:
        self._registry = registry

    def route(
        self,
        task: str,
        required_capabilities: Optional[List[str]] = None,
        preferred_agents: Optional[List[str]] = None,
        exclude_agents: Optional[List[str]] = None,
    ) -> Optional[str]:
        """
        Find the best available agent for the task.
        Returns agent_id or None if no suitable agent is available.
        """
        available = self._registry.available_agents()
        if not available:
            return None

        if exclude_agents:
            available = [a for a in available if a.agent_id not in exclude_agents]

        # If preferred agents are specified, filter to those first
        if preferred_agents:
            preferred = [a for a in available if a.agent_id in preferred_agents]
            if preferred:
                available = preferred

        if not required_capabilities:
            required_capabilities = self._infer_capabilities(task)

        # Score each candidate
        scored = [
            (agent, agent.capability_score(required_capabilities))
            for agent in available
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        if not scored or scored[0][1] == 0.0:
            # No capable agent found — return least-loaded available
            least_loaded = min(available, key=lambda a: a.load_factor)
            return least_loaded.agent_id

        return scored[0][0].agent_id

    def route_all(
        self,
        task: str,
        required_capabilities: Optional[List[str]] = None,
        count: int = 3,
    ) -> List[str]:
        """Return top-N agent IDs sorted by capability score."""
        available = self._registry.available_agents()
        if not required_capabilities:
            required_capabilities = self._infer_capabilities(task)
        scored = sorted(
            [(a, a.capability_score(required_capabilities)) for a in available],
            key=lambda x: x[1], reverse=True,
        )
        return [a.agent_id for a, _ in scored[:count]]

    def _infer_capabilities(self, task: str) -> List[str]:
        """
        Infer likely required capabilities from task description using
        simple keyword matching.
        """
        caps: List[str] = []
        task_lower = task.lower()
        _KEYWORD_MAP = {
            "search": ["web_search"],
            "browse": ["browser_navigate"],
            "code":   ["shell_exec", "file_write"],
            "write":  ["file_write"],
            "read":   ["file_read"],
            "git":    ["git_status", "git_diff"],
            "data":   ["data_analysis"],
            "chart":  ["data_chart"],
            "sql":    ["db_query"],
            "email":  ["email_draft"],
            "slack":  ["slack_send"],
            "image":  ["image_gen"],
            "pdf":    ["pdf_create"],
        }
        for keyword, matched_caps in _KEYWORD_MAP.items():
            if keyword in task_lower:
                caps.extend(matched_caps)
        return list(set(caps))


# ---------------------------------------------------------------------------
# ResultAggregator
# ---------------------------------------------------------------------------

class ResultAggregator:
    """
    Collect and aggregate results from multiple agents working in parallel.
    """

    def aggregate(
        self,
        results: List[Tuple[str, Optional[str], float]],  # (agent_id, result, confidence)
        strategy: AggregationStrategy = AggregationStrategy.BEST,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Aggregate results using the specified strategy.
        Returns (aggregated_result, confidence, metadata).
        """
        # Filter out None/empty results
        valid = [(aid, r, c) for aid, r, c in results if r]
        if not valid:
            return "", 0.0, {"strategy": strategy.value, "contributors": 0}

        if strategy == AggregationStrategy.FIRST:
            return self._first(valid)
        elif strategy == AggregationStrategy.BEST:
            return self._best(valid)
        elif strategy == AggregationStrategy.VOTE:
            return self._vote(valid)
        elif strategy == AggregationStrategy.MERGE:
            return self._merge(valid)
        return self._best(valid)

    def _first(self, results: List[Tuple[str, str, float]]) -> Tuple[str, float, Dict]:
        aid, result, conf = results[0]
        return result, conf, {"strategy": "first", "agent": aid, "contributors": 1}

    def _best(self, results: List[Tuple[str, str, float]]) -> Tuple[str, float, Dict]:
        best = max(results, key=lambda x: x[2])
        aid, result, conf = best
        return result, conf, {
            "strategy": "best",
            "agent": aid,
            "contributors": len(results),
            "all_confidences": [(a, c) for a, _, c in results],
        }

    def _vote(self, results: List[Tuple[str, str, float]]) -> Tuple[str, float, Dict]:
        """Majority vote — results that start with the same prefix are grouped."""
        from collections import Counter
        # Normalise: lowercase, strip whitespace, first 100 chars as key
        normalised = [(aid, r, c, r.strip().lower()[:100]) for aid, r, c in results]
        counts: Counter = Counter(n for _, _, _, n in normalised)
        winner_norm = counts.most_common(1)[0][0]
        # Return the actual result from the highest-confidence agent with this normalised form
        candidates = [(aid, r, c) for aid, r, c, n in normalised if n == winner_norm]
        best_candidate = max(candidates, key=lambda x: x[2])
        vote_confidence = counts[winner_norm] / len(results)
        return best_candidate[1], vote_confidence, {
            "strategy": "vote",
            "votes":    counts[winner_norm],
            "total":    len(results),
            "contributors": len(results),
        }

    def _merge(self, results: List[Tuple[str, str, float]]) -> Tuple[str, float, Dict]:
        """Merge: concatenate unique non-overlapping results."""
        seen_lines: Set[str] = set()
        merged_parts: List[str] = []
        for _, result, _ in sorted(results, key=lambda x: -x[2]):
            for line in result.split("\n"):
                stripped = line.strip()
                if stripped and stripped not in seen_lines:
                    seen_lines.add(stripped)
                    merged_parts.append(line)
        merged = "\n".join(merged_parts)
        avg_conf = sum(c for _, _, c in results) / len(results)
        return merged, avg_conf, {
            "strategy": "merge",
            "contributors": len(results),
            "original_lengths": [len(r) for _, r, _ in results],
        }


# ---------------------------------------------------------------------------
# DeadLetterQueue
# ---------------------------------------------------------------------------

class DeadLetterQueue:
    """
    Holds tasks that have failed all retry attempts.
    Supports inspection, manual re-queue, and poison detection.
    """

    def __init__(self, max_size: int = 1000) -> None:
        self._queue: deque[DeadLetterEntry] = deque(maxlen=max_size)
        self._lock  = threading.Lock()

    def push(self, ctx: DelegationContext) -> DeadLetterEntry:
        entry = DeadLetterEntry(
            task_id   = ctx.task_id,
            task      = ctx.task,
            agent_id  = ctx.agent_id,
            error     = ctx.error or "unknown error",
            retries   = ctx.retries,
            created_at = ctx.created_at,
        )
        with self._lock:
            self._queue.append(entry)
        log.warning("DLQ: task %s dead after %d retries — %s",
                    ctx.task_id, ctx.retries, ctx.error)
        return entry

    def list(self) -> List[DeadLetterEntry]:
        with self._lock:
            return list(self._queue)

    def pop(self, task_id: str) -> Optional[DeadLetterEntry]:
        """Remove and return entry by task_id."""
        with self._lock:
            for i, entry in enumerate(self._queue):
                if entry.task_id == task_id:
                    entry.requeued = True
                    del list(self._queue)[i]
                    return entry
        return None

    def clear(self) -> int:
        with self._lock:
            n = len(self._queue)
            self._queue.clear()
            return n

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            entries = list(self._queue)
        by_agent: Dict[str, int] = defaultdict(int)
        for e in entries:
            by_agent[e.agent_id] += 1
        return {
            "total": len(entries),
            "by_agent": dict(by_agent),
            "recent": [
                {"task_id": e.task_id, "task": e.task[:60], "error": e.error[:100]}
                for e in entries[-5:]
            ],
        }


# ---------------------------------------------------------------------------
# DelegationBus — unified facade
# ---------------------------------------------------------------------------

class DelegationBus:
    """
    Unified delegation bus: manages agents, routes tasks, tracks results.
    """

    def __init__(self) -> None:
        self.event_bus  = EventBus()
        self.registry   = AgentRegistry()
        self.router     = TaskRouter(self.registry)
        self.aggregator = ResultAggregator()
        self.dlq        = DeadLetterQueue()
        self._tasks:    Dict[str, DelegationContext] = {}
        self._lock      = threading.Lock()
        self._executor  = ThreadPoolExecutor(max_workers=16, thread_name_prefix="deleg")

    # ── Agent management ──────────────────────────────────────────────────────

    def register_agent(
        self,
        agent_id: str,
        capabilities: Optional[List[str]] = None,
        name: str = "",
        max_parallel: int = 3,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> AgentRegistration:
        """Register an agent with the bus."""
        reg = self.registry.register(
            agent_id, name=name, capabilities=capabilities,
            max_parallel=max_parallel, metadata=metadata,
        )
        self.event_bus.publish(
            "agent.registered",
            payload={"agent_id": agent_id, "capabilities": capabilities or []},
            agent_id=agent_id,
        )
        return reg

    def unregister_agent(self, agent_id: str) -> bool:
        ok = self.registry.unregister(agent_id)
        if ok:
            self.event_bus.publish("agent.unregistered", agent_id=agent_id)
        return ok

    def agent_heartbeat(self, agent_id: str, metadata: Optional[Dict] = None) -> None:
        """Called by agents to report they're alive."""
        reg = self.registry.get(agent_id)
        if reg:
            reg.last_active = time.time()
            if metadata:
                reg.metadata.update(metadata)

    # ── Task delegation ───────────────────────────────────────────────────────

    def delegate(
        self,
        task: str,
        to_agent: Optional[str] = None,
        required_capabilities: Optional[List[str]] = None,
        tools: Optional[List[str]] = None,
        max_retries: int = 3,
        timeout_sec: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> DelegationContext:
        """
        Create and start a delegation context.
        If to_agent is None, routes to the best available agent.
        Returns the DelegationContext immediately (async execution).
        """
        # Route to best agent if not specified
        agent_id = to_agent
        if not agent_id:
            agent_id = self.router.route(task, required_capabilities)
        if not agent_id:
            agent_id = "unassigned"

        ctx = DelegationContext(
            task=task,
            agent_id=agent_id,
            max_retries=max_retries,
            metadata=metadata or {},
        )
        if tools:
            ctx.tools_used = tools

        with self._lock:
            self._tasks[ctx.task_id] = ctx

        # Start tracking
        ctx.state      = TaskState.RUNNING
        ctx.started_at = time.time()
        self.registry.increment_load(agent_id)

        evt = self.event_bus.publish(
            "task.started",
            payload={"task": task[:200], "agent": agent_id},
            agent_id=agent_id,
            task_id=ctx.task_id,
        )
        ctx.add_event(evt)
        log.info("delegated task %s to agent %s", ctx.task_id, agent_id)
        return ctx

    def complete(
        self,
        task_id: str,
        result: str,
        confidence: float = 1.0,
    ) -> Optional[DelegationContext]:
        """Mark a task as successfully completed."""
        with self._lock:
            ctx = self._tasks.get(task_id)
        if ctx is None:
            return None

        ctx.state        = TaskState.DONE
        ctx.result       = result
        ctx.confidence   = confidence
        ctx.completed_at = time.time()
        self.registry.record_completion(ctx.agent_id, success=True)

        evt = self.event_bus.publish(
            "task.completed",
            payload={"result_len": len(result), "confidence": confidence},
            agent_id=ctx.agent_id,
            task_id=task_id,
        )
        ctx.add_event(evt)
        return ctx

    def fail(
        self,
        task_id: str,
        error: str,
        retry: bool = True,
    ) -> Optional[DelegationContext]:
        """Mark a task as failed. If retry=True and retries remain, re-queues it."""
        with self._lock:
            ctx = self._tasks.get(task_id)
        if ctx is None:
            return None

        ctx.error = error
        self.registry.record_completion(ctx.agent_id, success=False)

        if retry and ctx.retries < ctx.max_retries:
            ctx.retries += 1
            ctx.state   = TaskState.RETRYING
            ctx.agent_id = self.router.route(ctx.task) or ctx.agent_id
            ctx.started_at = time.time()
            self.registry.increment_load(ctx.agent_id)
            evt = self.event_bus.publish(
                "task.retrying",
                payload={"attempt": ctx.retries, "error": error[:200]},
                agent_id=ctx.agent_id,
                task_id=task_id,
                priority=EventPriority.HIGH,
            )
            ctx.add_event(evt)
            log.warning("task %s retrying (attempt %d/%d): %s",
                        task_id, ctx.retries, ctx.max_retries, error[:100])
        else:
            ctx.state        = TaskState.DEAD
            ctx.completed_at = time.time()
            self.dlq.push(ctx)
            evt = self.event_bus.publish(
                "task.dead",
                payload={"error": error[:200], "retries": ctx.retries},
                agent_id=ctx.agent_id,
                task_id=task_id,
                priority=EventPriority.URGENT,
            )
            ctx.add_event(evt)

        return ctx

    def get_context(self, task_id: str) -> Optional[DelegationContext]:
        with self._lock:
            return self._tasks.get(task_id)

    # ── Fan-out: parallel multi-agent dispatch ────────────────────────────────

    def fan_out(
        self,
        task: str,
        agents: Optional[List[str]] = None,
        count: int = 3,
        aggregation: str = "best",
        timeout_sec: float = 30.0,
        fn: Optional[Callable[[str, str], Tuple[str, float]]] = None,
    ) -> Tuple[str, float, Dict[str, Any]]:
        """
        Dispatch the same task to multiple agents in parallel.
        `fn(agent_id, task)` → (result_str, confidence).

        If fn is None, returns a stub (useful in tests).
        Returns (aggregated_result, confidence, metadata).
        """
        strategy = AggregationStrategy(aggregation)

        if agents is None:
            agents = self.router.route_all(task, count=count)

        if not agents:
            return "", 0.0, {"error": "no agents available"}

        futures: Dict[Future, str] = {}
        contexts: List[DelegationContext] = []
        for agent_id in agents:
            ctx = self.delegate(task, to_agent=agent_id)
            contexts.append(ctx)
            if fn:
                f = self._executor.submit(fn, agent_id, task)
                futures[f] = ctx.task_id

        # Collect results
        results: List[Tuple[str, Optional[str], float]] = []
        if fn and futures:
            done, _ = wait(futures.keys(), timeout=timeout_sec)
            for f in done:
                task_id = futures[f]
                try:
                    result_str, conf = f.result()
                    self.complete(task_id, result_str, conf)
                    results.append((task_id, result_str, conf))
                except Exception as exc:
                    self.fail(task_id, str(exc), retry=False)
                    results.append((task_id, None, 0.0))
        else:
            # No fn provided — return stub
            for ctx in contexts:
                results.append((ctx.agent_id, None, 0.0))

        return self.aggregator.aggregate(results, strategy)

    # ── Route ─────────────────────────────────────────────────────────────────

    def route(
        self,
        task: str,
        required_capabilities: Optional[List[str]] = None,
    ) -> Optional[str]:
        """Return the best agent_id for the given task."""
        return self.router.route(task, required_capabilities)

    # ── Introspection ─────────────────────────────────────────────────────────

    def list_tasks(
        self,
        state: Optional[TaskState] = None,
        agent_id: Optional[str] = None,
    ) -> List[DelegationContext]:
        with self._lock:
            tasks = list(self._tasks.values())
        if state:
            tasks = [t for t in tasks if t.state == state]
        if agent_id:
            tasks = [t for t in tasks if t.agent_id == agent_id]
        return sorted(tasks, key=lambda t: t.created_at, reverse=True)

    def stats(self) -> Dict[str, Any]:
        with self._lock:
            tasks = list(self._tasks.values())
        state_counts: Dict[str, int] = defaultdict(int)
        for t in tasks:
            state_counts[t.state.value] += 1
        return {
            "agents":   self.registry.stats(),
            "tasks":    {"total": len(tasks), "by_state": dict(state_counts)},
            "dlq":      self.dlq.stats(),
            "bus":      self.event_bus.stats(),
        }

    def shutdown(self) -> None:
        self.event_bus.shutdown()
        self._executor.shutdown(wait=False)


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_default_bus: Optional[DelegationBus] = None
_bus_lock = threading.Lock()


def get_bus() -> DelegationBus:
    """Return (or create) the session-scoped default DelegationBus."""
    global _default_bus
    with _bus_lock:
        if _default_bus is None:
            _default_bus = DelegationBus()
    return _default_bus


def register_agent(
    agent_id: str,
    capabilities: Optional[List[str]] = None,
    name: str = "",
) -> AgentRegistration:
    """One-liner: register an agent with the default bus."""
    return get_bus().register_agent(agent_id, capabilities=capabilities, name=name)


def delegate(task: str, to_agent: Optional[str] = None) -> DelegationContext:
    """One-liner: delegate a task via the default bus."""
    return get_bus().delegate(task, to_agent=to_agent)
