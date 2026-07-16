"""Causal session workload builders for multi-turn retrieval benchmarks.

MTRAG exposes turn order inside each conversation, but it does not expose a
global timestamp across conversations.  This module therefore provides two
explicit protocols instead of pretending that upstream file order is a
natural event log:

``session_round_robin``
    A seed-controlled interleaving of conversations.  It preserves every
    conversation's turn order and does not create topic/domain blocks.

``controlled_recurring_domain``
    A synthetic stress test that cycles through domain-sized blocks while
    still preserving every conversation's turn order.  Its output is marked
    as controlled so it cannot be reported as an observed production trace.

Construction only reads event identity, ``conversation_id``, ``turn_idx`` and,
for the controlled protocol, ``domain``.  In particular it never reads gold
supports (``sf_titles``) or retrieval feedback.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
import random
from typing import Any


SESSION_ROUND_ROBIN = "session_round_robin"
CONTROLLED_RECURRING_DOMAIN = "controlled_recurring_domain"


@dataclass(frozen=True)
class SessionProtocolAudit:
    """Compact, serialisable audit of a constructed session stream."""

    protocol: str
    controlled: bool
    total_events: int
    warmup_events: int
    evaluation_events: int
    conversations: int
    domains: int
    order_violations: int
    exact_duplicates: int
    domain_transitions: int
    domain_blocks: int
    window_size: int | None = None

    @property
    def duplicate_rate(self) -> float:
        return self.exact_duplicates / max(1, self.total_events)

    @property
    def domain_transition_rate(self) -> float:
        return self.domain_transitions / max(1, self.total_events - 1)

    @property
    def windows(self) -> int | None:
        if self.window_size is None:
            return None
        return math.ceil(self.evaluation_events / self.window_size)

    def as_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "controlled": self.controlled,
            "total_events": self.total_events,
            "warmup_events": self.warmup_events,
            "evaluation_events": self.evaluation_events,
            "conversations": self.conversations,
            "domains": self.domains,
            "order_violations": self.order_violations,
            "exact_duplicates": self.exact_duplicates,
            "duplicate_rate": round(self.duplicate_rate, 6),
            "domain_transitions": self.domain_transitions,
            "domain_transition_rate": round(
                self.domain_transition_rate, 6
            ),
            "domain_blocks": self.domain_blocks,
            "window_size": self.window_size,
            "windows": self.windows,
        }


def _event_identity(event: Mapping[str, Any]) -> tuple[Any, ...]:
    """Return an exact event identity without looking at gold evidence."""

    query_id = event.get("query_id")
    if query_id is not None:
        return ("query_id", str(query_id))
    conversation_id = event.get("conversation_id")
    turn_idx = event.get("turn_idx")
    if conversation_id is not None and turn_idx is not None:
        return ("session_turn", str(conversation_id), int(turn_idx))
    source_qidx = event.get("source_qidx")
    if source_qidx is not None:
        return ("source_qidx", source_qidx)
    qidx = event.get("qidx")
    if qidx is not None:
        return ("qidx", qidx)
    raise ValueError(
        "each event needs query_id, conversation_id/turn_idx, or qidx"
    )


def _group_sessions(
    queries: Sequence[Mapping[str, Any]],
    *,
    require_domain: bool,
) -> dict[str, list[dict[str, Any]]]:
    """Validate and copy source events, then sort each session by turn."""

    sessions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_identities: set[tuple[Any, ...]] = set()
    for event in queries:
        if not isinstance(event, Mapping):
            raise TypeError("queries must contain mapping-like events")
        conversation_id = event.get("conversation_id")
        if conversation_id is None or str(conversation_id) == "":
            raise ValueError("each event needs a non-empty conversation_id")
        try:
            turn_idx = int(event["turn_idx"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("each event needs an integer turn_idx") from exc
        if turn_idx < 0:
            raise ValueError("turn_idx must be non-negative")
        if require_domain and not str(event.get("domain") or ""):
            raise ValueError(
                "controlled recurring-domain workload requires domain"
            )

        copied = dict(event)
        copied["conversation_id"] = str(conversation_id)
        copied["turn_idx"] = turn_idx
        identity = _event_identity(copied)
        if identity in seen_identities:
            raise ValueError(f"duplicate input event: {identity}")
        seen_identities.add(identity)
        sessions[str(conversation_id)].append(copied)

    if not sessions:
        raise ValueError("queries must not be empty")

    for conversation_id, events in sessions.items():
        events.sort(key=lambda event: (
            event["turn_idx"], str(event.get("query_id", ""))
        ))
        turns = [event["turn_idx"] for event in events]
        if len(turns) != len(set(turns)):
            raise ValueError(
                f"conversation {conversation_id!r} has duplicate turn_idx"
            )
        if require_domain:
            session_domains = {
                str(event["domain"]) for event in events
            }
            if len(session_domains) != 1:
                raise ValueError(
                    "controlled domain blocks require one domain per "
                    f"conversation; {conversation_id!r} has "
                    f"{sorted(session_domains)}"
                )
    return dict(sessions)


def _requested_total(
    available: int,
    warmup_size: int,
    evaluation_size: int | None,
) -> tuple[int, int, int]:
    warmup_size = int(warmup_size)
    if warmup_size < 0:
        raise ValueError("warmup_size must be non-negative")
    if evaluation_size is None:
        evaluation_size = available - warmup_size
    evaluation_size = int(evaluation_size)
    if evaluation_size < 0:
        raise ValueError("evaluation_size must be non-negative or None")
    requested = warmup_size + evaluation_size
    if requested > available:
        raise ValueError(
            f"requested {requested} events, but only {available} are available"
        )
    return warmup_size, evaluation_size, requested


def _mark_and_split(
    scheduled: Sequence[Mapping[str, Any]],
    *,
    warmup_size: int,
    protocol: str,
    controlled: bool,
    window_size: int | None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], SessionProtocolAudit]:
    marked: list[dict[str, Any]] = []
    for position, source_event in enumerate(scheduled):
        event = dict(source_event)
        # qidx remains the upstream embedding row.  stream_pos is deliberately
        # separate so interleaving cannot silently corrupt embedding lookup.
        event["session_stream_pos"] = position
        event["workload_protocol"] = protocol
        event["workload_controlled"] = controlled
        event["global_time_order"] = "unobserved"
        event["session_order_preserved"] = True
        if controlled:
            domain = str(event["domain"])
            event["workload_state"] = f"domain:{domain}"
            event["workload_regime"] = domain
            event["constructor_space"] = "observed-domain-label"
        marked.append(event)

    warmup = marked[:warmup_size]
    evaluation = marked[warmup_size:]
    audit = audit_session_workload(
        marked,
        protocol=protocol,
        controlled=controlled,
        warmup_events=len(warmup),
        evaluation_events=len(evaluation),
        window_size=window_size,
    )
    if audit.order_violations:
        raise AssertionError(
            f"internal scheduling error: {audit.order_violations} "
            "session-order violations"
        )
    if audit.exact_duplicates:
        raise AssertionError(
            f"internal scheduling error: {audit.exact_duplicates} duplicates"
        )
    return evaluation, warmup, audit


def build_session_round_robin(
    queries: Sequence[Mapping[str, Any]],
    seed: int = 42,
    warmup_size: int = 0,
    evaluation_size: int | None = None,
    *,
    window_size: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], SessionProtocolAudit]:
    """Interleave sessions without inventing global chronology or using gold.

    A seeded shuffle chooses the initial conversation order.  The scheduler
    then emits one turn from each active conversation per round.  The result is
    deterministic for a fixed seed and preserves strict increasing
    ``turn_idx`` order inside every conversation.

    Returns ``(evaluation, warmup, audit)``.  Source records are never mutated.
    """

    sessions = _group_sessions(queries, require_domain=False)
    warmup_size, _, requested = _requested_total(
        sum(map(len, sessions.values())), warmup_size, evaluation_size
    )
    rng = random.Random(int(seed))
    conversation_ids = sorted(sessions)
    rng.shuffle(conversation_ids)
    active = deque(
        (conversation_id, deque(sessions[conversation_id]))
        for conversation_id in conversation_ids
    )

    scheduled: list[dict[str, Any]] = []
    while active and len(scheduled) < requested:
        conversation_id, pending = active.popleft()
        scheduled.append(pending.popleft())
        if pending:
            active.append((conversation_id, pending))

    return _mark_and_split(
        scheduled,
        warmup_size=warmup_size,
        protocol=SESSION_ROUND_ROBIN,
        controlled=False,
        window_size=window_size,
    )


def build_recurring_domain_workload(
    queries: Sequence[Mapping[str, Any]],
    seed: int = 42,
    warmup_size: int = 0,
    evaluation_size: int | None = None,
    *,
    block_size: int = 25,
    domain_order: Sequence[str] | None = None,
    window_size: int | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], SessionProtocolAudit]:
    """Create controlled recurring domain blocks with causal session order.

    Domain labels are observable benchmark metadata, not gold evidence.  A
    fixed domain cycle recurs until the requested number of events is emitted.
    Inside a domain, conversations are themselves round-robin interleaved.
    Every output event is explicitly marked ``workload_controlled=True``.
    """

    block_size = int(block_size)
    if block_size < 1:
        raise ValueError("block_size must be positive")
    sessions = _group_sessions(queries, require_domain=True)
    warmup_size, _, requested = _requested_total(
        sum(map(len, sessions.values())), warmup_size, evaluation_size
    )
    rng = random.Random(int(seed))

    sessions_by_domain: dict[str, list[str]] = defaultdict(list)
    for conversation_id, events in sessions.items():
        sessions_by_domain[str(events[0]["domain"])].append(conversation_id)
    available_domains = sorted(sessions_by_domain)
    if domain_order is None:
        cycle = list(available_domains)
        rng.shuffle(cycle)
    else:
        cycle = [str(domain) for domain in domain_order]
        if len(cycle) != len(set(cycle)):
            raise ValueError("domain_order must not contain duplicates")
        if set(cycle) != set(available_domains):
            raise ValueError(
                "domain_order must contain every available domain exactly "
                f"once; expected {available_domains}, got {cycle}"
            )

    active_by_domain: dict[
        str, deque[tuple[str, deque[dict[str, Any]]]]
    ] = {}
    for domain in available_domains:
        conversation_ids = sorted(sessions_by_domain[domain])
        rng.shuffle(conversation_ids)
        active_by_domain[domain] = deque(
            (conversation_id, deque(sessions[conversation_id]))
            for conversation_id in conversation_ids
        )

    scheduled: list[dict[str, Any]] = []
    while len(scheduled) < requested:
        made_progress = False
        for domain in cycle:
            active = active_by_domain[domain]
            for _ in range(block_size):
                if not active or len(scheduled) >= requested:
                    break
                conversation_id, pending = active.popleft()
                scheduled.append(pending.popleft())
                made_progress = True
                if pending:
                    active.append((conversation_id, pending))
            if len(scheduled) >= requested:
                break
        if not made_progress:
            break
    if len(scheduled) != requested:
        raise AssertionError(
            f"scheduler emitted {len(scheduled)} of {requested} events"
        )

    return _mark_and_split(
        scheduled,
        warmup_size=warmup_size,
        protocol=CONTROLLED_RECURRING_DOMAIN,
        controlled=True,
        window_size=window_size,
    )


def audit_session_workload(
    stream: Sequence[Mapping[str, Any]],
    *,
    protocol: str | None = None,
    controlled: bool | None = None,
    warmup_events: int = 0,
    evaluation_events: int | None = None,
    window_size: int | None = None,
) -> SessionProtocolAudit:
    """Audit order, exact duplication, and adjacent domain transitions.

    This diagnostic reads no evidence labels.  Call post-hoc support/reuse
    diagnostics separately when the experiment report needs cacheability.
    """

    events = list(stream)
    if warmup_events < 0 or warmup_events > len(events):
        raise ValueError("warmup_events is outside the stream")
    if evaluation_events is None:
        evaluation_events = len(events) - warmup_events
    if evaluation_events < 0 or warmup_events + evaluation_events != len(events):
        raise ValueError("warmup/evaluation counts must partition the stream")
    if window_size is not None and int(window_size) < 1:
        raise ValueError("window_size must be positive or None")

    seen: set[tuple[Any, ...]] = set()
    duplicates = 0
    last_turn: dict[str, int] = {}
    order_violations = 0
    domains: list[str] = []
    conversations: set[str] = set()
    for event in events:
        identity = _event_identity(event)
        duplicates += int(identity in seen)
        seen.add(identity)

        conversation_id = str(event.get("conversation_id") or "")
        if conversation_id:
            conversations.add(conversation_id)
            turn_idx = int(event["turn_idx"])
            if (
                conversation_id in last_turn
                and turn_idx <= last_turn[conversation_id]
            ):
                order_violations += 1
            last_turn[conversation_id] = turn_idx
        domain = event.get("domain")
        if domain is not None and str(domain):
            domains.append(str(domain))

    transitions = sum(
        left != right for left, right in zip(domains, domains[1:])
    )
    domain_blocks = int(bool(domains)) + transitions
    if protocol is None:
        labels = Counter(
            str(event.get("workload_protocol") or "unknown")
            for event in events
        )
        protocol = labels.most_common(1)[0][0] if labels else "unknown"
    if controlled is None:
        controlled = bool(events) and all(
            bool(event.get("workload_controlled", False))
            for event in events
        )

    return SessionProtocolAudit(
        protocol=str(protocol),
        controlled=bool(controlled),
        total_events=len(events),
        warmup_events=int(warmup_events),
        evaluation_events=int(evaluation_events),
        conversations=len(conversations),
        domains=len(set(domains)),
        order_violations=order_violations,
        exact_duplicates=duplicates,
        domain_transitions=transitions,
        domain_blocks=domain_blocks,
        window_size=int(window_size) if window_size is not None else None,
    )
