"""Tests for causal MTRAG session workload construction."""

from experiments.common.session_workload import (
    CONTROLLED_RECURRING_DOMAIN,
    SESSION_ROUND_ROBIN,
    audit_session_workload,
    build_recurring_domain_workload,
    build_session_round_robin,
)


def _fixture():
    queries = []
    qidx = 0
    for domain in ("alpha", "beta", "gamma"):
        for session in range(2):
            for turn in (1, 3, 5, 7):
                queries.append({
                    "qidx": qidx,
                    "query_id": f"{domain}-{session}<::>{turn}",
                    "conversation_id": f"{domain}-{session}",
                    "turn_idx": turn,
                    "domain": domain,
                    "question": f"question {qidx}",
                    "sf_titles": [f"gold-{domain}-{turn}"],
                })
                qidx += 1
    # Builders must recover session order instead of trusting source order.
    return list(reversed(queries))


def _assert_session_order(stream):
    turns = {}
    for event in stream:
        previous = turns.get(event["conversation_id"], -1)
        assert event["turn_idx"] > previous
        turns[event["conversation_id"]] = event["turn_idx"]


def test_session_round_robin_is_causal_disjoint_and_deterministic():
    queries = _fixture()
    source_snapshot = [dict(query) for query in queries]

    evaluation, warmup, audit = build_session_round_robin(
        queries,
        seed=9,
        warmup_size=5,
        evaluation_size=15,
        window_size=5,
    )
    repeated, repeated_warmup, _ = build_session_round_robin(
        queries,
        seed=9,
        warmup_size=5,
        evaluation_size=15,
        window_size=5,
    )

    combined = warmup + evaluation
    assert [row["query_id"] for row in repeated_warmup + repeated] == [
        row["query_id"] for row in combined
    ]
    assert len(warmup) == 5
    assert len(evaluation) == 15
    assert {row["query_id"] for row in warmup}.isdisjoint(
        row["query_id"] for row in evaluation
    )
    assert audit.protocol == SESSION_ROUND_ROBIN
    assert audit.controlled is False
    assert audit.order_violations == 0
    assert audit.exact_duplicates == 0
    assert audit.windows == 3
    assert all(row["qidx"] in range(24) for row in combined)
    assert [row["session_stream_pos"] for row in combined] == list(range(20))
    _assert_session_order(combined)
    assert queries == source_snapshot


def test_controlled_domain_blocks_recur_without_breaking_sessions():
    evaluation, warmup, audit = build_recurring_domain_workload(
        _fixture(),
        seed=3,
        warmup_size=0,
        evaluation_size=24,
        block_size=4,
        domain_order=("alpha", "beta", "gamma"),
        window_size=4,
    )

    assert warmup == []
    assert audit.protocol == CONTROLLED_RECURRING_DOMAIN
    assert audit.controlled is True
    assert audit.domain_transitions == 5
    assert audit.domain_blocks == 6
    assert audit.order_violations == 0
    assert audit.exact_duplicates == 0
    assert [row["domain"] for row in evaluation[::4]] == [
        "alpha", "beta", "gamma", "alpha", "beta", "gamma"
    ]
    assert all(row["workload_controlled"] for row in evaluation)
    assert all(
        row["workload_state"] == f"domain:{row['domain']}"
        for row in evaluation
    )
    _assert_session_order(evaluation)


def test_scheduling_does_not_depend_on_gold_supports():
    original = _fixture()
    relabeled = [
        {**query, "sf_titles": ["entirely-different-gold"]}
        for query in original
    ]

    first, first_warmup, _ = build_recurring_domain_workload(
        original, seed=19, warmup_size=3, evaluation_size=18, block_size=3
    )
    second, second_warmup, _ = build_recurring_domain_workload(
        relabeled, seed=19, warmup_size=3, evaluation_size=18, block_size=3
    )

    assert [row["query_id"] for row in first_warmup + first] == [
        row["query_id"] for row in second_warmup + second
    ]


def test_protocol_audit_detects_duplicate_and_order_violation():
    stream = [
        {
            "query_id": "a<::>2",
            "conversation_id": "a",
            "turn_idx": 2,
            "domain": "x",
        },
        {
            "query_id": "b<::>1",
            "conversation_id": "b",
            "turn_idx": 1,
            "domain": "y",
        },
        {
            "query_id": "a<::>1",
            "conversation_id": "a",
            "turn_idx": 1,
            "domain": "x",
        },
        {
            "query_id": "a<::>1",
            "conversation_id": "a",
            "turn_idx": 1,
            "domain": "x",
        },
    ]

    audit = audit_session_workload(stream)

    assert audit.order_violations == 2
    assert audit.exact_duplicates == 1
    assert audit.domain_transitions == 2


def test_controlled_domains_reject_cross_domain_conversation():
    queries = [
        {
            "query_id": "a<::>1",
            "conversation_id": "a",
            "turn_idx": 1,
            "domain": "x",
        },
        {
            "query_id": "a<::>2",
            "conversation_id": "a",
            "turn_idx": 2,
            "domain": "y",
        },
    ]

    try:
        build_recurring_domain_workload(queries)
    except ValueError as exc:
        assert "one domain per conversation" in str(exc)
    else:
        raise AssertionError("cross-domain conversation should be rejected")
