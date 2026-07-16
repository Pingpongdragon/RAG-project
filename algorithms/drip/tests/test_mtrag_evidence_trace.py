"""Unit tests for the embedding-free MTRAG evidence trace benchmark."""

from experiments.agent.run_mtrag_trace import (
    CausalDomainStateCache,
    WindowValueCache,
    calibrate_capacity,
    evaluate_trace,
)


def _event(query_id, domain, supports):
    return {
        "query_id": query_id,
        "conversation_id": query_id.split("<::>")[0],
        "turn_idx": int(query_id.rsplit("<::>", 1)[1]),
        "domain": domain,
        "sf_titles": list(supports),
    }


def test_capacity_is_computed_only_from_supplied_calibration_prefix():
    calibration = [
        _event("a<::>1", "x", ["x::a"]),
        _event("b<::>1", "x", ["x::a"]),
        _event("c<::>1", "y", ["y::b"]),
        _event("d<::>1", "y", ["y::c"]),
    ]
    first = calibrate_capacity(
        calibration,
        window_size=2,
        occurrence_coverage=0.90,
        window_quantile=0.90,
    )
    # A caller can change an arbitrarily large holdout without passing it to
    # calibration; capacity remains a function of the prefix alone.
    unseen_holdout = [
        _event(f"z{i}<::>1", "z", [f"z::{j}"])
        for i in range(20)
        for j in range(10)
    ]
    del unseen_holdout
    second = calibrate_capacity(
        list(calibration),
        window_size=2,
        occurrence_coverage=0.90,
        window_quantile=0.90,
    )

    assert first == second
    assert first["required_documents_per_window"] == [1, 2]
    assert first["reference_capacity"] == 2
    assert first["recommended_sweep"] == [1, 2, 4]


def test_multi_support_metrics_are_measured_before_current_feedback():
    calibration = [_event("a<::>1", "x", ["x::a"])]
    evaluation = [
        _event("a<::>2", "x", ["x::a", "x::b"]),
        _event("a<::>3", "x", ["x::b"]),
    ]
    result = evaluate_trace(
        calibration,
        evaluation,
        capacity=2,
        window_size=1,
        policy_names=("LRU",),
    )["LRU"]

    # First query sees only warm x::a: any hit but not an all-support hit.
    # x::b becomes resident only after that metric, then hits on query two.
    assert result["strict_all_support_hit_rate"] == 0.5
    assert result["any_support_hit_rate"] == 1.0
    assert result["evidence_coverage"] == 0.666667
    assert result["cache_writes"] == 1


def test_domain_state_proposals_never_contain_unobserved_documents():
    cache = CausalDomainStateCache(
        capacity=4,
        candidate_budget=2,
        minimum_confidence=0.0,
        minimum_weight_margin=0.0,
    )
    history = [
        [_event("x<::>1", "x", ["x::a"])],
        [_event("y<::>1", "y", ["y::a"])],
        [_event("x<::>2", "x", ["x::a"])],
        [_event("y<::>2", "y", ["y::a"])],
    ]
    observed = set()
    for index, window in enumerate(history):
        cache.begin_window(index)
        for event in window:
            supports = tuple(event["sf_titles"])
            cache.observe_query(supports, event)
            observed.update(supports)
        cache.end_window(window)
        assert set(cache.pending_candidates) <= observed
        assert all(
            set(candidates) <= observed
            for candidates in (cache.pending_experts or {}).values()
        )


def test_nonpredictive_topic_gate_is_exact_lru_fallback():
    calibration = [
        _event("a<::>1", "x", ["x::a"]),
        _event("b<::>1", "y", ["y::b"]),
    ]
    evaluation = [
        _event("a<::>2", "x", ["x::c"]),
        _event("b<::>2", "y", ["y::d"]),
        _event("a<::>3", "x", ["x::a"]),
    ]
    result = evaluate_trace(
        calibration,
        evaluation,
        capacity=2,
        window_size=1,
        policy_names=("LRU", "CausalDomainState"),
        candidate_budget=1,
    )
    diagnostics = result["CausalDomainState"]["topic_state_diagnostics"]

    assert diagnostics["gate_windows"] == 0
    assert result["CausalDomainState"]["proactive_document_fetches"] == 0
    assert result["CausalDomainState"]["prefetch_cache_writes"] == 0
    assert result["CausalDomainState"]["total_document_fetches"] == result[
        "CausalDomainState"
    ]["reactive_cold_evidence_reads"]
    for metric in (
        "strict_all_support_hit_rate",
        "any_support_hit_rate",
        "evidence_coverage",
        "cache_writes",
    ):
        assert result["CausalDomainState"][metric] == result["LRU"][metric]


def test_window_value_cache_updates_only_after_completed_window():
    cache = WindowValueCache(
        capacity=1,
        write_budget=1,
        candidate_budget=1,
        initial_price=0.0,
    )
    cache.begin_window(0)
    event = _event("a<::>1", "x", ["x::a"])
    cache.observe_query(("x::a",), event)
    assert "x::a" not in cache.residents
    cache.end_window([event])
    assert "x::a" in cache.residents


def test_domain_adapter_uses_only_observable_query_candidates():
    reactive = WindowValueCache(
        capacity=2, write_budget=2, candidate_mass=0.0, initial_price=0.0
    )
    routed = WindowValueCache(
        capacity=2, write_budget=2, candidate_mass=1.0, initial_price=0.0
    )
    event = _event("a<::>1", "x", ["x::gold"])
    event["ctx_titles"] = ["x::gold", "x::candidate"]
    for cache in (reactive, routed):
        cache.begin_window(0)
        cache.observe_query(("x::gold",), event)
        cache.end_window([event])
    assert "x::candidate" not in reactive.residents
    assert "x::candidate" in routed.residents


def test_warmup_fill_does_not_poison_evaluation_switching_price():
    cache = WindowValueCache(
        capacity=2, write_budget=2, initial_price=0.25
    )
    cache.begin_window(0)
    events = [
        _event("a<::>1", "x", ["x::a"]),
        _event("b<::>1", "x", ["x::b"]),
    ]
    for event in events:
        cache.observe_query(tuple(event["sf_titles"]), event)
    cache.end_window(events)
    assert cache.price > cache.initial_price
    cache.reset_evaluation_diagnostics()
    assert cache.price == cache.initial_price
    assert cache.dual_age == 0


def test_candidate_gate_uses_delayed_next_window_feedback():
    cache = WindowValueCache(
        capacity=3,
        write_budget=3,
        candidate_mass=1.0,
        adaptive_candidate_gate=True,
        initial_price=0.0,
    )
    first = _event("a<::>1", "x", ["x::gold"])
    first["ctx_titles"] = ["x::candidate"]
    cache.begin_window(0)
    cache.observe_query(("x::gold",), first)
    cache.end_window([first])
    assert cache.candidate_reliability == 0.0
    assert "x::candidate" not in cache.residents

    second = _event("a<::>2", "x", ["x::candidate"])
    cache.begin_window(1)
    cache.observe_query(("x::candidate",), second)
    cache.end_window([second])
    assert cache.candidate_reliability > 0.0
