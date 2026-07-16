"""主实验 protocol 不得读取未来 gold，并应控制 exact-query 重复。"""

import numpy as np

from experiments.common.factorized_workload import (
    GRADUAL,
    NATURAL,
    RECURRING,
    SHUFFLED,
    build_factorized_workload,
    resolve_workload,
    split_disjoint_source_pool,
)
from experiments.common.stream_protocol import (
    causal_prefix_init_kb,
    chronological_sample,
    embedding_content_fingerprint,
    query_identity,
    stream_sampling_diagnostics,
    support_reuse_diagnostics,
    query_drift_diagnostics,
    workload_factor_diagnostics,
    warmup_overlap_diagnostics,
)


def test_query_identity_ignores_empty_source_id():
    assert query_identity({"qidx": 7, "source_qidx": None}) == 7


def test_causal_prefix_init_ignores_gold_and_future_queries():
    documents = [
        {"doc_id": "d0", "title": "A", "text": ""},
        {"doc_id": "d1", "title": "B", "text": ""},
        {"doc_id": "d2", "title": "C", "text": ""},
    ]
    doc_embeddings = np.eye(3, dtype=np.float32)
    query_embeddings = np.eye(3, dtype=np.float32)
    prefix_a = [{"qidx": 0, "sf_titles": ["A"], "ctx_titles": ["A"]}]
    prefix_b = [{"qidx": 0, "sf_titles": ["C"], "ctx_titles": ["C"]}]

    init_a = causal_prefix_init_kb(
        documents, doc_embeddings, prefix_a, query_embeddings, budget=2,
        seed=11, candidate_topk=1)
    init_b = causal_prefix_init_kb(
        documents, doc_embeddings, prefix_b, query_embeddings, budget=2,
        seed=11, candidate_topk=1)

    assert init_a == init_b
    assert "d0" in init_a


def test_causal_prefix_fills_large_budget_deterministically():
    documents = [
        {"doc_id": f"d{index}", "title": str(index), "text": ""}
        for index in range(20)
    ]
    rng = np.random.default_rng(7)
    doc_embeddings = rng.normal(size=(20, 4))
    doc_embeddings /= np.linalg.norm(doc_embeddings, axis=1, keepdims=True)
    query_embeddings = doc_embeddings[:2]
    prefix = [{"qidx": 0}, {"qidx": 1}]

    first = causal_prefix_init_kb(
        documents, doc_embeddings, prefix, query_embeddings, budget=12,
        seed=9, candidate_topk=2)
    second = causal_prefix_init_kb(
        documents, doc_embeddings, prefix, query_embeddings, budget=12,
        seed=9, candidate_topk=2)

    assert first == second
    assert len(first) == 12


def test_support_reuse_diagnostics_are_posthoc_and_interpretable():
    stream = [
        {"qidx": 0, "sf_titles": ["a", "b"]},
        {"qidx": 1, "sf_titles": ["b", "c"]},
        {"qidx": 2, "sf_titles": ["a", "b"]},
        {"qidx": 3, "sf_titles": []},
    ]

    stats = support_reuse_diagnostics(stream, window_size=2).as_dict()

    assert stats["queries_with_support"] == 3
    assert stats["support_occurrences"] == 6
    assert stats["unique_supports"] == 3
    assert stats["repeated_support_occurrences"] == 3
    assert stats["repeated_support_rate"] == 0.5
    assert stats["queries_answerable_from_past"] == 1
    assert stats["past_answerable_query_rate"] == 0.333333
    assert stats["adjacent_window_jaccard_mean"] == 0.666667
    assert stats["max_support_frequency"] == 3


def test_embedding_fingerprint_tracks_inputs_but_not_gold_labels():
    documents = [{"title": "A", "text": "alpha"}]
    query_a = [{"question": "where", "sf_titles": ["A"]}]
    query_same_input = [{"question": "where", "sf_titles": ["B"]}]
    query_changed = [{"question": "when", "sf_titles": ["A"]}]

    first = embedding_content_fingerprint(documents, query_a)

    assert first == embedding_content_fingerprint(documents, query_same_input)
    assert first != embedding_content_fingerprint(documents, query_changed)
    assert first != embedding_content_fingerprint(
        [{"title": "A", "text": "changed"}], query_a)


def test_query_drift_diagnostics_separate_stable_and_shifted_streams():
    query_embeddings = np.asarray([
        [1.0, 0.0], [1.0, 0.0], [1.0, 0.0], [1.0, 0.0],
        [0.0, 1.0], [0.0, 1.0], [0.0, 1.0], [0.0, 1.0],
    ], dtype=np.float32)
    shifted = [
        {"qidx": index, "workload_regime": 0 if index < 4 else 1}
        for index in range(8)
    ]
    stable = [
        {"qidx": index, "workload_regime": 0}
        for index in range(4)
    ]

    shifted_stats = query_drift_diagnostics(
        shifted, query_embeddings, window_size=2).as_dict()
    stable_stats = query_drift_diagnostics(
        stable, query_embeddings, window_size=2).as_dict()

    assert shifted_stats["centroid_cosine_shift"] == 1.0
    assert shifted_stats["regime_js_divergence"] == 1.0
    assert stable_stats["centroid_cosine_shift"] == 0.0
    assert stable_stats["regime_js_divergence"] == 0.0


def test_warmup_overlap_diagnostics_detect_exact_query_leakage():
    warmup = [{"qidx": 0}, {"qidx": 1}, {"qidx": 1}]
    evaluation = [{"qidx": 1}, {"qidx": 2}]

    stats = warmup_overlap_diagnostics(
        warmup, evaluation, requested=4).as_dict()

    assert stats == {
        "requested": 4,
        "actual": 3,
        "unique": 2,
        "evaluation_overlap": 1,
        "overlap_rate": 0.333333,
    }


def test_window_span_reindexes_sparse_source_ids_for_embeddings():
    queries = [
        {"qidx": 100 + 7 * index, "event_ts": 1000 + index}
        for index in range(12)
    ]

    selected, _ = chronological_sample(
        queries,
        warmup_size=2,
        evaluation_size=4,
        mode="window_span",
        block_size=2,
    )

    assert [query["qidx"] for query in selected] == list(range(6))
    assert [query["source_qidx"] for query in selected[:2]] == [100, 107]
    assert selected[-1]["source_qidx"] == 177
    assert stream_sampling_diagnostics(selected).duplicates == 0
    # Sampling must not mutate loader-owned records.
    assert "source_qidx" not in queries[0]
    assert queries[0]["qidx"] == 100


def test_window_span_preserves_local_contiguity_and_global_time_coverage():
    queries = [
        {"qidx": index, "event_ts": 1000 + index}
        for index in range(30)
    ]

    selected, diagnostics = chronological_sample(
        queries,
        warmup_size=2,
        evaluation_size=12,
        mode="window_span",
        block_size=3,
    )
    evaluation_ids = [query["source_qidx"] for query in selected[2:]]
    blocks = [evaluation_ids[start:start + 3] for start in range(0, 12, 3)]

    assert all(
        block == list(range(block[0], block[0] + 3)) for block in blocks
    )
    assert blocks[0][0] == 2
    assert blocks[-1][-1] == 29
    assert diagnostics.block_size == 3
    assert diagnostics.selected_blocks == 4
    assert diagnostics.as_dict()["span_coverage"] > 0.9


def _factorized_fixture():
    documents = []
    queries = []
    title_to_idx = {}
    topic_words = ["alpha", "beta", "gamma", "delta"]
    for topic, word in enumerate(topic_words):
        for document_index in range(2):
            title = f"t{topic}-d{document_index}"
            title_to_idx[title] = len(documents)
            documents.append({
                "doc_id": title,
                "title": title,
                "text": f"{word} {word} evidence topic {topic}",
            })
            for question_index in range(2):
                queries.append({
                    "qidx": len(queries),
                    "question": f"question {topic} {document_index} {question_index}",
                    "sf_titles": [title],
                    "evidence_visibility": "direct",
                })
    return documents, queries, title_to_idx


def test_factorized_workload_uses_unique_queries_and_sparse_evidence_space():
    documents, queries, title_to_idx = _factorized_fixture()

    stream, warmup, construction = build_factorized_workload(
        queries,
        documents,
        title_to_idx,
        n_windows=4,
        window_size=4,
        workload=RECURRING,
        seed=7,
        latent_topics=4,
        n_regimes=2,
    )

    assert len(stream) == 16
    assert warmup == []
    assert len({query["qidx"] for query in stream}) == 16
    assert construction.exact_query_duplicates == 0
    assert construction.representation == "sparse-evidence-tfidf"
    assert all(
        query["constructor_space"] == "sparse-evidence-tfidf"
        for query in stream
    )
    window_supports = [
        {
            title
            for query in stream[start:start + 4]
            for title in query["sf_titles"]
        }
        for start in range(0, len(stream), 4)
    ]
    # Regime recurrence must mean evidence-working-set recurrence, while each
    # individual question remains unique.
    assert window_supports[0] == window_supports[2]
    assert window_supports[1] == window_supports[3]


def test_source_pool_split_is_stable_and_family_disjoint():
    documents, queries, title_to_idx = _factorized_fixture()
    calibration, calibration_stats = split_disjoint_source_pool(
        queries,
        documents,
        title_to_idx,
        role="calibration",
        seed=1729,
    )
    test, test_stats = split_disjoint_source_pool(
        queries,
        documents,
        title_to_idx,
        role="test",
        seed=1729,
    )

    assert calibration_stats.as_dict()["overlap_assertion"]
    assert calibration_stats.as_dict() == test_stats.as_dict() | {"role": "calibration"}
    assert {query["source_qidx"] for query in calibration}.isdisjoint(
        query["source_qidx"] for query in test
    )
    assert {
        title for query in calibration for title in query["sf_titles"]
    }.isdisjoint(
        title for query in test for title in query["sf_titles"]
    )
    assert {query["source_family_id"] for query in calibration}.isdisjoint(
        query["source_family_id"] for query in test
    )

    # Local loader order/qidx changes must not change the content-defined pool.
    reordered = []
    for qidx, query in enumerate(reversed(queries)):
        copied = dict(query)
        copied["qidx"] = qidx
        reordered.append(copied)
    reordered_calibration, _ = split_disjoint_source_pool(
        reordered,
        documents,
        title_to_idx,
        role="calibration",
        seed=1729,
    )
    assert {query["source_qidx"] for query in reordered_calibration} == {
        query["source_qidx"] for query in calibration
    }


def test_shuffled_control_preserves_regime_marginals():
    documents, queries, title_to_idx = _factorized_fixture()
    recurring, _, _ = build_factorized_workload(
        queries,
        documents,
        title_to_idx,
        n_windows=4,
        window_size=4,
        workload=RECURRING,
        seed=7,
        latent_topics=4,
        n_regimes=2,
    )
    documents, queries, title_to_idx = _factorized_fixture()
    shuffled, _, _ = build_factorized_workload(
        queries,
        documents,
        title_to_idx,
        n_windows=4,
        window_size=4,
        workload=SHUFFLED,
        seed=7,
        latent_topics=4,
        n_regimes=2,
    )

    recurring_states = [
        recurring[start]["workload_state"]
        for start in range(0, len(recurring), 4)
    ]
    shuffled_states = [
        shuffled[start]["workload_state"]
        for start in range(0, len(shuffled), 4)
    ]
    assert sorted(recurring_states) == sorted(shuffled_states)
    assert len({query["qidx"] for query in shuffled}) == len(shuffled)


def test_gradual_workload_moves_evidence_mass_monotonically():
    documents, queries, title_to_idx = _factorized_fixture()
    stream, _, construction = build_factorized_workload(
        queries,
        documents,
        title_to_idx,
        n_windows=4,
        window_size=4,
        workload=GRADUAL,
        seed=7,
        latent_topics=4,
        n_regimes=2,
    )

    regime_zero_counts = [
        sum(
            query["workload_regime"] == 0
            for query in stream[start:start + 4]
        )
        for start in range(0, len(stream), 4)
    ]
    assert regime_zero_counts == sorted(regime_zero_counts, reverse=True)
    assert regime_zero_counts[0] == 4
    assert regime_zero_counts[-1] == 0
    assert construction.exact_query_duplicates == 0


def test_reusable_multisupport_core_uses_anchor_families_without_query_cycles():
    documents = []
    title_to_idx = {}
    for title, text in (
        ("A", "alpha shared evidence"),
        ("B", "beta related evidence"),
        ("C", "gamma shared evidence"),
        ("D", "delta related evidence"),
    ):
        title_to_idx[title] = len(documents)
        documents.append({"doc_id": title, "title": title, "text": text})
    queries = []
    for pair in (("A", "B"), ("C", "D")):
        for variant in range(4):
            queries.append({
                "qidx": len(queries),
                "question": f"compare {pair[0]} {pair[1]} variant {variant}",
                "sf_titles": list(pair),
                "evidence_visibility": "direct",
            })

    stream, _, construction = build_factorized_workload(
        queries,
        documents,
        title_to_idx,
        n_windows=2,
        window_size=4,
        workload=RECURRING,
        seed=11,
        latent_topics=4,
        n_regimes=2,
        min_support_frequency=2,
        family_mode="anchor",
    )

    assert len(stream) == 8
    assert len({query["qidx"] for query in stream}) == 8
    assert construction.eligible_queries == 8
    assert construction.min_support_frequency == 2
    assert construction.family_mode == "anchor"
    assert construction.support_families == 2


def test_factorized_warmup_is_disjoint_and_uses_initial_regime():
    documents, queries, title_to_idx = _factorized_fixture()
    stream, warmup, construction = build_factorized_workload(
        queries,
        documents,
        title_to_idx,
        n_windows=2,
        window_size=4,
        warmup_size=4,
        workload=RECURRING,
        seed=7,
        latent_topics=4,
        n_regimes=2,
    )

    assert len(warmup) == 4
    assert construction.warmup_queries == 4
    assert {query["qidx"] for query in warmup}.isdisjoint(
        query["qidx"] for query in stream
    )
    assert {query["workload_state"] for query in warmup} == {
        stream[0]["workload_state"]
    }


def test_auto_workload_respects_dataset_protocol():
    natural_queries = [{"preserve_order": True}]
    controlled_queries = [{"preserve_order": False}]

    assert resolve_workload(natural_queries, "auto") == NATURAL
    assert resolve_workload(controlled_queries, "auto") == RECURRING


def test_workload_audit_does_not_call_one_shot_shift_predictable():
    stream = []
    for window, state in enumerate(["old", "old", "new", "new"]):
        for offset in range(2):
            stream.append({
                "qidx": 2 * window + offset,
                "workload_state": state,
                "sf_titles": [f"{state}-{offset}"],
                "evidence_visibility": "direct",
            })

    stats = workload_factor_diagnostics(stream, window_size=2).as_dict()

    assert stats["cross_regime_transitions"] == 1
    assert stats["causal_transition_accuracy"] == 0.0
    assert stats["recurrent_transition_rate"] == 0.0
    assert stats["labeled_visibility_rate"] == 1.0
