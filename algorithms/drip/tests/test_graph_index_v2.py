"""Focused smoke tests for relation-aware GraphIndex bridge scoring."""
import numpy as np

from algorithms.drip.cache_manager import DRIPCoreConfig, GraphIndex


def test_relation_path_beats_generic_shared_entity():
    cfg = DRIPCoreConfig(
        bridge_abs_threshold=0.01,
        bridge_relation_floor=0.01,
        bridge_relation_overlap_weight=0.5,
    )
    doc_pool = [
        {
            "doc_id": "a",
            "title": "Ada Lovelace",
            "text": "Ada Lovelace worked with Charles Babbage on the Analytical Engine.",
        },
        {
            "doc_id": "b",
            "title": "Charles Babbage",
            "text": "Charles Babbage designed the Analytical Engine.",
        },
        {
            "doc_id": "noise",
            "title": "Mathematics",
            "text": "Mathematics is a broad field with many researchers.",
        },
    ]
    d2p = {"a": 0, "b": 1, "noise": 2}
    graph = GraphIndex(cfg, d2p, doc_pool)
    graph.set_pool_entities({
        "a": ["Ada Lovelace", "Charles Babbage", "Mathematics"],
        "b": ["Charles Babbage"],
        "noise": ["Mathematics"],
    })
    doc_embs = np.eye(3, dtype=float)

    candidates = graph.graph_evidence(
        {"question": "Who worked with Ada Lovelace on the Analytical Engine?"},
        [(0, 0.9)],
        set(),
        doc_embs[:0],
        doc_embs,
    )

    assert candidates
    assert candidates[0][0] == 1
    assert graph.last_stats["bridge_after_threshold"] >= 1


if __name__ == "__main__":
    test_relation_path_beats_generic_shared_entity()
    print("GraphIndex v2 smoke test passed")
