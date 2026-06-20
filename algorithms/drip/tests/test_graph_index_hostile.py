"""Hostile/adversarial probes for relation-aware GraphIndex bridge scoring.

These mirror the failure modes listed in
``docs/design/GRAPH_INDEX_V2_EXECUTION_PLAN.md`` section 10:

  - a very high-degree hub entity must not out-rank the true bridge,
  - a weak/incorrect first hop must not fan out bridge candidates,
  - candidates below the absolute threshold must produce no evidence
    (no per-query max normalisation that amplifies noise),
  - the first-hop similarity gate must short-circuit before path expansion.

Run directly (no pytest needed):

    PYTHONPATH=. python algorithms/drip/tests/test_graph_index_hostile.py
"""
import numpy as np

from algorithms.drip.cache_manager import DRIPCoreConfig, GraphIndex


def _build(cfg, pool, ents):
    d2p = {d["doc_id"]: i for i, d in enumerate(pool)}
    graph = GraphIndex(cfg, d2p, pool)
    graph.set_pool_entities(ents)
    return graph, d2p


def test_true_bridge_beats_high_degree_hub():
    """B (sharing a rare entity) must out-rank 50 docs sharing a hub entity."""
    cfg = DRIPCoreConfig(
        bridge_abs_threshold=0.01,
        bridge_relation_floor=0.01,
        bridge_relation_overlap_weight=0.5,
    )
    pool = [
        {"doc_id": "A", "title": "Ada Lovelace",
         "text": "Ada Lovelace worked with Charles Babbage on the Analytical Engine in England."},
        {"doc_id": "B", "title": "Charles Babbage",
         "text": "Charles Babbage designed the Analytical Engine in England."},
    ]
    ents = {"A": ["Ada Lovelace", "Charles Babbage", "England"],
            "B": ["Charles Babbage", "England"]}
    for i in range(50):
        pool.append({"doc_id": f"N{i}", "title": f"Town {i}",
                     "text": "A town located in England."})
        ents[f"N{i}"] = ["England"]
    graph, _ = _build(cfg, pool, ents)
    embs = np.eye(len(pool), dtype=float)

    cands = graph.graph_evidence(
        {"question": "Who worked with Ada Lovelace on the Analytical Engine?"},
        [(0, 0.9)], set(), embs[:0], embs)

    assert cands, "expected at least one bridge candidate"
    assert cands[0][0] == 1, f"true bridge B should rank first, got {cands[:3]}"


def test_firsthop_sim_gate_blocks_weak_first_hop():
    """A first hop below bridge_min_firsthop_sim must spawn no candidates."""
    cfg = DRIPCoreConfig(
        bridge_abs_threshold=0.01,
        bridge_relation_floor=0.01,
        bridge_relation_overlap_weight=0.5,
        bridge_min_firsthop_sim=0.5,
    )
    pool = [
        {"doc_id": "A", "title": "Ada Lovelace",
         "text": "Ada Lovelace worked with Charles Babbage."},
        {"doc_id": "B", "title": "Charles Babbage",
         "text": "Charles Babbage designed the Analytical Engine."},
    ]
    ents = {"A": ["Ada Lovelace", "Charles Babbage"], "B": ["Charles Babbage"]}
    graph, _ = _build(cfg, pool, ents)
    embs = np.eye(2, dtype=float)

    # first-hop sim 0.3 < gate 0.5 -> no fan-out
    cands = graph.graph_evidence(
        {"question": "Who worked with Ada Lovelace?"},
        [(0, 0.3)], set(), embs[:0], embs)
    assert cands == [], f"weak first hop must be gated, got {cands}"

    # raise the first-hop sim above the gate -> candidate appears
    cands = graph.graph_evidence(
        {"question": "Who worked with Ada Lovelace?"},
        [(0, 0.9)], set(), embs[:0], embs)
    assert cands and cands[0][0] == 1, f"strong first hop should bridge, got {cands}"


def test_absolute_threshold_suppresses_lone_weak_candidate():
    """A single weak candidate below threshold yields no evidence.

    This guards against per-query max normalisation, which would have
    rescaled even a weak lone candidate up to bridge_alpha.
    """
    cfg = DRIPCoreConfig(
        bridge_abs_threshold=0.95,   # extreme: nothing should pass
        bridge_relation_floor=0.01,
        bridge_relation_overlap_weight=0.5,
    )
    pool = [
        {"doc_id": "A", "title": "Ada Lovelace",
         "text": "Ada Lovelace worked with Charles Babbage."},
        {"doc_id": "B", "title": "Charles Babbage",
         "text": "Charles Babbage designed the Analytical Engine."},
    ]
    ents = {"A": ["Ada Lovelace", "Charles Babbage"], "B": ["Charles Babbage"]}
    graph, _ = _build(cfg, pool, ents)
    embs = np.eye(2, dtype=float)

    cands = graph.graph_evidence(
        {"question": "Who worked with Ada Lovelace?"},
        [(0, 0.9)], set(), embs[:0], embs)
    assert cands == [], f"sub-threshold candidate must be dropped, got {cands}"
    assert graph.last_stats["bridge_after_threshold"] == 0


def test_degree_one_entity_creates_no_bridge():
    """An entity appearing in only one doc cannot bridge to a second doc."""
    cfg = DRIPCoreConfig(bridge_abs_threshold=0.01)
    pool = [
        {"doc_id": "A", "title": "Ada Lovelace",
         "text": "Ada Lovelace was a mathematician."},
        {"doc_id": "B", "title": "Charles Babbage",
         "text": "Charles Babbage was an inventor."},
    ]
    # No shared entity at all.
    ents = {"A": ["Ada Lovelace"], "B": ["Charles Babbage"]}
    graph, _ = _build(cfg, pool, ents)
    embs = np.eye(2, dtype=float)

    cands = graph.graph_evidence(
        {"question": "Who worked with Ada Lovelace?"},
        [(0, 0.9)], set(), embs[:0], embs)
    assert cands == [], f"no shared entity must mean no bridge, got {cands}"


if __name__ == "__main__":
    test_true_bridge_beats_high_degree_hub()
    test_firsthop_sim_gate_blocks_weak_first_hop()
    test_absolute_threshold_suppresses_lone_weak_candidate()
    test_degree_one_entity_creates_no_bridge()
    print("GraphIndex hostile probes passed")
