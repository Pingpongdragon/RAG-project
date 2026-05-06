"""
Motivation 1 — Single-hop sanity experiment.

Same framework as Motivation 2 (multi-hop, motivation_4), but with single-hop
queries to validate that QueryDrivenLive (dense alignment + same-window
write-through) really does help when the retrieval task is single-hop. This
produces the baseline evidence that demand-side query signal is real, and that
the remaining multi-hop gap comes from bridge retrieval rather than from the
idea of query-driven maintenance itself.

Single-hop here = HotpotQA "comparison" subset: each query independently
references ~2 documents that are dense-similar to the query (no bridging hop).
This is functionally a single-hop retrieval task while sharing the exact same
pool, embeddings, and evaluator as Motivation 2 — perfect controlled
comparison.
"""
from pathlib import Path

PROJECT_DIR = Path('/home/jyliu/RAG-project')
THIS_DIR    = Path(__file__).resolve().parent
DATA_DIR    = THIS_DIR / 'data'
FIG_DIR     = THIS_DIR / 'figures'
CACHE_DIR   = THIS_DIR / 'cache'

SEED = 42
EMBED_MODEL   = 'all-MiniLM-L6-v2'
SF_HIT_THRESH = 0.55
K_LIST = [1, 5, 10, 20]

_DEFAULT_CFG = {
    'n_source':       4000,
    'n_windows':      50,
    'window_size':    50,
    'n_clusters':     8,
    'top_head':       3,
    'kb_head_mult':   1.2,
}
DATASET_CONFIGS = {
    'squad': dict(_DEFAULT_CFG),
    'hotpotqa_comparison': dict(_DEFAULT_CFG, n_source=4000),
    # 2wiki_comparison evaluated at KB=8000 and excluded: 1.01 q/title,
    # QDC loses to RandomFIFO/Static. See ../DESIGN_DIRECTIONS.md.
    # Re-evaluated at KB=14000 (pass --kb-budget 14000) for competitive regime.
    '2wiki_comparison': dict(_DEFAULT_CFG, n_source=6000),
    # 2wiki_simple = comparison+inference+compositional (3 types, all 2-doc).
    # q/SF mean ~1.67 in train (vs 1.02 comp-only dev). Pragmatic near-direct
    # second testbed; inference/compositional involve mild bridging.
    '2wiki_simple': dict(_DEFAULT_CFG, n_source=15000),
    # 2wiki_bridge_comparison: 4-doc comparison-style with q/SF=2.90 (55.6% reuse).
    # Queries directly name two entities; bridges through 1 derived attribute.
    # Best structural near-direct candidate from local datasets.
    '2wiki_bridge_comparison': dict(_DEFAULT_CFG, n_source=8000),
    # TriviaQA Wikipedia: single-hop, q/SF≈1.35 at n=6000, query directly names SF entity.
    # First single-hop dataset satisfying BOTH conditions for QDC.
    # kb_head_mult=0.73: sets KB/pool≈33%, matching HotpotQA-comparison (no distractors in pool).
    'triviaqa_wikipedia': dict(_DEFAULT_CFG, n_source=6000, kb_head_mult=0.73),
    # FEVER: q/SF22488, Cond-B224898.6%, pool=SF+4xdistractor, drift: film->sport+music.
    # kb_head_mult=2.96: KB/pool224824%, matching HotpotQA (default 1.2 gives 10% 2192 RandomFIFO self-harms H1).
    'fever': dict(_DEFAULT_CFG, n_source=8000, kb_head_mult=1.8),
}

WRITE_CAP        = 200
PROBE_TOPK       = 50

DOC_ARRIVE       = 80
DOC_ADD_CAP      = WRITE_CAP
EDIT_BATCH       = WRITE_CAP
QD_TOP_K         = PROBE_TOPK
QD_REPLACE_CAP   = WRITE_CAP
FIFO_BATCH       = 40  # tight cap: ~LogDriven effective rate (200/5 windows); keeps RandomFIFO fair vs QDC empirical ~95/w and vs LogDriven amortised 40/w
FETCH_TOP_K      = PROBE_TOPK
LOG_FIX_TOP_K    = PROBE_TOPK
LOG_FIX_CAP      = WRITE_CAP
LOG_LAG_WINDOWS  = 5

STRATEGY_ORDER = [
    'Static', 'RandomFIFO', 'DocArrival', 'KnowledgeEdit',
    'OnDemandFetch', 'LogDrivenArrival', 'QueryDrivenCluster', 'Oracle',
]
STRATEGY_LABELS = {
    'Static':           'Static (no update)',
    'RandomFIFO':       'Random FIFO (blind ingest)',
    'DocArrival':       'Doc-Arrival (HippoRAG)',
    'KnowledgeEdit':    'Knowledge-Edit (RECIPE)',
    'OnDemandFetch':    'On-Demand Fetch (CRAG)',
    'LogDrivenArrival': 'Log-Driven (lagging)',
    'QueryDrivenCluster': 'Query-Driven Cluster (ours)',
    'Oracle':           'Oracle (upper bound)',
}
STRATEGY_STYLES = {
    'Static':           {'color': '#9CA3AF', 'marker': 'D', 'ls': '--'},
    'RandomFIFO':       {'color': '#D97706', 'marker': 'v', 'ls': '-.'},
    'DocArrival':       {'color': '#059669', 'marker': '^', 'ls': '-'},
    'KnowledgeEdit':    {'color': '#7C3AED', 'marker': 's', 'ls': '-'},
    'OnDemandFetch':    {'color': '#0891B2', 'marker': 'P', 'ls': ':'},
    'LogDrivenArrival': {'color': '#BE185D', 'marker': 'X', 'ls': '-.'},
    'QueryDrivenCluster': {'color': '#10B981', 'marker': 'D', 'ls': '-'},
    'Oracle':           {'color': '#DC2626', 'marker': '*', 'ls': '--'},
}

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('motivation_1')
