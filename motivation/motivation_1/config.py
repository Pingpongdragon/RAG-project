"""
Motivation 1 — Single-hop sanity experiment.

Same framework as Motivation 2 (multi-hop, motivation_4), but with single-hop
queries to validate that QueryDriven (dense topic alignment) DOES help when the
retrieval task is single-hop. This produces the BASELINE evidence that
QD signal is a real, useful signal — so the multi-hop failure (Motivation 2)
cannot be dismissed as "QD itself is broken".

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
    'n_source':       8000,
    'n_windows':      50,
    'window_size':    50,
    'n_clusters':     8,
    'top_head':       3,
    'kb_head_mult':   1.2,
}
DATASET_CONFIGS = {
    'hotpotqa_comparison': dict(_DEFAULT_CFG),
    '2wiki_comparison':     dict(_DEFAULT_CFG),
}

WRITE_CAP        = 200
PROBE_TOPK       = 50

DOC_ARRIVE       = 80
DOC_ADD_CAP      = WRITE_CAP
EDIT_BATCH       = WRITE_CAP
QD_TOP_K         = PROBE_TOPK
QD_REPLACE_CAP   = WRITE_CAP
FIFO_BATCH       = WRITE_CAP
FETCH_TOP_K      = PROBE_TOPK
LOG_FIX_TOP_K    = PROBE_TOPK
LOG_FIX_CAP      = WRITE_CAP
LOG_LAG_WINDOWS  = 5

STRATEGY_ORDER = [
    'Static', 'RandomFIFO', 'DocArrival', 'KnowledgeEdit',
    'OnDemandFetch', 'LogDrivenArrival', 'QueryDriven', 'Oracle',
]
STRATEGY_LABELS = {
    'Static':           'Static (no update)',
    'RandomFIFO':       'Random FIFO (blind ingest)',
    'DocArrival':       'Doc-Arrival (HippoRAG)',
    'KnowledgeEdit':    'Knowledge-Edit (RECIPE)',
    'OnDemandFetch':    'On-Demand Fetch (CRAG)',
    'LogDrivenArrival': 'Log-Driven (lagging)',
    'QueryDriven':      'Query-Driven (simple)',
    'Oracle':           'Oracle (upper bound)',
}
STRATEGY_STYLES = {
    'Static':           {'color': '#9CA3AF', 'marker': 'D', 'ls': '--'},
    'RandomFIFO':       {'color': '#D97706', 'marker': 'v', 'ls': '-.'},
    'DocArrival':       {'color': '#059669', 'marker': '^', 'ls': '-'},
    'KnowledgeEdit':    {'color': '#7C3AED', 'marker': 's', 'ls': '-'},
    'OnDemandFetch':    {'color': '#0891B2', 'marker': 'P', 'ls': ':'},
    'LogDrivenArrival': {'color': '#BE185D', 'marker': 'X', 'ls': '-.'},
    'QueryDriven':      {'color': '#2563EB', 'marker': 'o', 'ls': '-'},
    'Oracle':           {'color': '#DC2626', 'marker': '*', 'ls': '--'},
}

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('motivation_1')
