"""
Global constants and dataset configurations.

Embedding: all-MiniLM-L6-v2 (Sentence-BERT, 384-dim, cosine similarity).
  - Chosen for speed; production would use a larger model.

SF_HIT_THRESH: a document is considered a "hit" for a supporting fact
  if cosine similarity >= 0.55.  Calibrated on HotpotQA gold pairs.

K_LIST: standard IR recall cutoffs used in HippoRAG evaluation.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────
PROJECT_DIR = Path('/home/jyliu/RAG-project')
THIS_DIR    = Path(__file__).resolve().parent
DATA_DIR    = THIS_DIR / 'data'
FIG_DIR     = THIS_DIR / 'figures'
CACHE_DIR   = Path('/data/jyliu/RAG-project/motivation/motivation_4/cache')

# ── Reproducibility ────────────────────────────────
SEED = 42

# ── Embedding ──────────────────────────────────────
EMBED_MODEL   = 'all-MiniLM-L6-v2'
SF_HIT_THRESH = 0.55

# ── Evaluation ─────────────────────────────────────
K_LIST = [1, 5, 10, 20]

# ── Per-dataset experiment parameters ──────────────
# n_source:       how many raw items to sample from the JSON
# n_windows:      number of evaluation windows in the stream
# window_size:    queries per window
# n_clusters:     KMeans clusters for query topic modelling
# top_head:       top-N clusters treated as "head" (majority) topics
# kb_budget_pct:  KB size as fraction of doc pool
# use_focus_pool: if True, shrink pool to only docs reachable from stream
DATASET_CONFIGS = {
    'hotpotqa': {
        'n_source': 2000,
        'n_windows': 20, 'window_size': 50,
        'n_clusters': 8, 'top_head': 3,
        'kb_budget_pct': 0.10,
        'use_focus_pool': True,
    },
    '2wikimultihopqa': {
        'n_windows': 20, 'window_size': 25,
        'n_clusters': 8, 'top_head': 3,
        'kb_budget_pct': 0.10,
        'use_focus_pool': False,
    },
    'musique': {
        'n_windows': 20, 'window_size': 25,
        'n_clusters': 8, 'top_head': 3,
        'kb_budget_pct': 0.10,
        'use_focus_pool': False,
    },
}

# ── Strategy hyper-parameters ──────────────────────
# DocArrival  (models HippoRAG2 / LightRAG document-arrival pipeline)
DOC_ARRIVE   = 80     # docs sampled from pool each window
DOC_ADD_CAP  = 40     # max replacements per window

# KnowledgeEdit  (models RECIPE knowledge-edit pipeline)
EDIT_BATCH   = 30     # KB docs selected for edit each window

# QueryDriven  (our demand-side approach)
QD_TOP_K       = 50   # per-failing-query candidate retrieval width
QD_REPLACE_CAP = 60   # max replacements per window (at full drift)

# RandomFIFO  (blind supply-side baseline)
FIFO_BATCH     = 50   # random docs swapped per window

# OnDemandFetch  (CRAG / Agent-style passive search, no KB update)
FETCH_TOP_K    = 20   # docs fetched from pool per failing query

# LogDrivenArrival  (human-in-the-loop, fix previous window's failures)
LOG_FIX_TOP_K  = 30   # candidates per failing query
LOG_FIX_CAP    = 50   # max docs to queue for next window
LOG_LAG_WINDOWS = 5    # human review cycle: apply fix every N windows


# ── Strategy registry (display order, labels, plot styles) ──
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

# ── Logging ────────────────────────────────────────
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('motivation')
