"""
Global constants and dataset configurations.

Embedding: BAAI/bge-large-en-v1.5 (1024-dim, cosine similarity).
  - Chosen for speed; production would use a larger model.

SF_HIT_THRESH: a document is considered a "hit" for a supporting fact
  if cosine similarity >= 0.55.  Calibrated on HotpotQA gold pairs.

K_LIST: standard IR recall cutoffs used in HippoRAG evaluation.
"""
import os
from pathlib import Path

# ── Paths ──────────────────────────────────────────
PROJECT_DIR = Path('/home/jyliu/RAG-project')
THIS_DIR    = Path(__file__).resolve().parent
DATA_DIR    = THIS_DIR / 'data'
FIG_DIR     = THIS_DIR / 'figures'
CACHE_DIR   = THIS_DIR / 'cache'

# ── Reproducibility ────────────────────────────────
SEED = 42

# ── Embedding ──────────────────────────────────────
EMBED_MODEL   = 'BAAI/bge-large-en-v1.5'
SF_HIT_THRESH = 0.62  # calibrated for BGE-large (gold p10~0.62, pool p95~0.44)

# ── Evaluation ─────────────────────────────────────
K_LIST = [1, 5, 10, 20]

# ── Per-dataset experiment parameters ──────────────
# All datasets share the same shape under `--expanded`. Only n_source default
# may differ; everything else is overridden by CLI args (--n-windows /
# --window-size).  Field reference:
#   n_source       sample budget for raw JSON loading (CLI overrides)
#   n_windows      evaluation windows
#   window_size    queries per window
#   n_clusters     KMeans clusters for query topic modelling (head/tail split)
#   top_head       top-N clusters treated as head (majority); rest = tail
#   kb_head_mult   KB size = #head-context-docs * kb_head_mult
# Why head-mult instead of pool-pct: pools differ 10x across datasets but
# head-context size scales with #queries, so a mult-of-head budget is
# directly comparable.  mult=1.2 -> H1 nearly saturates head SFs (~80-100%
# init coverage) but tail coverage stays ~10-20% so the drift isn't
# pre-cushioned.
_DEFAULT_CFG = {
    'n_source':       3500,
    'n_windows':      50,
    'window_size':    50,
    'n_clusters':     8,
    'top_head':       3,
    'kb_head_mult':   1.2,
}
DATASET_CONFIGS = {
    'hotpotqa':        dict(_DEFAULT_CFG),
    '2wikimultihopqa': dict(_DEFAULT_CFG),
    'musique':         dict(_DEFAULT_CFG),
    # DOM/control snippets are much more template-like than Wikipedia passages;
    # a paragraph-calibrated 0.62 threshold creates false "covered" positives.
    'mind2web_agent':  dict(
        _DEFAULT_CFG, n_clusters=10, top_head=4, kb_head_mult=0.8,
        sf_hit_thresh=0.82),
}

# ── Strategy hyper-parameters ──────────────────────
# DESIGN PRINCIPLE: comparable batch sizes across strategies so the
# comparison reflects "different signal source" rather than "different
# throughput", BUT the budget must be large enough that each method's
# best-case behaviour can manifest -- otherwise everything collapses to
# Static and we just prove "we starved them", not "the signal is wrong".
#
# Calibration target:
#   - tail SF docs needed in H2 ~= 2.5k-5k unique docs
#   - want WRITE_CAP * n_H2_windows >= 2 * tail_SF (saturation headroom)
#   - 200/window * 25 windows = 5,000 writes -> ~2x tail SF set.
#   - PROBE_TOPK=50: ~25 failing queries * 50 candidates = 1,250 raw
#     candidates per window, dedup to a few hundred, fully feeds cap.

WRITE_CAP        = 200  # SHARED: max KB writes per window for all writers
PROBE_TOPK       = 50   # SHARED: per-probe candidate retrieval width

# DocArrival  (models HippoRAG2 / LightRAG document-arrival pipeline)
DOC_ARRIVE   = 80     # docs randomly sampled from pool each window (probe size)
DOC_ADD_CAP  = WRITE_CAP

# KnowledgeEdit  (models RECIPE knowledge-edit pipeline)
EDIT_BATCH   = WRITE_CAP   # KB docs selected for edit each window

# DRIP direct-demand path
QD_TOP_K       = PROBE_TOPK  # per-failing-query candidate retrieval width
QD_REPLACE_CAP = WRITE_CAP

# RandomFIFO  (blind supply-side baseline)
FIFO_BATCH       = 40  # tight cap: ~LogDriven effective rate (200/5 windows); keeps RandomFIFO fair vs QDC empirical ~95/w and vs LogDriven amortised 40/w

# OnDemandFetch  (CRAG / Agent-style passive search, no KB update)
FETCH_TOP_K    = PROBE_TOPK

# ARC-compatible AMAT model:
#   AMAT = AMAT_HIT_COST + L2AccessRate * AMAT_MISS_PENALTY
AMAT_HIT_COST     = float(os.environ.get('AMAT_HIT_COST', '1.0'))
AMAT_MISS_PENALTY = float(os.environ.get('AMAT_MISS_PENALTY', '10.0'))

# LogDrivenArrival  (human-in-the-loop, fix previous window's failures)
LOG_FIX_TOP_K  = PROBE_TOPK
LOG_FIX_CAP    = WRITE_CAP
LOG_LAG_WINDOWS = 5    # human review cycle: apply fix every N windows


# ── Strategy registry (display order, labels, plot styles) ──
STRATEGY_ORDER = [
    'LRU', 'FIFO', 'TinyLFU',
    'GPTCacheStyle', 'Proximity',
    'AgentRAGCache',
    'DRIPNOdetector', 'DRIP',
    'Oracle',
]
STRATEGY_LABELS = {
    'LRU':              'LRU Cache',
    'FIFO':             'FIFO Cache',
    'TinyLFU':          'TinyLFU Cache',
    'GPTCacheStyle':    'Semantic Cache (GPTCache)',
    'Proximity':        'Proximity Cache',
    'AgentRAGCache':    'ARC (DRF+Hubness)',
    'DRIP':             'DRIP (ours)',
    'DRIPNOdetector':    'DRIP w/o Detector',
    'Oracle':           'Oracle (upper bound)',
}
STRATEGY_STYLES = {
    'LRU':              {'color': '#F59E0B', 'marker': 'v', 'ls': '-.'},
    'FIFO':             {'color': '#7C3AED', 'marker': '^', 'ls': '--'},
    'TinyLFU':          {'color': '#0284C7', 'marker': 'p', 'ls': ':'},
    'GPTCacheStyle':    {'color': '#0891B2', 'marker': 'P', 'ls': ':'},
    'Proximity':        {'color': '#06B6D4', 'marker': 'h', 'ls': ':'},
    'AgentRAGCache':    {'color': '#111827', 'marker': 'o', 'ls': '-'},
    'DRIP':             {'color': '#0F766E', 'marker': '*', 'ls': '-'},
    'DRIPNOdetector':    {'color': '#2563EB', 'marker': 's', 'ls': '-'},
    'Oracle':           {'color': '#DC2626', 'marker': '*', 'ls': '--'},
}

# ── Logging ────────────────────────────────────────
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('motivation')

# BGE-style retrieval prefix (applied to QUERIES only, not docs)
BGE_QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '
