"""
Global constants and dataset configurations.

Embedding: BAAI/bge-large-en-v1.5 (1024-dim, cosine similarity).
  - Chosen for speed; production would use a larger model.

SF_HIT_THRESH: a document is considered a "hit" for a supporting fact
  if cosine similarity >= 0.55.  Calibrated on HotpotQA gold pairs.

K_LIST: standard IR recall cutoffs used in HippoRAG evaluation.
"""
from pathlib import Path

# ── Paths ──────────────────────────────────────────
PROJECT_DIR = Path('/home/jyliu/RAG_project')
THIS_DIR    = Path(__file__).resolve().parent
DATA_DIR    = THIS_DIR / 'data'
FIG_DIR     = THIS_DIR / 'figures'
CACHE_DIR   = Path('/home/jyliu/RAG_project/motivation/motivation_2/cache')

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

# QueryDriven / QueryDrivenLive  (demand-side writers)
QD_TOP_K       = PROBE_TOPK  # per-failing-query candidate retrieval width
QD_REPLACE_CAP = WRITE_CAP

# RandomFIFO  (blind supply-side baseline)
FIFO_BATCH       = 40  # tight cap: ~LogDriven effective rate (200/5 windows); keeps RandomFIFO fair vs QDC empirical ~95/w and vs LogDriven amortised 40/w

# OnDemandFetch  (CRAG / Agent-style passive search, no KB update)
FETCH_TOP_K    = PROBE_TOPK

# LogDrivenArrival  (human-in-the-loop, fix previous window's failures)
LOG_FIX_TOP_K  = PROBE_TOPK
LOG_FIX_CAP    = WRITE_CAP
LOG_LAG_WINDOWS = 5    # human review cycle: apply fix every N windows


# ── Strategy registry (display order, labels, plot styles) ──
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

# ── Logging ────────────────────────────────────────
import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('motivation')

# BGE-style retrieval prefix (applied to QUERIES only, not docs)
BGE_QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '
