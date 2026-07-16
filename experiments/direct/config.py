"""Direct QA 与自然 temporal workload 的统一实验配置。"""
import os
from pathlib import Path

PROJECT_DIR = Path('/home/jyliu/RAG-project')
THIS_DIR    = Path(__file__).resolve().parent
DATA_DIR    = THIS_DIR / 'data'
FIG_DIR     = THIS_DIR / 'figures'
CACHE_DIR   = THIS_DIR / 'cache'

DATA_SEED = int(os.environ.get('DATA_SEED', '42'))
SEED = int(os.environ.get('EXPERIMENT_SEED', '42'))
EMBED_MODEL   = os.environ.get('EMBED_MODEL', 'all-MiniLM-L6-v2')
SF_HIT_THRESH = float(os.environ.get('SF_HIT_THRESH', '0.55'))
BGE_QUERY_PREFIX = 'Represent this sentence for searching relevant passages: '
K_LIST = [1, 5, 10, 20]

_DEFAULT_CFG = {
    'n_source':       4000,
    'n_windows':      50,
    'window_size':    50,
    'kb_pool_ratio':  0.1,
}
DATASET_CONFIGS = {
    'squad': dict(_DEFAULT_CFG),
    'fever': dict(_DEFAULT_CFG, n_source=8000),
    # 将 DeepMind 官方评估元数据与本地正文镜像连接。问题严格保持 question_ts 顺序；
    # sweep 协议把前三个窗口保留为因果 warm-up 前缀。
    'streamingqa_official': dict(_DEFAULT_CFG, n_source=None, n_windows=50,
                                 window_size=500),
}

WRITE_CAP        = 200
PROBE_TOPK       = 50
FETCH_TOP_K      = PROBE_TOPK   # OnDemandFetch 每个 query 的文档池读取宽度。
AMAT_HIT_COST    = float(os.environ.get('AMAT_HIT_COST', '1.0'))
AMAT_MISS_PENALTY = float(os.environ.get('AMAT_MISS_PENALTY', '10.0'))

DOC_ARRIVE       = 80
DOC_ADD_CAP      = WRITE_CAP
EDIT_BATCH       = 8           # 每窗口约改写 KB(400) 的 2%，更贴近 RECIPE 设定。
QD_TOP_K         = PROBE_TOPK
QD_REPLACE_CAP   = WRITE_CAP
STRATEGY_ORDER = [
    'LRU', 'FIFO', 'TinyLFU', 'ClassicalARC',
    'GPTCacheStyle', 'Proximity',
    'AgentRAGCache',
    'DRIP',
    'Oracle',
]
STRATEGY_LABELS = {
    'LRU':                'LRU Cache',
    'FIFO':               'FIFO Cache',
    'TinyLFU':            'TinyLFU Cache',
    'ClassicalARC':       'Classical ARC',
    'GPTCacheStyle':      'Semantic Cache (GPTCache)',
    'Proximity':          'Proximity Cache',
    'AgentRAGCache':      'ARC (DRF+Hubness)',
    'DRIP':               'DRIP (ours)',
    'Oracle':             'Oracle (upper bound)',
}
STRATEGY_STYLES = {
    'LRU':                {'color': '#D97706', 'marker': 'v',  'ls': '-.'},
    'FIFO':               {'color': '#7C3AED', 'marker': '^',  'ls': '--'},
    'TinyLFU':            {'color': '#0284C7', 'marker': 'p',  'ls': ':'},
    'ClassicalARC':       {'color': '#2563EB', 'marker': 's',  'ls': '--'},
    'GPTCacheStyle':      {'color': '#0891B2', 'marker': 'P',  'ls': ':'},
    'Proximity':          {'color': '#06B6D4', 'marker': 'h',  'ls': ':'},
    'AgentRAGCache':      {'color': '#111827', 'marker': 'o',  'ls': '-'},
    'DRIP':               {'color': '#0F766E', 'marker': '*',  'ls': '-'},
    'Oracle':             {'color': '#DC2626', 'marker': '*',  'ls': '--'},
}

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger('experiments.direct')
