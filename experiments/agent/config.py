"""Agent trace loader 与 embedding 缓存配置。

这里不定义合成 topic drift。MIND 保留真实时间戳，Wizard of Wikipedia 与 MT-RAG
保留会话内 turn order；未来 evidence 标签只能在服务完成后作为反馈。
"""

import logging
import os
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
THIS_DIR = Path(__file__).resolve().parent
CACHE_DIR = THIS_DIR / "cache"
DATA_DIR = THIS_DIR / "data"

DATA_SEED = int(os.environ.get("DATA_SEED", "42"))
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-large-en-v1.5")
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
log = logging.getLogger("experiments.agent")
