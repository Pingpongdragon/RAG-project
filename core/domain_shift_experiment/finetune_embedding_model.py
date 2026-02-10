import os
# 🔴 [关键修改 1] 必须放在所有 import 之前！
# 强制 Tokenizers 只使用单线程，避免与 DataLoader 的多进程冲突导致死锁/变慢
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

import json
import logging
import torch
import gc
from pathlib import Path
from torch.utils.data import DataLoader
from sentence_transformers import SentenceTransformer, InputExample, losses

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

DATA_ROOT = Path("./data")
FIQA_DIR = DATA_ROOT / "raw_data" / "fiqa"
MODELS_DIR = DATA_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

EPOCHS = 2
MAX_SEQ_LENGTH = 512

TRAIN_TASKS = [
    {
        "name": "MiniLM-L6",
        "base_model": "sentence-transformers/all-MiniLM-L6-v2",
        "output_path": str(MODELS_DIR / "minilm_l6_fiqa_finetuned"),
        "batch_size": 64,  
        "use_grad_checkpoint": False 
    },
    {
        "name": "BGE-M3",
        "base_model": "BAAI/bge-m3", 
        "output_path": str(MODELS_DIR / "bge_m3_fiqa_finetuned"),
        "batch_size": 16,  
        "use_grad_checkpoint": True 
    }
]

def load_train_data():
    logging.info("📖 正在加载训练数据...")
    
    corpus = {}
    with open(FIQA_DIR / "corpus.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            doc = json.loads(line)
            corpus[doc['_id']] = doc['text']

    queries = {}
    with open(FIQA_DIR / "queries.jsonl", 'r', encoding='utf-8') as f:
        for line in f:
            q = json.loads(line)
            queries[q['_id']] = q['text']

    train_examples = []
    qrels_path = FIQA_DIR / "qrels" / "train.tsv"
    
    if not qrels_path.exists():
        raise FileNotFoundError(f"❌ 未找到训练数据: {qrels_path}")

    logging.info("🔨 正在构建训练样本...")
    with open(qrels_path, 'r', encoding='utf-8') as f:
        next(f)
        lines = f.readlines()
        # 为了快速验证速度，你可以先只取前 1000 条跑跑看，确认速度正常后再跑全量
        # lines = lines[:1000] 
        for line in lines:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                qid, docid = parts[0], parts[1]
                if qid in queries and docid in corpus:
                    train_examples.append(InputExample(texts=[queries[qid], corpus[docid]]))

    logging.info(f"✅ 训练集构建完成，共 {len(train_examples)} 个样本。")
    return train_examples

def train():
    torch.cuda.empty_cache()
    gc.collect()

    train_examples = load_train_data()
    
    for task in TRAIN_TASKS:
        logging.info("\n" + "="*60)
        logging.info(f"🚀 开始微调模型: {task['name']}")
        logging.info(f"   Batch Size: {task['batch_size']}")
        logging.info("="*60)
        
        model = SentenceTransformer(task['base_model'])
        model.max_seq_length = MAX_SEQ_LENGTH
        
        if task['use_grad_checkpoint']:
            logging.info("⚡ 已开启 Gradient Checkpointing")
            model[0].auto_model.gradient_checkpointing_enable()

        # 🔴 [关键修改 2] 优化 DataLoader 参数
        train_dataloader = DataLoader(
                train_examples, 
                shuffle=True, 
                batch_size=task['batch_size'],
                # 建议设置为 CPU 核心数的一半，或者 4-8 之间
                # 如果你的 CPU 核心很多，可以尝试 8 或 16
                num_workers=8,      
                pin_memory=True,    
                # 增加预取因子，让 CPU 提前准备更多数据
                prefetch_factor=2,  
                # 保持 worker 进程存活，避免每个 epoch 结束后重新创建进程的开销
                persistent_workers=True 
            )
        
        train_loss = losses.MultipleNegativesRankingLoss(model)
        
        # 🔴 [额外提示] 如果还是慢，可以尝试把 use_amp=True 改为 False 测一下
        # 虽然 amp 会加速，但在某些极端的驱动版本下可能引发异常，不过通常建议开启
        model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=EPOCHS,
            warmup_steps=int(len(train_dataloader) * 0.1),
            show_progress_bar=True,
            output_path=task['output_path'],
            use_amp=True 
        )
        logging.info(f"✅ {task['name']} 微调完成！")
        
        del model
        del train_loss # 显式删除 Loss 对象
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    train()