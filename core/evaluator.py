"""
评测模块: 基于文档内容的精确匹配
"""
import json
from pathlib import Path
from typing import List, Dict
import hashlib
from RAG_project.config.logger_config import logger
import numpy as np


def load_test_data(shift_type: str = "sudden") -> List[Dict]:
    """
    加载 domain shift 测试数据，并提取 gold_doc_ids
    """
    HERE = Path(__file__).parent
    data_file = HERE / "domain_shift_datasets" / "hotpot_shifts" / f"{shift_type}_4domains.jsonl"
    
    if not data_file.exists():
        raise FileNotFoundError(f"❌ 数据文件不存在: {data_file}")
    
    # 加载KB文档以建立 content -> doc_id 映射
    kb_content_map = _load_kb_content_mapping(HERE / "dataset_split_domain" / "hotpot_kb")
    
    # 加载triplet的gold_docs映射
    triplet_gold_map = _load_triplet_gold_docs(HERE / "dataset_split_domain" / "hotpot_triplets")
    
    queries = []
    with open(data_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            obj = json.loads(line)
            
            # 跳过metadata行
            if "metadata" in obj:
                logger.info(f"📋 数据集元信息: {obj['metadata'].get('shift_type')}")
                continue
            
            triplet_id = obj.get("triplet_id", "")
            if not triplet_id:
                continue
            
            # 从triplet获取gold_docs（文本列表）
            gold_docs_text = triplet_gold_map.get(triplet_id, [])
            
            # 转换为doc_id
            gold_doc_ids = []
            for doc_text in gold_docs_text:
                # 用前100字符作为key去映射中查找
                key = doc_text.strip()[:100]
                if key in kb_content_map:
                    gold_doc_ids.append(kb_content_map[key])
            
            queries.append({
                "query": obj["query"],
                "answer": obj["answer"],
                "gold_doc_ids": gold_doc_ids,
                "domain": obj.get("domain", "unknown"),
                "topic": obj.get("topic", ""),
                "triplet_id": triplet_id
            })
    
    valid_count = sum(1 for q in queries if q['gold_doc_ids'])
    logger.info(f"✅ 加载 {shift_type} 数据集: {len(queries)} 条（有效gold: {valid_count}）")
    return queries


def _load_kb_content_mapping(kb_dir: Path) -> Dict[str, str]:
    """
    建立 content前缀 -> doc_id 的映射
    返回: {content[:100]: doc_id}
    """
    mapping = {}
    domains = ["0_entertainment", "1_stem", "2_humanities", "3_lifestyle"]
    
    for domain in domains:
        kb_file = kb_dir / f"{domain}.jsonl"
        if not kb_file.exists():
            continue
        
        with open(kb_file, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                doc_id = obj.get("doc_id")
                text = obj.get("text", "")
                
                if doc_id and text:
                    key = text.strip()[:100]
                    mapping[key] = doc_id
    
    logger.info(f"✅ 建立 KB content映射: {len(mapping)} 条")
    return mapping


def _load_triplet_gold_docs(triplet_dir: Path) -> Dict[str, List[str]]:
    """
    加载triplet的gold_docs（文本）
    返回: {triplet_id: [doc_text1, doc_text2, ...]}
    """
    mapping = {}
    domains = ["0_entertainment", "1_stem", "2_humanities", "3_lifestyle"]
    
    for domain in domains:
        triplet_file = triplet_dir / f"{domain}.jsonl"
        if not triplet_file.exists():
            continue
        
        with open(triplet_file, 'r', encoding='utf-8') as f:
            for line in f:
                obj = json.loads(line)
                triplet_id = obj.get("triplet_id")
                gold_docs = obj.get("gold_docs", [])
                
                if triplet_id and gold_docs:
                    mapping[triplet_id] = gold_docs
    
    logger.info(f"✅ 加载 triplet gold_docs: {len(mapping)} 条")
    return mapping


def compute_retrieval_score(kb, domain: str, query_vec: np.ndarray, gold_doc_ids: List[str], step: int, top_k: int = 10) -> float:
    """
    计算检索得分 (Recall@k)
    """
    if not gold_doc_ids or not query_vec.any():
        return 0.0
    
    
    retrieved_docs = kb.search(query_vec, domain, step=step, top_k=top_k)
    
    if not retrieved_docs:
        return 0.0
    
    retrieved_ids = set(doc.doc_id for doc in retrieved_docs)
    gold_ids = set(gold_doc_ids)
    matched_count = len(retrieved_ids & gold_ids)
    
    # 调试（前3次）
    if step < 3:
        logger.info(f"🔍 Step {step} | Matched: {matched_count}/{len(gold_ids)}")
        if matched_count > 0:
            logger.info(f"   ✅ Gold: {list(gold_ids)[:2]}")
            logger.info(f"   ✅ Retrieved: {list(retrieved_ids)[:2]}")
    
    return matched_count / len(gold_ids)
