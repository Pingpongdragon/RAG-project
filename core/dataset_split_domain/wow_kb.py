"""
构建 WoW 全局知识库（所有候选文档）并按 Domain 分类
策略：
1. 先从 500 条 triplets 的相关文档中提取
2. 如果不足 2000 条，从剩余数据中补充到 2000 条
输出：wow_kb/{0_entertainment, 1_stem, 2_humanities, 3_lifestyle}.jsonl
"""
import json, hashlib
from pathlib import Path
from tqdm import tqdm
import os

HERE = Path(__file__).parent
TRIPLET_DIR = HERE / "wow_triplets"
OUTPUT_DIR = HERE / "wow_kb"

# KB 大小配置
TARGET_KB_SIZE = 2000

# 领域关键词（用于分类）
KEYWORDS = {
    "entertainment": ["music", "movie", "tv", "film", "actor", "actress", "celebrity", "game", "comic", "fiction", "beatles", "pop", "song", "album", "band", "xbox", "nintendo", "video game"],
    "stem": ["science", "technology", "physics", "biology", "chemistry", "computer", "internet", "space", "nasa", "machine", "robot", "species", "formula", "theory", "software", "engineering", "mathematics"],
    "humanities": ["history", "politics", "war", "battle", "army", "empire", "king", "queen", "president", "minister", "art", "literature", "writer", "philosophy", "religion", "democracy", "dynasty"],
    "lifestyle": ["sport", "football", "basketball", "baseball", "olympic", "league", "team", "coach", "food", "cooking", "fashion", "travel", "pet", "hobby", "garden", "car", "fitness"]
}

DOMAIN_NAMES = ["entertainment", "stem", "humanities", "lifestyle"]

# 领域到数字索引的映射
DOMAIN_TO_INDEX = {
    "entertainment": 0,
    "stem": 1,
    "humanities": 2,
    "lifestyle": 3
}

def get_domain(text):
    """根据关键词判断文本属于哪个领域"""
    if not text:
        return None
    
    text_lower = text.lower()
    scores = {}
    
    for domain, keywords in KEYWORDS.items():
        score = sum(1 for kw in keywords if kw in text_lower)
        scores[domain] = score
    
    # 找到得分最高的领域
    best_domain = max(scores, key=scores.get)
    
    # 如果最高分为 0，返回 None
    return best_domain if scores[best_domain] > 0 else None

def download_wow():
    """从 HuggingFace 加载 WoW 数据集"""
    print("📥 加载 Wizard of Wikipedia 数据集...")
    try:
        from datasets import load_dataset
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        dataset = load_dataset("chujiezheng/wizard_of_wikipedia", split="train")
        print(f"✅ 加载完成，共 {len(dataset)} 个样本")
        return dataset
        
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        exit(1)

def process():
    """构建 WoW 知识库"""
    
    # ===== 第一步：收集三元组中使用的 topics 和 gold_docs =====
    print("📋 Step 1: Loading triplets to extract used topics and gold docs...")
    used_topics = {d: set() for d in DOMAIN_NAMES}
    used_gold_docs = {d: set() for d in DOMAIN_NAMES}
    
    if TRIPLET_DIR.exists():
        for domain in DOMAIN_NAMES:
            domain_index = DOMAIN_TO_INDEX[domain]
            triplet_file = TRIPLET_DIR / f"{domain_index}_{domain}.jsonl"
            if triplet_file.exists():
                with open(triplet_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        used_topics[domain].add(item.get("topic", ""))
                        # 添加所有 gold_docs
                        for doc in item.get("gold_docs", []):
                            used_gold_docs[domain].add(doc)
        
        total_topics = sum(len(t) for t in used_topics.values())
        total_gold_docs = sum(len(g) for g in used_gold_docs.values())
        print(f"   Found {total_topics} used topics, {total_gold_docs} gold docs")
    else:
        print("   ⚠️ Triplet directory not found!")
        return
    
    # ===== 第二步：加载数据集 =====
    data = download_wow()
    
    # ===== 第三步：先提取三元组相关的文档 =====
    print("\n🔄 Step 2: Extracting docs from triplet topics...")
    buckets = {d: {} for d in DOMAIN_NAMES}
    used_sample_ids = {d: set() for d in DOMAIN_NAMES}  # 记录每个 domain 已使用的样本
    
    for idx, sample in enumerate(tqdm(data, desc="Processing triplet docs")):
        topics_list = sample.get('topics', [])
        knowledge_list = sample.get('knowledge', [])
        
        # 获取该对话的主 topic（通常是第一个）
        main_topic = topics_list[0] if topics_list else ""
        
        # 判断该 topic 属于哪个 domain
        domain = get_domain(main_topic)
        if domain is None:
            continue
        
        # 只处理三元组中使用的 topic
        if main_topic not in used_topics[domain]:
            continue
        
        used_sample_ids[domain].add(idx)
        
        # 提取该对话中的所有知识段落
        for knowledge_passages in knowledge_list:
            for passage in knowledge_passages:
                if not passage or "no_passages_used" in passage.lower():
                    continue
                
                # 使用 passage 作为文档内容
                doc_id = hashlib.md5(passage.encode()).hexdigest()
                
                if doc_id not in buckets[domain]:
                    buckets[domain][doc_id] = {
                        "doc_id": doc_id,
                        "dataset": "wow",
                        "domain": domain,
                        "title": main_topic,
                        "text": passage,
                        "source": "from_triplet"
                    }
    
    # ===== 第四步：验证 gold docs 是否都在 KB 中 =====
    print("\n🔍 Step 3: Verifying gold docs coverage...")
    for domain in DOMAIN_NAMES:
        texts_in_kb = {doc["text"] for doc in buckets[domain].values()}
        missing = used_gold_docs[domain] - texts_in_kb
        if missing:
            print(f"   ⚠️ {domain}: {len(missing)} gold docs missing, adding them...")
            # 将缺失的 gold docs 添加到 KB
            for gold_doc in missing:
                doc_id = hashlib.md5(gold_doc.encode()).hexdigest()
                if doc_id not in buckets[domain]:
                    # 尝试从 used_topics 中找一个 topic 作为 title
                    topic = list(used_topics[domain])[0] if used_topics[domain] else "unknown"
                    buckets[domain][doc_id] = {
                        "doc_id": doc_id,
                        "dataset": "wow",
                        "domain": domain,
                        "title": topic,
                        "text": gold_doc,
                        "source": "from_triplet"
                    }
        else:
            print(f"   ✅ {domain}: All gold docs covered")
    
    # ===== 第五步：如果不足 2000 条，从剩余数据中补充 =====
    print("\n🔄 Step 4: Filling KB to target size (2000 per domain)...")
    for domain in DOMAIN_NAMES:
        current_size = len(buckets[domain])
        print(f"   {domain}: {current_size} docs from triplets", end="")
        
        if current_size >= TARGET_KB_SIZE:
            print(" ✅ (already sufficient)")
            continue
        
        needed = TARGET_KB_SIZE - current_size
        print(f", need {needed} more...")
        
        added = 0
        for idx, sample in enumerate(data):
            if added >= needed:
                break
            
            # 跳过已使用的样本
            if idx in used_sample_ids[domain]:
                continue
            
            topics_list = sample.get('topics', [])
            main_topic = topics_list[0] if topics_list else ""
            
            # 判断 domain
            sample_domain = get_domain(main_topic)
            if sample_domain != domain:
                continue
            
            knowledge_list = sample.get('knowledge', [])
            
            for knowledge_passages in knowledge_list:
                if added >= needed:
                    break
                for passage in knowledge_passages:
                    if added >= needed:
                        break
                    if not passage or "no_passages_used" in passage.lower():
                        continue
                    
                    doc_id = hashlib.md5(passage.encode()).hexdigest()
                    
                    if doc_id not in buckets[domain]:
                        buckets[domain][doc_id] = {
                            "doc_id": doc_id,
                            "dataset": "wow",
                            "domain": domain,
                            "title": main_topic,
                            "text": passage,
                            "source": "filler"
                        }
                        added += 1
            
            used_sample_ids[domain].add(idx)
        
        print(f"      → Added {added} filler docs, total: {len(buckets[domain])}")
    
    # ===== 第六步：保存 =====
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n💾 Step 5: Saving KB...")
    for domain in DOMAIN_NAMES:
        domain_index = DOMAIN_TO_INDEX[domain]
        path = OUTPUT_DIR / f"{domain_index}_{domain}.jsonl"
        with open(path, 'w', encoding='utf-8') as f:
            for doc in buckets[domain].values():
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        from_triplet = sum(1 for d in buckets[domain].values() if d.get("source") == "from_triplet")
        filler = sum(1 for d in buckets[domain].values() if d.get("source") == "filler")
        print(f"✅ {domain}: {len(buckets[domain])} docs ({from_triplet} from triplets, {filler} fillers) → {path}")

if __name__ == "__main__":
    process()