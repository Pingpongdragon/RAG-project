"""
构建 HotpotQA 全局知识库（所有候选文档）并按 Domain 分类
策略：
1. 先从 500 条 triplets 的 context 中提取文档
2. 如果不足 5000 条，从剩余数据中补充到 5000 条
输出：hotpot_kb/{0_entertainment, 1_stem, 2_humanities, 3_lifestyle}.jsonl
"""
import json, hashlib
from pathlib import Path
from tqdm import tqdm

HERE = Path(__file__).parent
HOTPOT_FILE = HERE / "hotpot_train_v1.1.json"
TRIPLET_DIR = HERE / "hotpot_triplets"
OUTPUT_DIR = HERE / "hotpot_kb"

# KB 大小配置
TARGET_KB_SIZE = 8000  # 每个 domain 的目标 KB 大小

KEYWORDS = {
    0: ["music", "movie", "tv", "film", "actor", "actress", "celebrity", "game", "comic", "fiction", "beatles", "pop", "song", "album", "band", "xbox", "nintendo"],
    1: ["science", "technology", "physics", "biology", "chemistry", "computer", "internet", "space", "nasa", "machine", "robot", "species", "formula", "theory", "software", "engineering"],
    2: ["history", "politics", "war", "battle", "army", "empire", "king", "queen", "president", "minister", "art", "literature", "writer", "philosophy", "religion", "democracy", "dynasty"],
    3: ["sport", "football", "basketball", "baseball", "olympic", "league", "team", "coach", "food", "cooking", "fashion", "travel", "pet", "hobby", "garden", "car", "fitness"]
}

DOMAIN_NAMES = {0: "0_entertainment", 1: "1_stem", 2: "2_humanities", 3: "3_lifestyle"}

def get_domain(text):
    if not text: return None
    t = text.lower()
    scores = {k: sum(1 for kw in kws if kw in t) for k, kws in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None

def download_hotpot():
    """自动下载 HotpotQA 数据集"""
    if HOTPOT_FILE.exists():
        print(f"✅ {HOTPOT_FILE} already exists, skip download.")
        return
    
    print("📥 Downloading HotpotQA from HuggingFace...")
    try:
        from datasets import load_dataset
        import os
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        
        dataset = load_dataset("hotpot_qa", "distractor", split="train")
        
        data = []
        for item in tqdm(dataset, desc="Converting"):
            entry = {
                "_id": item["id"],
                "question": item["question"],
                "answer": item["answer"],
                "type": item["type"],
                "context": [
                    [title, sentences] 
                    for title, sentences in zip(item["context"]["title"], item["context"]["sentences"])
                ],
                "supporting_facts": [
                    [title, sent_id]
                    for title, sent_id in zip(item["supporting_facts"]["title"], item["supporting_facts"]["sent_id"])
                ]
            }
            data.append(entry)
        
        print(f"💾 Saving to {HOTPOT_FILE}...")
        with open(HOTPOT_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        print(f"✅ Downloaded {len(data)} samples")
        
    except Exception as e:
        print(f"❌ Download failed: {e}")
        exit(1)

def process():
    download_hotpot()
    
    if not HOTPOT_FILE.exists():
        print(f"❌ {HOTPOT_FILE} not found!")
        return
    
    # ===== Step 1: 收集三元组使用的 query IDs 和 supporting_facts =====
    print("📋 Step 1: Loading triplets to get used query IDs and supporting facts...")
    used_triplet_ids = set()
    triplet_gold_docs = {}  # 存储每个triplet的gold_docs
    
    if TRIPLET_DIR.exists():
        for dom_id in range(4):
            triplet_file = TRIPLET_DIR / f"{DOMAIN_NAMES[dom_id]}.jsonl"
            if triplet_file.exists():
                with open(triplet_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        item = json.loads(line)
                        triplet_id = item["triplet_id"]
                        used_triplet_ids.add(triplet_id)
                        # 保存supporting_facts用于标记gold docs
                        if "gold_docs" in item:
                            triplet_gold_docs[triplet_id] = item["gold_docs"]
        print(f"   Found {len(used_triplet_ids)} triplets")
    else:
        print("   ⚠️ Triplet directory not found!")
        return
    
    # ===== Step 2: 加载 HotpotQA =====
    print(f"📖 Step 2: Loading {HOTPOT_FILE}...")
    data = json.load(open(HOTPOT_FILE, encoding="utf-8"))
    
    buckets = {i: {} for i in range(4)}
    used_entries = set()
    
    # ===== Step 3: 从 triplet 相关的 query 中提取文档 =====
    print("🔄 Step 3: Extracting docs from triplet queries...")
    for entry in tqdm(data):
        entry_id = entry.get("_id", "")[:12]
        
        if entry_id not in used_triplet_ids:
            continue
        
        used_entries.add(entry_id)
        
        # 获取该triplet的gold_docs
        gold_docs = triplet_gold_docs.get(entry_id, [])
        # gold_docs是文档文本列表，使用前50个字符作为标识
        gold_prefixes = {doc[:50] for doc in gold_docs if isinstance(doc, str)}
        
        context = entry.get("context", [])
        
        for title, sentences in context:
            full_text = " ".join(sentences).strip()
            
            if not full_text:
                continue
            
            # ✅ 用 title + 前300字符分类（避免文本过长）
            doc_snippet = f"{title} {full_text[:300]}"
            dom = get_domain(doc_snippet)
            if dom is None:
                continue
            
            # 使用 title 作为文档 ID
            doc_id = hashlib.md5(title.encode()).hexdigest()
            
            # 🔧 标记是否为gold doc
            is_gold = any(full_text.startswith(prefix) for prefix in gold_prefixes)
            
            if doc_id not in buckets[dom]:
                buckets[dom][doc_id] = {
                    "doc_id": doc_id,
                    "dataset": "hotpotqa",
                    "domain": DOMAIN_NAMES[dom],
                    "title": title,  # ✅ 保留title用于匹配
                    "text": full_text,
                    "source": "from_triplet",
                    "is_gold_doc": is_gold,  # ✅ 标记gold doc
                    "from_triplet_id": entry_id  # ✅ 记录来源triplet
                }
        
    # ===== Step 4: 补充到目标数量 =====
    print("\n🔄 Step 4: Filling KB to target size...")
    for dom_id in range(4):
        current_size = len(buckets[dom_id])
        print(f"   {DOMAIN_NAMES[dom_id]}: {current_size} docs from triplets", end="")
        
        if current_size >= TARGET_KB_SIZE:
            print(" ✅ (already sufficient)")
            if current_size > TARGET_KB_SIZE:
                import random
                docs_list = list(buckets[dom_id].values())
                sampled = random.sample(docs_list, TARGET_KB_SIZE)
                buckets[dom_id] = {d["doc_id"]: d for d in sampled}
                print(f" → Sampled to {TARGET_KB_SIZE}")
            continue
        
        needed = TARGET_KB_SIZE - current_size
        print(f", need {needed} more...")
        
        added = 0
        for entry in data:
            if added >= needed:
                break
            
            entry_id = entry.get("_id", "")[:12]
            if entry_id in used_entries:
                continue
            
            context = entry.get("context", [])
            for title, sentences in context:
                if added >= needed:
                    break
                
                full_text = " ".join(sentences).strip()
                if not full_text:
                    continue
                
                # ✅ 用 title + 前300字符分类
                doc_snippet = f"{title} {full_text[:300]}"
                dom = get_domain(doc_snippet)
                if dom != dom_id:
                    continue
                
                doc_id = hashlib.md5(title.encode()).hexdigest()
                
                if doc_id not in buckets[dom_id]:
                    buckets[dom_id][doc_id] = {
                        "doc_id": doc_id,
                        "dataset": "hotpotqa",
                        "domain": DOMAIN_NAMES[dom_id],
                        "title": title,  # ✅ 保留title
                        "text": full_text,
                        "source": "filler",
                        "is_gold_doc": False,  # ✅ filler文档不是gold
                        "from_triplet_id": None
                    }
                    added += 1
            
            used_entries.add(entry_id)
        
        print(f"      → Added {added} filler docs, total: {len(buckets[dom_id])}")
    
    # ===== Step 5: 保存 =====
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print("\n💾 Step 5: Saving KB...")
    for dom_id, docs_map in buckets.items():
        path = OUTPUT_DIR / f"{DOMAIN_NAMES[dom_id]}.jsonl"
        with open(path, 'w', encoding='utf-8') as f:
            for doc in docs_map.values():
                f.write(json.dumps(doc, ensure_ascii=False) + '\n')
        
        # 统计
        from_triplet_count = sum(1 for d in docs_map.values() if d.get("source") == "from_triplet")
        gold_doc_count = sum(1 for d in docs_map.values() if d.get("is_gold_doc") == True)
        filler_count = sum(1 for d in docs_map.values() if d.get("source") == "filler")
        
        print(f"✅ {DOMAIN_NAMES[dom_id]}: {len(docs_map)} docs "
              f"({from_triplet_count} from triplets, {gold_doc_count} gold docs, {filler_count} fillers) → {path}")
    
    print("\n" + "="*70)
    print("📊 Summary:")
    print("="*70)
    total = sum(len(docs_map) for docs_map in buckets.values())
    total_gold = sum(sum(1 for d in docs_map.values() if d.get("is_gold_doc") == True) for docs_map in buckets.values())
    print(f"Total KB documents: {total}")
    print(f"Total gold documents: {total_gold}")
    print(f"Target per domain: {TARGET_KB_SIZE}")
    
if __name__ == "__main__":
    process()