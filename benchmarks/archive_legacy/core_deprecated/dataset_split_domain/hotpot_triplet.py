"""
提取 HotpotQA (Query, Answer, Gold Docs) 三元组并按 Domain 分类
输出：hotpot_triplets/{0_entertainment, 1_stem, 2_humanities, 3_lifestyle}.jsonl
"""
import json, hashlib
from pathlib import Path
from tqdm import tqdm

HERE = Path(__file__).parent
HOTPOT_FILE = HERE / "hotpot_train_v1.1.json"
OUTPUT_DIR = HERE / "hotpot_triplets"

# 每个 Domain 最多保留多少个三元组
MAX_TRIPLETS_PER_DOMAIN = 500  # 每个 domain 500 条 query

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
    
    print(f"📖 Loading {HOTPOT_FILE}...")
    data = json.load(open(HOTPOT_FILE, encoding="utf-8"))
    
    buckets = {i: [] for i in range(4)}
    domain_counts = {i: 0 for i in range(4)}
    
    print(f"🔄 Processing HotpotQA Triplets (Max {MAX_TRIPLETS_PER_DOMAIN} per domain)...")
    for entry in tqdm(data):
        question = entry.get("question", "")
        answer = entry.get("answer", "")
        
        dom = get_domain(question)
        if dom is None: continue
        
        # 如果该 domain 已达到上限，跳过
        if domain_counts[dom] >= MAX_TRIPLETS_PER_DOMAIN:
            continue
        
        # 提取 Gold Docs
        gold_docs = []
        supporting_facts = entry.get("supporting_facts", [])
        context = entry.get("context", [])
        
        for title, sent_id in supporting_facts:
            for ctx_title, ctx_sents in context:
                if ctx_title == title and sent_id < len(ctx_sents):
                    gold_docs.append(ctx_sents[sent_id])
                    break
        
        if not gold_docs:
            continue
        
        triplet = {
            "triplet_id": entry.get("_id", "")[:12],
            "dataset": "hotpotqa",
            "domain": DOMAIN_NAMES[dom],
            "type": entry.get("type", "unknown"),
            "query": question,
            "answer": answer,
            "gold_docs": gold_docs
        }
        
        buckets[dom].append(triplet)
        domain_counts[dom] += 1
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for dom_id, items in buckets.items():
        path = OUTPUT_DIR / f"{DOMAIN_NAMES[dom_id]}.jsonl"
        with open(path, 'w', encoding='utf-8') as f:
            for item in items:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
        print(f"✅ {DOMAIN_NAMES[dom_id]}: {len(items)} triplets → {path}")

if __name__ == "__main__":
    process()