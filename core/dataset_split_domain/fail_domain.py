"""
对比 HotpotQA 中用 Question 分类 vs 用 Document 分类的差异
"""
import json
from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

HERE = Path(__file__).parent
HOTPOT_FILE = HERE / "hotpot_train_v1.1.json"

KEYWORDS = {
    0: ["music", "movie", "tv", "film", "actor", "actress", "celebrity", "game", "comic", "fiction", "beatles", "pop", "song", "album", "band", "xbox", "nintendo"],
    1: ["science", "technology", "physics", "biology", "chemistry", "computer", "internet", "space", "nasa", "machine", "robot", "species", "formula", "theory", "software", "engineering"],
    2: ["history", "politics", "war", "battle", "army", "empire", "king", "queen", "president", "minister", "art", "literature", "writer", "philosophy", "religion", "democracy", "dynasty"],
    3: ["sport", "football", "basketball", "baseball", "olympic", "league", "team", "coach", "food", "cooking", "fashion", "travel", "pet", "hobby", "garden", "car", "fitness"]
}

DOMAIN_NAMES = {0: "Entertainment", 1: "STEM", 2: "Humanities", 3: "Lifestyle", None: "Unknown"}

def get_domain(text):
    """关键词分类"""
    if not text: return None
    t = text.lower()
    scores = {k: sum(1 for kw in kws if kw in t) for k, kws in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else None

def analyze_classification_difference():
    """分析问题分类 vs 文档分类的差异"""
    
    if not HOTPOT_FILE.exists():
        print(f"❌ {HOTPOT_FILE} not found! Please run hotpot_kb.py first.")
        return
    
    print(f"📖 Loading {HOTPOT_FILE}...")
    data = json.load(open(HOTPOT_FILE, encoding="utf-8"))
    
    # 统计数据
    total_samples = 0
    skipped_unknown_questions = 0  # 跳过的 unknown question
    skipped_unknown_docs = 0  # 🔧 新增：跳过的 unknown document
    total_docs = 0
    match_count = 0
    mismatch_count = 0
    
    # 详细记录
    mismatch_details = []
    confusion_matrix = defaultdict(lambda: defaultdict(int))
    
    print("\n🔍 Analyzing classification differences...")
    print("="*80)
    
    for entry in tqdm(data[:1000], desc="Processing samples"):
        question = entry.get("question", "")
        context = entry.get("context", [])
        
        # 问题分类
        q_domain = get_domain(question)
        
        # 🔧 跳过 question 为 Unknown 的样本
        if q_domain is None:
            skipped_unknown_questions += 1
            continue
        
        total_samples += 1
        
        # 遍历该问题的所有文档
        for title, sentences in context:
            full_text = " ".join(sentences).strip()
            if not full_text:
                continue
            
            # 文档分类（使用 title + 前300字符）
            doc_snippet = f"{title} {full_text[:300]}"
            d_domain = get_domain(doc_snippet)
            
            # 🔧 跳过 document 为 Unknown 的样本
            if d_domain is None:
                skipped_unknown_docs += 1
                continue
            
            total_docs += 1
            
            # 统计
            if q_domain == d_domain:
                match_count += 1
            else:
                mismatch_count += 1
                
                # 记录不匹配案例（只记录前20个）
                if len(mismatch_details) < 20:
                    mismatch_details.append({
                        "question": question,
                        "question_domain": DOMAIN_NAMES[q_domain],
                        "doc_title": title,
                        "doc_snippet": full_text[:150] + "...",
                        "doc_domain": DOMAIN_NAMES[d_domain]
                    })
            
            # 混淆矩阵
            confusion_matrix[q_domain][d_domain] += 1
    
    # ===== 输出报告 =====
    print("\n" + "="*80)
    print("📊 CLASSIFICATION COMPARISON REPORT")
    print("="*80)
    
    print(f"\n📈 Overall Statistics:")
    print(f"   Skipped Unknown questions: {skipped_unknown_questions}")
    print(f"   Skipped Unknown documents: {skipped_unknown_docs}")  # 🔧 新增输出
    print(f"   Total questions analyzed: {total_samples} (excluding Unknown)")
    print(f"   Total documents analyzed: {total_docs} (excluding Unknown)")  # 🔧 修改说明
    print(f"   ✅ Matches (Q ≡ D):       {match_count} ({match_count/total_docs*100:.1f}%)")
    print(f"   ❌ Mismatches (Q ≠ D):    {mismatch_count} ({mismatch_count/total_docs*100:.1f}%)")
    
    # 混淆矩阵（不包含 Unknown）
    print("\n📋 Confusion Matrix (Question → Document):")
    print("-"*80)
    header = "Question \\ Doc  |"
    for d in [0, 1, 2, 3]:  # 🔧 移除 None
        header += f" {DOMAIN_NAMES[d]:<12} |"
    print(header)
    print("-"*80)
    
    for q_dom in [0, 1, 2, 3]:
        row = f"{DOMAIN_NAMES[q_dom]:<16} |"
        for d_dom in [0, 1, 2, 3]:  # 🔧 移除 None
            count = confusion_matrix[q_dom][d_dom]
            row += f" {count:<12} |"
        print(row)
    
    # 详细案例
    if mismatch_details:
        print("\n" + "="*80)
        print("🔍 MISMATCH EXAMPLES (Top 20)")
        print("="*80)
        
        for i, case in enumerate(mismatch_details, 1):
            print(f"\n[Case {i}]")
            print(f"❓ Question: {case['question']}")
            print(f"   → Domain: {case['question_domain']}")
            print(f"\n📄 Document: {case['doc_title']}")
            print(f"   Content: {case['doc_snippet']}")
            print(f"   → Domain: {case['doc_domain']}")
            print("-"*80)
    
    # 错误率分析
    print("\n" + "="*80)
    print("⚠️  ERROR ANALYSIS")
    print("="*80)
    
    error_rate = mismatch_count / total_docs * 100 if total_docs > 0 else 0
    print(f"Overall Error Rate: {error_rate:.2f}%")
    
    if error_rate > 30:
        print("🔴 HIGH ERROR RATE! Using question for classification may be unreliable.")
        print("   Recommendation: Use document content for classification.")
    elif error_rate > 15:
        print("🟡 MODERATE ERROR RATE. Question-based classification has limitations.")
        print("   Recommendation: Consider using document content for better accuracy.")
    else:
        print("🟢 LOW ERROR RATE. Question and document domains are mostly aligned.")
    
    # 保存结果
    output_file = HERE / "classification_comparison.json"
    result = {
        "skipped_unknown_questions": skipped_unknown_questions,
        "skipped_unknown_docs": skipped_unknown_docs,  # 🔧 新增字段
        "total_samples": total_samples,
        "total_docs": total_docs,
        "match_count": match_count,
        "mismatch_count": mismatch_count,
        "error_rate": error_rate,
        "confusion_matrix": {
            DOMAIN_NAMES[q]: {DOMAIN_NAMES[d]: confusion_matrix[q][d] for d in confusion_matrix[q]}
            for q in confusion_matrix if q is not None
        },
        "mismatch_examples": mismatch_details
    }
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    
    print(f"\n💾 Full report saved to: {output_file}")

if __name__ == "__main__":
    analyze_classification_difference()