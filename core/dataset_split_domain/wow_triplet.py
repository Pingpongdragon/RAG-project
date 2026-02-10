"""
提取 WoW (Query, Answer, Gold Docs) 三元组并按 Domain 分类
输出：wow_triplets/{entertainment, stem, humanities, lifestyle}_triplets.json
"""
import json
from pathlib import Path
from tqdm import tqdm
import os

HERE = Path(__file__).parent
OUTPUT_DIR = HERE / "wow_triplets"
OUTPUT_DIR.mkdir(exist_ok=True)

# 每个 Domain 最多保留多少个三元组
MAX_TRIPLETS_PER_DOMAIN = 500

# 领域关键词（用于分类）
KEYWORDS = {
    "entertainment": ["music", "movie", "tv", "film", "actor", "actress", "celebrity", "game", "comic", "fiction", "beatles", "pop", "song", "album", "band", "xbox", "nintendo", "video game"],
    "stem": ["science", "technology", "physics", "biology", "chemistry", "computer", "internet", "space", "nasa", "machine", "robot", "species", "formula", "theory", "software", "engineering", "mathematics"],
    "humanities": ["history", "politics", "war", "battle", "army", "empire", "king", "queen", "president", "minister", "art", "literature", "writer", "philosophy", "religion", "democracy", "dynasty"],
    "lifestyle": ["sport", "football", "basketball", "baseball", "olympic", "league", "team", "coach", "food", "cooking", "fashion", "travel", "pet", "hobby", "garden", "car", "fitness"]
}

DOMAIN_NAMES = list(KEYWORDS.keys())


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
    """处理 WoW 数据集，提取 (query, answer, gold_docs) 三元组"""
    
    data = download_wow()
    domain_counts = {d: 0 for d in DOMAIN_NAMES}
    domain_triplets = {d: [] for d in DOMAIN_NAMES}
    
    # 统计信息
    total_samples = 0
    total_turns = 0
    valid_triplets = 0
    
    for sample in tqdm(data, desc="处理 WoW 数据"):
        total_samples += 1
        
        # WoW 数据结构：post, response, knowledge, labels, topics
        posts = sample.get('post', [])
        responses = sample.get('response', [])
        knowledge_list = sample.get('knowledge', [])
        labels = sample.get('labels', [])
        topics = sample.get('topics', [])
        
        # 构建对话历史
        conversation_history = []
        
        # 遍历每一轮对话
        for i in range(len(responses)):
            total_turns += 1
            
            # 当前用户的问题 (post)
            current_post = posts[i].strip() if i < len(posts) else ""
            if not current_post:
                continue
            
            # answer = 向导的回答 (response)
            answer = responses[i].strip() if i < len(responses) else ""
            if not answer:
                continue
            
            # gold_docs = 向导实际使用的知识段落
            if i >= len(knowledge_list) or i >= len(labels):
                # 更新历史后继续
                conversation_history.append(f"User: {current_post}")
                conversation_history.append(f"Wizard: {answer}")
                continue
            
            label_idx = labels[i]
            knowledge_passages = knowledge_list[i]
            
            # 获取实际使用的知识段落
            if label_idx >= len(knowledge_passages):
                conversation_history.append(f"User: {current_post}")
                conversation_history.append(f"Wizard: {answer}")
                continue
            
            gold_passage = knowledge_passages[label_idx].strip()
            
            # 跳过 "no_passages_used"
            if not gold_passage or "no_passages_used" in gold_passage.lower():
                conversation_history.append(f"User: {current_post}")
                conversation_history.append(f"Wizard: {answer}")
                continue
            
            valid_triplets += 1
            
            # 构建包含上下文历史的 query
            if conversation_history:
                # 拼接对话历史 + 当前问题
                query_with_context = " [SEP] ".join(conversation_history) + f" [SEP] User: {current_post}"
            else:
                # 第一轮对话，只有当前问题
                query_with_context = f"User: {current_post}"
            
            # 判断领域（使用 topic + query + answer + gold_passage）
            topic = topics[i] if i < len(topics) else ""
            combined_text = f"{topic} {current_post} {answer} {gold_passage}"
            domain = get_domain(combined_text)
            
            if domain is None:
                # 更新历史后继续
                conversation_history.append(f"User: {current_post}")
                conversation_history.append(f"Wizard: {answer}")
                continue
            
            # 检查该领域是否已满
            if domain_counts[domain] >= MAX_TRIPLETS_PER_DOMAIN:
                # 更新历史后继续
                conversation_history.append(f"User: {current_post}")
                conversation_history.append(f"Wizard: {answer}")
                continue
            
            # 添加到对应领域
            triplet = {
                'query': query_with_context,  # 包含完整对话历史的 query
                'answer': answer,
                'gold_docs': [gold_passage],  # 保持列表格式以与 HotpotQA 一致
                'topic': topic  # 额外保存 topic 信息
            }
            domain_triplets[domain].append(triplet)
            domain_counts[domain] += 1
            
            # 更新对话历史
            conversation_history.append(f"User: {current_post}")
            conversation_history.append(f"Wizard: {answer}")
            
            # 检查是否所有领域都已满
            if all(count >= MAX_TRIPLETS_PER_DOMAIN for count in domain_counts.values()):
                break
        
        if all(count >= MAX_TRIPLETS_PER_DOMAIN for count in domain_counts.values()):
            break
    
    # 打印统计信息
    print(f"\n=== 处理统计 ===")
    print(f"总样本数: {total_samples}")
    print(f"总对话轮次: {total_turns}")
    print(f"有效三元组: {valid_triplets}")
    print(f"\n各领域提取数量:")
    for domain in DOMAIN_NAMES:
        print(f"  {domain}: {domain_counts[domain]}")
    
    # 保存结果
    for domain in DOMAIN_NAMES:
        if domain_triplets[domain]:
            domain_index = DOMAIN_TO_INDEX[domain]
            output_file = OUTPUT_DIR / f"{domain_index}_{domain}.jsonl"
            with open(output_file, 'w', encoding='utf-8') as f:
                for triplet in domain_triplets[domain]:
                    f.write(json.dumps(triplet, ensure_ascii=False) + '\n')
            print(f"已保存 {domain}: {output_file}")
    print("\n处理完成！")

if __name__ == "__main__":
    process()