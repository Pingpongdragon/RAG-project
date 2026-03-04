import json
import random
import time
import os
import re
from typing import List, Dict, Any
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset

# ==========================================
# 1. Configuration (Local Qwen)
# ==========================================

client = OpenAI(
    api_key="none",
    base_url="http://202.45.128.234:5788/v1/"
)
MODEL_NAME = "/nfs/whlu/models/Qwen3-Coder-30B-A3B-Instruct"

DOMAIN_LABELS = ["entertainment", "stem", "humanities", "lifestyle"]

SYSTEM_PROMPT = """You are an advanced data annotator for a RAG router. 
Your task is to classify user queries into specific knowledge domains.
You must output the result in a strict JSON format."""

# ==========================================
# Prompt Templates
# ==========================================

# 用于 WoW 标注：利用全部信息（topic + knowledge + context + response）
WOW_LABEL_PROMPT_TEMPLATE = """
Analyze the following user query and determine the probability distribution across these 4 domains:

1. **Entertainment**: Movies, music, celebrities, video games, comics, fictional books.
2. **STEM**: Science, technology, engineering, mathematics, physics, biology, computer science, software.
3. **Humanities**: History, philosophy, religion, literature, art, social studies, war, ancient/modern history.
4. **Lifestyle**: Sports, food/cooking, travel, fashion, cars/vehicles, pets, hobbies, health/fitness.

**Conversation Topic:** "{topic}"
**Conversation Context:** {context}
**Knowledge Sources Used:** {knowledge_summary}
**User Query to classify:** "{query}"

**Instructions:**
1. Use ALL the above information (topic, context, knowledge) to deeply understand the domain, but classify **based on the user query itself**.
2. Assign a probability (float between 0.0 and 1.0) to each domain.
3. The sum of all probabilities **must equal 1.0**.
4. **Capture Uncertainty**: If the query is ambiguous or spans multiple domains, distribute the probability mass (e.g., [0.45, 0.05, 0.45, 0.05]). 
5. Output strictly in this JSON format:
{{
    "probabilities": [P_entertainment, P_stem, P_humanities, P_lifestyle],
    "reasoning": "A short explanation"
}}
"""

# 用于 WoW 生成 context summary：生成超简短的上下文摘要（最多15个词）
WOW_SUMMARY_PROMPT_TEMPLATE = """
Given the following multi-turn conversation, generate an EXTREMELY CONCISE context summary (max 15 words) that captures ONLY the key topic.

**Conversation Topic:** "{topic}"
**Recent Posts (last 3):** {recent_posts}
**Latest User Query:** "{query}"

**Instructions:**
1. Generate a summary in 10-15 words maximum.
2. Focus ONLY on the main topic, NOT specific details.
3. Do NOT repeat the user query.
4. Examples:
   - "Discussion about red hair genetics and frequency"
   - "Conversation on internet access benefits"
   - "Talk about science fiction and time travel"
5. Output strictly in this JSON format:
{{
    "summary": "A 10-15 word summary"
}}
"""

# 用于 HotpotQA（有 answer + supporting_facts 辅助标注）
HOTPOT_PROMPT_TEMPLATE = """
Analyze the following user query and determine the probability distribution across these 4 domains:

1. **Entertainment**: Movies, music, celebrities, video games, comics, fictional books.
2. **STEM**: Science, technology, engineering, mathematics, physics, biology, computer science, software.
3. **Humanities**: History, philosophy, religion, literature, art, social studies, war, ancient/modern history.
4. **Lifestyle**: Sports, food/cooking, travel, fashion, cars/vehicles, pets, hobbies, health/fitness.

**User Query:** "{query}"
**Answer:** "{answer}"
**Related Topics:** {supporting_facts}

**Instructions:**
1. Use the answer and related topics as hints to understand the user query's domain, but classify **based on the user query itself**.
2. Assign a probability (float between 0.0 and 1.0) to each domain.
3. The sum of all probabilities **must equal 1.0**.
4. **Capture Uncertainty**: If the query is ambiguous or spans multiple domains, distribute the probability mass (e.g., [0.45, 0.05, 0.45, 0.05]). 
5. Output strictly in this JSON format:
{{
    "probabilities": [P_entertainment, P_stem, P_humanities, P_lifestyle],
    "reasoning": "A short explanation"
}}
"""

# ==========================================
# 2. Data Loading (Mixed WoW + HotpotQA)
# ==========================================

def extract_knowledge_summary(knowledge_lists):
    """
    从 WoW 的 knowledge 字段中提取关键知识摘要。
    knowledge 格式: List[List[str]]，每轮对话对应一组知识条目。
    """
    key_facts = []
    seen = set()
    for turn_knowledge in knowledge_lists:
        for entry in turn_knowledge:
            if not isinstance(entry, str):
                continue
            # 跳过 no_passages_used
            if "no_passages_used" in entry:
                continue
            # 格式: "Topic __knowledge__ fact text"
            if "__knowledge__" in entry:
                parts = entry.split("__knowledge__", 1)
                if len(parts) == 2:
                    fact = parts[1].strip()
                    # 去重 + 限制长度
                    if fact and fact not in seen and len(fact) > 20:
                        seen.add(fact)
                        key_facts.append(fact)
            if len(key_facts) >= 5:  # 最多取5条关键知识
                break
        if len(key_facts) >= 5:
            break
    return key_facts


def load_mixed_data(sample_size=2000):
    queries = []
    print(f"📥 Loading datasets (Target total: {sample_size})...")
    
    samples_per_dataset = sample_size // 2
    
    # Load Wizard of Wikipedia
    try:
        print(f"  Loading Wizard of Wikipedia (target: {samples_per_dataset})...")
        ds_wow = load_dataset("chujiezheng/wizard_of_wikipedia", split="train", streaming=True)
        it_wow = iter(ds_wow)
        wow_count = 0
        
        for _ in range(samples_per_dataset * 5):
            if wow_count >= samples_per_dataset:
                break
            try:
                item = next(it_wow)
                
                if 'post' not in item or not item['post']:
                    continue
                
                posts = item['post']
                if not isinstance(posts, list) or len(posts) == 0:
                    continue
                
                # 提取 topic（post[0] 通常包含主题词）
                raw_topic = posts[0].strip() if isinstance(posts[0], str) else ""
                topic = raw_topic.split("\n")[0].strip() if raw_topic else ""
                
                # 提取用户发言：从 post[1:] 中取第一个完整句子作为 query
                user_query = None
                query_turn_idx = -1
                for idx, post in enumerate(posts[1:], start=1):
                    if isinstance(post, str) and len(post.strip()) > 15 and ' ' in post.strip():
                        user_query = post.strip()
                        query_turn_idx = idx
                        break
                
                if not user_query:
                    continue
                
                # 只保留最近3轮对话作为上下文（压缩）
                context_posts = []
                start_idx = max(0, query_turn_idx - 3)
                for p in posts[start_idx:query_turn_idx]:
                    if isinstance(p, str) and len(p.strip()) > 0:
                        # 截断每条 post 到最多 80 字符
                        truncated = p.strip()[:80]
                        context_posts.append(truncated)
                
                # 提取 knowledge 摘要（辅助标注用）
                knowledge = item.get('knowledge', [])
                knowledge_summary = extract_knowledge_summary(knowledge)
                
                # 提取 topics 列表
                topics_list = item.get('topics', [])
                
                queries.append({
                    "text": user_query,                    # 用户查询
                    "topic": topic,                        # 对话主题
                    "context": context_posts,              # 最近3轮的对话上下文（已截断）
                    "all_posts": [p.strip() for p in posts if isinstance(p, str) and len(p.strip()) > 0],
                    "knowledge_summary": knowledge_summary, # 关键知识条目（辅助标注）
                    "topics_list": topics_list,            # 全部主题标签
                    "source": "wizard_of_wikipedia"
                })
                wow_count += 1
                    
            except StopIteration:
                print(f"  ⚠️ WoW exhausted at {wow_count} samples")
                break
            except Exception as e:
                continue
                
        print(f"  ✓ Loaded {wow_count} samples from Wizard of Wikipedia")
    except Exception as e:
        print(f"⚠️ Wizard of Wikipedia load error: {e}")

    # Load HotpotQA
    try:
        print(f"  Loading HotpotQA (target: {samples_per_dataset})...")
        ds_hotpot = load_dataset("hotpot_qa", "distractor", split="train", streaming=True)
        it_hotpot = iter(ds_hotpot)
        hotpot_count = 0
        
        for _ in range(samples_per_dataset * 2):
            if hotpot_count >= samples_per_dataset:
                break
            try:
                item = next(it_hotpot)
                if 'question' in item and len(item['question'].strip()) > 10:
                    answer = item.get('answer', '').strip()
                    
                    supporting_facts = []
                    if 'supporting_facts' in item and item['supporting_facts']:
                        sf = item['supporting_facts']
                        if isinstance(sf, dict) and 'title' in sf:
                            supporting_facts = list(set(sf['title']))
                        elif isinstance(sf, list):
                            for fact in sf:
                                if isinstance(fact, (list, tuple)) and len(fact) > 0:
                                    supporting_facts.append(fact[0])
                            supporting_facts = list(set(supporting_facts))
                    
                    queries.append({
                        "text": item['question'].strip(),
                        "answer": answer,
                        "supporting_facts": supporting_facts,
                        "source": "hotpot_qa"
                    })
                    hotpot_count += 1
            except StopIteration:
                print(f"  ⚠️ HotpotQA exhausted at {hotpot_count} samples")
                break
            except Exception as e:
                continue
                
        print(f"  ✓ Loaded {hotpot_count} samples from HotpotQA")
    except Exception as e:
        print(f"⚠️ HotpotQA load error: {e}")

    print(f"\n📊 Total queries loaded: {len(queries)}")
    print(f"   - Wizard of Wikipedia: {sum(1 for q in queries if q['source'] == 'wizard_of_wikipedia')}")
    print(f"   - HotpotQA: {sum(1 for q in queries if q['source'] == 'hotpot_qa')}")
    
    random.shuffle(queries)
    return queries

# ==========================================
# 3. Qwen Labeler
# ==========================================

class QwenLabeler:
    
    def _call_llm(self, prompt, temperature=0.1):
        """统一的 LLM 调用接口"""
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
        )
        content = response.choices[0].message.content
        
        # 提取 JSON
        json_str = content
        if "```json" in content:
            match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
            if match:
                json_str = match.group(1)
        elif "{" in content:
            json_str = content[content.find("{"):content.rfind("}")+1]
        
        return json.loads(json_str)
    
    def generate_wow_summary(self, item):
        """
        为 WoW 多轮对话生成超简短的上下文摘要（10-15词）。
        这个 summary 会被拼接到 text 中，用于蒸馏训练。
        """
        topic = item.get("topic", "")
        context = item.get("context", [])  # 已经是最近3轮的截断版本
        query = item["text"]
        
        # 只传递最近3轮给 LLM（节省 token）
        recent_posts_str = json.dumps(context[-3:] if len(context) > 3 else context, ensure_ascii=False)
        
        prompt = WOW_SUMMARY_PROMPT_TEMPLATE.format(
            topic=topic,
            recent_posts=recent_posts_str,
            query=query
        )
        
        for attempt in range(3):
            try:
                data = self._call_llm(prompt, temperature=0.3)
                summary = data.get("summary", "").strip()
                # 验证长度（最多20词作为硬限制）
                if summary and len(summary.split()) <= 20:
                    return summary
            except Exception as e:
                if attempt == 2:
                    print(f"\n⚠️ Summary generation failed for '{query[:30]}...': {e}")
                time.sleep(0.3)
        
        # 兜底：用 topic 做简单 summary（5-8词）
        if topic:
            return f"Discussion about {topic}"
        return ""
    
    def annotate(self, item):
        """
        标注一条数据。
        
        对于 WoW 数据：
          1. 利用全部信息（topic, context, knowledge）做高质量标注
          2. 额外生成简短 context summary（10-15词），拼成 distill_text
        
        对于 HotpotQA 数据：
          1. 利用 answer + supporting_facts 做辅助标注
          2. distill_text 就是 query 本身（单轮无上下文）
        """
        text = item["text"]
        
        # ========== Step 1: 域标注 ==========
        if item["source"] == "wizard_of_wikipedia":
            topic = item.get("topic", "")
            context = item.get("context", [])
            knowledge_summary = item.get("knowledge_summary", [])
            
            context_str = json.dumps(context, ensure_ascii=False)
            knowledge_str = json.dumps(knowledge_summary[:5], ensure_ascii=False) if knowledge_summary else "[]"
            
            prompt = WOW_LABEL_PROMPT_TEMPLATE.format(
                topic=topic,
                context=context_str,
                knowledge_summary=knowledge_str,
                query=text
            )
        elif item["source"] == "hotpot_qa":
            answer = item.get("answer", "")
            supporting_facts = item.get("supporting_facts", [])
            supporting_facts_str = json.dumps(supporting_facts, ensure_ascii=False)
            prompt = HOTPOT_PROMPT_TEMPLATE.format(
                query=text,
                answer=answer,
                supporting_facts=supporting_facts_str
            )
        else:
            prompt = f"""
Classify the following query into domains: Entertainment, STEM, Humanities, Lifestyle.
Query: "{text}"
Output JSON: {{"probabilities": [P_ent, P_stem, P_hum, P_life], "reasoning": "..."}}
"""
        
        # 标注概率
        label_result = None
        for attempt in range(3):
            try:
                data = self._call_llm(prompt)
                probs = data.get("probabilities")
                
                if not probs or len(probs) != 4:
                    continue
                
                total = sum(probs)
                norm_probs = [round(float(p)/total, 4) if total > 0 else 0.25 for p in probs]
                hard_label_idx = norm_probs.index(max(norm_probs))
                
                label_result = {
                    "teacher_probs": norm_probs,
                    "hard_label": hard_label_idx,
                    "label_name": DOMAIN_LABELS[hard_label_idx],
                    "reasoning": data.get("reasoning", "")
                }
                break
                
            except Exception as e:
                if attempt == 2:
                    print(f"\n❌ Label error for '{text[:30]}...': {e}")
                time.sleep(0.5)
        
        if not label_result:
            return None
        
        # ========== Step 2: 构建 distill_text ==========
        if item["source"] == "wizard_of_wikipedia":
            # 多轮对话：生成简短 summary（10-15词）+ query 拼接
            summary = self.generate_wow_summary(item)
            if summary:
                distill_text = f"[context summary : {summary}] {text}"
            else:
                distill_text = text
        else:
            # HotpotQA 单轮：直接用 query
            distill_text = text
        
        return {
            "text": distill_text,           # 蒸馏训练用的输入（含简短上下文摘要）
            "query": text,                  # 原始纯 query（备用）
            **label_result
        }

# ==========================================
# 4. Main Execution
# ==========================================

def main():
    OUTPUT_FILE = "train_distill_mixed_qwenv5.jsonl"
    
    TOTAL_SAMPLES = 10000  # 先测试，之后改大
    raw_data = load_mixed_data(sample_size=TOTAL_SAMPLES)
    
    if len(raw_data) == 0:
        print("❌ 没有加载到任何数据,请检查数据集连接!")
        return
    
    # 打印前5个样本查看效果
    print("\n📝 Sample queries:")
    for i, item in enumerate(raw_data[:5]):
        src = item['source']
        txt = item['text'][:60]
        if src == "wizard_of_wikipedia":
            extra = f"topic={item.get('topic', 'N/A')}, context_turns={len(item.get('context', []))}"
        else:
            extra = f"answer={item.get('answer', 'N/A')[:30]}, facts={item.get('supporting_facts', [])[:2]}"
        print(f"  {i+1}. [{src}] {extra}")
        print(f"      query: {txt}...")
    
    labeler = QwenLabeler()
    print(f"\n🚀 Starting labeling with Qwen at {MODEL_NAME}...")
    
    valid_count = 0
    wow_with_summary = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in tqdm(raw_data):
            res = labeler.annotate(item)
            if res:
                res["dataset_source"] = item["source"]
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
                valid_count += 1
                
                # 统计有 summary 的 WoW 样本
                if item["source"] == "wizard_of_wikipedia" and "[" in res["text"] and "]" in res["text"]:
                    wow_with_summary += 1
                
                if valid_count % 10 == 0:
                    f.flush()

    print(f"\n✅ Finished! Saved {valid_count} samples to {OUTPUT_FILE}")
    print(f"   - WoW samples with context summary: {wow_with_summary}")
    print(f"   - HotpotQA samples (query only): {valid_count - wow_with_summary}")

if __name__ == "__main__":
    main()