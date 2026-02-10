import json
import random
import time
import os
from typing import List, Dict, Any
from tqdm import tqdm
from datasets import load_dataset

# ==========================================
# 0. Google GenAI Setup
# ==========================================
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# 读取环境变量 (你在 CMD 中 export 的那些)
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    raise ValueError("请先设置 GOOGLE_API_KEY 环境变量！")

# 配置 Google GenAI
genai.configure(api_key=GOOGLE_API_KEY)

# ==========================================
# 1. Configuration & Constants
# ==========================================

# 推荐使用 gemini-1.5-flash (速度快、便宜、适合标注) 或 gemini-1.5-pro (更强)
MODEL_NAME = "gemini-1.5-flash" 

# ⚠️ 必须与 Student 模型训练时的顺序完全一致！
DOMAIN_LABELS = ["entertainment", "stem", "humanities", "lifestyle"]

# System Prompt
SYSTEM_PROMPT = """You are an advanced data annotator for a RAG router. 
Your task is to classify user queries into specific knowledge domains.
You must output the result in a strict JSON format."""

USER_PROMPT_TEMPLATE = """
Analyze the following user query and determine the probability distribution across these 4 domains:

1. **Entertainment**: Movies, music, celebrities, video games, comics, fictional books.
2. **STEM**: Science, technology, engineering, mathematics, physics, biology, computer science, software.
3. **Humanities**: History, politics, philosophy, religion, literature, art, social studies, war.
4. **Lifestyle**: Sports, food/cooking, travel, fashion, cars/vehicles, pets, hobbies, health/fitness.

**Input Query:** "{query}"

**Instructions:**
1. Assign a probability (float between 0.0 and 1.0) to each domain.
2. The sum of all probabilities **must equal 1.0**.
3. **Capture Uncertainty**: If the query is ambiguous (e.g., "Sci-Fi Movie History"), distribute probabilities (e.g., [0.45, 0.05, 0.45, 0.05]).
4. Output strictly in the following JSON format:

{{
    "probabilities": [P_entertainment, P_stem, P_humanities, P_lifestyle],
    "reasoning": "A short explanation in English"
}}
"""

# ==========================================
# 2. Data Loading (HotpotQA + WoW)
# ==========================================
# (这部分代码保持不变，为了完整性我保留在这里)

def load_mixed_data(sample_size=200):
    queries = []
    print("📥 Loading HotpotQA (questions)...")
    try:
        ds_hotpot = load_dataset("hotpot_qa", "distractor", split="train", streaming=True)
        iterator = iter(ds_hotpot)
        for _ in range(sample_size // 2):
            item = next(iterator)
            queries.append({"text": item['question'], "source": "hotpot_qa"})
    except Exception as e:
        print(f"⚠️ Failed to load HotpotQA: {e}")

    print("📥 Loading Wizard of Wikipedia (dialogues)...")
    try:
        ds_wow = load_dataset("chujiezheng/wizard_of_wikipedia", split="train", streaming=True)
        iterator = iter(ds_wow)
        for _ in range(sample_size // 2):
            item = next(iterator)
            topic = item.get("chosen_topic", "")
            first_msg = ""
            history = item.get("history", [])
            if not history and 'dialog' in item:
                 history = item['dialog']
            if history and len(history) > 0:
                first_msg = history[0].get("text", "") if isinstance(history[0], dict) else history[0]

            query_text = first_msg if (first_msg and len(first_msg) > 10 and random.random() > 0.3) else f"Tell me about {topic}"
            queries.append({"text": query_text, "source": "wow"})
    except Exception as e:
        print(f"⚠️ Failed to load WoW: {e}")

    random.shuffle(queries)
    print(f"✅ Total queries prepared: {len(queries)}")
    return queries

# ==========================================
# 3. LLM Annotator (Google Gemini Implementation)
# ==========================================

class LLMLabeler:
    def __init__(self):
        # 初始化模型
        # generation_config 用于强制 JSON 输出
        self.model = genai.GenerativeModel(
            model_name=MODEL_NAME,
            system_instruction=SYSTEM_PROMPT,
            generation_config={"response_mime_type": "application/json"}
        )
        
        # 安全设置：关掉安全过滤，防止误杀普通 Query
        self.safety_settings = {
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

    def annotate(self, text):
        prompt = USER_PROMPT_TEMPLATE.format(query=text)
        
        for attempt in range(3):
            try:
                # 调用 Gemini API
                response = self.model.generate_content(
                    prompt, 
                    safety_settings=self.safety_settings
                )
                
                # 解析 JSON
                data = json.loads(response.text)
                probs = data.get("probabilities")
                
                # 验证数据
                if not probs or len(probs) != 4:
                    continue
                
                # 归一化
                total = sum(probs)
                if total == 0: continue
                norm_probs = [float(p)/total for p in probs]
                
                # 确定 Hard Label
                hard_label_idx = norm_probs.index(max(norm_probs))
                
                return {
                    "text": text,
                    "teacher_probs": norm_probs,
                    "hard_label": hard_label_idx,
                    "label_name": DOMAIN_LABELS[hard_label_idx],
                    "reasoning": data.get("reasoning", "")
                }
                
            except Exception as e:
                # print(f"Error: {e}") # 调试时打开
                time.sleep(1) # 遇到 Rate Limit 等待一下
        return None

# ==========================================
# 4. Main Execution
# ==========================================

def main():
    OUTPUT_FILE = "train_distill_mixed_gemini.jsonl"
    
    # 获取数据 (示例取 20 条，你可以改大)
    raw_data = load_mixed_data(sample_size=20) 
    
    labeler = LLMLabeler()
    
    print(f"🚀 Starting annotation using Google {MODEL_NAME}...")
    
    valid_count = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for item in tqdm(raw_data):
            text = item["text"]
            result = labeler.annotate(text)
            
            if result:
                result["dataset_source"] = item["source"]
                f.write(json.dumps(result, ensure_ascii=False) + "\n")
                valid_count += 1
    
    print(f"\n✅ Annotation complete! {valid_count}/{len(raw_data)} samples saved to {OUTPUT_FILE}")
    if valid_count > 0:
        print("Sample output:")
        with open(OUTPUT_FILE, "r") as f:
            print(f.readline())

if __name__ == "__main__":
    main()