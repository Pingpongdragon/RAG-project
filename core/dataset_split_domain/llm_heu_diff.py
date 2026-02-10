import json
import os
import random
import pandas as pd
import re
from pathlib import Path
from tqdm import tqdm
from typing import Optional
import sys

# 添加项目根目录到 Python 路径
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# 导入你的本地推理模块
from RAG_project.config import settings
from RAG_project.config.logger_config import configure_console_logger
from swift.llm import InferRequest, RequestConfig

logger = configure_console_logger(__name__)

# ================= 配置 =================
SAMPLE_SIZE = 100  # 抽取多少个文档进行对比
DEBUG_MODE = True  # 🔧 打开调试模式，查看模型原始输出
DEBUG_ALL_SAMPLES = False  # 🔧 只看前3个样本

# 你的原始关键词逻辑
KEYWORDS = {
    0: ["music", "movie", "tv", "film", "actor", "actress", "celebrity", "game", "comic", "fiction", "beatles", "pop", "song", "album", "band", "xbox", "nintendo"],
    1: ["science", "technology", "physics", "biology", "chemistry", "computer", "internet", "space", "nasa", "machine", "robot", "species", "formula", "theory", "software", "engineering"],
    2: ["history", "politics", "war", "battle", "army", "empire", "king", "queen", "president", "minister", "art", "literature", "writer", "philosophy", "religion", "democracy", "dynasty"],
    3: ["sport", "football", "basketball", "baseball", "olympic", "league", "team", "coach", "food", "cooking", "fashion", "travel", "pet", "hobby", "garden", "car", "fitness"],
}
DOMAIN_NAMES = {0: "Entertainment", 1: "STEM", 2: "Humanities", 3: "Lifestyle",4: "Unknown"}

# ================= 分类器定义 =================

def heuristic_classify(text):
    """你的原始关键词分类逻辑"""
    if not text: return 4
    t = text.lower()
    scores = {k: sum(1 for kw in kws if kw in t) for k, kws in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else 4

class LocalLLMClassifier:
    """使用本地模型的分类器 (基于你的项目架构)"""
    def __init__(self):
        logger.info("🔌 Initializing local model engine...")
        try:
            self.engine = settings.model_manager.get_engine(
                settings.MODEL_DIR, 
                settings.ADAPTER_DIR, 
                settings.MODEL_TYPE
            )
            logger.info(f"✅ Local model loaded: {settings.MODEL_DIR}")
            self.debug_count = 0
            self.failed_extractions = []  # 记录所有提取失败的样本
        except Exception as e:
            logger.error(f"❌ Failed to load local model: {e}")
            raise

    def classify(self, text: str) -> Optional[int]:
        """分类单个文本 - 修复版"""
        # 🔧 优化 Prompt：明确要求跳过思维过程,直接输出数字
        prompt = f"""Classify this text into one category. Reply with ONLY the number (0, 1, 2, 3 or 4). Do NOT include any explanations or additional text.

0 = Entertainment
1 = STEM  
2 = Humanities
3 = Lifestyle
4 = Unknown

Text: "{text[:300]}"

Category number:"""

        try:
            infer_request = [InferRequest(messages=[{'role': 'user', 'content': prompt}])]
            request_config = RequestConfig(
                max_tokens=1024,  
                temperature=0.0
            )
            resp_list = self.engine.infer(infer_request, request_config)
            raw_response = resp_list[0].choices[0].message.content
            
            # 🔍 调试：只打印前3个样本
            if DEBUG_ALL_SAMPLES or self.debug_count < 3:
                logger.info(f"\n{'='*60}")
                logger.info(f"🔍 Sample {self.debug_count + 1}")
                logger.info(f"📝 Input: {text[:80]}...")
                logger.info(f"🤖 Raw Output: {repr(raw_response)}")
            
            # 后处理：提取数字
            cleaned = self._extract_digit(raw_response, text[:100])
            
            if DEBUG_ALL_SAMPLES or self.debug_count < 3:
                logger.info(f"✅ Extracted: {cleaned}")
                logger.info(f"{'='*60}")
            
            self.debug_count += 1
            return cleaned if cleaned is not None else 4  # 🔧 提取失败时返回 4
            
        except Exception as e:
            logger.error(f"⚠️ LLM Classification Error: {e}")
            return 4  # 🔧 异常时返回 4

    def _extract_digit(self, raw_text: str, sample_text: str = "") -> Optional[int]:
        """从模型输出中提取数字 (0-4) - 处理 CoT 输出"""
        if not raw_text:
            return None
        
        original_text = raw_text
        
        # Step 1: 🔧 移除 <think>...</think> 标签及其内容
        raw_text = re.sub(r'<think>.*?</think>', '', raw_text, flags=re.DOTALL)
        
        # Step 2: 移除其他结束标签
        for tag in ['<|im_end|>', '<|endoftext|>']:
            index = raw_text.find(tag)
            if index != -1:
                raw_text = raw_text[index + len(tag):]
        
        # Step 3: 移除常见前缀
        cleaned = re.sub(r'^(Answer|Response|Output|Result|Classification|Category number)[\s:：]*', '', raw_text, flags=re.IGNORECASE)
        
        # Step 4: 多模式匹配
        patterns = [
            r'^[^\d]*([0-4])[^\d]*$',  # 单独的数字
            r'\b([0-4])\b',            # 单词边界
            r'([0-4])',                # 任意位置
        ]
        
        for pattern in patterns:
            match = re.search(pattern, cleaned.strip())
            if match:
                digit = int(match.group(1))
                return digit
        
        # Step 5: 失败记录
        if DEBUG_MODE and self.debug_count < 3:
            logger.warning(f"   ❌ Failed to extract from: {repr(cleaned[:100])}")
        
        self.failed_extractions.append({
            'sample': sample_text,
            'raw_output': original_text,
            'cleaned_output': cleaned
        })
        
        return None
    
    def print_diagnostic_summary(self):
        """打印诊断摘要"""
        if self.failed_extractions:
            logger.warning(f"\n{'='*60}")
            logger.warning(f"📋 DIAGNOSTIC SUMMARY: {len(self.failed_extractions)} Failed Extractions")
            logger.warning(f"{'='*60}")
            
            for i, failure in enumerate(self.failed_extractions[:5], 1):
                logger.warning(f"\n[Failed Case {i}]")
                logger.warning(f"Sample: {failure['sample']}")
                logger.warning(f"Raw: {repr(failure['raw_output'][:150])}")
                logger.warning(f"Cleaned: {repr(failure['cleaned_output'][:100])}")
                logger.warning("-" * 60)

# ================= 主流程 =================

def run_diff_check():
    # 1. 加载 HotpotQA 数据
    logger.info("📥 Loading HotpotQA samples...")
    from datasets import load_dataset
    for k in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"]:
        os.environ.pop(k, None)
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    
    dataset = load_dataset("hotpot_qa", "distractor", split="train", streaming=True)
    
    samples = []
    iterator = iter(dataset)
    
    logger.info(f"🔄 Extracting {SAMPLE_SIZE} documents...")
    with tqdm(total=SAMPLE_SIZE, desc="Loading samples") as pbar:
        while len(samples) < SAMPLE_SIZE:
            try:
                item = next(iterator)
                context = item.get("context", {})
                titles = context.get("title", [])
                sentences_list = context.get("sentences", [])
                
                if titles and sentences_list:
                    title = titles[0]
                    text = " ".join(sentences_list[0])
                    full_text = f"{title}: {text}"
                    
                    if len(full_text) > 50:
                        samples.append(full_text)
                        pbar.update(1)
            except StopIteration:
                break

    # 2. 初始化分类器
    llm_classifier = LocalLLMClassifier()
    results = []
    
    logger.info(f"\n🚀 Comparing Classifiers on {len(samples)} docs...")
    logger.info(f"   Using model: {settings.MODEL_DIR}\n")
    
    # 3. 运行对比
    for text in tqdm(samples, desc="Classifying"):
        res_heu = heuristic_classify(text)
        res_llm = llm_classifier.classify(text)
        match = (res_heu == res_llm)
        
        results.append({
            "text_snippet": text[:100] + "...",
            "heuristic": DOMAIN_NAMES.get(res_heu, "Unknown"),
            "llm": DOMAIN_NAMES.get(res_llm, "Unknown"),
            "heuristic_id": res_heu,
            "llm_id": res_llm,
            "match": match,
            "full_text": text
        })

    # 4. 打印诊断摘要
    llm_classifier.print_diagnostic_summary()

    # 5. 输出报告
    df = pd.DataFrame(results)
    total = len(df)
    matches = df['match'].sum()
    mismatches = total - matches
    agreement_rate = matches / total if total > 0 else 0
    
    print("\n" + "="*60)
    print(f"📊 Classification Comparison Report")
    print("="*60)
    print(f"Sample Size:      {total}")
    print(f"✅ Agreement:     {matches} ({agreement_rate:.1%})")
    print(f"❌ Disagreements: {mismatches}")
    print("="*60)
    
    # 6. 展示差异样本
    if mismatches > 0:
        print("\n👀 --- Top 10 Disagreements (Heuristic vs LLM) ---\n")
        diff_df = df[df['match'] == False].head(10)
        
        for idx, (_, row) in enumerate(diff_df.iterrows(), 1):
            print(f"[Case {idx}]")
            print(f"📄 Text: {row['text_snippet']}")
            print(f"🔴 Heuristic → {row['heuristic']}")
            print(f"🟢 LLM       → {row['llm']}")
            print("-" * 60)
    
    # 7. 领域分布对比
    print("\n📈 --- Domain Distribution Comparison ---")
    print(f"{'Domain':<15} | {'Heuristic':<10} | {'LLM':<10} | {'Difference':<10}")
    print("-" * 60)
    for dom_id, dom_name in DOMAIN_NAMES.items():
        heu_count = (df['heuristic_id'] == dom_id).sum()
        llm_count = (df['llm_id'] == dom_id).sum()
        diff = llm_count - heu_count
        print(f"{dom_name:<15} | {heu_count:<10} | {llm_count:<10} | {diff:+<10}")
    
    # 8. 保存结果
    output_file = Path(__file__).parent / "kb_diff_results.csv"
    df.to_csv(output_file, index=False, encoding='utf-8')
    
    if llm_classifier.failed_extractions:
        failed_log_file = Path(__file__).parent / "failed_extractions.json"
        with open(failed_log_file, 'w', encoding='utf-8') as f:
            json.dump(llm_classifier.failed_extractions, f, indent=2, ensure_ascii=False)
        logger.info(f"💾 Failed extraction details: {failed_log_file}")
    
    logger.info(f"💾 Full results: {output_file}")
    
    # 9. 统计 Unknown
    heu_unknown = (df['heuristic'] == 'Unknown').sum()
    llm_unknown = (df['llm'] == 'Unknown').sum()
    if heu_unknown > 0 or llm_unknown > 0:
        print(f"\n⚠️  Unknown classifications:")
        print(f"   Heuristic: {heu_unknown}")
        print(f"   LLM:       {llm_unknown}")

if __name__ == "__main__":
    run_diff_check()