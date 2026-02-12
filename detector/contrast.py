import json
import random
import time
import os
import re
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from tqdm import tqdm
from openai import OpenAI
from datasets import load_dataset
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# LLM API 配置（改为本地 Qwen-30B）
client = OpenAI(
    api_key="none",
    base_url="http://202.45.128.234:5788/v1/",
    timeout=60.0,
    max_retries=2
)
MODEL_NAME = "/nfs/whlu/models/Qwen3-Coder-30B-A3B-Instruct"

# 蒸馏模型配置
DISTILLED_MODEL_PATH = "/home/jyliu/labeling/detector/mini_router_best"  # ⭐ 改为绝对路径

# 领域标签
DOMAIN_LABELS = ["entertainment", "stem", "humanities", "lifestyle"]

# 关键词字典
KEYWORDS = {
    0: ["music", "movie", "tv", "film", "actor", "actress", "celebrity", "game", "comic", 
        "fiction", "beatles", "pop", "song", "album", "band", "xbox", "nintendo"],
    1: ["science", "technology", "physics", "biology", "chemistry", "computer", "internet", 
        "space", "nasa", "machine", "robot", "species", "formula", "theory", "software", "engineering"],
    2: ["history", "politics", "war", "battle", "army", "empire", "king", "queen", "president", 
        "minister", "art", "literature", "writer", "philosophy", "religion", "democracy", "dynasty"],
    3: ["sport", "football", "basketball", "baseball", "olympic", "league", "team", "coach", 
        "food", "cooking", "fashion", "travel", "pet", "hobby", "garden", "car", "fitness"]
}

DOMAIN_NAMES = {0: "0_entertainment", 1: "1_stem", 2: "2_humanities", 3: "3_lifestyle"}

# Prompt 配置
SYSTEM_PROMPT = """You are an advanced data annotator for a RAG router. 
Your task is to classify user queries into specific knowledge domains.
You must output the result in a strict JSON format."""

USER_PROMPT_TEMPLATE = """
Analyze the following user query and determine the probability distribution across these 4 domains:

1. **Entertainment**: Movies, music, celebrities, video games, comics, fictional books.
2. **STEM**: Science, technology, engineering, mathematics, physics, biology, computer science, software.
3. **Humanities**: History, philosophy, religion, literature, art, social studies, war, ancient/modern history.
4. **Lifestyle**: Sports, food/cooking, travel, fashion, cars/vehicles, pets, hobbies, health/fitness.

**Input Query:** "{query}"

**Instructions:**
1. Assign a probability (float between 0.0 and 1.0) to each domain.
2. The sum of all probabilities **must equal 1.0**.
3. **Capture Uncertainty**: If the query is ambiguous or spans multiple domains, distribute the probability mass (e.g., [0.45, 0.05, 0.45, 0.05]). 
4. Output strictly in this JSON format:
{{
    "probabilities": [P_entertainment, P_stem, P_humanities, P_lifestyle],
    "reasoning": "A short explanation"
}}
"""

# 测试配置
TEST_SAMPLE_SIZE = 500  # 测试样本数量
REFERENCE_SAMPLE_SIZE = 500  # ⭐ 改为全部数据都生成标签
MAX_WORKERS = 10  # 并发线程数

# ==========================================
# 2. 数据加载
# ==========================================

def load_test_data(sample_size: int, split: str = "validation") -> List[Dict[str, Any]]:
    """加载 HotpotQA 测试数据"""
    print(f"📥 正在从 HotpotQA {split} 集加载 {sample_size} 条数据...")
    
    queries = []
    try:
        ds = load_dataset("hotpot_qa", "distractor", split=split, streaming=True)
        ds_iter = iter(ds)
        
        for i in range(sample_size):
            try:
                item = next(ds_iter)
                queries.append({
                    "text": item['question'],
                    "id": i,
                    "answer": item.get('answer', ''),
                    "type": item.get('type', '')
                })
            except StopIteration:
                print(f"⚠️ 数据集只有 {len(queries)} 条数据")
                break
                
    except Exception as e:
        print(f"❌ 数据加载错误: {e}")
    
    print(f"✅ 成功加载 {len(queries)} 条数据")
    return queries

# ==========================================
# 3. 关键词分类器
# ==========================================

class KeywordClassifier:
    """基于关键词的简单分类器"""
    
    def __init__(self):
        self.keywords = KEYWORDS
        self.domain_names = DOMAIN_NAMES
    
    def get_domain(self, text: str) -> Optional[int]:
        """返回预测的领域索引"""
        if not text:
            return None
        
        t = text.lower()
        scores = {k: sum(1 for kw in kws if kw in t) for k, kws in self.keywords.items()}
        best = max(scores, key=scores.get)
        return best if scores[best] > 0 else None
    
    def annotate(self, text: str) -> Tuple[List[float], int, float, str]:
        """
        返回: (概率分布, 预测标签索引, 推理时间, 推理原因)
        """
        start_time = time.time()
        
        t = text.lower()
        scores = {k: sum(1 for kw in kws if kw in t) for k, kws in self.keywords.items()}
        
        # 计算概率分布
        total_score = sum(scores.values())
        if total_score == 0:
            probs = [0.25, 0.25, 0.25, 0.25]
            predicted_idx = 0
            reasoning = "No keywords matched, default to entertainment"
        else:
            probs = [scores[i] / total_score for i in range(4)]
            predicted_idx = max(scores, key=scores.get)
            matched_kws = [kw for kw in self.keywords[predicted_idx] if kw in t]
            reasoning = f"Matched keywords: {', '.join(matched_kws[:3])}"
        
        inference_time = time.time() - start_time
        return probs, predicted_idx, inference_time, reasoning

# ==========================================
# 4. LLM 标注器（优化版：支持并发）
# ==========================================

class LLMAnnotator:
    """LLM API 标注器（优化版：支持并发处理）"""
    
    def __init__(self, max_workers: int = 10):
        self.client = client
        self.model_name = MODEL_NAME
        self.max_workers = max_workers
    
    def annotate(self, text: str) -> Tuple[List[float], int, float, str]:
        """
        单个请求（优化版）
        返回: (概率分布, 预测标签索引, 推理时间, 推理原因)
        """
        prompt = USER_PROMPT_TEMPLATE.format(query=text)
        start_time = time.time()
        
        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.0,  # ⭐ 改为 0.0 确保确定性输出
                timeout=30
            )
            
            content = response.choices[0].message.content
            
            # 提取 JSON
            json_str = content
            if "```json" in content:
                match = re.search(r'```json\n(.*?)\n```', content, re.DOTALL)
                if match:
                    json_str = match.group(1)
            elif "{" in content:
                json_str = content[content.find("{"):content.rfind("}")+1]
            
            data = json.loads(json_str)
            probs = data.get("probabilities", [])
            
            if not probs or len(probs) != 4:
                probs = [0.25, 0.25, 0.25, 0.25]
                predicted_idx = 0
                reasoning = "Invalid format, using default"
            else:
                # 归一化
                total = sum(probs)
                probs = [round(float(p)/total, 4) if total > 0 else 0.25 for p in probs]
                predicted_idx = probs.index(max(probs))
                reasoning = data.get("reasoning", "")
            
            inference_time = time.time() - start_time
            return probs, predicted_idx, inference_time, reasoning
            
        except Exception as e:
            inference_time = time.time() - start_time
            error_msg = f"Error: {str(e)[:50]}"
            return [0.25, 0.25, 0.25, 0.25], 0, inference_time, error_msg
    
    def annotate_batch(self, texts: List[str]) -> List[Tuple[List[float], int, float, str]]:
        """
        批量并发请求 - 核心优化
        """
        results = [None] * len(texts)
        failed_count = 0
        
        print(f"🚀 使用 {self.max_workers} 线程并发处理...")
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_idx = {
                executor.submit(self.annotate, text): idx 
                for idx, text in enumerate(texts)
            }
            
            for future in tqdm(as_completed(future_to_idx), 
                             total=len(texts), 
                             desc="批量LLM推理",
                             unit="样本"):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result(timeout=60)
                except Exception as e:
                    failed_count += 1
                    results[idx] = ([0.25, 0.25, 0.25, 0.25], 0, 0, f"Batch error: {str(e)[:30]}")
        
        if failed_count > 0:
            print(f"⚠️ {failed_count}/{len(texts)} 个样本处理失败")
        
        return results

# ==========================================
# 5. 蒸馏模型标注器
# ==========================================

class DistilledModelAnnotator:
    """蒸馏模型标注器"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = None
        
        print(f"🔄 正在加载蒸馏模型: {model_path}")
        
        # ⭐ 先检查路径是否存在
        if not os.path.exists(model_path):
            print(f"⚠️ 蒸馏模型路径不存在: {model_path}")
            print("🔄 使用模拟模型进行演示...")
            self.is_real_model = False
            return
        
        # 检查 config.json
        config_path = os.path.join(model_path, "config.json")
        if not os.path.exists(config_path):
            print(f"⚠️ 找不到配置文件: {config_path}")
            self.is_real_model = False
            return
        
        try:
            # ⭐ 直接导入 DistilBERT 相关类
            from transformers import (
                DistilBertTokenizer, 
                DistilBertForSequenceClassification,
                DistilBertConfig
            )
            import torch
            from torch.nn.functional import softmax
            
            print(f"📦 加载 DistilBERT 模型...")
            
            # 加载配置
            config = DistilBertConfig.from_pretrained(model_path)
            print(f"   - 模型类型: {config.model_type}")
            print(f"   - 标签数量: {config.num_labels}")
            print(f"   - 词表大小: {config.vocab_size}")
            
            # 加载 tokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained(model_path)
            print(f"   - Tokenizer 加载成功")
            
            # 加载模型
            self.model = DistilBertForSequenceClassification.from_pretrained(
                model_path,
                config=config
            )
            print(f"   - 模型加载成功")
            
            # 设置设备
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.model.to(self.device)
            self.model.eval()
            
            print(f"✅ 蒸馏模型加载成功，设备: {self.device}")
            self.is_real_model = True
            
        except ImportError as e:
            print(f"⚠️ 缺少依赖库: {e}")
            print(f"   请运行: pip install transformers torch")
            self.is_real_model = False
        except Exception as e:
            print(f"⚠️ 蒸馏模型加载失败: {e}")
            print(f"   错误类型: {type(e).__name__}")
            import traceback
            print(f"   详细错误:\n{traceback.format_exc()}")
            print("🔄 使用模拟模型进行演示...")
            self.is_real_model = False
    
    def annotate(self, text: str) -> Tuple[List[float], int, float, str]:
        """
        返回: (概率分布, 预测标签索引, 推理时间, 推理原因)
        """
        start_time = time.time()
        
        if not self.is_real_model:
            time.sleep(0.001)
            probs = [random.random() for _ in range(4)]
            total = sum(probs)
            norm_probs = [p/total for p in probs]
            predicted_idx = norm_probs.index(max(norm_probs))
            inference_time = time.time() - start_time
            return norm_probs, predicted_idx, inference_time, "Simulated"
        
        try:
            import torch
            from torch.nn.functional import softmax
            
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=512
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = softmax(outputs.logits, dim=-1).cpu().numpy()[0]
            
            predicted_idx = np.argmax(probs)
            inference_time = time.time() - start_time
            
            return probs.tolist(), int(predicted_idx), inference_time, "Model prediction"
            
        except Exception as e:
            print(f"❌ 推理错误: {e}")
            inference_time = time.time() - start_time
            return [0.25, 0.25, 0.25, 0.25], 0, inference_time, "Error"
    
    def annotate_batch(self, texts: List[str]) -> List[Tuple[List[float], int, float, str]]:
        """
        批量推理（加速蒸馏模型）
        """
        if not self.is_real_model:
            return [self.annotate(text) for text in texts]
        
        try:
            import torch
            from torch.nn.functional import softmax
            
            results = []
            batch_size = 32  # 批处理大小
            
            print(f"🚀 蒸馏模型批量推理 (batch_size={batch_size})...")
            
            for i in tqdm(range(0, len(texts), batch_size), desc="蒸馏模型批量推理"):
                batch_texts = texts[i:i+batch_size]
                start_time = time.time()
                
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    truncation=True,
                    padding=True,
                    max_length=512
                ).to(self.device)
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs_batch = softmax(outputs.logits, dim=-1).cpu().numpy()
                
                batch_time = time.time() - start_time
                avg_time = batch_time / len(batch_texts)
                
                for probs in probs_batch:
                    predicted_idx = np.argmax(probs)
                    results.append((
                        probs.tolist(),
                        int(predicted_idx),
                        avg_time,
                        "Model prediction"
                    ))
            
            return results
            
        except Exception as e:
            print(f"❌ 批量推理错误: {e}")
            return [self.annotate(text) for text in texts]

# ==========================================
# 6. 评估器（优化版）
# ==========================================

class ModelEvaluator:
    """模型性能评估器（优化版）"""
    
    def __init__(self, test_data: List[Dict]):
        self.test_data = test_data
        self.results = {}
    
    def generate_reference_labels_batch(self, annotator, sample_size: int = None) -> List[int]:
        """
        Generate reference labels using LLM batch processing (optimized)
        """
        if sample_size is None:
            sample_size = len(self.test_data)
        
        print(f"\n🏷️ Using LLM to batch generate {sample_size} reference labels (as pseudo ground truth)...")
        print(f"⚠️  Note: These labels are LLM-generated, used to evaluate relative performance")
        
        reference_data = self.test_data[:sample_size]
        texts = [item["text"] for item in reference_data]
        
        # Use batch processing
        if hasattr(annotator, 'annotate_batch'):
            batch_results = annotator.annotate_batch(texts)
            reference_labels = [pred_idx for _, pred_idx, _, _ in batch_results]
        else:
            reference_labels = []
            for item in tqdm(reference_data, desc="Generating reference labels"):
                _, pred_idx, _, _ = annotator.annotate(item["text"])
                reference_labels.append(pred_idx)
        
        print(f"✅ Reference label generation complete ({len(reference_labels)} labels)")
        return reference_labels

    def evaluate_model(
        self, 
        annotator, 
        model_name: str, 
        ground_truth_labels: List[int] = None,
        save_details: bool = True,
        use_batch: bool = False
    ):
        """Evaluate a single model (optimized)"""
        print(f"\n{'='*80}")
        print(f"🔍 Evaluating: {model_name}")
        print(f"{'='*80}")
        
        predictions = []
        probabilities = []
        inference_times = []
        reasonings = []
        failed_samples = []
        
        # Check if batch processing should be used
        if use_batch and hasattr(annotator, 'annotate_batch'):
            print(f"🚀 Using batch concurrent processing...")
            texts = [item["text"] for item in self.test_data]
            batch_results = annotator.annotate_batch(texts)
            
            for i, (probs, pred_idx, inf_time, reasoning) in enumerate(batch_results):
                predictions.append(pred_idx)
                probabilities.append(probs)
                inference_times.append(inf_time)
                reasonings.append(reasoning)
                if "error" in reasoning.lower():
                    failed_samples.append(i)
        else:
            for item in tqdm(self.test_data, desc=f"Evaluating {model_name}"):
                try:
                    probs, pred_idx, inf_time, reasoning = annotator.annotate(item["text"])
                    predictions.append(pred_idx)
                    probabilities.append(probs)
                    inference_times.append(inf_time)
                    reasonings.append(reasoning)
                except Exception as e:
                    print(f"\n⚠️ Sample {item['id']} evaluation failed: {e}")
                    predictions.append(0)
                    probabilities.append([0.25, 0.25, 0.25, 0.25])
                    inference_times.append(0)
                    reasonings.append("Failed")
                    failed_samples.append(item['id'])
        
        # Performance statistics
        avg_inference_time = np.mean(inference_times)
        median_inference_time = np.median(inference_times)
        std_inference_time = np.std(inference_times)
        total_time = sum(inference_times)
        throughput = len(self.test_data) / total_time if total_time > 0 else 0
        
        result = {
            "model_name": model_name,
            "predictions": predictions,
            "probabilities": probabilities,
            "inference_times": inference_times,
            "reasonings": reasonings,
            "failed_samples": failed_samples,
            "stats": {
                "avg_inference_time": avg_inference_time,
                "median_inference_time": median_inference_time,
                "std_inference_time": std_inference_time,
                "total_time": total_time,
                "throughput": throughput,
                "total_samples": len(self.test_data),
                "failed_samples": len(failed_samples)
            }
        }
        
        # Calculate accuracy if ground truth labels exist
        if ground_truth_labels:
            accuracy = sum(p == g for p, g in zip(predictions, ground_truth_labels)) / len(predictions)
            
            # Per-class statistics
            class_stats = defaultdict(lambda: {"correct": 0, "total": 0})
            for pred, true in zip(predictions, ground_truth_labels):
                class_stats[true]["total"] += 1
                if pred == true:
                    class_stats[true]["correct"] += 1
            
            class_accuracies = {
                DOMAIN_LABELS[k]: v["correct"] / v["total"] if v["total"] > 0 else 0
                for k, v in class_stats.items()
            }
            
            result["stats"]["accuracy"] = accuracy
            result["stats"]["class_accuracies"] = class_accuracies
            result["ground_truth"] = ground_truth_labels
        
        self.results[model_name] = result
        
        # Print statistics
        print(f"\n📊 {model_name} Performance Statistics:")
        print(f"  Total Samples: {result['stats']['total_samples']}")
        print(f"  Failed Samples: {result['stats']['failed_samples']}")
        print(f"  Avg Inference Time: {avg_inference_time:.4f}s")
        print(f"  Median Inference Time: {median_inference_time:.4f}s")
        print(f"  Std Dev: {std_inference_time:.4f}s")
        print(f"  Total Time: {total_time:.2f}s ({total_time/60:.1f} min)")
        print(f"  Throughput: {throughput:.2f} samples/s")
        
        if "accuracy" in result["stats"]:
            print(f"  Overall Accuracy: {result['stats']['accuracy']:.4f} ({result['stats']['accuracy']*100:.2f}%)")
            print(f"  Per-Class Accuracy:")
            for domain, acc in result["stats"]["class_accuracies"].items():
                print(f"    - {domain}: {acc:.4f} ({acc*100:.2f}%)")
        
        # Save detailed results
        if save_details:
            self._save_detailed_results(model_name, result)
        
        return result
    
    def _save_detailed_results(self, model_name: str, result: Dict):
        """保存详细评估结果"""
        filename = f"evaluation_{model_name.replace(' ', '_').replace('/', '_').replace('(', '').replace(')', '')}.json"
        
        save_data = {
            "model_name": model_name,
            "stats": result["stats"],
            "samples": []
        }
        
        for i, item in enumerate(self.test_data):
            sample_result = {
                "id": item["id"],
                "text": item["text"],
                "prediction": result["predictions"][i],
                "predicted_label": DOMAIN_LABELS[result["predictions"][i]],
                "probabilities": result["probabilities"][i],
                "inference_time": result["inference_times"][i],
                "reasoning": result["reasonings"][i]
            }
            
            if "ground_truth" in result:
                sample_result["ground_truth"] = result["ground_truth"][i]
                sample_result["ground_truth_label"] = DOMAIN_LABELS[result["ground_truth"][i]]
                sample_result["correct"] = result["predictions"][i] == result["ground_truth"][i]
            
            save_data["samples"].append(sample_result)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, ensure_ascii=False, indent=2)
        
        print(f"  💾 详细结果已保存: {filename}")
    
    def compare_models(self):
        """Compare all models"""
        if len(self.results) < 2:
            print("\n⚠️ Need at least 2 models for comparison")
            return
        
        print(f"\n{'='*80}")
        print("📊 Model Performance Comparison")
        print(f"{'='*80}\n")
        
        # Create comparison table
        print(f"{'Metric':<25} ", end="")
        for model_name in self.results.keys():
            short_name = model_name[:20] + "..." if len(model_name) > 20 else model_name
            print(f"{short_name:<30} ", end="")
        print()
        print("-" * 110)
        
        # Inference time comparison
        print(f"{'Avg Inference Time (s)':<25} ", end="")
        for result in self.results.values():
            print(f"{result['stats']['avg_inference_time']:<30.6f} ", end="")
        print()
        
        print(f"{'Median Inference Time (s)':<25} ", end="")
        for result in self.results.values():
            print(f"{result['stats']['median_inference_time']:<30.6f} ", end="")
        print()
        
        # Throughput comparison
        print(f"{'Throughput (samples/s)':<25} ", end="")
        for result in self.results.values():
            print(f"{result['stats']['throughput']:<30.2f} ", end="")
        print()
        
        # Accuracy comparison
        if all("accuracy" in r["stats"] for r in self.results.values()):
            print(f"{'Accuracy':<25} ", end="")
            for result in self.results.values():
                acc = result['stats']['accuracy']
                print(f"{acc:<30.4f} ", end="")
            print()
        
        # Speed-up ratio
        print(f"\n⚡ Speed-up Ratio (relative to first model):")
        models = list(self.results.keys())
        base_time = self.results[models[0]]['stats']['avg_inference_time']
        
        for model_name in models:
            time_val = self.results[model_name]['stats']['avg_inference_time']
            speedup = base_time / time_val if time_val > 0 else 0
            print(f"  {model_name}: {speedup:.2f}x")
    
    def plot_comparison(self, save_path: str = "model_comparison.png"):
        """Plot performance comparison charts"""
        if len(self.results) < 2:
            print("\n⚠️ Need at least 2 models for comparison")
            return
        
        print(f"\n📈 Generating performance comparison charts...")
        
        models = list(self.results.keys())
        short_names = [m[:15] + "..." if len(m) > 15 else m for m in models]
        
        # Prepare data
        avg_times = [self.results[m]['stats']['avg_inference_time'] for m in models]
        throughputs = [self.results[m]['stats']['throughput'] for m in models]
        accuracies = [self.results[m]['stats'].get('accuracy', 0) for m in models]
        
        # Create subplots
        fig = plt.figure(figsize=(18, 10))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        # 1. Average inference time
        ax1 = fig.add_subplot(gs[0, 0])
        bars1 = ax1.bar(short_names, avg_times, color=colors)
        ax1.set_title('Average Inference Time', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Time (seconds)', fontsize=10)
        ax1.grid(axis='y', alpha=0.3)
        ax1.tick_params(axis='x', rotation=45)
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.4f}s', ha='center', va='bottom', fontsize=8)
        
        # 2. Throughput
        ax2 = fig.add_subplot(gs[0, 1])
        bars2 = ax2.bar(short_names, throughputs, color=colors)
        ax2.set_title('Throughput', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Samples/sec', fontsize=10)
        ax2.grid(axis='y', alpha=0.3)
        ax2.tick_params(axis='x', rotation=45)
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        # 3. Accuracy
        ax3 = fig.add_subplot(gs[0, 2])
        if any(acc > 0 for acc in accuracies):
            bars3 = ax3.bar(short_names, accuracies, color=colors)
            ax3.set_title('Accuracy (vs LLM Labels)', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Accuracy', fontsize=10)
            ax3.set_ylim(0, 1)
            ax3.grid(axis='y', alpha=0.3)
            ax3.tick_params(axis='x', rotation=45)
            for bar in bars3:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        else:
            ax3.text(0.5, 0.5, 'No Ground Truth\nAccuracy Unavailable',
                    ha='center', va='center', transform=ax3.transAxes,
                    fontsize=12)
            ax3.set_title('Accuracy', fontsize=12, fontweight='bold')
        
        # 4. Inference time distribution
        ax4 = fig.add_subplot(gs[1, :2])
        for i, model_name in enumerate(models):
            times = self.results[model_name]['inference_times']
            ax4.hist(times, bins=50, alpha=0.5, label=short_names[i], 
                    edgecolor='black', color=colors[i])
        ax4.set_title('Inference Time Distribution', fontsize=12, fontweight='bold')
        ax4.set_xlabel('Inference Time (seconds)', fontsize=10)
        ax4.set_ylabel('Frequency', fontsize=10)
        ax4.legend(loc='upper right')
        ax4.grid(axis='y', alpha=0.3)
        
        # 5. Speed-up ratio
        ax5 = fig.add_subplot(gs[1, 2])
        base_time = self.results[models[0]]['stats']['avg_inference_time']
        speedups = [base_time / self.results[m]['stats']['avg_inference_time'] 
                   for m in models]
        bars5 = ax5.barh(short_names, speedups, color=colors)
        ax5.set_title(f'Speed-up Ratio\n(vs {short_names[0]})', fontsize=12, fontweight='bold')
        ax5.set_xlabel('Speedup (x)', fontsize=10)
        ax5.axvline(x=1, color='red', linestyle='--', alpha=0.5)
        ax5.grid(axis='x', alpha=0.3)
        for bar in bars5:
            width = bar.get_width()
            ax5.text(width, bar.get_y() + bar.get_height()/2.,
                    f'{width:.1f}x', ha='left', va='center', fontsize=8)
        
        plt.suptitle('RAG Router Model Performance Comparison', fontsize=16, fontweight='bold')
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Performance comparison chart saved: {save_path}")
        
        # Plot class accuracies if available
        if all("class_accuracies" in r["stats"] for r in self.results.values()):
            self._plot_class_accuracies()
    
    def _plot_class_accuracies(self):
        """Plot per-class accuracy comparison"""
        print(f"📈 Generating per-class accuracy comparison chart...")
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        x = np.arange(len(DOMAIN_LABELS))
        width = 0.8 / len(self.results)
        
        models = list(self.results.keys())
        colors = plt.cm.Set3(np.linspace(0, 1, len(models)))
        
        for i, model_name in enumerate(models):
            class_accs = self.results[model_name]['stats']['class_accuracies']
            accs = [class_accs.get(domain, 0) for domain in DOMAIN_LABELS]
            offset = width * (i - len(models)/2 + 0.5)
            bars = ax.bar(x + offset, accs, width, 
                         label=model_name[:20] + "..." if len(model_name) > 20 else model_name,
                         alpha=0.8, color=colors[i])
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.2f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_xlabel('Domain', fontsize=12)
        ax.set_ylabel('Accuracy', fontsize=12)
        ax.set_title('Per-Domain Classification Accuracy (vs LLM Labels)', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels(DOMAIN_LABELS)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig('class_accuracies_comparison.png', dpi=300, bbox_inches='tight')
        print(f"✅ Class accuracy comparison chart saved: class_accuracies_comparison.png")
    
    def save_summary(self, filepath: str = "evaluation_summary.json"):
        """保存评估摘要"""
        summary = {
            "test_samples": len(self.test_data),
            "models": {}
        }
        
        for model_name, result in self.results.items():
            summary["models"][model_name] = result["stats"]
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        print(f"\n💾 评估摘要已保存: {filepath}")

# ==========================================
# 7. 主函数（优化版）
# ==========================================

def main():
    print("="*80)
    print("🚀 RAG Router Model Evaluation System (Optimized - Qwen-30B)")
    print("="*80)
    print(f"⚙️  Config: Test Samples={TEST_SAMPLE_SIZE}, Reference Samples={REFERENCE_SAMPLE_SIZE}, Workers={MAX_WORKERS}")
    print(f"🔧 Model: {MODEL_NAME}")
    
    # 1. Load test data
    test_data = load_test_data(TEST_SAMPLE_SIZE, split="validation")
    
    if len(test_data) == 0:
        print("❌ Failed to load test data, exiting")
        return
    
    # 2. Initialize evaluator
    evaluator = ModelEvaluator(test_data)
    
    # 3. Initialize annotators
    print("\n" + "="*80)
    print("🔧 Initializing models...")
    print("="*80)
    
    keyword_classifier = KeywordClassifier()
    llm_annotator = LLMAnnotator(max_workers=MAX_WORKERS)
    distilled_annotator = DistilledModelAnnotator(DISTILLED_MODEL_PATH)
    
    # 4. Generate reference labels (using LLM batch processing)
    print("\n⭐ Important Notes:")
    print("  - Reference labels (ground truth) generated by Qwen-30B")
    print("  - Qwen-30B accuracy should be close to 100% (except format errors)")
    print("  - Other models' accuracy represents consistency with Qwen-30B")
    
    ground_truth_labels = evaluator.generate_reference_labels_batch(
        llm_annotator, 
        sample_size=REFERENCE_SAMPLE_SIZE
    )
    
    # 5. Evaluate keyword classifier
    evaluator.evaluate_model(
        keyword_classifier,
        "Keyword Classifier",
        ground_truth_labels=ground_truth_labels,
        use_batch=False
    )
    
    # 6. Evaluate LLM model (using batch concurrent processing)
    evaluator.evaluate_model(
        llm_annotator,
        "Qwen3-30B (LLM API)",
        ground_truth_labels=ground_truth_labels,
        use_batch=True
    )
    
    # 7. Evaluate distilled model
    evaluator.evaluate_model(
        distilled_annotator,
        "Distilled Model",
        ground_truth_labels=ground_truth_labels,
        use_batch=True
    )
    
    # 8. Compare and visualize
    evaluator.compare_models()
    evaluator.plot_comparison()
    evaluator.save_summary()
    
    print("\n" + "="*80)
    print("✅ Evaluation Complete!")
    print("="*80)
    print("\nGenerated Files:")
    print("  📄 evaluation_Keyword_Classifier.json - Keyword classifier detailed results")
    print("  📄 evaluation_Qwen3-30B_LLM_API.json - LLM detailed results")
    print("  📄 evaluation_Distilled_Model.json - Distilled model detailed results")
    print("  📄 evaluation_summary.json - Evaluation summary")
    print("  📊 model_comparison.png - Performance comparison chart")
    print("  📊 class_accuracies_comparison.png - Class accuracy comparison chart")

if __name__ == "__main__":
    main()