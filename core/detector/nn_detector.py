import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import Counter, deque
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from dataclasses import dataclass, field
import logging
import json

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ==========================================
# 0. 标签映射（与训练时保持一致）
# ==========================================
LABEL_MAP = {
    "entertainment": 0,
    "stem": 1,
    "humanities": 2,
    "lifestyle": 3
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}

@dataclass
class DetectionResult:
    step: int
    query: str
    
    # Individual 预测
    predicted_domain: str
    confidence: float
    calibrated_probs: Dict[str, float]  # 校准后的概率
    entropy: float  # 预测熵（高熵 = OOD 信号）
    
    # Global Shift 检测
    is_global_shift: bool = False
    jsd_score: float = 0.0  # JS 散度
    psi_score: float = 0.0  # PSI 指标
    query_distribution: Dict[str, float] = field(default_factory=dict)
    
    # Local Shift 检测（领域内精度下降）
    local_accuracy: Dict[str, float] = field(default_factory=dict)
    local_shifts: Dict[str, bool] = field(default_factory=dict)

# ==========================================
# 1. Temperature Scaling 校准器
# ==========================================
class TemperatureScaling(nn.Module):
    """
    模型校准：解决小模型过度自信的问题
    参考论文：On Calibration of Modern Neural Networks (Guo et al., ICML 2017)
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)  # 初始温度

    def forward(self, logits):
        return logits / self.temperature

    def calibrate(self, model, val_loader, device, max_iter=50):
        """
        在验证集上学习最优温度参数
        """
        nll_criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=max_iter)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = 0.0
            with torch.no_grad():
                for batch in val_loader:
                    input_ids = batch['input_ids'].to(device)
                    mask = batch['attention_mask'].to(device)
                    labels = batch['hard_label'].to(device)
                    
                    logits = model(input_ids, attention_mask=mask).logits
                    loss += nll_criterion(self.forward(logits), labels)
            
            loss.backward()
            return loss
        
        optimizer.step(eval_loss)
        logger.info(f"✅ 校准完成，最优温度: T = {self.temperature.item():.3f}")
        return self.temperature.item()

# ==========================================
# 2. RAG Drift Detector
# ==========================================
class RAGDriftDetector:
    def __init__(
        self,
        model_path: str,
        kb_distribution: Dict[str, float],  # RAG 知识库的先验分布
        global_window_size: int = 100,
        local_window_size: int = 30,
        jsd_threshold: float = 0.1,  # JS 散度阈值
        psi_threshold: float = 0.2,  # PSI 阈值
        local_acc_drop_threshold: float = 0.15,  # 领域内精度下降阈值
        ood_entropy_threshold: float = 1.2,  # OOD 检测的熵阈值
        use_calibration: bool = True
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.domains = list(LABEL_MAP.keys())
        
        # ========== 1. 加载蒸馏模型 ==========
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()
        
        # ========== 2. 模型校准（可选） ==========
        self.use_calibration = use_calibration
        if use_calibration:
            self.temperature_scaler = TemperatureScaling().to(self.device)
            # 如果有验证集，可以在初始化时调用：
            # self.temperature_scaler.calibrate(self.model, val_loader, self.device)
        
        # ========== 3. KB 先验分布 ==========
        total = sum(kb_distribution.values())
        self.kb_dist = {k: v / total for k, v in kb_distribution.items()}
        logger.info(f"📊 RAG 知识库分布: {self.kb_dist}")
        
        # ========== 4. Global Shift 状态 ==========
        self.global_window = deque(maxlen=global_window_size)  # 存储校准后的概率向量
        self.jsd_threshold = jsd_threshold
        self.psi_threshold = psi_threshold
        self.ood_entropy_threshold = ood_entropy_threshold
        
        # ========== 5. Local Shift 状态（每个领域独立跟踪） ==========
        self.local_window_size = local_window_size
        self.local_windows = {d: deque(maxlen=local_window_size) for d in self.domains}
        self.local_baseline_acc = {d: 0.85 for d in self.domains}  # 初始基线
        self.local_acc_drop_threshold = local_acc_drop_threshold
        
        # ========== 6. 冷却期（避免频繁报警） ==========
        self.global_cooldown = 0
        self.local_cooldown = {d: 0 for d in self.domains}
        
        logger.info(f"🚀 Detector 初始化完成 | Device: {self.device}")

    def _predict_with_calibration(self, query: str) -> Dict:
        """
        预测 + 校准
        """
        inputs = self.tokenizer(
            query, 
            return_tensors="pt", 
            truncation=True, 
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            logits = self.model(**inputs).logits
            
            # 应用温度缩放
            if self.use_calibration:
                logits = self.temperature_scaler(logits)
            
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
        
        pred_id = int(np.argmax(probs))
        pred_entropy = entropy(probs, base=2)  # 用于 OOD 检测
        
        return {
            "domain": ID2LABEL[pred_id],
            "confidence": float(probs[pred_id]),
            "probs": {ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
            "entropy": float(pred_entropy)
        }

    def detect(
        self, 
        query: str, 
        step: int, 
        ground_truth: Optional[str] = None
    ) -> DetectionResult:
        """
        主检测函数
        :param query: 输入查询
        :param step: 当前步数
        :param ground_truth: 真实标签（用于 Local Shift 检测）
        """
        # ========== Individual 预测 ==========
        pred = self._predict_with_calibration(query)
        
        result = DetectionResult(
            step=step,
            query=query,
            predicted_domain=pred["domain"],
            confidence=pred["confidence"],
            calibrated_probs=pred["probs"],
            entropy=pred["entropy"]
        )
        
        # OOD 检测（高熵 = 未知类别）
        if pred["entropy"] > self.ood_entropy_threshold:
            logger.warning(f"⚠️ [OOD Signal] Query: '{query[:50]}...' | Entropy: {pred['entropy']:.3f}")
        
        # ========== 更新全局窗口 ==========
        self.global_window.append(pred["probs"])
        
        # ========== Global Shift 检测 ==========
        if len(self.global_window) >= 10:
            result = self._detect_global_shift(result)
        
        # ========== Local Shift 检测（需要真实标签） ==========
        if ground_truth is not None:
            result = self._detect_local_shift(result, ground_truth)
        
        return result

    def _detect_global_shift(self, result: DetectionResult) -> DetectionResult:
        """
        全局分布偏移检测（JS 散度 + PSI）
        """
        # 计算当前查询流的平均分布
        query_dist_vectors = np.array([list(p.values()) for p in self.global_window])
        avg_query_dist = np.mean(query_dist_vectors, axis=0)
        query_dist = {d: float(avg_query_dist[i]) for i, d in enumerate(self.domains)}
        
        result.query_distribution = query_dist
        
        # JS 散度
        kb_vec = np.array([self.kb_dist[d] for d in self.domains])
        query_vec = np.array([query_dist[d] for d in self.domains])
        jsd = jensenshannon(kb_vec, query_vec)
        result.jsd_score = float(jsd)
        
        # PSI (Population Stability Index)
        psi = self._calculate_psi(self.kb_dist, query_dist)
        result.psi_score = psi
        
        # 判定 Global Shift
        if self.global_cooldown == 0:
            if jsd > self.jsd_threshold or psi > self.psi_threshold:
                result.is_global_shift = True
                logger.warning(f"🚨 [Global Shift] Step {result.step}")
                logger.warning(f"   JSD: {jsd:.4f} (阈值: {self.jsd_threshold})")
                logger.warning(f"   PSI: {psi:.4f} (阈值: {self.psi_threshold})")
                logger.warning(f"   KB 分布:    {self.kb_dist}")
                logger.warning(f"   Query 分布: {query_dist}")
                self.global_cooldown = 50  # 冷却 50 步
        
        if self.global_cooldown > 0:
            self.global_cooldown -= 1
        
        return result

    def _detect_local_shift(
        self, 
        result: DetectionResult, 
        ground_truth: str
    ) -> DetectionResult:
        """
        局部领域精度下降检测
        """
        predicted = result.predicted_domain
        is_correct = (predicted == ground_truth)
        
        # 记录到真实领域的窗口
        self.local_windows[ground_truth].append(is_correct)
        
        # 检测每个领域
        for domain in self.domains:
            window = self.local_windows[domain]
            
            if len(window) >= 5:  # 最小样本数
                current_acc = sum(window) / len(window)
                result.local_accuracy[domain] = current_acc
                
                baseline = self.local_baseline_acc[domain]
                acc_drop = baseline - current_acc
                
                # Local Shift 判定
                if self.local_cooldown[domain] == 0:
                    if acc_drop > self.local_acc_drop_threshold:
                        result.local_shifts[domain] = True
                        logger.warning(f"⚠️ [Local Shift] Domain: {domain}")
                        logger.warning(f"   当前准确率: {current_acc:.2%}")
                        logger.warning(f"   基线准确率: {baseline:.2%}")
                        logger.warning(f"   下降幅度: {acc_drop:.2%}")
                        self.local_cooldown[domain] = 30
                    else:
                        result.local_shifts[domain] = False
                else:
                    result.local_shifts[domain] = False
                
                # 更新基线（指数滑动平均）
                if not result.local_shifts.get(domain, False):
                    self.local_baseline_acc[domain] = 0.9 * baseline + 0.1 * current_acc
                
                if self.local_cooldown[domain] > 0:
                    self.local_cooldown[domain] -= 1
        
        return result

    def _calculate_psi(self, expected: Dict, actual: Dict) -> float:
        """
        PSI (Population Stability Index)
        公式: PSI = Σ (actual% - expected%) * ln(actual% / expected%)
        """
        psi = 0.0
        for domain in self.domains:
            e = expected.get(domain, 1e-10)
            a = actual.get(domain, 1e-10)
            psi += (a - e) * np.log(a / e)
        return float(psi)

    def update_kb_distribution(self, new_kb_dist: Dict[str, float]):
        """
        外部触发 KB 更新时重置状态
        """
        total = sum(new_kb_dist.values())
        self.kb_dist = {k: v / total for k, v in new_kb_dist.items()}
        self.global_window.clear()
        self.global_cooldown = 0
        logger.info(f"🔄 KB 分布已更新: {self.kb_dist}")

    def reset_local_baseline(self, domain: str = None, new_baseline: float = 0.85):
        """
        重置领域基线（如重新标注数据后）
        """
        if domain:
            self.local_baseline_acc[domain] = new_baseline
            self.local_windows[domain].clear()
            logger.info(f"🔄 领域 {domain} 基线已重置为 {new_baseline}")
        else:
            for d in self.domains:
                self.local_baseline_acc[d] = new_baseline
                self.local_windows[d].clear()
            logger.info(f"🔄 所有领域基线已重置为 {new_baseline}")

# ==========================================
# 3. 使用示例
# ==========================================
if __name__ == "__main__":
    # RAG 知识库的先验分布（需要预先统计）
    kb_prior = {
        "entertainment": 0.15,
        "stem": 0.40,
        "humanities": 0.30,
        "lifestyle": 0.15
    }
    
    detector = RAGDriftDetector(
        model_path="./mini_router_best",
        kb_distribution=kb_prior,
        global_window_size=100,
        local_window_size=30,
        jsd_threshold=0.1,
        psi_threshold=0.2,
        local_acc_drop_threshold=0.15,
        use_calibration=True  # 启用校准
    )
    
    # 模拟数据流
    test_queries = [
        ("What company sponsored the Toyota Owners 400?", "entertainment"),
        ("How to implement gradient descent in PyTorch?", "stem"),
        ("The impact of Renaissance on art", "humanities"),
        ("Best workout routine for beginners", "lifestyle"),
        ("Quantum computing fundamentals", "stem"),
        ("Latest celebrity gossip 2024", "entertainment"),  # 可能触发 OOD
    ]
    
    print("\n" + "="*80)
    print("                    RAG DRIFT DETECTION DEMO")
    print("="*80 + "\n")
    
    for i, (query, gt) in enumerate(test_queries):
        result = detector.detect(query, step=i, ground_truth=gt)
        
        print(f"\n{'─'*80}")
        print(f"📍 Step {result.step} | Query: {result.query[:60]}...")
        print(f"{'─'*80}")
        
        # Individual 预测
        print(f"\n🔍 Individual Prediction:")
        print(f"   Predicted: {result.predicted_domain} (Conf: {result.confidence:.2%})")
        print(f"   Ground Truth: {gt}")
        print(f"   Entropy: {result.entropy:.3f} {'⚠️ OOD' if result.entropy > detector.ood_entropy_threshold else ''}")
        print(f"   Calibrated Probs: {result.calibrated_probs}")
        
        # Global Shift
        if result.query_distribution:
            print(f"\n🌍 Global Shift Detection:")
            print(f"   Status: {'🚨 SHIFT DETECTED' if result.is_global_shift else '✅ Normal'}")
            print(f"   JSD: {result.jsd_score:.4f} (阈值: {detector.jsd_threshold})")
            print(f"   PSI: {result.psi_score:.4f} (阈值: {detector.psi_threshold})")
            print(f"   Query 分布: {result.query_distribution}")
        
        # Local Shift
        if result.local_accuracy:
            print(f"\n📊 Local Shift Detection (Domain-wise Accuracy):")
            for domain, acc in result.local_accuracy.items():
                is_shift = result.local_shifts.get(domain, False)
                status = "⚠️ SHIFT" if is_shift else "✅ Normal"
                print(f"   {domain:15s}: {acc:.2%} {status}")
    
    print("\n" + "="*80 + "\n")