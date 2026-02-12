import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from torch.optim import AdamW
import json
import numpy as np
import os
import random
from tqdm import tqdm
from sklearn.metrics import classification_report

# ==========================================
# 0. 全局配置与标签映射
# ==========================================
LABEL_MAP = {
    "entertainment": 0,
    "stem": 1,
    "humanities": 2,
    "lifestyle": 3
}
ID2LABEL = {v: k for k, v in LABEL_MAP.items()}


class Config:
    data_path = "/home/jyliu/RAG_project/train_distill_mixed_qwen_v4.jsonl"

    model_name = "distilbert-base-uncased"
    num_labels = 4
    max_len = 128
    batch_size = 32
    lr = 3e-5
    epochs = 8              # 1万条数据，8轮足够收敛
    temperature = 2.0        # 降低温度，分布更锐利
    alpha = 0.3              # 1万条标注数据足够，更多依赖 hard label
    val_split = 0.1
    device = "cpu"
    save_dir = "/home/jyliu/RAG_project/detector/mini_router_best"
    seed = 42
    warmup_ratio = 0.1       # 10% warmup
    max_grad_norm = 1.0      # 梯度裁剪
    early_stopping_patience = 3  # 早停耐心值


# ==========================================
# 1. 固定随机种子
# ==========================================
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ==========================================
# 2. 数据集加载类
# ==========================================
class DistillationDataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len):
        self.data = []
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"找不到数据文件: {data_path}")

        with open(data_path, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    if 'text' in item and 'teacher_probs' in item and 'hard_label' in item:
                        self.data.append(item)

        print(f"✅ 成功加载数据，共 {len(self.data)} 条。")

        # 统计各类别分布
        label_counts = {}
        for item in self.data:
            lbl = item['hard_label']
            label_counts[lbl] = label_counts.get(lbl, 0) + 1
        print("📊 类别分布:")
        for lbl_id in sorted(label_counts.keys()):
            lbl_name = ID2LABEL.get(lbl_id, f"unknown_{lbl_id}")
            count = label_counts[lbl_id]
            ratio = count / len(self.data) * 100
            print(f"   {lbl_name} ({lbl_id}): {count} 条 ({ratio:.1f}%)")

        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        text = item['text']
        teacher_probs = torch.tensor(item['teacher_probs'], dtype=torch.float)
        hard_label = torch.tensor(item['hard_label'], dtype=torch.long)

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'teacher_probs': teacher_probs,
            'hard_label': hard_label
        }


# ==========================================
# 3. 损失函数
# ==========================================
def distillation_loss(student_logits, teacher_probs, hard_labels, temp, alpha):
    # Soft Loss: KL散度
    student_log_probs = F.log_softmax(student_logits / temp, dim=1)
    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temp ** 2)

    # Hard Loss: 交叉熵
    hard_loss = F.cross_entropy(student_logits, hard_labels)

    return alpha * soft_loss + (1.0 - alpha) * hard_loss


# ==========================================
# 4. 评估函数
# ==========================================
def evaluate(model, dataloader, device):
    """基础评估：返回准确率和验证损失"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            hard_labels = batch['hard_label'].to(device)

            outputs = model(input_ids, attention_mask=mask)
            logits = outputs.logits

            loss = F.cross_entropy(logits, hard_labels)
            total_loss += loss.item()

            preds = torch.argmax(logits, dim=1)
            correct += (preds == hard_labels).sum().item()
            total += hard_labels.size(0)

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    return accuracy, avg_loss


def evaluate_detailed(model, dataloader, device):
    """详细评估：输出每个类别的 precision / recall / f1"""
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            mask = batch['attention_mask'].to(device)
            hard_labels = batch['hard_label'].to(device)

            outputs = model(input_ids, attention_mask=mask)
            preds = torch.argmax(outputs.logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(hard_labels.cpu().numpy())

    target_names = [ID2LABEL[i] for i in range(Config.num_labels)]
    report = classification_report(all_labels, all_preds, target_names=target_names, digits=4)
    print(report)
    return report


# ==========================================
# 5. 训练主循环
# ==========================================
def train():
    set_seed(Config.seed)

    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    full_dataset = DistillationDataset(Config.data_path, tokenizer, Config.max_len)

    # 切分数据集
    val_size = int(len(full_dataset) * Config.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(Config.seed)
    )

    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, num_workers=2, pin_memory=True)

    print(f"📦 训练集: {len(train_dataset)} 条 | 验证集: {len(val_dataset)} 条")

    model = AutoModelForSequenceClassification.from_pretrained(Config.model_name, num_labels=Config.num_labels)
    model.to(Config.device)

    optimizer = AdamW(model.parameters(), lr=Config.lr, weight_decay=0.01)

    # 学习率调度器 (带 warmup)
    total_steps = len(train_loader) * Config.epochs
    warmup_steps = int(total_steps * Config.warmup_ratio)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps)

    print(f"\n🚀 开始在 {Config.device} 上训练...")
    print(f"   总步数: {total_steps} | Warmup 步数: {warmup_steps}")
    print(f"   温度: {Config.temperature} | Alpha(软损失权重): {Config.alpha}")

    best_acc = 0
    patience_counter = 0

    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{Config.epochs}")

        for batch in pbar:
            optimizer.zero_grad()

            input_ids = batch['input_ids'].to(Config.device)
            mask = batch['attention_mask'].to(Config.device)
            teacher_probs = batch['teacher_probs'].to(Config.device)
            hard_labels = batch['hard_label'].to(Config.device)

            outputs = model(input_ids, attention_mask=mask)
            loss = distillation_loss(outputs.logits, teacher_probs, hard_labels, Config.temperature, Config.alpha)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=Config.max_grad_norm)
            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

            # 训练准确率
            preds = torch.argmax(outputs.logits, dim=1)
            train_correct += (preds == hard_labels).sum().item()
            train_total += hard_labels.size(0)

            current_lr = scheduler.get_last_lr()[0]
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "lr": f"{current_lr:.2e}"})

        avg_train_loss = train_loss / len(train_loader)
        train_acc = train_correct / train_total

        # 验证
        val_acc, val_loss = evaluate(model, val_loader, Config.device)
        print(f"📊 Epoch {epoch+1} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        # 保存表现最好的模型 + 早停
        if val_acc > best_acc:
            best_acc = val_acc
            patience_counter = 0
            print(f"✨ 发现更好的模型 (Val Acc: {val_acc:.4f})，已保存至 {Config.save_dir}")
            model.save_pretrained(Config.save_dir)
            tokenizer.save_pretrained(Config.save_dir)
        else:
            patience_counter += 1
            print(f"⏳ 模型未提升，耐心值: {patience_counter}/{Config.early_stopping_patience}")
            if patience_counter >= Config.early_stopping_patience:
                print(f"🛑 早停触发！最佳 Val Acc: {best_acc:.4f}")
                break

    print(f"\n🏁 训练结束，最佳 Val Acc: {best_acc:.4f}")

    # 加载最佳模型做详细评估
    if os.path.exists(Config.save_dir):
        print("\n" + "=" * 50)
        print("      📋 最佳模型分类详细报告")
        print("=" * 50)
        best_model = AutoModelForSequenceClassification.from_pretrained(Config.save_dir).to(Config.device)
        evaluate_detailed(best_model, val_loader, Config.device)


# ==========================================
# 6. 在线检测器
# ==========================================
class OnlineDetector:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path).to(self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model.eval()

    def predict(self, query):
        inputs = self.tokenizer(query, return_tensors="pt", truncation=True, max_length=128).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = F.softmax(outputs.logits, dim=1).cpu().numpy()[0]

        pred_id = int(np.argmax(probs))
        entropy = -np.sum(probs * np.log(probs + 1e-9))

        return {
            "query": query,
            "top_label": ID2LABEL[pred_id],
            "confidence": float(probs[pred_id]),
            "all_probs": {ID2LABEL[i]: float(p) for i, p in enumerate(probs)},
            "entropy": float(entropy)
        }

    def predict_batch(self, queries, batch_size=32):
        """批量预测，提高推理效率"""
        results = []
        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i + batch_size]
            inputs = self.tokenizer(
                batch_queries,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=128
            ).to(self.device)

            if 'token_type_ids' in inputs:
                inputs.pop('token_type_ids')

            with torch.no_grad():
                outputs = self.model(**inputs)
                batch_probs = F.softmax(outputs.logits, dim=1).cpu().numpy()

            for j, (query, probs) in enumerate(zip(batch_queries, batch_probs)):
                pred_id = int(np.argmax(probs))
                entropy = -np.sum(probs * np.log(probs + 1e-9))
                results.append({
                    "query": query,
                    "top_label": ID2LABEL[pred_id],
                    "confidence": float(probs[pred_id]),
                    "all_probs": {ID2LABEL[k]: float(p) for k, p in enumerate(probs)},
                    "entropy": float(entropy)
                })
        return results


# ==========================================
# 7. 入口
# ==========================================
if __name__ == "__main__":
    # 1. 执行训练
    train()

    # 2. 测试推理 (使用保存的最佳模型)
    if os.path.exists(Config.save_dir):
        detector = OnlineDetector(Config.save_dir)
        print("\n" + "=" * 50)
        print("      🧪 Online Detection Test")
        print("=" * 50)

        samples = [
            # entertainment
            "What company sponsored the Toyota Owners 400 from 2007 to 2011?",
            "Who won the Best Actor Oscar in 2020?",
            # stem
            "How to write a transformer model in PyTorch?",
            "Explain the difference between TCP and UDP protocols.",
            # humanities
            "The impact of the French Revolution on modern democracy",
            "What are the main themes of Shakespeare's Hamlet?",
            # lifestyle
            "What are the best exercises for losing belly fat?",
            "How to make a perfect sourdough bread at home?",
        ]

        for q in samples:
            res = detector.predict(q)
            print(f"\nQ: {res['query']}")
            print(f"   🏷️  Domain: [{res['top_label'].upper()}] (Conf: {res['confidence']:.3f}, Entropy: {res['entropy']:.4f})")
            probs_str = " | ".join([f"{k}: {v:.3f}" for k, v in res['all_probs'].items()])
            print(f"   📊 {probs_str}")