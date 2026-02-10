import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
# 移除 AdamW
from transformers import AutoTokenizer, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
# 导入 PyTorch 原生的 AdamW
from torch.optim import AdamW 
import json
import numpy as np
import os
from tqdm import tqdm

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
    # 你的真实数据文件
    data_path = "train_distill_mixed_qwen.jsonl"
    
    model_name = "distilbert-base-uncased" 
    num_labels = 4         
    max_len = 128           # 适当增加长度以应对 HotpotQA 的长问题
    batch_size = 32        
    lr = 3e-5               # 微调通常使用较小的学习率
    epochs = 5             # 真实数据建议 5-10 轮
    temperature = 4.0      # 蒸馏温度
    alpha = 0.5            # 软硬 Loss 比例
    val_split = 0.1        # 10% 验证集
    device = "cuda" if torch.cuda.is_available() else "cpu"
    save_dir = "./mini_router_best"

# ==========================================
# 1. 数据集加载类
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
                    # 简单过滤确保字段完整
                    if 'text' in item and 'teacher_probs' in item:
                        self.data.append(item)
        
        print(f"✅ 成功加载数据，共 {len(self.data)} 条。")
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
# 2. 损失函数
# ==========================================
def distillation_loss(student_logits, teacher_probs, hard_labels, temp, alpha):
    # Soft Loss: KL散度
    student_log_probs = F.log_softmax(student_logits / temp, dim=1)
    soft_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean') * (temp ** 2)
    
    # Hard Loss: 交叉熵
    hard_loss = F.cross_entropy(student_logits, hard_labels)
    
    return alpha * soft_loss + (1.0 - alpha) * hard_loss

# ==========================================
# 3. 训练与验证函数
# ==========================================
def evaluate(model, dataloader, device):
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
            
            # 计算准确率
            preds = torch.argmax(logits, dim=1)
            correct += (preds == hard_labels).sum().item()
            total += hard_labels.size(0)
            
    return correct / total

def train():
    tokenizer = AutoTokenizer.from_pretrained(Config.model_name)
    full_dataset = DistillationDataset(Config.data_path, tokenizer, Config.max_len)
    
    # 切分数据集
    val_size = int(len(full_dataset) * Config.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size)
    
    model = AutoModelForSequenceClassification.from_pretrained(Config.model_name, num_labels=Config.num_labels)
    model.to(Config.device)
    
    optimizer = AdamW(model.parameters(), lr=Config.lr, weight_decay=0.01) 
    
    # 学习率调度器
    total_steps = len(train_loader) * Config.epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)
    
    print(f"\n🚀 开始在 {Config.device} 上训练...")
    best_acc = 0
    
    for epoch in range(Config.epochs):
        model.train()
        train_loss = 0
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
            optimizer.step()
            scheduler.step()
            
            train_loss += loss.item()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})
            
        # 验证
        val_acc = evaluate(model, val_loader, Config.device)
        print(f"📊 Epoch {epoch+1} | Train Loss: {train_loss/len(train_loader):.4f} | Val Acc: {val_acc:.4f}")
        
        # 保存表现最好的模型
        if val_acc > best_acc:
            best_acc = val_acc
            print(f"✨ 发现更好的模型，已保存至 {Config.save_dir}")
            model.save_pretrained(Config.save_dir)
            tokenizer.save_pretrained(Config.save_dir)

# ==========================================
# 4. 在线检测器
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

if __name__ == "__main__":
    # 1. 执行训练
    train()
    
    # 2. 测试推理 (使用保存的最佳模型)
    if os.path.exists(Config.save_dir):
        detector = OnlineDetector(Config.save_dir)
        print("\n" + "="*40)
        print("      Online Detection Test")
        print("="*40)
        
        samples = [
            "What company sponsored the Toyota Owners 400 from 2007 to 2011?",
            "How to write a transformer model in PyTorch?",
            "The impact of the French Revolution on modern democracy"
        ]
        
        for q in samples:
            res = detector.predict(q)
            print(f"\nQ: {res['query']}")
            print(f"Top Domain: [{res['top_label'].upper()}] (Conf: {res['confidence']:.2f})")
            print(f"Entropy: {res['entropy']:.4f}")