import sys
import os
import numpy as np
import matplotlib.pyplot as plt
import random
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics.pairwise import cosine_similarity

try:
    from river import drift
except ImportError:
    print("❌ 请先安装 river: pip install river")
    sys.exit(1)

# ================= 1. 构造合成数据 =================
def generate_synthetic_data():
    print("🧪 正在构造合成数据 (Synthetic Data)...")
    
    # 定义两组完全不相交的词汇
    vocab_a = [f"sport_word_{i}" for i in range(50)] # 模拟体育词汇
    vocab_b = [f"biz_word_{i}" for i in range(50)]   # 模拟商业词汇
    
    def generate_docs(vocab, count, min_len=10, max_len=20):
        docs = []
        for _ in range(count):
            # 随机生成句子
            length = random.randint(min_len, max_len)
            words = random.choices(vocab, k=length)
            docs.append(" ".join(words))
        return docs

    # 1. 训练集：混合 A 和 B，让 LDA 学会这两种模式
    train_a = generate_docs(vocab_a, 2000)
    train_b = generate_docs(vocab_b, 2000)
    train_corpus = train_a + train_b
    random.shuffle(train_corpus) # 打乱用于训练
    
    # 2. 验证集：用于计算 Topic A 的中心点
    # 我们希望系统认为 A 是"正常"的
    validation_source = generate_docs(vocab_a, 500)
    
    # 3. 数据流：A -> B -> A
    stream_normal_1 = generate_docs(vocab_a, 200)
    stream_shift = generate_docs(vocab_b, 200)
    stream_normal_2 = generate_docs(vocab_a, 200)
    
    return train_corpus, validation_source, stream_normal_1, stream_shift, stream_normal_2

# ================= 2. 训练与基准建立 =================
def train_and_get_centroid(train_docs, source_docs):
    print("⚙️ Training LDA on Synthetic Corpus...")
    
    # 词表是固定的，不需要过滤
    vectorizer = CountVectorizer()
    tf_train = vectorizer.fit_transform(train_docs)
    
    # 训练 LDA，强行让它把词汇分成 5 类 (哪怕我们只有2类真值，模拟真实情况)
    lda = LatentDirichletAllocation(n_components=5, random_state=42)
    lda.fit(tf_train)
    
    print("⚙️ Calculating Source Topic Centroid...")
    # 计算 Source (Topic A) 的中心点
    tf_source = vectorizer.transform(source_docs)
    topic_dist_source = lda.transform(tf_source)
    source_centroid = np.mean(topic_dist_source, axis=0).reshape(1, -1)
    
    return lda, vectorizer, source_centroid

# ================= 3. 运行流 =================
def run_experiment():
    # 1. 获取数据
    train, val_src, stream_1, stream_2, stream_3 = generate_synthetic_data()
    
    # 2. 训练模型
    lda, vec, centroid = train_and_get_centroid(train, val_src)
    
    # 3. 构造流
    full_stream = stream_1 + stream_2 + stream_3
    labels = ["Normal (A)"]*200 + ["Shift (B)"]*200 + ["Return (A)"]*200
    
    # 4. ADWIN 监控
    adwin = drift.ADWIN(delta=0.002)
    
    scores = []
    means = []
    drifts = []
    
    print("\n🚀 Running Synthetic Stream...")
    for i, text in enumerate(tqdm(full_stream)):
        # 转换
        tf = vec.transform([text])
        topic_dist = lda.transform(tf)
        
        # 计算相似度 (0~1)
        sim = cosine_similarity(topic_dist, centroid)[0][0]
        
        # 更新监控
        adwin.update(sim)
        
        scores.append(sim)
        means.append(adwin.estimation)
        
        if adwin.drift_detected:
            drifts.append(i)
            
    return scores, means, drifts, labels

# ================= 4. 绘图 =================
def plot_results(scores, means, drifts, labels):
    fig, ax = plt.subplots(figsize=(12, 6))
    x = range(len(scores))
    
    # 散点
    ax.scatter(x, scores, s=10, color='gray', alpha=0.3, label='Raw Similarity')
    # 均值线
    ax.plot(x, means, color='blue', linewidth=3, label='ADWIN Mean')
    
    # 阶段背景
    ax.axvspan(0, 200, color='green', alpha=0.1, label='Phase 1: Normal')
    ax.axvspan(200, 400, color='red', alpha=0.1, label='Phase 2: Shift')
    ax.axvspan(400, 600, color='green', alpha=0.1, label='Phase 3: Return')
    
    # 漂移线
    for d in drifts:
        ax.axvline(x=d, color='red', linestyle='--', linewidth=2)
        
    ax.set_title('Ideally Separated Data Shift Detection (LDA + ADWIN)', fontsize=14)
    ax.set_ylabel('Similarity to Source Topic', fontsize=12)
    ax.set_ylim(-0.1, 1.1)
    ax.legend(loc='lower left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('synthetic_shift.png', dpi=300)
    print("\n✅ Plot Saved: synthetic_shift.png")

if __name__ == "__main__":
    s, m, d, l = run_experiment()
    plot_results(s, m, d, l)