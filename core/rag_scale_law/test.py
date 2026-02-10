import time
import json
from pathlib import Path
from typing import List, Dict

try:
    import psutil
except ImportError:
    psutil = None
    print("⚠️ 未安装 psutil，内存占用统计将禁用。请运行: pip install psutil")

from tqdm import tqdm
import sys
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))

# 项目内依赖
from kb_base import ClusteredKnowledgeBase, load_kb_documents
from evaluator import load_test_data
from RAG_project.models.embeddings import embedding_service


def build_kb(doc_pool: Dict[str, List], total_capacity: int) -> ClusteredKnowledgeBase:
    """
    根据总容量构建 KB（平均分配到各 domain）
    """
    kb = ClusteredKnowledgeBase(capacity=total_capacity)
    per_domain_cap = total_capacity // 4

    for domain, docs in doc_pool.items():
        # 只取每域的前 per_domain_cap 个文档
        for doc in docs[:per_domain_cap]:
            kb.add_document(doc, step=0)

    return kb


def measure_latency(kb: ClusteredKnowledgeBase, queries: List[Dict], query_embeddings) -> float:
    """
    仅测检索耗时（不包含 embedding 编码），返回平均毫秒/查询
    """
    latencies_ms = []

    # 预热，避免首次开销影响结果
    warmup = min(50, len(queries))
    for i in range(warmup):
        _ = kb.search(query_embeddings[i], queries[i]["domain"], step=i, top_k=10)

    # 正式测量
    for i, q in enumerate(tqdm(queries, desc="Measuring retrieval latency")):
        qv = query_embeddings[i]
        t0 = time.perf_counter()
        _ = kb.search(qv, q["domain"], step=i, top_k=10)
        t1 = time.perf_counter()
        latencies_ms.append((t1 - t0) * 1000)

    avg_ms = sum(latencies_ms) / len(latencies_ms) if latencies_ms else 0.0
    return avg_ms


def measure_memory_after_build(kb_before_bytes: int) -> int:
    """
    返回构建后进程的 RSS（字节）。需要 psutil。
    """
    if psutil is None:
        return -1
    proc = psutil.Process()
    rss = proc.memory_info().rss  # bytes
    return rss - kb_before_bytes


def auto_capacity_list(doc_pool: Dict[str, List]) -> List[int]:
    """
    根据数据池自动生成容量列表（总容量），避免超过可用文档数
    例如：总可用 32k → [8k, 16k, 24k, 32k]
    """
    min_per_domain = min(len(docs) for docs in doc_pool.values())
    max_total = 4 * min_per_domain

    steps = [0.25, 0.5, 0.75, 1.0]
    sizes = sorted({int(max_total * s) for s in steps if int(max_total * s) > 0})
    return sizes


def benchmark():
    print("📚 加载文档池...")
    doc_pool = load_kb_documents()

    # 自动生成容量列表
    capacities = auto_capacity_list(doc_pool)
    print(f"🔧 测试总容量: {capacities} (平均每域 = 总容量/4)")

    # 加载固定查询集并批量编码（避免计入编码时间）
    print("🔎 加载查询数据并编码（sudden shift，500 条）...")
    queries = load_test_data(shift_type="sudden")
    query_texts = [q["query"] for q in queries]
    query_embeddings = embedding_service.encode(
        query_texts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    # 统计磁盘占用（数据集原始 KB 文件总大小，仅供参考）
    kb_dir = Path(__file__).parent / "dataset_split_domain" / "hotpot_kb"
    disk_bytes = 0
    if kb_dir.exists():
        for p in kb_dir.glob("*.jsonl"):
            disk_bytes += p.stat().st_size

    results = []

    # 基准进程内存
    base_rss = 0
    if psutil:
        base_rss = psutil.Process().memory_info().rss

    for total_cap in capacities:
        print(f"\n==============================================")
        print(f"⚙️ 构建 KB，总容量={total_cap}（每域≈{total_cap//4}）")

        kb = build_kb(doc_pool, total_cap)

        # 测检索平均耗时
        avg_ms = measure_latency(kb, queries, query_embeddings)

        # 测内存占用（构建后相对基线的增量）
        mem_delta_bytes = measure_memory_after_build(base_rss) if psutil else -1

        row = {
            "total_capacity": total_cap,
            "per_domain_capacity": total_cap // 4,
            "avg_latency_ms_per_query": round(avg_ms, 3),
            "memory_delta_MB": round(mem_delta_bytes / (1024 * 1024), 2) if mem_delta_bytes >= 0 else None,
            "kb_disk_usage_MB_reference": round(disk_bytes / (1024 * 1024), 2) if disk_bytes > 0 else None
        }
        results.append(row)

        print(f"✅ 结果: {row}")

    # 保存结果
    out_path = Path(__file__).parent / "kb_benchmark_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n📄 结果已保存 → {out_path}")
    print("📈 请将 avg_latency_ms_per_query 与 memory_delta_MB 画成两张曲线（随容量递增）")


if __name__ == "__main__":
    benchmark()