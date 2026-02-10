"""
构造 HotpotQA 4-Domain Shift 数据集
只生成 3 个数据集：
1. Sudden Shift (突变) - 4个domain依次切换
2. Gradual Drift (渐变) - 4个domain平滑过渡
3. Recurring Shift (周期性) - 4个domain循环出现
"""
import json
import random
from pathlib import Path
from typing import List, Dict

# 路径配置
HERE = Path(__file__).parent
TRIPLET_DIR = HERE.parent / "dataset_split_domain" / "hotpot_triplets"
OUTPUT_DIR = HERE / "hotpot_shifts"
OUTPUT_DIR.mkdir(exist_ok=True)

# Domain 配置（全部 4 个）
DOMAINS = ["0_entertainment", "1_stem", "2_humanities", "3_lifestyle"]
DOMAIN_NAMES = ["entertainment", "stem", "humanities", "lifestyle"]

def load_domain_data(domain_file: str) -> List[Dict]:
    """加载某个 domain 的所有 triplets"""
    file_path = TRIPLET_DIR / domain_file
    data = []
    
    if not file_path.exists():
        print(f"⚠️ 文件不存在: {file_path}")
        return data
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    
    return data

def save_shift_dataset(queries: List[Dict], output_file: Path, metadata: Dict = None):
    """保存 shift 数据集"""
    with open(output_file, 'w', encoding='utf-8') as f:
        # 保存元数据
        if metadata:
            f.write(json.dumps({"metadata": metadata}) + '\n')
        
        # 保存 queries
        for query in queries:
            f.write(json.dumps(query, ensure_ascii=False) + '\n')
    
    print(f"✅ 已保存: {output_file.name} ({len(queries)} queries)")

# ==================== 策略一：Sudden Shift ====================

def construct_sudden_shift(queries_per_domain: int = 125):
    """
    构造突变场景：Entertainment → STEM → Humanities → Lifestyle
    
    每个 domain 持续 queries_per_domain 个查询，然后瞬间切换
    
    时间轴：
        [0-124]: 100% Entertainment
        [125-249]: 100% STEM
        [250-374]: 100% Humanities
        [375-499]: 100% Lifestyle
    
    考察点：
        - Monitor 能多快检测到分布突变？
        - Controller 能多快完成 KB 更新？
    """
    print(f"\n{'='*60}")
    print(f"📊 构造 Sudden Shift: {' → '.join(DOMAIN_NAMES)}")
    print(f"{'='*60}")
    
    # 加载所有 domain 的数据
    all_data = {}
    for domain in DOMAINS:
        data = load_domain_data(f"{domain}.jsonl")
        if not data:
            print(f"❌ 数据加载失败: {domain}")
            return
        random.shuffle(data)
        all_data[domain] = data
    
    dataset = []
    
    # 突变模式：直接切换
    for domain_idx, domain in enumerate(DOMAINS):
        sampled = all_data[domain][:queries_per_domain]
        
        for query in sampled:
            step = len(dataset)
            dataset.append({
                **query,
                "step": step,
                "phase": domain_idx,
                "domain": domain
            })
        
        print(f"   Phase {domain_idx} ({domain}): steps {step-len(sampled)+1}-{step}")
    
    # 元数据
    metadata = {
        "shift_type": "sudden",
        "domains": DOMAINS,
        "domain_sequence": " → ".join(DOMAIN_NAMES),
        "total_queries": len(dataset),
        "queries_per_domain": queries_per_domain,
        "shift_points": [queries_per_domain * (i+1) for i in range(len(DOMAINS)-1)],
        "description": "Sudden shift across 4 domains: Entertainment → STEM → Humanities → Lifestyle"
    }
    
    # 保存
    output_file = OUTPUT_DIR / "sudden_4domains.jsonl"
    save_shift_dataset(dataset, output_file, metadata)

# ==================== 策略二：Gradual Drift ====================

def construct_gradual_drift(queries_per_transition: int = 125, transition_phases: int = 5):
    """
    构造渐变场景：相邻 domain 之间平滑过渡（避免重合）
    
    修改策略：
        Ent → STEM: Phase 0-3 (100% → 75% → 50% → 25%)  [跳过 0%]
        STEM → Hum: Phase 1-4 (75% → 50% → 25% → 0%)   [跳过 100%]
        Hum → Life: Phase 1-4 (75% → 50% → 25% → 0%)   [跳过 100%]
    
    时间轴：
        [0-24]:   100% Ent
        [25-49]:  75% Ent + 25% STEM
        [50-74]:  50% Ent + 50% STEM
        [75-99]:  25% Ent + 75% STEM
        [100-124]: 75% STEM + 25% Hum    ← 直接进入混合阶段
        [125-149]: 50% STEM + 50% Hum
        [150-174]: 25% STEM + 75% Hum
        [175-199]: 75% Hum + 25% Life    ← 直接进入混合阶段
        [200-224]: 50% Hum + 50% Life
        [225-249]: 25% Hum + 75% Life
        [250-274]: 100% Life
    
    总计: 275 queries (无重合)
    """
    print(f"\n{'='*60}")
    print(f"📊 构造 Gradual Drift: {' → '.join(DOMAIN_NAMES)}")
    print(f"{'='*60}")
    
    # 加载所有 domain 的数据
    all_data = {}
    for domain in DOMAINS:
        data = load_domain_data(f"{domain}.jsonl")
        if not data:
            print(f"❌ 数据加载失败: {domain}")
            return
        random.shuffle(data)
        all_data[domain] = data
    
    dataset = []
    queries_per_phase = queries_per_transition // transition_phases
    
    # 渐变模式：相邻 domain 之间渐变
    for i in range(len(DOMAINS) - 1):
        domain_a = DOMAINS[i]
        domain_b = DOMAINS[i + 1]
        
        print(f"\n   Transition {i}: {domain_a} → {domain_b}")
        
        # 第一个过渡：使用 phase 0-3（100%, 75%, 50%, 25%）
        # 后续过渡：使用 phase 1-4（75%, 50%, 25%, 0%）
        if i == 0:
            phase_range = range(0, transition_phases - 1)  # 0, 1, 2, 3
            print(f"      使用 Phase 0-{transition_phases-2} (跳过最后的纯净阶段)")
        else:
            phase_range = range(1, transition_phases)      # 1, 2, 3, 4
            print(f"      使用 Phase 1-{transition_phases-1} (跳过第一个纯净阶段)")
        
        for phase in phase_range:
            # 计算当前阶段的比例
            ratio_a = 1.0 - (phase / (transition_phases - 1))
            ratio_b = phase / (transition_phases - 1)
            
            num_a = int(queries_per_phase * ratio_a)
            num_b = queries_per_phase - num_a
            
            # 采样
            start_idx = len(dataset)  # 使用当前数据集长度作为采样起点
            phase_data_a = all_data[domain_a][start_idx % len(all_data[domain_a]): 
                                               (start_idx + num_a) % len(all_data[domain_a])]
            phase_data_b = all_data[domain_b][start_idx % len(all_data[domain_b]): 
                                               (start_idx + num_b) % len(all_data[domain_b])]
            
            # 如果采样跨越了数据边界，需要额外处理
            if len(phase_data_a) < num_a:
                phase_data_a.extend(all_data[domain_a][:num_a - len(phase_data_a)])
            if len(phase_data_b) < num_b:
                phase_data_b.extend(all_data[domain_b][:num_b - len(phase_data_b)])
            
            # 混合并打乱
            phase_data = phase_data_a + phase_data_b
            random.shuffle(phase_data)
            
            # 添加到数据集
            for query in phase_data:
                step = len(dataset)
                domain = domain_a if query in phase_data_a else domain_b
                dataset.append({
                    **query,
                    "step": step,
                    "transition": f"{domain_a}_to_{domain_b}",
                    "transition_id": i,
                    "phase": phase,
                    "domain": domain,
                    "ratio_a": ratio_a,
                    "ratio_b": ratio_b
                })
            
            print(f"      Phase {phase}: {ratio_a:.0%} {domain_a.split('_')[1]} + "
                  f"{ratio_b:.0%} {domain_b.split('_')[1]} (steps {step-len(phase_data)+1}-{step})")
    
    # 元数据
    metadata = {
        "shift_type": "gradual",
        "domains": DOMAINS,
        "domain_sequence": " → ".join(DOMAIN_NAMES),
        "total_queries": len(dataset),
        "queries_per_transition": queries_per_transition,
        "transition_phases": transition_phases,
        "description": "Gradual drift across 4 domains with smooth transitions (no overlap)",
        "note": "First transition uses phases 0-3, subsequent transitions use phases 1-4"
    }
    
    # 保存
    output_file = OUTPUT_DIR / "gradual_4domains.jsonl"
    save_shift_dataset(dataset, output_file, metadata)

# ==================== 策略三：Recurring Shift ====================

def construct_recurring_shift(queries_per_phase: int = 80, num_cycles: int = 2):
    """
    构造周期性场景：(Entertainment → STEM → Humanities → Lifestyle) × 2
    
    4个 domain 按顺序循环出现 2 次
    
    时间轴：
        Cycle 0: [0-79] Ent → [80-159] STEM → [160-239] Hum → [240-319] Life
        Cycle 1: [320-399] Ent → [400-479] STEM → [480-559] Hum → [560-639] Life
    
    考察点：
        - 能否识别并复用之前的 KB 内容？
        - 能否避免重复构建已见过的 domain KB？
    """
    print(f"\n{'='*60}")
    print(f"📊 构造 Recurring Shift: ({' → '.join(DOMAIN_NAMES)}) × {num_cycles}")
    print(f"{'='*60}")
    
    # 加载所有 domain 的数据
    all_data = {}
    for domain in DOMAINS:
        data = load_domain_data(f"{domain}.jsonl")
        if not data:
            print(f"❌ 数据加载失败: {domain}")
            return
        random.shuffle(data)
        all_data[domain] = data
    
    dataset = []
    
    for cycle in range(num_cycles):
        print(f"\n   Cycle {cycle}:")
        
        for domain_idx, domain in enumerate(DOMAINS):
            # 采样（允许重复使用）
            start_idx = (cycle * len(DOMAINS) + domain_idx) * queries_per_phase
            sampled_data = []
            for i in range(queries_per_phase):
                sampled_data.append(all_data[domain][(start_idx + i) % len(all_data[domain])])
            
            # 添加到数据集
            for query in sampled_data:
                step = len(dataset)
                dataset.append({
                    **query,
                    "step": step,
                    "cycle": cycle,
                    "phase": domain_idx,
                    "phase_name": f"{domain}_cycle{cycle}",
                    "domain": domain
                })
            
            print(f"      {domain} (steps {step-len(sampled_data)+1}-{step})")
    
    # 元数据
    metadata = {
        "shift_type": "recurring",
        "domains": DOMAINS,
        "domain_sequence": " → ".join(DOMAIN_NAMES),
        "total_queries": len(dataset),
        "queries_per_phase": queries_per_phase,
        "num_cycles": num_cycles,
        "pattern": f"({' → '.join(DOMAIN_NAMES)}) × {num_cycles}",
        "description": f"Recurring shift across 4 domains for {num_cycles} cycles"
    }
    
    # 保存
    output_file = OUTPUT_DIR / "recurring_4domains.jsonl"
    save_shift_dataset(dataset, output_file, metadata)

# ==================== 主函数 ====================

def main():
    """生成 3 个 domain shift 数据集"""
    
    print("\n" + "="*80)
    print("🚀 HotpotQA 4-Domain Shift 数据集构造")
    print("="*80)
    
    # 检查原始数据
    print("\n📂 检查原始数据...")
    for domain in DOMAINS:
        file_path = TRIPLET_DIR / f"{domain}.jsonl"
        if file_path.exists():
            with open(file_path, 'r') as f:
                num_lines = sum(1 for _ in f)
            print(f"   ✅ {domain}: {num_lines} triplets")
        else:
            print(f"   ❌ {domain}: 文件不存在")
            return
    
    # ==================== 1. Sudden Shift ====================
    print("\n" + "="*80)
    print("📊 1. Sudden Shift (突变)")
    print("="*80)
    construct_sudden_shift(queries_per_domain=125)
    
    # ==================== 2. Gradual Drift ====================
    print("\n" + "="*80)
    print("📊 2. Gradual Drift (渐变)")
    print("="*80)
    construct_gradual_drift(queries_per_transition=200, transition_phases=5)
    
    # ==================== 3. Recurring Shift ====================
    print("\n" + "="*80)
    print("📊 3. Recurring Shift (周期性)")
    print("="*80)
    construct_recurring_shift(queries_per_phase=62, num_cycles=2)
    
    print("\n" + "="*80)
    print("✅ 所有 Domain Shift 数据集构造完成！")
    print(f"📁 输出目录: {OUTPUT_DIR}")
    print("="*80)
    
    # 打印生成的文件
    print("\n📋 生成的文件列表:")
    all_files = sorted(OUTPUT_DIR.glob("*.jsonl"))
    for f in all_files:
        with open(f, 'r') as file:
            first_line = json.loads(file.readline())
            if "metadata" in first_line:
                meta = first_line["metadata"]
                print(f"\n   📄 {f.name}")
                print(f"      - 类型: {meta['shift_type']}")
                print(f"      - 序列: {meta['domain_sequence']}")
                print(f"      - 总查询数: {meta['total_queries']}")

if __name__ == "__main__":
    random.seed(42)
    main()