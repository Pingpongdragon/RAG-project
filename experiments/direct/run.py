#!/usr/bin/env python3
"""Direct-evidence 正式实验命令行入口。

职责仅有四项：解析参数、逐数据集调用 ``experiment.run_dataset``、保存 JSON、绘图。
数据读入与流构造见 ``workload.py``，因果评估见 ``evaluation.py``，指标汇总见
``reporting.py``。
"""

import argparse
import json
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
sys.path.append(str(PROJECT_ROOT))

if __package__:
    from .config import DATASET_CONFIGS, DATA_DIR, STRATEGY_ORDER, log
    from .experiment import run_dataset
    from .plotting import generate_figures
    from .reporting import print_summary
else:
    sys.path.insert(0, str(THIS_DIR))
    from config import DATASET_CONFIGS, DATA_DIR, STRATEGY_ORDER, log
    from experiment import run_dataset
    from plotting import generate_figures
    from reporting import print_summary

from experiments.common.factorized_workload import WORKLOAD_CHOICES


def parse_args():
    """解析命令行参数；默认使用无未来标签的 causal protocol。"""

    parser = argparse.ArgumentParser(description="DRIP direct-evidence 实验入口")
    parser.add_argument("--n-windows", type=int, default=50)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument(
        "--workload",
        choices=sorted(WORKLOAD_CHOICES),
        default="auto",
        help=(
            "auto: 时间日志保持官方顺序，普通 QA 使用 factorized_recurring；"
            "也可显式选择 natural_temporal 或 factorized_* 对照。"
        ),
    )
    parser.add_argument(
        "--freeze-until",
        type=int,
        default=0,
        help="冻结非 Oracle 策略在 [0, freeze_until) 窗口内的更新。",
    )
    parser.add_argument(
        "--factorized-min-support-frequency",
        type=int,
        default=1,
        help=(
            "每个 required support 在源数据中至少出现的 query 数；"
            "只影响 factorized_* workload。"
        ),
    )
    parser.add_argument(
        "--factorized-family-mode",
        choices=["auto", "exact", "anchor"],
        default="auto",
        help=(
            "factorized family 定义：single-support 默认 exact，"
            "direct multi-support 默认按共享 anchor。"
        ),
    )
    parser.add_argument("--n-source", type=int, default=None)
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASET_CONFIGS),
        default=["squad"],
    )
    parser.add_argument("--strategies", nargs="+", default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument(
        "--kb-budget",
        type=int,
        default=None,
        help="手动指定 KB budget；优先级高于 --kb-pool-ratio。",
    )
    parser.add_argument(
        "--kb-pool-ratio",
        type=float,
        default=None,
        help="按 pool 大小设置 KB budget，例如 0.1 表示 KB/pool=1/10。",
    )
    parser.add_argument(
        "--warmup-windows",
        type=int,
        default=3,
        help="因果初始化使用的评估前历史窗口数，不计入指标。",
    )
    parser.add_argument(
        "--temporal-sampling",
        choices=["prefix", "window_span"],
        default="prefix",
        help=(
            "自然时间流的采样方式：prefix 使用日志开头；"
            "window_span 覆盖完整时间且保持窗口内连续。"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    strategy_names = args.strategies or STRATEGY_ORDER
    all_results = {}
    for dataset_name in args.datasets:
        dataset_config = DATASET_CONFIGS[dataset_name].copy()
        dataset_config.update({
            "n_windows": args.n_windows,
            "window_size": args.window_size,
            "freeze_until": args.freeze_until,
        })
        log.info(
            "\n%s\n  Running: %s (workload=%s)\n%s",
            "=" * 60,
            dataset_name,
            args.workload,
            "=" * 60,
        )
        all_results[dataset_name] = run_dataset(
            dataset_name,
            dataset_config,
            strategy_names,
            loader_name=dataset_name,
            n_source=args.n_source,
            kb_budget_override=args.kb_budget,
            kb_pool_ratio=args.kb_pool_ratio,
            warmup_windows=args.warmup_windows,
            temporal_sampling=args.temporal_sampling,
            workload=args.workload,
            factorized_min_support_frequency=(
                args.factorized_min_support_frequency
            ),
            factorized_family_mode=args.factorized_family_mode,
        )

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    freeze_tag = (
        f"_freeze{args.freeze_until}" if args.freeze_until > 0 else ""
    )
    output_name = args.output or (
        f"results_{args.n_windows}w{freeze_tag}_{args.workload}.json"
    )
    output_path = DATA_DIR / output_name
    with open(output_path, "w") as output_file:
        json.dump(all_results, output_file, indent=2)
    log.info("Saved to %s", output_path)

    print_summary(all_results, strategy_names)
    generate_figures(
        all_results,
        strategy_names,
        suffix=f"_{Path(output_name).stem}",
    )


if __name__ == "__main__":
    main()
