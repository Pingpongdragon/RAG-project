#!/usr/bin/env python3
"""把多个 runner JSON 汇总成一个跨数据集 Markdown 表。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


LABELS = {
    "AgentRAGCache": "ARC",
    "DRIP": "DRIP-Reactive",
}


def full_stream_recall(summary: dict) -> float:
    """优先使用逐窗口值，避免把不同长度的 H1/H2 平均两次。"""

    values = summary.get("recall@5_per_window", ())
    if values:
        return float(np.mean(values))
    halves = [
        float(summary[key])
        for key in ("recall@5_h1", "recall@5_h2")
        if key in summary
    ]
    return float(np.mean(halves)) if halves else 0.0


def load_rows(paths: list[str], methods: tuple[str, ...]) -> list[dict]:
    rows = []
    for raw_path in paths:
        path = Path(raw_path)
        payload = json.loads(path.read_text())
        for dataset, result in payload.items():
            config = result["config"]
            ratio = 100.0 * float(
                config.get(
                    "kb_pool_ratio",
                    float(config["kb_budget"]) / max(1, int(config["pool_size"])),
                )
            )
            summaries = result["summary"]
            selected = methods or tuple(summaries)
            missing = set(selected) - set(summaries)
            if missing:
                raise KeyError(
                    f"{path} is missing requested methods: {sorted(missing)}"
                )
            for method in selected:
                summary = summaries[method]
                rows.append({
                    "dataset": dataset,
                    "ratio": ratio,
                    "method": method,
                    "has_answer": float(summary["has_answer_rate"]),
                    "amat": float(summary["amat"]),
                    "recall": full_stream_recall(summary),
                    "replacements": int(summary["replacement_count"]),
                    "source": str(path.resolve()),
                })
    return sorted(
        rows,
        key=lambda row: (row["dataset"], row["ratio"], row["method"]),
    )


def write_markdown(rows: list[dict], output: str) -> Path:
    lines = [
        "| Dataset | Ratio | Method | Has-Answer | AMAT | Recall@5 | Replacements |",
        "|:---|---:|:---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['dataset']} | {row['ratio']:.2f}% | "
            f"{LABELS.get(row['method'], row['method'])} | "
            f"{row['has_answer']:.2f} | {row['amat']:.3f} | "
            f"{row['recall']:.2f} | {row['replacements']} |"
        )
    lines.extend(["", "## Source JSON", ""])
    for source in sorted({row["source"] for row in rows}):
        lines.append(f"- `{source}`")
    output_path = Path(output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines) + "\n")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Summarize multiple cache runner JSON files."
    )
    parser.add_argument("paths", nargs="+")
    parser.add_argument("--methods", nargs="+", default=())
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = load_rows(args.paths, tuple(args.methods))
    output_path = write_markdown(rows, args.output)
    print(f"WROTE {len(rows)} rows: {output_path}")


if __name__ == "__main__":
    main()
