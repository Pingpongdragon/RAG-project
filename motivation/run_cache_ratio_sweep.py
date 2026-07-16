#!/usr/bin/env python3
"""Run the current DRIP cache-ratio benchmark with one fixed protocol.

The script intentionally keeps dataset construction, cache ratios, strategy set,
and output names in one place.  Launch one dataset per GPU, for example:

    python motivation/run_cache_ratio_sweep.py squad_direct --gpu 0
    python motivation/run_cache_ratio_sweep.py streamingqa_official --gpu 1

All runs are causal: the shared initial cache is built from a pre-evaluation
query prefix, and a policy updates only after the current window is evaluated.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys


PROJECT_DIR = Path(__file__).resolve().parent.parent

DEFAULT_RATIOS = (0.01, 0.02, 0.05, 0.10)
DEFAULT_STRATEGIES = (
    "LRU",
    "FIFO",
    "TinyLFU",
    "GPTCacheStyle",
    "Proximity",
    "AgentRAGCache",
    "DRIP",
)
DRIP_METHOD_VERSION = "drip-reactive-v1"


PROTOCOLS = {
    "squad_direct": {
        "runner": "experiments/direct/run.py",
        "args": (
            "--datasets", "squad",
            "--n-windows", "50",
            "--window-size", "300",
            "--workload", "factorized_recurring",
            "--n-source", "4000",
            "--warmup-windows", "3",
        ),
        "result_dataset": "squad",
        "note": (
            "15k unique SQuAD questions over 4k paragraphs with natural "
            "multi-query support reuse under recurring sparse-evidence "
            "regimes; ARC-compatible direct workload"
        ),
    },
    "streamingqa_official": {
        "runner": "experiments/direct/run.py",
        "args": (
            "--datasets", "streamingqa_official",
            "--n-windows", "50",
            "--window-size", "500",
            "--workload", "natural_temporal",
            "--warmup-windows", "3",
            "--temporal-sampling", "window_span",
        ),
        "note": (
            "official DeepMind question_ts order joined to local evidence text; "
            "25k evaluation queries after a 1.5k-query causal prefix"
        ),
    },
}


def ratio_tag(ratio: float) -> str:
    """Return an unambiguous four-decimal ratio tag, e.g. 0.01 -> 0100."""

    return f"{int(round(float(ratio) * 10000)):04d}"


def result_path(protocol: dict, output_name: str) -> Path:
    runner = PROJECT_DIR / protocol["runner"]
    return runner.parent / "data" / output_name


def result_dataset_name(protocol_name: str) -> str:
    """Return the JSON dataset key when a protocol uses a public alias."""

    protocol = PROTOCOLS[protocol_name]
    return str(protocol.get("result_dataset", protocol_name))


def is_complete_result(
    path: Path,
    dataset: str,
    ratio: float,
    expected_strategies: tuple[str, ...] = (),
) -> bool:
    """Only skip a result when it is valid and matches the requested protocol."""

    if not path.exists():
        return False
    try:
        payload = json.loads(path.read_text())
        result = payload[dataset]
        config = result["config"]
        observed_ratio = float(config["kb_pool_ratio"])
        method_matches = (
            "DRIP" not in expected_strategies
            or result["summary"].get("DRIP", {}).get("method_version")
            == DRIP_METHOD_VERSION
        )
        return (
            config.get("initialization") == "causal-prefix"
            and abs(observed_ratio - float(ratio)) <= 1.0 / max(
                1, int(config["pool_size"])
            )
            and bool(result.get("summary"))
            and set(expected_strategies).issubset(result["summary"])
            and method_matches
        )
    except (KeyError, TypeError, ValueError, json.JSONDecodeError):
        return False


def build_command(
    python: str,
    dataset: str,
    ratio: float,
    strategies: tuple[str, ...],
    output_prefix: str = "drip_reactive_sweep",
) -> tuple[list[str], Path]:
    protocol = PROTOCOLS[dataset]
    output_name = (
        f"{output_prefix}_{dataset}_r{ratio_tag(ratio)}.json"
    )
    command = [
        python,
        protocol["runner"],
        *protocol["args"],
        "--kb-pool-ratio", str(float(ratio)),
        "--strategies", *strategies,
        "--output", output_name,
    ]
    return command, result_path(protocol, output_name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run one dataset over a causal hot-cache capacity grid."
    )
    parser.add_argument("dataset", choices=tuple(PROTOCOLS))
    parser.add_argument(
        "--ratios", nargs="+", type=float, default=list(DEFAULT_RATIOS)
    )
    parser.add_argument(
        "--strategies", nargs="+", default=list(DEFAULT_STRATEGIES)
    )
    parser.add_argument(
        "--output-prefix",
        default="drip_reactive_sweep",
        help=(
            "Result filename prefix. Use a distinct value for ablations so "
            "main sweep JSON files are never overwritten."
        ),
    )
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    ratios = tuple(float(value) for value in args.ratios)
    if any(not 0.0 < value <= 1.0 for value in ratios):
        parser.error("every cache ratio must be in (0, 1]")
    strategies = tuple(args.strategies)
    if not args.output_prefix.replace("_", "").replace("-", "").isalnum():
        parser.error("--output-prefix may contain only letters, digits, _ and -")

    environment = os.environ.copy()
    environment.setdefault("HF_HUB_OFFLINE", "1")
    environment.setdefault("TRANSFORMERS_OFFLINE", "1")
    if args.gpu is not None:
        environment["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    print(f"dataset={args.dataset}")
    print(f"protocol={PROTOCOLS[args.dataset]['note']}")
    print(f"ratios={ratios}")
    print(f"strategies={strategies}")
    for ratio in ratios:
        command, output_path = build_command(
            args.python,
            args.dataset,
            ratio,
            strategies,
            output_prefix=args.output_prefix,
        )
        print(f"RESULT {ratio:.2%}: {output_path}")
        if not args.force and is_complete_result(
            output_path, result_dataset_name(args.dataset), ratio, strategies
        ):
            print(f"SKIP complete: {output_path}")
            continue
        print("RUN " + " ".join(command))
        if args.dry_run:
            continue
        subprocess.run(
            command,
            cwd=PROJECT_DIR,
            env=environment,
            check=True,
        )
        print(f"DONE {ratio:.2%}: {output_path}")


if __name__ == "__main__":
    main()
