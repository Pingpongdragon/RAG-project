"""Retired legacy ablation entry point.

The old benchmark patched ``KBUpdateAgent`` inside ``algorithms.drip``. That
agent pipeline has been removed; the canonical DRIP algorithm now lives in
``algorithms.drip.support_flow`` and is evaluated through motivation_2 against
the retained QueryDriven direct-demand baseline.
"""


def main():
    raise SystemExit(
        "benchmark/ablation_drift.py is retired. Run motivation/motivation_2/run.py "
        "with strategies QueryDriven DRIP for the current ablation."
    )


if __name__ == "__main__":
    main()
