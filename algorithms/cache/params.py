"""Injectable strategy hyper-parameters for the cache-policy family.

All cache strategies read their constants from this module's ``PARAMS`` object
instead of hard-coding them or importing from a specific experiment's config.
Each experiment (motivation_1 / motivation_2 / benchmark) calls
``params.update(**overrides)`` at startup to inject its own values, so the
*same* strategy code reproduces each experiment's numbers without forking.

Defaults below are the motivation_2 (multi-hop / BGE-large) values. The
motivation_1 (StreamingQA / MiniLM) experiment overrides SF_HIT_THRESH etc.
"""


class _Params:
    # ── core gating ──
    SEED          = 42
    SF_HIT_THRESH = 0.62    # mo2/BGE default; mo1 overrides to 0.55

    # ── shared budgets ──
    WRITE_CAP   = 200       # max KB writes per window for all writers
    PROBE_TOPK  = 50        # per-probe candidate retrieval width

    # ── DocArrival (LightRAG/HippoRAG doc-arrival) ──
    DOC_ARRIVE  = 80
    DOC_ADD_CAP = 200       # = WRITE_CAP

    # ── KnowledgeEdit (RECIPE) ──
    EDIT_BATCH  = 200       # = WRITE_CAP

    # ── QueryDriven / SemFlow (demand-side) ──
    QD_TOP_K       = 50     # = PROBE_TOPK
    QD_REPLACE_CAP = 200    # = WRITE_CAP

    # ── RandomFIFO ──
    FIFO_BATCH = 40

    # ── OnDemandFetch (CRAG) ──
    FETCH_TOP_K = 50        # = PROBE_TOPK

    # ── LogDrivenArrival ──
    LOG_FIX_TOP_K   = 50    # = PROBE_TOPK
    LOG_FIX_CAP     = 200   # = WRITE_CAP
    LOG_LAG_WINDOWS = 5

    def update(self, **overrides):
        """Inject experiment-specific values. Unknown keys raise to catch typos."""
        for k, v in overrides.items():
            if not hasattr(self, k):
                raise KeyError(f"unknown strategy param: {k}")
            setattr(self, k, v)

    def as_dict(self):
        return {k: getattr(self, k) for k in dir(self)
                if k.isupper() and not k.startswith('_')}


PARAMS = _Params()
