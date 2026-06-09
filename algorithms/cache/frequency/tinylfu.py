"""Frequency policy: TinyLFU (kept for ablation)."""
import numpy as np
from algorithms.cache.params import PARAMS as _P
from algorithms.cache.base import BaseStrategy
import logging
log = logging.getLogger("motivation")

class TinyLFU(BaseStrategy):
    """TinyLFU: frequency-sketch admission gate with LFU eviction.

    Core mechanism (Einziger et al., 2017 / Caffeine cache):
      - Access frequency is tracked per document (exact counts; in production
        a Count-Min Sketch approximates this over a bounded sliding window).
      - Admission gate: a candidate document d_new replaces victim d_old
        only if freq(d_new) >= freq(d_old).
      - New arriving documents always start at freq=0; once KB documents
        accumulate frequency they become progressively harder to evict.

    Admission source: random doc-arrival stream (identical to LRU/DocArrival).
    Eviction order: lowest frequency first (LFU).

    Key contrast with NQM:
      TinyLFU estimates frequency of OBSERVED (successful) accesses.
      NQM estimates unmet semantic demand from FAILED queries and projects
      it into document embedding space—a signal unavailable to any
      access-history policy.

    In the multi-hop setting, bridge documents are never directly retrieved
    (the query omits the bridge entity), so their frequency stays at 0.
    TinyLFU cannot protect them and performs no better than Static.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.rng = np.random.default_rng(_P.SEED + 700)
        self.all_ids = [d['doc_id'] for d in doc_pool]
        self.freq = {}   # pool_idx -> int (access count)

    def set_kb(self, ids):
        super().set_kb(ids)
        for did in self.kb:
            self.freq[self.d2p[did]] = 0

    def step(self, window_queries, window_query_embs, window_idx):
        if not self.kb:
            return
        kb_list = sorted(self.kb)
        kb_idx  = np.array([self.d2p[d] for d in kb_list])
        kb_emb  = self.doc_embs[kb_idx]
        norms   = np.linalg.norm(window_query_embs, axis=1, keepdims=True)
        nqe     = window_query_embs / np.clip(norms, 1e-10, None)
        q_kb    = nqe @ kb_emb.T
        max_s   = np.max(q_kb, axis=1)

        # 1) Increment frequency for docs that successfully serve queries.
        succ = max_s >= _P.SF_HIT_THRESH
        if succ.any():
            best_pos = np.argmax(q_kb[succ], axis=1)
            for pos in best_pos:
                pi = int(kb_idx[pos])
                self.freq[pi] = self.freq.get(pi, 0) + 1

        # 2) Process random doc arrivals through the TinyLFU admission gate.
        arrivals = self.rng.choice(
            self.all_ids, min(_P.DOC_ARRIVE, len(self.all_ids)), replace=False)
        self.maint_retrieval_cost += len(arrivals)

        # Eviction queue: ascending frequency (least-frequently-used first).
        kb_pos_set  = set(int(p) for p in kb_idx)
        evict_queue = sorted(kb_pos_set, key=lambda p: self.freq.get(p, 0))

        n  = 0
        ei = 0
        for did in arrivals:
            if n >= _P.DOC_ADD_CAP or ei >= len(evict_queue):
                break
            if did in self.kb:
                continue
            ni         = self.d2p[did]
            cand_freq  = self.freq.get(ni, 0)        # new arrivals: 0
            ep         = evict_queue[ei]
            victim_freq = self.freq.get(ep, 0)
            # TinyLFU gate: admit only if candidate has >= frequency as victim.
            if cand_freq >= victim_freq:
                self.kb.discard(self.p2d[ep])
                self.kb.add(did)
                self.freq.pop(ep, None)
                self.freq[ni] = 0
                ei += 1
                n  += 1
        self.update_cost += n


