"""Paradigm references: doc-arrival / knowledge-edit / random."""
import numpy as np
from algorithms.cache.params import PARAMS as _P
from algorithms.cache.base import BaseStrategy
import logging
log = logging.getLogger("motivation")

class Static(BaseStrategy):
    """No-update baseline.  KB is frozen after initialisation."""
    def step(self, window_queries, window_query_embs, window_idx):
        pass



class DocArrival(BaseStrategy):
    """Document-arrival-driven KB update (HippoRAG2 / LightRAG style).

    Each window, _P.DOC_ARRIVE documents are randomly sampled from the full pool
    (simulating new documents arriving in a real system). For each arrival:
      - If similarity to some KB doc > 0.7: replace that KB doc (update).
      - If similarity to all KB docs < 0.3: evict the stalest KB doc (insert).
      - Otherwise: skip (not novel enough and not redundant enough).
    At most _P.DOC_ADD_CAP replacements per window.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.rng = np.random.default_rng(_P.SEED + 100)
        self.all_ids = [d['doc_id'] for d in doc_pool]
        self.ts = {}

    def step(self, window_queries, window_query_embs, window_idx):
        if not self.kb:
            return
        kb_list = sorted(self.kb)
        kb_idx = np.array([self.d2p[d] for d in kb_list])
        kb_emb = self.doc_embs[kb_idx]
        arrivals = self.rng.choice(self.all_ids,
                                   min(_P.DOC_ARRIVE, len(self.all_ids)),
                                   replace=False)
        self.maint_retrieval_cost += len(arrivals)
        n = 0
        for did in arrivals:
            if n >= _P.DOC_ADD_CAP:
                break
            if did in self.kb:
                self.ts[did] = window_idx
                continue
            ni = self.d2p[did]
            ne = self.doc_embs[ni]
            sims = kb_emb @ ne
            best = float(np.max(sims))
            if best > 0.7:
                pos = int(np.argmax(sims))
                old = kb_list[pos]
                self.kb.discard(old)
                self.kb.add(did)
                self.ts[did] = window_idx
                kb_list[pos] = did
                kb_idx[pos] = ni
                kb_emb[pos] = ne
                n += 1
            elif best < 0.3:
                stale = min(self.kb, key=lambda d: self.ts.get(d, -1))
                self.kb.discard(stale)
                self.kb.add(did)
                self.ts[did] = window_idx
                n += 1
        self.update_cost += n



class KnowledgeEdit(BaseStrategy):
    """Knowledge-edit-driven KB update (RECIPE style).

    Each window, _P.EDIT_BATCH KB documents are randomly selected for "editing".
    For each, find the most similar non-KB document (0.4 < sim < 0.8) and
    swap it in.  This models the RECIPE paradigm where a knowledge graph is
    continuously revised via local edits rather than wholesale replacement.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.rng = np.random.default_rng(_P.SEED + 200)

    def step(self, window_queries, window_query_embs, window_idx):
        kb_list = sorted(self.kb)
        if not kb_list:
            return
        n_ed = min(_P.EDIT_BATCH, len(kb_list))
        targets = self.rng.choice(kb_list, n_ed, replace=False)
        self.maint_retrieval_cost += n_ed  # n_ed full-pool NN scans
        n = 0
        for tid in targets:
            tpi = self.d2p[tid]
            te = self.doc_embs[tpi]
            sims = self.doc_embs @ te
            for d in self.kb:
                if d in self.d2p:
                    sims[self.d2p[d]] = -1
            cands = np.where((sims > 0.4) & (sims < 0.8))[0]
            if len(cands) == 0:
                continue
            best = cands[np.argmax(sims[cands])]
            self.kb.discard(tid)
            self.kb.add(self.p2d[best])
            n += 1
        self.update_cost += n



class RandomFIFO(BaseStrategy):
    """Blind supply-side: random new docs replace oldest KB entries (FIFO).

    Models a naive scheduled ingest pipeline that periodically refreshes
    KB content without any relevance signal.  Each window, _P.FIFO_BATCH docs
    are randomly drawn from pool and replace the oldest-inserted KB entries.
    Demonstrates that blind supply-side updates inject noise faster than
    useful content, especially under drift.
    """
    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.rng = np.random.default_rng(_P.SEED + 300)
        self.all_ids = [d['doc_id'] for d in doc_pool]
        self.insert_order = []  # track insertion order for FIFO eviction

    def set_kb(self, ids):
        super().set_kb(ids)
        self.insert_order = list(ids)

    def step(self, window_queries, window_query_embs, window_idx):
        batch = min(_P.FIFO_BATCH, len(self.all_ids))
        arrivals = self.rng.choice(self.all_ids, batch, replace=False)
        self.maint_retrieval_cost += batch
        n = 0
        for did in arrivals:
            if did in self.kb:
                continue
            if not self.insert_order:
                break
            old = self.insert_order.pop(0)
            self.kb.discard(old)
            self.kb.add(did)
            self.insert_order.append(did)
            n += 1
        self.update_cost += n


