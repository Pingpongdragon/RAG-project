"""Classical Adaptive Replacement Cache (ARC) baseline.

This is the four-list ARC algorithm of Megiddo and Modha (FAST 2003), not the
unrelated ``AgentRAGCache`` baseline in :mod:`algorithms.cache.paradigm_ref`.
ARC maintains two resident LRU lists and two metadata-only ghost lists:

``T1``
    resident objects seen once recently;
``T2``
    resident objects seen at least twice;
``B1`` / ``B2``
    ghost histories for objects evicted from ``T1`` / ``T2``.

The adaptive target ``p`` controls how much of the resident cache ARC prefers
to devote to ``T1``.  A hit in ``B1`` increases ``p`` (favor recency), whereas
a hit in ``B2`` decreases it (favor frequency).

The benchmark runners impose an additional system write budget.  Hits always
update ARC metadata.  Every miss performs one cold-store read; a miss is
admitted only while the current window still has write budget.  A throttled
miss is not put into a ghost list because ARC ghost entries represent objects
that were previously resident, not arbitrary bypassed requests.  Consequently
the implementation is exactly classical ARC whenever ``WRITE_CAP`` does not
bind, and a write-throttled ARC policy otherwise.
"""

from __future__ import annotations

from collections import OrderedDict
from typing import Hashable

import numpy as np

from algorithms.cache.base import BaseStrategy
from algorithms.cache.params import PARAMS as _P


class ClassicalARC(BaseStrategy):
    """Classical ARC with exact-access and semantic-query entry points.

    The public constructor matches the repository's other ``BaseStrategy``
    policies.  Capacity is inferred from the resident set supplied through
    :meth:`set_kb`, as it is for all evidence-trace runners.
    """

    def __init__(self, name, doc_pool, doc_embs, title_to_idx):
        super().__init__(name, doc_pool, doc_embs, title_to_idx)
        self.capacity = 0
        self.p = 0.0
        # OrderedDict order is LRU -> MRU.  Values carry no information.
        self.t1: OrderedDict[Hashable, None] = OrderedDict()
        self.t2: OrderedDict[Hashable, None] = OrderedDict()
        self.b1: OrderedDict[Hashable, None] = OrderedDict()
        self.b2: OrderedDict[Hashable, None] = OrderedDict()

    @staticmethod
    def _append_mru(queue, key):
        queue.pop(key, None)
        queue[key] = None

    @staticmethod
    def _pop_lru(queue):
        key, _ = queue.popitem(last=False)
        return key

    def set_kb(self, ids):
        """Initialize residents as a deterministic, unbiased ``T1`` batch."""

        super().set_kb(ids)
        self.capacity = len(self.kb)
        self.p = 0.0
        self.t1.clear()
        self.t2.clear()
        self.b1.clear()
        self.b2.clear()

        # A batch-loaded cache has no real LRU order.  Match the other
        # baselines: stabilize IDs, then seed the equal-recency tie order.
        stable_ids = sorted(self.kb, key=self.d2p.__getitem__)
        rng = np.random.default_rng(int(_P.SEED) + 709)
        for index in rng.permutation(len(stable_ids)):
            self.t1[stable_ids[int(index)]] = None
        self._assert_invariants()

    @property
    def target_size(self):
        """Current adaptive recency target ``p`` (for diagnostics/tests)."""

        return float(self.p)

    def _replace(self, incoming):
        """ARC ``REPLACE``: move one resident into the matching ghost list."""

        choose_t1 = bool(self.t1) and (
            len(self.t1) > self.p
            or (incoming in self.b2 and len(self.t1) == self.p)
        )
        if choose_t1 or not self.t2:
            victim = self._pop_lru(self.t1)
            self._append_mru(self.b1, victim)
        else:
            victim = self._pop_lru(self.t2)
            self._append_mru(self.b2, victim)
        self.kb.discard(victim)

    def _request(self, document_id, allow_write, *, count_cold_read=True):
        """Process one ARC request; return whether it consumed a cache write."""

        # Case I: resident hit.  Both T1 and T2 hits become T2 MRU.
        if document_id in self.t1:
            self.t1.pop(document_id)
            self._append_mru(self.t2, document_id)
            self._assert_invariants()
            return False
        if document_id in self.t2:
            self._append_mru(self.t2, document_id)
            self._assert_invariants()
            return False

        # Every non-resident access fetches the object from the cold store,
        # including a repeated miss after the window's write cap is exhausted.
        if count_cold_read:
            self.maint_retrieval_cost += 1
        if not allow_write or self.capacity <= 0:
            self._assert_invariants()
            return False

        # Case II: ghost hit in B1 -> favor recency.
        if document_id in self.b1:
            delta = max(1.0, len(self.b2) / max(1, len(self.b1)))
            self.p = min(float(self.capacity), self.p + delta)
            self._replace(document_id)
            self.b1.pop(document_id)
            self._append_mru(self.t2, document_id)
            self.kb.add(document_id)
            self._assert_invariants()
            return True

        # Case III: ghost hit in B2 -> favor frequency.
        if document_id in self.b2:
            delta = max(1.0, len(self.b1) / max(1, len(self.b2)))
            self.p = max(0.0, self.p - delta)
            self._replace(document_id)
            self.b2.pop(document_id)
            self._append_mru(self.t2, document_id)
            self.kb.add(document_id)
            self._assert_invariants()
            return True

        # Case IV: completely new object.  This follows Algorithm 1 from the
        # ARC paper, including the two distinct |T1|+|B1| == c branches.
        if len(self.t1) + len(self.b1) == self.capacity:
            if len(self.t1) < self.capacity:
                self._pop_lru(self.b1)
                self._replace(document_id)
            else:
                victim = self._pop_lru(self.t1)
                self.kb.discard(victim)
        elif len(self.t1) + len(self.b1) < self.capacity:
            total = len(self.t1) + len(self.t2) + len(self.b1) + len(self.b2)
            if total >= self.capacity:
                if total == 2 * self.capacity:
                    self._pop_lru(self.b2)
                self._replace(document_id)

        self._append_mru(self.t1, document_id)
        self.kb.add(document_id)
        self._assert_invariants()
        return True

    def step(self, window_queries, window_query_embs, window_idx):
        """Replay one request window under a shared physical-write budget."""

        if self.capacity <= 0:
            return
        accesses = self._observed_access_positions(window_queries)
        writes = 0
        if accesses is not None:
            for pool_index in accesses:
                document_id = self.p2d[int(pool_index)]
                wrote = self._request(
                    document_id, writes < max(0, int(_P.WRITE_CAP))
                )
                writes += int(wrote)
            self.update_cost += writes
            return

        # Compatibility path for ordinary semantic-query workloads.  Request
        # order remains sequential because each ARC mutation affects the next
        # query's cache state.
        embeddings = np.asarray(window_query_embs)
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        normalized = embeddings / np.clip(norms, 1e-10, None)
        for query_embedding in normalized:
            if not self.kb:
                break
            residents = sorted(self.kb, key=self.d2p.__getitem__)
            resident_positions = np.asarray(
                [self.d2p[document_id] for document_id in residents],
                dtype=np.int64,
            )
            resident_scores = self.doc_embs[resident_positions] @ query_embedding
            best_resident = int(np.argmax(resident_scores))
            if float(resident_scores[best_resident]) >= _P.SF_HIT_THRESH:
                requested_id = residents[best_resident]
                cold_read_already_counted = False
            else:
                pool_scores = self.doc_embs @ query_embedding
                requested_id = self.p2d[int(np.argmax(pool_scores))]
                # The full-pool escalation itself is a cold read even in the
                # degenerate case where its top-1 object is already resident.
                self.maint_retrieval_cost += 1
                cold_read_already_counted = True
            wrote = self._request(
                requested_id,
                writes < max(0, int(_P.WRITE_CAP)),
                count_cold_read=not cold_read_already_counted,
            )
            writes += int(wrote)
        self.update_cost += writes

    def _assert_invariants(self):
        """Cheap internal checks; cache sizes here are small by construction."""

        resident = set(self.t1) | set(self.t2)
        ghost = set(self.b1) | set(self.b2)
        assert set(self.t1).isdisjoint(self.t2)
        assert set(self.b1).isdisjoint(self.b2)
        assert resident.isdisjoint(ghost)
        assert resident == self.kb
        assert len(resident) <= self.capacity
        assert len(self.b1) <= self.capacity
        assert len(self.b2) <= self.capacity
        assert len(resident) + len(ghost) <= 2 * self.capacity
        assert 0.0 <= self.p <= float(self.capacity)
