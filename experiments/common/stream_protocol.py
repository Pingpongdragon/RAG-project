"""实验流的因果采样、热缓存初始化与离线协议审计工具。

在线采样与初始化只使用时间顺序、query embedding 和文档 embedding，不读取
gold support 或未来窗口。实验结束后的诊断函数可以读取 ``sf_titles``、regime 与
visibility 标签，但只用于报告 drift、reuse、predictability 和 leakage，绝不把这些
量反馈给 cache policy。
"""

from dataclasses import dataclass
import hashlib

import numpy as np


def embedding_content_fingerprint(doc_pool, queries):
    """为 embedding cache 生成与实际 encoder 输入绑定的稳定指纹。

    只用 pool/query 数量会让相同规模的不同 seed 数据误读同一个 cache。本函数只
    哈希 title/text/question，不读取 gold support 标签。
    """

    digest = hashlib.sha256()
    for document in doc_pool:
        digest.update(str(document.get("title", "")).encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(document.get("text", "")).encode("utf-8"))
        digest.update(b"\n")
    digest.update(b"\xff")
    for query in queries:
        digest.update(str(query.get("question", "")).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()[:16]


@dataclass
class SamplingDiagnostics:
    """查询流中 exact-query 重复的可审计统计。"""

    total: int
    unique: int
    duplicates: int

    @property
    def duplicate_rate(self):
        return float(self.duplicates / max(1, self.total))

    def as_dict(self):
        return {
            "total": int(self.total),
            "unique": int(self.unique),
            "duplicates": int(self.duplicates),
            "duplicate_rate": round(self.duplicate_rate, 6),
        }


@dataclass
class WarmupOverlapDiagnostics:
    """Warm-up history 与正式评估流的 exact-query 重叠统计。"""

    requested: int
    actual: int
    unique: int
    evaluation_overlap: int

    def as_dict(self):
        return {
            "requested": int(self.requested),
            "actual": int(self.actual),
            "unique": int(self.unique),
            "evaluation_overlap": int(self.evaluation_overlap),
            "overlap_rate": round(
                self.evaluation_overlap / max(1, self.actual), 6),
        }


@dataclass
class TemporalSamplingDiagnostics:
    """自然时间流下采样对原始时间跨度的覆盖统计。"""

    mode: str
    source_events: int
    selected_events: int
    source_start: int | None
    source_end: int | None
    evaluation_start: int | None
    evaluation_end: int | None
    block_size: int | None = None
    selected_blocks: int | None = None

    def as_dict(self):
        source_span = (
            int(self.source_end - self.source_start)
            if self.source_start is not None and self.source_end is not None
            else None
        )
        evaluation_span = (
            int(self.evaluation_end - self.evaluation_start)
            if self.evaluation_start is not None
            and self.evaluation_end is not None
            else None
        )
        return {
            "mode": str(self.mode),
            "source_events": int(self.source_events),
            "selected_events": int(self.selected_events),
            "source_start": self.source_start,
            "source_end": self.source_end,
            "source_span_seconds": source_span,
            "evaluation_start": self.evaluation_start,
            "evaluation_end": self.evaluation_end,
            "block_size": self.block_size,
            "selected_blocks": self.selected_blocks,
            "evaluation_span_seconds": evaluation_span,
            "span_coverage": round(
                evaluation_span / max(1, source_span), 6)
            if evaluation_span is not None and source_span is not None
            else None,
        }


@dataclass
class SupportReuseDiagnostics:
    """Gold support 的后验可缓存性统计。

    这些量只用于评价一个已构造 query stream 是否存在可学习的 evidence reuse，
    绝不参与初始化、路由或 cache admission。
    """

    queries_with_support: int
    support_occurrences: int
    unique_supports: int
    repeated_support_occurrences: int
    queries_answerable_from_past: int
    adjacent_window_jaccard_mean: float
    max_support_frequency: int

    def as_dict(self):
        return {
            "queries_with_support": int(self.queries_with_support),
            "support_occurrences": int(self.support_occurrences),
            "unique_supports": int(self.unique_supports),
            "repeated_support_occurrences": int(
                self.repeated_support_occurrences),
            "repeated_support_rate": round(
                self.repeated_support_occurrences
                / max(1, self.support_occurrences),
                6,
            ),
            "queries_answerable_from_past": int(
                self.queries_answerable_from_past),
            "past_answerable_query_rate": round(
                self.queries_answerable_from_past
                / max(1, self.queries_with_support),
                6,
            ),
            "adjacent_window_jaccard_mean": round(
                float(self.adjacent_window_jaccard_mean), 6),
            "max_support_frequency": int(self.max_support_frequency),
        }


@dataclass
class QueryDriftDiagnostics:
    """不读取 gold 的 query-stream 分布变化统计。"""

    centroid_cosine_shift: float
    regime_js_divergence: float
    adjacent_centroid_shift_mean: float
    adjacent_centroid_shift_max: float

    def as_dict(self):
        return {
            "centroid_cosine_shift": round(
                float(self.centroid_cosine_shift), 6),
            "regime_js_divergence": round(
                float(self.regime_js_divergence), 6),
            "adjacent_centroid_shift_mean": round(
                float(self.adjacent_centroid_shift_mean), 6),
            "adjacent_centroid_shift_max": round(
                float(self.adjacent_centroid_shift_max), 6),
        }


@dataclass
class WorkloadFactorDiagnostics:
    """漂移、复用与可预测性三者的独立后验审计。

    ``causal_transition_accuracy`` 只允许用当前转移之前的 regime-transition
    counts 预测下一个 regime；第一次出现的转移记为不可预测。它因此不会把一个
    只发生一次的 A->B one-shot shift 误报成“完全可预测”。
    """

    windows: int
    regimes: int
    cross_regime_transitions: int
    cross_regime_evidence_jaccard_mean: float
    within_regime_repeated_support_rate: float
    causal_transition_accuracy: float
    recurrent_transition_rate: float
    labeled_visibility_rate: float

    def as_dict(self):
        return {
            "windows": int(self.windows),
            "regimes": int(self.regimes),
            "cross_regime_transitions": int(self.cross_regime_transitions),
            "cross_regime_evidence_jaccard_mean": round(
                float(self.cross_regime_evidence_jaccard_mean), 6),
            "drift_magnitude": round(
                1.0 - float(self.cross_regime_evidence_jaccard_mean), 6),
            "within_regime_repeated_support_rate": round(
                float(self.within_regime_repeated_support_rate), 6),
            "causal_transition_accuracy": round(
                float(self.causal_transition_accuracy), 6),
            "recurrent_transition_rate": round(
                float(self.recurrent_transition_rate), 6),
            "labeled_visibility_rate": round(
                float(self.labeled_visibility_rate), 6),
        }


def query_identity(query):
    """返回 query 的稳定身份；空的 ``source_qidx`` 不得覆盖有效 ``qidx``。"""

    if not isinstance(query, dict):
        return id(query)
    source_qidx = query.get("source_qidx")
    if source_qidx is not None:
        return source_qidx
    qidx = query.get("qidx")
    return qidx if qidx is not None else id(query)


def stream_sampling_diagnostics(stream):
    """按原始事件身份统计 exact-query 重复。

    自然日志抽样后，``qidx`` 会被重编号为连续的 embedding 行号；稳定身份保存在
    ``source_qidx``。受控流没有该字段时继续使用 ``qidx``。
    """

    identities = [query_identity(query) for query in stream]
    unique = len(set(identities))
    total = len(identities)
    return SamplingDiagnostics(total, unique, total - unique)


def warmup_overlap_diagnostics(warmup_stream, evaluation_stream, requested):
    """审计 warm-up 与 evaluation 是否共享同一条 query。

    该检查优先读取原始日志身份 ``source_qidx``，受控流退化到 ``qidx``；不读取
    gold support。主实验要求 ``evaluation_overlap == 0``，否则初始缓存可能间接
    看到被评估的未来 query。
    """

    def identity(query):
        return query_identity(query)

    warmup_ids = [identity(query) for query in warmup_stream]
    evaluation_ids = {identity(query) for query in evaluation_stream}
    return WarmupOverlapDiagnostics(
        requested=max(0, int(requested)),
        actual=len(warmup_ids),
        unique=len(set(warmup_ids)),
        evaluation_overlap=len(set(warmup_ids) & evaluation_ids),
    )


def chronological_sample(
    queries,
    warmup_size,
    evaluation_size,
    mode="prefix",
    block_size=None,
):
    """从自然日志中构造严格有序且 warm-up/evaluation 不重叠的流。

    ``prefix`` 只取日志开头；``window_span`` 在完整跨度上等距选择连续事件块，
    既保留长期时间覆盖，也不破坏每个评估窗口内部的自然 locality/reuse。

    返回的 ``qidx`` 始终重编号为 ``[0, n_selected)``，与随后计算的 query
    embedding 行号严格一致。原始日志身份保存在 ``source_qidx``，用于重复和
    warm-up 泄漏审计，避免把稀疏的原始索引误当成 dense 行号。
    """

    if mode not in {"prefix", "window_span"}:
        raise ValueError(
            "temporal sampling mode must be prefix or window_span"
        )
    queries = list(queries)
    warmup_size = max(0, int(warmup_size))
    evaluation_size = max(0, int(evaluation_size))
    required = warmup_size + evaluation_size
    if len(queries) < required:
        raise ValueError(
            f"chronological stream needs {required} events, found {len(queries)}"
        )

    indexed_queries = list(enumerate(queries))
    warmup = indexed_queries[:warmup_size]
    remaining = indexed_queries[warmup_size:]
    selected_blocks = None
    effective_block_size = None
    if mode == "prefix" or evaluation_size == 0:
        evaluation = remaining[:evaluation_size]
    else:
        effective_block_size = int(block_size or 0)
        if effective_block_size <= 0:
            raise ValueError("window_span requires a positive block_size")
        if evaluation_size % effective_block_size != 0:
            raise ValueError(
                "window_span evaluation_size must be divisible by block_size"
            )
        selected_blocks = evaluation_size // effective_block_size
        max_start = len(remaining) - effective_block_size
        starts = np.linspace(
            0,
            max_start,
            num=selected_blocks,
            dtype=np.int64,
        ) if selected_blocks else np.empty(0, dtype=np.int64)
        evaluation = []
        previous_end = 0
        for block_index, raw_start in enumerate(starts):
            start = max(int(raw_start), previous_end)
            remaining_blocks = selected_blocks - block_index - 1
            latest_start = (
                len(remaining)
                - effective_block_size * (remaining_blocks + 1)
            )
            start = min(start, latest_start)
            evaluation.extend(
                remaining[start:start + effective_block_size]
            )
            previous_end = start + effective_block_size

    selected = []
    for dense_index, (source_position, query) in enumerate(warmup + evaluation):
        if not isinstance(query, dict):
            raise TypeError("chronological queries must be dictionaries")
        selected_query = dict(query)
        source_identity = query.get("source_qidx")
        if source_identity is None:
            source_identity = query.get("qidx")
        selected_query["source_qidx"] = (
            source_position if source_identity is None else source_identity
        )
        selected_query["qidx"] = dense_index
        selected.append(selected_query)

    def timestamp(query):
        if not isinstance(query, dict):
            return None
        for field in ("event_ts", "question_ts", "timestamp"):
            value = query.get(field)
            if value is not None:
                return int(value)
        return None

    source_start = timestamp(queries[0]) if queries else None
    source_end = timestamp(queries[-1]) if queries else None
    evaluation_start = timestamp(evaluation[0][1]) if evaluation else None
    evaluation_end = timestamp(evaluation[-1][1]) if evaluation else None
    diagnostics = TemporalSamplingDiagnostics(
        mode=mode,
        source_events=len(queries),
        selected_events=len(warmup) + len(evaluation),
        source_start=source_start,
        source_end=source_end,
        evaluation_start=evaluation_start,
        evaluation_end=evaluation_end,
        block_size=effective_block_size,
        selected_blocks=selected_blocks,
    )
    return selected, diagnostics


def support_reuse_diagnostics(stream, window_size, support_field="sf_titles"):
    """量化历史窗口对未来 query 的 support 复用潜力。

    ``repeated_support_rate`` 是除首次出现外的 support occurrence 比例；
    ``past_answerable_query_rate`` 要求一个 query 的完整 support set 在其到达前
    都出现过；``adjacent_window_jaccard_mean`` 衡量相邻窗口 support 集合重合。
    读取 gold 的行为严格限制在实验结束后的 protocol audit。
    """

    supports_per_query = []
    frequencies = {}
    for query in stream:
        raw_supports = (
            query.get(support_field, ()) if isinstance(query, dict) else ())
        supports = frozenset(str(value) for value in raw_supports if value is not None)
        supports_per_query.append(supports)
        for support in supports:
            frequencies[support] = frequencies.get(support, 0) + 1

    seen = set()
    queries_with_support = 0
    answerable_from_past = 0
    repeated_occurrences = 0
    support_occurrences = 0
    for supports in supports_per_query:
        if not supports:
            continue
        queries_with_support += 1
        support_occurrences += len(supports)
        repeated_occurrences += len(supports & seen)
        if supports.issubset(seen):
            answerable_from_past += 1
        seen.update(supports)

    window_size = max(1, int(window_size))
    window_supports = [
        set().union(*supports_per_query[start:start + window_size])
        for start in range(0, len(supports_per_query), window_size)
    ]
    adjacent_jaccards = []
    for previous, current in zip(window_supports, window_supports[1:]):
        union = previous | current
        if union:
            adjacent_jaccards.append(len(previous & current) / len(union))

    return SupportReuseDiagnostics(
        queries_with_support=queries_with_support,
        support_occurrences=support_occurrences,
        unique_supports=len(frequencies),
        repeated_support_occurrences=repeated_occurrences,
        queries_answerable_from_past=answerable_from_past,
        adjacent_window_jaccard_mean=(
            float(np.mean(adjacent_jaccards)) if adjacent_jaccards else 0.0),
        max_support_frequency=max(frequencies.values(), default=0),
    )


def query_drift_diagnostics(stream, query_embs, window_size):
    """量化首尾分布差异与局部变化，不把构造标签交给策略。

    ``centroid_cosine_shift`` 衡量首尾 20% query 的表示中心变化；
    ``regime_js_divergence`` 衡量首尾 regime mixture 的差异，归一化到 [0,1]；
    相邻窗口 centroid shift 用于区分突变峰值与平滑迁移。
    """

    if not stream:
        return QueryDriftDiagnostics(0.0, 0.0, 0.0, 0.0)
    indices = np.asarray([
        int(query["qidx"]) for query in stream
        if isinstance(query, dict) and "qidx" in query
    ], dtype=np.int64)
    if len(indices) != len(stream):
        return QueryDriftDiagnostics(0.0, 0.0, 0.0, 0.0)
    embeddings = np.asarray(query_embs[indices], dtype=np.float64)
    embeddings /= np.clip(
        np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12, None)

    edge = max(1, len(stream) // 5)

    def normalized_centroid(block):
        center = np.asarray(block, dtype=np.float64).mean(axis=0)
        return center / max(float(np.linalg.norm(center)), 1e-12)

    first_center = normalized_centroid(embeddings[:edge])
    last_center = normalized_centroid(embeddings[-edge:])
    centroid_shift = max(0.0, 1.0 - float(first_center @ last_center))

    labels = [
        str(query.get("workload_regime", query.get("round", "unknown")))
        for query in stream
    ]
    values = sorted(set(labels))
    first_counts = np.asarray([
        labels[:edge].count(value) for value in values
    ], dtype=np.float64)
    last_counts = np.asarray([
        labels[-edge:].count(value) for value in values
    ], dtype=np.float64)
    first_prob = first_counts / max(float(first_counts.sum()), 1.0)
    last_prob = last_counts / max(float(last_counts.sum()), 1.0)
    mixture = 0.5 * (first_prob + last_prob)

    def kl_divergence(left, right):
        mask = left > 0.0
        return float(np.sum(
            left[mask] * np.log2(left[mask] / np.clip(right[mask], 1e-12, None))
        ))

    js_divergence = 0.5 * (
        kl_divergence(first_prob, mixture)
        + kl_divergence(last_prob, mixture)
    )

    window_size = max(1, int(window_size))
    window_centers = [
        normalized_centroid(embeddings[start:start + window_size])
        for start in range(0, len(embeddings), window_size)
        if len(embeddings[start:start + window_size])
    ]
    adjacent = [
        max(0.0, 1.0 - float(previous @ current))
        for previous, current in zip(window_centers, window_centers[1:])
    ]
    return QueryDriftDiagnostics(
        centroid_cosine_shift=centroid_shift,
        regime_js_divergence=min(1.0, max(0.0, js_divergence)),
        adjacent_centroid_shift_mean=(
            float(np.mean(adjacent)) if adjacent else 0.0),
        adjacent_centroid_shift_max=max(adjacent, default=0.0),
    )


def workload_factor_diagnostics(
    stream,
    window_size,
    support_field="sf_titles",
    regime_field="workload_state",
):
    """审计实际 stream 的 drift/reuse/predictability，而不是相信构造参数。

    该函数可以读取 gold support，但只在 stream 已构造完成后计算报告，不参与
    initialization、detector 或 cache policy。自然流没有 ``workload_state`` 时，
    使用 loader 提供的 ``round`` 或 ``category``。
    """

    window_size = max(1, int(window_size))
    windows = [
        list(stream[start:start + window_size])
        for start in range(0, len(stream), window_size)
        if stream[start:start + window_size]
    ]

    def query_regime(query):
        for field in (
            regime_field,
            "workload_regime",
            "round",
            "category",
        ):
            if isinstance(query, dict) and query.get(field) is not None:
                return str(query[field])
        return "unknown"

    window_regimes = []
    window_supports = []
    for window in windows:
        counts = {}
        supports = set()
        for query in window:
            regime = query_regime(query)
            counts[regime] = counts.get(regime, 0) + 1
            supports.update(
                str(value)
                for value in query.get(support_field, ())
                if value is not None
            )
        regime = min(counts, key=lambda value: (-counts[value], value))
        window_regimes.append(regime)
        window_supports.append(supports)

    changed_jaccards = []
    cross_regime = 0
    for previous_regime, regime, previous_support, support in zip(
        window_regimes,
        window_regimes[1:],
        window_supports,
        window_supports[1:],
    ):
        if previous_regime == regime:
            continue
        cross_regime += 1
        union = previous_support | support
        changed_jaccards.append(
            len(previous_support & support) / len(union) if union else 0.0
        )

    seen_by_regime = {}
    support_occurrences = 0
    repeated_occurrences = 0
    for query in stream:
        regime = query_regime(query)
        seen = seen_by_regime.setdefault(regime, set())
        supports = {
            str(value)
            for value in query.get(support_field, ())
            if value is not None
        }
        support_occurrences += len(supports)
        repeated_occurrences += len(supports & seen)
        seen.update(supports)

    # 压缩连续相同 regime，衡量 regime 之间的可复现迁移，而非窗口自环。
    states = []
    for regime in window_regimes:
        if not states or states[-1] != regime:
            states.append(regime)
    transition_counts = {}
    correct = recurrent = total = 0
    for source, target in zip(states, states[1:]):
        row = transition_counts.setdefault(source, {})
        total += 1
        if row:
            recurrent += 1
            predicted = min(
                row,
                key=lambda value: (-row[value], value),
            )
            correct += int(predicted == target)
        row[target] = row.get(target, 0) + 1

    visibility_values = []
    for query in stream:
        label = query.get("evidence_visibility") if isinstance(query, dict) else None
        if label is not None:
            visibility_values.append(float(str(label).lower() == "direct"))

    return WorkloadFactorDiagnostics(
        windows=len(windows),
        regimes=len(set(window_regimes)),
        cross_regime_transitions=cross_regime,
        cross_regime_evidence_jaccard_mean=(
            float(np.mean(changed_jaccards)) if changed_jaccards else 1.0
        ),
        within_regime_repeated_support_rate=(
            repeated_occurrences / max(1, support_occurrences)
        ),
        causal_transition_accuracy=correct / max(1, total),
        recurrent_transition_rate=recurrent / max(1, total),
        labeled_visibility_rate=(
            float(np.mean(visibility_values)) if visibility_values else 0.0
        ),
    )


def causal_prefix_init_kb(
    doc_pool,
    doc_embs,
    warmup_queries,
    query_embs,
    budget,
    seed=42,
    candidate_topk=32,
):
    """只用评估开始前的 query prefix 构造所有策略共用的初始热缓存。

    每个 warm-up query 从 cold corpus 检索 top-k，文档分数是跨 warm-up query 的
    ``max cosine``。若候选不足以填满容量，剩余位置用固定 seed 的 corpus 样本补齐。
    该过程不读取 gold support，也不读取被评估的未来窗口。
    """

    budget = max(0, min(int(budget), len(doc_pool)))
    if budget == 0:
        return set()

    warmup_indices = [
        int(query["qidx"])
        for query in warmup_queries
        if isinstance(query, dict) and "qidx" in query
    ]
    scores = {}
    if warmup_indices:
        # 容量变化实验中，固定 top-k 会让大 cache 主要由随机补位构成。让每个
        # warm-up query 的候选宽度随预算自适应，确保不同 ratio 都由同一种
        # query-conditioned initializer 主导，而不是容量越大随机比例越高。
        budget_width = int(np.ceil(budget / max(1, len(warmup_indices))))
        topk = max(int(candidate_topk), budget_width)
        topk = max(1, min(topk, len(doc_pool)))
        for start in range(0, len(warmup_indices), 64):
            block_indices = warmup_indices[start:start + 64]
            similarities = np.asarray(
                query_embs[block_indices] @ doc_embs.T)
            if topk >= similarities.shape[1]:
                candidate_columns = np.broadcast_to(
                    np.arange(similarities.shape[1]), similarities.shape)
            else:
                candidate_columns = np.argpartition(
                    similarities, -topk, axis=1)[:, -topk:]
            for row, candidates in zip(similarities, candidate_columns):
                for pool_idx in candidates:
                    pool_idx = int(pool_idx)
                    scores[pool_idx] = max(
                        scores.get(pool_idx, float("-inf")),
                        float(row[pool_idx]),
                    )

    selected = [
        pool_idx
        for pool_idx, _ in sorted(
            scores.items(), key=lambda item: (-item[1], item[0]))[:budget]
    ]
    if len(selected) < budget:
        selected_set = set(selected)
        remaining = [
            pool_idx for pool_idx in range(len(doc_pool))
            if pool_idx not in selected_set
        ]
        rng = np.random.default_rng(int(seed))
        need = min(budget - len(selected), len(remaining))
        if need:
            fill = rng.choice(remaining, size=need, replace=False)
            selected.extend(int(pool_idx) for pool_idx in fill)

    return {doc_pool[pool_idx]["doc_id"] for pool_idx in selected}
