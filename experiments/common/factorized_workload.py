"""表示独立、可审计的 evidence-working-set 漂移协议。

旧受控实验在 dense query embedding 上做 KMeans，再用相近表示检测漂移，容易把
benchmark construction 和 detector 成功绑定在一起。本模块只用于离线构造：

1. 在 gold evidence 文本的 sparse TF-IDF 空间中划分 latent evidence topics；
2. 将 topics 平衡合并成若干 working-set regimes；
3. 只改变 regime 的时间顺序或混合比例，产生 one-shot、gradual、
   recurring 和 shuffled control；
4. 原始 query 默认只使用一次，gold topic/regime 标签不提供给在线策略。

这不是为了让某个 cache policy 获胜，而是把四个实验因素拆开：漂移幅度、regime 内
evidence 复用、转移可预测性与 evidence 可见性。
自然日志（StreamingQA/MIND）不应经过本构造器重排。
"""

from collections import Counter, defaultdict
from dataclasses import dataclass
import hashlib
import json

import numpy as np

from experiments.common.stream_protocol import query_identity


ONE_SHOT = "factorized_one_shot"
GRADUAL = "factorized_gradual"
RECURRING = "factorized_recurring"
SHUFFLED = "factorized_shuffled"
STATIONARY = "factorized_stationary"
NATURAL = "natural_temporal"
AUTO = "auto"
FACTORIZED_WORKLOADS = {
    ONE_SHOT, GRADUAL, RECURRING, SHUFFLED, STATIONARY,
}
WORKLOAD_CHOICES = {AUTO, NATURAL, *FACTORIZED_WORKLOADS}
SOURCE_POOL_ALL = "all"
SOURCE_POOL_CALIBRATION = "calibration"
SOURCE_POOL_TEST = "test"
SOURCE_POOL_ROLES = {
    SOURCE_POOL_ALL,
    SOURCE_POOL_CALIBRATION,
    SOURCE_POOL_TEST,
}


@dataclass(frozen=True)
class FactorizedConstructionStats:
    representation: str
    latent_topics: int
    regimes: int
    requested_queries: int
    selected_queries: int
    warmup_queries: int
    available_queries: int
    eligible_queries: int
    exact_query_duplicates: int
    min_support_frequency: int
    family_mode: str
    support_families: int
    regime_counts: dict

    def as_dict(self):
        return {
            "representation": self.representation,
            "latent_topics": int(self.latent_topics),
            "regimes": int(self.regimes),
            "requested_queries": int(self.requested_queries),
            "selected_queries": int(self.selected_queries),
            "warmup_queries": int(self.warmup_queries),
            "available_queries": int(self.available_queries),
            "eligible_queries": int(self.eligible_queries),
            "exact_query_duplicates": int(self.exact_query_duplicates),
            "min_support_frequency": int(self.min_support_frequency),
            "family_mode": str(self.family_mode),
            "support_families": int(self.support_families),
            "regime_counts": {
                str(key): int(value)
                for key, value in self.regime_counts.items()
            },
        }


@dataclass(frozen=True)
class DisjointSourcePoolStats:
    """稳定 calibration/test source-family 划分的审计记录。

    样本池按照 gold support 的连通分量分配，而不是只按 exact question 划分。因此，
    共享同一 support 文档的两个问题，或通过共享文档连接起来的两个 multi-support
    family，绝不会被分到 calibration 与 test 两侧。
    """

    role: str
    seed: int
    calibration_fraction: float
    split_unit: str
    input_queries: int
    total_queries: int
    duplicate_source_queries_dropped: int
    calibration_queries: int
    test_queries: int
    calibration_supports: int
    test_supports: int
    calibration_families: int
    test_families: int
    query_overlap: int
    support_overlap: int
    family_overlap: int
    calibration_query_fingerprint: str
    test_query_fingerprint: str
    calibration_support_fingerprint: str
    test_support_fingerprint: str
    calibration_family_fingerprint: str
    test_family_fingerprint: str

    def as_dict(self):
        return {
            "role": str(self.role),
            "seed": int(self.seed),
            "calibration_fraction": float(self.calibration_fraction),
            "split_unit": str(self.split_unit),
            "stable_id_scheme": "sha256-content-v1",
            "input_queries": int(self.input_queries),
            "total_queries": int(self.total_queries),
            "duplicate_source_queries_dropped": int(
                self.duplicate_source_queries_dropped
            ),
            "calibration_queries": int(self.calibration_queries),
            "test_queries": int(self.test_queries),
            "calibration_supports": int(self.calibration_supports),
            "test_supports": int(self.test_supports),
            "calibration_families": int(self.calibration_families),
            "test_families": int(self.test_families),
            "query_overlap": int(self.query_overlap),
            "support_overlap": int(self.support_overlap),
            "family_overlap": int(self.family_overlap),
            "overlap_assertion": (
                self.query_overlap == self.support_overlap
                == self.family_overlap == 0
            ),
            "calibration_query_fingerprint": str(
                self.calibration_query_fingerprint
            ),
            "test_query_fingerprint": str(self.test_query_fingerprint),
            "calibration_support_fingerprint": str(
                self.calibration_support_fingerprint
            ),
            "test_support_fingerprint": str(
                self.test_support_fingerprint
            ),
            "calibration_family_fingerprint": str(
                self.calibration_family_fingerprint
            ),
            "test_family_fingerprint": str(
                self.test_family_fingerprint
            ),
        }


def _stable_digest(*values):
    payload = json.dumps(
        values, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _set_fingerprint(values):
    return _stable_digest(*sorted(str(value) for value in values))


def split_disjoint_source_pool(
    queries,
    doc_pool,
    title_to_idx,
    role,
    seed=1729,
    calibration_fraction=0.5,
    support_field="sf_titles",
    family_mode="auto",
):
    """选择稳定且 family 互不重叠的 calibration 或 test 源样本池。

    划分使用由内容决定的 support/query ID，不依赖 workload 随机种子或 loader 内部的
    ``qidx``。分配单位是 support 连通分量，因此 calibration 与 test 在构造上不存在
    exact query、support document 或 evidence family 重叠。构造标签只作为离线元数据，
    不会通过策略反馈接口返回给在线方法。
    """

    role = str(role)
    if role not in SOURCE_POOL_ROLES:
        raise ValueError(f"unknown source pool role: {role}")
    if role == SOURCE_POOL_ALL:
        return list(queries), None
    calibration_fraction = float(calibration_fraction)
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be in (0, 1)")
    if family_mode not in {"auto", "exact", "anchor"}:
        raise ValueError("family_mode must be auto, exact, or anchor")

    def stable_support_id(title):
        position = title_to_idx.get(title)
        if position is None:
            return _stable_digest("missing-support", str(title))
        document = doc_pool[int(position)]
        # SQuAD 展示的段落标题含有 loader 内部后缀。段落正文才是稳定的
        # source-family 标识；只有文档缺少正文时才退回使用标题。
        text = str(document.get("text", ""))
        source_value = text if text else str(document.get("title", title))
        return _stable_digest("support", source_value)

    records = []
    seen_query_ids = set()
    support_frequency = Counter()
    for query in queries:
        support_ids = tuple(sorted({
            stable_support_id(title)
            for title in query.get(support_field, ())
            if title in title_to_idx
        }))
        query_id = _stable_digest(
            "query",
            str(query.get("question", "")),
            str(query.get("answer", "")),
            support_ids,
        )
        # 在本协议中，内容完全相同的源记录属于同一个 exact request。这里只保留一个
        # 规范副本，不为了制造唯一 ID 而拼接依赖 loader 顺序的不稳定后缀。
        if query_id in seen_query_ids:
            continue
        seen_query_ids.add(query_id)
        support_frequency.update(support_ids)
        records.append((query, query_id, support_ids))

    parent = {}

    def find(value):
        parent.setdefault(value, value)
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left, right):
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            if left_root > right_root:
                left_root, right_root = right_root, left_root
            parent[right_root] = left_root

    for _, _, support_ids in records:
        if support_ids:
            first = support_ids[0]
            find(first)
            for support_id in support_ids[1:]:
                union(first, support_id)

    components = defaultdict(set)
    for support_id in parent:
        components[find(support_id)].add(support_id)
    component_id = {
        support_id: _stable_digest(
            "support-component", *sorted(components[find(support_id)])
        )
        for support_id in parent
    }

    pools = {
        SOURCE_POOL_CALIBRATION: [],
        SOURCE_POOL_TEST: [],
    }
    query_ids = {key: set() for key in pools}
    support_ids_by_pool = {key: set() for key in pools}
    family_ids = {key: set() for key in pools}
    denominator = float(1 << 64)
    for query, query_id, support_ids in records:
        mode = str(family_mode)
        if mode == "auto":
            mode = "exact" if len(support_ids) <= 1 else "anchor"
        if mode == "anchor" and support_ids:
            anchor = min(
                support_ids,
                key=lambda value: (-support_frequency[value], value),
            )
            family_key = ("anchor", anchor)
        else:
            family_key = ("exact", *support_ids)
        family_id = _stable_digest("family", *family_key)
        if support_ids:
            assignment_unit = component_id[support_ids[0]]
        else:
            assignment_unit = _stable_digest("unsupported-query", query_id)
        bucket = int(
            _stable_digest("source-pool", int(seed), assignment_unit)[:16], 16
        ) / denominator
        assigned_role = (
            SOURCE_POOL_CALIBRATION
            if bucket < calibration_fraction
            else SOURCE_POOL_TEST
        )
        copied = dict(query)
        # 在 factorized 构造器把 qidx 改写为连续 embedding 行号之前，先保存由内容
        # 决定的源请求标识。
        copied["source_qidx"] = query_id
        copied["source_family_id"] = family_id
        copied["source_pool_role"] = assigned_role
        pools[assigned_role].append(copied)
        query_ids[assigned_role].add(query_id)
        support_ids_by_pool[assigned_role].update(support_ids)
        family_ids[assigned_role].add(family_id)

    calibration = SOURCE_POOL_CALIBRATION
    test = SOURCE_POOL_TEST
    overlaps = (
        query_ids[calibration] & query_ids[test],
        support_ids_by_pool[calibration] & support_ids_by_pool[test],
        family_ids[calibration] & family_ids[test],
    )
    if any(overlaps):
        raise AssertionError(
            "source-pool leakage: "
            f"query={len(overlaps[0])}, support={len(overlaps[1])}, "
            f"family={len(overlaps[2])}"
        )
    if not pools[calibration] or not pools[test]:
        raise ValueError(
            "stable source split produced an empty pool; increase the source "
            "corpus or choose a different source_pool_seed"
        )

    stats = DisjointSourcePoolStats(
        role=role,
        seed=int(seed),
        calibration_fraction=calibration_fraction,
        split_unit="connected stable support-family component",
        input_queries=len(queries),
        total_queries=len(records),
        duplicate_source_queries_dropped=len(queries) - len(records),
        calibration_queries=len(pools[calibration]),
        test_queries=len(pools[test]),
        calibration_supports=len(support_ids_by_pool[calibration]),
        test_supports=len(support_ids_by_pool[test]),
        calibration_families=len(family_ids[calibration]),
        test_families=len(family_ids[test]),
        query_overlap=len(overlaps[0]),
        support_overlap=len(overlaps[1]),
        family_overlap=len(overlaps[2]),
        calibration_query_fingerprint=_set_fingerprint(query_ids[calibration]),
        test_query_fingerprint=_set_fingerprint(query_ids[test]),
        calibration_support_fingerprint=_set_fingerprint(
            support_ids_by_pool[calibration]
        ),
        test_support_fingerprint=_set_fingerprint(
            support_ids_by_pool[test]
        ),
        calibration_family_fingerprint=_set_fingerprint(
            family_ids[calibration]
        ),
        test_family_fingerprint=_set_fingerprint(family_ids[test]),
    )
    return pools[role], stats


def resolve_workload(queries, requested):
    """根据数据集协议选择唯一的 stream 构造方式。

    标记 ``preserve_order`` 的真实日志只能走官方时间顺序；普通 QA 数据只能走
    factorized evidence-regime 构造。``auto`` 是推荐入口，避免同时维护 workload
    和 drift 两套互相冲突的参数。
    """

    requested = str(requested)
    if requested not in WORKLOAD_CHOICES:
        raise ValueError(f"unknown workload: {requested}")
    preserve_order = bool(queries and queries[0].get("preserve_order", False))
    if requested == AUTO:
        return NATURAL if preserve_order else RECURRING
    if preserve_order and requested != NATURAL:
        raise ValueError(
            "natural traces require workload=natural_temporal (or auto)"
        )
    if not preserve_order and requested == NATURAL:
        raise ValueError(
            "controlled QA data require a factorized_* workload (or auto)"
        )
    return requested


def _support_key(query, support_field):
    return tuple(sorted(
        str(value) for value in query.get(support_field, ()) if value is not None
    ))


def _support_frequencies(queries, support_field):
    return Counter(
        support
        for query in queries
        for support in _support_key(query, support_field)
    )


def _family_key(query, support_field, support_frequencies, family_mode):
    """定义“证据家族”，但不复用 exact query。

    single-support 数据集以完整 support 为 family。对 direct multi-support
    query，``anchor`` 以全局出现频率最高的 support 为共享 working-set
    anchor；这保证后续 visit 复用 evidence，而不是复制问题文本。
    """

    supports = _support_key(query, support_field)
    mode = str(family_mode)
    if mode == "auto":
        mode = "exact" if len(supports) <= 1 else "anchor"
    if mode == "anchor" and supports:
        anchor = min(
            supports,
            key=lambda value: (-support_frequencies[value], value),
        )
        return ("anchor", anchor)
    return ("exact",) + supports


def _evidence_topic_labels(
    queries,
    doc_pool,
    title_to_idx,
    seed,
    support_field,
    latent_topics,
):
    """用 sparse evidence representation 标注 query，不读取 query embedding。"""

    from sklearn.cluster import MiniBatchKMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    support_titles = sorted({
        title
        for query in queries
        for title in query.get(support_field, ())
        if title in title_to_idx
    })
    if len(support_titles) < 2:
        raise ValueError("factorized workload requires at least two support docs")
    latent_topics = max(2, min(int(latent_topics), len(support_titles)))
    texts = []
    for title in support_titles:
        document = doc_pool[int(title_to_idx[title])]
        texts.append(
            f"{document.get('title', '')} {str(document.get('text', ''))[:1000]}"
        )
    matrix = TfidfVectorizer(
        stop_words="english",
        ngram_range=(1, 2),
        min_df=2 if len(texts) >= 20 else 1,
        max_features=8192,
        sublinear_tf=True,
    ).fit_transform(texts)
    labels = MiniBatchKMeans(
        n_clusters=latent_topics,
        random_state=int(seed),
        n_init=5,
        batch_size=min(512, max(32, len(texts))),
    ).fit_predict(matrix)
    title_topic = {
        title: int(label) for title, label in zip(support_titles, labels)
    }

    query_topics = []
    for query in queries:
        votes = Counter(
            title_topic[title]
            for title in query.get(support_field, ())
            if title in title_topic
        )
        if votes:
            topic = min(
                votes,
                key=lambda value: (-votes[value], value),
            )
        else:
            topic = int(query.get("qidx", len(query_topics))) % latent_topics
        query_topics.append(int(topic))
    return query_topics, latent_topics


def _balanced_regime_map(query_topics, n_regimes):
    """按 query mass 贪心平衡 latent topics，避免某个 regime 先耗尽。"""

    counts = Counter(int(value) for value in query_topics)
    n_regimes = max(2, min(int(n_regimes), len(counts)))
    loads = [0] * n_regimes
    topic_to_regime = {}
    for topic, count in sorted(counts.items(), key=lambda item: (-item[1], item[0])):
        regime = min(range(n_regimes), key=lambda value: (loads[value], value))
        topic_to_regime[int(topic)] = int(regime)
        loads[regime] += int(count)
    return topic_to_regime, n_regimes


class _FiniteFamilySampler:
    """跨 regime visit 复用 support family，但绝不复用 exact query。

    一个 evidence family（例如 SQuAD 的同一 paragraph）通常对应多个不同问题。
    若在第一次 visit 中连续耗尽整个 family，后续 regime recurrence 只会重复粗粒度
    topic，而不会重复真正的 evidence working set。这里维护一个有限 active-family
    cohort：每次 visit 优先从每个 active family 取一个尚未使用的 query，family
    耗尽后才补入新 family。于是复现的是 evidence，而不是 query 文本。
    """

    def __init__(
        self,
        queries,
        rng,
        support_field,
        support_frequencies,
        family_mode,
    ):
        families = defaultdict(list)
        for query in queries:
            key = _family_key(
                query,
                support_field,
                support_frequencies,
                family_mode,
            )
            families[key].append(query)
        family_keys = list(families)
        rng.shuffle(family_keys)
        # 在随机 tie-break 后优先选择可跨更多 visit 复用的 family；这直接控制
        # benchmark 的 reuse 轴，不向在线策略暴露 family 标签。
        family_keys.sort(key=lambda key: -len(families[key]))
        for key in family_keys:
            rng.shuffle(families[key])
        self._families = {
            key: list(families[key]) for key in family_keys
        }
        self._inactive = list(family_keys)
        self._active = []
        self._remaining = sum(len(items) for items in self._families.values())

    @property
    def family_count(self):
        return int(len(self._families))

    @property
    def remaining(self):
        return int(self._remaining)

    def _activate_until(self, target):
        while len(self._active) < int(target) and self._inactive:
            self._active.append(self._inactive.pop(0))

    def draw(self, count):
        count = int(count)
        if count > self.remaining:
            raise ValueError(
                f"factorized regime exhausted: need {count}, have {self.remaining}; "
                "reduce n_windows/window_size instead of cycling exact queries"
            )
        self._activate_until(min(count, len(self._families)))
        out = []
        while len(out) < count:
            if not self._active:
                raise ValueError("factorized regime has no non-empty evidence family")
            next_active = []
            for key in self._active:
                items = self._families[key]
                if len(out) < count and items:
                    out.append(items.pop())
                    self._remaining -= 1
                if items:
                    next_active.append(key)
            self._active = next_active
            self._activate_until(min(count, len(self._families)))
        return out


def _regime_schedule(workload, n_windows, n_regimes, rng):
    def uniform(regimes):
        regimes = tuple(int(value) for value in regimes)
        return {value: 1.0 / len(regimes) for value in regimes}

    old = tuple(range(max(1, n_regimes // 2)))
    new = tuple(range(max(1, n_regimes // 2), n_regimes)) or old
    if workload == ONE_SHOT:
        midpoint = n_windows // 2
        return [
            uniform(old if index < midpoint else new)
            for index in range(n_windows)
        ]
    if workload == GRADUAL:
        # 两端各保留 25% 稳定窗口，中间 50% 线性迁移。比例在生成
        # unique-query stream 前一次性确定，不会因为边跑边抽样而突然耗尽。
        stable = max(1, n_windows // 4)
        transition_start = stable
        transition_end = max(transition_start, n_windows - stable - 1)
        schedule = []
        for index in range(n_windows):
            if index < transition_start:
                alpha = 1.0
            elif index > transition_end:
                alpha = 0.0
            elif transition_end == transition_start:
                alpha = 0.5
            else:
                alpha = 1.0 - (
                    (index - transition_start)
                    / (transition_end - transition_start)
                )
            weights = {}
            if alpha > 0:
                weights.update({value: alpha / len(old) for value in old})
            if alpha < 1:
                weights.update({
                    value: (1.0 - alpha) / len(new) for value in new
                })
            schedule.append(weights)
        return schedule
    if workload == STATIONARY:
        all_regimes = uniform(range(n_regimes))
        return [dict(all_regimes) for _ in range(n_windows)]

    states = [index % n_regimes for index in range(n_windows)]
    if workload == SHUFFLED:
        # 保持 regime 边际窗口数与 recurring 完全一致，但选择因果历史
        # 多数转移预测最难的平衡序列。该搜索只查看 regime ID 的排列，
        # 不读取 query embedding、support 文本或任何 policy 输出。
        best = None
        best_score = None
        trials = max(256, 64 * int(n_windows))
        for _ in range(trials):
            candidate = list(states)
            rng.shuffle(candidate)
            if any(
                left == right
                for left, right in zip(candidate, candidate[1:])
            ):
                continue
            accuracy = _causal_transition_accuracy(candidate)
            transition_coverage = len(set(zip(candidate, candidate[1:])))
            score = (accuracy, -transition_coverage, tuple(candidate))
            if best_score is None or score < best_score:
                best = candidate
                best_score = score
        if best is not None:
            states = best
        else:
            rng.shuffle(states)
    return [{int(state): 1.0} for state in states]


def _causal_transition_accuracy(states):
    """只用当前转移之前的 counts 计算 schedule 可预测性。"""

    counts = {}
    correct = 0
    total = 0
    for source, target in zip(states, states[1:]):
        row = counts.setdefault(int(source), {})
        total += 1
        if row:
            predicted = min(
                row,
                key=lambda value: (-row[value], value),
            )
            correct += int(predicted == int(target))
        row[int(target)] = row.get(int(target), 0) + 1
    return correct / max(1, total)


def _allocate_window(active_weights, samplers, window_size):
    """按目标 evidence-regime mixture 分配窗口配额。

    largest-remainder rounding 保证每窗口数量精确为 ``window_size``；
    finite sampler 保证只使用尚未出现的 query。
    """

    active = [
        int(value) for value, weight in active_weights.items()
        if float(weight) > 0
    ]
    remaining = {
        value: int(samplers[value].remaining) for value in active
    }
    if sum(remaining.values()) < int(window_size):
        raise ValueError(
            "factorized active regimes do not contain enough unique queries"
        )
    weights = np.asarray(
        [float(active_weights[value]) for value in active], dtype=np.float64
    )
    weights /= weights.sum()
    raw = weights * int(window_size)
    counts = {
        value: min(int(np.floor(raw[index])), remaining[value])
        for index, value in enumerate(active)
    }
    leftover = int(window_size) - sum(counts.values())
    fractions = {
        value: float(raw[index] - np.floor(raw[index]))
        for index, value in enumerate(active)
    }
    while leftover:
        candidates = [
            value for value in active if counts[value] < remaining[value]
        ]
        if not candidates:
            raise ValueError(
                "factorized mixture exhausted before filling the window"
            )
        selected = min(
            candidates,
            key=lambda value: (-fractions[value], counts[value], value),
        )
        counts[selected] += 1
        fractions[selected] = 0.0
        leftover -= 1
    return {value: count for value, count in counts.items() if count}


def _state_label(active_weights):
    """把离线构造比例写入实验记录，不传给策略决策接口。"""

    return "+".join(
        f"{int(value)}:{float(weight):.4f}"
        for value, weight in sorted(active_weights.items())
        if float(weight) > 0
    )


def build_factorized_workload(
    queries,
    doc_pool,
    title_to_idx,
    n_windows,
    window_size,
    workload,
    seed=42,
    support_field="sf_titles",
    latent_topics=8,
    n_regimes=4,
    min_support_frequency=1,
    family_mode="auto",
    warmup_size=0,
):
    """构造互不重复的 warm-up、evaluation stream 与构造审计。"""

    if workload not in FACTORIZED_WORKLOADS:
        raise ValueError(f"unknown factorized workload: {workload}")
    normalized_queries = []
    for dense_index, query in enumerate(queries):
        copied = dict(query)
        original_identity = query_identity(query)
        if query.get("source_qidx") is not None or query.get("qidx") is not None:
            copied["source_qidx"] = original_identity
        copied["qidx"] = int(dense_index)
        normalized_queries.append(copied)
    queries = normalized_queries
    available_queries = len(queries)
    min_support_frequency = max(1, int(min_support_frequency))
    if family_mode not in {"auto", "exact", "anchor"}:
        raise ValueError("family_mode must be auto, exact, or anchor")
    support_frequencies = _support_frequencies(queries, support_field)
    queries = [
        query
        for query in queries
        if _support_key(query, support_field)
        and all(
            support_frequencies[support] >= min_support_frequency
            for support in _support_key(query, support_field)
        )
    ]
    requested = int(n_windows) * int(window_size)
    warmup_size = max(0, int(warmup_size))
    if len(queries) < requested + warmup_size:
        raise ValueError(
            "factorized workload needs "
            f"{requested + warmup_size} unique eligible queries "
            f"({warmup_size} warm-up + {requested} evaluation), "
            f"found {len(queries)} after min_support_frequency="
            f"{min_support_frequency}; load a larger source pool or lower the "
            "declared reuse floor"
        )
    rng = np.random.default_rng(int(seed) + 2027)
    query_topics, fitted_topics = _evidence_topic_labels(
        queries,
        doc_pool,
        title_to_idx,
        seed=seed,
        support_field=support_field,
        latent_topics=latent_topics,
    )
    topic_to_regime, fitted_regimes = _balanced_regime_map(
        query_topics, n_regimes
    )

    by_regime = defaultdict(list)
    for query, topic in zip(queries, query_topics):
        regime = int(topic_to_regime[topic])
        # warm-up protocol 需要从 evaluation 之外的同一初始 cohort 取历史。
        # 标签只留在实验记录中，不会被传给 cache policy 的决策接口。
        query["workload_topic"] = int(topic)
        query["workload_regime"] = regime
        query["constructor_space"] = "sparse-evidence-tfidf"
        copied = dict(query)
        by_regime[regime].append(copied)
    samplers = {
        regime: _FiniteFamilySampler(
            items,
            rng,
            support_field,
            support_frequencies,
            family_mode,
        )
        for regime, items in by_regime.items()
    }
    schedule = _regime_schedule(
        workload, int(n_windows), fitted_regimes, rng
    )

    warmup = []
    if warmup_size:
        initial_state = schedule[0]
        allocation = _allocate_window(initial_state, samplers, warmup_size)
        for regime, count in allocation.items():
            for query in samplers[regime].draw(count):
                copied = dict(query)
                copied["workload_window"] = -1
                copied["workload_state"] = _state_label(initial_state)
                warmup.append(copied)
        rng.shuffle(warmup)

    stream = []
    for window_index, active_weights in enumerate(schedule):
        allocation = _allocate_window(active_weights, samplers, window_size)
        window = []
        for regime, count in allocation.items():
            for query in samplers[regime].draw(count):
                copied = dict(query)
                copied["workload_window"] = int(window_index)
                copied["workload_state"] = _state_label(active_weights)
                window.append(copied)
        rng.shuffle(window)
        stream.extend(window)

    identities = [query_identity(query) for query in warmup + stream]
    stats = FactorizedConstructionStats(
        representation="sparse-evidence-tfidf",
        latent_topics=fitted_topics,
        regimes=fitted_regimes,
        requested_queries=requested,
        selected_queries=len(stream),
        warmup_queries=len(warmup),
        available_queries=available_queries,
        eligible_queries=len(queries),
        exact_query_duplicates=len(identities) - len(set(identities)),
        min_support_frequency=min_support_frequency,
        family_mode=str(family_mode),
        support_families=sum(
            sampler.family_count for sampler in samplers.values()
        ),
        regime_counts=dict(Counter(
            int(query["workload_regime"]) for query in stream
        )),
    )
    if stats.exact_query_duplicates:
        raise AssertionError(
            "factorized workload unexpectedly duplicated queries"
        )
    return stream, warmup, stats
