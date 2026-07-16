"""统一数据读入、查询流构造与因果热缓存初始化。

所有数据集首先由统一 loader 接口读成 ``doc_pool / queries / title_to_idx``。
之后才按 workload 类型分流：

* 自然时间流：只做时间范围采样，保持原始事件顺序；
* 受控 factorized 流：在同一套 evidence topic/family 上使用不同时间 schedule。

本模块不运行 cache policy，也不计算策略指标。gold support 只用于离线受控构造与
后验协议审计，构造标签不会传入在线策略。
"""

from dataclasses import dataclass

if __package__:
    from .config import log
    from . import config as experiment_config
    from .loaders import LOADERS
    from .loaders_temporal import TEMPORAL_LOADERS
    from .utils import compute_embeddings
else:
    from config import log
    import config as experiment_config
    from loaders import LOADERS
    from loaders_temporal import TEMPORAL_LOADERS
    from utils import compute_embeddings

from experiments.common.factorized_workload import (
    NATURAL,
    build_factorized_workload,
    resolve_workload,
)
from experiments.common.stream_protocol import (
    causal_prefix_init_kb,
    chronological_sample,
    query_drift_diagnostics,
    stream_sampling_diagnostics,
    support_reuse_diagnostics,
    warmup_overlap_diagnostics,
    workload_factor_diagnostics,
)
from experiments.visibility import require_visibility


@dataclass(frozen=True)
class PreparedWorkload:
    """runner 所需的统一数据对象。"""

    dataset_name: str
    doc_pool: list
    queries: list
    title_to_idx: dict
    doc_embs: object
    query_embs: object
    stream: list
    warmup_stream: list
    workload: str
    diagnostics: dict


def load_query_dataset(
    dataset_name,
    dataset_config,
    *,
    loader_name=None,
    n_source=None,
):
    """通过统一接口读取 temporal 或普通 QA 数据。

    每个 loader 都必须返回 ``(doc_pool, queries, title_to_idx)``。这里不构造漂移，
    因而更换自然流或受控流不会改变数据读入路径。
    """

    loader_key = loader_name or dataset_name
    if loader_key in TEMPORAL_LOADERS:
        loaded = TEMPORAL_LOADERS[loader_key]()
    else:
        source_size = (
            n_source if n_source is not None
            else dataset_config.get("n_source")
        )
        loaded = LOADERS[loader_key](n_source=source_size)
    require_visibility("direct", loader_key, loaded[1])
    return loaded


def prepare_workload(
    dataset_name,
    dataset_config,
    *,
    loader_name=None,
    n_source=None,
    warmup_windows=3,
    temporal_sampling="prefix",
    workload="auto",
    factorized_min_support_frequency=1,
    factorized_family_mode="auto",
):
    """统一读入数据，然后构造自然时间流或 factorized 受控流。"""

    num_windows = int(dataset_config["n_windows"])
    window_size = int(dataset_config["window_size"])
    warmup_size = int(warmup_windows) * window_size
    evaluation_size = num_windows * window_size

    doc_pool, queries, title_to_idx = load_query_dataset(
        dataset_name,
        dataset_config,
        loader_name=loader_name,
        n_source=n_source,
    )
    effective_workload = resolve_workload(queries, workload)

    temporal_stats = None
    if effective_workload == NATURAL:
        queries, sampling = chronological_sample(
            queries,
            warmup_size=warmup_size,
            evaluation_size=evaluation_size,
            mode=temporal_sampling,
            block_size=window_size,
        )
        temporal_stats = sampling.as_dict()
        log.info("Natural temporal sampling: %s", temporal_stats)

    embedding_tag = f"{dataset_name}_{num_windows}w_{window_size}s"
    doc_embs, query_embs = compute_embeddings(
        doc_pool,
        queries,
        tag=embedding_tag,
    )

    factorized_stats = None
    if effective_workload == NATURAL:
        warmup_stream = list(queries[:warmup_size])
        stream = list(queries[warmup_size:])
    else:
        stream, warmup_stream, construction = build_factorized_workload(
            queries,
            doc_pool,
            title_to_idx,
            n_windows=num_windows,
            window_size=window_size,
            workload=effective_workload,
            seed=int(experiment_config.DATA_SEED),
            min_support_frequency=factorized_min_support_frequency,
            family_mode=factorized_family_mode,
            warmup_size=warmup_size,
        )
        factorized_stats = construction.as_dict()

    diagnostics = _audit_workload(
        stream,
        warmup_stream,
        query_embs,
        window_size,
        warmup_size,
    )
    diagnostics.update({
        "temporal_sampling": temporal_stats,
        "factorized_construction": factorized_stats,
    })
    return PreparedWorkload(
        dataset_name=str(dataset_name),
        doc_pool=doc_pool,
        queries=queries,
        title_to_idx=title_to_idx,
        doc_embs=doc_embs,
        query_embs=query_embs,
        stream=stream,
        warmup_stream=warmup_stream,
        workload=str(effective_workload),
        diagnostics=diagnostics,
    )


def _audit_workload(
    stream,
    warmup_stream,
    query_embs,
    window_size,
    requested_warmup_queries,
):
    """集中执行不反馈给策略的协议审计。"""

    warmup = warmup_overlap_diagnostics(
        warmup_stream,
        stream,
        requested_warmup_queries,
    ).as_dict()
    if warmup["evaluation_overlap"]:
        raise ValueError(
            "Causal protocol violation: warm-up shares exact queries with "
            f"evaluation stream: {warmup}"
        )
    log.info("Warm-up/evaluation audit: %s", warmup)
    return {
        "stream_sampling": stream_sampling_diagnostics(stream).as_dict(),
        "support_reuse": support_reuse_diagnostics(
            stream, window_size
        ).as_dict(),
        "query_drift": query_drift_diagnostics(
            stream, query_embs, window_size
        ).as_dict(),
        "workload_factors": workload_factor_diagnostics(
            stream, window_size
        ).as_dict(),
        "warmup_audit": warmup,
    }


def build_initial_hot_cache(
    prepared,
    dataset_config,
    *,
    kb_budget_override=None,
    kb_pool_ratio=None,
):
    """仅用评估前历史，为所有策略构造相同的初始热缓存。"""

    pool_size = len(prepared.doc_pool)
    default_ratio = float(dataset_config.get("kb_pool_ratio", 0.1))
    budget = max(1, min(pool_size, int(round(pool_size * default_ratio))))
    if kb_budget_override is not None:
        override = max(1, min(pool_size, int(kb_budget_override)))
        log.info(
            "[%s] KB budget override: %s -> %s",
            prepared.dataset_name,
            budget,
            override,
        )
        budget = override
    elif kb_pool_ratio is not None:
        override = max(
            1,
            min(pool_size, int(round(pool_size * float(kb_pool_ratio)))),
        )
        log.info(
            "[%s] KB/pool ratio override: %s -> %s "
            "(pool=%s, ratio=%.4f)",
            prepared.dataset_name,
            budget,
            override,
            pool_size,
            float(kb_pool_ratio),
        )
        budget = override

    initial_kb = causal_prefix_init_kb(
        prepared.doc_pool,
        prepared.doc_embs,
        prepared.warmup_stream,
        prepared.query_embs,
        budget,
        seed=int(experiment_config.DATA_SEED) + 313,
    )
    log.info(
        "[%s] causal-prefix init: warmup_queries=%s, KB=%s",
        prepared.dataset_name,
        len(prepared.warmup_stream),
        len(initial_kb),
    )
    return budget, initial_kb
