"""单数据集实验的高层编排。

这里只连接 workload、evaluation 和 reporting 三层，不包含具体流构造、指标公式或
cache 更新细节。外部脚本需要复用实验时，只调用 ``run_dataset`` 即可。
"""

import time

if __package__:
    from . import config as experiment_config
    from .evaluation import evaluate_stream
    from .reporting import summarise_run
    from .workload import build_initial_hot_cache, prepare_workload
else:
    import config as experiment_config
    from evaluation import evaluate_stream
    from reporting import summarise_run
    from workload import build_initial_hot_cache, prepare_workload

from algorithms.cache.params import PARAMS as _P


def configure_shared_cache_parameters():
    """把 direct 实验的公共系统参数注入所有 baseline。"""

    _P.update(
        SEED=experiment_config.SEED,
        SF_HIT_THRESH=experiment_config.SF_HIT_THRESH,
        DOC_ARRIVE=experiment_config.DOC_ARRIVE,
        DOC_ADD_CAP=experiment_config.DOC_ADD_CAP,
        EDIT_BATCH=experiment_config.EDIT_BATCH,
        FETCH_TOP_K=experiment_config.FETCH_TOP_K,
        AMAT_HIT_COST=experiment_config.AMAT_HIT_COST,
        AMAT_MISS_PENALTY=experiment_config.AMAT_MISS_PENALTY,
    )


def run_dataset(
    dataset_name,
    dataset_config,
    strategies_to_run,
    loader_name=None,
    n_source=None,
    kb_budget_override=None,
    kb_pool_ratio=None,
    warmup_windows=3,
    temporal_sampling="prefix",
    workload="auto",
    factorized_min_support_frequency=1,
    factorized_family_mode="auto",
):
    """运行一个数据集，并返回向后兼容的结果字典。"""

    configure_shared_cache_parameters()
    start_time = time.time()
    prepared = prepare_workload(
        dataset_name,
        dataset_config,
        loader_name=loader_name,
        n_source=n_source,
        warmup_windows=warmup_windows,
        temporal_sampling=temporal_sampling,
        workload=workload,
        factorized_min_support_frequency=(
            factorized_min_support_frequency
        ),
        factorized_family_mode=factorized_family_mode,
    )
    kb_budget, initial_kb = build_initial_hot_cache(
        prepared,
        dataset_config,
        kb_budget_override=kb_budget_override,
        kb_pool_ratio=kb_pool_ratio,
    )
    strategies, tracking = evaluate_stream(
        prepared,
        dataset_config,
        strategies_to_run,
        initial_kb,
    )
    return summarise_run(
        prepared,
        kb_budget,
        int(dataset_config["window_size"]),
        strategies_to_run,
        strategies,
        tracking,
        time.time() - start_time,
        warmup_windows,
    )
