"""正式实验 loader 的 evidence visibility 边界检查。"""


def require_visibility(expected, loader_key, queries):
    """确保 loader 的全部 query 都属于指定 evidence 场景。"""

    invalid = [
        query for query in queries
        if query.get("evidence_visibility") != expected
    ]
    if not invalid:
        return

    counts = {}
    for query in invalid:
        label = query.get("evidence_visibility", "missing")
        counts[label] = counts.get(label, 0) + 1
    destination = "hidden" if expected == "direct" else "direct"
    raise ValueError(
        f"{expected.title()} loader {loader_key!r} returned incompatible "
        f"queries: {counts}. Move this workload to experiments/{destination}."
    )

