"""QA 分支严格校验可见性；agent trace 允许逐 query 混合标注。"""

from experiments.visibility import require_visibility


def test_direct_visibility_accepts_direct_queries():
    require_visibility(
        "direct", "direct_fixture",
        [{"evidence_visibility": "direct"}],
    )


def test_hidden_visibility_accepts_hidden_queries():
    require_visibility(
        "hidden", "hidden_fixture",
        [{"evidence_visibility": "hidden"}],
    )


def test_visibility_rejects_cross_branch_and_missing_labels():
    queries = [
        {"evidence_visibility": "hidden"},
        {"question": "missing label"},
    ]
    try:
        require_visibility("direct", "mixed_fixture", queries)
    except ValueError as error:
        assert "experiments/hidden" in str(error)
    else:
        raise AssertionError("cross-branch loader output must be rejected")


def test_agent_trace_can_carry_mixed_visibility_labels():
    queries = [
        {"evidence_visibility": "direct", "step_idx": 0},
        {"evidence_visibility": "hidden", "step_idx": 1},
    ]
    labels = {query["evidence_visibility"] for query in queries}
    assert labels == {"direct", "hidden"}
