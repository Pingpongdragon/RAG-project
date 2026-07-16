import unittest
from types import SimpleNamespace

import numpy as np

from algorithms.drip.config import DRIPConfig
from algorithms.drip.domain_adaptation import DomainAdapter
from algorithms.drip.policy import DRIP
from algorithms.drip.topic_partition import build_topic_partition
from benchmarks.run_controlled_topic_trace import _feedback_window


EMBEDDINGS = np.asarray([
    [1.0, 0.0],
    [0.9, 0.1],
    [0.0, 1.0],
    [0.1, 0.9],
], dtype=np.float32)


class DomainAdapterTest(unittest.TestCase):
    def setUp(self):
        self.partition = build_topic_partition(
            "metadata", document_labels=["a", "a", "b", "b"]
        )

    def test_prior_changes_ambiguous_route(self):
        adapter = DomainAdapter(
            self.partition,
            EMBEDDINGS,
            prior_rate=0.5,
            prior_weight=1.0,
            route_width=1,
            retrieve_topk=2,
        )
        query = np.asarray([[0.7, 0.7]], dtype=np.float32)
        self.assertEqual(adapter.route(query).queries[0].regions, (0,))
        adapter.observe([2, 3, 2])
        self.assertEqual(adapter.route(query).queries[0].regions, (1,))

    def test_query_only_ablation_ignores_prior(self):
        adapter = DomainAdapter(
            self.partition,
            EMBEDDINGS,
            prior_rate=1.0,
            prior_weight=0.0,
            route_width=1,
            retrieve_topk=2,
        )
        query = np.asarray([[0.7, 0.7]], dtype=np.float32)
        before = adapter.route(query).queries[0]
        adapter.observe([2, 3])
        after = adapter.route(query).queries[0]
        self.assertEqual(before.regions, after.regions)
        self.assertEqual(before.documents, after.documents)

    def test_results_stay_inside_regions_and_budget(self):
        adapter = DomainAdapter(
            self.partition,
            EMBEDDINGS,
            route_width=1,
            retrieve_topk=2,
            candidate_budget=1,
        )
        routed = adapter.route(np.asarray([
            [1.0, 0.0], [0.0, 1.0]
        ], dtype=np.float32))
        self.assertLessEqual(len(routed.unique_documents), 1)
        for item in routed.queries:
            allowed = {
                position for topic in item.regions
                for position in self.partition.hard_bucket(topic)
            }
            self.assertTrue(set(item.documents) <= allowed)


class DomainPolicyCausalityTest(unittest.TestCase):
    def test_prepare_does_not_update_prior_or_hot_cache(self):
        documents = [
            {"doc_id": f"d{i}", "title": f"t{i}", "topic": topic}
            for i, topic in enumerate(["a", "a", "b", "b"])
        ]
        policy = DRIP(
            "domain",
            documents,
            EMBEDDINGS,
            {document["title"]: i for i, document in enumerate(documents)},
            DRIPConfig.domain_adapt(
                candidate_budget=2,
                metadata_field="topic",
                domain_prior_rate=0.5,
                domain_prior_weight=1.0,
                domain_route_width=1,
                domain_retrieve_topk=2,
                initial_dual_price=10.0,
            ),
        )
        policy.set_kb({"d0", "d1"})
        prior = policy.domain_adapter.prior.copy()
        hot = set(policy.kb)
        query_embedding = EMBEDDINGS[[2]]
        policy.prepare_window([{}], query_embedding, 0)
        np.testing.assert_allclose(policy.domain_adapter.prior, prior)
        self.assertEqual(policy.kb, hot)

        policy.step([{"access_title": "t2"}], query_embedding, 0)
        self.assertGreater(policy.domain_adapter.prior[1], prior[1])


class ControlledRunnerAlignmentTest(unittest.TestCase):
    def test_multisupport_feedback_keeps_original_query_rows(self):
        dataset = SimpleNamespace(
            title_to_idx={"a": 0, "b": 1, "c": 2},
            doc_pool=[
                {"title": "a"}, {"title": "b"}, {"title": "c"}
            ],
        )
        feedback, positions, supports, query_rows = _feedback_window(
            dataset,
            [
                {"sf_titles": ["a", "b"]},
                {"sf_titles": ["c"]},
            ],
        )
        self.assertEqual([item["access_title"] for item in feedback], [
            "a", "b", "c"
        ])
        np.testing.assert_array_equal(positions, [0, 1, 2])
        self.assertEqual(supports, [{0, 1}, {2}])
        np.testing.assert_array_equal(query_rows, [0, 0, 1])



if __name__ == "__main__":
    unittest.main()
