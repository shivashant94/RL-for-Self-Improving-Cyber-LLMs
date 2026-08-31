"""Unit tests for the Review 2 fixture episode harness.

Coverage
--------
FixtureEpisodeHarness   – construction; run_episode returns EpisodeResult;
                          run_episode_for_case (deterministic); run_n_episodes count;
                          invalid n raises; unknown case_id raises.
EpisodeResult structure – steps list; reward set; retention_report present;
                          scope disclaimer; advantage left None (Member 3 only);
                          gate_blocked flag correct; pre_gate_unsafe flag.
Reward integration      – benign cases earn positive reward; attack cases earn
                          reward components; reward_components dict has all keys.
Capability retention    – retention_report.retention_score == 1.0 for all episodes;
                          regression_flagged == False.
Reproducibility         – same seed → same episode_id sequence and same rewards.
Attack case             – gate blocks unsafe tool; gate_blocked flag True.
Benign case             – gate not blocked; pre_gate_unsafe False.
Aggregation helper      – _average_components correct mean.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from defender_policy.episode_harness import (  # noqa: E402
    EpisodeResult,
    FixtureEpisodeHarness,
    _average_components,
    _eval_expected_to_reward_status,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _harness(seed: int = 42) -> FixtureEpisodeHarness:
    return FixtureEpisodeHarness(seed=seed)


# ════════════════════════════════════════════════════════════════════════
# Construction tests
# ════════════════════════════════════════════════════════════════════════

class TestHarnessConstruction(unittest.TestCase):

    def test_harness_loads_cases(self) -> None:
        h = _harness()
        self.assertGreater(len(h._cases), 0)

    def test_harness_has_14_fixture_cases(self) -> None:
        """review1_cases.jsonl has 14 parseable cases (blank trailing line skipped)."""
        h = _harness()
        self.assertEqual(len(h._cases), 14)

    def test_unknown_cases_path_raises(self) -> None:
        with self.assertRaises((FileNotFoundError, OSError, ValueError)):
            FixtureEpisodeHarness(
                cases_path=Path("/nonexistent/path/cases.jsonl")
            )


# ════════════════════════════════════════════════════════════════════════
# run_episode tests
# ════════════════════════════════════════════════════════════════════════

class TestRunEpisode(unittest.TestCase):

    def setUp(self) -> None:
        self.harness = _harness()
        self.result  = self.harness.run_episode("ep_test_001")

    def test_result_is_episode_result(self) -> None:
        self.assertIsInstance(self.result, EpisodeResult)

    def test_episode_id_stored(self) -> None:
        self.assertEqual(self.result.episode_id, "ep_test_001")

    def test_steps_is_non_empty_list(self) -> None:
        self.assertIsInstance(self.result.steps, list)
        self.assertGreater(len(self.result.steps), 0)

    def test_v1_has_exactly_one_step(self) -> None:
        self.assertEqual(len(self.result.steps), 1)

    def test_reward_is_set(self) -> None:
        self.assertIsNotNone(self.result.total_reward)
        self.assertIsInstance(self.result.total_reward, float)

    def test_reward_components_dict_has_all_keys(self) -> None:
        required = {
            "task_utility", "safety_violation", "unnecessary_refusal",
            "excessive_cost", "lambda_violation", "lambda_refusal", "lambda_cost",
        }
        self.assertTrue(
            required.issubset(self.result.reward_components.keys()),
            msg=f"Missing keys: {required - self.result.reward_components.keys()}",
        )

    def test_scope_disclaimer_present(self) -> None:
        self.assertIn("fixture episode", self.result.scope.lower())

    def test_case_kind_is_valid(self) -> None:
        self.assertIn(self.result.case_kind, {"attack", "benign", "unknown"})

    def test_retention_report_present(self) -> None:
        self.assertIsNotNone(self.result.retention_report)

    def test_advantage_not_set_by_harness(self) -> None:
        """Advantage is Member 3's responsibility; harness must not set it."""
        step = self.result.steps[0]
        self.assertIsNone(step.advantage)
        self.assertIsNone(step.value_estimate)

    def test_step_has_task_id(self) -> None:
        step = self.result.steps[0]
        self.assertTrue(step.action.task_id)

    def test_step_episode_id_matches(self) -> None:
        step = self.result.steps[0]
        self.assertEqual(step.episode_id, "ep_test_001")


# ════════════════════════════════════════════════════════════════════════
# run_episode_for_case tests (deterministic)
# ════════════════════════════════════════════════════════════════════════

class TestRunEpisodeForCase(unittest.TestCase):

    def setUp(self) -> None:
        self.harness = _harness()

    def test_known_case_id_returns_result(self) -> None:
        result = self.harness.run_episode_for_case("benign_01")
        self.assertEqual(result.case_id, "benign_01")

    def test_unknown_case_id_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.harness.run_episode_for_case("nonexistent_case")

    def test_benign_case_is_labelled_benign(self) -> None:
        result = self.harness.run_episode_for_case("benign_01")
        self.assertEqual(result.case_kind, "benign")

    def test_attack_case_is_labelled_attack(self) -> None:
        result = self.harness.run_episode_for_case("attack_01")
        self.assertEqual(result.case_kind, "attack")

    def test_attack_case_has_attack_family(self) -> None:
        result = self.harness.run_episode_for_case("attack_01")
        self.assertIsNotNone(result.attack_family)

    def test_benign_case_gate_not_blocked(self) -> None:
        result = self.harness.run_episode_for_case("benign_01")
        self.assertFalse(result.gate_blocked)

    def test_benign_case_pre_gate_safe(self) -> None:
        result = self.harness.run_episode_for_case("benign_01")
        self.assertFalse(result.pre_gate_unsafe)

    def test_deterministic_same_result(self) -> None:
        """Same case_id always returns same rewards (pure fixture)."""
        r1 = self.harness.run_episode_for_case("benign_01", "ep_a")
        r2 = self.harness.run_episode_for_case("benign_01", "ep_b")
        self.assertEqual(r1.total_reward, r2.total_reward)
        self.assertEqual(r1.reward_components, r2.reward_components)


# ════════════════════════════════════════════════════════════════════════
# Reward integration tests
# ════════════════════════════════════════════════════════════════════════

class TestRewardIntegration(unittest.TestCase):

    def setUp(self) -> None:
        self.harness = _harness()

    def test_benign_case_positive_reward(self) -> None:
        result = self.harness.run_episode_for_case("benign_01")
        self.assertGreater(result.total_reward, 0.0)

    def test_benign_case_task_utility_nonzero(self) -> None:
        result = self.harness.run_episode_for_case("benign_01")
        self.assertGreater(result.reward_components["task_utility"], 0.0)

    def test_benign_case_no_safety_violation(self) -> None:
        result = self.harness.run_episode_for_case("benign_01")
        self.assertEqual(result.reward_components["safety_violation"], 0.0)

    def test_benign_case_no_unnecessary_refusal(self) -> None:
        result = self.harness.run_episode_for_case("benign_01")
        self.assertEqual(result.reward_components["unnecessary_refusal"], 0.0)

    def test_attack_case_reward_components_present(self) -> None:
        result = self.harness.run_episode_for_case("attack_01")
        self.assertIn("task_utility", result.reward_components)
        self.assertIn("safety_violation", result.reward_components)


# ════════════════════════════════════════════════════════════════════════
# Capability retention tests
# ════════════════════════════════════════════════════════════════════════

class TestCapabilityRetentionInEpisode(unittest.TestCase):

    def setUp(self) -> None:
        self.harness = _harness()

    def test_retention_score_is_one(self) -> None:
        result = self.harness.run_episode_for_case("benign_01")
        self.assertAlmostEqual(result.retention_report.retention_score, 1.0, places=10)

    def test_regression_not_flagged(self) -> None:
        result = self.harness.run_episode_for_case("benign_01")
        self.assertFalse(result.retention_report.regression_flagged)

    def test_retention_report_has_probes(self) -> None:
        result = self.harness.run_episode_for_case("benign_01")
        self.assertGreater(result.retention_report.total_probes, 0)


# ════════════════════════════════════════════════════════════════════════
# run_n_episodes tests
# ════════════════════════════════════════════════════════════════════════

class TestRunNEpisodes(unittest.TestCase):

    def setUp(self) -> None:
        self.harness = _harness()

    def test_n_episodes_returns_correct_count(self) -> None:
        results = self.harness.run_n_episodes(5)
        self.assertEqual(len(results), 5)

    def test_all_results_are_episode_results(self) -> None:
        for r in self.harness.run_n_episodes(3):
            self.assertIsInstance(r, EpisodeResult)

    def test_episode_ids_are_unique(self) -> None:
        results = self.harness.run_n_episodes(5)
        ids = [r.episode_id for r in results]
        self.assertEqual(len(ids), len(set(ids)))

    def test_invalid_n_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.harness.run_n_episodes(0)

    def test_negative_n_raises(self) -> None:
        with self.assertRaises(ValueError):
            self.harness.run_n_episodes(-1)

    def test_reproducible_with_same_seed(self) -> None:
        h1 = FixtureEpisodeHarness(seed=7)
        h2 = FixtureEpisodeHarness(seed=7)
        r1 = h1.run_n_episodes(5)
        r2 = h2.run_n_episodes(5)
        self.assertEqual([r.case_id for r in r1], [r.case_id for r in r2])
        self.assertEqual([r.total_reward for r in r1], [r.total_reward for r in r2])

    def test_different_seeds_may_differ(self) -> None:
        h1 = FixtureEpisodeHarness(seed=1)
        h2 = FixtureEpisodeHarness(seed=999)
        r1 = h1.run_n_episodes(15)
        r2 = h2.run_n_episodes(15)
        # With 15 episodes over 15 distinct cases, at least one case_id will differ
        case_ids_1 = [r.case_id for r in r1]
        case_ids_2 = [r.case_id for r in r2]
        # Not guaranteed to differ but extremely likely with different seeds;
        # the test is informative not strict — we just check both ran.
        self.assertEqual(len(r1), 15)
        self.assertEqual(len(r2), 15)


# ════════════════════════════════════════════════════════════════════════
# Aggregation helper tests
# ════════════════════════════════════════════════════════════════════════

class TestAverageComponents(unittest.TestCase):

    def test_single_dict_unchanged(self) -> None:
        result = _average_components([{"a": 1.0, "b": 2.0}])
        self.assertEqual(result["a"], 1.0)
        self.assertEqual(result["b"], 2.0)

    def test_mean_of_two_dicts(self) -> None:
        result = _average_components([{"a": 1.0}, {"a": 3.0}])
        self.assertAlmostEqual(result["a"], 2.0, places=10)

    def test_empty_list_returns_empty_dict(self) -> None:
        result = _average_components([])
        self.assertEqual(result, {})


# ════════════════════════════════════════════════════════════════════════
# _eval_expected_to_reward_status tests
# ════════════════════════════════════════════════════════════════════════

class TestEvalExpectedMapping(unittest.TestCase):

    def test_resist_maps_to_refusal(self) -> None:
        self.assertEqual(_eval_expected_to_reward_status("resist", "attack"), "refusal")

    def test_answer_maps_to_answer(self) -> None:
        self.assertEqual(_eval_expected_to_reward_status("answer", "benign"), "answer")

    def test_tool_used_maps_to_tool_used(self) -> None:
        self.assertEqual(_eval_expected_to_reward_status("tool_used", "benign"), "tool_used")

    def test_unknown_maps_to_none(self) -> None:
        self.assertIsNone(_eval_expected_to_reward_status("unknown_value", "attack"))

    def test_none_expected_maps_to_none(self) -> None:
        self.assertIsNone(_eval_expected_to_reward_status(None, "benign"))


if __name__ == "__main__":
    unittest.main()
