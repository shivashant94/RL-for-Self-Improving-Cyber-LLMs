"""Unit tests for evaluate_checkpoint.py.

Coverage
--------
evaluate_checkpoint()   – default fixture_baseline run; in_distribution metrics present;
                          heldout blocked by default; allow_heldout=True loads heldout;
                          wrong heldout path raises FileNotFoundError;
                          custom checkpoint_id stored; seed stored;
                          dataset_version stored; evaluation_scope has disclaimer.
SliceMetrics            – frozen; all expected fields; n_cases > 0.
CheckpointEvalResult    – heldout_ood is None when allow_heldout=False;
                          heldout_ood populated when allow_heldout=True;
                          in_distribution ASR is 0 for fixture baseline;
                          per_family_asr present and all-zero for fixture;
                          tool_block_rate == 1.0 for fixture.
Heldout guard           – calling without allow_heldout does NOT load heldout data;
                          calling with allow_heldout=True DOES load it;
                          bad path raises FileNotFoundError when allow_heldout=True.
Reward integration      – avg_total_reward is a float; present in SliceMetrics.
Reproducibility         – same seed + same adapter → same in_distribution metrics.
"""

import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from defender_policy.evaluate_checkpoint import (  # noqa: E402
    CheckpointEvalResult,
    SliceMetrics,
    evaluate_checkpoint,
)
from defender_policy.model_adapter import FixtureModelAdapter  # noqa: E402


# ── Helpers ───────────────────────────────────────────────────────────────────

def _run(allow_heldout: bool = False, **kwargs) -> CheckpointEvalResult:
    return evaluate_checkpoint(
        checkpoint_id=kwargs.pop("checkpoint_id", "test_fixture"),
        adapter=FixtureModelAdapter(),
        allow_heldout=allow_heldout,
        seed=kwargs.pop("seed", 42),
        **kwargs,
    )


# ════════════════════════════════════════════════════════════════════════
# evaluate_checkpoint — basic contract
# ════════════════════════════════════════════════════════════════════════

class TestEvaluateCheckpointBasic(unittest.TestCase):

    def setUp(self) -> None:
        self.result = _run()

    def test_returns_checkpoint_eval_result(self) -> None:
        self.assertIsInstance(self.result, CheckpointEvalResult)

    def test_checkpoint_id_stored(self) -> None:
        self.assertEqual(self.result.checkpoint_id, "test_fixture")

    def test_seed_stored(self) -> None:
        self.assertEqual(self.result.seed, 42)

    def test_dataset_version_stored(self) -> None:
        self.assertIsNotNone(self.result.dataset_version)
        self.assertTrue(self.result.dataset_version)

    def test_evaluation_scope_present(self) -> None:
        self.assertTrue(self.result.evaluation_scope)

    def test_allow_heldout_false_by_default(self) -> None:
        self.assertFalse(self.result.allow_heldout)

    def test_heldout_ood_is_none_by_default(self) -> None:
        self.assertIsNone(self.result.heldout_ood)

    def test_in_distribution_is_present(self) -> None:
        self.assertIsNotNone(self.result.in_distribution)
        self.assertIsInstance(self.result.in_distribution, SliceMetrics)

    def test_raw_indist_report_is_dict(self) -> None:
        self.assertIsInstance(self.result.raw_report_indist, dict)

    def test_raw_heldout_report_is_none_by_default(self) -> None:
        self.assertIsNone(self.result.raw_report_heldout)


# ════════════════════════════════════════════════════════════════════════
# SliceMetrics — in-distribution
# ════════════════════════════════════════════════════════════════════════

class TestInDistributionMetrics(unittest.TestCase):

    def setUp(self) -> None:
        self.metrics = _run().in_distribution

    def test_slice_name_is_in_distribution(self) -> None:
        self.assertEqual(self.metrics.slice_name, "in_distribution")

    def test_n_cases_positive(self) -> None:
        self.assertGreater(self.metrics.n_cases, 0)

    def test_asr_is_zero_for_fixture(self) -> None:
        """Fixture baseline must resist all attacks."""
        self.assertAlmostEqual(self.metrics.attack_success_rate, 0.0, places=10)

    def test_tool_block_rate_is_one(self) -> None:
        self.assertAlmostEqual(self.metrics.tool_block_rate, 1.0, places=10)

    def test_per_family_asr_present(self) -> None:
        self.assertIsInstance(self.metrics.per_family_asr, dict)
        self.assertGreater(len(self.metrics.per_family_asr), 0)

    def test_per_family_asr_all_zero(self) -> None:
        for family, asr in self.metrics.per_family_asr.items():
            self.assertAlmostEqual(asr, 0.0, places=10, msg=f"Family {family} has ASR > 0")

    def test_avg_total_reward_is_float(self) -> None:
        self.assertIsInstance(self.metrics.avg_total_reward, float)

    def test_all_rates_in_unit_interval(self) -> None:
        for name, val in [
            ("asr", self.metrics.attack_success_rate),
            ("benign_success", self.metrics.benign_task_success_rate),
            ("false_refusal", self.metrics.false_refusal_rate),
            ("tool_block_rate", self.metrics.tool_block_rate),
        ]:
            self.assertGreaterEqual(val, 0.0, msg=name)
            self.assertLessEqual(val, 1.0, msg=name)

    def test_metrics_is_frozen(self) -> None:
        with self.assertRaises((AttributeError, TypeError)):
            self.metrics.attack_success_rate = 99.0  # type: ignore[misc]

    def test_evaluation_scope_has_disclaimer(self) -> None:
        self.assertTrue(self.metrics.evaluation_scope)


# ════════════════════════════════════════════════════════════════════════
# Held-out guard tests
# ════════════════════════════════════════════════════════════════════════

class TestHeldoutGuard(unittest.TestCase):

    def test_heldout_not_loaded_without_flag(self) -> None:
        result = _run(allow_heldout=False)
        self.assertIsNone(result.heldout_ood)

    def test_heldout_loaded_with_flag(self) -> None:
        result = _run(allow_heldout=True)
        self.assertIsNotNone(result.heldout_ood)
        self.assertIsInstance(result.heldout_ood, SliceMetrics)

    def test_heldout_slice_name(self) -> None:
        result = _run(allow_heldout=True)
        self.assertEqual(result.heldout_ood.slice_name, "heldout_ood")

    def test_heldout_n_cases_positive(self) -> None:
        result = _run(allow_heldout=True)
        self.assertGreater(result.heldout_ood.n_cases, 0)

    def test_heldout_asr_is_zero_for_fixture(self) -> None:
        result = _run(allow_heldout=True)
        self.assertAlmostEqual(result.heldout_ood.attack_success_rate, 0.0, places=10)

    def test_bad_heldout_path_raises(self) -> None:
        with self.assertRaises(FileNotFoundError):
            evaluate_checkpoint(
                checkpoint_id="test",
                adapter=FixtureModelAdapter(),
                heldout_cases_path=Path("/nonexistent/heldout.jsonl"),
                allow_heldout=True,
            )

    def test_allow_heldout_stored_in_result(self) -> None:
        result = _run(allow_heldout=True)
        self.assertTrue(result.allow_heldout)

    def test_raw_heldout_report_present_when_evaluated(self) -> None:
        result = _run(allow_heldout=True)
        self.assertIsNotNone(result.raw_report_heldout)
        self.assertIsInstance(result.raw_report_heldout, dict)


# ════════════════════════════════════════════════════════════════════════
# Reproducibility
# ════════════════════════════════════════════════════════════════════════

class TestReproducibility(unittest.TestCase):

    def test_same_seed_same_asr(self) -> None:
        r1 = _run(seed=42)
        r2 = _run(seed=42)
        self.assertEqual(
            r1.in_distribution.attack_success_rate,
            r2.in_distribution.attack_success_rate,
        )

    def test_same_seed_same_avg_reward(self) -> None:
        r1 = _run(seed=7)
        r2 = _run(seed=7)
        self.assertAlmostEqual(
            r1.in_distribution.avg_total_reward,
            r2.in_distribution.avg_total_reward,
            places=10,
        )

    def test_same_seed_same_n_cases(self) -> None:
        r1 = _run(seed=1)
        r2 = _run(seed=1)
        self.assertEqual(
            r1.in_distribution.n_cases,
            r2.in_distribution.n_cases,
        )


# ════════════════════════════════════════════════════════════════════════
# Custom checkpoint_id
# ════════════════════════════════════════════════════════════════════════

class TestCustomCheckpointId(unittest.TestCase):

    def test_custom_id_stored(self) -> None:
        result = _run(checkpoint_id="sft-epoch-3")
        self.assertEqual(result.checkpoint_id, "sft-epoch-3")

    def test_scope_references_custom_id(self) -> None:
        result = _run(checkpoint_id="mappo-iter-50")
        self.assertIn("mappo-iter-50", result.evaluation_scope)

    def test_fixture_baseline_scope_has_fixture_note(self) -> None:
        result = evaluate_checkpoint(
            checkpoint_id="fixture_baseline",
            adapter=FixtureModelAdapter(),
        )
        self.assertIn("fixture", result.evaluation_scope.lower())


if __name__ == "__main__":
    unittest.main()
