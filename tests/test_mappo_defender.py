"""Unit tests for mappo_defender.py and capability_retention.py.

Coverage
--------
MAPPOConfig          – invalid ranges raise ValueError; defaults are valid.
compute_kl_penalty   – zero when current==ref; positive when diverged;
                       monotone in beta_kl; empty list returns 0; bad beta raises.
compute_policy_loss  – zero ratio & zero advantage; clipping on high ratio;
                       clipping on low ratio; negative advantage clips correctly;
                       empty input raises; mismatched length raises.
compute_value_loss   – perfect prediction → 0; off-prediction > 0; mismatched raises.
compute_entropy_bonus – proportional to coefficient; negative coefficient raises.
compute_total_loss   – components appear correctly in output; MAPPOLossOutput frozen.
defender_update_step – end-to-end: benign step; clipped step; zero-KL step.
CapabilityRetention  – fixture probe bank; run_capability_probe result shape;
                       all fixture probes pass against ReviewOneBaselineDefender;
                       regression flag raised below threshold; per-category rates.
"""

import math
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from defender_policy.capability_retention import (  # noqa: E402
    FIXTURE_PROBES,
    CapabilityRetentionReport,
    ProbeCase,
    ProbeResult,
    run_capability_probe,
)
from defender_policy.mappo_defender import (  # noqa: E402
    MAPPOConfig,
    MAPPOLossOutput,
    compute_entropy_bonus,
    compute_kl_penalty,
    compute_policy_loss,
    compute_total_loss,
    compute_value_loss,
    defender_update_step,
)


# ════════════════════════════════════════════════════════════════════════
# MAPPOConfig tests
# ════════════════════════════════════════════════════════════════════════

class TestMAPPOConfig(unittest.TestCase):

    def test_defaults_are_valid(self) -> None:
        cfg = MAPPOConfig()
        self.assertGreater(cfg.clip_epsilon, 0)
        self.assertGreaterEqual(cfg.beta_kl, 0)
        self.assertGreaterEqual(cfg.value_loss_coeff, 0)

    def test_negative_beta_kl_raises(self) -> None:
        with self.assertRaises(ValueError):
            MAPPOConfig(beta_kl=-0.1)

    def test_clip_epsilon_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            MAPPOConfig(clip_epsilon=0.0)

    def test_clip_epsilon_above_one_raises(self) -> None:
        with self.assertRaises(ValueError):
            MAPPOConfig(clip_epsilon=1.1)

    def test_gamma_above_one_raises(self) -> None:
        with self.assertRaises(ValueError):
            MAPPOConfig(gamma=1.01)

    def test_gae_lambda_above_one_raises(self) -> None:
        with self.assertRaises(ValueError):
            MAPPOConfig(gae_lambda=1.01)

    def test_negative_value_loss_coeff_raises(self) -> None:
        with self.assertRaises(ValueError):
            MAPPOConfig(value_loss_coeff=-0.01)

    def test_zero_beta_kl_is_valid(self) -> None:
        cfg = MAPPOConfig(beta_kl=0.0)
        self.assertEqual(cfg.beta_kl, 0.0)

    def test_config_is_immutable(self) -> None:
        cfg = MAPPOConfig()
        with self.assertRaises((AttributeError, TypeError)):
            cfg.beta_kl = 99.0  # type: ignore[misc]


# ════════════════════════════════════════════════════════════════════════
# compute_kl_penalty tests
# ════════════════════════════════════════════════════════════════════════

class TestComputeKLPenalty(unittest.TestCase):

    def test_zero_when_identical(self) -> None:
        lp = [-1.0, -2.0, -0.5]
        result = compute_kl_penalty(lp, lp, beta_kl=1.0)
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_positive_when_diverged(self) -> None:
        current = [-1.0, -2.0]
        ref     = [-2.0, -3.0]     # current is sharper → KL > 0
        result = compute_kl_penalty(current, ref, beta_kl=1.0)
        self.assertGreater(result, 0.0)

    def test_scales_with_beta_kl(self) -> None:
        current = [-1.0]
        ref     = [-2.0]
        low  = compute_kl_penalty(current, ref, beta_kl=0.1)
        high = compute_kl_penalty(current, ref, beta_kl=1.0)
        self.assertAlmostEqual(high / low, 10.0, places=8)

    def test_empty_list_returns_zero(self) -> None:
        self.assertEqual(compute_kl_penalty([], [], beta_kl=1.0), 0.0)

    def test_negative_beta_kl_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_kl_penalty([-1.0], [-1.0], beta_kl=-0.1)

    def test_mismatched_length_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_kl_penalty([-1.0, -2.0], [-1.0], beta_kl=1.0)

    def test_zero_beta_gives_zero_penalty(self) -> None:
        result = compute_kl_penalty([-1.0, -2.5], [-0.5, -1.5], beta_kl=0.0)
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_single_token_manual(self) -> None:
        """Manual calculation: KL = exp(lp_cur) * (lp_cur - lp_ref)."""
        lp_cur, lp_ref, beta = -1.0, -2.0, 1.0
        expected = beta * (math.exp(lp_cur) * (lp_cur - lp_ref))
        result = compute_kl_penalty([lp_cur], [lp_ref], beta_kl=beta)
        self.assertAlmostEqual(result, expected, places=10)


# ════════════════════════════════════════════════════════════════════════
# compute_policy_loss tests
# ════════════════════════════════════════════════════════════════════════

class TestComputePolicyLoss(unittest.TestCase):

    def test_zero_ratio_zero_advantage(self) -> None:
        """r=1 (log_ratio=0), A=0 → loss=0."""
        result = compute_policy_loss([0.0], [0.0], clip_epsilon=0.2)
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_positive_advantage_reduces_loss(self) -> None:
        """Positive advantage with ratio=1 → negative of positive value → should be negative."""
        result = compute_policy_loss([1.0], [0.0], clip_epsilon=0.2)
        self.assertLess(result, 0.0)

    def test_high_ratio_is_clipped(self) -> None:
        """Very high ratio should be clipped to 1+ε."""
        eps = 0.2
        adv = 1.0
        # Clipped value = (1+eps) * adv = 1.2; unclipped = huge
        # Loss = -clipped = -1.2
        result = compute_policy_loss([adv], [math.log(100.0)], clip_epsilon=eps)
        self.assertAlmostEqual(result, -(1.0 + eps) * adv, places=5)

    def test_low_ratio_is_clipped_positive_advantage(self) -> None:
        """Very low ratio with positive advantage should be clipped to 1-ε."""
        eps = 0.2
        adv = 1.0
        # Clipped value = (1-eps) * adv = 0.8; unclipped = tiny
        result = compute_policy_loss([adv], [math.log(0.001)], clip_epsilon=eps)
        # min(unclipped, clipped) = unclipped (very small * adv) → less than clipped
        # so the min picks the unclipped side (worst case for the policy)
        self.assertGreater(result, -(1.0 - eps) * adv)

    def test_negative_advantage_clips_correctly(self) -> None:
        """Negative advantage with high ratio → clipped at 1+ε.

        r ≈ 100, A = -1.0
        unclipped = 100 * (-1) = -100   (very negative)
        clipped   = (1+ε) * (-1) = -1.2 (less negative)
        min(-100, -1.2) = -100   → negated loss = +100

        Wait — min picks the more negative value (-100), so loss = -(-100/1) = +100.
        But the clip is meant to prevent large updates. For negative advantage,
        clip(r, lo, hi)*A = (1+ε)*(-1) = -1.2 and unclipped = -100.
        min(-100, -1.2) = -100 → loss = +100 (unclipped dominates).
        This is by PPO design: for A<0 the pessimistic (most-penalising) side wins.
        """
        eps = 0.2
        adv = -1.0
        result = compute_policy_loss([adv], [math.log(100.0)], clip_epsilon=eps)
        # The unclipped side wins: loss = -(r * A) = -(-100) = +100
        self.assertAlmostEqual(result, 100.0, places=2)

    def test_empty_advantages_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_policy_loss([], [], clip_epsilon=0.2)

    def test_mismatched_lengths_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_policy_loss([1.0, 2.0], [0.0], clip_epsilon=0.2)

    def test_invalid_clip_epsilon_zero_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_policy_loss([1.0], [0.0], clip_epsilon=0.0)

    def test_invalid_clip_epsilon_over_one_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_policy_loss([1.0], [0.0], clip_epsilon=1.5)

    def test_multi_step_mean(self) -> None:
        """Loss is the mean across steps, not the sum."""
        eps = 0.2
        single = compute_policy_loss([1.0], [0.0], clip_epsilon=eps)
        double = compute_policy_loss([1.0, 1.0], [0.0, 0.0], clip_epsilon=eps)
        self.assertAlmostEqual(single, double, places=10)


# ════════════════════════════════════════════════════════════════════════
# compute_value_loss tests
# ════════════════════════════════════════════════════════════════════════

class TestComputeValueLoss(unittest.TestCase):

    def test_perfect_prediction_gives_zero(self) -> None:
        result = compute_value_loss([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_off_prediction_gives_positive_loss(self) -> None:
        result = compute_value_loss([0.0], [1.0])
        self.assertAlmostEqual(result, 1.0, places=10)

    def test_mse_is_mean_not_sum(self) -> None:
        result = compute_value_loss([0.0, 0.0], [1.0, 1.0])
        self.assertAlmostEqual(result, 1.0, places=10)

    def test_empty_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_value_loss([], [])

    def test_mismatched_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_value_loss([1.0], [1.0, 2.0])


# ════════════════════════════════════════════════════════════════════════
# compute_entropy_bonus tests
# ════════════════════════════════════════════════════════════════════════

class TestComputeEntropyBonus(unittest.TestCase):

    def test_zero_coefficient_gives_zero(self) -> None:
        result = compute_entropy_bonus([0.5, 0.7], entropy_coeff=0.0)
        self.assertAlmostEqual(result, 0.0, places=10)

    def test_bonus_is_negative(self) -> None:
        """Entropy bonus subtracts from loss (negative sign)."""
        result = compute_entropy_bonus([1.0], entropy_coeff=0.01)
        self.assertLess(result, 0.0)

    def test_scales_with_entropy(self) -> None:
        """Higher entropy → more negative bonus (larger magnitude)."""
        low  = compute_entropy_bonus([0.5], entropy_coeff=0.01)
        high = compute_entropy_bonus([1.0], entropy_coeff=0.01)
        self.assertLess(high, low)   # more negative = less in numerical value

    def test_negative_coeff_raises(self) -> None:
        with self.assertRaises(ValueError):
            compute_entropy_bonus([1.0], entropy_coeff=-0.01)

    def test_empty_list_returns_zero(self) -> None:
        result = compute_entropy_bonus([], entropy_coeff=0.01)
        self.assertAlmostEqual(result, 0.0, places=10)


# ════════════════════════════════════════════════════════════════════════
# compute_total_loss tests
# ════════════════════════════════════════════════════════════════════════

class TestComputeTotalLoss(unittest.TestCase):

    def _make_output(self, **kwargs) -> MAPPOLossOutput:
        cfg = MAPPOConfig()
        defaults = dict(policy_loss=0.5, kl_penalty=0.1, value_loss=0.2,
                        entropy_bonus=-0.05, config=cfg)
        defaults.update(kwargs)
        return compute_total_loss(**defaults)

    def test_components_sum_correctly(self) -> None:
        cfg = MAPPOConfig(value_loss_coeff=0.5)
        out = compute_total_loss(
            policy_loss=1.0,
            kl_penalty=0.2,
            value_loss=0.4,
            entropy_bonus=-0.1,
            config=cfg,
        )
        # 1.0 + 0.2 + 0.5*0.4 + (-0.1) = 1.0 + 0.2 + 0.2 - 0.1 = 1.3
        self.assertAlmostEqual(out.total_loss, 1.3, places=10)

    def test_output_fields_match_inputs(self) -> None:
        out = self._make_output()
        self.assertEqual(out.policy_loss, 0.5)
        self.assertEqual(out.kl_penalty, 0.1)
        self.assertEqual(out.value_loss, 0.2)
        self.assertEqual(out.entropy_bonus, -0.05)

    def test_output_is_immutable(self) -> None:
        out = self._make_output()
        with self.assertRaises((AttributeError, TypeError)):
            out.total_loss = 99.0  # type: ignore[misc]


# ════════════════════════════════════════════════════════════════════════
# defender_update_step end-to-end tests
# ════════════════════════════════════════════════════════════════════════

class TestDefenderUpdateStep(unittest.TestCase):

    def _benign_step(self) -> MAPPOLossOutput:
        return defender_update_step(
            advantages=[1.0],
            log_ratios=[0.0],        # ratio=1 (no change)
            current_logprobs=[-1.0],
            ref_logprobs=[-1.0],     # identical → KL=0
            value_estimates=[1.0],
            returns=[1.0],           # perfect prediction → value_loss=0
            entropies=[0.5],
            config=MAPPOConfig(beta_kl=0.1, value_loss_coeff=0.5, entropy_coeff=0.01),
        )

    def test_zero_kl_when_current_equals_ref(self) -> None:
        out = self._benign_step()
        self.assertAlmostEqual(out.kl_penalty, 0.0, places=10)

    def test_zero_value_loss_when_perfect(self) -> None:
        out = self._benign_step()
        self.assertAlmostEqual(out.value_loss, 0.0, places=10)

    def test_total_loss_has_all_components(self) -> None:
        out = self._benign_step()
        self.assertIsInstance(out.total_loss, float)
        self.assertIsInstance(out.policy_loss, float)
        self.assertIsInstance(out.kl_penalty, float)
        self.assertIsInstance(out.entropy_bonus, float)

    def test_higher_beta_kl_increases_loss_when_diverged(self) -> None:
        kwargs = dict(
            advantages=[1.0], log_ratios=[0.0],
            current_logprobs=[-1.0], ref_logprobs=[-2.0],
            value_estimates=[1.0], returns=[1.0],
        )
        low  = defender_update_step(**kwargs, config=MAPPOConfig(beta_kl=0.0))
        high = defender_update_step(**kwargs, config=MAPPOConfig(beta_kl=1.0))
        self.assertGreater(high.total_loss, low.total_loss)

    def test_clipped_step_output_is_bounded(self) -> None:
        out = defender_update_step(
            advantages=[1.0],
            log_ratios=[math.log(100.0)],   # very large ratio → clipped
            current_logprobs=[-1.0],
            ref_logprobs=[-1.0],
            value_estimates=[0.0],
            returns=[1.0],
            config=MAPPOConfig(clip_epsilon=0.2, beta_kl=0.0, value_loss_coeff=0.0),
        )
        # policy_loss should be exactly -(1+ε)*A = -1.2 when clipped
        self.assertAlmostEqual(out.policy_loss, -1.2, places=5)


# ════════════════════════════════════════════════════════════════════════
# capability_retention tests
# ════════════════════════════════════════════════════════════════════════

class TestCapabilityRetentionFixtures(unittest.TestCase):

    def test_fixture_probe_bank_is_non_empty(self) -> None:
        self.assertGreater(len(FIXTURE_PROBES), 0)

    def test_all_probe_ids_unique(self) -> None:
        ids = [p.probe_id for p in FIXTURE_PROBES]
        self.assertEqual(len(ids), len(set(ids)))

    def test_all_expected_statuses_are_valid(self) -> None:
        valid = {"answer", "tool_used"}
        for probe in FIXTURE_PROBES:
            self.assertIn(probe.expected_status, valid, msg=f"{probe.probe_id}")

    def test_probe_is_immutable(self) -> None:
        probe = FIXTURE_PROBES[0]
        with self.assertRaises((AttributeError, TypeError)):
            probe.probe_id = "new_id"  # type: ignore[misc]


class TestRunCapabilityProbe(unittest.TestCase):

    def setUp(self) -> None:
        self.report = run_capability_probe()

    def test_report_total_matches_fixture_bank(self) -> None:
        self.assertEqual(self.report.total_probes, len(FIXTURE_PROBES))

    def test_all_fixture_probes_pass_baseline(self) -> None:
        """ReviewOneBaselineDefender must pass all benign capability probes."""
        self.assertEqual(
            self.report.failed, 0,
            msg=f"Failed probes: {[r.probe_id for r in self.report.results if not r.passed]}",
        )

    def test_retention_score_is_one_for_baseline(self) -> None:
        self.assertAlmostEqual(self.report.retention_score, 1.0, places=10)

    def test_regression_not_flagged_at_full_score(self) -> None:
        self.assertFalse(self.report.regression_flagged)

    def test_regression_flagged_below_threshold(self) -> None:
        # Use a threshold above 1.0 to force a flag
        report = run_capability_probe(threshold=1.01)
        self.assertTrue(report.regression_flagged)

    def test_per_category_rates_are_in_range(self) -> None:
        for cat, rate in self.report.per_category_pass_rate.items():
            self.assertGreaterEqual(rate, 0.0, msg=cat)
            self.assertLessEqual(rate, 1.0, msg=cat)

    def test_scope_disclaimer_present(self) -> None:
        self.assertIn("capability-retention probe", self.report.scope.lower())

    def test_result_list_matches_total(self) -> None:
        self.assertEqual(len(self.report.results), self.report.total_probes)

    def test_custom_probe_bank(self) -> None:
        custom = [ProbeCase(
            probe_id="custom_01",
            probe_category="test",
            user_task="Summarize the email.",
            untrusted_content="All services healthy.",
            expected_status="answer",
        )]
        report = run_capability_probe(probes=custom)
        self.assertEqual(report.total_probes, 1)
        self.assertTrue(report.results[0].passed)


if __name__ == "__main__":
    unittest.main()
