"""Unit tests for Part 9: ablation configs and capability-retention analysis.

Coverage
--------
AblationConfigs     – all three config files exist; valid JSON; required keys present;
                      no_kl has beta_kl==0.0; high_kl has beta_kl>=1.0;
                      no_gate documents gate invariant; gate_policy field present in all.
MAPPOConfigFromAblation – loaded beta_kl values are valid MAPPOConfig inputs;
                           no_kl and high_kl beta_kl differ from each other and the default.
CapabilityRetentionAnalysis – run_capability_probe at default threshold → no regression;
                               at custom threshold 1.01 → regression flagged;
                               per_category_pass_rate keys non-empty;
                               all probe categories covered in report;
                               report has scope/disclaimer attributes.
RetentionScriptSmoke – script exits 0 with --dry-run (subprocess call).
CSVAndJSON          – _build_report returns required top-level keys;
                      per_probe list length matches FIXTURE_PROBES;
                      all per_probe dicts have probe_id, passed, note fields.
"""

import json
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIGS_DIR = ROOT / "configs" / "ablation_configs"
sys.path.insert(0, str(ROOT / "src"))

from defender_policy.capability_retention import (  # noqa: E402
    FIXTURE_PROBES,
    run_capability_probe,
)
from defender_policy.mappo_defender import MAPPOConfig  # noqa: E402


# ════════════════════════════════════════════════════════════════════════
# Ablation config file tests
# ════════════════════════════════════════════════════════════════════════

class TestAblationConfigFiles(unittest.TestCase):

    def _load(self, filename: str) -> dict:
        path = CONFIGS_DIR / filename
        self.assertTrue(path.exists(), msg=f"Missing: {path}")
        with path.open(encoding="utf-8") as fh:
            return json.load(fh)

    # ── File existence ────────────────────────────────────────────────

    def test_no_kl_config_exists(self) -> None:
        self.assertTrue((CONFIGS_DIR / "ablation_no_kl.json").exists())

    def test_high_kl_config_exists(self) -> None:
        self.assertTrue((CONFIGS_DIR / "ablation_high_kl.json").exists())

    def test_no_gate_config_exists(self) -> None:
        self.assertTrue((CONFIGS_DIR / "ablation_no_gate.json").exists())

    # ── Valid JSON ────────────────────────────────────────────────────

    def test_no_kl_is_valid_json(self) -> None:
        self._load("ablation_no_kl.json")  # raises on parse error

    def test_high_kl_is_valid_json(self) -> None:
        self._load("ablation_high_kl.json")

    def test_no_gate_is_valid_json(self) -> None:
        self._load("ablation_no_gate.json")

    # ── No-KL specific ────────────────────────────────────────────────

    def test_no_kl_beta_is_zero(self) -> None:
        cfg = self._load("ablation_no_kl.json")
        self.assertEqual(cfg["mappo_config"]["beta_kl"], 0.0)

    def test_no_kl_has_ablation_id(self) -> None:
        cfg = self._load("ablation_no_kl.json")
        self.assertIn("no_kl", cfg["_ablation_id"])

    def test_no_kl_has_gate_policy(self) -> None:
        cfg = self._load("ablation_no_kl.json")
        self.assertIn("gate_policy", cfg)

    def test_no_kl_gate_policy_mentions_active(self) -> None:
        cfg = self._load("ablation_no_kl.json")
        self.assertIn("ACTIVE", cfg["gate_policy"])

    def test_no_kl_has_capability_retention_section(self) -> None:
        cfg = self._load("ablation_no_kl.json")
        self.assertIn("capability_retention", cfg)

    def test_no_kl_retention_threshold_below_one(self) -> None:
        cfg = self._load("ablation_no_kl.json")
        self.assertLess(cfg["capability_retention"]["threshold"], 1.0)

    # ── High-KL specific ─────────────────────────────────────────────

    def test_high_kl_beta_is_one(self) -> None:
        cfg = self._load("ablation_high_kl.json")
        self.assertGreaterEqual(cfg["mappo_config"]["beta_kl"], 1.0)

    def test_high_kl_has_gate_policy(self) -> None:
        cfg = self._load("ablation_high_kl.json")
        self.assertIn("gate_policy", cfg)

    def test_high_kl_retention_threshold_is_one(self) -> None:
        cfg = self._load("ablation_high_kl.json")
        self.assertAlmostEqual(cfg["capability_retention"]["threshold"], 1.0, places=5)

    # ── No-Gate specific ─────────────────────────────────────────────

    def test_no_gate_documents_why_infeasible(self) -> None:
        cfg = self._load("ablation_no_gate.json")
        self.assertIn("_WHY_THE_GATE_CANNOT_BE_DISABLED", cfg)

    def test_no_gate_why_is_non_empty_list(self) -> None:
        cfg = self._load("ablation_no_gate.json")
        self.assertIsInstance(cfg["_WHY_THE_GATE_CANNOT_BE_DISABLED"], list)
        self.assertGreater(len(cfg["_WHY_THE_GATE_CANNOT_BE_DISABLED"]), 0)

    def test_no_gate_has_feasible_alternative(self) -> None:
        cfg = self._load("ablation_no_gate.json")
        self.assertIn("_FEASIBLE_ALTERNATIVE", cfg)
        self.assertTrue(cfg["_FEASIBLE_ALTERNATIVE"])

    def test_no_gate_gate_policy_mentions_always(self) -> None:
        cfg = self._load("ablation_no_gate.json")
        self.assertIn("ALWAYS", cfg.get("gate_policy", ""))

    def test_no_gate_review3_table_entry_present(self) -> None:
        cfg = self._load("ablation_no_gate.json")
        self.assertIn("_review3_table_entry", cfg)

    def test_no_gate_table_entry_says_not_run(self) -> None:
        cfg = self._load("ablation_no_gate.json")
        entry = cfg["_review3_table_entry"]
        self.assertIn("NOT_RUN", entry["status"])


# ════════════════════════════════════════════════════════════════════════
# MAPPOConfig compatibility
# ════════════════════════════════════════════════════════════════════════

class TestAblationMAPPOConfig(unittest.TestCase):

    def _mappo_cfg_from_file(self, filename: str) -> MAPPOConfig:
        path = CONFIGS_DIR / filename
        raw = json.loads(path.read_text(encoding="utf-8"))
        mc = raw["mappo_config"]
        return MAPPOConfig(
            clip_epsilon=mc["clip_epsilon"],
            beta_kl=mc["beta_kl"],
            value_loss_coeff=mc["value_loss_coeff"],
            entropy_coeff=mc["entropy_coeff"],
            gamma=mc["gamma"],
            gae_lambda=mc["gae_lambda"],
            max_grad_norm=mc["max_grad_norm"],
        )

    def test_no_kl_loads_into_mappo_config(self) -> None:
        cfg = self._mappo_cfg_from_file("ablation_no_kl.json")
        self.assertEqual(cfg.beta_kl, 0.0)

    def test_high_kl_loads_into_mappo_config(self) -> None:
        cfg = self._mappo_cfg_from_file("ablation_high_kl.json")
        self.assertGreaterEqual(cfg.beta_kl, 1.0)

    def test_no_kl_and_high_kl_differ_on_beta(self) -> None:
        low  = self._mappo_cfg_from_file("ablation_no_kl.json")
        high = self._mappo_cfg_from_file("ablation_high_kl.json")
        self.assertLess(low.beta_kl, high.beta_kl)

    def test_ablation_betas_differ_from_default(self) -> None:
        default_beta = MAPPOConfig().beta_kl   # 0.1
        no_kl  = self._mappo_cfg_from_file("ablation_no_kl.json")
        high_kl = self._mappo_cfg_from_file("ablation_high_kl.json")
        self.assertNotEqual(no_kl.beta_kl,  default_beta)
        self.assertNotEqual(high_kl.beta_kl, default_beta)


# ════════════════════════════════════════════════════════════════════════
# Capability-retention analysis tests
# ════════════════════════════════════════════════════════════════════════

class TestCapabilityRetentionAnalysis(unittest.TestCase):

    def setUp(self) -> None:
        self.report = run_capability_probe(threshold=1.0)

    def test_no_regression_at_default_threshold(self) -> None:
        self.assertFalse(self.report.regression_flagged)

    def test_regression_flagged_above_threshold(self) -> None:
        report = run_capability_probe(threshold=1.01)
        self.assertTrue(report.regression_flagged)

    def test_per_category_rates_non_empty(self) -> None:
        self.assertGreater(len(self.report.per_category_pass_rate), 0)

    def test_all_fixture_categories_present(self) -> None:
        expected_cats = {p.probe_category for p in FIXTURE_PROBES}
        reported_cats = set(self.report.per_category_pass_rate.keys())
        self.assertEqual(expected_cats, reported_cats)

    def test_scope_field_present(self) -> None:
        self.assertTrue(self.report.scope)

    def test_results_length_matches_fixture_probes(self) -> None:
        self.assertEqual(len(self.report.results), len(FIXTURE_PROBES))

    def test_all_probes_pass(self) -> None:
        failed = [r.probe_id for r in self.report.results if not r.passed]
        self.assertEqual(failed, [], msg=f"Failed probes: {failed}")

    def test_custom_threshold_stored(self) -> None:
        report = run_capability_probe(threshold=0.75)
        self.assertAlmostEqual(report.threshold, 0.75, places=10)


# ════════════════════════════════════════════════════════════════════════
# _build_report JSON structure tests
# ════════════════════════════════════════════════════════════════════════

class TestBuildReport(unittest.TestCase):
    """Import the helper from the script and test its output."""

    @classmethod
    def setUpClass(cls) -> None:
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "run_cap",
            ROOT / "scripts" / "run_capability_retention_analysis.py",
        )
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)
        cls.report = run_capability_probe(threshold=1.0)
        cls.report_dict = cls.module._build_report(cls.report, "fixture_baseline")

    def test_required_top_level_keys(self) -> None:
        required = {
            "checkpoint_id", "generated_at", "total_probes", "passed", "failed",
            "retention_score", "threshold", "regression_flagged",
            "per_category_pass_rate", "per_probe", "scope", "disclaimer",
        }
        self.assertTrue(required.issubset(self.report_dict.keys()),
                        msg=f"Missing: {required - self.report_dict.keys()}")

    def test_per_probe_length_matches_fixtures(self) -> None:
        self.assertEqual(len(self.report_dict["per_probe"]), len(FIXTURE_PROBES))

    def test_per_probe_items_have_required_fields(self) -> None:
        required = {"probe_id", "probe_category", "expected_status",
                    "actual_status", "passed", "note"}
        for item in self.report_dict["per_probe"]:
            self.assertTrue(required.issubset(item.keys()),
                            msg=f"Missing keys in {item}")

    def test_checkpoint_id_stored_in_report(self) -> None:
        self.assertEqual(self.report_dict["checkpoint_id"], "fixture_baseline")

    def test_disclaimer_present(self) -> None:
        self.assertTrue(self.report_dict["disclaimer"])

    def test_retention_score_in_unit_interval(self) -> None:
        score = self.report_dict["retention_score"]
        self.assertGreaterEqual(score, 0.0)
        self.assertLessEqual(score, 1.0)


# ════════════════════════════════════════════════════════════════════════
# Script smoke test (subprocess, --dry-run)
# ════════════════════════════════════════════════════════════════════════

class TestRetentionScriptSmoke(unittest.TestCase):

    def test_script_exits_zero_dry_run(self) -> None:
        script = ROOT / "scripts" / "run_capability_retention_analysis.py"
        result = subprocess.run(
            [sys.executable, str(script), "--dry-run"],
            capture_output=True, text=True,
        )
        self.assertEqual(
            result.returncode, 0,
            msg=f"Script failed:\n{result.stdout}\n{result.stderr}",
        )

    def test_script_dry_run_prints_no_regression(self) -> None:
        script = ROOT / "scripts" / "run_capability_retention_analysis.py"
        result = subprocess.run(
            [sys.executable, str(script), "--dry-run", "--threshold", "1.0"],
            capture_output=True, text=True,
        )
        self.assertNotIn("REGRESSION", result.stdout)

    def test_script_dry_run_shows_retention_score(self) -> None:
        script = ROOT / "scripts" / "run_capability_retention_analysis.py"
        result = subprocess.run(
            [sys.executable, str(script), "--dry-run"],
            capture_output=True, text=True,
        )
        self.assertIn("Score", result.stdout)


if __name__ == "__main__":
    unittest.main()
