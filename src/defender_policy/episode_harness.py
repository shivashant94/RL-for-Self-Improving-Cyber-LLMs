"""Fixture episode harness for Review 2 self-play smoke testing.

Responsibilities
----------------
1. Sample one (user_task, untrusted_content, case_metadata) triple from
   fixture data (review1_cases.jsonl) per episode.
2. Run a single-turn episode through the Defender pipeline:
       FixtureModelAdapter.act() → RolloutAdapter.step() → apply_reward_to_step()
3. Run the capability-retention probe at episode end (benign-task check).
4. Return an ``EpisodeResult`` containing the full ``List[TrajectoryStep]``,
   all reward components, the gate decision, and the retention probe report.

Design rules
------------
- No live network, shell, or credentials.  All data is fixture-only.
- The gate is NEVER bypassed.  Every tool proposal goes through PolicyGate.
- The harness does NOT compute advantages or value estimates — those are
  Member 3's responsibility.  ``TrajectoryStep.advantage`` and
  ``.value_estimate`` are left None.
- Episode length is 1 turn (single observation → action) in v1 because the
  fixture policy is stateless.  Multi-turn is a Review 3 extension.
- All results are labelled "fixture episode; not a trained-LLM result".
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .capability_retention import CapabilityRetentionReport, run_capability_probe
from .model_adapter import DefenderObservation, FixtureModelAdapter
from .rewards import RewardWeights, apply_reward_to_step
from .rollout_adapter import RolloutAdapter, TrajectoryStep

PROJECT_ROOT = Path(__file__).resolve().parents[2]  # src/defender_policy/ → src/ → repo root


# ── Case loading ──────────────────────────────────────────────────────────────

def _load_cases(cases_path: Path) -> List[dict]:
    """Load all non-empty lines from a JSONL evaluation case file."""
    cases: List[dict] = []
    with cases_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                cases.append(json.loads(line))
    if not cases:
        raise ValueError(f"No cases found in {cases_path}")
    return cases


def _default_cases_path() -> Path:
    return PROJECT_ROOT / "data" / "evaluation" / "review1_cases.jsonl"


# ── Episode result ────────────────────────────────────────────────────────────

@dataclass
class EpisodeResult:
    """All artifacts produced by one fixture episode run.

    Fields
    ------
    episode_id          – unique identifier for this episode.
    case_id             – the evaluation case used as the episode scenario.
    case_kind           – "attack" or "benign".
    attack_family       – attack family or None.
    steps               – list of TrajectorySteps (one per turn; v1: always 1).
    total_reward        – sum of per-step rewards.
    reward_components   – averaged reward components across steps.
    gate_blocked        – True if any step's gate blocked a tool call.
    pre_gate_unsafe     – True if any step had a pre-gate unsafe proposal.
    retention_report    – capability-retention probe run after episode end.
    scope               – always "fixture episode; not a trained-LLM result".
    """
    episode_id: str
    case_id: str
    case_kind: str
    attack_family: Optional[str]
    steps: List[TrajectoryStep]
    total_reward: float
    reward_components: Dict[str, float]
    gate_blocked: bool
    pre_gate_unsafe: bool
    retention_report: CapabilityRetentionReport
    scope: str = "fixture episode; not a trained-LLM result"


# ── Episode harness ───────────────────────────────────────────────────────────

class FixtureEpisodeHarness:
    """Single-turn episode harness over fixture evaluation cases.

    Parameters
    ----------
    cases_path : Path, optional
        Path to the evaluation JSONL file.  Defaults to
        ``data/evaluation/review1_cases.jsonl``.
    reward_weights : RewardWeights, optional
        Penalty weights for the reward function.  Uses defaults if None.
    seed : int, optional
        Random seed for reproducible case sampling.
    """

    ACTION_SCHEMA_VERSION = "gate-v1"

    def __init__(
        self,
        cases_path: Optional[Path] = None,
        reward_weights: Optional[RewardWeights] = None,
        seed: int = 42,
    ) -> None:
        self._cases = _load_cases(cases_path or _default_cases_path())
        self._weights = reward_weights or RewardWeights()
        self._rng = random.Random(seed)
        self._adapter = FixtureModelAdapter()
        self._rollout = RolloutAdapter()

    # ── Public API ────────────────────────────────────────────────────────

    def run_episode(self, episode_id: str = "episode_000") -> EpisodeResult:
        """Sample one fixture case, run one turn, return EpisodeResult."""
        case = self._rng.choice(self._cases)
        return self._run_case(case, episode_id)

    def run_episode_for_case(
        self,
        case_id: str,
        episode_id: str = "episode_000",
    ) -> EpisodeResult:
        """Run an episode for a specific case_id (deterministic for tests)."""
        matching = [c for c in self._cases if c["id"] == case_id]
        if not matching:
            raise ValueError(f"Case '{case_id}' not found in fixture cases.")
        return self._run_case(matching[0], episode_id)

    def run_n_episodes(self, n: int, base_id: str = "ep") -> List[EpisodeResult]:
        """Run ``n`` randomly sampled episodes and return all results."""
        if n < 1:
            raise ValueError(f"n must be >= 1, got {n}")
        return [self.run_episode(episode_id=f"{base_id}_{i:04d}") for i in range(n)]

    # ── Internal ──────────────────────────────────────────────────────────

    def _run_case(self, case: dict, episode_id: str) -> EpisodeResult:
        """Execute one turn for the given case dict and return EpisodeResult."""
        task_id   = case.get("id", "unknown")
        user_task = case.get("user_task", "")
        untrusted = case.get("untrusted_content", "")
        case_kind = case.get("kind", "unknown")
        attack_family = case.get("attack_family", None)
        expected_status = case.get("expected", None)

        # Map evaluation 'expected' to reward-compatible status
        expected_reward_status = _eval_expected_to_reward_status(expected_status, case_kind)
        is_benign = case_kind == "benign"

        # Build observation
        obs = DefenderObservation(
            task_id=task_id,
            user_task=user_task,
            untrusted_content=untrusted,
            safe_context=(),
            allowed_action_schema=self.ACTION_SCHEMA_VERSION,
        )

        # Model → rollout → reward
        raw = self._adapter.act(obs)
        step = self._rollout.step(obs, raw, episode_id=episode_id, step_index=0)
        apply_reward_to_step(
            step,
            weights=self._weights,
            expected_status=expected_reward_status,
            is_benign_task=is_benign,
        )

        steps = [step]

        # Aggregate
        total_reward = sum(s.reward or 0.0 for s in steps)
        avg_components = _average_components([s.reward_components for s in steps])
        gate_blocked   = any(
            s.action.gate_result is not None and not s.action.gate_result.allowed
            for s in steps
        )
        pre_gate_unsafe = any(s.action.pre_gate_unsafe for s in steps)

        # Capability retention probe (run once at episode end)
        retention = run_capability_probe()

        return EpisodeResult(
            episode_id=episode_id,
            case_id=task_id,
            case_kind=case_kind,
            attack_family=attack_family,
            steps=steps,
            total_reward=total_reward,
            reward_components=avg_components,
            gate_blocked=gate_blocked,
            pre_gate_unsafe=pre_gate_unsafe,
            retention_report=retention,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _eval_expected_to_reward_status(expected: Optional[str], case_kind: str) -> Optional[str]:
    """Translate evaluation case 'expected' field to reward expected_status."""
    mapping = {
        "resist": "refusal",   # attack successfully resisted → Defender refused
        "answer": "answer",
        "tool_used": "tool_used",
    }
    return mapping.get(expected or "", None)


def _average_components(components_list: List[Dict[str, float]]) -> Dict[str, float]:
    """Average reward component dicts across steps."""
    if not components_list:
        return {}
    keys = components_list[0].keys()
    result: Dict[str, float] = {}
    for k in keys:
        values = [c[k] for c in components_list if k in c]
        result[k] = sum(values) / len(values) if values else 0.0
    return result
