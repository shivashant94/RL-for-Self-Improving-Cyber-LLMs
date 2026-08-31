"""Rollout adapter: transforms raw model output into a validated DefenderAction.

Responsibilities
----------------
1. Accept a ``RawModelOutput`` from the model adapter.
2. If the output contains a tool proposal, route it through the unchanged
   Review 1 policy gate and record the gate decision.
3. Package the result as a ``DefenderAction`` with stored logprob, entropy,
   gate outcome, and action mask version.
4. Record a ``TrajectoryStep`` for use by the environment (Member 3).

Invariants
----------
- Every proposed tool call goes through the gate.  The gate is NEVER bypassed.
- If the gate blocks a call, the Defender receives a safe denial and must answer
  without the tool result.
- ``DefenderAction`` contains NO hidden critic state, attacker labels, or global
  state.  Those belong to ``GlobalState`` (Member 3 only).
- Pre-gate unsafe proposals and post-gate violations are tracked separately so
  they appear as distinct metrics.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Optional

from .gate import GateResult, PolicyGate, ToolCall
from .model_adapter import DefenderObservation, RawModelOutput


# ── DefenderAction ────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DefenderAction:
    """Validated output of the Defender for one environment step.

    Fields
    ------
    task_id          – echoes observation.task_id for trajectory alignment.
    answer_text      – text answer (non-empty when tool_call is None).
    tool_call        – the proposed ToolCall (None if Defender answered directly).
    gate_result      – the gate's decision (None if no tool was proposed).
    log_prob         – log-probability of the action tokens (from model adapter).
    entropy          – entropy of the output distribution (None if unavailable).
    action_mask_version – version of the gate's action schema at step time.
    pre_gate_unsafe  – True if the raw model output contained an unsafe tool
                       proposal that the gate blocked (pre-gate violation metric).
    """
    task_id: str
    answer_text: str
    tool_call: Optional[ToolCall]
    gate_result: Optional[GateResult]
    log_prob: float
    entropy: Optional[float]
    action_mask_version: str
    pre_gate_unsafe: bool = False


# ── TrajectoryStep ────────────────────────────────────────────────────────────

@dataclass
class TrajectoryStep:
    """One step in the Defender's episode trajectory.

    Produced by the rollout adapter; consumed by the environment (Member 3)
    to build ``Trajectory`` objects for the MAPPO update.

    Fields marked ``# critic_only`` must NOT be populated by Member 2.
    They are filled in by Member 3's environment after the fact.
    """
    episode_id: str
    step_index: int
    observation: DefenderObservation
    action: DefenderAction
    reward: Optional[float] = None          # filled by rewards.py
    reward_components: dict[str, float] = field(default_factory=dict)
    done: bool = False
    # Fields below are reserved for Member 3 / centralized critic
    advantage: Optional[float] = None       # critic_only: filled by Member 3
    value_estimate: Optional[float] = None  # critic_only: filled by Member 3


# ── Unsafe-proposal detection (pre-gate heuristic) ───────────────────────────

_UNSAFE_TOOL_NAMES = frozenset({
    "export_data", "send_email", "delete_record", "write_file",
    "execute_command", "browse_url", "shell", "send_sms",
})

_TRAVERSAL_PATTERN = re.compile(r"\.\./|~[/\\]|^/")


def _is_pre_gate_unsafe(raw: RawModelOutput) -> bool:
    """Heuristic: did the raw model output look unsafe BEFORE the gate ran?

    Used for the ``pre_gate_unsafe_proposal_rate`` metric.  The gate is the
    authoritative enforcer; this flag is for diagnostic metric separation only.
    """
    if raw.tool_name is None:
        return False
    if raw.tool_name in _UNSAFE_TOOL_NAMES:
        return True
    # Path traversal in any argument value
    for v in raw.tool_args.values():
        if isinstance(v, str) and _TRAVERSAL_PATTERN.search(v):
            return True
    return False


# ── Rollout adapter ───────────────────────────────────────────────────────────

class RolloutAdapter:
    """Converts a ``RawModelOutput`` into a ``DefenderAction`` via the gate.

    Parameters
    ----------
    gate : PolicyGate
        The unchanged Review 1 gate.  Never bypassed.
    action_mask_version : str
        Version tag for the current gate allowlist, stored on every action for
        reproducibility.  Increment this when the gate config changes.
    """

    ACTION_MASK_VERSION: str = "gate-v1"

    def __init__(
        self,
        gate: Optional[PolicyGate] = None,
        action_mask_version: str = ACTION_MASK_VERSION,
    ) -> None:
        self.gate = gate or PolicyGate()
        self.action_mask_version = action_mask_version

    # ── Public API ────────────────────────────────────────────────────────

    def step(
        self,
        observation: DefenderObservation,
        raw: RawModelOutput,
        episode_id: str = "episode_000",
        step_index: int = 0,
    ) -> TrajectoryStep:
        """Validate ``raw`` through the gate and return a ``TrajectoryStep``.

        Parameters
        ----------
        observation : DefenderObservation
            The observation the model acted on.
        raw : RawModelOutput
            The model's unvalidated proposal.
        episode_id : str
            Identifier for the current self-play episode.
        step_index : int
            Position of this step within the episode.
        """
        action = self._to_defender_action(observation.task_id, raw)
        return TrajectoryStep(
            episode_id=episode_id,
            step_index=step_index,
            observation=observation,
            action=action,
        )

    # ── Internal ──────────────────────────────────────────────────────────

    def _to_defender_action(self, task_id: str, raw: RawModelOutput) -> DefenderAction:
        """Route the raw output through the gate and build a DefenderAction."""
        pre_gate_unsafe = _is_pre_gate_unsafe(raw)

        if raw.tool_name is None or raw.tool_name == "_fixture_tool":
            # Pure text answer — no gate execution needed.
            return DefenderAction(
                task_id=task_id,
                answer_text=raw.text,
                tool_call=None,
                gate_result=None,
                log_prob=raw.log_prob,
                entropy=raw.entropy,
                action_mask_version=self.action_mask_version,
                pre_gate_unsafe=False,
            )

        # Build ToolCall and run through gate.
        tool_call = ToolCall(
            tool=raw.tool_name,
            arguments=raw.tool_args,
            purpose=raw.tool_purpose or f"Model-proposed call to {raw.tool_name}",
        )
        gate_result = self.gate.execute(tool_call)

        if gate_result.allowed:
            answer_text = ""
        else:
            # Gate blocked — Defender answers with a safe denial.
            answer_text = gate_result.message

        return DefenderAction(
            task_id=task_id,
            answer_text=answer_text,
            tool_call=tool_call,
            gate_result=gate_result,
            log_prob=raw.log_prob,
            entropy=raw.entropy,
            action_mask_version=self.action_mask_version,
            pre_gate_unsafe=pre_gate_unsafe,
        )
