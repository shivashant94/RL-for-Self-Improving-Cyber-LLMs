"""Defender model adapter: abstract interface between the policy loop and the model.

Design
------
Both the deterministic fixture baseline and any real SFT-trained model must expose
the same ``act()`` contract so the rest of the pipeline (rollout, MAPPO update,
evaluation) is model-agnostic.

The adapter never authorises or executes a tool call.  It only proposes an action.
Every proposed ToolCall must still pass through the Review 1 policy gate before
any execution can happen.

Interfaces (from three-review-architecture.md)
----------------------------------------------
Defender observation (input to act()):
    task_id          – unique case identifier
    user_task        – legitimate user-facing instruction
    untrusted_content – content from an untrusted source (may be empty)
    safe_context     – prior approved, redacted tool results (may be empty list)
    allowed_action_schema – version tag of the gate's current action allowlist

Defender action (returned by act()):
    Defined in rollout_adapter.py as ``DefenderAction``.

Fixture mock
------------
``FixtureModelAdapter`` wraps ``ReviewOneBaselineDefender`` so the same integration
tests pass without a real LLM.
"""

import abc
from dataclasses import dataclass, field
from typing import Any, Optional


# ── Observation ───────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class DefenderObservation:
    """Decentralised observation visible to the Defender actor only.

    Invariant: this object must NEVER contain hidden attacker intent, protected
    labels, or critic-only global state.  Those belong in ``GlobalState`` (Member 3).
    """
    task_id: str
    user_task: str
    untrusted_content: str
    safe_context: tuple[str, ...]   # prior approved, redacted tool results
    allowed_action_schema: str      # e.g. "gate-v1" — version of the enforcement policy


# ── Raw model output (pre-gate) ───────────────────────────────────────────────

@dataclass(frozen=True)
class RawModelOutput:
    """Unvalidated output from the model.  Must not be executed directly.

    Fields
    ------
    text        – free-text answer if the model chose to answer directly.
    tool_name   – proposed tool name if the model chose a tool call (may be None).
    tool_args   – proposed arguments (may be empty dict).
    tool_purpose – proposed purpose string (may be empty).
    log_prob    – sum of token log-probabilities for this output (-inf if unavailable).
    entropy     – approximate entropy of the output distribution (None if unavailable).
    """
    text: str
    tool_name: Optional[str] = None
    tool_args: dict[str, Any] = field(default_factory=dict)
    tool_purpose: str = ""
    log_prob: float = 0.0
    entropy: Optional[float] = None


# ── Abstract adapter ──────────────────────────────────────────────────────────

class BaseModelAdapter(abc.ABC):
    """Abstract base class for Defender model adapters.

    Subclasses implement ``_generate`` to wrap a specific model backend.
    The public ``act`` method validates inputs and enforces the interface contract.
    """

    # The action schema version this adapter was built against.
    ACTION_SCHEMA_VERSION: str = "gate-v1"

    def act(self, observation: DefenderObservation) -> RawModelOutput:
        """Produce a raw model output for ``observation``.

        This method validates the observation schema and delegates to
        ``_generate``.  The returned ``RawModelOutput`` is a PROPOSAL only;
        any tool call in it must still pass through the policy gate.

        Raises
        ------
        ValueError
            If the observation is structurally invalid or uses an unsupported
            action schema version.
        """
        if not observation.task_id or not observation.task_id.strip():
            raise ValueError("DefenderObservation.task_id must be a non-empty string.")
        if not isinstance(observation.user_task, str):
            raise ValueError("DefenderObservation.user_task must be a string.")
        if not isinstance(observation.untrusted_content, str):
            raise ValueError("DefenderObservation.untrusted_content must be a string.")
        if observation.allowed_action_schema != self.ACTION_SCHEMA_VERSION:
            raise ValueError(
                f"Observation uses action schema {observation.allowed_action_schema!r} "
                f"but this adapter only supports {self.ACTION_SCHEMA_VERSION!r}."
            )
        return self._generate(observation)

    @abc.abstractmethod
    def _generate(self, observation: DefenderObservation) -> RawModelOutput:
        """Backend-specific generation.  Called only after input validation."""


# ── Fixture adapter (wraps ReviewOneBaselineDefender) ────────────────────────

class FixtureModelAdapter(BaseModelAdapter):
    """Deterministic fixture adapter that wraps ``ReviewOneBaselineDefender``.

    Used for integration tests and smoke tests when no real LLM is available.
    Produces deterministic outputs with synthetic log_prob=-1.0 and entropy=0.0
    to allow downstream components to exercise the full pipeline path.

    This adapter is NOT a trained model and must never be presented as one.
    """

    def __init__(self) -> None:
        from .baseline import ReviewOneBaselineDefender
        self._defender = ReviewOneBaselineDefender()

    def _generate(self, observation: DefenderObservation) -> RawModelOutput:
        outcome = self._defender.respond(
            observation.user_task,
            observation.untrusted_content,
        )

        if outcome.status == "tool_used" and outcome.tool_result is not None:
            # The fixture defender already ran the tool internally; we re-expose
            # the tool proposal shape so the rollout adapter can re-route it
            # through the gate.  We reconstruct a plausible proposal from the
            # outcome rather than recording actual arguments (which are safe in
            # this fixture context but opaque to the caller).
            return RawModelOutput(
                text="",
                tool_name="_fixture_tool",   # sentinel — rollout adapter handles this
                tool_args={},
                tool_purpose="Fixture tool execution (deterministic).",
                log_prob=-1.0,
                entropy=0.0,
            )

        # For answer / safe_answer / refusal, return the text.
        return RawModelOutput(
            text=outcome.response,
            tool_name=None,
            tool_args={},
            tool_purpose="",
            log_prob=-1.0,
            entropy=0.0,
        )


# ── Stub for future SFT/MAPPO model ──────────────────────────────────────────

class SFTModelAdapter(BaseModelAdapter):
    """Placeholder adapter for the real SFT-trained Defender model.

    Status: BLOCKED_EXTERNAL — requires team-confirmed base model, training
    stack, and GPU budget.  This class provides the correct interface so
    downstream code compiles and tests pass against the fixture adapter.

    To activate: replace the ``NotImplementedError`` body with the real
    model-loading and generation code.  Do NOT change the ``act()`` contract.
    """

    def __init__(self, checkpoint_path: Optional[str] = None) -> None:
        self.checkpoint_path = checkpoint_path

    def _generate(self, observation: DefenderObservation) -> RawModelOutput:
        raise NotImplementedError(
            "SFTModelAdapter._generate is BLOCKED_EXTERNAL.  "
            "Load the team-approved checkpoint and implement generation here.  "
            "Do not fabricate results."
        )
