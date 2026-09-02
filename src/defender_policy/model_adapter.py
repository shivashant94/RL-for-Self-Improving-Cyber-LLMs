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
import json
import re
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


# ── Real SFT-trained model adapter ────────────────────────────────────────────

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


class SFTModelAdapter(BaseModelAdapter):
    """Adapter for a real LoRA-tuned Defender checkpoint, produced by
    ``src/train_sft.py`` against ``data/sft/train_formatted.jsonl``.

    Loading (torch/transformers/peft) is deferred to first use, so importing
    this module never requires those packages — only actually generating
    from a checkpoint does.  Without a ``checkpoint_path``, ``act()`` still
    raises ``NotImplementedError`` rather than silently falling back to a
    fixture: there is no model to be honest about yet.

    Prompt and output format mirror training exactly:
      - prompt:  "User: {user_task}\\n[untrusted content]\\n{untrusted_content}\\nAssistant:"
        (matches scripts/format_sft_for_training.py)
      - a tool proposal is emitted as ``<tool_call>{"tool":...,"arguments":...,
        "purpose":...}</tool_call>`` (matches the SFT corpus's benign_tool_use
        targets); anything else is treated as a direct text answer.
    """

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        base_model: str = "Qwen/Qwen2.5-0.5B",
        max_new_tokens: int = 200,
        device: Optional[str] = None,
    ) -> None:
        self.checkpoint_path = checkpoint_path
        self.base_model = base_model
        self.max_new_tokens = max_new_tokens
        self._device_override = device
        self._model = None
        self._tokenizer = None
        self._torch = None

    def _load(self) -> None:
        if self._model is not None:
            return
        if not self.checkpoint_path:
            raise NotImplementedError(
                "SFTModelAdapter has no checkpoint_path.  Pass "
                "checkpoint_path=<LoRA adapter dir produced by train_sft.py> "
                "once a real Defender checkpoint exists.  Do not fabricate results."
            )
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
            from peft import PeftModel
        except ImportError as exc:
            raise ImportError(
                "SFTModelAdapter needs torch, transformers, and peft installed "
                "(see requirements.txt) to load a real checkpoint."
            ) from exc

        is_mac = not torch.cuda.is_available() and torch.backends.mps.is_available()
        device = self._device_override or ("cpu" if is_mac else "auto")

        tokenizer = AutoTokenizer.from_pretrained(self.base_model)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        base = AutoModelForCausalLM.from_pretrained(
            self.base_model, device_map=device, torch_dtype=torch.float32,
        )
        model = PeftModel.from_pretrained(base, self.checkpoint_path)
        model.eval()

        self._torch = torch
        self._tokenizer = tokenizer
        self._model = model

    @staticmethod
    def _build_prompt(observation: DefenderObservation) -> str:
        lines = [observation.user_task]
        if observation.untrusted_content:
            lines.append(f"[untrusted content]\n{observation.untrusted_content}")
        user_turn = "\n".join(lines)
        return f"User: {user_turn}\nAssistant:"

    @staticmethod
    def _parse_completion(
        completion: str, log_prob: float, entropy: Optional[float]
    ) -> RawModelOutput:
        match = _TOOL_CALL_RE.search(completion)
        if match:
            try:
                payload = json.loads(match.group(1))
            except json.JSONDecodeError:
                # Malformed tool-call syntax is not our call to fix — pass the raw
                # text through and let the policy gate's GATE-003 reject it.
                return RawModelOutput(text=completion, log_prob=log_prob, entropy=entropy)
            return RawModelOutput(
                text="",
                tool_name=payload.get("tool"),
                tool_args=payload.get("arguments") or {},
                tool_purpose=payload.get("purpose", ""),
                log_prob=log_prob,
                entropy=entropy,
            )
        return RawModelOutput(text=completion, log_prob=log_prob, entropy=entropy)

    def _generate(self, observation: DefenderObservation) -> RawModelOutput:
        self._load()
        torch = self._torch
        prompt = self._build_prompt(observation)
        inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)

        with torch.no_grad():
            output = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=self._tokenizer.pad_token_id,
            )

        prompt_len = inputs["input_ids"].shape[1]
        generated_ids = output.sequences[0][prompt_len:]
        completion = self._tokenizer.decode(generated_ids, skip_special_tokens=True).strip()

        log_prob = 0.0
        entropy_terms = []
        for step_logits, token_id in zip(output.scores, generated_ids):
            log_probs = torch.log_softmax(step_logits[0], dim=-1)
            log_prob += log_probs[token_id].item()
            probs = log_probs.exp()
            entropy_terms.append(-(probs * log_probs).sum().item())
        entropy = sum(entropy_terms) / len(entropy_terms) if entropy_terms else None

        return self._parse_completion(completion, log_prob, entropy)
