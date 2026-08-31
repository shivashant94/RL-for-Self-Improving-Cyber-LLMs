"""Defender MAPPO policy-update computations (pure-Python, framework-agnostic).

Design constraints
------------------
- All functions are pure (no side effects, no I/O).
- Input/output types are Python floats and lists — no PyTorch/NumPy dependency.
  When a real training loop integrates, it wraps these in tensor ops; the logic
  and unit tests remain unchanged.
- The KL penalty is applied HERE, not in rewards.py, so the two concerns are
  cleanly separated:
    * rewards.py  → environment signal  (task_utility, safety_violation, ...)
    * mappo_defender.py → PPO loss       (policy_loss + KL_penalty + value_loss)
- The gate is NOT touched during training.  Its allowlist is fixed.

Defender objective (from three-review-architecture.md §Defender objective)
---------------------------------------------------------------------------
  R_def = task_utility - λ_attack·safety_violation - λ_refusal·unnecessary_refusal

  PPO surrogate loss:
    L_CLIP(θ) = -E[ min(r·A, clip(r, 1-ε, 1+ε)·A) ]

  Total training loss:
    L_total = L_CLIP + β_KL·KL(π_θ || π_ref) + c_v·L_value

  where:
    r = exp(log π_θ(a|s) - log π_ref(a|s))   (probability ratio)
    A = GAE advantage (provided by Member 3's critic)
    π_ref = SFT warm-up policy (frozen reference)
    β_KL tunes the KL penalty strength; tuned on held-out benign tasks
    c_v is the value-loss coefficient
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional


# ── Hyperparameters ───────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MAPPOConfig:
    """Hyperparameters for the Defender PPO update.

    All values must be validated non-negative (or within required ranges).
    Stored on every experiment record so ablations are reproducible.
    """
    clip_epsilon: float = 0.2          # PPO clip range [1-ε, 1+ε]
    beta_kl: float = 0.1               # KL penalty coefficient
    value_loss_coeff: float = 0.5      # weight of the value-function loss
    entropy_coeff: float = 0.01        # optional entropy bonus (encourages exploration)
    gamma: float = 0.99                # discount factor
    gae_lambda: float = 0.95           # GAE λ for advantage estimation
    max_grad_norm: float = 0.5         # gradient clipping (applied by training loop)

    def __post_init__(self) -> None:
        checks = [
            ("clip_epsilon",     self.clip_epsilon,     0.0, 1.0, True),   # strict lo
            ("beta_kl",          self.beta_kl,          0.0, None, False),
            ("value_loss_coeff", self.value_loss_coeff, 0.0, None, False),
            ("entropy_coeff",    self.entropy_coeff,    0.0, None, False),
            ("gamma",            self.gamma,            0.0, 1.0, False),
            ("gae_lambda",       self.gae_lambda,       0.0, 1.0, False),
            ("max_grad_norm",    self.max_grad_norm,    0.0, None, False),
        ]
        for name, val, lo, hi, strict_lo in checks:
            if strict_lo and val <= lo:
                raise ValueError(f"MAPPOConfig.{name}={val} must be > {lo}")
            elif not strict_lo and val < lo:
                raise ValueError(f"MAPPOConfig.{name}={val} must be >= {lo}")
            if hi is not None and val > hi:
                raise ValueError(f"MAPPOConfig.{name}={val} must be <= {hi}")


# ── KL penalty ────────────────────────────────────────────────────────────────

def compute_kl_penalty(
    current_logprobs: List[float],
    ref_logprobs: List[float],
    beta_kl: float,
) -> float:
    """Token-level KL divergence penalty: β_KL · KL(π_θ || π_ref).

    KL(π_θ || π_ref) = Σ_t  π_θ(t) · log(π_θ(t) / π_ref(t))
                     ≈ Σ_t  exp(lp_θ(t)) · (lp_θ(t) - lp_ref(t))

    Parameters
    ----------
    current_logprobs : list of float
        Per-token log-probabilities from the current policy π_θ.
    ref_logprobs : list of float
        Per-token log-probabilities from the frozen SFT reference π_ref.
    beta_kl : float
        Penalty coefficient (>= 0).

    Returns
    -------
    float
        The scaled KL penalty scalar.  Zero when current == reference;
        positive when the current policy has diverged.

    Raises
    ------
    ValueError
        If the two sequences have different lengths or beta_kl is negative.
    """
    if beta_kl < 0:
        raise ValueError(f"beta_kl must be >= 0, got {beta_kl}")
    if len(current_logprobs) != len(ref_logprobs):
        raise ValueError(
            f"current_logprobs (len={len(current_logprobs)}) and "
            f"ref_logprobs (len={len(ref_logprobs)}) must have the same length."
        )
    if not current_logprobs:
        return 0.0

    kl_sum = 0.0
    for lp_cur, lp_ref in zip(current_logprobs, ref_logprobs):
        # Numerically stable: KL contribution per token
        kl_sum += math.exp(lp_cur) * (lp_cur - lp_ref)

    return beta_kl * kl_sum


# ── PPO clipped policy loss ───────────────────────────────────────────────────

def compute_policy_loss(
    advantages: List[float],
    log_ratios: List[float],
    clip_epsilon: float,
) -> float:
    """PPO clipped surrogate policy loss (negated, so gradient descent minimises it).

    L_CLIP = -mean[ min(r·A, clip(r, 1-ε, 1+ε)·A) ]

    where r = exp(log_ratio)  and  log_ratio = log π_θ(a|s) - log π_old(a|s).

    Parameters
    ----------
    advantages : list of float
        GAE advantage estimates (provided by Member 3's centralized critic).
        Must not be empty.
    log_ratios : list of float
        log π_θ(a|s) - log π_old(a|s) for each step.
        Must have the same length as advantages.
    clip_epsilon : float
        Clip range.  Typically 0.1–0.3.  Must be in (0, 1].

    Returns
    -------
    float
        The mean clipped policy loss (positive scalar; lower = better policy).

    Raises
    ------
    ValueError
        On empty inputs, mismatched lengths, or invalid clip_epsilon.
    """
    if clip_epsilon <= 0 or clip_epsilon > 1:
        raise ValueError(f"clip_epsilon must be in (0, 1], got {clip_epsilon}")
    if not advantages:
        raise ValueError("advantages must be non-empty.")
    if len(advantages) != len(log_ratios):
        raise ValueError(
            f"advantages (len={len(advantages)}) and log_ratios "
            f"(len={len(log_ratios)}) must have the same length."
        )

    lo = 1.0 - clip_epsilon
    hi = 1.0 + clip_epsilon
    total = 0.0

    for adv, log_r in zip(advantages, log_ratios):
        r = math.exp(log_r)
        unclipped = r * adv
        clipped   = max(lo, min(hi, r)) * adv
        total    += min(unclipped, clipped)

    return -(total / len(advantages))  # negated: minimise loss → maximise reward


# ── Value function loss ───────────────────────────────────────────────────────

def compute_value_loss(
    value_estimates: List[float],
    returns: List[float],
) -> float:
    """Mean-squared-error value function loss.

    L_value = mean[ (V(s) - R)² ]

    Parameters
    ----------
    value_estimates : list of float
        Critic value predictions V(s) for each step.
    returns : list of float
        Monte-Carlo or GAE returns R for each step.
    """
    if not value_estimates:
        raise ValueError("value_estimates must be non-empty.")
    if len(value_estimates) != len(returns):
        raise ValueError(
            f"value_estimates (len={len(value_estimates)}) and "
            f"returns (len={len(returns)}) must have the same length."
        )
    total = sum((v - r) ** 2 for v, r in zip(value_estimates, returns))
    return total / len(value_estimates)


# ── Entropy bonus ─────────────────────────────────────────────────────────────

def compute_entropy_bonus(entropies: List[float], entropy_coeff: float) -> float:
    """Entropy bonus to encourage exploration.

    Returns -entropy_coeff * mean(entropies).  Subtracted from total loss so
    higher entropy → lower loss → gradient encourages diversity.
    """
    if entropy_coeff < 0:
        raise ValueError(f"entropy_coeff must be >= 0, got {entropy_coeff}")
    if not entropies:
        return 0.0
    return -entropy_coeff * (sum(entropies) / len(entropies))


# ── Total MAPPO loss ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class MAPPOLossOutput:
    """Full loss breakdown for one Defender PPO update step."""
    total_loss: float
    policy_loss: float
    kl_penalty: float
    value_loss: float
    entropy_bonus: float
    config: MAPPOConfig


def compute_total_loss(
    policy_loss: float,
    kl_penalty: float,
    value_loss: float,
    entropy_bonus: float,
    config: MAPPOConfig,
) -> MAPPOLossOutput:
    """Combine all loss components into the final Defender training loss.

    L_total = L_CLIP + β_KL·KL + c_v·L_value + entropy_bonus

    Note: entropy_bonus is already negated by compute_entropy_bonus() so it
    reduces the total loss (encouraging exploration).
    """
    total = (
        policy_loss
        + kl_penalty
        + config.value_loss_coeff * value_loss
        + entropy_bonus
    )
    return MAPPOLossOutput(
        total_loss=total,
        policy_loss=policy_loss,
        kl_penalty=kl_penalty,
        value_loss=value_loss,
        entropy_bonus=entropy_bonus,
        config=config,
    )


# ── Convenience: full update step ────────────────────────────────────────────

def defender_update_step(
    advantages: List[float],
    log_ratios: List[float],
    current_logprobs: List[float],
    ref_logprobs: List[float],
    value_estimates: List[float],
    returns: List[float],
    entropies: Optional[List[float]] = None,
    config: Optional[MAPPOConfig] = None,
) -> MAPPOLossOutput:
    """Compute the full Defender MAPPO loss for one minibatch.

    Parameters
    ----------
    advantages      – GAE advantages from Member 3's critic.
    log_ratios      – log π_θ/π_old per step (current vs. old policy).
    current_logprobs – per-token logprobs from π_θ (for KL vs SFT reference).
    ref_logprobs    – per-token logprobs from frozen π_SFT (for KL penalty).
    value_estimates – V(s) from the critic.
    returns         – target returns (GAE or Monte-Carlo).
    entropies       – per-step approximate entropy (optional; 0 if None).
    config          – MAPPO hyperparameters (defaults if None).
    """
    cfg = config or MAPPOConfig()
    entropies = entropies or [0.0] * len(advantages)

    p_loss  = compute_policy_loss(advantages, log_ratios, cfg.clip_epsilon)
    kl_pen  = compute_kl_penalty(current_logprobs, ref_logprobs, cfg.beta_kl)
    v_loss  = compute_value_loss(value_estimates, returns)
    ent_bon = compute_entropy_bonus(entropies, cfg.entropy_coeff)

    return compute_total_loss(p_loss, kl_pen, v_loss, ent_bon, cfg)
