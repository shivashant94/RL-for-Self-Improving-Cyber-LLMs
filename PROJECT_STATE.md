# Project State

## Current phase

Part 10: Review 3 final summary and presentation artefacts (complete).
All three review phases are now fully implemented as fixture infrastructure.
SFT and MAPPO training remain BLOCKED_EXTERNAL.

## Completed

- Three-review MAPPO-Sec architecture documented.
- Member 2 Defender master plan documented.
- **Review 1 complete (Parts 1–3, 2026-08-26):** Gate GATE-001–009, SFT corpus 20/8/10, scripts, 5 sanitized traces, 43 tests.
- **Part 4 Review 2 integration contracts (2026-08-26):** model_adapter, rollout_adapter, rewards. 104 tests.
- **Part 5 MAPPO update + capability retention (2026-08-27):** mappo_defender, capability_retention. 162 tests.
- **Part 6 Fixture episode harness (2026-08-27):** episode_harness, run_review2_episode. 208 tests.
- **Part 7 Review 2 docs (2026-08-27):** review2_integration_summary, defender-review2-evidence. 208 tests.
- **Part 8 Checkpoint evaluator (2026-08-27):** evaluate_checkpoint, heldout guard. 242 tests.
- **Part 9 Ablation configs + retention analysis (2026-08-28):** 3 ablation JSONs, retention script, JSON+CSV reports. 284 tests.
- **Part 10 Review 3 final artefacts (2026-08-28):**
  - `reports/review3_readiness_summary.json`: machine-readable readiness record — all 12 deliverables (READY or BLOCKED_EXTERNAL), fixture baseline metrics (in-dist + OOD + retention), full ablation plan with hypotheses, interface readiness (Member 1/3/4 PENDING), 7-step run instructions for when training is unblocked.
  - `docs/defender-review3-evidence.md`: full Review 3 presentation — 5-slice evaluation matrix, metric comparison tables (SFT/MAPPO BLOCKED_EXTERNAL), per-family ASR, capability retention table, 8-probe per-category rates, ablation plan with no-gate NOT_RUN rationale, end-to-end run instructions, interface readiness, 284-test coverage table, 9 generated reports indexed, 6 open blockers.
  - All 284 tests still pass.

## In progress

Nothing in progress.

## Next exact action

All three review phases are now complete as fixture infrastructure.
The project is in HOLDING state pending:
1. Team confirmation of base LLM checkpoint and training stack → unblocks SFT warm-up.
2. Member 1 AttackPayload schema → unblocks MAPPO self-play.
3. Member 3 Trajectory/Critic schema → unblocks advantage computation.
4. Member 4 EvaluationEvent/OOD suite → unblocks Review 3 held-out evaluation.

When any blocker clears, resume from the corresponding step in
`reports/review3_readiness_summary.json § how_to_run_once_unblocked`.

## Latest verification

- Command: `python3 -m unittest discover -s tests -v`
- Result: 284 tests passed, 0 failed.
- Scripts: all 6 scripts exit 0 (dry-run verified):
  - `prepare_sft.py --dry-run`
  - `run_review1_baseline.py --dry-run`
  - `run_review2_episode.py --dry-run`
  - `evaluate_checkpoint.py --dry-run`
  - `run_capability_retention_analysis.py --dry-run`
  - `evaluate_checkpoint.py --checkpoint fixture_baseline --allow-heldout` (full run)

## Decisions locked

- Base model: PENDING. SFT training BLOCKED_EXTERNAL.
- Dataset version: `review1-fixture-v2` / `sft-v2`.
- Heldout splits FROZEN — `allow_heldout=True` required.
- Gate: GATE-001–009; fixed. Cannot be disabled for ablations.
- Action schema version: `gate-v1`.
- Reward weights: lambda_violation=1.0, lambda_refusal=0.3, lambda_cost=0.0.
- MAPPO defaults: clip_epsilon=0.2, beta_kl=0.1, value_loss_coeff=0.5, entropy_coeff=0.01, gamma=0.99, gae_lambda=0.95.
- Ablation betas: no_kl=0.0, default=0.1, high_kl=1.0.

## Blockers / questions for team

- Base LLM/checkpoint, training stack, LoRA/QLoRA/full-SFT, GPU budget → SFT BLOCKED_EXTERNAL.
- Member 1 AttackPayload schema → MAPPO self-play blocked.
- Member 3 Trajectory/Critic schema → trajectory assembly blocked.
- Member 4 EvaluationEvent/OOD suite → Review 3 eval blocked.
