# MAPPO-Sec — Defender Policy

The Defender workstream for the MAPPO-Sec capstone: an LLM that completes legitimate
cyber-support tasks while resisting direct and indirect prompt injection. Owned by
Member 2 (Defender Policy Lead). See `docs/three-review-architecture.md` for the whole-team
system design, and `docs/member2-defender-policy-master-plan.md` for this workstream's plan.

## Threat model, in one paragraph

An LLM that can read untrusted content (email, logs, documents) and also propose tool calls has
collapsed two different kinds of input into one channel: the task the user asked for, and
whatever text happens to be embedded in the content it was told to read. A prompt injection
attack exploits this — it is not a jailbreak trick, it is a confusion-of-authority bug. If the
model that can be persuaded by that content is also the model that authorizes tool execution,
the system has no real boundary. So the Defender **proposes**; it never **authorizes**.

## Constraints (non-negotiable)

- The Defender may propose a tool call. It never executes or authorizes its own tool call —
  every proposal passes through the policy gate (`src/defender_policy/gate.py`), which is
  deterministic code, not a learned policy, and is fixed during SFT and MAPPO training.
- The gate allowlist cannot be disabled for training or "for now" convenience. See
  `configs/ablation_configs/ablation_no_gate.json` for why the no-gate ablation is documented but
  never executed.
- No real secrets, live inboxes, shells, or unrestricted network access appear anywhere in
  evaluation. All fixtures are synthetic and inert.

## Tool contract

Every proposed action is a structured `ToolCall`: `{tool, arguments, purpose}`. The gate checks it
against nine serial rules (`GATE-001`–`GATE-009`, see `docs/defender-review2-evidence.md` §4) —
empty/unknown tool name, invalid arguments, oversized purpose, secret-access keywords,
external-URL/code-exec requests, path traversal, indirect injection in a tool result, and runtime
exceptions. A call must clear all nine to reach the sandbox; failing any one exits to a block.
Approved results are marked untrusted and redacted before returning to the Defender's context.

## Setup

```bash
git clone https://github.com/shivashant94/RL-for-Self-Improving-Cyber-LLMs.git
cd RL-for-Self-Improving-Cyber-LLMs
pip install -r requirements.txt   # only needed for real SFT training/inference, not for tests
```

## Run commands

**Acceptance tests (no ML dependencies required):**
```bash
python3 -m unittest discover -s tests -v
```

**Fixture baseline (rule-based, not a trained model):**
```bash
python3 scripts/prepare_sft.py
python3 scripts/run_review1_baseline.py
python3 scripts/evaluate_checkpoint.py --checkpoint fixture_baseline
python3 scripts/run_capability_retention_analysis.py --checkpoint fixture_baseline
```

**Real SFT training** (on Colab/Kaggle with a GPU runtime — see `docs/infra-training-setup.md`):
```bash
python3 scripts/format_sft_for_training.py --split train
python3 scripts/format_sft_for_training.py --split validation
python3 src/train_sft.py \
    --dataset_path data/sft/train_formatted.jsonl \
    --model_name Qwen/Qwen2.5-0.5B \
    --output_dir ./defender_checkpoints
```

**Evaluate a real checkpoint** (once downloaded locally — checkpoint directories are gitignored,
never commit them):
```bash
python3 scripts/evaluate_checkpoint.py \
    --checkpoint <id> --checkpoint-path defender_checkpoints/checkpoint-N
python3 scripts/run_capability_retention_analysis.py \
    --checkpoint <id> --checkpoint-path defender_checkpoints/checkpoint-N
```

The scripts write reproducible manifests and metrics under `reports/`. Fixture-baseline results
are deterministic and validate the safety environment; they are deliberately never represented as
a trained model's performance. Real-checkpoint results carry their own scope disclaimer — see
`docs/defender-final-report.md` for the current real result and its limitations.

## What is included

- A strict, allowlisted tool registry and a fail-closed policy gate (`GATE-001`–`009`).
- Mock, read-only fixtures for email, documents, and records; an inert text sandbox that
  classifies text but never executes code or fetches URLs.
- A versioned SFT corpus (`data/sft/`) covering four behaviors: benign answer, benign tool use,
  injection resistance, safe refusal — with disjoint train/validation/held-out splits.
- `SFTModelAdapter`: loads a real LoRA checkpoint and routes its output through the same gate,
  reward, and evaluation pipeline as the fixture baseline.
- A MAPPO Defender loss (PPO clip + KL anchor to the SFT reference + value loss + entropy bonus),
  a decomposed reward function, and a fixture self-play episode harness.
- An 8-probe capability-retention suite and three ablation configs (no-KL, high-KL, no-gate).
- 291 tests, all passing without any ML dependency installed (real-checkpoint code paths are
  lazily imported so the test suite never requires torch/transformers/peft to run).

## Limitations (current, honest)

- No model call proposes anything outside this policy gate — but a proposal that satisfies all
  nine gate rules is not itself judged for intent; that judgment is the trained policy's job.
- Real-model ASR/capability numbers currently rely on a keyword-based heuristic classifier
  (`_AdapterShim` in `evaluate_checkpoint.py`), not a real judge. This is provisional pending
  Member 4's `EvaluationEvent`/judge criteria (`three-review-architecture.md`).
- Self-play and ablation results require Member 1's `AttackPayload` schema and Member 3's
  `Trajectory`/`GlobalState`/environment — none of which exist in this repo yet.
- The one real checkpoint trained so far (`defender_checkpoints/checkpoint-20`) used a minimal
  smoke-test budget (1 epoch, 20 examples) and should not be read as representative of the
  Defender's real capability. See `docs/defender-final-report.md`.

## Documentation index

| Doc | Contents |
|---|---|
| `docs/three-review-architecture.md` | Whole-team system design and interfaces |
| `docs/member2-defender-policy-master-plan.md` | This workstream's full Review 1–3 plan |
| `docs/defender-review2-evidence.md` | Gate rules, reward/loss formulas, fixture metrics, real-checkpoint results §13 |
| `docs/defender-review3-evidence.md` | Review 3 evaluation matrix and ablation plan |
| `docs/defender-final-report.md` | Trace appendix, reproducibility package, final real-checkpoint results |
| `docs/infra-training-setup.md` | Colab/Kaggle SFT training setup guide |
