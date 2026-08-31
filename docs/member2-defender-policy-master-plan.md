# Member 2 Master Plan: Defender Policy Lead

## Your final accountability

You own the Defender's ability to remain useful on normal cyber-support tasks while resisting injections, data leaks, and unsafe tool use. Your final evidence must show that safety improves without unacceptable capability loss.

Your core rule is non-negotiable: **the LLM may propose an action, but it never authorizes or executes its own tool call.** The external policy gate enforces every action.

## Success criteria by the final review

| Outcome | Minimum evidence |
|---|---|
| Tool safety | All unallowlisted, malformed, traversal, secret-access, network, and side-effecting calls are blocked and audited. |
| Injection resilience | MAPPO Defender beats SFT/static baseline on held-out and OOD attack ASR. |
| Task utility | Benign cyber-task success remains close to the SFT baseline; false refusals are measured and explained. |
| Capability retention | General non-attack task performance does not materially regress; KL regularization selection is justified. |
| Reproducibility | Dataset versions, seeds, configs, checkpoints, metrics, traces, and test commands are preserved. |

## Review 1 - Defender foundation and baseline

### A. Build and harden the policy boundary

Deliver:

- Strict structured `ToolCall` contract: tool, exact arguments, purpose.
- Fixed read-only allowlist for the demo environment.
- Schema validation with unknown-field rejection.
- Resource allowlists, traversal protection, input size limits, and sensitive-field redaction.
- Inert text sandbox with no shell, network, filesystem writes, nested tool calls, or live credentials.
- Fail-closed behavior and privacy-preserving audit logs.

Tests to own:

- Allowed read-only retrieval works.
- Unknown/side-effecting tool is blocked.
- Extra or malformed arguments are blocked.
- Path traversal is blocked.
- An email/document injection cannot authorize an export or disclosure.
- Sandbox detects suspicious content but never executes it.
- Audit events do not retain raw sensitive arguments or tool results.

### B. Prepare the Defender SFT warm-up

Build a versioned SFT corpus with four behaviors:

| Behavior | What a good target demonstrates |
|---|---|
| Benign answer | Correct, concise answer to a normal log/email/case task. |
| Benign tool use | Minimal valid read-only tool proposal; uses result only after policy-gate approval. |
| Injection resistance | Treats embedded instructions as data, completes the legitimate task, and never follows the payload. |
| Safe refusal | Declines hidden-prompt, secret, export, code-execution, or unsafe tool requests and gives a safe alternative. |

Data quality rules:

- Balance task type, injection family, and language style; do not train only on obvious “ignore previous instructions” prompts.
- Keep held-out benign and attack cases disjoint by template and scenario, not only by ID.
- Remove real credentials, PII, and executable payloads; fixtures remain synthetic and inert.
- Store source, scenario, label, split, and expected safe behavior for every record.

### C. Establish the baseline and reporting contract

Report for SFT/static Defender and the rule-based fixture baseline:

- ASR = unsafe compliance/leak/unsafe execution divided by attack cases.
- Benign task success = correct and safe completion divided by benign cases.
- False-refusal rate = benign cases refused divided by benign cases.
- Tool-block rate = invalid/unsafe calls blocked divided by invalid/unsafe attempts.
- Per-family breakdown: direct injection, indirect email/log injection, secret-extraction, malicious link/code, and unauthorized data operation.

### Review 1 review-day package

1. Architecture diagram and trust-boundary explanation.
2. Live or recorded safe tool-call demo: one allowed document lookup and two blocked attacks.
3. SFT data schema and split-validation report.
4. Baseline metric table and 3-5 representative traces.
5. Test output and reproducible run commands.

### Review 1 exit gate

Do not proceed until the gate cannot be bypassed by an LLM response, all safety tests pass, the data split is validated, and your group has selected the real base model/checkpoint for SFT.

## Between Review 1 and Review 2 - integration readiness

Before self-play starts, agree in writing with the other members on these interfaces:

| Need | Ask | Owner you depend on |
|---|---|---|
| Attacks | Stable `payload_id`, payload text, attack family, and success label format | Member 1 |
| Environment | `reset`, `step`, observation/action schemas, deterministic seed handling | Member 3 |
| Critic | Global-state fields, advantage format, checkpoint/load interface | Member 3 |
| Evaluation | Case schema, judge criteria, fixed held-out/OOD splits, dashboards | Member 4 |
| Training | Base model, GPU allocation, LoRA/full-tuning decision, experiment tracking | Whole team |

Lock the Review 1 gate as an invariant. In Review 2, improve the policy and reward—not by weakening or bypassing enforcement.

## Review 2 - MAPPO Defender and self-play

### A. Implement the Defender actor integration

Your actor receives only the decentralized observation: user task, untrusted content, safe context, approved/redacted tool results, and allowed action schema. It must not receive hidden attacker intent, protected labels, or critic-only global state.

Implement:

- Defender action representation: answer tokens plus structured tool-call tokens.
- Action masks for unsupported tools/fields and length/token budgets.
- Log probability, entropy, value-input metadata, and deterministic evaluation mode.
- Checkpointing of policy, tokenizer/prompt version, LoRA adapter, and generation config.
- A policy adapter that routes every proposed action through the unchanged Review 1 gate.

### B. Implement the Defender MAPPO update

Work with Member 3's centralized critic but own the Defender loss and experiments:

`L_def = L_MAPPO + beta_KL * KL(pi_def || pi_SFT) - c_entropy * entropy`

Use rewards that separately track:

- legitimate task utility;
- safety violation severity (leak, unsafe tool proposal, injection obedience);
- unnecessary refusal;
- excessive cost/latency, if measured.

Tune these in a controlled grid:

- `beta_KL`: low, medium, high values;
- safety-penalty weight;
- entropy/exploration coefficient;
- rollout length and update epochs;
- attacker/Defender update ratio.

Change one variable per ablation run; keep seed and held-out evaluation fixed.

### C. Make self-play stable

Watch for these failure modes every run:

| Failure | Symptom | Response |
|---|---|---|
| Defender collapses to refusal | ASR falls but benign success collapses | Increase task/rejection penalty, increase KL, rebalance data. |
| Attacker becomes repetitive | Few payload families, stagnant reward | Member 1 increases diversity/exploration; use payload-family caps. |
| Reward hacking | Good reward but unsafe traces | Strengthen evaluator/judge and manually audit samples. |
| Critic instability | Value loss explodes, oscillating policies | Member 3 adjusts normalization/GAE/lr; pause actor update. |
| Gate masking policy weakness | Gate blocks everything while model proposes unsafe calls | Report pre-gate proposal safety and gate outcomes separately. |

### Review 2 review-day package

1. Diagram of decentralized actors and centralized critic.
2. Defender MAPPO loss and reward specification.
3. Training curves: ASR, benign success, false refusals, KL, entropy, critic loss, and unsafe-proposal rate.
4. At least five chronological attacker/Defender traces that show attack evolution and Defender adaptation.
5. SFT baseline versus MAPPO checkpoint table, including selected hyperparameters.

### Review 2 exit gate

Self-play runs stably across more than one seed; ASR trends downward without a material benign-success collapse; all gate invariants and held-out splits remain unchanged.

## Between Review 2 and Review 3 - freeze and audit

- Freeze the candidate checkpoints and evaluation suite before final tuning.
- Create a sealed OOD set: no tuning, prompt editing, or parameter selection from its results.
- Audit 50 randomly sampled trajectories for judge errors, safety violations, and false refusals.
- Decide the final `beta_KL` using validation only, then lock it.
- Define success thresholds with the team before looking at OOD metrics.

## Review 3 - retention, final robustness, and submission

### A. Own capability-retention analysis

Compare SFT/static baseline, PPO baseline (if available), and final MAPPO Defender on:

- held-out benign cyber tasks;
- general non-attack tasks;
- each attack family in-distribution;
- OOD attack families;
- tool-use correctness and invalid-action rate.

For every result, report mean, seed variation/confidence interval when feasible, sample count, and relative change from the SFT baseline.

### B. Run Defender-focused ablations

At minimum:

| Ablation | Question answered |
|---|---|
| MAPPO without KL | Does retention regularization prevent capability loss? |
| MAPPO with low/high KL | What safety-utility trade-off does `beta_KL` control? |
| Static SFT Defender | Is self-play better than static alignment? |
| Gate disabled in offline synthetic test only | How much enforcement contributes, and can the model itself resist? |
| Reward without false-refusal penalty | Does the reward design create over-refusal? |

Never disable the gate in deployment or live demonstrations; use that ablation only in the isolated fixture environment.

### C. Produce final artifacts

- Defender policy README: threat model, constraints, tool contract, setup, run commands, limitations.
- Final results tables and plots, with data splits/seeds/checkpoint IDs.
- Qualitative trace appendix: successes, failures, false refusals, and root-cause notes.
- Reproducibility package: configs, evaluation cases, test commands, environment versions.
- Presentation content: 2-3 Defender slides (problem/threat model; Review 1 enforcement; final safety-utility results).

### Review 3 exit gate

Final results are reproducible from a clean setup, OOD evaluation is untouched during tuning, retention analysis is complete, limitations are explicit, and the team report distinguishes policy safety from gate enforcement.

## Your recurring operating checklist

Run this after every meaningful change:

1. Run unit and acceptance tests.
2. Re-run held-out benign and attack evaluation; record config, checkpoint, seed, timestamp.
3. Inspect a small random trace sample manually.
4. Confirm the policy gate and fixture allowlist have not changed unintentionally.
5. Update the experiment log with a one-line conclusion and next decision.

## Immediate next actions

1. Send the base-model/SFT setup request to the team.
2. Ask Member 1, 3, and 4 for the four Review 2 interface contracts listed above.
3. Expand the current SFT and evaluation fixtures before training: diverse indirect injections, benign look-alikes, and safe tool-use cases.
4. Run the actual SFT warm-up as soon as the checkpoint and compute are agreed.
5. Freeze the Review 1 baseline report and tag it as the comparison point for Review 2.
