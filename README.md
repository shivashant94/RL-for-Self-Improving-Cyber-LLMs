# MAPPO LLM Capstone

This repository begins with the Review 1 Defender foundation: a safe, testable tool-execution boundary for the Defender LLM.

## Run the acceptance tests

```bash
python3 -m unittest discover -s tests -v
```

## Run the Review 1 baseline

```bash
python3 scripts/prepare_sft.py
python3 scripts/run_review1_baseline.py
```

The scripts write reproducible manifests and metrics under `reports/`. The included baseline is deterministic and fixture-only; it validates the safety environment. It is deliberately not represented as a trained LLM. Connect the team-approved base model to `configs/sft_warmup.json` for actual SFT, while retaining the same data split and policy gate.

## What is included

- A strict, allowlisted tool registry.
- A fail-closed policy gate that validates structured model-proposed tool calls.
- Mock, read-only fixtures for email, documents, and records.
- An inert text sandbox: it classifies text only and cannot execute code or fetch URLs.
- Structured audit events and Review 1 acceptance tests.

No model call or real external tool is included. Those should be connected later behind this policy gate, never directly from the Defender policy.
