# Evidence included here

This folder holds only curated evidence: the five one-time evaluations and the RL-first runs. The full project
evidence is about 814 MB and is not in this repository. It includes every trajectory,
Kaggle round, source snapshot and the trained LoRA adapters.

| Directory | What it is |
| --- | --- |
| `native-final-test-v1/` | First sealed test (24 tasks), four Qwen3-0.6B stages, opened once |
| `scaleup-final-test-2-v1/` | Second sealed test (48 tasks), five arms, opened once |
| `scaleup-humanevalfix-v1/` | HumanEvalFix in the native tool format (117 tasks), three arms, run once |
| `native-repair-pilot-20261030-v1/config.json` | Frozen source fingerprint and task manifest that `tests/test_curriculum.py` checks |
| `*-registry.json` | One-time-opening records for each sealed evaluation |
| `scaleup-frozen.json` | Checkpoints and rules frozen before the second sealed test was opened |
| `final-test-3-v1/` | Third sealed test (191 tasks), four Qwen3-1.7B arms, opened once. `final3-results.json` has the rules and per-task outcomes; `sealed-tasks.json.gz` holds the tasks, including hidden cases |
| `humanevalfix-rlfirst-v1/` | HumanEvalFix in the two-turn format (114 tasks), four arms, run once |
| `rlfirst-frozen.json` | Adapter hashes, protocol fingerprints and rules, frozen before sealed test 3 was opened |
| `rlfirst-rl-hardened-v1/`, `rlfirst-rl-weak-v1/` | The two RL runs: plan, per-chunk training logs, and the full audit (`rl-verification.json`). The weak run's crashed first chunk is kept with its diagnosis |
| `rlfirst-sft-v1/` | The supervised arm's plan and training log |
| `rlfirst-dev-*-v1/` | Development evaluations of the four arms (graded results) |
| `rlfirst-settings-v2/`, `rlfirst-pool-v3/`, `rlfirst-sweep-v1/` | Preflight decision, task-pool summary, and the base-model sweep audit (the 35 MB pool itself is not included; `scripts/build_rlfirst_pool.py` rebuilds it) |

Each run directory contains these files:

- `config.json`: models, versions and fingerprints.
- `experiment-plan.json`: what was fixed before the run.
- `report.json`: per-task results.
- `independent-verification.json`: the audit that regraded every submission.
- `results.html`: a readable replay of every episode; open it in a browser.

The RL-first runs use `eval.json` (every episode, per arm), `final3-results.json` or
`eval-results.json` (graded outcomes), and `rl-verification.json` (the training audit).

Paths inside these files, and in `docs/`, refer to the original project folder. Many of
those directories are not included here.
