# Evidence included here

This folder holds only the evidence for the three one-time evaluations. The full project
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

Each run directory contains these files:

- `config.json`: models, versions and fingerprints.
- `experiment-plan.json`: what was fixed before the run.
- `report.json`: per-task results.
- `independent-verification.json`: the audit that regraded every submission.
- `results.html`: a readable replay of every episode; open it in a browser.

Paths inside these files, and in `docs/`, refer to the original project folder. Many of
those directories are not included here.
