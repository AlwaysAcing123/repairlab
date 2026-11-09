# RepairLab

RepairLab trains small open-weight models (Qwen3 0.6B and 1.7B) to debug Python functions
with tools, using only free GPUs. Each result is measured on a sealed test set that is opened once.

## Results

On a sealed test of 48 unseen programs, opened once after every checkpoint and decision
rule was frozen:

| Model | Sealed test (48) | HumanEvalFix (117, run once) |
| --- | ---: | ---: |
| Qwen3-1.7B, untrained | 5 | 22 |
| Qwen3-0.6B, supervised + RL | 17 | 44 |
| Qwen3-1.7B, supervised only | 27 | — |
| Qwen3-1.7B, supervised with written diagnoses | 16 | — |
| **Qwen3-1.7B, supervised + RL** | **29** | **59** |

- **Training worked.** The untrained 1.7B solved 5 of 48; trained, it solved 29. Against
  the trained 0.6B, it won 17 of the tasks only one of them solved and lost 5 (exact sign
  test p = 0.017, under a rule written before the run).
- **Supervised training did most of the work.** RL on top of the supervised model gained
  4 tasks and lost 2, which is indistinguishable from chance. The RL stage was small: 6
  cycles, 1,320 sampled attempts, 24 optimizer steps.
- **Teaching the model to write a diagnosis before each fix hurt** (16 against 27,
  p = 0.019). It learned to quote the failing test correctly, but the fix often didn't
  follow from the quote.
- **The models still repeat themselves.** About half of the 1.7B model's later edits
  repeat an earlier one.
- **HumanEvalFix** consists of 164 human-written bugs, converted mechanically to the same
  tool format. 117 remained after dropping three groups:
  - 29 that couldn't be converted;
  - 10 that failed a control check;
  - 8 that resembled a training program.

  The ranking matches the sealed test. These are adapted scores, not comparable to
  published leaderboards.

Full numbers are in [the 1.7B results](docs/scaleup-results-20261106.md), its
[plan, written before any result](docs/scaleup-plan-20261105.md), and the earlier
[0.6B RL results](docs/native-rl-results-20261105.md). The first sealed test, which is
also spent, scored 0, 5, 7 and 9 of 24 across the four 0.6B stages.

## How it works

**Task.** Each task gives the model a buggy function, its specification and a few public
examples. The model has four tools (read, test, edit, submit) and at most eight actions.

**Grading.** A submission is run on every input in a small declared domain, and its
outputs are compared with the reference implementation's. The inputs and expected
outputs are hidden from the model. The candidate runs in a fresh Docker container
with these restrictions:

- no network access;
- a read-only filesystem;
- dropped Linux capabilities;
- memory, CPU, process and time limits.

The grader runs outside the container, on the host. Before a task is admitted, it must
pass four controls:

1. the broken code fails;
2. the reference fix passes;
3. a second, different fix passes;
4. code that hard-codes the public examples fails.

**Training.**

- **Supervised:** LoRA fine-tuning on 480 next-action examples from scripted repairs,
  including recovery after a failed test.
- **RL:** GRPO-style training on top. The reward is the hidden grader's pass or fail.
- **Compute:** free Kaggle T4 GPUs, about 22 GPU hours for the 1.7B sequence.

**Evaluation discipline.**

- Rules are written before each run.
- Choices between checkpoints use development tasks only.
- Sealed tests are opened once and recorded in a registry.
- Every run is audited by regrading every submission, first on a copy and then in place.
- Reruns reproduce the saved trajectories exactly.

## Limitations

- **The tasks are small single-function helpers.** The AI agent that built the project
  wrote them, and no person has reviewed them independently. They are not real-world bugs.
- **Each result comes from one seed and greedy decoding.**
- **RL's contribution is unresolved.** The RL stage was too small to measure its effect.
- **HumanEvalFix may be contaminated.** HumanEval is public, so the models may have seen
  it during pretraining.
- **The audits are programmatic.** Nobody has replicated the results independently.

## Repository layout

| Path | Contents |
| --- | --- |
| `repairlab/` | Tasks, sandboxed executor, grader, episodes, training-data builders |
| `scripts/` | Kaggle runners, notebook builders, audits |
| `tests/` | Unit and Docker integration tests |
| `docs/` | Plans written before each result, results reports, and the [chronological log](docs/project-log.md) |
| `artifacts/` | Reports, audits and replays for the three one-time evaluations (see [its README](artifacts/README.md)) |
| `data/humanevalfix/` | Pinned HumanEvalFix snapshot with its MIT license |

## Running it

You need Python 3.9 or later and Docker. The package itself has no dependencies.

```sh
python3 -B -m repairlab list     # tasks and their input domains
make test                        # unit tests
make integration-test            # sandbox tests, needs Docker
make demo                        # scripted repair replay; no model involved
```

GPU runs use private Kaggle notebooks through the Kaggle CLI. The notebook IDs in
`scripts/` name the author's Kaggle account, so change them to run on yours.

## How this was built

AI coding agents wrote the code, tasks and reports: OpenAI Codex first, then Claude
Code. The author chose the direction, set the rules and made the decisions.
[PROVENANCE.md](PROVENANCE.md) records who did what, and
[VALIDATION.md](VALIDATION.md) records what was actually exercised.

## License

The code is under the MIT license (see [LICENSE](LICENSE)). The HumanEvalFix data keeps its
own MIT license in `data/humanevalfix/LICENSE`. Model weights and trained adapters are not
included.
