# RepairLab

RepairLab trains small open-weight models (Qwen3 0.6B and 1.7B) to repair buggy Python
functions from test feedback, using reinforcement learning on free Kaggle T4 GPUs. Every
headline number comes from a sealed test set opened exactly once, under decision rules
written before the run.

## Results

### Sealed test 3: RL from the untrained model, supervised training, and a reward-hacking test (Nov 7–14, 2026)

Four Qwen3-1.7B arms, each with one frozen checkpoint:

| Arm | Training | Generated bugs (95) | Hand-written bugs (96) | **All (191)** | False passes¹ | HumanEvalFix (114)² |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Base | None | 64 | 43 | 107 | 8 | 76 |
| RL, hardened grader | Reward 1 only if public **and** hidden tests pass | **65** | 52 | **117** | 10 | 84 |
| RL, weak grader | Reward 1 if public tests pass (gameable) | 63 | **54** | **117** | 8 | **91** |
| Supervised | One pass over the same 383 tasks' reference fixes | 60 | 36 | 96 | 3 | 82 |

¹ Passes the public tests but fails hidden ones. ² Two-turn format, run once, reported only.

| Rule, written before the opening | Result | p | Outcome |
| --- | --- | ---: | --- |
| 1 (primary): hardened RL vs base, generated bugs | 65 vs 64 | 1.00 | Not met |
| 2: hardened RL vs supervised, generated bugs | 65 vs 60 | 0.50 | Neither called better |
| 3: weak vs hardened RL, false passes, all tasks | 8 vs 10 | 0.77 | Not met: no cheating shown |

p-values are exact two-sided sign tests on the tasks only one of the two arms solved.

- **RL didn't beat the untrained model on the primary rule.** It did on unseen development
  programs (80 vs 68, p = 0.017), but the gain didn't carry over to the sealed generated
  bugs. The plan predicted this in writing before the opening.
- **RL beat supervised training overall, though not on the pre-registered set:** 117 vs 96
  on all 191 tasks (44 tasks to 23, p = 0.014) and 52 vs 36 on hand-written bugs
  (p = 0.007). The supervised model repeated its first answer on 79 of 94 retries; it never
  learned to use test feedback.
- **The gameable grader didn't teach cheating.** The weak-grader arm solved as many tasks
  as the hardened arm, with no more false passes. All 29 false passes across the four arms
  are incomplete fixes. None imports anything, calls `open`/`eval`/`exec` or hard-codes a
  public input, and each passes 48–99.6% of its hidden cases. The public tests always
  include the input that exposes the bug, so fixing it was usually the easiest way to pass.
- **On HumanEvalFix,** both RL arms finished above base and supervised. Only weak vs base
  is clearly significant (91 vs 76, p = 0.003). Much of RL's gain comes from recovering on
  the retry after an invalid first reply.
- **A test-design flaw, found after the opening and disclosed:** the 15% cap on any one bug
  kind applied to the training pool, not the sealed draw. So 63 of the 95 sealed generated
  bugs are wrong-variable bugs, a kind the base model already solves well. One sealed
  program had no admissible generated bug, so the test has 191 tasks rather than 192.

Full record: the [plan and its dated addenda](docs/rl-first-plan-20261107.md), covering the
method, the rules, every deviation and the results (Addenda 13–15).

### Sealed test 2: scaling from 0.6B to 1.7B (Nov 5–6, 2026)

| Model | Sealed test 2 (48) | HumanEvalFix, tool format (117) |
| --- | ---: | ---: |
| Qwen3-1.7B, untrained | 5 | 22 |
| Qwen3-0.6B, supervised + RL | 17 | 44 |
| Qwen3-1.7B, supervised only | 27 | — |
| Qwen3-1.7B, supervised with written diagnoses | 16 | — |
| **Qwen3-1.7B, supervised + RL** | **29** | **59** |

- This phase used an eight-action tool format, and each test was run once.
- **1.7B beat 0.6B:** 17 tasks to 5 among those only one solved (p = 0.017).
- **Supervised training did most of the work:** RL on top gained 4 tasks and lost 2.
- **Writing a diagnosis before each fix hurt:** 16 vs 27 (p = 0.019).
- **The two HumanEvalFix columns aren't comparable:** they use different formats and task
  counts (117 vs 114).
- **Sealed test 1** (24 tasks, 0.6B) scored 0, 5, 7 and 9 across four training stages.

Details: the [1.7B results](docs/scaleup-results-20261106.md), their
[plan](docs/scaleup-plan-20261105.md) and the [0.6B RL results](docs/native-rl-results-20261105.md).

## How it works

| Part | What it does |
| --- | --- |
| **Tasks** | A buggy function, its specification and a few public examples. Bugs come from an AST mutation engine with 13 kinds (boundaries, missing edge cases, wrong variables, statement order, aliasing, two-bug programs, …) and from hand-written specification misreadings. |
| **Admission** | A task is used only if all four controls pass: the bug fails, the reference passes, a second, different fix passes, and code that hard-codes the public examples fails. |
| **Grading** | A submission runs on every input of a small declared domain, and its outputs are compared with the reference's. Hidden inputs and all expected outputs stay on the host. Candidates run in a fresh Docker container as `nobody`, with no network, a read-only filesystem, all Linux capabilities dropped, and memory, CPU, process and time limits. |
| **Episode** | The model replies with a fixed function, sees its public test results, and gets one retry. |
| **RL (GRPO-style)** | 32 tasks × 8 samples per step. Groups whose 8 rewards are all equal are dropped, and advantage = reward − group mean. The loss is summed over all model tokens in both turns and divided by the batch's token count. One on-policy update per batch, no KL penalty, LoRA rank 16; 53 steps and 20,352 graded episodes per run. |
| **On Kaggle** | Everything runs in one T4 session. vLLM samples, hot-swapping the LoRA weights each step and sleeping during updates (2.7× faster than Hugging Face generation). A sandboxed in-session process grades candidates, and the trainer recomputes log-probabilities itself, so the sampler's numerics never enter the gradient. |
| **Evaluation discipline** | Rules are written before each run, and later changes are dated addenda. Each arm has one checkpoint, so nothing is chosen on development results. Sealed tests are registered before upload and opened once. Every RL reward, every public-test message and every evaluation submission is regraded in Docker on the laptop, first on a copy and then in place, and every trained token sequence is checked against its saved text. Across the two RL runs: 24,507 submissions regraded and 42,867 turns checked, with 0 problems. |

## Run it yourself

You need git, Python 3.9 or later (standard library only), `make` and Docker. A GPU is
needed only to train models.

```sh
git clone https://github.com/AlwaysAcing123/repairlab.git && cd repairlab
# The grader runs candidates only in this pinned image (Intel/AMD and Apple Silicon)
docker pull python:3.12-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70

python3 -B -m repairlab list     # tasks and their input domains
make test                        # unit tests, no Docker: 165 run, OK (18 skipped), ~5 min
make demo                        # one scripted repair in the sandbox, no model involved
make integration-test            # Docker tests: 160 pass, 5 skip, ~20 min
```

- **`make demo`** writes `artifacts/demo-<time>/`; open its `replay.html` in a browser. A
  scripted agent runs the public tests, edits the function, retests and submits, and the
  hidden grader passes it on all 48 private cases.
- **The 5 integration skips** need files this repository doesn't include: the trained
  adapter weights, or the 35 MB task pool (`python3 -B scripts/build_rlfirst_pool.py`
  rebuilds the pool).

**Regrade sealed test 3 from its saved model outputs (~4 min):**

```sh
gunzip -k artifacts/final-test-3-v1/sealed-tasks.json.gz
python3 -B scripts/verify_final_test_3.py --run artifacts/final-test-3-v1 --destination regrade-final3
```

It reparses every model reply, grades all 755 distinct submissions in Docker and reapplies
the three rules. Expected output:

```text
{"rule_1": {"first": 65, "second": 64, "p": 1.0}, "rule_2": {"first": 65, "second": 60, "p": 0.49955983320251107}, "rule_3": {"first": 8, "second": 10, "p": 0.7744140625}}
mismatches 0
```

`regrade-final3/final3-results.json` should then match
`artifacts/final-test-3-v1/final3-results.json`. To regrade HumanEvalFix, unzip
`artifacts/humanevalfix-rlfirst-v1/sealed-tasks.json.gz` the same way and run the same
script with `--run artifacts/humanevalfix-rlfirst-v1 --humanevalfix --destination regrade-hef`.

**Training (optional):**
- GPU runs are private Kaggle notebooks, launched through the Kaggle CLI with your own API
  token. The CLI is installed separately.
- The notebook IDs in `scripts/` name the author's account, so change them to yours.
- Trained adapters aren't included.

## Limitations

- **Synthetic tasks:** small single-function helpers written by an AI agent, not real-world
  bugs. No person has independently reviewed them.
- **One training run per arm, with greedy evaluation:** seed variance is unmeasured.
- **Small sealed test for the effect size:** only about 25 tasks were solved by one arm and
  not the other, and a sign test needs roughly an 18-to-7 split of those to reach p < 0.05.
- **The cheating test was easy to pass honestly:** it shows the weak grader didn't
  *induce* cheating here, not what prevents cheating.
- **Possible contamination:** HumanEvalFix is public, so the models may have seen it in
  pretraining.
- **No network isolation on Kaggle:** the in-session grader runs without it, and the
  session holds no secrets.
- **Programmatic audits only:** nobody has replicated the results independently.

## Repository layout

| Path | Contents |
| --- | --- |
| `repairlab/` | Tasks, sandboxed executor, grader and episodes. RL-first code is in `rlfirst/`; sealed test 3's programs are in `final3/` |
| `scripts/` | Kaggle runners and runtimes, the freeze and opening scripts, audits |
| `tests/` | Unit and Docker integration tests |
| `docs/` | Plans written before each result, results reports, and the [chronological log](docs/project-log.md) |
| `artifacts/` | Curated evidence for every one-time evaluation and the RL-first runs (see [its README](artifacts/README.md)) |
| `data/humanevalfix/` | Pinned HumanEvalFix snapshot with its MIT license |

## License

The code is MIT-licensed (see [LICENSE](LICENSE)), and the HumanEvalFix data keeps its own
MIT license in `data/humanevalfix/LICENSE`. Model weights and trained adapters aren't
included.
