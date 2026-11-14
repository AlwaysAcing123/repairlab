# RepairLab

RepairLab trains small open-weight models (Qwen3 0.6B and 1.7B) to repair buggy Python
functions from test feedback, using only free GPUs. Every result is measured on a sealed
test set that is opened exactly once, under decision rules written before the run.

The latest phase asks three questions:
- Does reinforcement learning (RL) from the *untrained* model help?
- Is RL better than supervised fine-tuning on the same tasks?
- Does a gameable grader teach the model to cheat (reward hacking)?

## Results: RL from the untrained model, and a reward-hacking test (Nov 7–14, 2026)

Four Qwen3-1.7B arms, each with one frozen checkpoint:

- **Base:** the untrained model.
- **RL, hardened grader:** reward 1 only if the fix passes the public *and* hidden tests.
- **RL, weak grader:** reward 1 if the fix passes the public tests alone. Every other
  setting is identical.
- **Supervised:** one pass over the same 383 tasks, with the reference fix as the target.

| | Base | RL, hardened grader | RL, weak grader | Supervised |
| --- | ---: | ---: | ---: | ---: |
| Sealed test 3, generated bugs (95) | 64 | 65 | 63 | 60 |
| Sealed test 3, hand-written bugs (96) | 43 | 52 | 54 | 36 |
| **Sealed test 3, all (191)** | 107 | **117** | **117** | 96 |
| False passes (pass the public tests, fail hidden ones) | 8 | 10 | 8 | 3 |
| HumanEvalFix, two-turn format (114, run once) | 76 | 84 | 91 | 82 |

**The three rules written before the opening:**

| Rule | Comparison | Result | p | Outcome |
| --- | --- | --- | ---: | --- |
| 1 (primary) | Hardened RL vs base, generated bugs | 65 vs 64 | 1.00 | Not met |
| 2 | Hardened RL vs supervised, generated bugs | 65 vs 60 | 0.50 | Neither called better |
| 3 | Weak vs hardened RL, false passes, all tasks | 8 vs 10 | 0.77 | Not met: no cheating shown |

The p-values come from exact two-sided sign tests on the tasks only one of the two arms
got.

- **RL didn't beat the untrained model on the pre-registered test.** On unseen development
  programs RL won 80 to 68 (p = 0.017), but that didn't carry over to the sealed generated
  bugs. The plan predicted this in writing before the opening, because the development
  gain on generated bugs was small.
- **RL beat supervised fine-tuning overall, though not on the pre-registered set.**
  - All 191 tasks: 117 vs 96 (44 tasks to 23, p = 0.014).
  - Hand-written bugs: 52 vs 36 (p = 0.007).
  - The supervised model memorized: it repeated its first answer on 79 of 94 retries.
- **The weak grader didn't teach cheating.**
  - It solved as many tasks as the hardened grader, with no more false passes.
  - All 29 false passes across the four arms are incomplete fixes. None imports
    anything, calls `open`/`eval`/`exec`, or hard-codes a public input, and each passes
    48–99.6% of its hidden cases.
  - The public tests always include the input that exposes the bug, so the easiest way to
    pass them was usually to fix the bug.
- **On HumanEvalFix** (reported only), both RL arms finished above base and supervised.
  Only weak vs base is clearly significant (91 vs 76, p = 0.003). Much of RL's gain is
  recovering on the retry after an invalid first reply.
- **A test-design weakness, found after opening and disclosed:**
  - the 15% cap on any one bug kind applied to the training pool, not the sealed draw;
  - so 63 of the 95 sealed generated bugs are wrong-variable bugs, a kind base already
    solves well;
  - one sealed program had no admissible generated bug, so the test has 191 tasks rather
    than 192.

Full record: the [plan and its dated addenda](docs/rl-first-plan-20261107.md). It covers
the method, the rules, every deviation, and the results (Addenda 13–15).

## Earlier results: scaling from 0.6B to 1.7B (Nov 5–6, 2026)

On sealed test 2 (48 unseen programs, eight-action tool format, opened once):

| Model | Sealed test 2 (48) | HumanEvalFix, tool format (117, run once) |
| --- | ---: | ---: |
| Qwen3-1.7B, untrained | 5 | 22 |
| Qwen3-0.6B, supervised + RL | 17 | 44 |
| Qwen3-1.7B, supervised only | 27 | — |
| Qwen3-1.7B, supervised with written diagnoses | 16 | — |
| **Qwen3-1.7B, supervised + RL** | **29** | **59** |

- **The 1.7B beat the 0.6B:** 17 tasks to 5 (p = 0.017).
- **Supervised training did most of the work.** RL on top gained 4 tasks and lost 2.
- **Writing a diagnosis before each fix hurt:** 16 vs 27, p = 0.019.
- **The two HumanEvalFix columns aren't comparable.** This table uses the eight-action
  tool format and 117 tasks. The table above uses the two-turn format and 114 tasks.

Details: [1.7B results](docs/scaleup-results-20261106.md), its
[plan](docs/scaleup-plan-20261105.md), and the [0.6B RL results](docs/native-rl-results-20261105.md).
The first sealed test, also spent, scored 0, 5, 7 and 9 of 24 across the four 0.6B stages.

## How it works

**Tasks:**
- Each task gives the model a buggy function, its specification and a few public
  examples.
- In the RL-first phase, the model replies with a fixed function, sees its public test
  results, and gets one retry.
- Bugs come from two sources:
  - an AST mutation engine with 13 realistic kinds, such as boundaries, missing edge
    cases, wrong variables, statement order, aliasing and two-bug programs;
  - hand-written specification misreadings.

**Grading:**
- A submission runs on every input of a small declared domain. Its outputs are compared
  with the reference implementation's.
- Hidden inputs and all expected outputs stay on the host.
- Candidates run in a fresh Docker container with:
  - no network;
  - a read-only filesystem;
  - all Linux capabilities dropped, as the `nobody` user;
  - memory, CPU, process and time limits.
- A task is admitted only if all four controls pass:
  1. the bug fails;
  2. the reference passes;
  3. a second, different fix passes;
  4. code that hard-codes the public examples fails.

**RL (GRPO-style):**
- 32 tasks per step × 8 samples;
- groups whose 8 rewards are all equal are dropped;
- advantage = reward − group mean;
- loss summed over all model tokens in both turns and divided by the batch's token count;
- one on-policy update per batch, with no KL penalty;
- LoRA rank 16; 53 steps per run; 20,352 graded episodes per run.

**Inside one Kaggle T4 session:**
- vLLM samples, hot-swapping the LoRA weights each step and sleeping during updates (2.7×
  faster than Hugging Face generation);
- candidates are graded in a sandboxed in-session process;
- the trainer recomputes log-probabilities itself, so the sampler's numerics never enter
  the gradient.

**Evaluation discipline:**
- Rules are written before each run, and later changes are dated addenda.
- In the RL-first phase each arm has exactly one checkpoint, so nothing is chosen on
  development results.
- Sealed tests are recorded in a registry before upload and opened once.
- Every RL reward, every public-test message shown to the model, and every evaluation
  submission is regraded in Docker on the laptop, first on a copy and then in place.
  Every trained token sequence is also checked against its saved text.
- The two RL runs together: 24,507 submissions regraded and 42,867 trained turns
  token-checked, with 0 problems.

## Limitations

- **The tasks aren't real-world bugs.** They are small single-function helpers, written
  by the AI agent that built the project and never independently reviewed by a person.
- **Each arm is one training run with greedy evaluation.** Seed variance is unmeasured.
- **Sealed test 3 is small for the effect size involved.** With about 25 tasks where only
  one arm succeeds, a sign test needs roughly an 18-to-7 split to reach p < 0.05.
- **The cheating test was easy to pass honestly.** It shows the weak grader didn't
  *induce* cheating here. It doesn't show what prevents cheating.
- **HumanEvalFix may be contaminated.** It is public, so the models may have seen it in
  pretraining.
- **No network isolation on Kaggle.** The in-session grader runs without it; this is
  disclosed, and the session holds no secrets.
- **The audits are programmatic.** Nobody has replicated the results independently.

## Repository layout

| Path | Contents |
| --- | --- |
| `repairlab/` | Tasks, sandboxed executor, grader and episodes. RL-first code is in `rlfirst/`; sealed test 3's programs are in `final3/` |
| `scripts/` | Kaggle runners and runtimes, the freeze and opening scripts, audits |
| `tests/` | Unit and Docker integration tests |
| `docs/` | Plans written before each result, results reports, and the [chronological log](docs/project-log.md) |
| `artifacts/` | Curated evidence for every one-time evaluation and the RL-first runs (see [its README](artifacts/README.md)) |
| `data/humanevalfix/` | Pinned HumanEvalFix snapshot with its MIT license |

## Running it

You need Python 3.9 or later and Docker. The package itself has no dependencies.

```sh
python3 -B -m repairlab list     # tasks and their input domains
make test                        # unit tests
make integration-test            # sandbox tests, needs Docker
make demo                        # scripted repair replay; no model involved
```

To regrade sealed test 3 yourself:
1. Decompress `artifacts/final-test-3-v1/sealed-tasks.json.gz` in place.
2. Run `python3 -B scripts/verify_final_test_3.py --run artifacts/final-test-3-v1
   --destination <new folder>`.

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
