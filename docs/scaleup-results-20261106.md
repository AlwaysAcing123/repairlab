# Qwen3-1.7B results — November 6, 2026

**Under the rule fixed before any 1.7B run, 1.7B counts as better.** On the second
sealed final test, opened once, the final 1.7B checkpoint solved **29 of 48** tasks and
the 0.6B RL adapter **17 of 48**. Of the tasks only one of them solved, 1.7B had 17 and
0.6B had 5 (exact sign test p = 0.017; the rule needed p < 0.05).

Rules, runs, and every incident are in [the plan](scaleup-plan-20261105.md), written
before any 1.7B result. One seed, one greedy attempt per task, AI-authored tasks.

## Second final test, opened once

All five frozen arms ran all 48 sealed tasks, unseeded, under the development rules.
No checkpoint was chosen on these results.

| Measure | 0.6B RL | 1.7B untrained | 1.7B plain | 1.7B notes | **1.7B RL** |
| --- | ---: | ---: | ---: | ---: | ---: |
| Successful repairs | 17 | 5 | 27 | 16 | **29** |
| Reached the action limit unsubmitted | 28 | 7 | 19 | 28 | 16 |
| Invalid actions | 7 | 23 | 17 | 8 | 11 |
| Later edits that repeat an earlier edit | 64 of 84 | 25 of 28 | 25 of 52 | 62 of 82 | 25 of 51 |
| Real recoveries | 1 | 0 | 2 | 1 | 5 |
| Generated tokens | 14,061 | 11,642 | 16,329 | 18,664 | 9,781 |

A real recovery is a task solved after a public test failed on a candidate the model
itself had edited.

| Pair (sign test) | Only the first solved | Only the second solved | p |
| --- | ---: | ---: | ---: |
| **1.7B RL vs 0.6B RL** (the rule) | 17 | 5 | **0.017** |
| 1.7B RL vs 1.7B untrained | 26 | 2 | 0.000003 |
| 1.7B plain vs 1.7B untrained | 24 | 2 | 0.00001 |
| 1.7B RL vs 1.7B plain | 4 | 2 | 0.69 |
| 1.7B notes vs 1.7B plain | 4 | 15 | 0.019 |

Every submitted candidate was freshly graded again by the audit, which matched the run
on a copy and in place. No grade timed out.

## What this shows

- **The bigger model, trained the same way, is better on unseen programs.** 29 against
  17 is the first gain on a sealed test that a sign test calls unlikely by chance
  between two trained models.
- **Training does most of the work.** Untrained, 1.7B solved 5 of 48: it submitted on
  41 tasks, usually its first idea, and in 25 development episodes it submitted a
  candidate that had just failed a test. Plain supervised training alone added 24
  tasks against 2.
- **RL's effect is again unresolved.** It gained 4 tasks and lost 2 against its start
  here, and 2 against 0 on development. That is the same size as at 0.6B. It did make
  answers shorter (9,781 tokens against 16,329) and had the most real recoveries (5).
- **The written notes hurt.** The notes variant solved 16 against plain's 27 (p = 0.019),
  and lost on development too (29 against 34). It learned the form: 85 of its 132 edits
  correctly quote a failing case from the test output. But the diagnosis after the
  quote is often wrong, and 62 of its 82 later edits repeat an earlier edit, the 0.6B
  rate. A sentence that sounds like reasoning did not make the fix follow from the
  evidence.
- **Repetition fell where the model was stronger, not where it was told to reason.**
  About half of the plain and RL adapters' later edits repeat an earlier one, against
  three quarters for 0.6B RL and the notes variant.

## Development (48 tasks: pilot plus the opened first final test)

| Run | Arms and successes |
| --- | --- |
| 1. Untrained | 1.7B untrained 19; 0.6B RL 24, reproducing all 48 saved trajectories |
| 3. Variant choice | plain 34, notes 29 (10 against 5, p = 0.30): rule 1 chose plain |
| 5. Final choice | RL 36, plain 34 (2 against 0, p = 0.5): rule 2 chose RL |

Plain's rerun in run 5 reproduced all 48 of its run-3 trajectories. Development
overstated the untrained model most: 19 there, 5 on the sealed test.

## Training

| Run | Result |
| --- | --- |
| Supervised, plain | 180 steps on the 480 recovery samples; loss 0.77 to 0.001; peak 6.8 GiB; 2.1 h |
| Supervised, notes | Same samples, with notes on all 120 repair edits; loss 1.33 to 0.09; peak 6.9 GiB; 2.3 h |
| RL from plain | 6 cycles, 1,320 sampled attempts, 2,663 trained turns, 24 optimizer steps; 6 to 13 of the sampled programs gave signal per cycle |

Without gradient checkpointing, one supervised step on the longest sample ran out of
memory (13.9 GiB of 14.6); with it, the same step peaked at 6.8 GiB.

The RL audit recomputed every reward, advantage, program choice, and update payload,
decoded all 4,804 sampled turns, and freshly regraded 581 submissions. It found one
wrong reward: a sleep of this computer interrupted a grade in cycle 4, turning a correct
attempt into a failure. Its effect is described in the plan; it can only have worked
against RL.

## HumanEvalFix native, run once

These 117 tasks were written by people, not by the AI that wrote every other task here.
They were converted to the native tool format by a mechanical rule, and tasks resembling
a training program were dropped before any model ran. Reported only; nothing was chosen
on them.

| Measure | 0.6B RL | 1.7B untrained | **1.7B RL** |
| --- | ---: | ---: | ---: |
| Successful repairs | 44 | 22 | **59** |
| Reached the action limit unsubmitted | 66 | 8 | 48 |
| Invalid actions | 24 | 33 | 13 |
| Later edits that repeat an earlier edit | 166 of 189 | 36 of 51 | 107 of 155 |
| Real recoveries | 4 | 1 | 10 |

| Pair (sign test) | Only the first solved | Only the second solved | p |
| --- | ---: | ---: | ---: |
| 1.7B RL vs 0.6B RL | 28 | 13 | 0.028 |
| 1.7B RL vs 1.7B untrained | 39 | 2 | < 0.000001 |
| 1.7B untrained vs 0.6B RL | 10 | 32 | 0.0009 |

The direction matches the sealed test: the trained 1.7B is best, and training matters
more than size. HumanEval is public and widely copied, so both models may have seen
these functions in pretraining, which can inflate every column alike.

The converted sources come from Python's `ast.unparse`, whose output differs between
versions: the run used the system Python 3.9.6, which writes `for (a, b) in`, while
the admission run used 3.12, which writes `for a, b in`. The code is the same either
way. Converting again under 3.9 gave identical controls, drops, and separation results
(`artifacts/humanevalfix-native-admission-py39`), and the audit ran under 3.9.

## Evidence

| Run | Directory | Audit |
| --- | --- | --- |
| Preflight | `artifacts/scaleup-preflight-v1` | `preflight-decision.json` |
| Untrained development | `artifacts/scaleup-dev-base-v3` | `independent-verification.json`, `rerun-verification.json` |
| Supervised plain, notes | `artifacts/scaleup-sft-plain-v1`, `-notes-v1` | `independent-verification.json` |
| Variant choice | `artifacts/scaleup-dev-sft-v2` | `independent-verification.json` |
| RL | `artifacts/scaleup-rl-v1` | `rl-verification.json` |
| Final choice | `artifacts/scaleup-dev-rl-v2` | `independent-verification.json`, `rerun-verification.json` |
| Frozen record | `artifacts/scaleup-frozen.json` | |
| Second final test | `artifacts/scaleup-final-test-2-v1` | `independent-verification.json` |
| HumanEvalFix native | `artifacts/scaleup-humanevalfix-v1` | `independent-verification.json` |

Registries: `artifacts/final-test-2-registry.json` and
`artifacts/humanevalfix-native-registry.json`, both completed. The second final test is
now spent. The whole 1.7B sequence used about 22 free GPU hours over two weekly quotas,
including about 2 hours lost to the incidents in the plan. Each audit ran on a copy first, then once in place; the two agreed
on everything except grading times. The tasks are AI-authored and not independently
human-reviewed, and the audits are programmatic, not independent replication.
