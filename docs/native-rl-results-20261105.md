# RL training, comparison, and final test — November 5, 2026

**Under the rule fixed before the run, RL did not help.** On the 24 development tasks
the RL adapter solved **15/24**, against **17/24** for the recovery adapter it started
from, in the same run. Both passed **14/24** seeded tasks, the same 14.

The sealed final test was then opened once, as planned, whatever these results. There
RL solved **9/24** against **7/24** for the recovery adapter. The two splits disagree
by two tasks each way (p = 0.5 on each), so there is no reliable difference between
the two adapters.

The final test does confirm the larger result on programs no choice was tuned on:
**original 0/24, first adapter 5/24, recovery 7/24, RL 9/24**. Every trained
checkpoint scored far below its development score.

These are results with one seed and one attempt per task. Rules, method, and launch
commands are in [the RL plan](rl-plan-20261104.md), written before any RL result.

## RL training

Six cycles each sampled 4 attempts on each of the 30 RL programs (120 attempts) from
the current adapter, graded them in Docker, and ran one update on Kaggle. Every setting
matched the plan. The checkpoint is the adapter after cycle 6.

| Cycle | Successes | Programs with signal | Trained turns | Invalid calls |
| ---: | ---: | ---: | ---: | ---: |
| 1 | 39/120 | 7 | 194 | 37 |
| 2 | 44/120 | 5 | 136 | 38 |
| 3 | 44/120 | 7 | 196 | 27 |
| 4 | 46/120 | 6 | 160 | 28 |
| 5 | 51/120 | 7 | 190 | 26 |
| 6 | 52/120 | 5 | 124 | 18 |

Training reward rose from 0.325 to 0.433, and invalid calls halved. Learning was
concentrated: 13 programs were never solved in any cycle and 4 always were, so only
5 to 7 programs per cycle gave any signal. The whole run was 24 optimizer steps. As the
plan says, rising reward on the training programs is not evidence of better repair.

The audit recomputed every reward, advantage, and update payload from saved episodes,
checked the adapter chain across cycles, and decoded all 4,638 sampled turns back to
their saved text. Two infrastructure incidents interrupted the run without changing
the method. Both are described in the plan. RL used 4.65 GPU hours. Final adapter file
SHA-256: `179592c22a530e43bb8bc6d12e0baea34ed298b75ff07f5881ce1626fc576571`.

## Autonomous development comparison

All four checkpoints ran all 24 development tasks in one run under the frozen tools,
budgets, and success rule.

| Measure | Original | First adapter | Recovery | RL |
| --- | ---: | ---: | ---: | ---: |
| Successful repairs | 1/24 | 13/24 | **17/24** | 15/24 |
| Submitted on a failing test | 0 | 11 | 0 | 0 |
| Reached the action limit unsubmitted | 13 | 0 | 6 | 7 |
| Invalid actions | 115 | 0 | 2 | 0 |
| Generated tokens | 5,938 | 2,105 | 3,153 | 3,175 |

**The 72 reruns of the three earlier checkpoints matched their saved trajectories
exactly.** The RL difference therefore comes from its weights, not from run noise.

| RL against recovery | Tasks |
| --- | ---: |
| Both passed | 15 |
| Recovery only | 2 |
| RL only | 0 |
| Both failed | 7 |

RL lost two tasks. On `closed-window` it changed both bounds instead of one. That
passed both public examples but failed 28 of 196 private checks. On `running-totals`
it submitted `[sum(v) for v in values]` as an edit four times, until it ran out of
actions. That is the repetition RL was meant to reduce.

## Seeded recovery probe

Each attempt began after a scripted read, a kept-bug edit, and a failing test. **This is
not an autonomous score.**

| Measure | Recovery | RL |
| --- | ---: | ---: |
| Changed the source after the failure | 23/24 | 23/24 |
| Submitted the failing candidate | 0/24 | 0/24 |
| Passed all public and private checks | 14/24 | 14/24 |
| Reached the action limit unsubmitted | 8 | 9 |
| Invalid actions | 3 | 4 |

Both passed the same 14 tasks. The recovery adapter's rerun matched its saved seeded
trajectories 24/24.

**Decision:** "RL helped" required more autonomous development successes than the
recovery adapter **and** at least as many seeded passes. 15 is fewer than 17, so the
rule is not met.

## Final test, opened once

`scripts/run_final_test.py` recorded the opening in `artifacts/final-test-registry.json`
before uploading anything. All four checkpoints ran all 24 sealed tasks, unseeded,
under the development rules. No checkpoint was chosen on these results.

| Measure | Original | First adapter | Recovery | RL |
| --- | ---: | ---: | ---: | ---: |
| Successful repairs | 0/24 | 5/24 | 7/24 | **9/24** |
| Submitted on a failing test | 0 | 19 | 0 | 0 |
| Reached the action limit unsubmitted | 15 | 0 | 16 | 14 |
| Invalid actions | 123 | 1 | 5 | 0 |
| Generated tokens | 8,454 | 2,368 | 5,277 | 5,131 |

| Pair (sign test) | Only the first passed | Only the second passed | p |
| --- | ---: | ---: | ---: |
| Original vs first adapter | 0 | 5 | 0.0625 |
| Original vs recovery | 0 | 7 | 0.016 |
| Original vs RL | 0 | 9 | 0.004 |
| First adapter vs recovery | 2 | 4 | 0.69 |
| Recovery vs RL | 0 | 2 | 0.5 |

RL's two extra tasks, `power-of-two` and `second-largest-distinct`, were both real
recoveries: it revised after a failing test, retested, and submitted a passing repair.
Recovery training's main change also held on unseen programs. The recovery and RL
adapters never submitted on a failing test, while the first adapter did so on 19
tasks.

The one sealed task with a reviewed near pair among the RL programs, `quiet-hours`
(paired with `within-tolerance`), was failed by every checkpoint. It cannot explain any
gain.

## What this shows

- **Training transferred to unseen programs.** Recovery and RL each beat the original on
  the final test by margins a sign test calls unlikely by chance (p = 0.016, p = 0.004).
- **RL's effect is unresolved.** Across 48 development and final tasks it gained 2 and
  lost 2 against its starting point. The small budget allows a real effect too small
  to see, or none.
- **Debugging is still mostly repetition.** On the final test, 40 of the recovery
  adapter's 53 later edits repeated an earlier edit, and 38 of RL's 51. Most failures
  ran out of actions doing this.
- **The final test is now spent.** It has been seen once and can no longer serve as an
  unseen test. Comparing any future checkpoint needs a new sealed split.

The tasks are AI-authored and not independently human-reviewed. Audits are
programmatic, not independent replication.

## Evidence

| Run | Directory | Audit |
| --- | --- | --- |
| RL training | `artifacts/native-rl-training-v1` | `rl-verification.json` |
| Development comparison | `artifacts/native-rl-comparison-v1` | `independent-verification.json` |
| Seeded probe | `artifacts/native-rl-seeded-v1` | `independent-verification.json` |
| Final test | `artifacts/native-final-test-v1` | `independent-verification.json` |

Each audit ran on a scratch copy first, then once on the run itself. The two agreed on
everything except grading times. Free GPU quota used this week: 7.60 of 30 hours.
