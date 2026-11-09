# Recovery training and comparison — November 2, 2026

**Under the rules fixed before these runs, recovery training worked.** The recovery
adapter revised after a failed test on **23/24** seeded tasks against **1/24** for the
first adapter, and autonomous successes rose from **13/24 to 17/24**. It never submitted
on a failing test in either comparison.

Revising is not yet debugging. Most gains came from better first edits, and the attempts
that ran out of actions mostly repeated earlier edits. RL may not start yet: all three
gate conditions must hold, and one does not. No RL ran.

These are development results with one checkpoint, seed, and attempt per task. The
autonomous gain alone is not statistically reliable (p = 0.125).

## Training

Training continued the frozen first adapter on 480 next-action samples from 30 new
training programs for 180 steps, as configured before the run. The final step was the
checkpoint. No wrong repair was a target, and no evaluation solution entered training.

| Measure | Value |
| --- | ---: |
| First-step loss | 0.0948 |
| Final-step loss | 0.00004 |
| Longest sequence | 1,639 tokens |
| Peak GPU memory | 11.6 GiB |
| Training time | 2,191 seconds |
| Free GPU quota used | 0.64 hours |

Loss near zero suggests the model memorized these programs, which is why only the
results on unseen tasks below count. Final adapter file SHA-256:
`d7e7194d9623137a60d170dc3fdebe3957d2454f8352a08ea455addfab192853`.

## Autonomous comparison

Both adapters ran all 24 development tasks under the pilot's tools, budgets, and success
rule.

| Measure | First adapter | Recovery adapter |
| --- | ---: | ---: |
| Successful repairs | 13/24 | **17/24** |
| Submitted on a failing test | 11 | **0** |
| Reached the action limit unsubmitted | 0 | 6 |
| Invalid actions | 0 | 2 |
| Generated tokens | 2,105 | 3,153 |

| Paired outcome | Tasks |
| --- | ---: |
| Both passed | 13 |
| Recovery only | 4 |
| First adapter only | 0 |
| Both failed | 7 |

The first adapter's rerun matched **all 24 saved trajectories exactly**, despite the
Kaggle software change noted in the [expansion report](native-repair-expansion-20261102.md).
Its 13/24 is therefore reproducible.

## Seeded recovery probe

Each attempt began after a scripted read, a kept-bug edit, and a failing test, leaving
five autonomous actions. **This is not an autonomous repair score.**

| Measure | First adapter | Recovery adapter |
| --- | ---: | ---: |
| Changed the source after the failure | 1/24 | **23/24** |
| Submitted the failing candidate | 23/24 | **0/24** |
| Passed all public and private checks | 1/24 | **14/24** |
| Reached the action limit unsubmitted | 0 | 8 |
| Passed public, failed private | 0 | 2 |

Paired outcomes were 1 both passed, 13 recovery only, 0 first adapter only, and 10
neither (sign test p = 0.0002).

## What actually improved

**Three of the four autonomous gains were not simple recovery from a bad first edit.**

| Gained task | What happened |
| --- | --- |
| `first-match` | Better first edit; no revision needed |
| `consecutive-failures` | Better first edit; no revision needed |
| `drop-missing` | First edit passed, then an unneeded edit broke it, then it recovered |
| `next-scheduled-slot` | Real recovery: first edit failed, revision passed |

`cache-expiry` passed under both, but the recovery adapter's first edit failed and took
two revisions. The new data taught better first edits as well as recovery, and the
experiment cannot fully separate the two.

**Revising is often not debugging.** All six autonomous candidates left at the action
limit were wrong. Five of those attempts repeated an earlier edit. On `stable-unique` the
model wrote one edit four times. On `rate-window` it kept summing timestamps instead of
counting them, rewriting the same logic while the test output showed the wrong result.

**The failure often does not change the fix.** Across 23 tasks, the model's first edit was
identical with and without a seeded failure on **15**. Of the 14 seeded successes, 13
needed one edit. When its first attempt was wrong, it rarely found the fix later.

**Hidden tests mattered.** `first-failure` passed its public tests but only 9/29 private
checks, autonomously and in the probe. `closed-window` passed its public tests in the
probe but only 168/196 private checks.

## Metric limit

`revised_after_failure` counts any source-changing edit after a failed test. In
autonomous runs that includes a normal first edit after testing the untouched bug, which
accounts for the first adapter's two. The seeded probe avoids this. Even there, the
metric shows a source change, not a meaningful one.

## Decision

| Fixed rule | Result | Met |
| --- | --- | --- |
| Seeded revisions exceed the first adapter's | 23 vs 1 | **Yes** |
| Autonomous successes do not fall | 17 vs 13 | **Yes** |
| RL gate 1: revises on at least half of seeded tasks | 23/24 | **Yes** |
| RL gate 2: autonomous success from 20% to 85% | 71% | **Yes** |
| RL gate 3: enough unseen training programs remain | 0 of 30 | **No** |

Recovery training worked, and the 71% rate leaves mixed outcomes for RL to learn from.
Gate 3 blocks RL: all 30 training programs entered recovery training, and the near-zero
loss suggests RL on them would give little signal. Reusing development tasks is ruled
out, so RL needs new training programs under the same admission and leakage checks.

## Execution and audit

The audits confirmed every conversation, action, budget, checkpoint hash, notebook,
and archive across both comparisons. They freshly regraded **all 82 submitted
candidates** (42 autonomous, 40 seeded), all agreeing with saved grades. Sealed final-test
tasks used: **0**. Post-hoc grades of the six unsubmitted autonomous candidates are
diagnostic and change no score. Task authorship limits are as in the expansion report.

The first training attempt never reached a GPU; see [the plan](splits-and-recovery-plan-20261030.md).
The whole sequence, including the expansion, used **1.52 free GPU hours**.

## Evidence

- `artifacts/native-recovery-training-v2/`: bundle, training log, audit, and adapter.
- `artifacts/native-recovery-training-v1/`: the rejected first attempt, preserved.
- `artifacts/native-recovery-comparison-v1/`: autonomous comparison and post-hoc grades.
- `artifacts/native-recovery-seeded-v1/`: seeded probe.

Each comparison includes `results.html`, `report.json`, and `independent-verification.json`.
