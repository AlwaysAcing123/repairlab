# Separated splits and recovery experiment plan — October 30, 2026

This record was written **before** any of the runs below, so their rules cannot be
fitted to their results. Nothing here is a model score. Status, launch paths, and the
launch note were updated afterward; the rules were not.

## Status

| Step | State |
| --- | --- |
| 1. Frozen 18-task expansion | **Done:** original 1/18, adapter 8/18. See [the report](native-repair-expansion-20261102.md). |
| 2. Separated splits | **Done.** 246/246 authored controls matched; see `artifacts/split-admission-20261030-v1/`. |
| 3. Recovery training and comparison | **Done:** rules met; autonomous 13/24 to 17/24. See [the report](native-recovery-comparison-20261102.md). |
| 4. RL | **Done:** rule not met; development 15/24 against recovery's 17/24, final test 9/24 against 7/24. See [the RL report](native-rl-results-20261105.md). |

The GPU steps need a Kaggle token file, kept outside the project.

## Splits

| Split | Tasks | Public | Private | Status |
| --- | ---: | ---: | ---: | --- |
| Training | 30 new programs | 91 | 3,861 | Targets may enter training |
| Development | 24 existing tasks | 51 | 3,865 | Reused for choices; not held out |
| Final test | 24 new programs | 77 | 5,819 | **Opened once on November 5; no longer unseen** |
| Final test 2 | 48 new programs | 144 | 17,997 | **Opened once on November 6; no longer unseen.** See [its record](final-test-2-20261105.md) and [the 1.7B results](scaleup-results-20261106.md). |

Development is deliberately unchanged, so the pilot and warm-up audits stay valid and
`pilot_tasks()` keeps its manifest. Training and final-test programs are new, balanced
across the same three families and four bug categories. Each training program also has
one plausible wrong repair that fails a public test. The eight toy warm-up programs
remain training material; the four warm-up development programs stay development.

All new code lives in `repairlab/curriculum/`. `source_fingerprint()` covers only
top-level `repairlab/*.py`, so the expansion carries the **same source fingerprint as
the pilot**, which is stronger evidence of a frozen comparison than a claim would be.

### Separation checks

`leakage_report()` compares all 2,648 cross-split pairs, including the 12 warm-up programs. Identical solutions after
identifier anonymization are a hard failure. Near-identical source, specification
overlap, and behavioral agreement across shared inputs are listed for human review.
Solutions are compared because they are what a model could memorize; a trivial buggy
line such as `return a - b` appearing twice is a review item, not a leak.

Result: **no hard failures, no behavioral near-duplicates, 25 near-source pairs.** All
25 were read individually and share only a short skeleton. They are pinned in
`REVIEWED_NEAR_PAIRS`, so any new near pair fails tests until someone reads it. Tests
inject a renamed copy and a differently written equivalent to prove the check fires.

This review caught one real problem. Draft training program `window-sums` used
`range(len(values) - width + 1)` over sliced windows, which is almost exactly the
*bug* in development task `chunk-items`. It was replaced by `nth-from-end`.

Broad patterns still cross splits: a list filter is a list filter. Separation here
means no shared program, renamed copy, or mutation, not absence of shared technique.

### The seal

`reject_final()` runs in every model-facing path and rejects sealed tasks by ID,
specification, buggy source, and reference. Comparing content matters:
`Task.fingerprint()` covers the ID, so a renamed sealed task passed an earlier draft of
this guard until a test caught it. `run_matched_comparison.py` has no task selector.
The final test should be opened once, after every checkpoint and rule is frozen.

## Recovery training

**Problem measured so far:** one episode, `stable-unique`. The adapter kept the bug,
saw a failing test, and submitted. Whether that is typical is what the expansion and
the seeded probe below will show.

**A format mismatch in the first adapter.** It trained under a diagnostic system
prompt, an instructed user prompt ("Inspect, test, edit, retest, then submit"), and
test observations lacking `status` and `mutated_input`. Autonomous evaluation uses
none of those. It worked anyway, but the new data matches evaluation exactly: prompts
come from the shared `transcript.py`, and every observation comes from a real `Episode`
against the real executor.

**Wrong repairs are never targets.** Each sample supervises one next action. Wrong
edits appear only as prior history, so the model learns to respond to a failure, not
to write one. A test pins this and `verify_recovery_training.py` re-checks it.

Each program yields 16 next-action samples, 480 in total:

| Trajectory | Actions (`*` = history only) |
| --- | --- |
| direct | read, edit(fix), test, submit |
| recover-retained | read\*, edit(kept bug)\*, test, edit(fix), test, submit |
| recover-wrong | read\*, edit(wrong)\*, test, edit(fix), test, submit |
| recover-twice | read\*, edit(kept bug)\*, test\*, edit(wrong)\*, test, edit(fix), test, submit |

The kept-bug edit reproduces the observed failure shape: the faulty source plus a
redundant final line. `recover-twice` uses all eight actions, which is the real budget.

**Configuration, fixed now:** continue from the frozen first adapter, LoRA unchanged,
AdamW learning rate 0.0001, 180 optimizer steps, 8 samples per step (three passes over
the data), seed 7. Micro-batch 1 because the first run peaked at 8.9 GiB at about 780
tokens and these histories are roughly twice as long on a 15 GiB T4. The **final step
is the checkpoint**; steps 60 and 120 are saved only as recovery evidence.

## Planned runs, in order

1. **Expansion.** Original versus first adapter on the remaining 18 tasks under the
   frozen rules. Report it apart from the six-task pilot, then a descriptive 24-task
   total with `report_development_comparison.py`.
2. **Recovery training** as configured above.
3. **Autonomous comparison.** First adapter versus recovery adapter on all 24
   development tasks under the same rules. First-adapter reruns double as a determinism
   check against saved trajectories.
4. **Seeded recovery probe.** Same arms and tasks. Each attempt starts after a scripted
   read, kept-bug edit, and failing test, with the remaining five actions autonomous.
   This measures recovery on every task instead of only where the model happens to
   fail. It uses only the buggy source, so no solution is revealed. **It is not an
   autonomous repair score** and is reported separately.

## Decision rules, fixed before results

Recovery training **worked** only if the seeded probe shows more `revised_after_failure`
than the first adapter **and** autonomous successes do not fall on the 24 tasks.
A seeded gain with an autonomous loss is a failed trade, not a partial win.

RL may start only when all of these hold. Otherwise fix the blocking item first.

1. The recovery adapter revises after a failed test on at least half of seeded tasks.
2. Autonomous development success is between 20% and 85%. RL needs mixed outcomes: all
   failures give it no reward signal, and near-perfect scores leave no gain to measure.
3. Training has enough unseen programs left to avoid reusing development tasks.

RL comparisons use original, supervised, and RL checkpoints under equal budgets, on
development tasks until choices freeze. The final test opens once, at the end. Binary
independent correctness stays the initial reward. A better training reward alone is
not evidence of better repair on unfamiliar programs, and no gain is assumed.

## What the dry run does and does not show

`tests/fake_kaggle.py` scripts model responses from authored references. The dry run
exercises the real runners, real Docker grading, real batch hashing, and the real
audits on 36 expansion and 48 seeded episodes. It does **not** execute either GPU
runtime. For those, the evidence is narrower: every attribute they use also appears in
the runtimes already proven on a T4, and the one semantic change is
`PeftModel.from_pretrained(..., is_trainable=True)`.

## Launch

```sh
TOKEN=/absolute/path/to/kaggle-token
KAGGLE=.venv/bin/kaggle

python3 -B scripts/run_native_repair_pilot.py --task-set remaining \
  --output artifacts/native-repair-expansion-20261030-v1 \
  --token-file "$TOKEN" --kaggle "$KAGGLE"
python3 -B scripts/verify_native_repair_pilot.py \
  artifacts/native-repair-expansion-20261030-v1 --regrade
python3 -B scripts/report_development_comparison.py \
  --pilot artifacts/native-repair-pilot-20261030-v1 \
  --expansion artifacts/native-repair-expansion-20261030-v1 \
  --output artifacts/native-repair-expansion-20261030-v1/development-summary.json

python3 -B scripts/run_recovery_training.py \
  --output artifacts/native-recovery-training-v2 --token-file "$TOKEN" --kaggle "$KAGGLE"
.venv/bin/python -B scripts/verify_recovery_training.py artifacts/native-recovery-training-v2

python3 -B scripts/run_matched_comparison.py --recovery-run artifacts/native-recovery-training-v2 \
  --output artifacts/native-recovery-comparison-v1 --token-file "$TOKEN" --kaggle "$KAGGLE"
python3 -B scripts/verify_matched_comparison.py artifacts/native-recovery-comparison-v1 --regrade

python3 -B scripts/run_matched_comparison.py --recovery-run artifacts/native-recovery-training-v2 \
  --seeded --output artifacts/native-recovery-seeded-v1 --token-file "$TOKEN" --kaggle "$KAGGLE"
python3 -B scripts/verify_matched_comparison.py artifacts/native-recovery-seeded-v1 --regrade
```

The first training attempt, `artifacts/native-recovery-training-v1`, never reached a
GPU: Kaggle rejected its 1.95 MB notebook because notebooks must be under 1 MB. Training
data is now compressed inside the notebook, and `Kaggle.run` checks the size before
upload. Data, configuration, and rules are unchanged. The failed attempt is preserved,
and the retry uses a new directory, `-v2`.

Expected free-quota use is about 2 GPU hours in total, from the pilot's 0.28 hours for
12 episodes and the warm-up's 0.17 hours. Do not edit `repairlab/` during a run:
`finish()` rejects any source change. Every run directory must be new.
