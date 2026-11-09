# Native repair expansion — November 2, 2026

The original Qwen3-0.6B model completed **1/18 successful repairs**, while the frozen
supervised adapter completed **8/18** on the 18 development tasks outside the six-task
pilot. Both ran under the pilot's tools, limits, and success rule. No training occurred.

This is a development comparison with one model, checkpoint, seed, and attempt per
condition. It extends the pilot; it is not a generalization claim or an RL result.

## Results

| Task | Original | Frozen adapter |
| --- | --- | --- |
| allowed-status | Failed submitted repair | Passed |
| bounded-queue | Action limit, unsubmitted | Failed submitted repair |
| cache-expiry | Failed submitted repair | Passed |
| chunk-items | Failed submitted repair | Failed submitted repair |
| consecutive-failures | Failed submitted repair | Failed submitted repair |
| drop-missing | Failed submitted repair | Failed submitted repair |
| first-failure | Action limit, unsubmitted | Failed submitted repair |
| first-match | Action limit, unsubmitted | Failed submitted repair |
| next-scheduled-slot | Failed submitted repair | Failed submitted repair |
| optional-name | Action limit, unsubmitted | Failed submitted repair |
| page-slice | Action limit, unsubmitted | Passed |
| password-length | Failed submitted repair | Passed |
| payload-limit | Failed submitted repair | Passed |
| positive-batch | Action limit, unsubmitted | Failed submitted repair |
| rate-window | Action limit, unsubmitted | Failed submitted repair |
| remaining-budget | Passed | Passed |
| required-fields | Action limit, unsubmitted | Passed |
| running-totals | Action limit, unsubmitted | Passed |

The tasks have **38 public examples and 3,066 private checks**.

| Measure | Original | Frozen adapter |
| --- | ---: | ---: |
| Successful submitted repairs | 1/18 | 8/18 |
| Submitted attempts | 9/18 | 18/18 |
| Valid native actions | 27/110 | 76/76 |
| Valid public test calls | 0 | 22 |
| Edit calls | 18 | 18 |
| Generated tokens, all attempts | 4,388 | 1,609 |

| Paired outcome | Tasks |
| --- | ---: |
| Adapter only | 7 |
| Original only | 0 |
| Both passed | 1 |
| Both failed | 10 |

The exact two-sided sign test on the expansion alone gives p = 0.016. The descriptive
24-task total is **original 1/24, adapter 13/24**, with 12 adapter-only tasks and none
for the original. Its p = 0.0005 is descriptive because the pilot's tasks informed later
work and the two parts ran in different software environments (below). All of these
tasks are development tasks, reused for choices, and they were authored by AI without
independent human review.

## Observed behavior

**The adapter never revised after its own edit failed a test.** It made exactly one edit
per task. Four attempts tested the original bug before editing, which is not recovery.
All ten failures ended the same way: the latest public test failed and the next action
was submit. All eight successes had a passing latest test, so the test result separated
success from failure perfectly, and the model did not act on it.

The adapter's action sequence was `read → edit → test → submit` on 14 tasks and
`read → test → edit → test → submit` on four. Neither depended on what a test showed.
Its score here is therefore a measure of first-edit quality plus reliable tool use, not
of repair by iteration. The pilot's `stable-unique` failure was typical rather than
unusual, which makes recovery the right next training target.

The original model made one valid edit first on every task. After that, 83 of its 110
actions were invalid: 37 had unexpected arguments, 28 could not be parsed as an
assistant call, and 18 did not contain exactly one tool call. It never ran a test.

Post-hoc grading of its nine unsubmitted candidates found one correct (`page-slice`),
two changed but incorrect, and six identical to the buggy source. None change any
score: 1/18 is an end-to-end workflow score under the predeclared submission rule. See
`unsubmitted-candidate-diagnostic.json`. As in the pilot, the gain combines better tool
use, patch generation, and workflow completion, and cannot isolate each one.

**The pilot's 5/6 did not carry over: the adapter scored 8/18 here.** The pilot's tasks
were the first two in each category rather than a random sample, and six tasks are few.
A later rerun in the current environment reproduced all 24 adapter trajectories exactly,
including 5/6 on those tasks, so the difference comes from the tasks.

## Environment change

Kaggle's image changed after the pilot. `torch` moved from `2.11.0+cu128` to
`2.11.0+cu130` and `huggingface-hub` from 1.29.0 to 1.33.0. All pinned packages, the
model revision, adapter hashes, source fingerprint, and tasks match. The environment
was identical across all eight rounds and both conditions, so the comparison in this
run is unaffected. Only the combined 24-task total crosses the two environments. The
later adapter rerun matched every saved trajectory, so the change did not affect the
adapter's 13/24. The original model's pilot arm has not been rerun.

## Execution and audit

The run used the pilot's source fingerprint unchanged. The audit in
`independent-verification.json` confirmed:

- every raw response, parsed native action, and reconstructed public-only conversation;
- budgets, source and task fingerprints, and aggregate scores;
- unchanged adapter and base fingerprints in all eight rounds;
- notebook privacy, server source equality, and downloaded archive hashes;
- fresh local regrades of **all 27 submitted candidates**, all agreeing with saved grades.

Eight private notebook versions completed. Free GPU quota moved from **30.00 to 29.67
hours**. This is a programmatic audit, not independent human replication.

Two adapter failures, `positive-batch` and `optional-name`, failed the public tests but
passed every private check. This is expected. Their only bug-exposing inputs (`[]` and
`None`) are public, and private checks cover the rest of the finite domain. Success
requires both sets, so no incorrect candidate can score as a success this way. A scan of
all 78 tasks found the same pattern in only these two development tasks, two final-test
tasks (`uniform-values`, `power-of-two`), and one training task (`shipping-tier`). The
buggy source fails its public tests on all five. The scan ran only buggy sources
through the existing grader and involved no model.

## Evidence

Run directory: `artifacts/native-repair-expansion-20261030-v1/`.

- `experiment-plan.json`, `config.json`, and `source-snapshot/`.
- `rounds/1/` through `rounds/8/`: requests, notebooks, archives, and raw responses.
- `episodes/original/` and `episodes/supervised/`: actions, tool results, and grades.
- `report.json`, `independent-verification.json`, `development-summary.json`, and
  `unsubmitted-candidate-diagnostic.json`.
- `results.html`: expandable saved trajectories and final grades.

The verifier's file name retains the pilot's. It was generalized, and re-running it on
the pilot reproduced every shared field.
