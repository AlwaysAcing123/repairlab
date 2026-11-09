# Native autonomous repair pilot — October 30, 2026

The original Qwen3-0.6B model completed **0/6 successful repairs**, while the saved
supervised adapter completed **5/6** under the same native tools and limits. These
were autonomous attempts: no replacement solutions or requested next actions were
supplied after the initial task. The six selected repair references were excluded
from the earlier supervised training.

This is a small synthetic **development** comparison with one model, checkpoint,
seed, and attempt per condition. It establishes a working repair prototype, not
broad generalization, a statistically reliable training gain, or an RL result.
No training occurred during this experiment.

## Frozen comparison

The task rule was the first two authored tasks in each of three categories. It was
fixed before generating either model's outputs:

| Task | Original | Saved supervised adapter |
| --- | --- | --- |
| quantity-range | Failed submitted repair | Passed |
| closed-window | Failed submitted repair | Passed |
| positive-total | Action limit, unsubmitted | Passed |
| stable-unique | Action limit, unsubmitted | Failed submitted repair |
| retry-decision | Action limit, unsubmitted | Passed |
| capped-backoff | Action limit, unsubmitted | Passed |

The tasks have **13 public examples and 799 independent private checks** across
declared finite input domains. The two conditions used identical initial prompts,
native `read`, `test`, `edit`, and `submit` schemas, public execution feedback, greedy
decoding, seed 7, and the same pinned model revision:

`Qwen/Qwen3-0.6B` at `c1899de289a04d12100db370d81485cdf75e47ca`.

Each attempt allowed eight actions including submit, 512 generated tokens per call,
4,096 generated tokens per episode, and 12,288 context tokens. Invalid calls consumed
an action. No automatic submission, repair retry, best-of selection, model-output
cleanup, or evaluation-driven checkpoint selection occurred. Only an explicit
submission passing both public and private checks counted as success.

The fixed final adapter from the earlier 60-step supervised warm-up was reloaded
through PEFT from the attached private Kaggle notebook output. Its physical file
SHA-256 was checked before loading:

`2dd4773919ca4c13934d796fd0e38ed77aeb8df12055e2cd8f3b69992ac47375`.

The original model and adapter condition used separate base-model instances. Both
base parameter fingerprints matched the prior baseline, and the adapter tensor
fingerprint matched the saved checkpoint. All fingerprints remained unchanged before
and after every round. The model, tokenizer template, software, GPU, and decoding
configuration stayed constant across all eight inference rounds.

## Observed behavior

| Measure | Original | Saved adapter |
| --- | ---: | ---: |
| Successful submitted repairs | 0/6 | 5/6 |
| Submitted attempts | 2/6 | 6/6 |
| Valid native actions | 8/40 | 24/24 |
| Invalid actions | 32 | 0 |
| Valid public test calls | 0 | 6 |
| Edit calls | 6 | 6 |
| Generated tokens, all attempts | 1,550 | 496 |

The adapter followed `read → edit → test → submit` on every task. Five edits passed
all public and private checks. For stable deduplication it retained `sorted(set(values))`
and added an unreachable duplicate return. Public execution reported `[1,2]` instead
of the required `[2,1]`. The model nevertheless submitted the candidate. It did not
revise after a failing test, so successful recovery from observed failure remains
unproven.

The original model made an edit first on all tasks, then returned prose and malformed
or schema-invalid calls. Its attempted tests often supplied inputs or `public_tests`
arguments to the zero-argument test tool. Those calls were rejected and consumed
budget. Two attempts eventually submitted unchanged faulty behavior; four never
submitted before the action limit.

Post-hoc grading found the original model's unsubmitted capped-backoff candidate was
correct. It still scores as a failed episode under the predeclared submission rule.
Thus 0/6 is an end-to-end workflow score, not a claim that every original-model
candidate was incorrect. The other three unsubmitted candidates failed their checks.

The result combines better tool use, patch generation, and workflow completion.
It cannot isolate how much each contributes. This native comparison also differs
from the older single-response 8/24 baseline and custom-JSON 0/24 tool-loop experiments;
those scores are not interchangeable.

## Execution and audit

Private staged Kaggle notebooks generated one next action per unfinished attempt.
Only public conversations and checkpoint identifiers were uploaded. Model-generated
Python was executed locally in the existing restricted disposable container runtime;
expected private outputs remained with the host grader. The GPU notebook never ran
generated Python. Public tool observations excluded execution timing, and submission
returned only `{"submitted": true}` to the model.

The saved-evidence audit checked all 64 raw responses, independently extracted native
calls, reconstructed all conversations, checked exact action identities and code
fingerprints, and verified action/token budgets and aggregate scores. It also checked
notebook privacy, submitted/server source equality, downloaded archive hashes, and
the saved adapter/base fingerprints. **All eight submitted candidates were freshly
regraded locally**, with results agreeing with the original grades. Four unsubmitted
candidates were separately checked as post-hoc diagnosis without changing scores or
feeding results back; see `unsubmitted-candidate-diagnostic.json`.

All **57 local preflight tests** passed, including nine Docker integration checks.
An earlier run had a transient container startup failure in the five-second integration
fixture; the unchanged suite passed on repeat. The initial failure log is preserved.
Lint checks passed. This is a programmatic local evidence audit, not independent human
replication or a proof that the executor is secure against all adversarial programs.

The eight private [Kaggle notebook versions](https://www.kaggle.com/code/alwaysacing/repairlab-native-repair-pilot)
completed. Free GPU quota moved from **22.27 to 21.99 hours**, approximately 0.28 hours
including repeated environment startup, subject to rounded readings. Recorded
inference plus end-of-round weight hashing totaled **206.50 seconds**, excluding
initial setup/model loading. Peak allocated GPU memory was about **3.25 GiB**.
No paid inference, new optimizer steps, RL, publishing, or deployment occurred.

## Reproduction evidence

Run directory: `artifacts/native-repair-pilot-20261030-v1/`.

- `experiment-plan.json`, `config.json`, `source-snapshot/`, and preflight logs.
- `rounds/1/` through `rounds/8/`: public requests, exact uploaded/server notebooks,
  privacy metadata, original archives/logs, raw responses, parameter hashes, and
  round verification records.
- `episodes/original/` and `episodes/supervised/`: actual actions, tool results, diffs,
  final sources, submission grades, and raw model calls.
- `report.json`, `independent-verification.json`, `audit.log`, `run.log`, quota records,
  and `unsubmitted-candidate-diagnostic.json`.
- `results.html`: expandable saved trajectories and final grades.

Implementation: `repairlab/native_repair.py`, `scripts/native_repair_runtime.py`,
`scripts/build_native_repair_notebook.py`, `scripts/run_native_repair_pilot.py`,
`scripts/verify_native_repair_pilot.py`, and `tests/test_native_repair.py`.

## Immediate next steps

1. Evaluate the same frozen adapter and original model on the remaining 18 development
   tasks under the same rules. Preserve these six results; report the six-task pilot
   and subsequent expansion separately before any combined descriptive total.
2. Design program/group-separated training, development, and untouched final-test
   sets. Preserve task admission controls and prevent repair/reference leakage.
3. Add supervised recovery demonstrations using training programs: failed test,
   revised patch, retest, and submission. Compare against this checkpoint on fixed
   development tasks without training on their solutions.
4. Once reliable mixed-success repair behavior and separate evaluation sets exist,
   run a small reward-based training experiment. Compare original, supervised, and
   RL checkpoints with matched budgets. Keep a final test unused until choices freeze.

HumanEvalFix remains available for a separate external evaluation; its assertion
adapter and protocol differ from this custom finite-domain grading environment.
