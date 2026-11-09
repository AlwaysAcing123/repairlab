# RepairLab project log (chronological)

Adapt an open-weight model to repair Python programs using execution feedback, and
measure improvement on independent problems under matched conditions. The name is
provisional. The first specialty is small Python backend helper functions.

## What exists now

- **24 original development tasks**, eight each for validation, collection processing,
  and control-flow helpers. Each has a specification, buggy implementation, public
  examples, independent grading inputs, reference repair, and alternative repair.
- A bounded **read → test → edit → submit** interface, with eight actions by default.
  Edits replace one complete function, at most 8 KiB. Invalid actions consume budget.
- A container executor and a host-owned grader. The model sees public test feedback;
  it never gets private grading results during the episode.
- Task admission controls, saved trajectories, readable diffs, a small HTML replay,
  version fingerprints, and a model-output scoring command.
- An uploadable notebook for a **single-response, original-model feasibility baseline**
  on free Kaggle or Colab. It contains only public task information.
- A pinned snapshot of all **164 Python HumanEvalFix tasks**, with its MIT license,
  an external assertion-test adapter, control checks, and a separate inference notebook.

**First actual baseline: 8/24 repairs passed (33.3%)**, using the original Qwen3-0.6B
in one response without execution feedback on October 29, 2026. See
[the baseline report](baseline-20261029.md) for controls, failure categories,
cost, and evidence. That baseline involved no fine-tuning or RL. Authored controls and scripted
reference replays remain labeled separately from actual model results.

**October 30 paired tool-loop result: 0/24 without execution feedback and 0/24 with
feedback.** All 48 final candidates remained the buggy inputs; invalid actions and
premature submission dominated. See [the paired report](paired-revision-20261030.md).
This is a tool-use readiness finding, not evidence that feedback is generally useless.

**Native-tool diagnostic: 11/12 valid requested calls**, with all four native
serializer/parser round-trips passing. Three generated source arguments matched the
requested functions and executed correctly on checked inputs. One malformed opening
tag remains. These instructed synthetic probes are not an autonomous repair score.
The predeclared 12/12 readiness gate was not met at that stage, prompting the supervised
warm-up below. See [the native diagnostic](native-tools-20261030.md).

**First actual supervised adapter completed:** 60 LoRA steps on 64 next-action
examples from eight new training programs. Requested tool calls improved from
**12/16 to 16/16** on four separate development programs and **11/12 to 12/12** on the
earlier diagnostic. Saved adapter tensors, masked labels, training/evaluation
separation, and raw outputs were independently audited. These are tool-use checks,
not autonomous repair scores or RL gains. See [the supervised report](native-supervised-warmup-20261030.md).

**Native autonomous repair pilot completed: original 0/6, saved supervised adapter
5/6.** The adapter independently read, edited, tested, and submitted each task. It
made no invalid calls; the original made 32. All eight submitted candidates were
freshly regraded, and checkpoint/conversation evidence was audited. The remaining
failure submitted despite a failed public test, so recovery behavior is still missing.
This is a six-task development pilot with no new training or RL. See
[the repair pilot report](native-repair-pilot-20261030.md).

**Separated splits exist.** 30 new training programs and a **sealed 24-task final test**
were authored in `repairlab/curriculum/`. All 246 authored controls matched, no solution
crosses a split, and no model saw the final test until it was opened once on November 5. See
[the plan](splits-and-recovery-plan-20261030.md), which fixed the rules and the RL
gate before any later result existed.

**Expansion completed: original 1/18, first adapter 8/18** on the other 18 development
tasks. All ten adapter failures submitted on a failing test. Across 24 tasks the
adapter scored 13/24 and reproduced exactly when rerun. See
[the expansion report](native-repair-expansion-20261102.md).

**Recovery training worked under the fixed rules: 13/24 rose to 17/24 autonomously**, and
revisions after a seeded failure rose from 1/24 to 23/24. It never submitted on a failing
test. Most gains still came from better first edits, and failed attempts often repeat an
edit, so this is not yet real debugging. These are development results. See
[the recovery report](native-recovery-comparison-20261102.md).

**RL did not help under the fixed rule: 15/24 against the recovery adapter's 17/24** on
development tasks, with the same 14/24 seeded passes. Six RL cycles on 30 new programs
raised training reward from 0.325 to 0.433. **The sealed final test, opened once
afterward, scored original 0/24, first adapter 5/24, recovery 7/24, RL 9/24.** Training
transferred to unseen programs, but RL against recovery is two tasks each way across
both splits, which is no reliable difference. Repeated edits still dominate failures.
See [the RL report](native-rl-results-20261105.md).

**A second sealed test of 48 new tasks** was built for the Qwen3-1.7B comparison. All
192 authored controls matched, and it is separated from all 120 earlier programs. See
[its record](final-test-2-20261105.md).

**Qwen3-1.7B counts as better under the rule fixed before it ran.** On that second test,
opened once, the final 1.7B checkpoint (supervised, then RL) solved **29/48** against
**17/48** for the 0.6B RL adapter (17 against 5 tasks only one solved, p = 0.017).
Untrained, 1.7B solved 5/48. RL against its supervised start is again unresolved (4
against 2). A variant trained to write a diagnosis before each fix did worse (16/48):
it quotes the failing test correctly but still repeats its edits. On 117 human-written
HumanEvalFix tasks in the same tool format, run once, the final 1.7B solved 59 against
44 for 0.6B RL (p = 0.028). The second test is now spent. See [the 1.7B results](scaleup-results-20261106.md) and
[the plan](scaleup-plan-20261105.md).

## First look

Use Python 3.9+ and a running Docker Desktop/Engine. The host package has no external
runtime dependencies; commands below run directly from this directory.

```sh
python3 -B -m repairlab list
make test
make integration-test
make admit
make demo
```

`make demo` saves an `episode.json` and `replay.html` under a new `artifacts/demo-*`
directory. Open the HTML in a browser. It shows real executions of a scripted repair,
including the initial failure, edit, public retest, and independent final grade.
It is prominently labeled **SCRIPTED REFERENCE / SAVED REPLAY / NO MODEL INVOLVED**.

If the pinned Python image is absent, pull it once:

```sh
docker pull python:3.12-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70
```

Every run gets a new directory; existing evidence is never intentionally overwritten.
The first local admission and demonstration were saved as development diagnostics.
Use the final validation record for the admitted dataset version.

## Run the first model baseline

1. Open `notebooks/RepairLab_Baseline.ipynb` in Kaggle or Colab as a private notebook.
2. Select one free GPU, enable model-download access, and check your live quota.
3. Run the cells in order. There are 24 prompts, one response per prompt, and at most
   1,024 new tokens per response. No paid API or training is involved.
4. Download and extract the output archive. Generated Python stays text in the notebook;
   it is not executed beside cloud credentials or the model.
5. Score the archive locally:

```sh
python3 -B -m repairlab score-patches \
  --predictions /absolute/path/to/extracted/predictions.jsonl \
  --metadata /absolute/path/to/extracted/model-config.json \
  --output artifacts/baseline-001
```

The notebook uses `Qwen/Qwen3-0.6B` at the exact revision already used by CompanyGym,
as a **feasibility candidate**, not a final choice of coding model. It uses greedy
decoding, thinking disabled, seed 7, and a fixed token budget, and records packages,
GPU, dtype, raw responses, elapsed generation time, and peak GPU memory.

The first baseline has **no tools or execution feedback**. A paired tool-loop rollout
is now implemented using staged private Kaggle batches: generate one next action per
unfinished episode, apply actions locally, then upload only public conversations for
the next round. There are no callbacks or credentials in these batch notebooks.
Do not compare the two protocols as if their budgets were identical. The HumanEvalFix
notebook is structurally checked but has not run on a GPU.

The paired runner is `scripts/run_batch_revision.py`, with its public-data notebook
builder in `scripts/build_batch_revision_notebook.py` and audit in
`scripts/verify_batch_revision.py`. Both arms use identical parsing, prompts, decoding,
and budgets. `--allow-json-fences` accepts one enclosing JSON code fence; the original
strict-format diagnostic is preserved separately. Read the paired report before
launching more runs. First establish reliable tool use, then measure supervised
warm-up separately from subsequent RL. That first supervised warm-up has now been
measured; the next experiment should load its saved adapter into a small native
autonomous repair loop before starting RL.

The score report keeps all 24 tasks in the denominator, including missing generations
and invalid patches. Infrastructure failures stop the run and are recorded separately.
Per-task episodes link the patch, public information, and final grade to the aggregate.

## What the evaluation means

Each task has four admission controls:

| Candidate | Expected outcome |
| --- | --- |
| Starting broken implementation | Fail |
| Reference repair | Pass |
| Alternative repair | Pass |
| Public-example hard-coding with `None` elsewhere | Fail |

The shortcut is also checked separately to confirm that it actually passes all public
examples. These controls validate the authored tasks and evaluator; they are not model
repair attempts. They are initial checks, not comprehensive validation of every possible
incorrect or alternative patch.

The grader compares actual JSON outputs against protected expectations, with output
types preserved (`True` does not equal integer `1`). It rejects input mutation,
exceptions, malformed output, resource failures, and unsubmitted episodes. Hidden
expected values and reference solutions for these custom tasks never enter the
execution container. The external HumanEvalFix adapter has a different test boundary,
described below.

For each task we enumerate every input in a **small declared finite domain**, excluding
public examples from the private set. `python3 -B -m repairlab list` reports the exact
domain and counts. Passing establishes agreement on those inputs; it does not prove
correctness over all Python values or establish production readiness.

All 24 tasks are **development material**. We intentionally have not invented a training
or final-test split from this pilot. Related mutations or renamed functions must stay
in one program group when the next datasets are authored. Reusing these development
tasks after training will not produce a held-out result. Program groups here are authored
semantic groups, not a guarantee that the tasks have no shared algorithmic patterns.

## Execution boundary

Each test or submission starts a fresh unprivileged Python container with no network,
a read-only root filesystem, dropped Linux capabilities, no privilege escalation,
128 MiB memory, one CPU, 32 processes, a 3-second CPU limit, a 12-second wall-clock
limit, bounded host-captured output, and an 8 MiB temporary filesystem. Only the worker
file is mounted. There is no project mount, Docker socket, home directory, credential,
reference implementation, or expected-answer file in a custom-task candidate container.

The host sends candidate source and inputs, and independently compares returned values.
Candidate stdout does not determine the grade. Containers are forcibly removed after
each invocation, including timeouts. Tests exercise network denial, absent host files,
input mutation, exceptions, loops, output flooding, and forged verdict fields.

This is a restricted container boundary using the host's Docker trust model, not a
formally verified defense against hostile code or container-runtime vulnerabilities.
Do not mount secrets or expand candidate permissions to make a failed model pass.

## Existing benchmarks and next experiment

[HumanEvalFix](https://github.com/bigcode-project/octopack) is integrated as an
**explicit local adaptation**: Python, docstrings plus public examples, complete-source
responses, and the authors' original assertion tests. A pinned snapshot, source URL,
checksum, revision, and license live under `data/humanevalfix/`.

```sh
python3 -B -m repairlab external-admit --output artifacts/humanevalfix-controls-001
python3 -B scripts/build_baseline_notebook.py --humanevalfix
```

Run `notebooks/RepairLab_HumanEvalFix.ipynb` on a free GPU to generate patches, then:

```sh
python3 -B -m repairlab external-score \
  --predictions /absolute/path/to/extracted/predictions.jsonl \
  --metadata /absolute/path/to/extracted/model-config.json \
  --output artifacts/humanevalfix-baseline-001
```

Prompting, source extraction, helper imports, timeout, and execution environment differ
from the authors' official harness. **These are adapted scores, not a reproduction of
published leaderboard scores.** Tests use their own Python assertion semantics, rather
than the custom tasks' type-sensitive JSON comparison. Public examples can overlap
the external final tests, as in the source dataset.

The external assertion suite executes with candidate code inside the container; a
deliberately adversarial candidate could inspect or interfere with in-process tests.
It is a conventional non-adversarial repair check, **not our RL reward oracle**. Use
the stronger host-owned output oracle for initial RL. Do not train on HumanEvalFix
test tasks or claim that public data was absent from a model's pretraining. Report
reference-control failures and already-passing broken programs rather than silently
filtering tasks to improve results. The full 164-task denominator is preserved.

Use the initial baseline to choose a useful difficulty range. Author separate training
programs and freeze a new evaluation set. Then measure supervised training separately
from supervised-plus-RL, with the same model interface, prompts, and inference budgets.
Use binary independent correctness as the initial terminal RL reward. If supervised
warm-up is unnecessary, an RL-only feasibility run can be measured separately.

Keep actual weight-change evidence, checkpoint hashes, task/source versions, trajectories,
model revisions, and compute use. A better training reward alone is not evidence of
better repair on unfamiliar programs. No positive learning outcome is assumed.

## Relationship to the earlier projects

`companygym` remains the source of prior training/notebook/checkpoint lessons, including
the need to distinguish supervised gains from incremental RL gains. Its simulator,
email tasks, adapters, and scores have not been relabeled as coding results.

`incidentbench-agentrollout` remains the realistic connection-leak case and independent
application verifier for a later integration experiment. Its application/database are
not used by this prototype or touched by its tests.

This directory starts a small standalone implementation rather than copying those
whole systems. See `PROVENANCE.md` for authorship and reuse, and `VALIDATION.md` for
what was actually exercised.
