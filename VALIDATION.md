# Validation — October 29, 2026

This is the original environment-validation record, prepared before the GPU baseline.
The subsequent model run and notebook fixes are recorded in
`docs/baseline-20261029.md`; the original measurements below remain historical evidence.
The later paired rollout, 41-test preflight, and conversation audit are recorded in
`docs/paired-revision-20261030.md`.
The subsequent native diagnostic and actual supervised adapter are documented in
`docs/native-tools-20261030.md` and `docs/native-supervised-warmup-20261030.md`.

Later split admission and dry-run evidence is recorded in
`docs/splits-and-recovery-plan-20261030.md`.

This records environment/evaluator validation. **There are no model scores or new
training results.** The machine-readable evidence is in
`artifacts/validation-final/summary.json`; raw test output is in the same directory.

## Custom development suite

- 24 authored fixtures, eight per category: validation, collections, control flow.
- 51 public cases and 3,865 private cases across the exact declared finite domains.
- **96/96** expected repair-control outcomes matched: broken, reference, alternative,
  and public-example shortcut for each task.
- **24/24** shortcuts separately passed the public examples, confirming that final
  rejection was not merely a failure to match those examples.
- The saved scripted reference episode failed its initial public test, applied a real
  edit, passed the public retest, and passed independent final checks.
- The source fingerprint was unchanged throughout the final admission run.

Evidence: `artifacts/admission-final/`, `artifacts/demo-final/`, and
`artifacts/validation-final/pilot-manifest.json`.

The 24 semantic fixtures share several algorithmic patterns; do not represent them
as 24 independent bug families or 24 unseen algorithms. All are development material,
with no training or final test split. Passing is limited to the declared domains.

## HumanEvalFix adaptation

- All **164/164 reference repairs passed** the original assertion tests.
- All **164/164 buggy versions were rejected**. Three hit the candidate CPU limit:
  `Python/10`, `Python/156`, and `Python/160`. These resource failures are recorded
  separately from ordinary assertion failures.
- All **328 expected control outcomes matched** after fixing the control reporter to
  recognize candidate resource failures as legitimate rejection of buggy programs.
- No cases were silently removed, and no model-generated patch was evaluated.
- The source fingerprint was unchanged during the final external control run.

Evidence: `artifacts/humanevalfix-controls-final/`. Dataset revision, checksum, source
URL, and MIT license are in `data/humanevalfix/`.

The protocol is a local docstrings-plus-examples adaptation, with complete-source
responses. It is not a reproduction of the authors' published results. Its assertion
tests run with the candidate in the container and must not be used as a secure RL
reward oracle. The stronger custom-task output comparison keeps expected answers on
the host. Neither boundary is a formal proof against arbitrary hostile code.

## Automated checks

**23/23 tests passed**, including nine Docker integration tests. These cover tool
budget exhaustion, frozen submission, invalid edits, signature preservation, hidden
feedback exclusion, stable manifests, missing/invalid prediction denominators,
notebook public-data boundaries, external control classification, actual execution,
type-sensitive output checking, input mutation, exceptions, network denial, host-file
isolation, infinite loops, output flooding, forged verdict fields, and external
assertion execution. One test can cover multiple assertions.

Ruff 0.17.2 lint and formatting checks passed for all 15 Python source/test/script
files. It was available from the earlier local project's environment; the host CLI
and tests themselves use the Python standard library.

Both generated notebooks passed Python syntax and public-payload checks. Their cells
contain no saved outputs. **Neither notebook has been executed on a GPU.** Inference
dependency compatibility, free GPU availability, current account quota, and model
success therefore remain unmeasured in this prototype.

## Separated splits — October 30, 2026

- **30 training and 24 final-test tasks**: 168 public and 9,680 private cases.
- **246/246** authored controls matched: broken, reference, alternative, shortcut, and
  a wrong repair for each training program. All buggy programs and wrong repairs fail
  a public test, so a demonstrated test failure is real.
- **2,648 cross-split pairs** compared. No solution matches across splits after
  identifier anonymization, no behavioral near-duplicates, and 25 near-source pairs,
  each read by hand and pinned. Injected leaks prove the checks fire.
- **88/88 tests** passed, including nine Docker tests and two scripted dry runs of the
  real runners. Ruff lint and formatting passed for `repairlab`, `scripts`, and `tests`.
- The six-task audit, re-run with the generalized verifier on a copy, reproduced every
  original field, including all 8 fresh regrades. Top-level `repairlab/*.py` is
  byte-identical to the pilot snapshot, so its source fingerprint is unchanged.

Evidence: `artifacts/split-admission-20261030-v1/`. No model ran. Neither new GPU
runtime has executed. The tasks are AI-authored and not independently human-reviewed.

## GPU runs — November 2, 2026

- **Expansion:** all 27 submitted candidates freshly regraded; source fingerprint equal
  to the pilot's. Kaggle's `torch` build and `huggingface-hub` version had changed.
- **Recovery training:** 180 steps on all 480 samples; no wrong repair as a target and
  no evaluation solution in training.
- **Comparisons:** all 82 submitted candidates freshly regraded across 96 episodes. All
  24 first-adapter trajectories matched the saved pilot and expansion trajectories.
  Sealed final-test tasks used: 0.
- **90/90 tests** passed on the final code, including Docker tests and both scripted dry
  runs. Ruff lint and formatting passed. The exact Kaggle token appears in no project
  file.
- **Dry-run gap:** the scripted Kaggle stand-in did not enforce Kaggle's 1 MB notebook
  limit, so the first training upload was rejected. `Kaggle.run` now checks size first.

These audits are programmatic, not independent human replication. All scores are on
development tasks. See [the recovery report](docs/native-recovery-comparison-20261102.md).

## GPU runs — November 4 and 5, 2026

- **RL training:** 6 cycles of 120 sampled attempts on the 30 RL programs, fixed
  configuration. Every reward, advantage, and update payload recomputed; adapter chain
  checked; all 4,638 sampled turns decoded to their saved text. No development or
  final-test task entered training.
- **Development comparison:** 96 episodes; all 72 reruns of the three earlier
  checkpoints matched their saved trajectories exactly. The seeded probe's recovery
  rerun matched 24/24.
- **Final test:** opened once and registered before upload; 96 episodes on all 24
  sealed tasks; every submitted candidate freshly regraded.
- Each audit ran on a scratch copy, then once in place; the two differed only in
  grading times.
- **102/102 tests** passed, including Docker tests and the scripted RL dry run (108/108 after the second sealed test was added). Ruff
  lint and formatting passed.
- **Dry-run gaps:** the scripted Kaggle stand-in could not reproduce a T4 out-of-memory
  error in batched sampling, or a CLI exit error after a complete download. Both were
  fixed in `scripts/` and are described in [the RL plan](docs/rl-plan-20261104.md).

See [the RL report](docs/native-rl-results-20261105.md).

## Qwen3-1.7B runs — November 5 and 6, 2026

- **Preflight:** measured memory before any task run. Without gradient checkpointing
  a supervised step ran out of memory; with it, the step peaked at 6.8 GiB.
- **Supervised training:** both variants audited for targets, notes, masks, tensors,
  and separation; no wrong repair as a target, no evaluation solution in training.
- **RL:** 6 cycles, 1,320 sampled attempts. Every reward, advantage, program choice, and
  update payload recomputed; all 4,804 sampled turns decoded to their saved text; 581
  submissions freshly regraded. One reward was wrong because this computer slept
  mid-grade; it is reported, not corrected.
- **Comparisons:** a new runtime evaluates arms on both model sizes in one notebook. Its
  0.6B rerun reproduced all 48 saved trajectories, and its plain rerun all 48 of run 3.
- **Second final test:** opened once and registered before upload; 240 episodes; all
  142 submissions freshly regraded; no grade timed out.
- **HumanEvalFix native:** 117 converted tasks, Docker controls, and training-separation
  checks before any model ran; run once.
- Each audit ran on a copy, then once in place; the two differed only in grading times.
- **137/137 tests** passed on the final code, including Docker tests and a scripted dry
  run of the whole 1.7B sequence through the second test's opening and its refusal to
  open again. Ruff lint and formatting passed. Top-level `repairlab/*.py` is unchanged.
- **Dry-run gaps:** the scripted Kaggle stand-in never executes a runtime, so it could
  not catch PEFT refusing dotted adapter names or rounding a second adapter loaded onto
  a shared base. A running notebook's fingerprint checks caught both before any
  generation. Network outages and laptop sleep were also outside the dry run.

See [the 1.7B results](docs/scaleup-results-20261106.md).

## Version and limits

Final package source SHA-256:
`b7146a77a83d26626e1e4e477d4380c79ae158b3ced5a548deb9134e81f5741c`

Custom task manifest SHA-256:
`9829fbbd7e8db746505bcd6fb95205b181107cd207c63e79b52dbb014f30c407`

Earlier `first-demo`, `admission-v1`, and `humanevalfix-controls-v1` directories are
development diagnostics, not the final admitted version. In particular, the first
custom draft had 23 fixtures, and the first external reporter incorrectly marked
three correctly rejected resource failures as control mismatches. Their original
records were retained, and final runs use the corrected implementation.

No GPU job, paid inference, fine-tuning, publication, or deployment was performed.
The prior CompanyGym and IncidentBench experiments and application data were preserved.
