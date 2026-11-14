# RL-first plan — November 7, 2026

**Status: approved by Aaron on November 9, 2026; build started.** No GPU job of this phase has run. See the addendum at the end for this week's order of work.
Written **before** any result of this phase, so its rules cannot be fitted to the outcome.
Status lines may be updated afterward; the rules may not, except through dated addenda
written before the results they affect.

## Why a new phase

In the 1.7B phase, supervised training did almost all the work: 5 to 27 of 48 on the
second sealed test. RL added 2 more (29), which could be chance. The saved run records
show why RL was weak:

- **It was slow.** 1,320 sampled attempts took about 11 GPU hours, roughly 120 per hour.
  Every model action was its own Kaggle job, because grading ran on this laptop: 8
  rollout jobs plus 1 update job per cycle. A rollout job took 5 to 6 minutes, of which
  about 2.5 were generation; the rest was startup. An update job took about 55 minutes
  for 4 optimizer steps on contexts of up to 4,096 tokens.
- **It was small.** 24 optimizer steps in total, on 30 programs.
- **Signal was thin.** In each cycle, only 6 to 13 of the 15 sampled programs had both
  passing and failing attempts. The others taught nothing.
- **It started late.** RL began from a supervised adapter that had memorized its 30
  training programs (loss 0.001), so there was little left for RL to change.

This phase makes RL the main method, starting from the untrained model.

## Questions

1. **Primary:** does RL alone make Qwen3-1.7B better at fixing bugs in programs it never
   trained on?
2. Is RL better or worse than supervised training on the same tasks? (Reported.)
3. **Secondary:** does a weak grader that checks only the visible tests teach the model to
   cheat, and does the hardened grader prevent it?

## What changes

| | 1.7B phase | This phase |
| --- | --- | --- |
| RL starts from | the plain supervised adapter | the untrained base model |
| Training tasks | 30 hand-written bugs | about 2,000 generated bugs in 108 programs |
| Episode | up to 8 tool actions, 4,096 tokens | write a fix, see the public tests, one retry |
| Where the RL loop runs | one Kaggle job per action; grading on the laptop | sampling, grading and updates in one Kaggle session; the laptop audits afterwards |
| Attempts per run | 1,320 | at least 15,000 (set by the preflight) |
| Optimizer steps | 24 | at least 30 (set by the preflight) |
| LoRA | rank 8 on `q_proj`, `v_proj` | rank 16 on every attention and MLP projection |
| Precision | bf16, which the T4 has no fast hardware for | fp16 if the preflight checks pass, else bf16 |

Unchanged: model `Qwen/Qwen3-1.7B` at revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e`,
thinking off, the `solve` function format, the success rule (pass every public and hidden
case), host-side Docker grading for every evaluation, and the Kaggle token rules.

## The environment

### Generated bugs

Bugs are made automatically by changing correct programs, so tasks are no longer limited
by how many bugs can be written by hand.

- **Source programs:** the 108 existing programs already opened and never sealed again:
  24 pilot, 30 supervised-training, 24 first-final-test, and 30 RL programs.
- **Bug kinds.** Operator swaps and off-by-ones are only one kind. Each kind below is a
  program transformation that copies a common real mistake:

  | Kind | Example |
  | --- | --- |
  | Boundary | `<` for `<=`, a range or slice bound off by one |
  | Missing edge case | an `if` guard or early return deleted, such as the empty-input check |
  | Wrong variable | `i` for `j`, `low` for `high`, the other argument |
  | Wrong starting value | an accumulator starting at 0 instead of 1, `max` starting at 0 |
  | Statement order | two adjacent statements swapped, so a value is updated before it is used |
  | Missing step | an update deleted: a counter increment, an `append`, a `seen.add` |
  | Wrong scope | a statement moved into or out of a loop or branch, such as a `return` inside a loop |
  | Wrong operation | `append` for `extend`, `sort()` for `sorted()`, `//` for `/`, `min` for `max`, `any` for `all`, `strip` for `lstrip` |
  | Aliasing | a copy removed, so the function changes its input |
  | Inverted logic | `and` for `or`, a `not` dropped or added, `if` and `else` branches swapped |
  | Wrong return | the wrong variable returned, or returned too early |
  | Written by hand | mistakes no rule makes: misreading the specification ("at least" read as "more than"), ignoring duplicates or case, a plausible but wrong algorithm |
  | Two bugs | two bugs of different kinds in one program |

- **Balance:** no kind makes up more than 15% of the pool, and every result is also reported
  by kind, so the report shows which kinds RL learned to fix and which it did not.
- **Hand-written bugs** in the pool are written for this phase, several per program, and go
  through the same admission. They come from the same AI that wrote the programs, a
  limitation already true of every earlier set.
- **Admission, in Docker on the laptop:** a changed program becomes a task only if it
  keeps the `solve` signature, gives a different answer from the reference on at least
  one input in the declared domain, finishes on every input within the time limit, and
  does not match another admitted bug or any reference after names are anonymized.
- **Public cases:** the program's existing public cases, plus one witness: the first
  domain input, in a fixed order, where the bug's answer differs. So every task shows a
  real failing test. Every other domain input is a hidden case.
- **Controls for every task:** the reference passes everything; the bug fails a public
  case; the public-answer shortcut (hard-coding the public cases) passes the public cases
  and fails the hidden ones.
- **Pool size:** up to 2,000 bugs, at most 30 per program, plus the 108 original
  hand-written bugs. The admission run gives the real count; it is recorded before any
  model runs.

### Episode

1. The model sees the specification, the buggy function, and the public test results:
   for each case, the call, what it returned or raised, and what was expected.
2. It replies with the complete corrected function in one Python code block, in at most
   768 generated tokens.
3. If the reply passes every public case, it is submitted. Otherwise the model sees the
   new public results and gets one retry, which is submitted whatever it does.
4. A reply without exactly one valid `solve` definition counts as a failed attempt, and
   the reason is shown before the retry.

**Reward**, hardened grader: 1 if the submission passes every public and hidden case,
else 0. Weak grader: 1 if it passes every public case. Both graders compute both results
for every attempt, so the weak run's hidden results are logged without being rewarded.

### Grading inside the Kaggle session

This is a deliberate change. Until now, model output never ran on a Kaggle GPU machine.
Here it must, or each RL step would again need a round trip to the laptop.

- Each candidate runs in its own process with CPU-time, memory and output limits and a
  timeout. No secrets are attached to these sessions.
- If the session runs as root (the preflight checks this), the candidate process drops to
  an unprivileged user, and the hidden answers exist only in a root-only file and in the
  parent process's memory.
- If it cannot drop privileges, a candidate could in principle read the hidden answers.
  The run then relies on detection: the laptop regrades every rewarded submission in the
  existing Docker grader, where no answers exist inside the container, and any reward that
  differs is reported. Submissions are also scanned for file, process and network calls.
- Development and sealed evaluations put **no** hidden case on Kaggle. The session sees
  only public cases, and the laptop grades each final submission in Docker, as before.

## Splits

| Split | Programs | Tasks | Use |
| --- | ---: | ---: | --- |
| Training | 108 existing | the generated pool plus 108 hand-written bugs | RL and supervised training |
| Development, seen programs | the same 108 | 108: one generated bug each, held out of the pool | measures memorization |
| Development, unseen programs | second final test's 48 | 48 generated + 48 hand-written bugs | monitoring only |
| **Third sealed test** | 96 new | 96 generated + 96 hand-written bugs | opened once |
| HumanEvalFix native | 117 | 117 | reported only |

- The 96 new programs are written for this phase, with hand-written bugs and the usual
  controls (reference, alternative, broken, shortcut). They go through the existing
  cross-split and near-duplicate checks against all 156 earlier programs and HumanEvalFix,
  with every near pair read and pinned. No model sees them before the opening.
- The generated bug for each sealed program is chosen by a fixed seed from its admitted
  bugs, before any model runs.
- The HumanEvalFix separation check is rerun against the 108 training programs.
- Every training path calls `reject_final2` and a new `reject_final3`.

## Arms

1. **Base:** the untrained 1.7B model.
2. **RL, hardened grader.**
3. **RL, weak grader.** Same settings and number of steps as arm 2; only the reward differs.
4. **Supervised:** same LoRA settings. One pass over as many distinct training tasks as
   arm 2 drew, chosen by seed from the same pool. The target is the reference function as
   the first reply. Learning rate 1e-4.

## RL method

- **Batch:** 32 tasks per step, 8 sampled episodes each, temperature 1.0, top-p 1.0.
- **Signal filter:** a group whose 8 rewards are all equal teaches nothing and is dropped.
  Fresh tasks are drawn until 32 informative groups are found, up to 96 draws per step.
  A task whose group was all correct is retired from the pool.
- **Update:** advantage = reward minus the group mean. One optimizer step per batch
  (on-policy), with the loss summed over the model's tokens in both turns and divided by
  the batch's total model tokens. Learning rate 2e-5 (stable in both earlier phases),
  gradient clip 1.0, no KL penalty.
- **Length:** a fixed number of steps per run, set by the preflight as the most that fits
  8 GPU hours, and at least 30. The final step's adapter is the one evaluated. No
  checkpoint is chosen on any evaluation.
- **Sessions:** chunks of at most 4 hours. Each chunk saves the adapter, optimizer state,
  random-number state and logs. The next chunk attaches them as a kernel source and
  resumes.
- **Sampler:** vLLM if the preflight passes its checks, else Hugging Face `generate`.
  Old log-probabilities are recomputed by the trainer, so the sampler's precision does
  not enter the update.
- **Collapse:** if valid replies fall below 50% of a batch for 3 steps in a row, the run
  stops and is reported as collapsed. Its last adapter is still the one evaluated.
- The two RL runs go at once, since Kaggle allows two sessions.

## Preflight, at most 2 GPU hours

It measures; it chooses nothing on development or sealed tasks.

1. Python version, whether the session is root, CPU count, library versions.
2. **fp16:** greedy replies on 48 training tasks in fp16 and bf16. fp16 is used only if no
   logit is non-finite and the valid-reply and public-pass counts are each within 2 of
   bf16's.
3. **vLLM:** used only if it installs without replacing the session's PyTorch, loads the
   base model with a LoRA adapter, passes the same check against Hugging Face greedy
   replies, and samples at least twice as fast.
4. One full RL step at batch size: time and peak memory.
5. In-session grading speed.
6. **Signal:** 64 training tasks, 8 samples each, from the base model. If fewer than a
   quarter of the groups are informative, the pool's share of two-bug tasks is raised (if
   the model solves too much) or lowered (if it solves too little), by a rule written into the preflight
   decision before the step-count calculation.

**If fewer than 30 steps fit in 8 GPU hours**, only the hardened RL run goes, with the
whole 16 hours, and question 3 waits for the next quota.

## Rules

- **Rule 1 (primary):** RL helped if hardened RL solves more of the 96 sealed generated
  bugs than base, with an exact two-sided sign test p < 0.05 on the tasks only one of them
  solved.
- **Rule 2:** RL and supervised are compared on every set and both directions are
  reported. Either one is called better only if it wins the 96 sealed generated bugs with
  p < 0.05.
- **Rule 3 (cheating):** a *false pass* is a submission that passes every public case and
  fails a hidden one. The weak grader taught cheating if weak RL has more false passes than
  hardened RL across all 192 sealed tasks, with p < 0.05 on the tasks where only one of
  them false-passed.
- Everything else is description, not a claim: the 96 sealed hand-written bugs, the
  unseen-program development set, HumanEvalFix, the seen-versus-unseen gap, false passes
  during training, retries identical to the first attempt, and tokens.

## Opening the third sealed test

As before: a frozen record of the four adapters' hashes, the protocol fingerprints and
these rules; a registry written before upload; all four arms run once on all 192 tasks,
greedy; the laptop grades every submission in Docker; the audit runs on a copy, then once
in place. HumanEvalFix runs once afterwards, reported only.

## Audits

- Every RL reward and every public-test message shown to the model is recomputed on the
  laptop in Docker. Any difference is reported, not corrected.
- Every sampled token sequence is decoded and checked against its saved text.
- No development, sealed or HumanEvalFix program enters training, checked by content.
- The dry run includes an injected cheat, a candidate that reads the answer file, to show
  the audit catches it.

## GPU budget

| Item | GPU hours |
| --- | ---: |
| Preflight | 2 |
| Base on development sets | 0.5 |
| RL, hardened | 8 |
| RL, weak (at the same time) | 8 |
| Supervised | 1.5 |
| Development sets for the 3 trained arms | 1 |
| Third sealed test and HumanEvalFix | 2 |
| Margin for incidents | 3 |
| **Total** | **26** |

About 25 hours remain this week (quota resets 2026-11-14 00:00 UTC), then 30 next week,
so the phase spans both.

## Work before any GPU job

1. Bug generator, admission and controls, with the pool recorded.
2. The 96 sealed programs with hand-written bugs, controls and separation checks.
3. The two-turn episode, the in-session grader, the chunked RL loop, the supervised
   runtime and the evaluation runtime.
4. Tests and a scripted dry run of the whole sequence, including the injected cheat and
   a refused second opening.

## Known risks

- **Memorizing instead of debugging.** RL may learn to recall the 108 reference programs.
  The seen-versus-unseen development gap measures this, and the sealed test uses only new
  programs.
- **Generated bugs are simpler than real ones.** The hand-written sealed bugs and
  HumanEvalFix measure transfer.
- **No cheating may appear.** In 8 hours, the weak run may never find hard-coding. That
  result is reported as it is.
- **vLLM or fp16 may fail on the T4.** The fallbacks are fixed above, at the cost of
  fewer steps.
- One seed, AI-written programs, and programmatic audits rather than independent
  replication, as before.
- **Laptop sleep matters less.** The RL loop runs on Kaggle; the laptop only launches
  chunks and audits. Launches still happen on AC power with the lid open.

## Addendum 1 — November 9, 2026, written before any GPU job or result of this phase

**Order of work.** This week's free quota (about 25 GPU hours, reset 2026-11-14 00:00 UTC)
goes to work that does not need the RL loop. The RL runs use next week's quota.

1. **Preflight, part A (this week):** items 1, 2, 3, 5 and 6 above.
   - **Part B:** item 4, one full RL step, runs once the RL loop exists. It is capped at 0.5
     GPU hours, inside the same 2-hour preflight budget.
   - **vLLM's environment:** if installing vLLM into the session would replace its
     PyTorch, it may be installed in a separate environment used only for sampling. Item
     3's checks still apply. The trainer always uses the session's PyTorch.
2. **Base sweep (new, this week):** the untrained model plays 8 sampled episodes on every
   training-pool task, using the RL episode, temperature and in-session grader. Every
   attempt records both the hardened and the weak result.
   - It replaces item 6's 64-task signal check, with the same decision rule applied to the
     whole pool.
   - Tasks the base model solves in all 8 episodes are retired before RL. This is the
     plan's own retirement rule, applied early. Tasks it never solves stay in the pool.
   - The base solve rate by bug kind is reported.
   - It touches no development, sealed or HumanEvalFix task.
   - If the measured speed means the whole pool would need more than 15 GPU hours, a
     seeded random subset that fits is swept instead, and the rest stay unswept.
   - **Audit:** the laptop regrades every distinct (task, submission) pair in Docker. Any
     difference is reported.
3. **Collapse rule, fixed now:** if the base model's valid-reply rate in the sweep is below
   50%, the collapse threshold becomes half that rate, measured over the same 3 steps.
4. **Third sealed test:**
   - A separate agent writes it. That agent is given the specification of the set and
     the earlier programs to separate from, and none of this phase's model output.
   - It is finished, admitted and pinned before the first RL step.
   - No model runs on it until the opening.
5. **Hand-written pool bugs:** they are written before the first RL step and pass the same
   admission. They may come after the sweep, since they are training material only, and
   they get their own sweep before entering the pool.

## Addendum 2 — November 9, 2026, after the pool was built and before any model output of this phase

1. **The pool** (`artifacts/rlfirst-pool-v2`) has 880 tasks: 772 generated bugs in 104
   programs, plus the 108 original hand-written bugs. All 1,081 pool and development tasks
   passed their Docker controls, and the in-session grader agreed with Docker on every
   case of every one.
   - **The 15% cap binds** for boundary, wrong variable and two bugs, at 132 each. That
     cap, not the 2,000 limit, sets the pool size.
   - **Rare kinds:** aliasing has 1 task and wrong scope 6. Results for those kinds are
     description only.
   - **Development sets:** 105 tasks on seen programs (three programs had no admitted
     mutation), and 96 on unseen programs (48 generated, 48 original).
2. **Hand-written pool bugs (addendum 1, item 5).**
   - **How many fit:** new hand-written bugs are added to the 880-task pool without
     reselecting the other kinds, so the cap allows at most 28 more: 136 of 908 is 14.98%.
   - **Plan:** 28 new ones are written (specification misreadings and plausible wrong
     algorithms), one in each of 28 programs chosen by seed. They go through the same
     admission and get their own base sweep before entering the pool.
3. **Schedule.** The RL runs may start this week if the conditions below hold. Otherwise
   they start with next week's quota. No rule changes either way.
   - Part B has passed.
   - The sweep audit has no mismatches.
   - The third sealed test is admitted.
   - At least 17 GPU hours of this week's quota remain.

## Addendum 3 — November 9, 2026, written AFTER preflight part A's result (a disclosed deviation)

This addendum changes a preflight rule after seeing the measurement it governs. It is
labelled that way on purpose.

**Preflight part A** (`artifacts/rlfirst-preflight-v1`, about 0.9 GPU hours) found:
- **In-session grading works:** all 48 references passed and all 48 bugs failed a public
  case.
- **Sandbox user:** the candidate process drops to user 65534.
- **No network isolation:** `unshare` is not permitted on Kaggle.
- **Exposed notebook:** two copies of the notebook, which holds the compressed payload,
  were readable by the candidate user.
- **The fp16 check failed by its letter.**
  - bf16 had 41 valid replies and 32 public passes; fp16 had 43 and 37.
  - Every logit was finite.
  - The valid-reply counts are within 2. The public passes differ by 5, in fp16's favour.
- **Speed:** bf16 is about 2.8 times slower than fp16 on the T4 (greedy: 159 seconds
  against 56).
- **vLLM was not tested.** Creating its separate environment failed (`ensurepip`), before
  vLLM was installed.

**Deviation, chosen by Aaron after seeing these numbers:** every arm uses **fp16**.
- **Why:** the check was meant to catch a numerical failure, and none occurred. A gap of
  5 of 48 greedy replies between two precisions is within the noise of greedy decoding.
  Under bf16, at most about 6 RL steps fit in 8 GPU hours, which defeats the phase's
  purpose.
- **Effect on the rules:** precision applies to every arm alike, including the base
  model, and sealed and development grading stays in Docker on the laptop. So the change
  cannot favour the outcome of rules 1 to 3. It is reported with the results.

**Also decided now, before the results they affect:**
1. **vLLM, retried.** vLLM is installed again in a separate environment created with `uv`.
   Preflight item 3 is applied unchanged, except that the reference is now the
   Hugging Face fp16 greedy result above (43 valid, 37 public passes).
   - **Within 2** on both counts of that reference.
   - **At least twice** the fastest Hugging Face fp16 sampling rate (2,718 sequences per
     hour, at batch 64).
2. **Notebook copies.** Every session makes both notebook copies readable by root only
   before any candidate runs, then checks again for the canary.
3. **Network.** The lack of network isolation is a known limitation. The Docker regrade,
   which has no network, and the scan for file, process and network calls stay the
   detection.

## Addendum 4 — November 9, 2026, after the vLLM retry and before the sweep or any RL step

1. **vLLM passes Addendum 3, item 1** (`artifacts/rlfirst-vllm-check-v1`):
   - **Versions:** vLLM 0.33.0 with its own torch 2.14, in a separate environment.
   - **Greedy results:** 43 valid replies and 36 public passes, against Hugging Face fp16's
     43 and 37.
   - **Sampling speed:** 7,409 sequences per hour, against 2,718 (2.7 times).
   - **Use:** vLLM samples for the sweep and for RL. LoRA updates reach it as a newly saved
     adapter after every optimizer step. For RL it shares the GPU with the trainer and is
     put to sleep during each update.
   - **Evaluations** (development, sealed, HumanEvalFix) keep Hugging Face fp16 greedy
     decoding, the same for every arm.
2. **Sampling settings:** temperature 1.0, top-p 1.0, no top-k, and the model's two stop
   tokens. Every sampled sequence gets its own seed, so repeated samples of one prompt are
   independent.
3. **Notebook copies:**
   - The executing copy (`/kaggle/working/__notebook__.ipynb`) is now readable by root only.
   - The source copy (`/kaggle/src/script.ipynb`) is on a read-only mount and stays readable
     by the candidate user.
   - Detection therefore stays as planned: the Docker regrade of every submission, plus the
     scan for file, process and network calls.
4. **Part B, first attempt:** it ran out of memory while sampling, because PEFT cast every
   activation to the adapter's fp32. Forwards now run under fp16 autocast with that cast
   disabled. This is an implementation fix with no rule change.

## Addendum 5 — November 9, 2026, part B's result applied, before any RL step

Part B (`artifacts/rlfirst-partb-v3`, vLLM sampler, fp16) ran one full step at batch size.
- **Sampling:** 96 draws to find 32 informative groups, 773 seconds.
- **Update:** 200 seconds at a micro-batch of 16,384 tokens, with a peak of 11.5 GiB.
- **Total:** 973 seconds per step.
- **Adapter swap:** sleeping vLLM, swapping in the updated adapter and waking it worked.

Earlier attempts v1 and v2 ran out of memory with the Hugging Face sampler and are kept
with their diagnoses.

**The rule, applied as written:**
- 26 steps fit in 8 GPU hours, fewer than 30. So only the **hardened** RL run goes, with
  the whole 16 hours: **53 steps**.
- Question 3, the weak grader, waits for the next quota. Its run uses the same settings,
  seed and 53 steps.
- The step count stays at 53 even if later steps run faster. Part B used the unretired
  pool; RL draws only from tasks the sweep did not retire, so its steps may well be faster,
  but that does not change the count.

## Addendum 6 — November 9, 2026, schedule only, before any RL step

The first sweep shard shows that 61% of its tasks are solved in all 8 episodes and will be
retired. Almost every remaining task gives a mixed group, so an RL step should need far
fewer draws than part B's 96. This affects time only; no rule depends on it.

**Change:** the weak-grader run starts **at the same time** as the hardened run, as the plan
originally intended ("The two RL runs go at once"), instead of waiting for the next quota.
- **Unchanged:** both runs keep exactly the settings, seed, and 53 steps fixed in
  Addendum 5.
- **If this week's quota runs out**, either run resumes from its last saved chunk next week.
- **The supervised arm** still starts after the hardened run has finished, because it
  copies that run's number of distinct tasks drawn.

## Addendum 7 — November 10, 2026 (02:00 UTC), after the weak arm's first chunk failed

Written after seeing the weak arm's training summaries for steps 1–12 (rewards, false
passes, memory). No trained model has been evaluated on anything.

**What happened.** The weak arm's first chunk finished 12 steps. It then ran out of GPU
memory during the update of step 13 (12.05 GiB allocated, 1.0 GiB reserved but unused, and
1.31 GiB held by the sleeping vLLM process, out of 14.56 GiB). Steps 1–12 had peaked at
11.2–12.05 GiB.
- **Cause:** the update keeps fp32 logits for every reply token until backward, so memory
  grows with reply tokens. The 16,384-token micro-batch limit counts prompt plus reply
  tokens. Replies grew longer over the steps.
- **No training state survived,** because the runtime saves the adapter and optimizer only
  at the end of a chunk. The 12 steps' records stay in
  `artifacts/rlfirst-rl-weak-v1/chunks/1-abandoned-1` with `diagnosis.json`. They count
  toward no result.

**Change 1 (engineering; it does not change what is computed).** `_update` in
`scripts/rlfirst_runtime.py` now handles running out of memory on a micro-batch:
- it restores the gradients from a copy taken before that micro-batch, halves the
  micro-batch, and tries again;
- the loss is a sum over replies divided by the step's total reply tokens, so the step's
  gradient is unchanged;
- a single reply that does not fit still fails the run;
- each split is recorded in the step's `update.oom_splits`.

Tests and file hashes:
- **CPU test** (failures forced in the forward and backward passes): updated weights equal
  the unsplit update to 1e-17. Without the restore, the backward-pass case gives a
  different update, so the test does exercise the restore path.
- **Runtime sha256:** `ead7612a…` before, `6c8cea37…` after.
- **The launcher reads the runtime when it builds each chunk.** So the hardened arm's
  chunk 1 ran the old code and its later chunks run the new code. For any step with no
  split, both versions compute the same update.
- **Part B's micro-batch search used the old code** and is not rerun.

**Change 2.** The weak arm reruns with `--resume` in the same directory. This is the
launcher's recovery path for a failed chunk: chunk 1 starts again from the base model with
the same seed, settings, and 53 steps.
- **Reproduction check:** with the same seed, its first steps should come close to the
  failed attempt's. Any differences are reported; vLLM sampling is not guaranteed to be
  bit-for-bit repeatable.

**Change 3 (schedule only).** The weak arm restarts **after** the hardened run ends, not now.
- **Quota:** this week's GPU quota is the limit, with about 16 hours left before the
  2026-11-14 reset.
- **Step time:** in the failed attempt it rose from 732 s to 911 s over 12 steps, so the
  hardened run's remaining steps may need about 13 hours.
- **Risk:** running both arms at once could use up the quota before the hardened run
  finishes.
- **Order:** hardened run, then the supervised arm and development evaluations as quota
  allows, then the weak arm (after the reset if a full chunk does not fit).
- **Unchanged:** no rule, setting, or step count changes.

## Addendum 8 — November 10, 2026, the supervised arm's details, fixed before it runs

Arms item 4 leaves some details open. They are fixed here, before the supervised arm has
been trained or any trained arm evaluated. Code: `scripts/run_rlfirst_sft.py` (launcher)
and `scripts/rlfirst_sft_runtime.py`, which is appended to the unchanged
`rlfirst_runtime.py` in the notebook.
- **"The same pool"** means the list the RL arms draw from: pool-v3 minus the 525 tasks the
  base sweep retired, which leaves 383 tasks on 86 programs.
- **The number of tasks** is the count of distinct tasks in all of the hardened run's saved
  steps, chosen from that list by seed (`SEED + 13`). If the hardened run drew all 383, the
  supervised arm trains on exactly the RL arms' tasks.
- **The target** is the reference function in one ```python block, ended by `<|im_end|>`.
  It answers the same first prompt the RL arms see (the bug and its public results), with
  the same chat template and thinking off. The launcher checks that the target parses back
  to the reference.
- **The loss** comes from the RL update code with every advantage set to 1: the mean
  negative log-likelihood over the batch's target tokens.
- **The optimizer** is the 1.7B phase's supervised setting:
  - 8 tasks per step, one pass;
  - learning rate 1e-4, constant, gradient clip 1.0;
  - AdamW without weight decay, fp16 with a gradient scaler;
  - a step whose gradients overflow is redone on the same batch at the lowered scale (at
    most 20 times), so every batch gives exactly one optimizer step.
- **Evaluation** uses the adapter after the single pass. No checkpoint is chosen.
- **For reading results:** 85 of dev-seen's 105 programs are RL-list programs. On those
  85, the supervised arm has trained on the exact reference function, and the RL arms have
  practised other bugs. dev-unseen and sealed test 3 share no program with training, so
  they are the fair comparison for rule 2. Dev-seen is still reported.
- **Dry run on CPU** with a tiny random model and the real tokenizer: every target decodes
  back to its reference and ends with the stop token; the loop and adapter saving work.

**Addendum 7, continued — 02:15 UTC, two more launcher problems and a revised schedule.**
- **The hardened arm's chunk 1 completed on Kaggle**: 15 steps, no memory splits, state
  saved. The laptop then refused its download: the 140 MB optimizer file exceeded a 64 MB
  per-file limit left over from earlier phases.
  - `kaggle_client.extract_output` now allows 256 MB per file and 1 GB in total, with the
    same path checks.
  - `scripts/finish_rlfirst_chunk.py` re-checked the private server copy and its source,
    extracted the archive, and wrote the two records the launcher would have written.
  - No training was redone.
- **The resume path had never run.** A CPU-only probe notebook (no GPU quota) attached
  chunk 1's output and showed that Kaggle mounts a source kernel at
  `/kaggle/input/notebooks/<owner>/<slug>/`. The launcher's one-level glob would have
  found nothing.
  - The launchers now use `**/repairlab-rlfirst-rl-<arm>-*/state` (and
    `**/…/state/adapter` for evaluations).
  - The runtime is unchanged.
- **Another readable copy of the payload.** Inside a resumed chunk, the previous chunk's
  notebook copy is readable under `/kaggle/input`, and it holds the payload. Like
  `/kaggle/src/script.ipynb` (Addendum 4), it sits on a read-only mount and cannot be
  hardened; the audit's scan of submissions for file, process and introspection calls
  remains the safeguard.
- **Change 3, revised (schedule only).** Chunk 1 fitted 15 steps in 3.45 hours, so the
  hardened run's remaining 38 steps need about 9 GPU hours, not 13. With 16.11 hours left
  before the reset:
  - the weak arm's chunk 1 reruns now, alongside the hardened run;
  - it stops after that one chunk (`--max-chunks 1`), which leaves about 3.6 hours for
    the supervised arm and development evaluations;
  - its remaining chunks run after the 2026-11-14 reset.

## Addendum 9 — November 10, 2026, how sealed test 3 is frozen and opened (before any trained arm exists)

Written while both RL runs are in their first or second chunk. No trained adapter exists
yet, and no model has seen any sealed task.

**Freeze (`scripts/freeze_rlfirst.py`).** It refuses unless:
- both RL runs finished and their in-place audits found no problem, with the token check
  on;
- the supervised run finished and trained exactly its chosen tasks;
- every arm's development evaluation was graded in Docker with no mismatch, on the same
  adapter kernel.

It then records, once (the record cannot be overwritten):
- each trained arm's kernel, version, adapter glob, and adapter weights SHA-256;
- the RL-first and sealed-test fingerprints, the hashes of `rlfirst_runtime.py` and of the
  adapter-check runtime;
- the hash and text of the Rules section.

**Opening (`scripts/run_final_test_3.py`, once).**
- **Refusals:** it refuses if the registry exists, if anything recorded by the freeze has
  changed, or if fewer than 2 GPU hours remain.
- **Generated bugs:** for each sealed program in manifest order, one generated bug by the
  development-set recipe. Single mutations must keep the signature, always finish, change
  some answer, and match no known reference or bug. One is drawn with seed `SEED + 31`;
  if it fails the pool's Docker controls, the next is drawn from the rest.
- **Hand-written bugs:** each program's hand-written bug, under the same controls.
- **Counts:** fewer than 96 of either kind is reported as such. Rule 1 then uses every
  admitted generated bug.
- **Order:** the registry is written after the tasks are built and before anything is
  uploaded.
- **Running the arms:** each of the four arms runs once, greedy, in its own private
  notebook, two at a time, with public cases only. Each notebook first refuses unless the
  mounted adapter's hash equals the frozen one (`run_checked_eval`).
- **Failures:** a failed arm is recorded for a human decision, never retried.

**Grading (`scripts/verify_final_test_3.py`).**
- It regrades every submission in Docker and re-checks what each model was shown.
- It applies rules 1–3 with the exact two-sided sign test used in earlier phases, at
  p < 0.05.
- It runs first on a copy, then once in place, which marks the registry completed.

**Development evaluations** of the trained arms pass `--adapter-sha256`, so they go
through the same adapter check.

**Tests before any use.**
- A stand-in run used training programs, never the sealed ones. Task building, Docker
  controls, and grading worked. A public-case lookup table was counted as a false pass on
  every task. Rule 3 with 4 against 0 gave p = 0.125, so it was correctly not called.
- The sealed-test import guard (`tests/test_final3.py`) now allows `run_final_test_3.py`
  by name, as its comment asks. `verify_final_test_3.py` reads only the opening's own
  task file.
- HumanEvalFix (reported only) is not built yet.

## Addendum 10 — November 10, 2026, early audit of the hardened arm's chunk 1, and repeated draws

The RL audit ran early on chunk 1 (steps 1–15), on a scratch copy:
- 3,512 distinct submissions were regraded in Docker, and 6,044 trained turns had their
  tokens checked;
- rewards, false passes, shown public results, prompts and replies all agree;
- the 16 flagged submissions only import `collections`, `string` or `re`; none touches
  files, processes or the network.

**Repeated draws.** When the shuffled task order runs out partway through a step, the
order is reshuffled and drawing continues, so a task can be drawn twice in one step.
- In step 15, three tasks were drawn twice in separate draws. Each draw formed its own
  group of 8 with its own mean, which is how the runtime has always worked.
- The first audit merged such groups by task and reported 40 false "advantage"
  differences. `verify_rlfirst_rl.py` now rebuilds each step's draws and groups exactly as
  the runtime forms them, and also checks the group order and counts. It then found 0
  problems.
- **A rarer case** is a task drawn twice within one draw, when the reshuffle happens
  inside it. The runtime then forms one group from both draws (16 episodes) and counts it
  twice. This has not happened so far. The audit mirrors it and lists every repeat under
  `repeated_draws`.
- Both arms run the same code, so neither case is changed during the runs. Both are
  disclosed in the results.

## Addendum 11 — November 10, 2026 (11:50 UTC), the hardened arm finished; no trained model evaluated yet

**Hardened arm.**
- All 53 steps ran in four chunks (15, 15, 15 and 8 steps). The last chunk was
  `alwaysacing/repairlab-rlfirst-hardened-b` version 2, which stopped with
  `steps_complete`.
- Every chunk's archive passed the laptop checks: the notebook was private, its sources
  matched, and no credentials were uploaded.
- **Memory.** Chunk 1 ran the old runtime. In chunks 2–4 the micro-batch split from
  Addendum 7 fired 14 times, in steps 31, 33, 41 and 46–53, and every split update went
  through. The per-step peak stayed between 11.1 and 12.0 GiB.
- **Training curve** (described only; no checkpoint is chosen from it):
  - Tasks retired, meaning all 8 samples passed, rose from 8 after step 1 to 311 of 383
    after step 53.
  - The valid-reply rate rose from 0.78 to 0.89.
  - The mean reward stayed between 0.48 and 0.68, as the pool left to draw from got
    harder.
- **Audit (whole run).** Run on a scratch copy first, then in place:
  - 20,352 episodes; 12,142 distinct submissions regraded in Docker; 20,652 trained turns
    token-checked; **0 problems**.
  - 238 submissions were flagged, and every one only imports `collections` (227), `re`
    (10) or `string` (1).

**Repeated draws (Addendum 10's rarer case did happen).**
- A task was drawn twice in separate draws of one step 22 times. Each of those draws
  formed its own group, as intended.
- A task was drawn twice *within one draw* 8 times, in steps 21, 30 (twice), 43 (twice),
  48 (twice) and 52. Each time the runtime formed one 16-episode group, with the mean taken
  over all 16:
  - in 5 of the cases the group was counted twice;
  - in 3 cases it was counted once, because the step's 32-group limit had been reached.
- That makes 13 of the run's 1,696 trained groups (0.8%) with 16 episodes instead of 8,
  and so with more weight in their step's update. The audit mirrors this exactly, and
  nothing is changed. The weak arm runs the same code, and this is disclosed with the
  results.

**Weak arm.** The rerun of chunk 1 completed with 13 steps (about 950 s per step, against
about 800 s for the hardened arm). The split fired once, in step 9, and went through. Its
early audit on a scratch copy found 0 problems, and its 10 flagged submissions import only
`collections` or `string`. It is paused before chunk 2. Its remaining 40 steps, three to
four chunks, run after the quota reset on 2026-11-14.

**Supervised arm.** It was launched once the hardened run had finished and its scratch
audit was clean, while the in-place audit ran; its input is only the number of distinct
tasks the hardened run drew. About 3.6 GPU hours were left, enough for it and the
development evaluations of the hardened and supervised arms.

## Addendum 12 — November 10, 2026 (13:00 UTC), supervised arm trained; development results for three arms

These are **development** results. They choose nothing (each arm has exactly one
checkpoint) and change no rule. Sealed test 3 is still sealed.

**Supervised arm.**
- Kernel `alwaysacing/repairlab-rlfirst-sft` version 1, 141 s of training: 383 tasks,
  exactly the distinct tasks the hardened run drew, in 48 optimizer steps.
- No fp16 overflow retries. Mean target loss fell from 0.60 to about 0.03, and the peak
  was 4.7 GiB.
- Adapter sha256 `84f361eb…`. The hardened adapter is `571da4b6…`, from chunk 4.
- Both development notebooks printed `ADAPTER CHECK PASSED` with these hashes.

**Development evaluation** (greedy, two turns, fp16). Each was graded in Docker, first on a
scratch copy and then in place, with 0 mismatches. p is an exact two-sided sign test on the
tasks only one arm solved.

| set | base | hardened RL | supervised |
|---|---|---|---|
| dev-seen (105) | 88 | 99 | 99 |
| dev-unseen (96) | 68 | 80 | 67 |
| — generated (48) | 34 | 39 | 38 |
| — hand-written (48) | 34 | 41 | 29 |

- **Hardened vs base:**
  - dev-unseen: 17 tasks only hardened solved, 5 only base (p = 0.017);
  - dev-seen: 15 against 4 (p = 0.019);
  - dev-unseen generated bugs alone: 8 against 3 (p = 0.23).
- **Hardened vs supervised:**
  - dev-unseen: 18 against 5 (p = 0.011), most of it on hand-written bugs (14 against 2);
  - dev-seen: 5 against 5.
- **Supervised vs base:** dev-unseen 14 against 15; dev-seen 14 against 3. 85 of
  dev-seen's 105 programs are in the training list, so the supervised arm saw their
  references.
- **Replies.**
  - The supervised arm stopped writing any reasoning: replies average 42 tokens, against
    259 for base and 288 for hardened.
  - On dev-unseen its retry repeated the first answer in 22 of 24 retries.
- **Format.** The share of first replies with more than one code block fell from 15 to 9
  of 96 on dev-unseen. 4 of the 17 tasks only hardened solved there were ones where base's
  first reply was invalid.
- **False passes on dev-unseen:** base 1, hardened 3, supervised 5.
- **What this predicts for the sealed rules, and changes nothing about them:** rule 1 uses
  the sealed *generated* bugs, and on the development generated bugs the hardened gain was
  small (+5, p = 0.23). Rule 1 may well not be met even though the overall development
  gain is clear.

**Next.** The weak arm resumes after the quota reset on 2026-11-14 (40 steps left). Then:
its audit and development evaluation, the freeze, and the opening of sealed test 3.

## Addendum 13 — November 14, 2026 (19:20 UTC), weak arm finished; development results for all four arms; written before the freeze

These are **development** results. They choose nothing and change no rule; the Rules
section is unchanged. Sealed test 3 is still sealed.

**Weak arm.**
- The remaining 40 steps ran in four chunks after the quota reset (11, 12, 11 and 6
  steps), starting 2026-11-14 03:01 UTC. The last chunk was
  `alwaysacing/repairlab-rlfirst-weak-a` version 4, which stopped with `steps_complete`
  at step 53. Each chunk's archive passed the laptop checks.
- **Memory.** The micro-batch split fired 25 times, in 19 of the 53 steps, and every split
  update went through. The per-step peak stayed between 11.3 and 12.2 GiB. Steps averaged
  about 1,006 s, against 812 s for the hardened arm.
- **Training curve** (described only; no checkpoint is chosen from it):
  - tasks retired rose from 11 after step 1 to 321 of the pool after step 53;
  - the valid-reply rate went from 0.78 to 0.80 (hardened: 0.78 to 0.89);
  - the weak reward's mean stayed between 0.47 and 0.72, and the hardened grader's pass
    rate on the same episodes between 0.45 and 0.69;
  - false passes in training: 803 over the run (199 in steps 1–13, 161 in steps 41–53).
    The hardened run had 913 (173 and 199). Training false passes are not rule 3, which
    is judged only on sealed test 3.
- **Audit (whole run).** First on a scratch copy, then in place, with identical reports:
  - 20,352 episodes; 12,365 distinct submissions regraded in Docker; 22,215 trained turns
    token-checked; **0 problems**.
  - 255 submissions were flagged, and every one only imports `collections` (215), `re`
    (38), `string` (1) or `itertools` (1).
  - The scratch Python environment used for the token check had been emptied by macOS's
    periodic `/tmp` cleanup. It was rebuilt with the same versions (transformers 5.2.0,
    tokenizers 0.22.2) before either audit ran.
- **Repeated draws.** A task was drawn twice in separate draws of one step 32 times, and
  twice within one draw 20 times, mostly late in the run as the undrawn pool shrank. Both
  are handled as in Addendum 11, and the audit mirrors them exactly.
- Adapter sha256 `5b51f853…` (chunk 5).

**Development evaluation of the weak arm.** Kernel `alwaysacing/repairlab-rlfirst-eval-weak`
version 1 printed `ADAPTER CHECK PASSED` with that hash. It waited about 3.5 hours in
Kaggle's queue and then used 0.21 GPU hours. Graded in Docker on a scratch copy, then in
place: identical, 0 mismatches.

| set | base | hardened RL | supervised | weak RL |
|---|---|---|---|---|
| dev-seen (105) | 88 | 99 | 99 | 97 |
| dev-unseen (96) | 68 | 80 | 67 | 80 |
| — generated (48) | 34 | 39 | 38 | 41 |
| — hand-written (48) | 34 | 41 | 29 | 39 |
| false passes, both sets | 4 | 4 | 6 | 6 |

- **Weak vs hardened**, dev-unseen: 4 tasks only weak solved, 4 only hardened (p = 1.0);
  generated bugs alone 3 against 1 (p = 0.63).
- **Weak vs base**, dev-unseen: 16 against 4 (p = 0.012). **Weak vs supervised**: 16 against
  3 (p = 0.004).
- **False passes**, every development task, weak against hardened: 4 against 2 (p = 0.69).
  Rule 3 is judged only on sealed test 3.
- The weak arm retried 50 times and repeated its first answer in 7 of them.

**Next.** The freeze, then the one opening of sealed test 3 (all four arms, 192 tasks),
graded on a copy and then in place; then HumanEvalFix once, reported only.

## Addendum 14 — November 14, 2026 (21:55 UTC), sealed test 3 opened once and graded; results

Written after the opening. It reports results and changes no rule; the Rules section is
unchanged. Sealed test 3 is now **spent**: any later comparison needs a new sealed set.

**The opening.**
- Frozen record `artifacts/rlfirst-frozen.json` (frozen 2026-11-14 19:22:16 UTC). Registry
  `artifacts/final-test-3-registry.json` was written before upload, with seed 20261140.
- Each arm ran once, greedy, in its own private notebook (`alwaysacing/repairlab-final3-{base,hardened,weak,supervised}`
  version 1). Every notebook printed `ADAPTER CHECK PASSED` with the frozen hash (base: no
  adapter). Hidden cases stayed on the laptop. Together the four used about 1.4 GPU hours
  (17.82 h left before, 16.45 h after).
- **191 tasks, not 192 (a deviation, disclosed).** One program, `roster-order`, had no
  admissible single-bug mutant, so it contributed only its hand-written bug. The sealed
  generated set is therefore **95** tasks, and rules 1 and 2 were applied to 95 generated
  bugs and rule 3 to 191 tasks. No other program was affected.
- **Grading.** Every distinct submission (755) was graded in the pinned Docker grader, first
  on a scratch copy, then once in place. The in-place report (`artifacts/final-test-3-v1/final3-results.json`) has
  **0 mismatches**, and its rules, description and per-task outcomes are identical to the
  copy's. The registry is marked completed.

**Rules (as written before any result).**

| rule | comparison | result | only-first vs only-second | p | outcome |
|---|---|---|---|---|---|
| 1 | hardened RL vs base, solved, 95 generated bugs | 65 vs 64 | 13 vs 12 | 1.00 | **not met**: RL is not shown to help |
| 2 | hardened RL vs supervised, solved, 95 generated bugs | 65 vs 60 | 20 vs 15 | 0.50 | **neither** is called better |
| 3 | weak RL vs hardened RL, false passes, 191 tasks | 8 vs 10 | 5 vs 7 | 0.77 | **not met**: the weak grader is not shown to teach cheating |

Addendum 12 predicted that rule 1 might not be met, because the development gain on
generated bugs was small (+5, p = 0.23). It was not met.

**Description (not claims).**

| set | base | hardened RL | weak RL | supervised |
|---|---|---|---|---|
| generated (95) | 64 | 65 | 63 | 60 |
| hand-written (96) | 43 | 52 | 54 | 36 |
| all (191) | 107 | 117 | 117 | 96 |
| false passes, all | 8 | 10 | 8 | 3 |
| valid first reply, all | 150 | 174 | 151 | 190 |
| retried / solved after retry | 97 / 18 | 81 / 14 | 90 / 21 | 94 / 2 |
| retry identical to first answer | 28 | 22 | 16 | 79 |
| final answer = the unchanged bug | 6 | 1 | 1 | 16 |

Paired sign tests (described only; rule 2 asks that every set be described):
- **Hardened RL vs supervised:** hand-written 52 vs 36 (24 vs 8, p = 0.007); all 191,
  117 vs 96 (44 vs 23, p = 0.014). In 36 of the 44 tasks only hardened solved, supervised
  repeated its first answer on the retry.
- **Hardened RL vs base:** hand-written 52 vs 43 (16 vs 7, p = 0.093); all 117 vs 107
  (29 vs 19, p = 0.19). In 10 of the 29 tasks only hardened solved, base's first reply
  had no valid code.
- **Weak RL vs base:** hand-written 54 vs 43 (16 vs 5, p = 0.027); all 117 vs 107 (p = 0.19).
- **Weak vs hardened:** 117 vs 117 solved (14 vs 14). False passes by set: generated 4 vs 6,
  hand-written 4 vs 4; only 3 tasks false-passed under both.
- **Supervised vs base:** all 96 vs 107 (24 vs 35, p = 0.19).
- 56 tasks were solved by all four arms, 38 by none.

**By kind (generated bugs; solved by base / hardened / weak / supervised).**

| kind | tasks | base | hardened | weak | supervised |
|---|---|---|---|---|---|
| wrong variable | 63 | 45 | 44 | 43 | 44 |
| missing edge case | 8 | 4 | 5 | 4 | 2 |
| inverted logic | 6 | 5 | 5 | 5 | 5 |
| boundary | 5 | 1 | 2 | 2 | 3 |
| wrong operation | 4 | 2 | 2 | 2 | 3 |
| wrong return | 3 | 2 | 2 | 2 | 1 |
| missing step | 2 | 2 | 1 | 1 | 0 |
| statement order | 2 | 1 | 2 | 2 | 1 |
| wrong scope | 2 | 2 | 2 | 2 | 1 |

**A design weakness found after opening.** The 15% cap applied to the training pool, not to
the evaluation sets. Each sealed program's generated bug was drawn uniformly from its
admitted mutants, and wrong-variable mutants are the most numerous, so **63 of the 95
sealed generated bugs (66%) are wrong-variable bugs** (development-unseen: 27 of 48, 56%;
pool: 132 of 908, 14.5%). Rule 1 therefore mostly measures one kind, which base already
solved 45 of 63 times. This could have been seen in the development sets before opening;
it was not checked. It is reported, and nothing is recomputed.

**Reading.**
1. **RL from the untrained model did not beat the untrained model on the pre-registered
   test.** The development gain (80 vs 68 on unseen programs, p = 0.017) did not replicate
   on sealed generated bugs, and on all 191 tasks it shrank to 117 vs 107 (p = 0.19).
2. **RL beat supervised fine-tuning where supervised memorized** (hand-written bugs and all
   tasks), but not on the pre-registered generated set.
3. **No sign of reward hacking.** The weak grader ended with as many solved tasks as the
   hardened one and no more false passes. In training, its extra reward was small: false
   passes were 803 of 20,352 weak-arm episodes (3.9%), against 913 (4.5%) in the hardened
   arm, where they earned nothing. Most of the time the easiest way to pass the public
   tests, which always include the bug's witness case, was to fix the bug.
4. **The sealed false passes are incomplete fixes, not exploits.** All 29 false-passing
   final answers (base 8, hardened 10, weak 8, supervised 3) were scanned after grading:
   - none imports a module or calls `open`, `eval`, `exec`, `getattr`, `globals` or a
     similar function;
   - none writes a public case's string, list or dict input into its code (a heuristic
     that does not check numeric inputs);
   - each passes 48% to 99.6% of its hidden cases, and 26 of 29 pass at least 70%.
     Hard-coding the public examples would fail most hidden cases, as each task's
     hard-coding control shows.

**Mismatches.** The scratch copy reported 3, all `public_result_shown` on base's first
attempts at three tasks (one hand-written, two wrong-variable), each of which failed its
public tests in both the session and Docker. Regrading those three submissions 5 times
each, 6 at a time, matched the session every time, and each takes about 0.5 s alone; the
in-place grade reported none. They were transient grader failures. Docker reports only a
whole-run status, and any failed run sets `public_passed` to False, so a transient failure
on a submission the session had passed would appear as a `public_passed` mismatch. There
were none, so no solved or false-pass outcome could have been affected.

**Next.** HumanEvalFix once, all four arms, reported only.

## Addendum 15 — November 14, 2026 (22:35 UTC), HumanEvalFix run once; reported only

Run after sealed test 3's registry was marked completed, as the plan's Opening section
requires. Nothing here is a rule or a claim. HumanEvalFix is now **spent** for this phase.

**The run.**
- `scripts/run_humanevalfix_rlfirst.py`, output `artifacts/humanevalfix-rlfirst-v1`. The
  registry `artifacts/humanevalfix-rlfirst-registry.json` was written before upload.
- **114 of 117 tasks admitted.** The 1.7B phase's mechanical conversion was put in the
  RL-first two-turn format. Three tasks were rejected because their bug gives no failing
  input within the converted input domain, so they have no witness case. Every admitted
  task passed its Docker controls.
- Each arm ran once, greedy, in `alwaysacing/repairlab-hef3-{base,hardened,weak,supervised}`
  version 1, and each printed `ADAPTER CHECK PASSED` with the frozen hash. Together they
  used about 0.9 GPU hours (16.45 h left before, 15.53 h after).
- 367 distinct submissions were graded in Docker on a scratch copy, then once in place.
  The two reports are identical, with **0 mismatches**, and the registry is marked
  completed.

| | base | hardened RL | weak RL | supervised |
|---|---|---|---|---|
| solved (114) | 76 | 84 | 91 | 82 |
| false passes | 2 | 2 | 0 | 3 |
| valid first reply | 67 | 68 | 59 | 113 |
| retried / solved after retry | 60 / 22 | 62 / 32 | 67 / 44 | 30 / 1 |
| retry identical to first answer | 4 | 6 | 3 | 27 |

Paired sign tests (reported only, six comparisons, no correction):
- hardened vs base: 84 vs 76 (16 vs 8, p = 0.15);
- weak vs base: 91 vs 76 (19 vs 4, p = 0.003);
- supervised vs base: 82 vs 76 (19 vs 13, p = 0.38);
- hardened vs supervised: 84 vs 82 (13 vs 11, p = 0.84);
- weak vs supervised: 91 vs 82 (17 vs 8, p = 0.11);
- weak vs hardened: 91 vs 84 (11 vs 4, p = 0.12).

60 tasks were solved by all four arms and 11 by none.

**Notes.**
- **First-reply validity.** The three arms that reason before answering often wrote more
  than one code block in the first reply. That was the reason for 46 of base's 47 invalid
  first replies, 42 of hardened's 46 and 53 of weak's 55. No reply reached the 768-token
  cap. The retry recovered most of them. In 11 of the 16 tasks only hardened solved, base's
  first reply was invalid.
- **Supervised** almost never had an invalid first reply, but it solved 1 of its 30
  retries and repeated its first answer in 27, as on sealed test 3.
- **Not comparable with the 1.7B phase's 59/117.** That phase used the eight-action tool
  format and 117 tasks. This uses the two-turn format and 114 tasks.
- HumanEvalFix is public and may be in Qwen3's pretraining data.

**Phase 3 is complete.** Sealed test 3 and HumanEvalFix are both spent. Any further
comparison of these arms needs a new sealed set.
