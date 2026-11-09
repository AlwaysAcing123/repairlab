# Qwen3-1.7B plan — November 5, 2026

Written **before** any Qwen3-1.7B run, so its rules cannot be fitted to the outcome.
Nothing here is a model score. Status lines may be updated afterward; the rules may not.

## Questions

1. **Primary:** does the whole training pipeline on Qwen3-1.7B produce a better repair
   model than the best 0.6B pipeline, on programs no choice was tuned on?
2. Does training still help at 1.7B, compared with the untrained 1.7B model?
3. Does a short written diagnosis before each demonstrated fix make the model use the
   test output, instead of repeating edits?

## What changes from the 0.6B pipeline, and why

| Change | Reason |
| --- | --- |
| Model `Qwen/Qwen3-1.7B`, revision `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (Apache-2.0) | Next size up that fits a free T4. Its chat template and tokenizer are byte-identical to the 0.6B revision's. |
| Gradient checkpointing in every 1.7B training step | 0.6B training peaked at 12.5 GB of the T4's 15.6 GB without it, and 1.7B layers are twice as wide. Checkpointing recomputes activations instead of storing them; it does not change the gradients. |
| One supervised stage from the base model, on the 480 recovery samples | The 0.6B warm-up data used a different prompt and observation format from evaluation. The recovery samples already include direct repairs and match evaluation exactly. |
| Two supervised variants: **plain** and **notes** | Tests question 3. The variants differ only in the text of the supervised edit turns. |
| RL: 8 attempts per program instead of 4, on the programs nearest a 50% pass rate | 0.6B RL got a learning signal from only 5 to 7 of 30 programs per cycle. |
| Development set: 48 tasks | The 24 pilot tasks plus the 24 opened first-final-test tasks. Spent as a test, the latter are fine for choices. |
| Second outside measure: HumanEvalFix in the native tool format, reported only | Every task so far was written by the same AI. |

Nothing else changes: system prompt, tools, budgets (8 actions including submit, 512
generated tokens per call, 4,096 per episode, 12,288 context tokens), greedy decoding,
seed 7, and the success rule (an explicit submit that passes every public and private
check). LoRA is unchanged: rank 8, alpha 16, no dropout, `q_proj` and `v_proj`.

### The notes variant

In the notes variant, every supervised **edit** turn carries one short line of text
before its tool call. Read, test, and submit turns stay bare, as in the plain variant.

- After a failing test: `Failing case: solve(ARGS) returned VALUE, expected EXPECTED.
  Cause: CAUSE. Fix: FIX.` The case is the first failing public case in the real test
  output of that demonstration, so the line quotes what the model saw.
- After reading only (direct repair): `Cause: CAUSE. Fix: FIX.`

Cause and fix are written by hand for each of the 30 training programs, twice: once for
the original bug and once for that program's plausible wrong repair. A wrong repair
still never becomes a target; only its explanation appears, in the turn that replaces
it with the reference.

## Runs, in order

| # | Run | Arms or data | GPU estimate |
| ---: | --- | --- | ---: |
| 0 | Preflight: memory and speed on the T4 | Longest training sample, longest RL sequence, worst sampling batch. No task is attempted. | 0.3 h |
| 1 | Untrained 1.7B on development | `1.7b-base`, plus `0.6b-rl` as a rerun check | 0.8 h |
| 2 | Supervised training | `1.7b-plain`, `1.7b-notes` | 5 h |
| 3 | Development comparison | `1.7b-plain`, `1.7b-notes` | 1.1 h |
| 4 | RL from the chosen variant | 30 RL programs | 9 h |
| 5 | Development comparison | chosen variant, `1.7b-rl` | 1.1 h |
| 6 | Freeze every checkpoint and choice | none | 0 |
| 7 | Final test 2, opened once | all five arms on all 48 sealed tasks | 1.8 h |
| 8 | HumanEvalFix native, run once | `0.6b-rl`, `1.7b-base`, final 1.7B | 3 h |

About 22 GPU hours in total, an estimate until the preflight measures speed. That spans
two weekly quotas (22.4 h left until November 7, then 30 h).

`0.6b-rl` is the 0.6B RL adapter, the end of the 0.6B pipeline. On the 48 tasks where
both 0.6B adapters ran, recovery and RL tie at 24 each, so the choice is the pipeline's
end point rather than a score.

### Preflight rule

The preflight runs one training step on the longest supervised sample (both variants),
an RL update step on synthetic sequences of 4,096, 6,144, and 8,192 tokens, and sampling
at batch caps of 60, 40, and 25 million. Every 1.7B notebook then uses:

- gradient checkpointing for every training step;
- RL `max_sequence_tokens` = the longest tested length whose step peaks below 13 GiB;
- the sampling square cap = the largest tested cap whose worst batch peaks below 13 GiB.

If the supervised step exceeds 13 GiB even with checkpointing, the plan stops for a new
design. The preflight also records the 1.7B base-parameter fingerprint that later runs
must match. It is a plumbing check, not a result.

## Decision rules

1. **Variant:** RL starts from the supervised variant with more autonomous development
   successes in run 3. On a tie, notes, because RL can then reward the written
   diagnosis along with the actions.
2. **Final 1.7B checkpoint:** `1.7b-rl` if it solves more development tasks than its
   starting variant in run 5, otherwise that starting variant.
3. **Primary, "1.7B counts as better":** on final test 2, the final 1.7B checkpoint
   solves more tasks than `0.6b-rl` **and** the exact two-sided sign test over tasks
   that only one of them solves gives p < 0.05.

Reported with sign tests, not used for any decision: final 1.7B against `1.7b-base`,
notes against plain, RL against its starting variant, and every arm on HumanEvalFix.
Differences of a few tasks are reported as unresolved, not as gains.

### How debugging is measured

For each arm on final test 2:

- **Repeated edits:** among edits after an episode's first edit, the share that
  submits a source identical to an earlier edit in the same episode. For 0.6B it was
  40 of 53 (recovery) and 38 of 51 (RL) on the first final test.
- **Real recoveries:** tasks solved after a public test failed on a candidate the model
  itself had edited.

These describe behavior. They decide nothing.

## Final test 2 and HumanEvalFix: opened once

After run 5, `scripts/freeze_scaleup.py` writes `artifacts/scaleup-frozen.json`: every
arm's checkpoint hashes, the variant chosen by rule 1, and the final checkpoint chosen by
rule 2, each recomputed from the saved run reports. It refuses to overwrite itself.

`scripts/run_final_test_2.py --open-sealed-final-test-2` then:

1. refuses to run without the frozen record, or if `artifacts/final-test-2-registry.json`
   exists;
2. writes that registry entry **before** uploading anything, so a failed opening cannot
   be retried silently;
3. runs all five arms on all 48 sealed tasks, in manifest order, unseeded, under the
   development rules;
4. records the successes in the registry.

HumanEvalFix native uses its own registry, `artifacts/humanevalfix-native-registry.json`,
the same way. No checkpoint is chosen on either result, and every arm is reported.

## HumanEvalFix native

The 164 HumanEvalFix Python tasks are converted mechanically, before any model runs:

- kept only if the declaration is one function, with `typing` imports at most, no
  default or variable arguments, and every assertion in the example and hidden tests
  has the form `assert candidate(literal arguments) == literal` (or a bare or negated
  call, meaning True or False);
- the function is renamed `solve`, recursive calls included, and annotations are
  removed;
- example-test cases are public; the other test cases are private;
- each converted task must pass Docker controls: the reference passes every case and
  the buggy version fails at least one. Tasks failing a control are dropped, not
  repaired;
- tasks with any hard or near match to a training program (supervised, RL, or warm-up)
  are dropped, by the same thresholds as every earlier leakage check.

A first count found 136 convertible tasks before controls. HumanEval is public and
widely copied, so both model sizes may have seen it in pretraining. Absolute scores may
be inflated; comparisons between arms are what this measures.

## What would make this a null result

If the final 1.7B checkpoint does not beat `0.6b-rl` under rule 3, the 1.7B pipeline is
not shown to be better, whatever its development scores. That is reported plainly.

## Addenda, written after the preflight and before any task was attempted

**Preflight results** (`artifacts/scaleup-preflight-v1`, 0.24 GPU hours):

| Step | Peak memory | Time |
| --- | ---: | ---: |
| Supervised step, longest sample (1,636 tokens), with checkpointing | 6.8 GiB | 7 s |
| The same step **without** checkpointing | out of memory at 13.9 GiB | — |
| RL step, 4,096 tokens | 8.8 GiB | 22 s |
| RL step, 6,144 and 8,192 tokens | out of memory | — |
| Sampling, 16 prompts at cap 60 million, 512 new tokens | 11.8 GiB | 226 s |
| Greedy decoding, 2,000-token prompt, 256 new tokens | 4.7 GiB | 17 s |

By the rule, every 1.7B training step uses gradient checkpointing, RL
`max_sequence_tokens` is **4,096**, and the sampling cap stays **60 million**. The
base-parameter fingerprint is `9131bb92…3b04`. The preflight's notes samples predate
one change below and were about 60 tokens shorter; the training runs record their own
peak memory.

**Two gaps the plan did not cover, closed now:**

1. An RL turn longer than 4,096 tokens is left out of that cycle's update and counted
   in its payload. It is never truncated. The plan had no rule for this.
2. A repair edit keeps its note when it reappears in a later sample's history, as the
   model's own note would during evaluation. The first draft of the notes data
   showed the note only in the target turn.

**Speed.** bfloat16 runs slowly on the T4, so the estimate rises from about 22 to about
33 GPU hours. Two weekly quotas still cover it.

**HumanEvalFix native admission** (`artifacts/humanevalfix-native-admission-v2`): 135
of 164 tasks convert. Ten fail a Docker control, because the reference returns a tuple,
which is not JSON, or changes its input. Eight match a training program and are
dropped, including HumanEval's gcd, maximum, bracket-balance, and sortedness tasks,
which resemble the authored `euclid-gcd`, `highest-score`, `balanced-parens`, and
`nondecreasing-series`. **117 tasks are scored.** The first admission run,
`-v1`, also dropped three tasks whose bug loops forever. The plan counts that as
failing, so `-v2` keeps them, and the audit accepts a timeout that recurs on regrade.

## Incidents

- **Run 1, first attempt** (`artifacts/scaleup-dev-base-v1`): the notebook stopped after
  two minutes, before generating anything, because PEFT refuses adapter names containing
  a dot, such as `0.6b-rl`. The scripted Kaggle stand-in never executes the runtime, so
  the dry run could not catch it. `scripts/sized_eval_runtime.py` now names adapters by
  arm position. Nothing in `repairlab/` changed. The attempt is kept.
- **Run 1, second attempt** (`-v2`): Kaggle refused the upload because both supervised
  trainings already held the account's limit of two GPU sessions. No GPU time was used.
  Run 1 starts as `-v3` once a training finishes.
- **Run 3, first attempt** (`artifacts/scaleup-dev-sft-v1`): the notebook stopped before
  generating anything, because the second adapter on the shared 1.7B base did not match
  its saved fingerprint. PEFT 0.18.1 creates a later adapter in the base's bfloat16,
  loads the saved float32 weights into it, and only then casts it to float32, so the
  weights were rounded. The first adapter, created by `from_pretrained`, is not
  affected, which is why the single-adapter runs matched. The runtime now loads each
  later adapter a second time, into its float32 copy, and the fingerprint check confirms
  the exact weights. The attempt is kept; the rerun is `-v2`.
- **Run 4, cycle 1, round 8:** this computer briefly could not resolve `api.kaggle.com`
  while polling a running notebook, and the client treated the failed status call as
  fatal. Status, metadata, and download calls are now retried, keeping every failed
  log; pushes are never retried. The run resumed with `--resume`: the partial cycle is
  kept as `cycles/1-abandoned-1`, and cycle 1 restarted from the plain adapter. About
  1.4 GPU hours were lost. Nothing in `repairlab/` changed.
- **This computer slept during RL** (lid closed, on battery) several times, pausing the
  local runner. Kaggle notebooks kept running; nothing was lost except time.
- **One wrong RL reward.** In cycle 4, a sleep began while sample `s3` of
  `valid-clock-time` was being graded, so its public tests "timed out" after 19 seconds,
  against a 12-second limit, while its private tests graded normally in 1.3. The
  candidate is correct: a fresh regrade passes every check. The group's true rewards
  were 8 of 8, which gives no signal. The false 0 instead put 32 of the cycle's 423
  turns into the update, pushing down one correct attempt (advantage -0.875) and
  slightly up seven others (+0.125). The RL audit found it. **Decision, made before
  run 5's results:** keep the checkpoint and report the error. It can only have
  worked against RL, it touched one of six small updates, and redoing cycles 4 to 6
  would cost about four GPU hours. The audit now reports a saved timeout that regrades
  normally as an infrastructure fault rather than crashing. The saved grade still
  stands as the score.
- **A second network outage** during run 5's first attempt (`scaleup-dev-rl-v1`) lasted
  longer than the five minutes of retries. Read-only calls now retry for about an hour.
  Run 5 reruns as `-v2`.
- **New grading safeguard for the openings, decided before either opens:** final test 2
  and HumanEvalFix native grade a timed-out candidate up to two more times and log each
  retry in the run directory. An infinite loop times out every time, so only a grade
  the host interrupted can change. The development runs had no timeouts at all (140
  submissions and 328 public tests, all graded normally).
- **The audit's independent parser was too lenient.** One sampled RL turn wrote a valid
  test call followed by an invented tool response. The native parser rejected it, and
  the episode counted an invalid action, which is correct. The audit's regex parser
  accepted the first block, so it now also refuses stray tool tags.
- **HumanEvalFix sources depend on the Python version.** The runners use the system
  Python 3.9.6, the admission run and tests used 3.12, and `ast.unparse` writes tuple
  targets differently in the two. The code is semantically identical. Admission rerun
  under 3.9 (`artifacts/humanevalfix-native-admission-py39`) gave the same controls,
  drops, and separation results, and the HumanEvalFix audit ran under 3.9. No other
  task is affected, because every other source is fixed text.

## Status

| Run | State |
| --- | --- |
| 0. Preflight | Done: proceed (see addenda) |
| 1. Untrained 1.7B | Done (`scaleup-dev-base-v3`): `1.7b-base` 19/48; `0.6b-rl` 24/48, reproducing all 48 saved trajectories |
| 2. Supervised training | Done: `scaleup-sft-plain-v1`, `scaleup-sft-notes-v1`, both audited. They ran before run 1, which decides nothing, because both GPU slots were free first |
| 3. Variant comparison | Done (`scaleup-dev-sft-v2`): plain 34/48, notes 29/48 (10 against 5 tasks, p = 0.30). Rule 1 chooses **plain** |
| 4. RL | Done (`scaleup-rl-v1`, audited): six cycles, 1,320 sampled attempts, 4,804 sampled turns decoded |
| 5. RL comparison | Done (`scaleup-dev-rl-v2`): RL 36/48, plain 34/48 (2 against 0 tasks, p = 0.5). Rule 2 chooses **`1.7b-rl`** |
| 6. Freeze | Done: `artifacts/scaleup-frozen.json` |
| 7. Final test 2 | Done, opened once (`scaleup-final-test-2-v1`): 1.7B RL 29/48, 0.6B RL 17/48 (17 against 5, p = 0.017). **Rule 3 is met: 1.7B counts as better** |
| 8. HumanEvalFix native | Done, run once (`scaleup-humanevalfix-v1`): 1.7B RL 59/117, 0.6B RL 44/117, untrained 22/117 |

Results: [the 1.7B report](scaleup-results-20261106.md).
