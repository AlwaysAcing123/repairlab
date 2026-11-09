# First supervised native-tool adapter — October 30, 2026

**An actual LoRA adapter was trained and saved.** After 60 supervised optimizer steps,
the model issued the requested valid tool call on **16/16 separate-program probes**,
compared with **12/16 for the original model**. The earlier diagnostic improved from
**11/12 to 12/12**. No regressions were observed on these 28 instructed checks.

These are development interface results. They do not establish autonomous debugging,
general repair improvement, production readiness, or an RL gain. This is one small
supervised pilot with one seed and a fixed final checkpoint.

## Data and training

The training set contains **64 next-action examples from eight new toy programs** in
seven semantic groups: integer powers (square/cube), absolute value, negation, pair
addition, pair multiplication, string reversal, and first-character extraction.
Each contributes a demonstrated read/test/edit/test/submit workflow plus three direct
tool requests. All teacher actions use the same native Qwen function-call interface.

The **four separate development programs** cover parity, integer floor division,
joining words, and distinct-element counting. Each contributes four instructed
tool-use contexts, for 16 probes. Program/group IDs do not overlap with training.
These 16 contexts are not 16 independent programs. The old 12 diagnostic probes are
repeated and reported separately; none was a training target. The original 24 backend
repair tasks and HumanEvalFix repairs were not used for training.

Every new reference implementation passed its declared input checks, and every
faulty implementation failed, giving **24/24 expected program-control outcomes**.
Golden public examples were checked when constructing the fixtures. The data are
synthetic and AI-assisted, not independently human-reviewed production incidents.

Frozen configuration:

- Base: `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- LoRA rank 8, alpha 16, dropout 0; `q_proj` and `v_proj` modules only.
- **1,146,880 trainable parameters**; the original base parameters stayed frozen.
- AdamW, learning rate 0.0001, weight decay 0, gradient clipping at 1.0.
- Microbatch 2, gradient accumulation 4, effective batch 8, 60 optimizer steps.
- 480 sample presentations in total; all 64 unique training examples appeared.
- Maximum training length 2,048 tokens, without truncation. Entire observed histories
  and padding are masked from loss; only the next assistant completion is supervised.
- Native tokenizer template and TRL 1.8.0 Qwen response schema; Transformers 5.2.0,
  PEFT 0.18.1, seed 7, one free Kaggle T4.

The original model was evaluated first. The same prompts, tools, greedy decoding,
thinking-disabled setting, and 256-output-token/4,096-context-token limits were then
used with the trained adapter. A step-30 checkpoint is preserved as recovery evidence;
**step 60 was selected before training**, without evaluation-driven checkpoint choice.

## What changed

| Development check | Original | Supervised adapter |
| --- | ---: | ---: |
| Four separate programs, 16 instructed contexts | 12/16 | **16/16** |
| Historical native diagnostic | 11/12 | **12/12** |
| Exact source arguments, separate-program edit probes | 4/4 | 4/4 |
| Exact source arguments, historical edit probes | 3/3 | 3/3 |

The four separate-program failures were all `test` requests. The original model tried
to pass a `public_tests` argument to the zero-argument tool, sometimes with malformed
JSON. The adapter instead produced the permitted native call:

```text
<tool_call>
{"name": "test", "arguments": {}}
</tool_call>
```

The historical failure used `<tool>` instead of `<tool_call>`; the adapter produced
the correct opening tag. These are specific tool-contract improvements. Source-copying
checks were already correct and stayed correct. The evaluation explicitly requests
tools and sometimes supplies replacement source, so success is not autonomous fault
diagnosis, patch discovery, or end-to-end repair.

## Evidence of actual training

The saved initial and final adapter files contain **112 matching tensor names**.
An independent local read of the safetensors files confirmed their SHA-256 fingerprints
and a nonzero **L2 parameter change of 1.28190640361835**. Initial LoRA B tensors were
zero, and final adapter values differed. Base parameter fingerprints before and after
training match the original native diagnostic:

`2d4bad117c2ae98f4a69740ac05d48c237237d84711349a396c5e98866553204`

All 60 losses and gradient norms were finite. Low training loss alone is not the result;
the matched before/after development checks provide the behavioral observation.

The local audit independently decoded every supervised token span and verified that
it equals the intended native teacher action. It checked all prefix masks, padded-label
rules in the frozen runtime, training-batch IDs, group separation, model revision,
saved tensor names/hashes/deltas, raw evaluation calls, and aggregate scores. The
submitted and server notebook sources match, and Kaggle confirmed the notebook private.
**49/49 local preflight tests passed**, including nine Docker integration tests.

This is a local audit of saved evidence rather than an independent human replication.
Evaluation programs were excluded from training, but the small, synthetic development
set is not a pristine final test and does not establish broad generalization.

## Saved adapter and compute

The reusable adapter is in:

`artifacts/native-supervised-warmup-20261030-v1/extracted/adapter/`

It contains `adapter_model.safetensors` (about 4.4 MiB), `adapter_config.json` with the
pinned base revision, tokenizer files, and the native chat template. Load it on the
specified original base model through PEFT. The adapter alone is not a complete model.
No checkpoint was published or deployed.

The private [Kaggle run](https://www.kaggle.com/code/alwaysacing/repairlab-native-supervised-warmup)
version 1 completed. Quota moved from **22.44 to 22.27 free GPU hours**, approximately
**0.17 hours** including environment/session overhead, subject to rounded readings.
The measured original-evaluation/training/final-evaluation phase took **513.32 seconds**,
excluding initial model setup and dataset tokenization; peak allocated GPU memory was
approximately **8.88 GiB**. No paid inference or RL ran, and the worker is finished.

## Reproduction artifacts

Run root: `artifacts/native-supervised-warmup-20261030-v1/`.

- `experiment-plan.json`, `bundle.json`, `data-validation.json`, `source-snapshot/`.
- `upload/`, `server-v1/`: exact notebook and privacy/source confirmation.
- `download/`: unchanged original result archive and execution log.
- `extracted/initial_adapter.safetensors`, `extracted/checkpoint-30/`, `extracted/adapter/`.
- `extracted/tokenized-training.json`, `tokenization-preflight.json`, training logs and batch IDs.
- `extracted/evaluation/original/` and `extracted/evaluation/supervised/`: all raw calls.
- `extracted/weight-evidence.json`, `comparison.json`, and runtime configuration.
- `independent-verification.json`, `results.html`, preflight log, and quota/worker records.

Implementation: `repairlab/tool_warmup.py`, `scripts/build_native_warmup.py`,
`scripts/native_warmup_runtime.py`, and `scripts/verify_native_warmup.py`.

## Next experiment

Load this saved adapter into a **small native autonomous repair pilot**, alongside
the original model under the same native tools and budget. Check whether the model
actually chooses tests, responds to observed failures, generates a repair, and submits
it. The improved instructed probes justify that pilot; they do not substitute for it.

Keep supervised improvements separate from any future RL gain. Establish a working
mixed-success repair baseline and separate training/final-evaluation programs before
running reward-based optimization. A further positive result is not assumed.
