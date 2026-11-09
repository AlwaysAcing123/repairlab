# Paired revision-loop result — October 30, 2026

**0/24 repairs passed without execution feedback; 0/24 passed with feedback.**
The full comparison contains 48 original-model episodes and 264 model calls across
eight staged inference rounds. All final candidates were exactly the original buggy
functions. No correct final candidate was left unsubmitted.

This exposes a tool-use readiness problem under the custom JSON scaffold. It does
not establish that execution feedback is generally ineffective. There was no training.

## Frozen setup

- Original `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`;
  no adapter, optimizer, or parameter changes; greedy decoding, seed 7, thinking disabled.
- Same 24 development tasks, public examples, input domains, and private grading
  checks as the original 8/24 single-response baseline. No final held-out set exists.
- Identical initial prompts, parser, tools, decoding, and maximum budgets in both
  conditions. Pair order alternates by task. Only execution feedback availability changes.
- Eight actions including submission; at most 512 new tokens per call, 4,096 output
  tokens per episode, and a 12,288-token context budget without silent truncation.
- Plain JSON or exactly one enclosing JSON/unlabeled code fence accepted. Extra prose,
  missing closing fences, multiple objects, duplicate keys, and invalid actions rejected.
  Invalid actions consume budget; no automatic submission or semantic output repair.
- `test` returns public results in one condition and a fixed unavailable message in
  the other. Final checks stay local and never enter the model conversation.

The earlier 8/24 baseline used a different interface, prompt, and budget. It is a
reference, not a matched-budget estimate of interaction's effect.

## Results

| Measurement | Without feedback | With feedback |
| --- | ---: | ---: |
| Correct submitted repairs | **0/24** | **0/24** |
| Submitted, incorrect | 12 | 8 |
| Action limit, no submission | 12 | 16 |
| Model calls/actions | 120 | 144 |
| Invalid actions | 79 | 79 |
| Valid public test calls | 8 | 35 |
| Valid edit calls | 21 | 22 |
| Final source unchanged | 24/24 | 24/24 |
| Input tokens over all calls | 47,608 | 65,629 |
| Generated tokens | 1,976 | 2,256 |

`report.json` counts attempted tools; valid-call counts above exclude rejected calls.
Edits can re-submit the same source. Withheld tests do not execute candidate code.

Feedback changed behavior toward more tests but yielded no repairs. Common failures
were unclosed code fences, malformed objects imitating tool errors, repeatedly testing
the same fault, and submitting unchanged source. The current scaffold limits what this
experiment tells us about the model's underlying coding knowledge.

## Verification

**41/41 preflight tests passed**, including nine real Docker integration tests. Six
scripted known-correct control episodes also passed through the paired interface on
validation, collection, and control-flow tasks. Those are explicitly non-model results.

The saved-evidence audit checked every archived response, conversation history, parsed
action, public observation, budget, fingerprint, and aggregate score. All initial pair
prompts match; no hidden grades enter model histories. All eight notebook versions are
private and match their submitted cell sources.

Full parameter hashes match before/after each batch and across all batches. GPU, dtype,
packages, decoding, seed, and chat-template fingerprint match. The local execution
image content ID and worker bytes match the first baseline. Docker lost its repository
digest alias, so resolution now falls back to that exact previously recorded immutable
content ID; neither the grader nor the task data changed.

This is a separate local audit of evidence, not an independent human reproduction or
a proof of correctness outside the declared finite domains.

## Preserved diagnostics and transport

The initial strict JSON attempt rejected all 48 first actions as fenced output. It was
stopped after three applied rounds; its already-submitted fourth batch finished normally
and was archived without application. It is an incomplete interface diagnostic, not
a complete score per condition. The fresh comparison accepted one enclosing JSON fence
in both arms. No further parsing change occurred during that completed comparison.

Automatic approval review rejected the proposed live Cloudflare connection because a
temporary bearer credential would pass through an external service. The rejected
credential-bearing request did not execute and the tunnel was stopped. Actual model
evaluation used staged private notebooks without credentials or callbacks. Generated
Python was executed only in local restricted containers.

## Compute and next decision

The completed comparison used approximately **0.27 free GPU quota hours**, and the
stopped diagnostic approximately **0.12**, totaling about **0.39 hours** from rounded
readings. Remaining quota was **22.47 hours**. Model loops totaled **264.46 seconds**,
excluding startup, loading, queueing, and local execution; peak allocated memory was
about **1.73 GiB**. Every submitted worker completed. No paid inference or training ran.

Do not start terminal-reward RL with this failed loop. First verify minimal interactions
using the model's native tool-call template/parser, already used in CompanyGym. That
can distinguish custom-interface incompatibility from broader tool-use weakness.
Keep interface gains separate from weight-training claims. If supervised warm-up is
needed, author independent validated training programs, evaluate that checkpoint
separately, and only then measure additional RL under a frozen working interface.

## Evidence

Completed run: `artifacts/paired-revision-20261030-json-fences-v1/`.

- `experiment-plan.json`, `config.json`, `source-snapshot/`: frozen setup and implementation.
- `rounds/1` through `rounds/8`: public requests, uploaded/server notebooks, original output
  archives, raw responses, privacy confirmation, and unchanged-weight fingerprints.
- `episodes/`: all 48 local model trajectories, including final private grades.
- `report.json`, `independent-verification.json`, `failure-diagnosis.json`, `results.html`.
- `scripted-controls/`: six separately labeled positive controls.
- `preflight-tests.log`, quota readings, and `final-worker-status.txt`.

The private [Kaggle notebook](https://www.kaggle.com/code/alwaysacing/repairlab-paired-revision-batches)
versions 1–4 are the incomplete strict diagnostic; versions 5–12 are the completed
comparison. Each version is one inference round, not the aggregate result.

The strict diagnostic is preserved in `artifacts/paired-revision-20261030-batches-v1/`;
the rejected live-transport preparation, with zero model episodes, is in
`artifacts/paired-revision-20261029-v1/`.
