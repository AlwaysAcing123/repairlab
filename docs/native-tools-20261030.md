# Native tool-interface diagnostic — October 30, 2026

**11/12 instructed probes produced a valid native call to the requested tool.** All
four native serializer/parser round-trips passed. The three edit source arguments
exactly matched their requested functions and ran correctly over the checked inputs.
One probe emitted a malformed opening tool tag. No weights were trained.

This is an interface diagnostic, not an autonomous repair benchmark. Four probes
contain scripted conversation histories; they are not full model-driven episodes.
Explicit source-edit instructions do not establish independent code-repair ability.

## Question and frozen protocol

The preceding custom JSON tool loop scored 0/24 in both feedback conditions and
produced many invalid actions. This experiment asks whether the model's native
function-call interface can carry the tools and code arguments correctly.

- Original `Qwen/Qwen3-0.6B`, revision `c1899de289a04d12100db370d81485cdf75e47ca`.
- Same starting parameter fingerprint as the preceding paired model; no adapter,
  optimizer, training examples added to weights, or weight updates.
- Native tokenizer chat template with `tools=`; structured assistant `tool_calls`
  and actual `tool` message roles; TRL 1.8.0 `qwen3_schema` through Transformers
  5.2.0 `tokenizer.parse_response`.
- Native raw completions retained with special tokens (`skip_special_tokens=False`).
- Four serializer/parser round-trips: read, test, edit, submit.
- Twelve synthetic probes: eight direct or paraphrased requests, plus four follow-up
  requests with scripted prior tool calls and results.
- One generation per probe, greedy decoding, thinking disabled, seed 7, at most 256
  output tokens and a 4,096-token context budget without truncation.
- No output rewriting, missing-tag correction, constrained decoding, or retry of a
  failed probe. Requested tool and exact-source match are recorded separately.

The readiness gate was written before submission: all four contract round-trips
and all twelve requested tool calls must pass before a larger native repair pilot.
**It was not met**, and no larger native repair evaluation was launched.

## Observations

| Check | Result |
| --- | ---: |
| Native contract round-trips | 4/4 |
| Valid native call to requested tool | 11/12 |
| Follow-up probes with scripted tool histories | 4/4 |
| Exact source arguments on three edit probes | 3/3 |
| Local execution of those three source arguments | 3/3 |

The failed probe asked: “I want to inspect the current implementation. Use read first.”
The model emitted this malformed opening tag:

```text
<tool>
{"name": "read", "arguments": {}}
</tool_call><|im_end|>
```

The native parser correctly found no valid complete tool call. The response used 16
tokens, below the 256-token limit, so it was not an output-budget truncation.

The direct `read` probe passed, as did both `test` and both `submit` requests, two
explicit source replacement requests, and the four scripted follow-up contexts.
One follow-up contained a failing public example and requested an edit to return
`x+1`; its generated native source argument was correct. This is a useful signal
that native tool messaging deserves further work, not a repair benchmark result.

The native and custom JSON runs have different probes and prompts, so their percentages
must not be presented as a controlled repair improvement. No weight-training gain is
claimed. The format and parser are the changes, not learned parameters.

## Independent checks and execution boundary

**45/45 preflight tests passed**, including the existing nine Docker integration
tests. An additional local audit independently extracted the raw `<tool_call>` JSON
blocks with duplicate-key checks and compared the results with the native parser.
It verified all probe histories, requested tools, source arguments, token limits,
template fingerprint, unchanged weight hashes, notebook privacy, and server source.
The native parser and independent extraction agreed on every probe, including the failure.

The three source arguments were executed only after download, in the pinned restricted
local containers, over every integer input from -5 through 5. These are labeled
instructed-edit controls, not autonomous or held-out repair scores. No model-generated
code executed in the GPU notebook, and no account credential was uploaded.

## Compute and evidence

The private [Kaggle diagnostic](https://www.kaggle.com/code/alwaysacing/repairlab-native-tool-diagnostic)
version 1 completed. Its model loop took **17.58 seconds**, excluding environment setup,
model loading, hashing, and artifact saving. Peak allocated GPU memory was approximately
**1.53 GiB**. Live quota readings moved from **22.47 to 22.44 free GPU hours**—about
**0.03 hours** including session overhead, subject to rounding. The worker is complete.

Evidence is preserved in `artifacts/native-tools-diagnostic-20261030-v1/`:

- `experiment-plan.json`: prespecified question, readiness gate, probes, and source hash.
- `upload/` and `server-v1/`: local and server notebook plus privacy metadata.
- `download/`: original result archive and server log, unchanged.
- `extracted/`: raw completions, parsed calls, round-trips, template, model hashes, and runtime.
- `independent-verification.json`: independent raw-output and probe audit.
- `edit-execution-checks.json`: three separately labeled local execution controls.
- `preflight-tests.log`, quota readings, and `final-worker-status.txt`.
- `results.html`: readable probe-by-probe report, including the failed raw response.

Implementation: `repairlab/native_tools.py`, `scripts/build_native_diagnostic.py`,
`scripts/verify_native_diagnostic.py`, and `tests/test_native_tools.py`.

## Next step

Use this native interface as the basis for the next development experiment. A small
supervised tool-use warm-up on independently authored examples can be measured
separately from any later RL. It must include the model's own native assistant/tool
format, validated source arguments, and terminal submission behavior. Keep these
diagnostic probes as development material rather than claiming an untouched test set.

Before longer training, establish a minimal autonomous repair pilot that can issue
valid tests and edits and submit a working function. Preserve the original-model
comparison under the same native interface. Do not lower the existing diagnostic
gate retroactively, relabel interface fixes as learning, or start sparse-reward RL
against the failed custom loop.
