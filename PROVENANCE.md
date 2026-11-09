# Authorship and reuse

This initial prototype was implemented with Codex assistance on October 29, 2026,
from the project owner's supplied directions and discussion. The owner chose the
specialist-model direction, code repair as the initial specialty, and free Kaggle or
Colab as the compute preference. They should distinguish their own work from this
AI-assisted implementation when discussing ownership in interviews.

The 24 pilot programs, specifications, mutations, alternative repairs, test domains,
executor, grader, episode implementation, notebook generator, and tests were authored
in this chat. They have not undergone independent human review. They are synthetic
backend-inspired helpers, not mined production bugs. None came from a public benchmark
or an external model teacher dataset.

The separate HumanEvalFix snapshot comes from `bigcode-project/octopack` at the
revision and checksum in `data/humanevalfix/snapshot.json`, with the original MIT
license preserved beside it. Its 164 Python tasks are external evaluation material,
not original project work. The adapter, control runner, and public-only notebook
are AI-assisted project implementation; their protocol differs from the authors'
official harness. External assert-based evaluation is not used as an RL reward oracle.

Read-only review of CompanyGym informed source/task fingerprints, checkpoint-evidence
requirements, worker diagnostics, notebook portability, and the first feasibility
model/revision. Read-only review of IncidentBench informed independent grading,
disposable execution, and positive/negative repair controls. No original experiment
was modified, and their results do not validate this environment or its future model.

The pinned Python container image was reused from the local IncidentBench runtime.
The GPU notebook relies on PyTorch, Hugging Face Transformers, and Qwen3-0.6B.
Its first real GPU baseline ran on October 29, 2026 and scored 8/24 custom development
tasks under a single-response protocol. No weights were trained. The HumanEvalFix GPU
baseline is still pending. Dataset/model licensing and redistribution notices must
be reviewed before an external release.

On October 30, Codex implemented and ran the paired revision comparison, local action
application, staged private Kaggle transport, and saved-evidence audit. The complete
comparison scored 0/24 in both conditions. No weights were trained. Scripted controls
are labeled separately. The owner authorized continuation and free Kaggle compute.

An initial live transport was rejected by automatic approval review before the
credential-bearing request executed. Its tunnel was stopped. `revision.py` and
`build_revision_notebook.py` retain the unused live-controller implementation as
development history; the supported experiment uses `batch_revision.py` and the batch
scripts, without callback credentials.

Later on October 30, Codex added the native Qwen tool schemas, synthetic diagnostic
probes, notebook builder, schema validation tests, and independent raw-output audit.
The unchanged model produced valid requested native calls on 11/12 probes. The
diagnostic included scripted histories and explicit edit instructions, so it is not
evidence of autonomous debugging or model training. All four native serialization
round-trips and three local instructed-edit execution checks passed. No larger
repair run was launched because the predeclared interface gate was not met.

The owner then authorized the supervised warm-up. Codex authored eight new training
programs, four separate development programs, 64 next-action demonstrations, a
completion-masked LoRA runtime, and a separate audit of saved tensors, token labels,
data boundaries, and evaluation outputs. The original model and fixed step-60 adapter
were compared under the same native interface. Scores were 12/16 versus 16/16 on the
new program probes and 11/12 versus 12/12 on the historical diagnostic. This is actual
supervised training with a saved adapter, not an RL result or autonomous repair claim.

The owner subsequently authorized a six-task native autonomous repair comparison.
Codex implemented the matched runner, private staged inference notebooks, checkpoint
reload checks, tool application, tests, evidence audit, and saved replay. The frozen
task rule selected two authored development tasks per category before inference. The
original model scored 0/6 and the saved supervised adapter 5/6, with no new training.
All eight submitted candidates were freshly graded again. The adapter failed to revise
its deduplication candidate after a public failure. These are small development results,
not an unseen final-test or RL claim. The old project directories remain unchanged.

Still on October 30, the project moved to Claude Code, which implemented the expansion
runner and generalized audit, the `repairlab/curriculum/` splits, sealed final test,
leakage checks, recovery demonstrations, matched comparison, GPU runtimes, and scripted
dry-run harness. Those 54 new tasks are AI-authored, like the original 24, and have not
undergone independent human review. No model has run on them. The evidence for the
earlier experiments was re-audited on a copy and not modified.

On November 2, Claude Code ran the expansion, recovery training, and both checkpoint
comparisons on private Kaggle GPUs, then audited each run. The first adapter scored 13/24
and the recovery adapter 17/24 on development tasks. A rejected first training upload
led to compressed notebook data; the data, configuration, and rules did not change. The
30 training programs have now entered training. Sealed final-test tasks have still had
**zero model evaluations**, and no RL has run.

On November 4 and 5, Claude Code authored 30 new RL training programs in
`repairlab/rl/`, which are also AI-authored and not human-reviewed, and fixed the RL
method and decision rules before any RL result. It then ran six RL cycles, the
four-checkpoint development comparison, and the seeded probe, and opened the sealed
final test once. RL scored 15/24 on development against 17/24 for its starting
adapter, and 9/24 against 7/24 on the final test. The final test has now been used and
is no longer unseen. Claude Code then authored a second sealed test of 48 tasks in
`repairlab/final2/`, also AI-authored and not human-reviewed, with zero model
evaluations.

On November 5 and 6, at the owner's request, Claude Code wrote and fixed the Qwen3-1.7B
plan before any 1.7B run, then ran it on private Kaggle GPUs. New code is in
`repairlab/scaleup/`. The 60 diagnoses used as notes were written by Claude Code and
checked against real test output. HumanEvalFix was converted to the native format by
a mechanical rule; its tasks remain external material under the original MIT license.
Qwen3-1.7B is used under Apache-2.0 at a pinned revision. The second final test was then
opened once and is no longer unseen.
