"""Supervised recovery demonstrations: after a failed public test, revise, retest, submit.

The first adapter trained on a different system prompt, an instructed user prompt, and
test observations without ``status`` or ``mutated_input``. Here every prompt is exactly
the autonomous evaluation format and every observation comes from a real Episode.

Wrong repairs appear only as history. No sample has one as its target, because that
would teach the model to write the bug rather than to recover from it.
"""

import json

from ..episode import Episode
from ..native_tools import assistant_call
from ..revision import MODEL, REVISION
from ..tasks import canonical, pilot_tasks, sha256
from .final_test import final_tasks
from .splits import reject_final
from .training_programs import training_programs
from .transcript import initial_messages, retained_repair, scripted_step

PROTOCOL = "native-recovery-supervised-v1"
CONFIG = {
    "optimizer_steps": 180,
    "micro_batch_size": 1,
    "gradient_accumulation": 8,
    "learning_rate": 0.0001,
    "max_sequence_tokens": 2560,
    "seed": 7,
    "checkpoint_steps": [60, 120],
}
WARM_START = {
    "kernel_source": "alwaysacing/repairlab-native-supervised-warmup",
    "upstream_version": 1,
    "file_sha256": "2dd4773919ca4c13934d796fd0e38ed77aeb8df12055e2cd8f3b69992ac47375",
    "tensor_sha256": "245ac4651a2fb8018a8c699bc6219341ddf85a674d8584372bd5aa4905213c82",
    "base_sha256": "2d4bad117c2ae98f4a69740ac05d48c237237d84711349a396c5e98866553204",
}


class MemoizedExecutor:
    """Identical candidates and cases give identical grades; avoid repeat containers."""

    def __init__(self, executor):
        self.executor = executor
        self.results = {}

    def run(self, source, cases, *, reveal=False):
        key = (source, canonical([c.record() for c in cases]), reveal)
        if key not in self.results:
            self.results[key] = self.executor.run(source, cases, reveal=reveal)
        return self.results[key]


def trajectories(program):
    """Actions as (tool, arguments, supervised). Unsupervised actions are history only."""
    task = program.task
    retained = {"source": retained_repair(task.buggy)}
    wrong = {"source": program.wrong}
    fix = {"source": task.reference}
    tail = [("edit", fix, True), ("test", {}, True), ("submit", {}, True)]
    return {
        "direct": [("read", {}, True), *tail],
        "recover-retained": [
            ("read", {}, False),
            ("edit", retained, False),
            ("test", {}, True),
            *tail,
        ],
        "recover-wrong": [
            ("read", {}, False),
            ("edit", wrong, False),
            ("test", {}, True),
            *tail,
        ],
        "recover-twice": [
            ("read", {}, False),
            ("edit", retained, False),
            ("test", {}, False),
            ("edit", wrong, False),
            ("test", {}, True),
            *tail,
        ],
    }


def demonstrations(executor, programs=None):
    programs = training_programs() if programs is None else programs
    reject_final([p.task for p in programs])
    executor = MemoizedExecutor(executor)
    samples, seen = [], {}
    for program in programs:
        task = program.task
        for shape, actions in trajectories(program).items():
            episode = Episode(task, executor, max_actions=8)
            history = initial_messages(task)
            fixed = False
            for index, (tool, arguments, supervised) in enumerate(actions):
                prompt = json.loads(canonical(history))
                observation = scripted_step(episode, history, tool, arguments)
                fixed = fixed or arguments.get("source") == task.reference
                if tool == "test" and observation["passed"] is not fixed:
                    raise ValueError("Demonstration test outcome is wrong: " + task.id)
                if not supervised:
                    continue
                if tool == "edit" and arguments["source"] != task.reference:
                    raise ValueError("A wrong repair may never be a target: " + task.id)
                key = canonical(prompt)
                completion = assistant_call(tool, arguments)
                if seen.setdefault(key, completion) != completion:
                    raise ValueError("Two targets for one prompt: " + task.id)
                samples.append(
                    {
                        "sample_id": f"{task.id}-{shape}-{index}",
                        "split": "train",
                        "program_id": task.id,
                        "group_id": program.group,
                        "trajectory": shape,
                        "prompt": prompt,
                        "completion": [completion],
                    }
                )
            if episode.result["success"] is not True:
                raise ValueError("Demonstration did not repair " + task.id)
    return samples


def bundle(executor):
    train = demonstrations(executor)
    text = canonical(train)
    for task in pilot_tasks() + final_tasks():
        for source in (task.reference, task.alternative):
            if json.dumps(source)[1:-1] in text:
                raise ValueError("Evaluation solution in training data: " + task.id)
        if '"task_id":"' + task.id + '"' in text.replace('\\"', '"'):
            raise ValueError("Evaluation task in training data: " + task.id)
    body = {
        "protocol": PROTOCOL,
        "config": CONFIG,
        "warm_start": WARM_START,
        "model": MODEL,
        "model_revision": REVISION,
        "train": train,
        "provenance": "New authored training programs; no evaluation solution, external-model output, or sealed final-test content.",
    }
    return {**body, "bundle_sha256": sha256(canonical(body))}
