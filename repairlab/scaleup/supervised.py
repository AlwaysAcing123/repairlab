"""Supervised data for Qwen3-1.7B: the 480 recovery samples, plain or with notes.

The plain variant is exactly the 0.6B recovery data. The notes variant differs only
in its repair edits, which first state the failing case, its cause, and the fix. The
failing case is read from the real test output just before the edit. A repair edit
keeps its note when it reappears in a later sample's history, as the model's own
would. Wrong repairs stay history only and carry no note.
"""

import json

from ..curriculum.final_test import final_tasks
from ..curriculum.recovery import demonstrations
from ..curriculum.training_programs import training_programs
from ..final2.admission import reject_final2
from ..final2.programs import final2_tasks
from ..tasks import canonical, pilot_tasks, sha256
from .models import LARGE
from .notes import DIAGNOSES, note

PROTOCOL = "sized-supervised-v1"
VARIANTS = ("plain", "notes")
CONFIG = {
    "optimizer_steps": 180,
    "micro_batch_size": 1,
    "gradient_accumulation": 8,
    "learning_rate": 0.0001,
    "max_sequence_tokens": 2560,
    "seed": 7,
    "checkpoint_steps": [60, 120],
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "v_proj"],
    "gradient_checkpointing": True,
}


def _edit_source(message):
    for call in message.get("tool_calls") or []:
        if call["function"]["name"] == "edit":
            return call["function"]["arguments"]["source"]
    return None


def annotated(messages, program):
    """Give every repair edit its note, computed from the messages before it."""
    result, last_edit = [], None
    for message in messages:
        source = _edit_source(message)
        if source == program.task.reference:
            observation = json.loads(result[-1]["content"])
            text = note(program.task.id, observation, last_edit == program.wrong)
            message = {**message, "content": text}
        if source is not None:
            last_edit = source
        result.append(message)
    return result


def with_notes(samples, programs):
    """Add the diagnosis line to repair edits, in targets and histories alike."""
    by_id = {p.task.id: p for p in programs}
    result = []
    for sample in samples:
        program = by_id[sample["program_id"]]
        messages = annotated(sample["prompt"] + sample["completion"], program)
        result.append({**sample, "prompt": messages[:-1], "completion": messages[-1:]})
    return result


def variant_samples(executor, variant, programs=None):
    programs = training_programs() if programs is None else programs
    if variant not in VARIANTS:
        raise ValueError("Unknown supervised variant: " + variant)
    if set(DIAGNOSES) != {p.task.id for p in training_programs()}:
        raise ValueError("Every training program needs a diagnosis")
    reject_final2([p.task for p in programs])
    samples = demonstrations(executor, programs)
    return samples if variant == "plain" else with_notes(samples, programs)


def bundle(executor, variant):
    train = variant_samples(executor, variant)
    text = canonical(train)
    for task in pilot_tasks() + final_tasks() + final2_tasks():
        for source in (task.reference, task.alternative):
            if json.dumps(source)[1:-1] in text:
                raise ValueError("Evaluation solution in training data: " + task.id)
        if '"task_id":"' + task.id + '"' in text.replace('\\"', '"'):
            raise ValueError("Evaluation task in training data: " + task.id)
    body = {
        "protocol": PROTOCOL,
        "variant": variant,
        "config": CONFIG,
        "model": LARGE["model"],
        "model_revision": LARGE["model_revision"],
        "base_sha256": LARGE["base_sha256"],
        "train": train,
        "provenance": "The 30 authored training programs of the 0.6B recovery data; notes "
        "are hand-written diagnoses. No evaluation solution, external-model output, or "
        "sealed content.",
    }
    return {**body, "bundle_sha256": sha256(canonical(body))}
