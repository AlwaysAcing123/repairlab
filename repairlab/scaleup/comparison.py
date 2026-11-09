"""Matched comparison across model sizes: each arm names its base model and adapter.

Episodes, tools, budgets, prompts, and grading are those of ``curriculum.comparison``.
Each split has one fixed task list. The second sealed test accepts only all of its
tasks, unseeded; every other split refuses its tasks by content.
"""

import json
from pathlib import Path

from ..curriculum import fingerprint as curriculum_fingerprint
from ..curriculum.comparison import (
    ENVIRONMENT_KEYS,
    MAX_ACTIONS,
    SEEDED_ACTIONS,
    MatchedComparison,
    seed_failure,
)
from ..curriculum.transcript import initial_messages
from ..episode import Episode
from ..experiments import source_fingerprint, write_json
from ..final2.admission import reject_final2
from ..final2.programs import final2_tasks
from ..native_repair import SYSTEM
from ..native_tools import tools
from ..revision import BUDGET
from ..tasks import canonical, sha256
from . import fingerprint
from .humanevalfix_native import native_tasks
from .models import MODELS, development_tasks

PROTOCOL = "sized-matched-comparison-v1"
GREEDY = {"do_sample": False, "num_beams": 1, "enable_thinking": False}


def split_tasks(split):
    if split == "development":
        return development_tasks()
    if split == "final_test_2":
        return final2_tasks()
    if split == "humanevalfix_native":
        return native_tasks()
    raise ValueError("Unknown split: " + split)


def debugging_metrics(task, record):
    """Repeated edits and real recoveries, from saved events only."""
    events = record["events"]
    edits = [
        e["action"]["source"]
        for e in events
        if e["action"].get("tool") == "edit" and "error" not in e["observation"]
    ]
    buggy = sha256(task.buggy)
    failed_own_edit = any(
        e["action"].get("tool") == "test"
        and e["observation"].get("passed") is False
        and e["source_sha256"] != buggy
        for e in events
    )
    return {
        "edits": len(edits),
        "later_edits": max(0, len(edits) - 1),
        "repeated_edits": sum(source in edits[:i] for i, source in enumerate(edits) if i),
        "real_recovery": bool(record["result"]["success"] and failed_own_edit),
    }


class SizedComparison(MatchedComparison):
    protocol = PROTOCOL

    def __init__(self, executor, destination, arms, tasks, seeded=False, split="development"):
        tasks = list(tasks)
        fixed = [t.fingerprint() for t in split_tasks(split)] if split != "rl_training" else None
        if split == "final_test_2":
            if seeded or [t.fingerprint() for t in tasks] != fixed:
                raise ValueError("The second final test runs every sealed task once, unseeded")
        else:
            reject_final2(tasks)
            if fixed is not None and [t.fingerprint() for t in tasks] != fixed[: len(tasks)]:
                raise ValueError("Tasks must be the fixed list for split " + split)
        self.tasks, self.arms, self.seeded = tasks, dict(arms), seeded
        self.models = {}
        for spec in self.arms.values():
            model = MODELS[spec["model_key"]]
            if model["base_sha256"] is None:
                raise ValueError("Base fingerprint not yet measured: " + model["model"])
            adapter = spec["adapter"]
            if adapter is not None and adapter["base_sha256"] != model["base_sha256"]:
                raise ValueError("Adapter was trained on a different base model")
            self.models[spec["model_key"]] = model
        self.destination = Path(destination)
        self.destination.mkdir(parents=True, exist_ok=False)
        self.config = {
            "protocol": self.protocol,
            "models": self.models,
            "budget": BUDGET,
            "tools": tools(),
            "system_message": SYSTEM,
            "arms": self.arms,
            "task_ids": [t.id for t in tasks],
            "full_task_list": fixed is None or len(tasks) == len(fixed),
            "seeded_failure": seeded,
            "seeded_actions": SEEDED_ACTIONS if seeded else 0,
            "split": split,
            "decoding": GREEDY,
            "new_training": False,
            "executor": executor.metadata(),
            "source_sha256": source_fingerprint(),
            "curriculum_sha256": curriculum_fingerprint(),
            "scaleup_sha256": fingerprint(),
            **self._extra_config(),
        }
        names = list(self.arms)
        self.states = {}
        for index, task in enumerate(tasks):
            offset = index % len(names)
            for name in names[offset:] + names[:offset]:
                episode = Episode(task, executor, max_actions=MAX_ACTIONS)
                messages = initial_messages(task)
                if seeded:
                    seed_failure(episode, messages)
                self.states[task.id + "--" + name] = {
                    "task": task,
                    "condition": name,
                    "episode": episode,
                    "messages": messages,
                    "model_calls": [],
                }
        self.config["ordered_jobs"] = list(self.states)
        write_json(self.destination / "config.json", self.config)
        write_json(self.destination / "status.json", {"status": "running"})
        self.pending, self.round_number, self.applied, self.environment = None, 0, set(), None

    def _extra_batch(self):
        return {"kind": "sized_eval", "decoding": GREEDY, "seed": 7}

    def export(self):
        if self.pending is not None:
            return self.pending
        requests = []
        for ident, state in self.states.items():
            if state["episode"].done:
                continue
            used = sum(c.get("output_tokens", 0) for c in state["model_calls"])
            requests.append(
                {
                    "episode_id": ident,
                    "condition": state["condition"],
                    "decision": len(state["model_calls"]) + 1,
                    "messages": json.loads(canonical(state["messages"])),
                    "max_new_tokens": min(512, 4096 - used),
                    "max_context_tokens": 12288,
                }
            )
        if not requests:
            return None
        self.round_number += 1
        self.pending = {
            "round": self.round_number,
            "protocol": self.protocol,
            "models": self.models,
            "tools": tools(),
            "budget": BUDGET,
            "arms": self.arms,
            "requests": requests,
            **self._extra_batch(),
        }
        self.pending["batch_sha256"] = sha256(canonical(self.pending))
        write_json(
            self.destination / "rounds" / str(self.round_number) / "public-requests.json",
            self.pending,
        )
        return self.pending

    def _check_metadata(self, batch_hash, metadata):
        if (
            self.pending is None
            or batch_hash in self.applied
            or batch_hash != self.pending["batch_sha256"]
        ):
            raise ValueError("Duplicate, stale, or unknown batch")
        if (
            metadata.get("protocol") != self.protocol
            or metadata.get("budget") != BUDGET
            or metadata.get("batch_sha256") != batch_hash
        ):
            raise ValueError("Wrong protocol, budget, or batch")
        loaded = metadata.get("models", {})
        if set(loaded) != set(self.models):
            raise ValueError("Wrong base models in model evidence")
        for key, model in self.models.items():
            evidence = loaded[key]
            if evidence.get("model_revision") != model["model_revision"]:
                raise ValueError("Wrong model revision")
            for name in ("base_initial_sha256", "base_final_sha256"):
                if evidence.get(name) != model["base_sha256"]:
                    raise ValueError("Base parameters changed or differ from the baseline")
        evidence = metadata.get("arms", {})
        if set(evidence) != set(self.arms):
            raise ValueError("Wrong arms in model evidence")
        for name, spec in self.arms.items():
            arm, adapter = evidence[name], spec["adapter"]
            if arm.get("model_key") != spec["model_key"]:
                raise ValueError("Arm ran on the wrong base model")
            if adapter is None:
                if arm.get("adapter_file_sha256") is not None:
                    raise ValueError("An adapter was attached to an untrained arm")
                continue
            if arm.get("adapter_file_sha256") != adapter["file_sha256"]:
                raise ValueError("Wrong adapter checkpoint file")
            for key in ("adapter_initial_sha256", "adapter_final_sha256"):
                if arm.get(key) != adapter["tensor_sha256"]:
                    raise ValueError("Adapter parameters changed or differ from the checkpoint")
        environment = {k: metadata.get(k) for k in ENVIRONMENT_KEYS}
        if self.environment is not None and environment != self.environment:
            raise ValueError("Inference environment changed between rounds")
        return environment

    def finish(self):
        report = super().finish()
        if fingerprint() != self.config["scaleup_sha256"]:
            raise ValueError("Source changed during the comparison")
        for row in report["rows"]:
            state = self.states[row["episode_id"]]
            row["debugging"] = debugging_metrics(state["task"], state["episode"].record())
        for name, arm in report["arms"].items():
            selected = [r["debugging"] for r in report["rows"] if r["condition"] == name]
            arm.update(
                later_edits=sum(d["later_edits"] for d in selected),
                repeated_edits=sum(d["repeated_edits"] for d in selected),
                real_recoveries=sum(d["real_recovery"] for d in selected),
            )
        report.update(models=self.models, full_task_list=self.config["full_task_list"])
        write_json(self.destination / "report.json", report)
        return report
