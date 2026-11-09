"""Matched checkpoint comparison with any number of arms.

Autonomous mode follows the frozen pilot rules. Seeded mode starts each attempt after
a scripted read, retained-bug edit, and failing test, then lets the model continue.
That measures recovery on every task, but it is not an autonomous repair score.

Sealed tasks are refused unless ``split="final_test"`` names every sealed task, in
order and unseeded. That is the single path for opening the final test.
"""

import json
from collections import Counter
from pathlib import Path

from ..episode import Episode
from ..experiments import source_fingerprint, write_json
from ..native_repair import SYSTEM
from ..native_tools import parsed_action, tools
from ..revision import BUDGET, MODEL, REVISION
from ..tasks import canonical, sha256
from . import fingerprint
from .final_test import final_tasks
from .splits import reject_final
from .transcript import initial_messages, retained_repair, scripted_step, tool_message

PROTOCOL = "native-matched-comparison-v1"
SEEDED_ACTIONS = 3
MAX_ACTIONS = 8
ENVIRONMENT_KEYS = ("gpu", "dtype", "packages", "chat_template_sha256", "decoding", "seed")


def seed_failure(episode, messages):
    task = episode.task
    scripted_step(episode, messages, "read")
    scripted_step(episode, messages, "edit", {"source": retained_repair(task.buggy)})
    observation = scripted_step(episode, messages, "test")
    if observation["passed"] is not False:
        raise ValueError("Seeded failure unexpectedly passed: " + task.id)


def recovery_metrics(events):
    """What the model did after each failed public test, from saved events only."""
    tools_used = [e["action"].get("tool") for e in events]
    failed = [
        i
        for i, e in enumerate(events)
        if tools_used[i] == "test" and e["observation"].get("passed") is False
    ]
    if not failed:
        return {"saw_failure": False}
    first = failed[0]
    revised = [
        i
        for i in range(first + 1, len(events))
        if tools_used[i] == "edit"
        and "diff" in events[i]
        and "error" not in events[i]["observation"]
    ]
    return {
        "saw_failure": True,
        "next_tool": tools_used[first + 1] if first + 1 < len(events) else None,
        "revised": bool(revised),
        "retested": any(tools_used[i] == "test" for i in range(revised[0] + 1, len(events)))
        if revised
        else False,
        "submitted_on_failure": tools_used[-1] == "submit"
        and events[-1]["source_sha256"] == events[failed[-1]]["source_sha256"],
    }


class MatchedComparison:
    protocol = PROTOCOL

    def __init__(self, executor, destination, arms, tasks, seeded=False, split="development"):
        tasks = list(tasks)
        if split == "final_test":
            sealed = [t.fingerprint() for t in final_tasks()]
            if seeded or [t.fingerprint() for t in tasks] != sealed:
                raise ValueError("The final test runs every sealed task once, unseeded")
        else:
            reject_final(tasks)
        self.tasks, self.arms, self.seeded = tasks, dict(arms), seeded
        self.destination = Path(destination)
        self.destination.mkdir(parents=True, exist_ok=False)
        self.config = {
            "protocol": self.protocol,
            "model": MODEL,
            "model_revision": REVISION,
            "budget": BUDGET,
            "tools": tools(),
            "system_message": SYSTEM,
            "arms": self.arms,
            "base_sha256": next(a["base_sha256"] for a in self.arms.values() if a),
            "task_ids": [t.id for t in tasks],
            "seeded_failure": seeded,
            "seeded_actions": SEEDED_ACTIONS if seeded else 0,
            "split": split,
            "new_training": False,
            "executor": executor.metadata(),
            "source_sha256": source_fingerprint(),
            "curriculum_sha256": fingerprint(),
            **self._extra_config(),
        }
        names = list(self.arms)
        self.states = {}
        for index, task in enumerate(tasks):
            offset = index % len(names)
            for arm in names[offset:] + names[:offset]:
                episode = Episode(task, executor, max_actions=MAX_ACTIONS)
                messages = initial_messages(task)
                if seeded:
                    seed_failure(episode, messages)
                self.states[task.id + "--" + arm] = {
                    "task": task,
                    "condition": arm,
                    "episode": episode,
                    "messages": messages,
                    "model_calls": [],
                }
        self.config["ordered_jobs"] = list(self.states)
        write_json(self.destination / "config.json", self.config)
        write_json(self.destination / "status.json", {"status": "running"})
        self.pending, self.round_number, self.applied, self.environment = None, 0, set(), None

    def _extra_config(self):
        return {}

    def _extra_batch(self):
        return {}

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
            "model": MODEL,
            "model_revision": REVISION,
            "tools": tools(),
            "budget": BUDGET,
            "arms": self.arms,
            "base_sha256": self.config["base_sha256"],
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
            or metadata.get("model_revision") != REVISION
            or metadata.get("batch_sha256") != batch_hash
        ):
            raise ValueError("Wrong protocol, budget, model, or batch")
        evidence = metadata.get("arms", {})
        if set(evidence) != set(self.arms):
            raise ValueError("Wrong arms in model evidence")
        for name, adapter in self.arms.items():
            arm = evidence[name]
            for key in ("base_initial_sha256", "base_final_sha256"):
                if arm.get(key) != self.config["base_sha256"]:
                    raise ValueError("Base parameters changed or differ from the baseline")
            if adapter is None:
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

    def _check_outputs(self, outputs):
        requests = {r["episode_id"]: r for r in self.pending["requests"]}
        ids = [r["episode_id"] for r in outputs]
        if len(ids) != len(requests) or set(ids) != set(requests) or len(set(ids)) != len(ids):
            raise ValueError("Missing, duplicate, or unexpected responses")
        for row in outputs:
            request = requests[row["episode_id"]]
            if (
                row.get("decision") != request["decision"]
                or row.get("condition") != request["condition"]
            ):
                raise ValueError("Wrong response identity")
            if row.get("generation_error"):
                raise ValueError("Generation failure; diagnose before continuing")
            if row.get("stop"):
                if row["stop"] not in {"context_budget", "output_token_budget"}:
                    raise ValueError("Invalid model stop reason")
                continue
            if (
                not isinstance(row.get("raw"), str)
                or len(row["raw"].encode()) > 32768
                or type(row.get("input_tokens")) is not int
                or row["input_tokens"] < 1
                or type(row.get("output_tokens")) is not int
                or not 0 < row["output_tokens"] <= request["max_new_tokens"]
                or row["input_tokens"] + request["max_new_tokens"] > request["max_context_tokens"]
            ):
                raise ValueError("Invalid generation or token budget")

    def apply(self, batch_hash, metadata, outputs):
        environment = self._check_metadata(batch_hash, metadata)
        self._check_outputs(outputs)
        self.environment = environment
        for row in outputs:
            state = self.states[row["episode_id"]]
            episode = state["episode"]
            call = {**row, "messages": json.loads(canonical(state["messages"]))}
            if row.get("stop"):
                episode.stop(row["stop"])
            else:
                try:
                    action = parsed_action(row.get("parsed"))
                    native_error = None
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    native_error = str(exc)
                    action = {"invalid_native_output": row["raw"][:8192]}
                observation = episode.step(action)
                call["native_error"] = native_error
                if native_error is None:
                    state["messages"].extend(
                        [row["parsed"], tool_message(action["tool"], observation)]
                    )
                else:
                    state["messages"].extend(
                        [
                            {"role": "assistant", "content": row["raw"]},
                            {
                                "role": "user",
                                "content": "Native action rejected: "
                                + native_error
                                + ". Use exactly one valid native tool call. "
                                + canonical(observation),
                            },
                        ]
                    )
            state["model_calls"].append(call)
            if episode.done:
                self._save_episode(state)
        self.applied.add(batch_hash)
        self.pending = None
        write_json(
            self.destination / "status.json",
            {
                "status": "round_complete",
                "rounds": self.round_number,
                "episodes_finished": sum(s["episode"].done for s in self.states.values()),
                "episodes_planned": len(self.states),
            },
        )

    def _save_episode(self, state):
        record = {
            "actor": "model",
            "condition": state["condition"],
            "seeded_actions": self.config["seeded_actions"],
            "model_calls": state["model_calls"],
            **state["episode"].record(),
        }
        write_json(
            self.destination / "episodes" / state["condition"] / (state["task"].id + ".json"),
            record,
        )
        print(
            state["task"].id,
            state["condition"],
            record["result"]["success"],
            record["result"]["termination"],
            flush=True,
        )

    def finish(self):
        if self.pending is not None or any(not s["episode"].done for s in self.states.values()):
            raise ValueError("Unfinished episodes")
        if (
            source_fingerprint() != self.config["source_sha256"]
            or fingerprint() != self.config["curriculum_sha256"]
        ):
            raise ValueError("Source changed during the comparison")
        rows = []
        for ident, state in self.states.items():
            record = state["episode"].record()
            rows.append(
                {
                    "episode_id": ident,
                    "task_id": state["task"].id,
                    "family": state["task"].family,
                    "condition": state["condition"],
                    "success": record["result"]["success"],
                    "termination": record["result"]["termination"],
                    "actions": record["actions"],
                    "edits": record["edits"],
                    "test_calls": record["test_calls"],
                    "invalid_actions": record["invalid_actions"],
                    "output_tokens": sum(c.get("output_tokens", 0) for c in state["model_calls"]),
                    "recovery": recovery_metrics(record["events"]),
                }
            )
        arms = {}
        for arm in self.arms:
            selected = [r for r in rows if r["condition"] == arm]
            recoveries = [r["recovery"] for r in selected if r["recovery"]["saw_failure"]]
            arms[arm] = {
                "tasks": len(selected),
                "successes": sum(r["success"] for r in selected),
                "termination_counts": dict(Counter(r["termination"] for r in selected)),
                "invalid_actions": sum(r["invalid_actions"] for r in selected),
                "output_tokens": sum(r["output_tokens"] for r in selected),
                "episodes_with_failed_test": len(recoveries),
                "next_tool_after_first_failure": dict(
                    Counter(str(r["next_tool"]) for r in recoveries)
                ),
                "revised_after_failure": sum(r["revised"] for r in recoveries),
                "retested_after_revision": sum(r["retested"] for r in recoveries),
                "submitted_on_failure": sum(r["submitted_on_failure"] for r in recoveries),
                "recovered": sum(r["success"] for r in selected if r["recovery"]["saw_failure"]),
            }
        report = {
            "protocol": self.protocol,
            "split": self.config["split"],
            "seeded_failure": self.seeded,
            "tasks_per_condition": len(self.tasks),
            "rounds": self.round_number,
            "arms": arms,
            "rows": rows,
            "new_training": False,
        }
        write_json(self.destination / "report.json", report)
        write_json(self.destination / "status.json", {"status": "completed", "episodes": len(rows)})
        return report
