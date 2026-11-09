"""Credential-free staged transport for the matched revision experiment.

Only public model conversations are exported. Candidate execution and all private
grading remain local. One batch advances each unfinished episode by one decision.
"""

import json
from collections import Counter

from .episode import Episode, parse_action
from .experiments import new_run, source_fingerprint, write_json
from .revision import BUDGET, MODEL, PROTOCOL, REVISION, SYSTEM, FeedbackExecutor
from .tasks import canonical, pilot_tasks, sha256


def parse_batch_action(raw, allow_json_fences=False):
    if allow_json_fences and isinstance(raw, str):
        lines = raw.strip().splitlines()
        if len(lines) >= 3 and lines[0] in {"```json", "```"} and lines[-1] == "```":
            raw = "\n".join(lines[1:-1])
    return parse_action(raw)


class BatchRevision:
    def __init__(self, executor, destination, tasks=None, allow_json_fences=False):
        self.tasks = pilot_tasks() if tasks is None else list(tasks)
        self.destination, self.config = new_run(destination, PROTOCOL, executor)
        self.config.update(
            protocol=PROTOCOL + ("-json-fences" if allow_json_fences else ""),
            allow_json_fences=allow_json_fences,
            transport="staged-private-kaggle-batches",
            system_message=SYSTEM,
            budget=BUDGET,
            model=MODEL,
            model_revision=REVISION,
            seed=7,
            credentials_uploaded=False,
        )
        self.states = {}
        for index, task in enumerate(self.tasks):
            arms = (
                ("without_feedback", "with_feedback")
                if index % 2 == 0
                else ("with_feedback", "without_feedback")
            )
            for arm in arms:
                ident = task.id + "--" + arm
                self.states[ident] = {
                    "condition": arm,
                    "task": task,
                    "episode": Episode(
                        task,
                        FeedbackExecutor(executor, arm == "with_feedback"),
                        max_actions=BUDGET["max_actions"],
                    ),
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": canonical(task.public())},
                    ],
                    "model_calls": [],
                }
        self.config["ordered_jobs"] = list(self.states)
        write_json(self.destination / "config.json", self.config)
        self.round_number = 0
        self.pending = None
        self.initial_model_metadata = None
        self.applied_batches = set()

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
                    "decision": len(state["model_calls"]) + 1,
                    "messages": json.loads(canonical(state["messages"])),
                    "max_new_tokens": min(
                        BUDGET["max_call_output_tokens"], BUDGET["max_episode_output_tokens"] - used
                    ),
                    "max_context_tokens": BUDGET["max_context_tokens"],
                }
            )
        if not requests:
            return None
        self.round_number += 1
        self.pending = {
            "round": self.round_number,
            "protocol": self.config["protocol"],
            "model": MODEL,
            "model_revision": REVISION,
            "budget": BUDGET,
            "requests": requests,
        }
        self.pending["batch_sha256"] = sha256(canonical(self.pending))
        write_json(
            self.destination / "rounds" / str(self.round_number) / "public-requests.json",
            self.pending,
        )
        return self.pending

    def apply(self, batch_hash, metadata, outputs):
        if batch_hash in self.applied_batches:
            raise ValueError("Batch already applied; no duplicate execution permitted")
        if self.pending is None or batch_hash != self.pending["batch_sha256"]:
            raise ValueError("Wrong or stale batch")
        if metadata.get("model") != MODEL or metadata.get("model_revision") != REVISION:
            raise ValueError("Wrong model revision")
        if metadata.get("protocol") != self.config["protocol"] or metadata.get("budget") != BUDGET:
            raise ValueError("Wrong protocol or budget")
        initial_hash = metadata.get("initial_weight_sha256")
        if not isinstance(initial_hash, str) or len(initial_hash) != 64:
            raise ValueError("Missing initial weight fingerprint")
        if initial_hash != metadata.get("final_weight_sha256"):
            raise ValueError("Weights changed during inference")
        if self.initial_model_metadata is not None:
            for key in (
                "initial_weight_sha256",
                "model",
                "model_revision",
                "dtype",
                "gpu",
                "packages",
                "seed",
                "decoding",
                "chat_template_sha256",
            ):
                if metadata.get(key) != self.initial_model_metadata.get(key):
                    raise ValueError("Inference environment changed across rounds: " + key)
        expected = {r["episode_id"]: r for r in self.pending["requests"]}
        if len(outputs) != len(expected) or len({r["episode_id"] for r in outputs}) != len(outputs):
            raise ValueError("Missing or duplicate model responses")
        if {r["episode_id"] for r in outputs} != set(expected):
            raise ValueError("Unexpected model episode IDs")
        # Validate the whole batch before any candidate is executed.
        for row in outputs:
            request = expected[row["episode_id"]]
            if row.get("decision") != request["decision"]:
                raise ValueError("Wrong decision number")
            if row.get("generation_error"):
                raise ValueError("GPU generation failed; diagnose before continuing")
            if row.get("stop"):
                if row["stop"] not in {"context_budget", "output_token_budget"}:
                    raise ValueError("Invalid stop reason")
                continue
            if (
                not isinstance(row.get("raw"), str)
                or len(row["raw"].encode()) > 32768
                or type(row.get("input_tokens")) is not int
                or row["input_tokens"] < 1
                or type(row.get("output_tokens")) is not int
                or row["output_tokens"] < 1
                or row["output_tokens"] > request["max_new_tokens"]
                or row["input_tokens"] + request["max_new_tokens"] > request["max_context_tokens"]
            ):
                raise ValueError("Invalid model response or token budget violation")
        self.initial_model_metadata = (
            metadata if self.initial_model_metadata is None else self.initial_model_metadata
        )
        for row in outputs:
            state = self.states[row["episode_id"]]
            episode = state["episode"]
            state["model_calls"].append(
                {**row, "messages": json.loads(canonical(state["messages"]))}
            )
            if row.get("stop"):
                episode.stop(row["stop"])
            else:
                observation = episode.step(
                    parse_batch_action(row["raw"], self.config["allow_json_fences"])
                )
                state["messages"].extend(
                    [
                        {"role": "assistant", "content": row["raw"]},
                        {"role": "user", "content": "Tool result: " + canonical(observation)},
                    ]
                )
            if episode.done:
                record = {
                    "condition": state["condition"],
                    "actor": "original_model",
                    "model_calls": state["model_calls"],
                    **episode.record(),
                }
                write_json(
                    self.destination
                    / "episodes"
                    / state["condition"]
                    / (state["task"].id + ".json"),
                    record,
                )
                print(
                    row["episode_id"],
                    record["result"]["success"],
                    record["result"]["termination"],
                    flush=True,
                )
        self.applied_batches.add(batch_hash)
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

    def finish(self):
        if self.pending is not None or any(not s["episode"].done for s in self.states.values()):
            raise ValueError("Unfinished episodes")
        if source_fingerprint() != self.config["source_sha256"]:
            raise ValueError("Local source changed during run")
        rows = []
        for ident, state in self.states.items():
            record = state["episode"].record()
            rows.append(
                {
                    "episode_id": ident,
                    "task_id": state["task"].id,
                    "condition": state["condition"],
                    "family": state["task"].family,
                    "success": record["result"]["success"],
                    "termination": record["result"]["termination"],
                    "actions": record["actions"],
                    "test_calls": record["test_calls"],
                    "edits": record["edits"],
                    "invalid_actions": record["invalid_actions"],
                    "input_tokens": sum(c.get("input_tokens", 0) for c in state["model_calls"]),
                    "output_tokens": sum(c.get("output_tokens", 0) for c in state["model_calls"]),
                }
            )
        arms = {}
        for arm in ("without_feedback", "with_feedback"):
            selected = [r for r in rows if r["condition"] == arm]
            arms[arm] = {
                "tasks": len(selected),
                "successes": sum(r["success"] for r in selected),
                "termination_counts": dict(Counter(r["termination"] for r in selected)),
                **{
                    key: sum(r[key] for r in selected)
                    for key in (
                        "actions",
                        "test_calls",
                        "edits",
                        "invalid_actions",
                        "input_tokens",
                        "output_tokens",
                    )
                },
            }
        pairs = [
            {
                arm: self.states[task.id + "--" + arm]["episode"].result["success"]
                for arm in ("without_feedback", "with_feedback")
            }
            for task in self.tasks
        ]
        report = {
            "protocol": self.config["protocol"],
            "transport": "staged-private-kaggle-batches",
            "split": "development",
            "source_unchanged": True,
            "weights_unchanged": True,
            "rounds": self.round_number,
            "arms": arms,
            "rows": rows,
            "paired": {
                "both_pass": sum(all(p.values()) for p in pairs),
                "feedback_only": sum(
                    p["with_feedback"] and not p["without_feedback"] for p in pairs
                ),
                "no_feedback_only": sum(
                    p["without_feedback"] and not p["with_feedback"] for p in pairs
                ),
                "both_fail": sum(not any(p.values()) for p in pairs),
            },
        }
        write_json(self.destination / "report.json", report)
        write_json(self.destination / "status.json", {"status": "completed", "episodes": len(rows)})
        return report
