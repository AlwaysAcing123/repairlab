"""Six-task native autonomous repair comparison using a frozen supervised adapter."""

import json
from collections import Counter

from .episode import Episode
from .experiments import new_run, source_fingerprint, write_json
from .native_tools import SYSTEM as DIAGNOSTIC_SYSTEM
from .native_tools import parsed_action, tools
from .revision import BUDGET, MODEL, REVISION
from .tasks import canonical, pilot_tasks, sha256

PROTOCOL = "native-autonomous-repair-pilot-v1"
TASK_IDS = (
    "quantity-range",
    "closed-window",
    "positive-total",
    "stable-unique",
    "retry-decision",
    "capped-backoff",
)
SYSTEM = (
    DIAGNOSTIC_SYSTEM.replace(
        "This session is a tool-interface diagnostic.",
        "Repair the supplied function to satisfy its specification and public examples, then submit the corrected function.",
    )
    + """
You may read, test, edit, and retest as needed. Preserve the supplied solve signature.
The source must contain exactly one function. Inputs and outputs are JSON values;
preserve output types and do not mutate inputs. Only public tests are available.
Final correctness checks are independent and are never revealed during the attempt.
You have at most 8 actions INCLUDING submit, 512 generated tokens per call, and 4096
generated tokens in the entire episode. Only a submitted repair can succeed.
"""
)


def selected_tasks():
    by_id = {t.id: t for t in pilot_tasks()}
    return [by_id[ident] for ident in TASK_IDS]


class NativeRepairPilot:
    def __init__(self, executor, destination, adapter, tasks=None):
        self.tasks = selected_tasks() if tasks is None else list(tasks)
        self.destination, self.config = new_run(destination, PROTOCOL, executor)
        self.config.update(
            protocol=PROTOCOL,
            model=MODEL,
            model_revision=REVISION,
            budget=BUDGET,
            tools=tools(),
            system_message=SYSTEM,
            adapter=adapter,
            task_ids=[t.id for t in self.tasks],
            conditions=["original", "supervised"],
            transport="staged-private-kaggle-batches",
            seed=7,
            new_training=False,
        )
        self.states = {}
        for index, task in enumerate(self.tasks):
            arms = ("original", "supervised") if index % 2 == 0 else ("supervised", "original")
            for arm in arms:
                self.states[task.id + "--" + arm] = {
                    "task": task,
                    "condition": arm,
                    "episode": Episode(task, executor, max_actions=8),
                    "messages": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": canonical(task.public())},
                    ],
                    "model_calls": [],
                }
        self.config["ordered_jobs"] = list(self.states)
        write_json(self.destination / "config.json", self.config)
        self.pending, self.round_number = None, 0
        self.applied = set()
        self.environment = None

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
            "protocol": PROTOCOL,
            "model": MODEL,
            "model_revision": REVISION,
            "tools": tools(),
            "budget": BUDGET,
            "adapter": self.config["adapter"],
            "requests": requests,
        }
        self.pending["batch_sha256"] = sha256(canonical(self.pending))
        write_json(
            self.destination / "rounds" / str(self.round_number) / "public-requests.json",
            self.pending,
        )
        return self.pending

    def apply(self, batch_hash, metadata, outputs):
        if (
            self.pending is None
            or batch_hash in self.applied
            or batch_hash != self.pending["batch_sha256"]
        ):
            raise ValueError("Duplicate, stale, or unknown native batch")
        if metadata.get("protocol") != PROTOCOL or metadata.get("budget") != BUDGET:
            raise ValueError("Wrong native protocol or budget")
        if metadata.get("model_revision") != REVISION or metadata.get("batch_sha256") != batch_hash:
            raise ValueError("Wrong model or batch")
        adapter = self.config["adapter"]
        if metadata.get("adapter_file_sha256") != adapter["file_sha256"]:
            raise ValueError("Wrong adapter checkpoint file")
        for name in ("original", "adapted_base"):
            if (
                metadata.get(name + "_initial_sha256") != adapter["base_sha256"]
                or metadata.get(name + "_final_sha256") != adapter["base_sha256"]
            ):
                raise ValueError("Base parameters changed or differ from the baseline")
        if (
            metadata.get("adapter_initial_sha256") != adapter["tensor_sha256"]
            or metadata.get("adapter_final_sha256") != adapter["tensor_sha256"]
        ):
            raise ValueError("Adapter parameters changed or differ from the saved checkpoint")
        environment = {
            k: metadata.get(k)
            for k in ("gpu", "dtype", "packages", "chat_template_sha256", "decoding", "seed")
        }
        if self.environment is not None and environment != self.environment:
            raise ValueError("Inference environment changed between rounds")
        requests = {r["episode_id"]: r for r in self.pending["requests"]}
        if (
            len(outputs) != len(requests)
            or len({r["episode_id"] for r in outputs}) != len(outputs)
            or set(requests) != {r["episode_id"] for r in outputs}
        ):
            raise ValueError("Missing, duplicate, or unexpected native responses")
        for row in outputs:
            request = requests[row["episode_id"]]
            if (
                row.get("decision") != request["decision"]
                or row.get("condition") != request["condition"]
            ):
                raise ValueError("Wrong native response identity")
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
                raise ValueError("Invalid native generation or token budget")
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
                        [
                            row["parsed"],
                            {
                                "role": "tool",
                                "name": action["tool"],
                                "content": canonical(
                                    {k: v for k, v in observation.items() if k != "elapsed_seconds"}
                                ),
                            },
                        ]
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
                record = {
                    "actor": "model",
                    "condition": state["condition"],
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

    def finish(self):
        if self.pending is not None or any(not s["episode"].done for s in self.states.values()):
            raise ValueError("Unfinished native episodes")
        if source_fingerprint() != self.config["source_sha256"]:
            raise ValueError("Source changed during pilot")
        rows = []
        for ident, state in self.states.items():
            episode = state["episode"].record()
            rows.append(
                {
                    "episode_id": ident,
                    "task_id": state["task"].id,
                    "family": state["task"].family,
                    "condition": state["condition"],
                    "success": episode["result"]["success"],
                    "termination": episode["result"]["termination"],
                    "actions": episode["actions"],
                    "edits": episode["edits"],
                    "test_calls": episode["test_calls"],
                    "invalid_actions": episode["invalid_actions"],
                    "input_tokens": sum(c.get("input_tokens", 0) for c in state["model_calls"]),
                    "output_tokens": sum(c.get("output_tokens", 0) for c in state["model_calls"]),
                }
            )
        arms = {}
        for arm in ("original", "supervised"):
            selected = [r for r in rows if r["condition"] == arm]
            arms[arm] = {
                "tasks": len(selected),
                "successes": sum(r["success"] for r in selected),
                "termination_counts": dict(Counter(r["termination"] for r in selected)),
                **{
                    k: sum(r[k] for r in selected)
                    for k in (
                        "actions",
                        "edits",
                        "test_calls",
                        "invalid_actions",
                        "input_tokens",
                        "output_tokens",
                    )
                },
            }
        pairs = [
            {
                arm: self.states[t.id + "--" + arm]["episode"].result["success"]
                for arm in ("original", "supervised")
            }
            for t in self.tasks
        ]
        report = {
            "protocol": PROTOCOL,
            "split": "development",
            "tasks_per_condition": len(self.tasks),
            "arms": arms,
            "rows": rows,
            "source_unchanged": True,
            "weights_unchanged": True,
            "rounds": self.round_number,
            "new_training": False,
            "paired": {
                "both_pass": sum(all(p.values()) for p in pairs),
                "adapter_only": sum(p["supervised"] and not p["original"] for p in pairs),
                "original_only": sum(p["original"] and not p["supervised"] for p in pairs),
                "both_fail": sum(not any(p.values()) for p in pairs),
            },
        }
        write_json(self.destination / "report.json", report)
        write_json(self.destination / "status.json", {"status": "completed", "episodes": len(rows)})
        return report
