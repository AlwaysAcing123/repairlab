"""The same bounded tool contract for baseline evaluation and later training."""

import difflib
import json

from .execution import validate_source
from .tasks import canonical, sha256

INTERFACE_VERSION = "repairlab-tools-v1"
SYSTEM_MESSAGE = """Repair the given Python function to satisfy its specification.
Return one JSON object per turn, without Markdown or other text. Available actions:
{"tool":"read"}
{"tool":"test"}
{"tool":"edit","source":"def solve(...):\\n    ...\\n"}
{"tool":"submit"}
Edit replaces the complete function. Return exactly one function named solve,
without decorators, annotations, or default arguments. Preserve the signature.
Inputs and outputs are JSON values; preserve output types and do not mutate inputs.
Tests give public feedback. Final checks are independent and are not revealed.
You have 8 actions INCLUDING submit, and at most 8192 UTF-8 bytes per source.
Only a submitted repair can succeed. Explanations are not graded.
"""


class Episode:
    def __init__(self, task, executor, max_actions=8):
        if type(max_actions) is not int or max_actions < 1:
            raise ValueError("A positive action budget is required")
        self.task = task
        self.executor = executor
        self.max_actions = max_actions
        self.source = task.buggy
        self.events = []
        self.done = False
        self.result = None

    def initial_messages(self):
        system = SYSTEM_MESSAGE.replace("8 actions", f"{self.max_actions} actions")
        return [
            {"role": "system", "content": system},
            {"role": "user", "content": canonical(self.task.public())},
        ]

    def step(self, action):
        if self.done:
            raise ValueError("Episode has ended")
        if len(self.events) >= self.max_actions:
            raise ValueError("Action budget exhausted")
        before = self.source
        try:
            if not isinstance(action, dict):
                raise ValueError("Action must be a JSON object")
            tool = action.get("tool")
            required = {"tool", "source"} if tool == "edit" else {"tool"}
            if set(action) != required:
                raise ValueError("Unexpected or missing action fields")
            if tool == "read":
                observation = {"source": self.source}
            elif tool == "test":
                observation = self.executor.run(self.source, self.task.public_cases, reveal=True)
            elif tool == "edit":
                validate_source(action["source"], self.task.signature)
                self.source = action["source"]
                observation = {"edited": True}
            elif tool == "submit":
                # Private grades are recorded for the researcher, never fed back to the model.
                public = self.executor.run(self.source, self.task.public_cases)
                private = self.executor.run(self.source, self.task.private_cases)
                self.result = {
                    "success": public["passed"] and private["passed"],
                    "reward": int(public["passed"] and private["passed"]),
                    "termination": "submitted",
                    "public": public,
                    "private": private,
                }
                self.done = True
                observation = {"submitted": True}
            else:
                raise ValueError("Unknown tool; choose read, test, edit, or submit")
        except (ValueError, TypeError) as exc:
            observation = {"error": str(exc)}
        event = {
            "index": len(self.events) + 1,
            "action": action,
            "observation": observation,
            "source_sha256": sha256(self.source),
        }
        if self.source != before:
            event["diff"] = "".join(
                difflib.unified_diff(
                    before.splitlines(keepends=True),
                    self.source.splitlines(keepends=True),
                    fromfile="before.py",
                    tofile="after.py",
                )
            )
        self.events.append(event)
        if not self.done and len(self.events) == self.max_actions:
            self.stop("action_budget")
        return observation

    def stop(self, reason):
        if not self.done:
            self.done = True
            self.result = {"success": False, "reward": 0, "termination": reason}

    def record(self):
        if not self.done:
            raise ValueError("Cannot save an unfinished episode")
        return {
            "interface_version": INTERFACE_VERSION,
            "task": self.task.public(),
            "task_sha256": self.task.fingerprint(),
            "max_actions": self.max_actions,
            "events": self.events,
            "final_source": self.source,
            "final_source_sha256": sha256(self.source),
            "result": self.result,
            "actions": len(self.events),
            "test_calls": sum(
                e["action"].get("tool") == "test"
                for e in self.events
                if isinstance(e["action"], dict)
            ),
            "edits": sum(
                e["action"].get("tool") == "edit"
                for e in self.events
                if isinstance(e["action"], dict)
            ),
            "invalid_actions": sum("error" in e["observation"] for e in self.events),
        }


def parse_action(text):
    from .execution import reject_constant, unique_keys

    try:
        return json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    except (ValueError, TypeError, RecursionError):
        # Represent malformed text as an invalid action; it still consumes an action.
        return {"invalid_model_output": str(text)[:8192]}
