"""Paired revision experiment: remote model inference, local protected execution.

The authenticated HTTP surface exposes only current public messages and bounded
actions. It has no filesystem, terminal, reference-solution, or private-grade route.
"""

import argparse
import hmac
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from .episode import Episode, parse_action
from .execution import DockerExecutor, InfrastructureError, reject_constant, unique_keys
from .experiments import new_run, source_fingerprint, write_json
from .tasks import canonical, pilot_tasks, sha256

PROTOCOL = "paired-revision-feedback-v1"
MODEL = "Qwen/Qwen3-0.6B"
REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
BUDGET = {
    "max_actions": 8,
    "max_episode_output_tokens": 4096,
    "max_call_output_tokens": 512,
    "max_context_tokens": 12288,
}
SYSTEM = """You repair Python functions. Use the following tools by returning exactly ONE JSON object per turn:
{"tool":"read"}
{"tool":"test"}
{"tool":"edit","source":"def solve(...):\\n    ...\\n"}
{"tool":"submit"}
Do not use Markdown or put text outside the JSON object. Edit replaces the ENTIRE function.
Preserve the given solve signature; do not use annotations, decorators, or defaults.
Inputs and outputs are JSON values. Preserve output types and do not mutate inputs.
Public examples are provided. The test tool returns execution feedback when available;
otherwise it returns a fixed feedback-unavailable message. Independent final checks
are never revealed. You may inspect, revise, and test within 8 total actions INCLUDING
submit. Each response is limited to 512 tokens, with at most 4096 generated tokens per
episode. Only submitting a repair can succeed. Submit before your action budget ends.
"""


class FeedbackExecutor:
    def __init__(self, executor, enabled):
        self.executor, self.enabled = executor, enabled

    def run(self, source, cases, *, reveal=False):
        if reveal and not self.enabled:
            return {
                "feedback": "unavailable",
                "message": "Execution feedback is withheld in this condition.",
            }
        result = self.executor.run(source, cases, reveal=reveal)
        # Wall-clock noise must not change the model's next prompt.
        return {k: v for k, v in result.items() if k != "elapsed_seconds"} if reveal else result


class RevisionExperiment:
    def __init__(self, executor, destination, tasks=None):
        self.tasks = pilot_tasks() if tasks is None else list(tasks)
        self.destination, self.config = new_run(destination, PROTOCOL, executor)
        self.executor = executor
        self.jobs = []
        for index, task in enumerate(self.tasks):
            # Alternate pair order to reduce a systematic first/second-arm effect.
            arms = (
                ("without_feedback", "with_feedback")
                if index % 2 == 0
                else ("with_feedback", "without_feedback")
            )
            self.jobs.extend((task, arm) for arm in arms)
        self.config.update(
            protocol=PROTOCOL,
            system_message=SYSTEM,
            budget=BUDGET,
            model=MODEL,
            model_revision=REVISION,
            seed=7,
            ordered_jobs=[f"{t.id}--{arm}" for t, arm in self.jobs],
        )
        write_json(self.destination / "config.json", self.config)
        self.position = 0
        self.episode = None
        self.messages = []
        self.calls = []
        self.records = []
        self.cache = {}
        self.model_metadata = None
        self.failed = None
        self.finished = False
        self.lock = threading.RLock()
        self._status("waiting_for_model")

    def _status(self, status):
        write_json(
            self.destination / "status.json",
            {
                "status": status,
                "episodes_finished": len(self.records),
                "episodes_planned": len(self.jobs),
                "error": self.failed,
            },
        )

    def metadata(self, data):
        if data.get("model") != MODEL or data.get("model_revision") != REVISION:
            raise ValueError("Wrong model or revision")
        if data.get("protocol") != PROTOCOL or data.get("budget") != BUDGET:
            raise ValueError("Wrong protocol or budgets")
        if (
            not isinstance(data.get("initial_weight_sha256"), str)
            or len(data["initial_weight_sha256"]) != 64
        ):
            raise ValueError("Initial weight fingerprint required")
        if self.model_metadata is not None and data != self.model_metadata:
            raise ValueError("Model metadata changed during experiment")
        self.model_metadata = data
        write_json(self.destination / "model-config.json", data)
        return {"accepted": True}

    def work(self):
        if self.failed:
            raise InfrastructureError(self.failed)
        if self.model_metadata is None:
            raise ValueError("Register model metadata first")
        if self.position == len(self.jobs):
            return {"done": True}
        if self.episode is None:
            task, arm = self.jobs[self.position]
            self.episode = Episode(
                task,
                FeedbackExecutor(self.executor, arm == "with_feedback"),
                max_actions=BUDGET["max_actions"],
            )
            self.messages = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": canonical(task.public())},
            ]
            self.calls = []
            self._status("evaluating")
        task, arm = self.jobs[self.position]
        used = sum(c["output_tokens"] for c in self.calls)
        return {
            "done": False,
            "episode_id": f"{task.id}--{arm}",
            "decision": len(self.calls) + 1,
            "messages": json.loads(canonical(self.messages)),
            "max_new_tokens": min(
                BUDGET["max_call_output_tokens"], BUDGET["max_episode_output_tokens"] - used
            ),
            "max_context_tokens": BUDGET["max_context_tokens"],
        }

    def action(self, request):
        ident = request.get("request_id")
        if not isinstance(ident, str) or len(ident) > 160:
            raise ValueError("Missing request ID")
        digest = sha256(canonical(request))
        if ident in self.cache:
            previous_digest, reply = self.cache[ident]
            if previous_digest != digest:
                raise ValueError("A request ID cannot be reused with different content")
            return reply
        work = self.work()
        if work["done"]:
            raise ValueError("No active episode")
        if ident != f"{work['episode_id']}:{work['decision']}":
            raise ValueError("Stale or out-of-order action")
        if request.get("stop"):
            if request["stop"] not in {"context_budget", "generation_error", "output_token_budget"}:
                raise ValueError("Invalid stop reason")
            self.episode.stop(request["stop"])
            self.calls.append(
                {
                    "stop": request["stop"],
                    "output_tokens": 0,
                    "input_tokens": request.get("input_tokens", 0),
                    "messages": list(self.messages),
                }
            )
            observation = {"stopped": True}
        else:
            raw = request.get("raw")
            input_tokens, output_tokens = request.get("input_tokens"), request.get("output_tokens")
            if not isinstance(raw, str) or len(raw.encode()) > 32768:
                raise ValueError("Invalid raw generation")
            if (
                type(input_tokens) is not int
                or type(output_tokens) is not int
                or input_tokens < 1
                or output_tokens < 1
                or output_tokens > work["max_new_tokens"]
                or input_tokens + work["max_new_tokens"] > work["max_context_tokens"]
            ):
                raise ValueError("Token budget violation")
            call = {
                "decision": work["decision"],
                "raw": raw,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "allowance": work["max_new_tokens"],
                "messages": list(self.messages),
                "elapsed_seconds": request.get("elapsed_seconds"),
                "hit_token_cap": output_tokens == work["max_new_tokens"],
            }
            observation = self.episode.step(parse_action(raw))
            self.calls.append(call)
            self.messages.extend(
                [
                    {"role": "assistant", "content": raw},
                    {"role": "user", "content": "Tool result: " + canonical(observation)},
                ]
            )
        reply = {"accepted": True, "episode_done": self.episode.done, "observation": observation}
        if self.episode.done:
            self._save_episode()
        self.cache[ident] = (digest, reply)
        return reply

    def _save_episode(self):
        task, arm = self.jobs[self.position]
        record = {
            "actor": "original_model",
            "condition": arm,
            "model_calls": self.calls,
            **self.episode.record(),
        }
        write_json(self.destination / "episodes" / arm / f"{task.id}.json", record)
        self.records.append(
            {
                "task_id": task.id,
                "condition": arm,
                "success": record["result"]["success"],
                "termination": record["result"]["termination"],
                "actions": record["actions"],
                "edits": record["edits"],
                "test_calls": record["test_calls"],
                "invalid_actions": record["invalid_actions"],
                "input_tokens": sum(c["input_tokens"] for c in self.calls),
                "output_tokens": sum(c["output_tokens"] for c in self.calls),
            }
        )
        print(
            task.id, arm, record["result"]["success"], record["result"]["termination"], flush=True
        )
        self.position += 1
        self.episode = None
        self._status(
            "awaiting_final_fingerprint" if self.position == len(self.jobs) else "evaluating"
        )

    def finish(self, data):
        if self.position != len(self.jobs):
            raise ValueError("Not all episodes finished")
        if data.get("final_weight_sha256") != self.model_metadata["initial_weight_sha256"]:
            raise ValueError("Model parameters changed during evaluation")
        if source_fingerprint() != self.config["source_sha256"]:
            raise ValueError("Local source changed during evaluation")
        summary = {
            "protocol": PROTOCOL,
            "split": "development",
            "rows": self.records,
            "source_unchanged": True,
            "model_weights_unchanged": True,
            "arms": {
                arm: {
                    "tasks": sum(r["condition"] == arm for r in self.records),
                    "successes": sum(r["condition"] == arm and r["success"] for r in self.records),
                }
                for arm in ("without_feedback", "with_feedback")
            },
        }
        pairs = {
            task.id: {r["condition"]: r["success"] for r in self.records if r["task_id"] == task.id}
            for task in self.tasks
        }
        summary["paired"] = {
            "both_pass": sum(all(v.values()) for v in pairs.values()),
            "feedback_only": sum(
                v["with_feedback"] and not v["without_feedback"] for v in pairs.values()
            ),
            "no_feedback_only": sum(
                v["without_feedback"] and not v["with_feedback"] for v in pairs.values()
            ),
            "both_fail": sum(not any(v.values()) for v in pairs.values()),
        }
        write_json(self.destination / "completion.json", data)
        write_json(self.destination / "report.json", summary)
        self.finished = True
        self._status("completed")
        return {"completed": True, "episodes": len(self.records)}


def server(experiment, token, port=0):
    if not isinstance(token, str) or len(token) < 32:
        raise ValueError("Use a strong ephemeral bearer token")

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_args):
            pass  # Never log headers or bearer tokens.

        def reply(self, status, body):
            payload = canonical(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(payload)

        def authorized(self):
            return hmac.compare_digest(self.headers.get("Authorization", ""), "Bearer " + token)

        def do_GET(self):
            if not self.authorized():
                return self.reply(401, {"error": "unauthorized"})
            if self.path != "/work":
                return self.reply(404, {"error": "unknown route"})
            with experiment.lock:
                try:
                    return self.reply(200, experiment.work())
                except (ValueError, InfrastructureError) as exc:
                    return self.reply(409, {"error": str(exc)})

        def do_POST(self):
            if not self.authorized():
                return self.reply(401, {"error": "unauthorized"})
            if self.path not in {"/metadata", "/action", "/finish"}:
                return self.reply(404, {"error": "unknown route"})
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 262144:
                    raise ValueError("Invalid request length")
                self.connection.settimeout(20)
                data = json.loads(
                    self.rfile.read(length),
                    parse_constant=reject_constant,
                    object_pairs_hook=unique_keys,
                )
                if not isinstance(data, dict):
                    raise ValueError("Expected JSON object")
                with experiment.lock:
                    if experiment.failed:
                        raise InfrastructureError(experiment.failed)
                    method = {
                        "/metadata": experiment.metadata,
                        "/action": experiment.action,
                        "/finish": experiment.finish,
                    }[self.path]
                    result = method(data)
                return self.reply(200, result)
            except InfrastructureError as exc:
                experiment.failed = str(exc)
                experiment._status("infrastructure_error")
                return self.reply(503, {"error": str(exc)})
            except (ValueError, TypeError, KeyError, TimeoutError) as exc:
                return self.reply(400, {"error": str(exc)})

    return ThreadingHTTPServer(("127.0.0.1", port), Handler)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--port", type=int, default=0)
    args = parser.parse_args()
    experiment = RevisionExperiment(DockerExecutor(), args.output)
    httpd = server(experiment, args.token_file.read_text().strip(), args.port)
    write_json(args.output / "local-endpoint.json", {"port": httpd.server_port})
    print("Coordinator ready on loopback port", httpd.server_port, flush=True)
    try:
        httpd.serve_forever(poll_interval=0.2)
    finally:
        httpd.server_close()


if __name__ == "__main__":
    main()
