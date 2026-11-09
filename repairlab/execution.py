"""Container execution and host-owned, type-sensitive result comparison."""

import ast
import json
import os
import selectors
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from .tasks import canonical, sha256

IMAGE = "python:3.12-slim-bookworm@sha256:6a0e06f5ba0b373d051aabd3fffb9b873f27f9a2a3a14bcbe37dee6f0e3bbc70"
MAX_SOURCE_BYTES = 8192


class InfrastructureError(RuntimeError):
    """An execution environment failed, so no model result can be assigned."""


def validate_source(source, signature=None):
    if not isinstance(source, str) or len(source.encode()) > MAX_SOURCE_BYTES:
        raise ValueError("Source must be text with at most 8192 UTF-8 bytes")
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError) as exc:
        raise ValueError("Invalid Python source") from exc
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        raise ValueError("Submit exactly one function definition: def solve(...)")
    function = tree.body[0]
    if function.name != "solve" or function.decorator_list:
        raise ValueError("The function must be named solve and have no decorators")
    if function.args.defaults or any(x is not None for x in function.args.kw_defaults):
        raise ValueError("Default argument expressions are not supported")
    if function.returns or any(
        arg.annotation
        for arg in (function.args.posonlyargs + function.args.args + function.args.kwonlyargs)
    ):
        raise ValueError("Annotations are not supported in the pilot")
    if signature is not None:
        expected = ast.parse(f"def solve({signature}):\n    pass\n").body[0].args
        if ast.dump(function.args) != ast.dump(expected):
            raise ValueError("Preserve the supplied function signature")


def bounded_run(command, payload, timeout, max_output=1048576):
    """Bound host-side captured output as well as wall time (communicate does not)."""
    with tempfile.TemporaryFile() as stdin:
        stdin.write(payload.encode())
        stdin.seek(0)
        process = subprocess.Popen(
            command, stdin=stdin, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        chunks = {"stdout": bytearray(), "stderr": bytearray()}
        deadline = time.monotonic() + timeout
        status = None
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, "stdout")
                selector.register(process.stderr, selectors.EVENT_READ, "stderr")
                while selector.get_map():
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        status = "timeout"
                        break
                    for key, _ in selector.select(min(remaining, 0.2)):
                        block = os.read(key.fileobj.fileno(), 65536)
                        if not block:
                            selector.unregister(key.fileobj)
                            continue
                        chunks[key.data].extend(block)
                        if sum(len(c) for c in chunks.values()) > max_output:
                            status = "output_limit"
                            break
                    if status:
                        break
            if status:
                process.kill()
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            status = "timeout"
            process.kill()
            process.wait(timeout=5)
        finally:
            process.stdout.close()
            process.stderr.close()
            if process.poll() is None:
                process.kill()
                process.wait(timeout=5)
        return (
            status,
            process.returncode,
            {k: bytes(v[:max_output]).decode(errors="replace") for k, v in chunks.items()},
        )


class DockerExecutor:
    def __init__(self, timeout=12, image=IMAGE):
        self.timeout = timeout
        self.image = image
        self.resolved_reference = image
        try:
            result = subprocess.run(
                ["docker", "image", "inspect", image], capture_output=True, text=True, timeout=15
            )
            if result.returncode and image == IMAGE:
                # The original baseline recorded this exact local content ID.
                # Docker can retain it while losing its repository digest alias.
                # Resolve the immutable ID, never a mutable tag or a newer image.
                self.resolved_reference = IMAGE.split("@", 1)[1]
                result = subprocess.run(
                    ["docker", "image", "inspect", self.resolved_reference],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise InfrastructureError("Docker is unavailable") from exc
        if result.returncode:
            raise InfrastructureError("Start Docker and pull the pinned Python image; see README")
        self.image_id = json.loads(result.stdout)[0]["Id"]
        if self.resolved_reference != image and self.image_id != self.resolved_reference:
            raise InfrastructureError("Local image content ID differs from the original baseline")

    def metadata(self):
        return {
            "backend": "docker",
            "image": self.image,
            "image_id": self.image_id,
            "resolved_reference": self.resolved_reference,
            "worker_sha256": sha256(Path(__file__).with_name("worker.py").read_text()),
            "timeout_seconds": self.timeout,
            "cpu_seconds": 3,
            "memory_mb": 128,
            "pids": 32,
            "network": "none",
            "root_filesystem": "read_only",
        }

    def _request(self, request):
        name = "repairlab-" + uuid.uuid4().hex
        worker = Path(__file__).with_name("worker.py").resolve()
        command = [
            "docker",
            "run",
            "--rm",
            "--pull=never",
            "--name",
            name,
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--pids-limit=32",
            "--memory=128m",
            "--memory-swap=128m",
            "--cpus=1",
            "--user=65534:65534",
            "--log-driver=none",
            "--tmpfs=/tmp:rw,noexec,nosuid,size=8m",
            "--mount",
            f"type=bind,src={worker},dst=/runner.py,readonly",
            "--interactive",
            "--entrypoint=python",
            self.image_id,
            "-I",
            "-B",
            "/runner.py",
        ]
        payload = canonical(request)
        started = time.monotonic()
        try:
            status, code, output = bounded_run(command, payload, self.timeout)
        except OSError as exc:
            raise InfrastructureError("Could not launch Docker") from exc
        finally:
            # Killing the CLI does not guarantee killing its container.
            subprocess.run(["docker", "rm", "--force", name], capture_output=True, timeout=15)
        elapsed = time.monotonic() - started
        result = {
            "passed": False,
            "status": "execution_error",
            "elapsed_seconds": round(elapsed, 4),
        }
        if status:
            return {**result, "status": status}
        if code in (125, 126, 127):
            raise InfrastructureError("Container could not start: " + output["stderr"][:500])
        if code:
            return {
                **result,
                "status": "resource_limit" if code in (137, 152) else "execution_error",
            }
        try:
            data = json.loads(
                output["stdout"], parse_constant=reject_constant, object_pairs_hook=unique_keys
            )
        except (ValueError, RecursionError):
            return {**result, "status": "invalid_output"}
        if not isinstance(data, dict):
            return {**result, "status": "invalid_output"}
        return {**result, "status": "returned", "data": data}

    def run(self, source, cases, *, reveal=False):
        validate_source(source)
        # Custom-task expected outputs stay on the host.
        result = self._request({"source": source, "inputs": [c.args for c in cases]})
        data = result.pop("data", None)
        result.update(passed_cases=0, total_cases=len(cases))
        if data is None:
            return result
        if "load_error" in data:
            return {**result, "status": "load_error"}
        rows = data.get("results")
        if not isinstance(rows, list) or len(rows) != len(cases):
            return {**result, "status": "invalid_output"}
        checks = []
        for case, actual in zip(cases, rows):
            ok = (
                isinstance(actual, dict)
                and "value" in actual
                and "error" not in actual
                and actual.get("mutated_input") is False
                and canonical(actual["value"]) == canonical(case.expected)
            )
            # Candidate output cannot override the host's verdict with a 'passed' key.
            checks.append(
                {
                    **(
                        {
                            k: v
                            for k, v in actual.items()
                            if k in {"value", "error", "mutated_input"}
                        }
                        if isinstance(actual, dict)
                        else {}
                    ),
                    "passed": ok,
                }
            )
        passed = sum(row["passed"] for row in checks)
        result.update(passed=passed == len(cases), passed_cases=passed, status="graded")
        if reveal:
            result["cases"] = [
                {**row, "args": case.args, "expected": case.expected}
                for case, row in zip(cases, checks)
            ]
        return result

    def run_suite(self, source, tests):
        """Official-style external assert runner; not the custom output oracle.

        External tests execute in the candidate container. This is suitable for
        non-adversarial patch evaluation, not reward-hacking-resistant RL grading.
        """
        if not isinstance(source, str) or len(source.encode()) > 65536:
            raise ValueError("External patch must be text under 64 KiB")
        try:
            ast.parse(source)
        except (SyntaxError, ValueError, RecursionError) as exc:
            raise ValueError("Invalid Python patch") from exc
        result = self._request({"mode": "external_assert_suite", "source": source, "tests": tests})
        data = result.pop("data", None)
        if data is None:
            return result
        if set(data) != {"suite_passed", "error"} or type(data["suite_passed"]) is not bool:
            return {**result, "status": "invalid_output"}
        return {
            **result,
            "passed": data["suite_passed"],
            "status": "graded",
            "error": data["error"],
        }


def reject_constant(value):
    raise ValueError("Nonfinite JSON number")


def unique_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON key")
        result[key] = value
    return result
