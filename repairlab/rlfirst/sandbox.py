"""In-session grading without Docker, for the Kaggle RL loop and the base sweep.

The candidate runs in its own process, using the same ``worker.py`` as the Docker grader
and the same host-side comparison (``DockerExecutor.run`` is inherited unchanged). The
worker receives inputs only; expected answers stay in this parent process. Limits are
CPU time and file size (set by the worker), address space, process count, and wall
time. When the parent runs as root, the child drops to an unprivileged user and, where
``unshare`` works, loses network access. The laptop regrades everything in Docker.
"""

import json
import os
import resource
import selectors
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from ..execution import DockerExecutor, reject_constant, unique_keys
from ..tasks import canonical, sha256

WORKER = Path(__file__).parent.parent / "worker.py"
NOBODY = 65534


class LocalExecutor(DockerExecutor):
    def __init__(self, timeout=12, memory_bytes=1 << 30, isolate_network=None):
        self.timeout = timeout
        self.memory_bytes = memory_bytes
        self.root = os.geteuid() == 0 if hasattr(os, "geteuid") else False
        self.workdir = Path(tempfile.mkdtemp(prefix="repairlab-sandbox-"))
        self.worker = self.workdir / "runner.py"
        shutil.copyfile(WORKER, self.worker)
        os.chmod(self.workdir, 0o755)
        os.chmod(self.worker, 0o444)
        if isolate_network is None:
            isolate_network = self.root and _unshare_works()
        self.isolate_network = isolate_network
        # macOS ignores or rejects RLIMIT_AS; the laptop only prefilters, Docker decides.
        self.enforce_memory = sys.platform.startswith("linux")

    def metadata(self):
        return {
            "backend": "local_process",
            "python": sys.version.split()[0],
            "worker_sha256": sha256(WORKER.read_text()),
            "timeout_seconds": self.timeout,
            "cpu_seconds": 3,
            "memory_bytes": self.memory_bytes if self.enforce_memory else None,
            "dropped_to_user": NOBODY if self.root else None,
            "network_isolated": self.isolate_network,
        }

    def _preexec(self):
        os.setsid()
        if self.enforce_memory:
            resource.setrlimit(resource.RLIMIT_AS, (self.memory_bytes, self.memory_bytes))
        if hasattr(resource, "RLIMIT_NPROC") and self.root:
            resource.setrlimit(resource.RLIMIT_NPROC, (32, 32))
        if self.root:
            os.setgroups([])
            os.setgid(NOBODY)
            os.setuid(NOBODY)

    def _request(self, request):
        command = [sys.executable, "-I", "-B", str(self.worker)]
        if self.isolate_network:
            command = ["unshare", "--net", "--"] + command
        payload = canonical(request)
        started = time.monotonic()
        status, code, output = _bounded(
            command,
            payload,
            self.timeout,
            preexec=self._preexec,
            cwd=str(self.workdir),
        )
        elapsed = time.monotonic() - started
        result = {
            "passed": False,
            "status": "execution_error",
            "elapsed_seconds": round(elapsed, 4),
        }
        if status:
            return {**result, "status": status}
        if code:
            # Killed by a signal (negative code) or the worker's CPU or file limit.
            limited = code < 0 or code in (137, 152)
            return {**result, "status": "resource_limit" if limited else "execution_error"}
        try:
            data = json.loads(
                output["stdout"], parse_constant=reject_constant, object_pairs_hook=unique_keys
            )
        except (ValueError, RecursionError):
            return {**result, "status": "invalid_output"}
        if not isinstance(data, dict):
            return {**result, "status": "invalid_output"}
        return {**result, "status": "returned", "data": data}


def _unshare_works():
    try:
        result = subprocess.run(["unshare", "--net", "--", "true"], capture_output=True, timeout=10)
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def _bounded(command, payload, timeout, preexec, cwd, max_output=1048576):
    """Like ``execution.bounded_run``, plus a process group, limits, and a clean environment."""
    with tempfile.TemporaryFile() as stdin:
        stdin.write(payload.encode())
        stdin.seek(0)
        process = subprocess.Popen(
            command,
            stdin=stdin,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=preexec,  # noqa: PLW1509 - limits must apply before exec
            cwd=cwd,
            env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8"},
            close_fds=True,
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
                _kill(process)
            process.wait(timeout=max(0.1, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            status = "timeout"
            _kill(process)
            process.wait(timeout=5)
        finally:
            process.stdout.close()
            process.stderr.close()
            if process.poll() is None:
                _kill(process)
                process.wait(timeout=5)
        return (
            status,
            process.returncode,
            {k: bytes(v[:max_output]).decode(errors="replace") for k, v in chunks.items()},
        )


def _kill(process):
    try:
        os.killpg(process.pid, 9)
    except (ProcessLookupError, PermissionError):
        process.kill()
