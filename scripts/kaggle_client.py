"""Private staged Kaggle notebooks: build, submit, confirm, wait, download, extract.

The account token is read from an owner-only local file into the official client's
environment. It is never included in notebooks, requests, output, or run metadata.
"""

import base64
import gzip
import json
import os
import re
import subprocess
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE_LIMIT_BYTES = 1000000  # Kaggle rejects notebooks of 1 MB or more.
# Read-only calls ride out about an hour without network, 30 seconds apart; a laptop
# that changed networks once lost DNS for longer than five minutes mid-run.
OUTAGE_ATTEMPTS = 120
INSTALL = (
    "import os,sys,subprocess\nos.environ['CUDA_VISIBLE_DEVICES']='0'\n"
    "os.environ['TOKENIZERS_PARALLELISM']='false'\nos.environ['HF_HUB_DISABLE_IMPLICIT_TOKEN']='1'\n"
    "subprocess.check_call([sys.executable,'-m','pip','install','transformers==5.2.0','trl==1.8.0','peft==0.18.1'])\n"
    "import torch\nif not torch.cuda.is_available():\n    raise RuntimeError('Select a GPU runtime')\n"
)


def embedded_payload(payload, compress):
    text = json.dumps(payload, ensure_ascii=False)
    if not compress:
        return "import json\nPAYLOAD=json.loads(" + repr(text) + ")\n"
    encoded = base64.b64encode(gzip.compress(text.encode(), mtime=0)).decode()
    return (
        "import base64,gzip,json\nPAYLOAD=json.loads(gzip.decompress(base64.b64decode("
        + repr(encoded)
        + ")))\n"
    )


def build_notebook(
    destination, name, heading, payload, runtime, entry, output_glob, compress=False
):
    """Embed public data and a verbatim runtime; save a traceback archive on failure.

    Compressed payloads hide their text from `Kaggle.run`'s credential scan, so callers
    using `compress` must check the raw payload themselves.
    """
    texts = [
        ("markdown", heading),
        ("code", INSTALL),
        ("code", embedded_payload(payload, compress)),
        ("code", (ROOT / "scripts" / runtime).read_text()),
        (
            "code",
            "try:\n    " + entry + "(PAYLOAD)\nexcept Exception:\n"
            "    import traceback, shutil\n    from pathlib import Path\n"
            "    for folder in Path('.').glob('" + output_glob + "'):\n"
            "        if folder.is_dir():\n"
            "            (folder/'failure.txt').write_text(traceback.format_exc())\n"
            "            shutil.make_archive(str(folder), 'zip', folder)\n    raise\n",
        ),
    ]
    cells = []
    for index, (kind, text) in enumerate(texts):
        cell = {"cell_type": kind, "metadata": {}, "id": f"{name}-{index:02d}", "source": text}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "kaggle": {"isInternetEnabled": True, "isGpuEnabled": True},
        },
        "cells": cells,
    }
    Path(destination).write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n")
    return notebook


class Kaggle:
    def __init__(self, executable, token_file):
        self.executable = str(executable)
        self.token = Path(token_file).read_text().strip()
        self.environment = {
            **os.environ,
            "KAGGLE_API_TOKEN": self.token,
            "KAGGLE_CONFIG_DIR": str(Path(token_file).parent),
        }

    def call(self, arguments, logfile):
        result = subprocess.run(
            [self.executable, *arguments],
            env=self.environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        text = (result.stdout + result.stderr).replace(self.token, "[REDACTED]")
        logfile.parent.mkdir(parents=True, exist_ok=True)
        logfile.write_text(text)
        if result.returncode:
            raise RuntimeError("Kaggle command failed; inspect " + str(logfile))
        return text

    def require_gpu_quota(self, logfile):
        quota = json.loads(self.call(["quota", "--format", "json"], logfile))
        if float(next(r["remaining"] for r in quota if r["resource"] == "GPU")[:-1]) <= 0:
            raise RuntimeError("No free GPU quota available")

    def _repeatable(self, arguments, logfile, attempts=3, pause_seconds=30):
        """Run a read-only command again after a failure, keeping each failed log.

        The CLI once exited nonzero after a complete download, and the host once lost
        DNS while polling a running notebook. Reading again is harmless; a push is never
        retried, because a repeated push would start a second notebook.
        """
        for attempt in range(1, attempts + 1):
            try:
                return self.call(arguments, logfile)
            except (RuntimeError, subprocess.TimeoutExpired):
                if attempt == attempts:
                    raise
                if logfile.exists():
                    index = 1
                    while (failed := logfile.with_stem(f"{logfile.stem}-failed-{index}")).exists():
                        index += 1
                    logfile.rename(failed)
                time.sleep(pause_seconds)

    def _download(self, reference, folder, attempts=None, pause_seconds=30):
        self._repeatable(
            [
                "kernels",
                "output",
                reference,
                "-p",
                str(folder / "download"),
                "--file-pattern",
                r".*(\.log|\.zip)$",
                "--page-size",
                "200",
            ],
            folder / "download.log",
            OUTAGE_ATTEMPTS if attempts is None else attempts,
            pause_seconds,
        )

    def run(self, folder, notebook, metadata, poll_seconds=20):
        """Submit a saved upload folder; returns extracted output, archive, and version."""
        upload = folder / "upload"
        if self.token in json.dumps(notebook):
            raise RuntimeError("Credential would enter the notebook")
        if (upload / metadata["code_file"]).stat().st_size >= SOURCE_LIMIT_BYTES:
            raise RuntimeError("Notebook exceeds Kaggle's 1 MB source limit")
        (upload / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
        response = self.call(
            ["kernels", "push", "-p", str(upload), "--accelerator", "NvidiaTeslaT4"],
            folder / "submission-response.txt",
        )
        version = re.search(r"Kernel version (\d+) successfully pushed", response)
        if not version:
            raise RuntimeError("Unknown submission outcome; inspect server state before retrying")
        reference = metadata["id"] + "/" + version.group(1)
        self._repeatable(
            ["kernels", "pull", reference, "-p", str(folder / "server"), "-m"],
            folder / "pull.log",
            attempts=OUTAGE_ATTEMPTS,
        )
        server = json.loads((folder / "server" / "kernel-metadata.json").read_text())
        if server.get("is_private") is not True:
            raise RuntimeError("Server did not confirm private notebook")
        remote = json.loads(next((folder / "server").glob("*.ipynb")).read_text())
        if [c["source"] for c in remote["cells"]] != [c["source"] for c in notebook["cells"]]:
            raise RuntimeError("Submitted notebook source mismatch")
        last_status = None
        while True:
            status = self._repeatable(
                ["kernels", "status", reference],
                folder / "worker-status.txt",
                attempts=OUTAGE_ATTEMPTS,
            )
            if status != last_status:
                print(status.strip(), flush=True)
                last_status = status
            if '"KernelWorkerStatus.COMPLETE"' in status:
                break
            if '"KernelWorkerStatus.ERROR"' in status or '"KernelWorkerStatus.CANCEL' in status:
                self._download(reference, folder)
                raise RuntimeError("GPU worker failed; diagnose saved outputs before retrying")
            time.sleep(poll_seconds)
        self._download(reference, folder)
        archives = list((folder / "download").glob("*.zip"))
        if len(archives) != 1:
            raise RuntimeError("Expected exactly one output archive")
        extracted = folder / "extracted"
        with zipfile.ZipFile(archives[0]) as archive:
            for member in archive.infolist():
                if (
                    Path(member.filename).is_absolute()
                    or ".." in Path(member.filename).parts
                    or member.file_size > 64000000
                ):
                    raise RuntimeError("Unsafe or unexpectedly large output archive")
            archive.extractall(extracted)
        return extracted, archives[0], int(version.group(1))
