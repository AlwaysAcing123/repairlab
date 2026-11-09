"""Run a frozen original-versus-supervised native repair comparison.

`--task-set pilot` is the original six-task pilot; `--task-set remaining` is the
expansion to the other 18 development tasks under the same rules and checkpoint.
The account token is read from an owner-only local file into the official client's
environment. It is never included in notebooks, requests, output, or run metadata.
"""

import argparse
import ast
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from build_native_repair_notebook import build  # noqa: E402

from repairlab.curriculum.splits import remaining_development_tasks  # noqa: E402
from repairlab.execution import DockerExecutor  # noqa: E402
from repairlab.experiments import write_json  # noqa: E402
from repairlab.native_repair import NativeRepairPilot, selected_tasks  # noqa: E402

TASK_SETS = {
    "pilot": {
        "tasks": selected_tasks,
        "kernel": "alwaysacing/repairlab-native-repair-pilot",
        "title": "RepairLab Native Repair Pilot",
        "selection_rule": "First two tasks per category in the authored manifest; development pilot, not a held-out final test.",
    },
    "remaining": {
        "tasks": remaining_development_tasks,
        "kernel": "alwaysacing/repairlab-native-repair-expansion",
        "title": "RepairLab Native Repair Expansion",
        "selection_rule": "Every authored development task outside the frozen six-task pilot, in manifest order; fixed by the pilot report before inference. Development data, not a held-out final test.",
    },
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    parser.add_argument("--task-set", choices=sorted(TASK_SETS), default="pilot")
    parser.add_argument("--kernel")
    args = parser.parse_args()
    task_set = TASK_SETS[args.task_set]
    args.kernel = args.kernel or task_set["kernel"]
    tasks = task_set["tasks"]()
    token = args.token_file.read_text().strip()
    environment = os.environ.copy()
    environment["KAGGLE_API_TOKEN"] = token
    environment["KAGGLE_CONFIG_DIR"] = str(args.token_file.parent)
    adapter_path = (
        ROOT
        / "artifacts/native-supervised-warmup-20261030-v1/extracted/adapter/adapter_model.safetensors"
    )
    adapter = {
        "kernel_source": "alwaysacing/repairlab-native-supervised-warmup",
        "upstream_version": 1,
        "file_sha256": "2dd4773919ca4c13934d796fd0e38ed77aeb8df12055e2cd8f3b69992ac47375",
        "tensor_sha256": "245ac4651a2fb8018a8c699bc6219341ddf85a674d8584372bd5aa4905213c82",
        "base_sha256": "2d4bad117c2ae98f4a69740ac05d48c237237d84711349a396c5e98866553204",
    }
    if hashlib.sha256(adapter_path.read_bytes()).hexdigest() != adapter["file_sha256"]:
        raise RuntimeError("Local saved checkpoint differs from the frozen adapter")
    experiment = NativeRepairPilot(DockerExecutor(), args.output, adapter, tasks)
    root = experiment.destination
    write_json(
        root / "experiment-plan.json",
        {
            **experiment.config,
            "selection_rule": task_set["selection_rule"],
            "tasks": [t.id for t in tasks],
            "episodes": 2 * len(tasks),
            "success_rule": "Explicit submit within budget; all public and private checks must pass.",
            "no_retries_or_best_of": True,
            "no_new_training": True,
        },
    )
    for name in ("repairlab", "scripts", "tests"):
        shutil.copytree(
            ROOT / name,
            root / "source-snapshot" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    shutil.copyfile(ROOT / "pyproject.toml", root / "source-snapshot/pyproject.toml")

    def kaggle(arguments, logfile):
        result = subprocess.run(
            [str(args.kaggle), *arguments],
            env=environment,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        text = (result.stdout + result.stderr).replace(token, "[REDACTED]")
        logfile.parent.mkdir(parents=True, exist_ok=True)
        logfile.write_text(text)
        if result.returncode:
            raise RuntimeError("Kaggle command failed; inspect " + str(logfile))
        return text

    try:
        quota = kaggle(["quota", "--format", "json"], root / "quota-before.json")
        if (
            float(next(r["remaining"] for r in json.loads(quota) if r["resource"] == "GPU")[:-1])
            <= 0
        ):
            raise RuntimeError("No free GPU quota available")
        while True:
            batch = experiment.export()
            if batch is None:
                break
            if batch["round"] > 8:
                raise RuntimeError("Unexpected ninth round")
            folder = root / "rounds" / str(batch["round"])
            upload = folder / "upload"
            upload.mkdir(exist_ok=False)
            notebook = build(upload / "RepairLab_Native_Repair_Round.ipynb", batch)
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    compile(ast.parse(cell["source"]), "inference-client", "exec")
            if token in json.dumps(notebook):
                raise RuntimeError("Credential would enter the notebook")
            write_json(
                upload / "kernel-metadata.json",
                {
                    "id": args.kernel,
                    "title": task_set["title"],
                    "code_file": "RepairLab_Native_Repair_Round.ipynb",
                    "language": "python",
                    "kernel_type": "notebook",
                    "is_private": True,
                    "enable_gpu": True,
                    "enable_tpu": False,
                    "enable_internet": True,
                    "dataset_sources": [],
                    "competition_sources": [],
                    "kernel_sources": [adapter["kernel_source"]],
                    "model_sources": [],
                },
            )
            print(
                "Submitting round",
                batch["round"],
                "with",
                len(batch["requests"]),
                "unfinished episodes",
                flush=True,
            )
            response = kaggle(
                ["kernels", "push", "-p", str(upload), "--accelerator", "NvidiaTeslaT4"],
                folder / "submission-response.txt",
            )
            version = re.search(r"Kernel version (\d+) successfully pushed", response)
            if not version:
                raise RuntimeError(
                    "Unknown submission outcome; inspect server state before retrying"
                )
            reference = args.kernel + "/" + version.group(1)
            kaggle(
                ["kernels", "pull", reference, "-p", str(folder / "server"), "-m"],
                folder / "pull.log",
            )
            server_meta = json.loads((folder / "server" / "kernel-metadata.json").read_text())
            if server_meta.get("is_private") is not True:
                raise RuntimeError("Server did not confirm private notebook")
            server_source = next((folder / "server").glob("*.ipynb"))
            remote = json.loads(server_source.read_text())
            if [c["source"] for c in remote["cells"]] != [c["source"] for c in notebook["cells"]]:
                raise RuntimeError("Submitted notebook source mismatch")
            last_status = None
            while True:
                status = kaggle(["kernels", "status", reference], folder / "worker-status.txt")
                if status != last_status:
                    print("Round", batch["round"], status.strip(), flush=True)
                    last_status = status
                if '"KernelWorkerStatus.COMPLETE"' in status:
                    break
                if any(
                    s in status
                    for s in ('"KernelWorkerStatus.ERROR"', '"KernelWorkerStatus.CANCEL')
                ):
                    kaggle(
                        [
                            "kernels",
                            "output",
                            reference,
                            "-p",
                            str(folder / "download"),
                            "--file-pattern",
                            r".*(\.log|\.zip)$",
                        ],
                        folder / "download.log",
                    )
                    raise RuntimeError("GPU worker failed; diagnose saved outputs before retrying")
                time.sleep(20)
            kaggle(
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
            )
            archives = list((folder / "download").glob("*.zip"))
            if len(archives) != 1:
                raise RuntimeError("Expected exactly one round output archive")
            extracted = folder / "extracted"
            with zipfile.ZipFile(archives[0]) as archive:
                for member in archive.infolist():
                    if (
                        Path(member.filename).is_absolute()
                        or ".." in Path(member.filename).parts
                        or member.file_size > 10000000
                    ):
                        raise RuntimeError("Unsafe or unexpectedly large output archive")
                archive.extractall(extracted)
            metadata = json.loads((extracted / "model-config.json").read_text())
            if (
                metadata["batch_sha256"] != batch["batch_sha256"]
                or metadata["round"] != batch["round"]
            ):
                raise RuntimeError("Downloaded outputs target a different batch")
            outputs = [
                json.loads(line)
                for line in (extracted / "responses.jsonl").read_text().splitlines()
            ]
            print("Grading/applying round", batch["round"], "locally", flush=True)
            experiment.apply(batch["batch_sha256"], metadata, outputs)
            write_json(
                folder / "round-verification.json",
                {
                    "server_private": True,
                    "server_sources_match": True,
                    "response_count": len(outputs),
                    "batch_sha256": batch["batch_sha256"],
                    "archive_sha256": hashlib.sha256(archives[0].read_bytes()).hexdigest(),
                    "worker_complete": True,
                    "credentials_uploaded": False,
                },
            )
        report = experiment.finish()
        kaggle(["quota", "--format", "json"], root / "quota-after.json")
        print("NATIVE REPAIR RUN COMPLETE", json.dumps(report["arms"]), flush=True)
    except BaseException as exc:
        write_json(
            root / "status.json",
            {"status": "needs_diagnosis", "error": str(exc), "round": experiment.round_number},
        )
        raise


if __name__ == "__main__":
    main()
