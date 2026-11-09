"""Compare saved checkpoints of either model size on all 48 development tasks.

Arms are named, not configured: `0.6b-rl`, `1.7b-base`, `1.7b-plain`, `1.7b-notes`, and
`1.7b-rl`, read from their saved run records. ``--tasks`` exists for scripted dry runs;
real runs use every development task, and the plan records which happened.
"""

import argparse
import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kaggle_client import Kaggle, build_notebook  # noqa: E402

from repairlab.execution import DockerExecutor  # noqa: E402
from repairlab.experiments import write_json  # noqa: E402
from repairlab.scaleup.comparison import SizedComparison  # noqa: E402
from repairlab.scaleup.models import RL_06B, arm, development_tasks  # noqa: E402
from repairlab.tasks import sha256  # noqa: E402

KERNEL = "alwaysacing/repairlab-sized-comparison"
TITLE = "RepairLab Sized Comparison"
ARMS = ("0.6b-rl", "1.7b-base", "1.7b-plain", "1.7b-notes", "1.7b-rl")


class SuspensionSafeExecutor:
    """Grade again after a timeout, logging each retry into the run directory.

    A laptop that slept mid-grade once turned a correct RL candidate's normal grade into
    a timeout. A real timeout, such as an infinite loop, recurs on every attempt, so only
    a grade the host interrupted can change.
    """

    def __init__(self, executor, log, attempts=3):
        self.executor, self.log, self.attempts = executor, Path(log), attempts

    def metadata(self):
        return {**self.executor.metadata(), "timeout_attempts": self.attempts}

    def run(self, source, cases, *, reveal=False):
        for attempt in range(1, self.attempts + 1):
            result = self.executor.run(source, cases, reveal=reveal)
            if result.get("status") != "timeout" or attempt == self.attempts:
                return result
            with self.log.open("a") as handle:
                record = {"attempt": attempt, "source_sha256": sha256(source), "result": result}
                handle.write(json.dumps(record) + "\n")


def arm_specs(names, plain_run=None, notes_run=None, rl_run=None):
    specs = {"0.6b-rl": lambda: arm("0.6b", RL_06B), "1.7b-base": lambda: arm("1.7b")}
    for name, run, record in (
        ("1.7b-plain", plain_run, "scaleup-adapter.json"),
        ("1.7b-notes", notes_run, "scaleup-adapter.json"),
        ("1.7b-rl", rl_run, "rl-adapter.json"),
    ):
        if run is not None:
            specs[name] = lambda run=run, record=record: arm(
                "1.7b", json.loads((Path(run) / record).read_text())
            )
    missing = [n for n in names if n not in specs]
    if missing:
        raise SystemExit("No saved run given for arms: " + ", ".join(missing))
    return {name: specs[name]() for name in names}


def run_sized_rounds(client, experiment, root, kernel, title):
    """Submit every round, apply it locally, and finish; shared by every fixed task set."""
    sources = sorted(
        {a["adapter"]["kernel_source"] for a in experiment.arms.values() if a["adapter"]}
    )
    try:
        while True:
            batch = experiment.export()
            if batch is None:
                break
            if batch["round"] > 8:
                raise RuntimeError("Unexpected ninth round")
            folder = root / "rounds" / str(batch["round"])
            upload = folder / "upload"
            upload.mkdir(exist_ok=False)
            if client.token in json.dumps(batch):
                raise RuntimeError("Credential would enter the notebook")
            notebook = build_notebook(
                upload / "RepairLab_Sized_Round.ipynb",
                "sized",
                "# RepairLab matched comparison across model sizes\n\n"
                "One next action per unfinished episode, with identical native tools and budgets. "
                "Model output is never executed on this GPU. Only public task conversations and "
                "checkpoint hashes are embedded, compressed to fit Kaggle's 1 MB notebook limit. "
                "No training occurs.",
                batch,
                "sized_eval_runtime.py",
                "run_sized_round",
                "repairlab-sized-round-*",
                compress=True,
            )
            for cell in notebook["cells"]:
                if cell["cell_type"] == "code":
                    compile(ast.parse(cell["source"]), "inference-client", "exec")
            print(
                "Submitting round",
                batch["round"],
                "with",
                len(batch["requests"]),
                "unfinished episodes",
                flush=True,
            )
            extracted, archive, version = client.run(
                folder,
                notebook,
                {
                    "id": kernel,
                    "title": title,
                    "code_file": "RepairLab_Sized_Round.ipynb",
                    "language": "python",
                    "kernel_type": "notebook",
                    "is_private": True,
                    "enable_gpu": True,
                    "enable_tpu": False,
                    "enable_internet": True,
                    "dataset_sources": [],
                    "competition_sources": [],
                    "kernel_sources": sources,
                    "model_sources": [],
                },
            )
            metadata = json.loads((extracted / "model-config.json").read_text())
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
                    "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                    "worker_complete": True,
                    "credentials_uploaded": False,
                    "kernel_version": version,
                },
            )
        report = experiment.finish()
        client.call(["quota", "--format", "json"], root / "quota-after.json")
        print("SIZED COMPARISON COMPLETE", json.dumps(report["arms"]), flush=True)
        return report
    except BaseException as exc:
        write_json(
            root / "status.json",
            {"status": "needs_diagnosis", "error": str(exc), "round": experiment.round_number},
        )
        raise


def snapshot(root):
    for name in ("repairlab", "scripts", "tests"):
        shutil.copytree(
            ROOT / name,
            root / "source-snapshot" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    parser.add_argument("--arms", required=True)
    parser.add_argument("--plain-run", type=Path)
    parser.add_argument("--notes-run", type=Path)
    parser.add_argument("--rl-run", type=Path)
    parser.add_argument("--tasks", type=int)
    args = parser.parse_args()
    names = args.arms.split(",")
    if set(names) - set(ARMS) or len(set(names)) != len(names):
        raise SystemExit("Arms must be distinct names from: " + ", ".join(ARMS))
    arms = arm_specs(names, args.plain_run, args.notes_run, args.rl_run)
    client = Kaggle(args.kaggle, args.token_file)
    tasks = development_tasks()[: args.tasks]
    executor = SuspensionSafeExecutor(DockerExecutor(), args.output / "timeout-retries.jsonl")
    experiment = SizedComparison(executor, args.output, arms, tasks)
    root = experiment.destination
    write_json(
        root / "experiment-plan.json",
        {
            **experiment.config,
            "selection_rule": "All 48 development tasks in order: the 24 pilot tasks, then the "
            "24 tasks of the first final test, opened November 5.",
            "uses_fixed_configuration": experiment.config["full_task_list"],
            "episodes": len(arms) * len(experiment.tasks),
            "success_rule": "Explicit submit within budget; all public and private checks must pass.",
            "score_meaning": "Autonomous development repair score.",
            "no_retries_or_best_of": True,
        },
    )
    snapshot(root)
    client.require_gpu_quota(root / "quota-before.json")
    run_sized_rounds(client, experiment, root, KERNEL, TITLE)


if __name__ == "__main__":
    main()
