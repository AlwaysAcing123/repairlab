"""Compare saved checkpoints on all 24 development tasks under the frozen rules.

Tasks are not selectable: sealed final-test tasks have no path through this script.
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

from repairlab.curriculum.comparison import MatchedComparison  # noqa: E402
from repairlab.curriculum.recovery import WARM_START  # noqa: E402
from repairlab.execution import DockerExecutor  # noqa: E402
from repairlab.experiments import write_json  # noqa: E402
from repairlab.tasks import pilot_tasks  # noqa: E402

KERNEL = "alwaysacing/repairlab-native-matched-comparison"
TITLE = "RepairLab Native Matched Comparison"


def arm_specs(names, recovery_run, rl_run=None):
    specs = {"original": None, "supervised": WARM_START}
    if recovery_run is not None:
        specs["recovery"] = json.loads((recovery_run / "recovery-adapter.json").read_text())
    if rl_run is not None:
        specs["rl"] = json.loads((rl_run / "rl-adapter.json").read_text())
    return {name: specs[name] for name in names}


def run_rounds(client, experiment, root, arms, kernel, title):
    """Submit every round, apply it locally, and finish; shared with other fixed task sets."""
    sources = sorted({a["kernel_source"] for a in arms.values() if a})
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
            notebook = build_notebook(
                upload / "RepairLab_Matched_Round.ipynb",
                "matched",
                "# RepairLab matched checkpoint comparison\n\n"
                "One next action per unfinished episode, with identical native tools and budgets. "
                "Model output is never executed on this GPU. Only public task conversations and "
                "checkpoint hashes are embedded. No training occurs.",
                batch,
                "matched_eval_runtime.py",
                "run_matched_round",
                "repairlab-matched-round-*",
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
            extracted, archive, _ = client.run(
                folder,
                notebook,
                {
                    "id": kernel,
                    "title": title,
                    "code_file": "RepairLab_Matched_Round.ipynb",
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
                },
            )
        report = experiment.finish()
        client.call(["quota", "--format", "json"], root / "quota-after.json")
        print("MATCHED COMPARISON COMPLETE", json.dumps(report["arms"]), flush=True)
        return report
    except BaseException as exc:
        write_json(
            root / "status.json",
            {"status": "needs_diagnosis", "error": str(exc), "round": experiment.round_number},
        )
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    parser.add_argument("--kernel", default=KERNEL)
    parser.add_argument("--arms", default="supervised,recovery")
    parser.add_argument("--recovery-run", type=Path)
    parser.add_argument("--rl-run", type=Path)
    parser.add_argument("--seeded", action="store_true")
    args = parser.parse_args()
    arms = arm_specs(args.arms.split(","), args.recovery_run, args.rl_run)
    client = Kaggle(args.kaggle, args.token_file)
    experiment = MatchedComparison(DockerExecutor(), args.output, arms, pilot_tasks(), args.seeded)
    root = experiment.destination
    write_json(
        root / "experiment-plan.json",
        {
            **experiment.config,
            "selection_rule": "All 24 development tasks in manifest order.",
            "episodes": len(arms) * len(experiment.tasks),
            "success_rule": "Explicit submit within budget; all public and private checks must pass.",
            "score_meaning": "Seeded recovery probe, not an autonomous repair score."
            if args.seeded
            else "Autonomous development repair score.",
            "no_retries_or_best_of": True,
        },
    )
    for name in ("repairlab", "scripts", "tests"):
        shutil.copytree(
            ROOT / name,
            root / "source-snapshot" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    client.require_gpu_quota(root / "quota-before.json")
    run_rounds(client, experiment, root, arms, args.kernel, TITLE)


if __name__ == "__main__":
    main()
