"""Open the sealed final test exactly once: every sealed task, all four frozen arms.

The registry entry is written before anything is uploaded, so a second opening is
refused even if this run fails partway. A failed opening needs a human decision, not
a silent retry. Decoding, tools, budgets, and grading match the development runs.
"""

import argparse
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kaggle_client import Kaggle  # noqa: E402
from run_matched_comparison import arm_specs, run_rounds  # noqa: E402

from repairlab.curriculum.comparison import MatchedComparison  # noqa: E402
from repairlab.curriculum.final_test import final_tasks  # noqa: E402
from repairlab.execution import DockerExecutor  # noqa: E402
from repairlab.experiments import write_json  # noqa: E402

KERNEL = "alwaysacing/repairlab-native-final-test"
TITLE = "RepairLab Native Final Test"
ARMS = ("original", "supervised", "recovery", "rl")
REGISTRY = ROOT / "artifacts" / "final-test-registry.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    parser.add_argument("--recovery-run", type=Path, required=True)
    parser.add_argument("--rl-run", type=Path, required=True)
    parser.add_argument("--registry", type=Path, default=REGISTRY)
    parser.add_argument("--open-sealed-final-test", action="store_true", required=True)
    args = parser.parse_args()
    if args.registry.exists():
        raise SystemExit("The sealed final test was already opened: " + str(args.registry))
    arms = arm_specs(ARMS, args.recovery_run, args.rl_run)
    client = Kaggle(args.kaggle, args.token_file)
    experiment = MatchedComparison(
        DockerExecutor(), args.output, arms, final_tasks(), split="final_test"
    )
    root = experiment.destination
    write_json(
        args.registry,
        {
            "status": "opened",
            "run": str(args.output),
            "arms": arms,
            "task_ids": experiment.config["task_ids"],
            "rule": "Opened once after every checkpoint and rule was frozen; never reopen.",
        },
    )
    write_json(
        root / "experiment-plan.json",
        {
            **experiment.config,
            "selection_rule": "Every sealed final-test task in manifest order, opened once.",
            "episodes": len(arms) * len(experiment.tasks),
            "success_rule": "Explicit submit within budget; all public and private checks must pass.",
            "score_meaning": "Autonomous repair score on unseen final-test programs.",
            "no_retries_or_best_of": True,
            "checkpoint_selection_on_final_test": False,
        },
    )
    for name in ("repairlab", "scripts", "tests"):
        shutil.copytree(
            ROOT / name,
            root / "source-snapshot" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    client.require_gpu_quota(root / "quota-before.json")
    report = run_rounds(client, experiment, root, arms, KERNEL, TITLE)
    write_json(
        args.registry,
        {
            "status": "completed",
            "run": str(args.output),
            "arms": arms,
            "task_ids": experiment.config["task_ids"],
            "successes": {name: arm["successes"] for name, arm in report["arms"].items()},
            "rule": "Opened once after every checkpoint and rule was frozen; never reopen.",
        },
    )


if __name__ == "__main__":
    main()
