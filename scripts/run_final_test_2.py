"""Open the second sealed final test exactly once: all 48 tasks, all five frozen arms.

Needs the frozen checkpoint record. The registry entry is written before anything is
uploaded, so a second opening is refused even if this run fails partway; a failed
opening needs a human decision, not a silent retry. Decoding, tools, budgets, and
grading match the development runs. HumanEvalFix native opens the same way.
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kaggle_client import Kaggle  # noqa: E402
from run_scaleup_comparison import (  # noqa: E402
    SuspensionSafeExecutor,
    run_sized_rounds,
    snapshot,
)

from repairlab.execution import DockerExecutor  # noqa: E402
from repairlab.experiments import write_json  # noqa: E402
from repairlab.final2.programs import final2_tasks  # noqa: E402
from repairlab.scaleup.comparison import SizedComparison  # noqa: E402

FROZEN = ROOT / "artifacts" / "scaleup-frozen.json"
REGISTRY = ROOT / "artifacts" / "final-test-2-registry.json"


def open_once(args, registry, frozen, names, tasks, split, kernel, title, meaning):
    if registry.exists():
        raise SystemExit("Already opened: " + str(registry))
    arms = {name: frozen["arms"][name] for name in names}
    client = Kaggle(args.kaggle, args.token_file)
    executor = SuspensionSafeExecutor(DockerExecutor(), args.output / "timeout-retries.jsonl")
    experiment = SizedComparison(executor, args.output, arms, tasks, split=split)
    root = experiment.destination
    entry = {
        "status": "opened",
        "run": str(args.output),
        "frozen_record": str(args.frozen),
        "arms": arms,
        "final_1.7b": frozen["final_1.7b"],
        "task_ids": experiment.config["task_ids"],
        "rule": "Opened once after every checkpoint and rule was frozen; never reopen.",
    }
    registry.parent.mkdir(parents=True, exist_ok=True)
    with registry.open("x") as handle:
        handle.write(json.dumps(entry, indent=2) + "\n")
    write_json(
        root / "experiment-plan.json",
        {
            **experiment.config,
            "frozen": frozen,
            "selection_rule": "Every task of the split in manifest order, opened once.",
            "episodes": len(arms) * len(experiment.tasks),
            "success_rule": "Explicit submit within budget; all public and private checks must pass.",
            "score_meaning": meaning,
            "no_retries_or_best_of": True,
            "checkpoint_selection_on_this_split": False,
        },
    )
    snapshot(root)
    client.require_gpu_quota(root / "quota-before.json")
    report = run_sized_rounds(client, experiment, root, kernel, title)
    write_json(
        registry,
        {
            **entry,
            "status": "completed",
            "successes": {name: arm["successes"] for name, arm in report["arms"].items()},
        },
    )
    return report


def parser_for(doc, flag):
    parser = argparse.ArgumentParser(description=doc)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    parser.add_argument("--frozen", type=Path, default=FROZEN)
    parser.add_argument("--registry", type=Path)
    parser.add_argument(flag, action="store_true", required=True)
    return parser


def main():
    args = parser_for(__doc__, "--open-sealed-final-test-2").parse_args()
    frozen = json.loads(args.frozen.read_text())
    open_once(
        args,
        args.registry or REGISTRY,
        frozen,
        list(frozen["arms"]),
        final2_tasks(),
        "final_test_2",
        "alwaysacing/repairlab-final-test-2",
        "RepairLab Final Test 2",
        "Autonomous repair score on unseen second-final-test programs.",
    )


if __name__ == "__main__":
    main()
