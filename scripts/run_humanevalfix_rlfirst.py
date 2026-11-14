"""HumanEvalFix once, after sealed test 3, for all four frozen arms (plan, Opening; reported only).

Runs only after sealed test 3's registry is marked completed. It uses the 1.7B phase's
mechanical conversion (``native_tasks``: 117 scored tasks, admitted before any model ran)
in the RL-first two-turn format:
- the hand-written HumanEvalFix bug;
- its example tests as public cases, plus the first failing input when no example fails
  (as for every RL-first task);
- the other tests hidden.
Every task passes the pool's Docker controls. The registry entry is written before upload,
and HumanEvalFix is never reopened. Grade with
``verify_final_test_3.py --humanevalfix``, first on a copy, then in place.

Usage:
  python3 -B scripts/run_humanevalfix_rlfirst.py --frozen artifacts/rlfirst-frozen.json \
    --preflight artifacts/rlfirst-settings-v2 --output artifacts/humanevalfix-rlfirst-v1 \
    --token-file ~/.kaggle/access_token --kaggle .venv/bin/kaggle
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from freeze_rlfirst import ARMS, protocol  # noqa: E402
from kaggle_client import Kaggle  # noqa: E402
from run_final_test_3 import MIN_QUOTA_HOURS, admit, quota_hours, run_arms  # noqa: E402
from run_final_test_3 import REGISTRY as FINAL3_REGISTRY  # noqa: E402
from run_rlfirst_eval import public_only  # noqa: E402

from repairlab.execution import DockerExecutor  # noqa: E402
from repairlab.experiments import write_json  # noqa: E402
from repairlab.rlfirst.mutate import HAND_WRITTEN  # noqa: E402
from repairlab.rlfirst.pool import evaluate, record  # noqa: E402
from repairlab.rlfirst.sandbox import LocalExecutor  # noqa: E402
from repairlab.scaleup.humanevalfix_native import native_tasks  # noqa: E402

REGISTRY = ROOT / "artifacts" / "humanevalfix-rlfirst-registry.json"


def humanevalfix_rows(sandbox, docker):
    rows, report = [], []
    for program in native_tasks():
        passed = evaluate(sandbox, program, program.buggy)
        if passed is None or all(passed):
            report.append({"task": program.id, "rejected": "no failing input within limits"})
            continue
        built, info, control = admit(
            docker, program, program.buggy, HAND_WRITTEN, passed, "humanevalfix_native"
        )
        if not control["admitted"]:
            report.append({"task": program.id, "rejected": control["reason"]})
            continue
        rows.append(
            {
                **record(built),
                **info,
                "set": "humanevalfix",
                "description": "HumanEvalFix hand-written bug",
                "split": "humanevalfix_native",
                "bug_public_result": control["bug_public_result"],
            }
        )
    return rows, report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    args = parser.parse_args()
    if REGISTRY.exists():
        raise SystemExit("HumanEvalFix was already run for this phase: " + str(REGISTRY))
    if json.loads(FINAL3_REGISTRY.read_text()).get("status") != "completed":
        raise SystemExit("Sealed test 3 must be opened and graded first")
    frozen = json.loads(args.frozen.read_text())
    changed = [k for k, v in protocol().items() if frozen.get(k) != v]
    if changed:
        raise SystemExit("Changed since the freeze: " + ", ".join(changed))
    decision = json.loads((args.preflight / "preflight-decision.json").read_text())
    client = Kaggle(args.kaggle, args.token_file)
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    if quota_hours(client, root / "quota-check.json") < MIN_QUOTA_HOURS:
        raise SystemExit(f"Less than {MIN_QUOTA_HOURS} GPU hours left; run after the reset")
    rows, report = humanevalfix_rows(LocalExecutor(), DockerExecutor())
    write_json(root / "sealed-tasks.json", {"tasks": rows})
    write_json(root / "admission.json", {"tasks": len(rows), "report": report})
    tasks = [public_only(r) for r in rows]
    if any(t["hidden"] for t in tasks):
        raise SystemExit("Hidden cases would leave the laptop")
    entry = {
        "status": "opened",
        "run": str(root),
        "frozen_record": str(args.frozen),
        "frozen_sha256": hashlib.sha256(args.frozen.read_bytes()).hexdigest(),
        "arms": {a: frozen["arms"][a]["adapter_sha256"] for a in ARMS},
        "task_ids": [r["task_id"] for r in rows],
        "rule": "Reported only; run once after sealed test 3; never rerun.",
    }
    with REGISTRY.open("x") as handle:
        handle.write(json.dumps(entry, indent=2) + "\n")
    write_json(
        root / "experiment-plan.json",
        {
            "kind": "rlfirst_humanevalfix",
            "frozen": frozen,
            "episodes": len(ARMS) * len(tasks),
            "decoding": "greedy, two turns, Hugging Face fp16 (as the development evaluations)",
            "reported_only": True,
        },
    )
    for name in ("repairlab", "scripts", "tests"):
        shutil.copytree(
            ROOT / name,
            root / "source-snapshot" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    outcomes = run_arms(client, root, frozen, tasks, decision, "hef3", "RepairLab HumanEvalFix")
    client.call(["quota", "--format", "json"], root / "quota-after.json")
    failed = [a for a, o in outcomes.items() if o["status"] != "downloaded"]
    status = "failed_partway_needs_human_decision" if failed else "downloaded_needs_grading"
    write_json(REGISTRY, {**entry, "status": status, "outcomes": outcomes})
    write_json(root / "status.json", {"status": status, "outcomes": outcomes})
    print("HUMANEVALFIX", status, len(rows), "tasks", flush=True)


if __name__ == "__main__":
    main()
