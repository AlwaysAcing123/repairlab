"""Greedy evaluation of one RL-first arm on the development sets (never a sealed test).

The Kaggle session receives specifications, buggy functions and public cases only; hidden
cases stay on the laptop, where ``verify_rlfirst_eval.py`` grades every final submission
in Docker. An adapter, if any, comes from a finished kernel's output as a kernel source.

Usage:
  python3 -B scripts/run_rlfirst_eval.py --arm base --pool artifacts/rlfirst-pool-v2 \
    --preflight artifacts/rlfirst-preflight-v1 --output artifacts/rlfirst-dev-base-v1 \
    --token-file ~/.kaggle/access_token --kaggle .venv/bin/kaggle \
    [--adapter-kernel alwaysacing/repairlab-rlfirst-hardened-b --adapter-glob '**/repairlab-rlfirst-rl-hardened-*/state/adapter']
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_client import Kaggle  # noqa: E402
from run_rl_training import kernel_metadata, submit  # noqa: E402
from run_rlfirst_preflight import shipped_code  # noqa: E402
from run_rlfirst_sft import combined_runtime  # noqa: E402

from repairlab.experiments import write_json  # noqa: E402
from repairlab.rlfirst import fingerprint  # noqa: E402
from repairlab.scaleup.models import LARGE  # noqa: E402
from repairlab.tasks import sha256  # noqa: E402

SETS = ("dev-seen", "dev-unseen")


def public_only(row):
    return {
        "task_id": row["task_id"],
        "kind": row["kind"],
        "specification": row["specification"],
        "signature": row["signature"],
        "buggy": row["buggy"],
        "public": row["public"],
        "hidden": [],
        "bug_public_result": row["bug_public_result"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    parser.add_argument("--adapter-kernel")
    parser.add_argument("--adapter-glob")
    parser.add_argument(
        "--adapter-sha256", help="refuse in-session unless the mounted adapter has this hash"
    )
    args = parser.parse_args()
    if bool(args.adapter_kernel) != bool(args.adapter_glob):
        raise SystemExit("Give both --adapter-kernel and --adapter-glob, or neither")
    client = Kaggle(args.kaggle, args.token_file)
    decision = json.loads((args.preflight / "preflight-decision.json").read_text())
    rows = []
    for name in SETS:
        rows += json.loads((args.pool / f"{name}.json").read_text())["tasks"]
    tasks = [public_only(r) for r in rows]
    payload = {
        "kind": "rlfirst_eval",
        "arm": args.arm,
        "model": LARGE,
        "dtype": "fp16" if decision["use_fp16"] else "bf16",
        "code": shipped_code(),
        "tasks": tasks,
        "batch_size": decision["batch_size"],
        "adapter_glob": args.adapter_glob,
        "adapter_sha256": args.adapter_sha256,
    }
    if any(t["hidden"] for t in payload["tasks"]):
        raise SystemExit("Hidden cases would leave the laptop")
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    write_json(
        root / "experiment-plan.json",
        {
            "kind": "rlfirst_development_evaluation",
            "arm": args.arm,
            "sets": list(SETS),
            "tasks": len(tasks),
            "adapter_kernel": args.adapter_kernel,
            "adapter_glob": args.adapter_glob,
            "adapter_sha256": args.adapter_sha256,
            "dtype": payload["dtype"],
            "pool_summary_sha256": sha256((args.pool / "summary.json").read_text()),
            "rlfirst_fingerprint": fingerprint(),
            "rule": "docs/rl-first-plan-20261107.md, Splits and Arms (development only)",
        },
    )
    for name in ("repairlab", "scripts", "tests"):
        shutil.copytree(
            ROOT / name,
            root / "source-snapshot" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "final3"),
        )
    runtime, entry = "rlfirst_runtime.py", "run_eval"
    if args.adapter_sha256:
        runtime = combined_runtime(root / "runtime-combined.py", "rlfirst_checked_eval_runtime.py")
        entry = "run_checked_eval"
    client.require_gpu_quota(root / "quota-before.json")
    kernel = f"alwaysacing/repairlab-rlfirst-eval-{args.arm}"
    try:
        submit(
            client,
            root / "run",
            f"rlfirst-eval-{args.arm}",
            f"# RepairLab RL-first development evaluation ({args.arm})\n\nGreedy two-turn "
            "episodes with public cases only; hidden grading happens elsewhere.",
            payload,
            runtime,
            entry,
            "repairlab-rlfirst-eval-*",
            kernel_metadata(
                kernel,
                kernel.split("/")[1].replace("-", " ").title(),
                "RepairLab_Rlfirst_Eval.ipynb",
                [args.adapter_kernel] if args.adapter_kernel else [],
            ),
        )
        client.call(["quota", "--format", "json"], root / "quota-after.json")
        write_json(root / "status.json", {"status": "completed_needs_grading"})
        print("EVAL", args.arm, "downloaded; run verify_rlfirst_eval.py", flush=True)
    except BaseException as exc:
        write_json(root / "status.json", {"status": "needs_diagnosis", "error": str(exc)[:500]})
        raise


if __name__ == "__main__":
    main()
