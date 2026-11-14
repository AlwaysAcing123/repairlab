"""RL-first preflight, part B (plan item 4, addendum 1): one full RL step at batch size.

Samples one step's 32 informative groups with the base model plus a fresh adapter, then
times the update at three micro-batch token budgets. The laptop chooses the largest budget
that fits under 13 GiB and the number of steps that fit 8 GPU hours (at least 30, or only
the hardened arm runs with 16 hours). Nothing is saved as a model.

Usage:
  python3 -B scripts/run_rlfirst_partb.py --pool artifacts/rlfirst-pool-v2 \
    --preflight artifacts/rlfirst-preflight-v1 --output artifacts/rlfirst-partb-v1 \
    --token-file ~/.kaggle/access_token --kaggle .venv/bin/kaggle
"""

import argparse
import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_client import Kaggle  # noqa: E402
from run_rl_training import kernel_metadata, submit  # noqa: E402
from run_rlfirst_preflight import payload_tasks, shipped_code  # noqa: E402
from run_rlfirst_rl import CHUNK_SECONDS, SETTINGS  # noqa: E402

from repairlab.experiments import write_json  # noqa: E402
from repairlab.rlfirst import fingerprint  # noqa: E402
from repairlab.rlfirst.pool import SEED  # noqa: E402
from repairlab.scaleup.models import LARGE  # noqa: E402

KERNEL = "alwaysacing/repairlab-rlfirst-partb"
BUDGETS = [16384, 8192, 4096]
LIMIT_GIB = 13.0
RUN_HOURS = 8.0
CHUNK_OVERHEAD_SECONDS = 15 * 60  # model load, install, and state saving per chunk


def decide(results):
    fitting = [r for r in results["updates"] if r.get("ok") and r["peak_gib"] < LIMIT_GIB]
    if not fitting:
        return {"proceed": False, "reason": "no micro-batch budget fits"}
    best = max(fitting, key=lambda r: r["micro_batch_tokens"])
    step_seconds = results["sampling"]["seconds"] + best["seconds"]
    chunks = math.ceil(RUN_HOURS * 3600 / CHUNK_SECONDS)
    usable = RUN_HOURS * 3600 - chunks * CHUNK_OVERHEAD_SECONDS
    steps = int(usable // step_seconds)
    return {
        "proceed": True,
        "micro_batch_tokens": best["micro_batch_tokens"],
        "step_seconds": round(step_seconds),
        "sampling_seconds": results["sampling"]["seconds"],
        "update_seconds": best["seconds"],
        "update_peak_gib": best["peak_gib"],
        "steps_in_8_hours": steps,
        "both_arms": steps >= 30,
        "steps": steps if steps >= 30 else int((2 * usable) // step_seconds),
        "rule": "docs/rl-first-plan-20261107.md, RL method 'Length' and Preflight",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    args = parser.parse_args()
    client = Kaggle(args.kaggle, args.token_file)
    decision = json.loads((args.preflight / "preflight-decision.json").read_text())
    rows = json.loads((args.pool / "pool.json").read_text())["tasks"]
    programs, tasks = payload_tasks(rows)
    payload = {
        "kind": "rlfirst_partb",
        "model": LARGE,
        "dtype": "fp16" if decision["use_fp16"] else "bf16",
        "sampler": "vllm" if decision["use_vllm"] else "hf",
        "code": shipped_code(),
        "programs": programs,
        "tasks": tasks,
        "seed": SEED + 3,
        "batch_size": decision["batch_size"],
        "rl": dict(SETTINGS),
        "micro_budgets": BUDGETS,
    }
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    write_json(
        root / "experiment-plan.json",
        {
            "kind": "rlfirst_preflight_part_b_not_a_result",
            "budgets": BUDGETS,
            "settings": SETTINGS,
            "dtype": payload["dtype"],
            "rlfirst_fingerprint": fingerprint(),
        },
    )
    for name in ("repairlab", "scripts", "tests"):
        shutil.copytree(
            ROOT / name,
            root / "source-snapshot" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "final3"),
        )
    client.require_gpu_quota(root / "quota-before.json")
    try:
        extracted, _ = submit(
            client,
            root / "run",
            "rlfirst-partb",
            "# RepairLab RL-first preflight (part B)\n\nOne full RL step at batch size on "
            "training-pool tasks, timed at three micro-batch sizes. Nothing is saved.",
            payload,
            "rlfirst_runtime.py",
            "run_part_b",
            "repairlab-rlfirst-partb-*",
            kernel_metadata(KERNEL, "RepairLab Rlfirst Partb", "RepairLab_Rlfirst_PartB.ipynb", []),
        )
        results = json.loads(next(extracted.rglob("partb.json")).read_text())
        verdict = decide(results)
        write_json(root / "part-b-decision.json", verdict)
        client.call(["quota", "--format", "json"], root / "quota-after.json")
        write_json(root / "status.json", {"status": "completed"})
        print("PART B", json.dumps(verdict), flush=True)
    except BaseException as exc:
        write_json(root / "status.json", {"status": "needs_diagnosis", "error": str(exc)[:500]})
        raise


if __name__ == "__main__":
    main()
