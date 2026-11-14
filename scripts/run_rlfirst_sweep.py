"""RL-first base sweep (plan addendum 1): the untrained model, 8 sampled episodes per task.

Runs on training-pool tasks only, split into shards that each fit about four hours at the
preflight's measured speed, two Kaggle sessions at a time. If the whole pool would need
more than 15 GPU hours, a seeded random subset that fits is swept and the rest stay
unswept. Rewards are audited afterwards in Docker by ``verify_rlfirst_sweep.py``.

Usage:
  python3 -B scripts/run_rlfirst_sweep.py --pool artifacts/rlfirst-pool-v2 \
    --preflight artifacts/rlfirst-preflight-v1 --output artifacts/rlfirst-sweep-v1 \
    --token-file ~/.kaggle/access_token --kaggle .venv/bin/kaggle
"""

import argparse
import json
import random
import shutil
import sys
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_client import Kaggle, embedded_payload  # noqa: E402
from run_rl_training import kernel_metadata, submit  # noqa: E402
from run_rlfirst_preflight import payload_tasks, shipped_code  # noqa: E402

from repairlab.experiments import write_json  # noqa: E402
from repairlab.rlfirst import fingerprint  # noqa: E402
from repairlab.rlfirst.pool import SEED  # noqa: E402
from repairlab.scaleup.models import LARGE  # noqa: E402
from repairlab.tasks import sha256  # noqa: E402

SAMPLES = 8
SHARD_HOURS = 4.0
BUDGET_HOURS = 15.0
ASSUMED_TURNS = 1.8  # generations per episode before the sweep measures it
PAYLOAD_LIMIT = 900_000
KERNELS = ("alwaysacing/repairlab-rlfirst-sweep-a", "alwaysacing/repairlab-rlfirst-sweep-b")


def plan_shards(rows, sequences_per_hour):
    episodes_per_hour = sequences_per_hour / ASSUMED_TURNS
    tasks_per_hour = episodes_per_hour / SAMPLES
    rng = random.Random(SEED + 2)
    order = list(rows)
    rng.shuffle(order)
    fit = int(tasks_per_hour * BUDGET_HOURS)
    subset = len(order) > fit
    chosen = order[:fit] if subset else order
    # At least two shards, so both Kaggle sessions work at once.
    size = max(1, min(int(tasks_per_hour * SHARD_HOURS), -(-len(chosen) // 2)))
    shards = [chosen[i : i + size] for i in range(0, len(chosen), size)]
    return shards, subset, tasks_per_hour


def payload_for(shard_rows, decision, index):
    programs, tasks = payload_tasks(shard_rows)
    return {
        "kind": "rlfirst_sweep",
        "shard": index,
        "model": LARGE,
        "dtype": "fp16" if decision["use_fp16"] else "bf16",
        "sampler": "vllm" if decision["use_vllm"] else "hf",
        "code": shipped_code(),
        "programs": programs,
        "tasks": tasks,
        "samples": SAMPLES,
        "batch_size": decision["batch_size"],
        "tasks_per_chunk": 64,
        "seed": SEED + 1000 * index,
    }


def fits(payload):
    return len(embedded_payload(payload, True).encode()) < PAYLOAD_LIMIT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    parser.add_argument("--lanes", type=int, default=2, choices=(1, 2))
    args = parser.parse_args()
    client = Kaggle(args.kaggle, args.token_file)
    decision = json.loads((args.preflight / "preflight-decision.json").read_text())
    if not decision["in_session_grading_ok"]:
        raise SystemExit("Preflight: in-session grading failed its controls")
    rows = json.loads((args.pool / "pool.json").read_text())["tasks"]
    shards, subset, tasks_per_hour = plan_shards(rows, decision["sampler_sequences_per_hour"])
    payloads = []
    for index, shard in enumerate(shards):
        payload = payload_for(shard, decision, index)
        if not fits(payload):
            raise SystemExit(f"Shard {index} payload too large; lower SHARD_HOURS")
        payloads.append(payload)
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    write_json(
        root / "experiment-plan.json",
        {
            "kind": "rlfirst_base_sweep_training_pool_only",
            "samples": SAMPLES,
            "decision": decision,
            "estimated_tasks_per_gpu_hour": round(tasks_per_hour, 1),
            "subset": subset,
            "tasks": sum(len(s) for s in shards),
            "pool_tasks": len(rows),
            "shards": [[r["task_id"] for r in s] for s in shards],
            "pool_summary_sha256": sha256((args.pool / "summary.json").read_text()),
            "rlfirst_fingerprint": fingerprint(),
            "rule": "docs/rl-first-plan-20261107.md, Addendum 1 item 2",
        },
    )
    for name in ("repairlab", "scripts", "tests"):
        shutil.copytree(
            ROOT / name,
            root / "source-snapshot" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "final3"),
        )
    client.require_gpu_quota(root / "quota-before.json")
    failures = []

    def lane(kernel, indices):
        for index in indices:
            folder = root / "shards" / str(index)
            try:
                submit(
                    client,
                    folder,
                    f"rlfirst-sweep-{index}",
                    f"# RepairLab RL-first base sweep, shard {index}\n\nThe untrained model plays "
                    "8 sampled two-turn episodes on training-pool tasks. Nothing is trained.",
                    payloads[index],
                    "rlfirst_runtime.py",
                    "run_sweep",
                    "repairlab-rlfirst-sweep-*",
                    kernel_metadata(
                        kernel,
                        kernel.split("/")[1].replace("-", " ").title(),
                        "RepairLab_Rlfirst_Sweep.ipynb",
                        [],
                    ),
                )
                write_json(folder / "status.json", {"status": "completed"})
            except BaseException as exc:  # noqa: BLE001 - one lane's failure is recorded
                failures.append((index, str(exc)[:300]))
                write_json(
                    folder / "status.json", {"status": "needs_diagnosis", "error": str(exc)[:500]}
                )
                return

    kernels = KERNELS[: args.lanes]
    lanes = [list(range(i, len(shards), len(kernels))) for i in range(len(kernels))]
    threads = [threading.Thread(target=lane, args=(k, ix)) for k, ix in zip(kernels, lanes) if ix]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    client.call(["quota", "--format", "json"], root / "quota-after.json")
    status = "completed" if not failures else "needs_diagnosis"
    write_json(root / "status.json", {"status": status, "failures": failures})
    print("SWEEP", status, failures, flush=True)


if __name__ == "__main__":
    main()
