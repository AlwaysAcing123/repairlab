"""Train the supervised arm (plan, Arms item 4; Addendum 8) after the hardened RL run.

Counts the distinct tasks the finished hardened run drew (every episode of every saved
step, abandoned chunks excluded). Chooses that many tasks by seed from the same list the
RL arms draw from (the pool minus the tasks the base sweep retired), and trains one pass
on them in one Kaggle session. The notebook embeds ``rlfirst_runtime.py`` followed by
``rlfirst_sft_runtime.py``, so the RL arms' loading, LoRA, prompt and update code is
reused unchanged.

Usage:
  python3 -B scripts/run_rlfirst_sft.py --pool artifacts/rlfirst-pool-v3 \
    --preflight artifacts/rlfirst-settings-v2 --sweep artifacts/rlfirst-sweep-v1 \
    --rl-run artifacts/rlfirst-rl-hardened-v1 --output artifacts/rlfirst-sft-v1 \
    --token-file ~/.kaggle/access_token --kaggle .venv/bin/kaggle
"""

import argparse
import gzip
import json
import os
import random
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_client import Kaggle, embedded_payload  # noqa: E402
from run_rl_training import kernel_metadata, submit  # noqa: E402
from run_rlfirst_preflight import payload_tasks, shipped_code  # noqa: E402
from run_rlfirst_rl import PAYLOAD_LIMIT, SETTINGS  # noqa: E402

from repairlab.experiments import write_json  # noqa: E402
from repairlab.rlfirst import fingerprint  # noqa: E402
from repairlab.rlfirst.pool import SEED  # noqa: E402
from repairlab.scaleup.models import LARGE  # noqa: E402
from repairlab.tasks import sha256  # noqa: E402

KERNEL = "alwaysacing/repairlab-rlfirst-sft"
SFT = {
    "batch_tasks": 8,
    "lr": 1e-4,
    "clip": 1.0,
    "micro_batch_tokens": None,  # filled from part B, as in the RL arms
    "max_sequence_tokens": SETTINGS["max_sequence_tokens"],
    "max_overflow_retries": 20,
}


def drawn_tasks(run):
    """Distinct task ids in every saved step of a finished RL run (abandoned chunks excluded)."""
    status = json.loads((run / "status.json").read_text())
    if status.get("status") != "completed":
        raise SystemExit(f"{run} has not finished: {status}")
    seen = set()
    files = [
        p
        for p in run.glob("chunks/*/extracted/**/steps/step-*.json.gz")
        if "abandoned" not in str(p.relative_to(run))
    ]
    numbers = sorted(int(p.name.split("-")[1].split(".")[0]) for p in files)
    if numbers != list(range(1, status["state_step"] + 1)):
        raise SystemExit(f"Saved steps {numbers[:3]}...{numbers[-3:]} do not match the run")
    for path in files:
        for record in json.loads(gzip.open(path, "rt").read())["episodes"]:
            seen.add(record["task_id"])
    return seen


def choose(rl_rows, count, seed):
    """``count`` tasks by seed from the RL arms' list (all of it if ``count`` covers it)."""
    ids = sorted(r["task_id"] for r in rl_rows)
    random.Random(seed).shuffle(ids)
    chosen = set(ids[:count])
    return [r for r in rl_rows if r["task_id"] in chosen]


def combined_runtime(destination, extra="rlfirst_sft_runtime.py"):
    """``rlfirst_runtime.py`` followed by ``extra``, saved for the notebook builder."""
    text = (ROOT / "scripts" / "rlfirst_runtime.py").read_text()
    text += f"\n\n# ---- {extra} ----\n\n"
    text += (ROOT / "scripts" / extra).read_text()
    compile(text, "rlfirst-combined-runtime", "exec")
    destination.write_text(text)
    return os.path.relpath(destination, ROOT / "scripts")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--partb", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--rl-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    args = parser.parse_args()
    client = Kaggle(args.kaggle, args.token_file)
    decision = json.loads((args.preflight / "preflight-decision.json").read_text())
    part_b = json.loads((args.partb / "part-b-decision.json").read_text())
    sweep = json.loads((args.sweep / "sweep-verification.json").read_text())
    rl_plan = json.loads((args.rl_run / "experiment-plan.json").read_text())
    if rl_plan["arm"] != "hardened":
        raise SystemExit("The supervised arm copies the hardened run's task count")
    retired = set(sweep["retired_before_rl"])
    rows = json.loads((args.pool / "pool.json").read_text())["tasks"]
    rl_rows = [r for r in rows if r["task_id"] not in retired]
    drawn = drawn_tasks(args.rl_run)
    if not drawn <= {r["task_id"] for r in rl_rows}:
        raise SystemExit("The RL run drew a task outside the RL list")
    seed = SEED + 13
    chosen = choose(rl_rows, len(drawn), seed)
    programs, tasks = payload_tasks(chosen)
    references = {}
    for row in chosen:
        if references.setdefault(row["program"], row["reference"]) != row["reference"]:
            raise SystemExit("One program has two references: " + row["program"])
    payload = {
        "kind": "rlfirst_sft",
        "model": LARGE,
        "dtype": "fp16" if decision["use_fp16"] else "bf16",
        "code": shipped_code(),
        "programs": programs,
        "tasks": tasks,
        "references": references,
        "seed": seed,
        "sft": {**SFT, "micro_batch_tokens": part_b["micro_batch_tokens"]},
    }
    if len(embedded_payload(payload, True).encode()) >= PAYLOAD_LIMIT:
        raise SystemExit("Supervised payload too large for one notebook")
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    write_json(
        root / "experiment-plan.json",
        {
            "kind": "rlfirst_supervised_training",
            "hardened_run": str(args.rl_run),
            "hardened_distinct_tasks_drawn": len(drawn),
            "rl_list_tasks": len(rl_rows),
            "chosen_tasks": len(chosen),
            "chosen_task_ids": sorted(r["task_id"] for r in chosen),
            "chosen_equals_hardened_drawn": {r["task_id"] for r in chosen} == drawn,
            "seed": seed,
            "dtype": payload["dtype"],
            "settings": payload["sft"],
            "pool_summary_sha256": sha256((args.pool / "summary.json").read_text()),
            "rlfirst_fingerprint": fingerprint(),
            "rule": "docs/rl-first-plan-20261107.md, Arms item 4 and Addendum 8",
        },
    )
    for name in ("repairlab", "scripts", "tests"):
        shutil.copytree(
            ROOT / name,
            root / "source-snapshot" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "final3"),
        )
    runtime = combined_runtime(root / "runtime-combined.py")
    client.require_gpu_quota(root / "quota-before.json")
    try:
        extracted, version = submit(
            client,
            root / "run",
            "rlfirst-sft",
            "# RepairLab RL-first supervised arm\n\nOne pass of supervised LoRA training on "
            "training-pool tasks only; the target is the reference function.",
            payload,
            runtime,
            "run_sft",
            "repairlab-rlfirst-sft-*",
            kernel_metadata(KERNEL, "RepairLab Rlfirst Sft", "RepairLab_Rlfirst_SFT.ipynb", []),
        )
        log = json.loads(next(extracted.rglob("log.json")).read_text())
        if sorted(log["order"]) != sorted(r["task_id"] for r in chosen):
            raise RuntimeError("Trained tasks differ from the chosen tasks")
        client.call(["quota", "--format", "json"], root / "quota-after.json")
        result = {
            "status": "completed",
            "kernel": KERNEL,
            "version": version,
            "optimizer_steps": log["optimizer_steps"],
            "overflow_retries": sum(s["overflow_retries"] for s in log["steps"]),
            "adapter_glob": "**/repairlab-rlfirst-sft-*/state/adapter",
        }
        write_json(root / "status.json", result)
        print("SFT", json.dumps(result), flush=True)
    except BaseException as exc:
        write_json(root / "status.json", {"status": "needs_diagnosis", "error": str(exc)[:500]})
        raise


if __name__ == "__main__":
    main()
