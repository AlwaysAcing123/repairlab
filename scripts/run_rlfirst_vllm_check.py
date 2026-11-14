"""Addendum 3: harden the notebook copies, re-check the canary, and retry vLLM (item 3).

Uses the same 48 training-pool tasks as preflight part A. vLLM is chosen only if it
loads the base model with a LoRA adapter, its greedy valid-reply and public-pass counts
are each within 2 of Hugging Face fp16's (43 and 37), and it samples at least twice as
fast as Hugging Face fp16's best rate (2,718 sequences per hour).

Usage:
  python3 -B scripts/run_rlfirst_vllm_check.py --pool artifacts/rlfirst-pool-v3 \
    --preflight artifacts/rlfirst-preflight-v1 --output artifacts/rlfirst-vllm-check-v1 \
    --token-file ~/.kaggle/access_token --kaggle .venv/bin/kaggle
"""

import argparse
import json
import secrets
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_client import Kaggle  # noqa: E402
from run_rl_training import kernel_metadata, submit  # noqa: E402
from run_rlfirst_preflight import SPEED, payload_tasks, shipped_code  # noqa: E402

from repairlab.experiments import write_json  # noqa: E402
from repairlab.rlfirst import fingerprint  # noqa: E402
from repairlab.scaleup.models import LARGE  # noqa: E402

KERNEL = "alwaysacing/repairlab-rlfirst-vllm-check"


def decide(results, preflight):
    fp16 = next(r for r in preflight["dtype_check"] if r["dtype"] == "fp16")
    hf_best = max(r["sequences_per_hour"] for r in preflight["hf_sampling"] if r["ok"])
    vllm = results.get("vllm", {})
    rate = 0
    if vllm.get("run_ok") and vllm.get("sample_seconds"):
        rate = round(vllm["sample_sequences"] / vllm["sample_seconds"] * 3600)
    use = bool(
        vllm.get("run_ok")
        and abs(vllm.get("valid_replies", -99) - fp16["valid_replies"]) <= 2
        and abs(vllm.get("public_passes", -99) - fp16["public_passes"]) <= 2
        and rate >= 2 * hf_best
    )
    return {
        "use_vllm": use,
        "vllm_sequences_per_hour": rate,
        "hf_fp16_sequences_per_hour": hf_best,
        "vllm_valid_replies": vllm.get("valid_replies"),
        "vllm_public_passes": vllm.get("public_passes"),
        "hf_fp16_valid_replies": fp16["valid_replies"],
        "hf_fp16_public_passes": fp16["public_passes"],
        "readable_after_hardening": results.get("readable_after_hardening"),
        "canary_after_hardening": results.get("canary_after_hardening", {}).get(
            "readable_paths_with_payload_text"
        ),
        "rule": "docs/rl-first-plan-20261107.md, Addendum 3 item 1",
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
    plan = json.loads((args.preflight / "experiment-plan.json").read_text())
    preflight = json.loads(next((args.preflight / "run").rglob("preflight.json")).read_text())
    rows = {r["task_id"]: r for r in json.loads((args.pool / "pool.json").read_text())["tasks"]}
    chosen = [rows[task_id] for task_id in plan["task_ids"]]
    programs, tasks = payload_tasks(chosen)
    canary = "canary-" + secrets.token_hex(16)
    payload = {
        "kind": "rlfirst_vllm_check",
        "model": LARGE,
        "code": shipped_code(),
        "programs": programs,
        "tasks": tasks,
        "speed": SPEED,
        "canary": canary,
    }
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    write_json(
        root / "experiment-plan.json",
        {
            "kind": "rlfirst_vllm_check_not_a_result",
            "task_ids": plan["task_ids"],
            "rlfirst_fingerprint": fingerprint(),
            "rule": "docs/rl-first-plan-20261107.md, Addendum 3",
        },
    )
    for name in ("repairlab", "scripts", "tests"):
        shutil.copytree(
            ROOT / name,
            root / "source-snapshot" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "final3"),
        )
    client.require_gpu_quota(root / "quota-before.json")
    heading = (
        "# RepairLab RL-first vLLM check\n\nHardens the notebook copies, re-checks the canary, "
        "and measures vLLM against the earlier fp16 reference on 48 training-pool tasks.\n\n"
        f"Marker: {canary}"
    )
    try:
        extracted, _ = submit(
            client,
            root / "run",
            "rlfirst-vllm-check",
            heading,
            payload,
            "rlfirst_runtime.py",
            "run_vllm_check",
            "repairlab-rlfirst-vllm-*",
            kernel_metadata(
                KERNEL, "RepairLab Rlfirst Vllm Check", "RepairLab_Rlfirst_Vllm.ipynb", []
            ),
        )
        results = json.loads(next(extracted.rglob("vllm-check.json")).read_text())
        decision = decide(results, preflight)
        write_json(root / "vllm-decision.json", decision)
        client.call(["quota", "--format", "json"], root / "quota-after.json")
        write_json(root / "status.json", {"status": "completed"})
        print("VLLM", json.dumps(decision), flush=True)
    except BaseException as exc:
        write_json(root / "status.json", {"status": "needs_diagnosis", "error": str(exc)[:500]})
        raise


if __name__ == "__main__":
    main()
