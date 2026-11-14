"""RL-first training for one arm (hardened or weak grader), in resumable Kaggle chunks.

Each chunk runs ``run_rl`` for at most ``CHUNK_SECONDS``, then saves the adapter, optimizer,
random-number state and pool state. The next chunk runs on the arm's other kernel slug and
attaches the previous chunk's output as a kernel source. Both arms use the same seed and
settings; only the reward differs. Tasks the base sweep solved 8 of 8 times are retired.

Usage:
  python3 -B scripts/run_rlfirst_rl.py --arm hardened --steps N --pool artifacts/rlfirst-pool-v2 \
    --preflight artifacts/rlfirst-preflight-v1 --partb artifacts/rlfirst-partb-v1 \
    --sweep artifacts/rlfirst-sweep-v1 \
    --output artifacts/rlfirst-rl-hardened-v1 --token-file ~/.kaggle/access_token \
    --kaggle .venv/bin/kaggle [--resume]
"""

import argparse
import json
import shutil
import sys
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

CHUNK_SECONDS = 4 * 3600 - 30 * 60  # four-hour sessions, minus loading and saving
SETTINGS = {
    "groups": 32,
    "samples": 8,
    "max_draws": 96,
    "draw_chunk": 24,
    "lr": 2e-5,
    "clip": 1.0,
    "max_sequence_tokens": 4096,
    "chunk_seconds": CHUNK_SECONDS,
    "vllm_utilization": 0.55,
}
PAYLOAD_LIMIT = 900_000


def slugs(arm):
    return [f"alwaysacing/repairlab-rlfirst-{arm}-a", f"alwaysacing/repairlab-rlfirst-{arm}-b"]


def build_payload(args, decision, part_b, sweep):
    rows = json.loads((args.pool / "pool.json").read_text())["tasks"]
    retired = set(sweep["retired_before_rl"])
    rows = [r for r in rows if r["task_id"] not in retired]
    programs, tasks = payload_tasks(rows)
    payload = {
        "kind": "rlfirst_rl",
        "arm": args.arm,
        "model": LARGE,
        "dtype": "fp16" if decision["use_fp16"] else "bf16",
        "sampler": "vllm" if decision["use_vllm"] else "hf",
        "code": shipped_code(),
        "programs": programs,
        "tasks": tasks,
        "seed": SEED + 7,
        "batch_size": decision["batch_size"],
        "rl": {
            **SETTINGS,
            "steps_total": args.steps,
            "micro_batch_tokens": part_b["micro_batch_tokens"],
            "first_step_estimate": part_b["step_seconds"],
            "collapse_threshold": sweep["collapse_threshold"],
        },
    }
    if len(embedded_payload(payload, True).encode()) >= PAYLOAD_LIMIT:
        raise SystemExit("RL payload too large for one notebook")
    return payload, len(retired)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("hardened", "weak"), required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--partb", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--max-chunks",
        type=int,
        help="start at most this many chunks in this invocation, then pause (quota planning)",
    )
    args = parser.parse_args()
    if args.steps < 30:
        raise SystemExit("The plan requires at least 30 steps")
    client = Kaggle(args.kaggle, args.token_file)
    decision = json.loads((args.preflight / "preflight-decision.json").read_text())
    part_b = json.loads((args.partb / "part-b-decision.json").read_text())
    sweep = json.loads((args.sweep / "sweep-verification.json").read_text())
    if sweep["mismatch_count"]:
        raise SystemExit("The sweep audit has mismatches; resolve them before RL")
    payload, retired = build_payload(args, decision, part_b, sweep)
    root = args.output
    if args.resume:
        if not (root / "experiment-plan.json").exists():
            raise SystemExit("Nothing to resume")
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json(
            root / "experiment-plan.json",
            {
                "kind": "rlfirst_rl_training",
                "arm": args.arm,
                "steps": args.steps,
                "settings": payload["rl"],
                "dtype": payload["dtype"],
                "seed": payload["seed"],
                "tasks": len(payload["tasks"]),
                "retired_before_rl": retired,
                "pool_summary_sha256": sha256((args.pool / "summary.json").read_text()),
                "rlfirst_fingerprint": fingerprint(),
                "rule": "docs/rl-first-plan-20261107.md, RL method and Addendum 1",
            },
        )
        for name in ("repairlab", "scripts", "tests"):
            shutil.copytree(
                ROOT / name,
                root / "source-snapshot" / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "final3"),
            )
    names = slugs(args.arm)
    started = 0
    while True:
        done = sorted(int(p.parent.name) for p in (root / "chunks").glob("*/chunk-result.json"))
        chunk = (done[-1] + 1) if done else 1
        if done:
            last = json.loads((root / "chunks" / str(done[-1]) / "chunk-result.json").read_text())
            if last["stop_reason"] in ("steps_complete", "collapsed"):
                write_json(root / "status.json", {"status": "completed", **last})
                print("RL", args.arm, "COMPLETE", json.dumps(last), flush=True)
                return
        if args.max_chunks is not None and started >= args.max_chunks:
            write_json(root / "status.json", {"status": "paused", "next_chunk": chunk})
            print("RL", args.arm, "PAUSED before chunk", chunk, flush=True)
            return
        started += 1
        folder = root / "chunks" / str(chunk)
        if folder.exists():
            index = 1
            while (folder.parent / f"{chunk}-abandoned-{index}").exists():
                index += 1
            folder.rename(folder.parent / f"{chunk}-abandoned-{index}")
        kernel = names[(chunk - 1) % 2]
        previous = names[chunk % 2]
        chunk_payload = dict(payload)
        sources = []
        if chunk > 1:
            # Kaggle mounts a source kernel at /kaggle/input/notebooks/<owner>/<slug>/ (a
            # CPU probe showed this before chunk 2; a one-level glob found nothing).
            chunk_payload["resume_glob"] = f"**/repairlab-rlfirst-rl-{args.arm}-*/state"
            sources = [previous]
        client.require_gpu_quota(folder / "quota-before.json")
        write_json(root / "status.json", {"status": "running", "chunk": chunk})
        extracted, version = submit(
            client,
            folder,
            f"rlfirst-rl-{args.arm}-{chunk}",
            f"# RepairLab RL-first RL ({args.arm} grader), chunk {chunk}\n\n"
            "Samples, grades and updates in one session on training-pool tasks only.",
            chunk_payload,
            "rlfirst_runtime.py",
            "run_rl",
            "repairlab-rlfirst-rl-*",
            kernel_metadata(
                kernel,
                kernel.split("/")[1].replace("-", " ").title(),
                "RepairLab_Rlfirst_RL.ipynb",
                sources,
            ),
        )
        log = json.loads(next(extracted.rglob("log.json")).read_text())
        result = {
            "chunk": chunk,
            "kernel": kernel,
            "version": version,
            "stop_reason": log["stop_reason"],
            "state_step": log["state_step"],
            "steps_this_chunk": len(log["steps"]),
        }
        if chunk > 1 and result["steps_this_chunk"] == 0 and log["stop_reason"] == "chunk_time":
            raise SystemExit("A chunk made no progress; diagnose before continuing")
        write_json(folder / "chunk-result.json", result)
        print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
