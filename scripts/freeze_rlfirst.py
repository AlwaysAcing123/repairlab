"""Freeze the four RL-first arms and the rules before sealed test 3 opens (plan, Opening).

Refuses unless:
- both RL runs finished (all steps, or a collapse) and their in-place audits found no
  problem;
- the supervised run finished and trained exactly its chosen tasks;
- each arm's development evaluation was graded in Docker with no mismatch.
It then records each trained arm's kernel, version, adapter glob, and the SHA-256 of its
adapter weights (from the laptop's copy of the kernel output). It also records the
protocol fingerprints, the runtime hashes, and the hash of the plan's Rules section. The
record is written once and never overwritten.

Usage:
  python3 -B scripts/freeze_rlfirst.py --hardened-run artifacts/rlfirst-rl-hardened-v1 \
    --weak-run artifacts/rlfirst-rl-weak-v1 --sft-run artifacts/rlfirst-sft-v1 \
    --dev base=artifacts/rlfirst-dev-base-v1 --dev hardened=... --dev weak=... \
    --dev supervised=...
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from repairlab.final3 import fingerprint as final3_fingerprint  # noqa: E402
from repairlab.rlfirst import fingerprint  # noqa: E402

FROZEN = ROOT / "artifacts" / "rlfirst-frozen.json"
PLAN = ROOT / "docs" / "rl-first-plan-20261107.md"
ARMS = ("base", "hardened", "weak", "supervised")
RUNTIMES = ("rlfirst_runtime.py", "rlfirst_checked_eval_runtime.py")


def file_sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rules_text():
    """The plan's Rules section, from its heading up to the next heading."""
    text = PLAN.read_text()
    start = text.index("\n## Rules\n")
    return text[start : text.index("\n## ", start + 1)]


def protocol():
    return {
        "rlfirst_fingerprint": fingerprint(),
        "final3_fingerprint": final3_fingerprint(),
        "runtimes_sha256": {n: file_sha256(ROOT / "scripts" / n) for n in RUNTIMES},
        "rules_sha256": hashlib.sha256(rules_text().encode()).hexdigest(),
    }


def one(paths, what):
    paths = sorted(paths)
    if len(paths) != 1:
        raise SystemExit(f"Expected one {what}, found {len(paths)}")
    return paths[0]


def rl_arm(run, arm):
    plan = json.loads((run / "experiment-plan.json").read_text())
    status = json.loads((run / "status.json").read_text())
    if plan["arm"] != arm or status.get("status") != "completed":
        raise SystemExit(f"{run} is not a finished {arm} run: {status}")
    if status["stop_reason"] not in ("steps_complete", "collapsed"):
        raise SystemExit(f"{run} stopped for {status['stop_reason']}")
    audit = json.loads((run / "rl-verification.json").read_text())
    if audit["problem_count"] or audit["steps"] != status["state_step"] or not audit["token_check"]:
        raise SystemExit(f"{run} audit is not clean and complete")
    folder = run / "chunks" / str(status["chunk"])
    weights = one(
        folder.glob("extracted/**/state/adapter/adapter_model.safetensors"), "final adapter"
    )
    return {
        "run": str(run),
        "kernel": status["kernel"],
        "kernel_version": status["version"],
        "adapter_glob": f"**/repairlab-rlfirst-rl-{arm}-*/state/adapter",
        "adapter_sha256": file_sha256(weights),
        "steps": status["state_step"],
        "stop_reason": status["stop_reason"],
        "audit_flagged_submissions": audit["flagged_count"],
    }


def sft_arm(run):
    status = json.loads((run / "status.json").read_text())
    plan = json.loads((run / "experiment-plan.json").read_text())
    if status.get("status") != "completed":
        raise SystemExit(f"{run} is not finished: {status}")
    log = json.loads(one((run / "run" / "extracted").rglob("log.json"), "supervised log").read_text())
    if sorted(log["order"]) != plan["chosen_task_ids"]:
        raise SystemExit("The supervised arm did not train exactly its chosen tasks")
    weights = one(
        (run / "run" / "extracted").glob("**/state/adapter/adapter_model.safetensors"),
        "supervised adapter",
    )
    return {
        "run": str(run),
        "kernel": status["kernel"],
        "kernel_version": status["version"],
        "adapter_glob": status["adapter_glob"],
        "adapter_sha256": file_sha256(weights),
        "optimizer_steps": status["optimizer_steps"],
        "tasks": len(log["order"]),
    }


def development(run, arm, spec):
    plan = json.loads((run / "experiment-plan.json").read_text())
    results = json.loads((run / "eval-results.json").read_text())
    if plan["arm"] != arm or results["mismatches"]:
        raise SystemExit(f"{run} is not a clean development evaluation of {arm}")
    if spec.get("kernel") != plan.get("adapter_kernel"):
        raise SystemExit(f"{run} evaluated a different adapter kernel than the {arm} run")
    if plan.get("adapter_sha256") not in (None, spec.get("adapter_sha256")):
        raise SystemExit(f"{run} checked a different adapter hash")
    return {
        "run": str(run),
        "adapter_hash_checked_in_session": plan.get("adapter_sha256") is not None,
        "solved": {name: part["solved"] for name, part in results["summary"].items()},
        "tasks": {name: part["tasks"] for name, part in results["summary"].items()},
        "false_passes": {name: part["false_passes"] for name, part in results["summary"].items()},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hardened-run", type=Path, required=True)
    parser.add_argument("--weak-run", type=Path, required=True)
    parser.add_argument("--sft-run", type=Path, required=True)
    parser.add_argument("--dev", action="append", required=True, help="arm=run directory")
    parser.add_argument("--output", type=Path, default=FROZEN)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Already frozen: " + str(args.output))
    dev = dict(item.split("=", 1) for item in args.dev)
    if sorted(dev) != sorted(ARMS):
        raise SystemExit("Give one --dev for each of " + ", ".join(ARMS))
    arms = {
        "base": {"kernel": None, "adapter_glob": None, "adapter_sha256": None},
        "hardened": rl_arm(args.hardened_run, "hardened"),
        "weak": rl_arm(args.weak_run, "weak"),
        "supervised": sft_arm(args.sft_run),
    }
    for arm in ARMS:
        arms[arm]["development"] = development(Path(dev[arm]), arm, arms[arm])
    record = {
        "kind": "rlfirst_frozen_record",
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "arms": arms,
        **protocol(),
        "plan": str(PLAN.relative_to(ROOT)),
        "rules": rules_text(),
        "opening": "scripts/run_final_test_3.py, once, all four arms on all sealed tasks",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps({a: arms[a].get("adapter_sha256") for a in ARMS}, indent=2))


if __name__ == "__main__":
    main()
