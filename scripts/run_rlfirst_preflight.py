"""RL-first preflight, part A (plan addendum 1): measure, choose nothing on evaluation tasks.

On one private Kaggle T4 it records the environment, checks in-session grading against
known controls, searches the disk as the unprivileged candidate user for a canary placed
in the notebook text, compares greedy fp16 and bf16 replies on 48 training-pool tasks,
measures Hugging Face sampling and grading speed, and tries vLLM. The laptop then applies
the plan's fp16 and vLLM rules to the saved measurements.

Usage:
  python3 -B scripts/run_rlfirst_preflight.py --pool artifacts/rlfirst-pool-v2 \
    --output artifacts/rlfirst-preflight-v1 --token-file ~/.kaggle/access_token \
    --kaggle .venv/bin/kaggle
"""

import argparse
import json
import random
import secrets
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_client import Kaggle  # noqa: E402
from run_rl_training import kernel_metadata, submit  # noqa: E402

from repairlab.experiments import write_json  # noqa: E402
from repairlab.rlfirst import fingerprint  # noqa: E402
from repairlab.rlfirst.pool import (  # noqa: E402
    SEED,
    from_record,
    ordered_cases,
    reject_evaluation_material,
    training_sources,
)
from repairlab.scaleup.models import LARGE  # noqa: E402
from repairlab.tasks import canonical, sha256  # noqa: E402

KERNEL = "alwaysacing/repairlab-rlfirst-preflight"
SHIPPED = (
    "repairlab/__init__.py",
    "repairlab/tasks.py",
    "repairlab/execution.py",
    "repairlab/worker.py",
    "repairlab/rlfirst/__init__.py",
    "repairlab/rlfirst/episode.py",
    "repairlab/rlfirst/sandbox.py",
)
SPEED = {"batch_sizes": [32, 48, 64], "samples_per_prompt": 8}


def shipped_code():
    return {name: (ROOT / name).read_text() for name in SHIPPED}


def payload_tasks(rows):
    """Programs once (ordered cases) plus per-task public indices; training pool only."""
    sources = {t.id: t for _, t in training_sources()}
    templates = {}
    tasks = []
    programs = {}
    for row in rows:
        program = sources[row["program"]]
        templates[row["task_id"]] = from_record(row, program)
        order = ordered_cases(program)
        index = {canonical(c.args): i for i, c in enumerate(order)}
        public_index = [index[canonical(c["args"])] for c in row["public"]]
        hidden_index = sorted(set(range(len(order))) - set(public_index))
        if [index[canonical(c["args"])] for c in row["hidden"]] != hidden_index:
            raise ValueError("Hidden cases do not match the program order: " + row["task_id"])
        programs[program.id] = {
            "specification": program.specification,
            "signature": program.signature,
            "cases": [[c.args, c.expected] for c in order],
        }
        tasks.append(
            {
                "task_id": row["task_id"],
                "program": program.id,
                "kind": row["kind"],
                "buggy": row["buggy"],
                "public_index": public_index,
                "bug_public_result": row["bug_public_result"],
            }
        )
    reject_evaluation_material(list(templates.values()))
    return programs, tasks


def choose(pool_rows, count=48):
    """Seeded, kind-balanced, one task per program where possible."""
    rng = random.Random(SEED + 1)
    by_kind = {}
    for row in pool_rows:
        by_kind.setdefault(row["kind"], []).append(row)
    for rows in by_kind.values():
        rng.shuffle(rows)
    chosen, programs = [], set()
    kinds = sorted(by_kind)
    while len(chosen) < count:
        progressed = False
        for kind in kinds:
            options = [r for r in by_kind[kind] if r["program"] not in programs] or by_kind[kind]
            if not options or len(chosen) >= count:
                continue
            row = options[0]
            by_kind[kind].remove(row)
            chosen.append(row)
            programs.add(row["program"])
            progressed = True
        if not progressed:
            break
    return chosen


def decide(results):
    """The plan's fp16 and vLLM rules, applied to saved measurements only."""
    rows = {r["dtype"]: r for r in results.get("dtype_check", [])}
    bf16, fp16 = rows.get("bf16"), rows.get("fp16")
    use_fp16 = bool(
        bf16
        and fp16
        and fp16["all_logits_finite"]
        and abs(fp16["valid_replies"] - bf16["valid_replies"]) <= 2
        and abs(fp16["public_passes"] - bf16["public_passes"]) <= 2
    )
    hf = [r for r in results.get("hf_sampling", []) if r["ok"]]
    hf_best = max((r["sequences_per_hour"] for r in hf), default=0)
    vllm = results.get("vllm", {})
    vllm_rate = 0
    if vllm.get("run_ok") and vllm.get("sample_seconds"):
        vllm_rate = round(vllm["sample_sequences"] / vllm["sample_seconds"] * 3600)
    reference = fp16 if use_fp16 else bf16
    use_vllm = bool(
        vllm.get("run_ok")
        and reference
        and abs(vllm.get("valid_replies", -99) - reference["valid_replies"]) <= 2
        and abs(vllm.get("public_passes", -99) - reference["public_passes"]) <= 2
        and vllm_rate >= 2 * hf_best
    )
    controls = results.get("in_session_controls", {})
    grading_ok = controls.get("reference_passes") == controls.get("tasks") and controls.get(
        "bug_fails_public"
    ) == controls.get("tasks")
    canary = results.get("canary", {})
    return {
        "use_fp16": use_fp16,
        "use_vllm": use_vllm,
        "hf_best_sequences_per_hour": hf_best,
        "vllm_sequences_per_hour": vllm_rate,
        "sampler_sequences_per_hour": vllm_rate if use_vllm else hf_best,
        "batch_size": max((r["batch_size"] for r in hf), default=32),
        "in_session_grading_ok": grading_ok,
        "privilege_drop": results.get("environment", {}).get("sandbox", {}).get("dropped_to_user"),
        "network_isolated": results.get("environment", {})
        .get("sandbox", {})
        .get("network_isolated"),
        "payload_readable_by_candidate": canary.get("readable_paths_with_payload_text"),
        "rule": "docs/rl-first-plan-20261107.md, Preflight items 2-3 and Addendum 1",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    args = parser.parse_args()
    client = Kaggle(args.kaggle, args.token_file)
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    pool_rows = json.loads((args.pool / "pool.json").read_text())["tasks"]
    chosen = choose(pool_rows)
    programs, tasks = payload_tasks(chosen)
    canary = "canary-" + secrets.token_hex(16)
    references = {row["task_id"]: row["reference"] for row in chosen}
    payload = {
        "kind": "rlfirst_preflight",
        "model": LARGE,
        "code": shipped_code(),
        "programs": programs,
        "tasks": tasks,
        "references": references,
        "speed": SPEED,
        "canary": canary,
    }
    raw = json.dumps(payload)
    if client.token in raw:
        raise RuntimeError("Credential would enter the notebook")
    write_json(
        root / "experiment-plan.json",
        {
            "kind": "rlfirst_preflight_part_a_not_a_result",
            "task_ids": [t["task_id"] for t in tasks],
            "kinds": sorted({t["kind"] for t in tasks}),
            "pool_summary_sha256": sha256((args.pool / "summary.json").read_text()),
            "rlfirst_fingerprint": fingerprint(),
            "speed": SPEED,
            "rule": "docs/rl-first-plan-20261107.md, Preflight and Addendum 1",
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
        "# RepairLab RL-first preflight (part A)\n\n"
        "Measures the environment, in-session grading, fp16 against bf16, sampling speed, "
        "and vLLM on 48 training-pool tasks. Nothing is trained or saved as a model.\n\n"
        f"Marker: {canary}"
    )
    try:
        extracted, _ = submit(
            client,
            root / "run",
            "rlfirst-preflight",
            heading,
            payload,
            "rlfirst_runtime.py",
            "run_preflight",
            "repairlab-rlfirst-preflight-*",
            kernel_metadata(
                KERNEL, "RepairLab Rlfirst Preflight", "RepairLab_Rlfirst_Preflight.ipynb", []
            ),
        )
        results = json.loads(next(extracted.rglob("preflight.json")).read_text())
        decision = decide(results)
        write_json(root / "preflight-decision.json", decision)
        client.call(["quota", "--format", "json"], root / "quota-after.json")
        write_json(root / "status.json", {"status": "completed"})
        print("PREFLIGHT", json.dumps(decision), flush=True)
    except BaseException as exc:
        write_json(root / "status.json", {"status": "needs_diagnosis", "error": str(exc)[:500]})
        raise


if __name__ == "__main__":
    main()
