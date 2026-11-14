"""Open sealed test 3 exactly once: all four frozen arms on all sealed tasks (plan, Opening).

Needs the frozen record (``scripts/freeze_rlfirst.py``) and refuses if the protocol
fingerprints, runtimes, or rules changed since it was written.

Steps:
1. **Build the sealed tasks on the laptop.** No model is involved. For each of the 96
   sealed programs, in manifest order:
   - one generated bug, by the development-set recipe: single mutations that keep the
     signature, always finish, change some answer, and match no known bug or reference;
     one is drawn with a fixed seed, and if it fails the Docker controls the next is drawn
     from the rest;
   - the program's hand-written bug.
   Every task passes the pool's Docker controls.
2. **Write the registry entry before anything is uploaded.** A second opening is then
   refused even if this one fails partway; a failed opening needs a human decision, not a
   retry.
3. **Run each arm once, greedy,** in its own private notebook (two at a time), using
   specifications, buggy functions, and public cases only. Each notebook first checks that
   the mounted adapter is byte-identical to the frozen one.
4. **Download.** Grading, the rules, and the audit are in ``verify_final_test_3.py``:
   first on a copy, then once in place.

Usage:
  python3 -B scripts/run_final_test_3.py --frozen artifacts/rlfirst-frozen.json \
    --preflight artifacts/rlfirst-settings-v2 --pool artifacts/rlfirst-pool-v3 \
    --output artifacts/final-test-3-v1 --token-file ~/.kaggle/access_token \
    --kaggle .venv/bin/kaggle
"""

import argparse
import hashlib
import json
import random
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_rlfirst_pool import controls  # noqa: E402
from freeze_rlfirst import ARMS, protocol  # noqa: E402
from kaggle_client import Kaggle  # noqa: E402
from run_rl_training import kernel_metadata, submit  # noqa: E402
from run_rlfirst_eval import public_only  # noqa: E402
from run_rlfirst_preflight import shipped_code  # noqa: E402
from run_rlfirst_sft import combined_runtime  # noqa: E402

from repairlab.execution import DockerExecutor  # noqa: E402
from repairlab.experiments import write_json  # noqa: E402
from repairlab.final3.programs import final3_tasks  # noqa: E402
from repairlab.rlfirst.mutate import HAND_WRITTEN  # noqa: E402
from repairlab.rlfirst.pool import (  # noqa: E402
    SEED,
    _ident,
    _key,
    admitted_singles,
    evaluate,
    make_task,
    ordered_cases,
    record,
    training_sources,
)
from repairlab.rlfirst.sandbox import LocalExecutor  # noqa: E402
from repairlab.scaleup.models import LARGE  # noqa: E402
from repairlab.tasks import canonical  # noqa: E402

REGISTRY = ROOT / "artifacts" / "final-test-3-registry.json"
SEALED_SEED = SEED + 31
MIN_QUOTA_HOURS = 2.0


def known_keys(pool):
    """Normalized token sequences of every earlier reference and bug, and of the sealed ones.

    The development sets' rows carry the second sealed test's references and bugs, so
    that test is covered without loading it here.
    """
    keys = set()
    for _, task in training_sources():
        keys |= {_key(task.reference), _key(task.buggy)}
    for task in final3_tasks():
        keys |= {_key(task.reference), _key(task.buggy)}
    for name in ("pool.json", "dev-seen.json", "dev-unseen.json"):
        for row in json.loads((pool / name).read_text())["tasks"]:
            keys |= {_key(row["reference"]), _key(row["buggy"])}
    return keys


def admit(docker, program, text, kind, passed, split="final3"):
    """A built task and its Docker control row (the pool's controls)."""
    built, info = make_task(program, _ident(program.id, text), kind, text, passed, split)
    flags = {canonical(c.args): ok for c, ok in zip(ordered_cases(program), passed)}
    return built, info, controls(docker, built, flags)


def sealed_tasks(pool, sandbox, docker):
    rng = random.Random(SEALED_SEED)
    seen = known_keys(pool)
    rows, report = [], []

    def row(built, info, control, group, description):
        return {
            **record(built),
            **info,
            "set": group,
            "description": description,
            "split": "final_test_3",
            "bug_public_result": control["bug_public_result"],
        }

    for program in final3_tasks():
        singles = []
        for single in admitted_singles(sandbox, program):
            key = _key(single["text"])
            if key not in seen:
                seen.add(key)
                singles.append(single)
        left = list(singles)
        generated = None
        while left and generated is None:
            single = rng.choice(left)
            left.remove(single)
            built, info, control = admit(
                docker, program, single["text"], single["mutation"].kind, single["passed"]
            )
            if control["admitted"]:
                generated = row(built, info, control, "generated", single["mutation"].description)
            else:
                report.append({"program": program.id, "generated_rejected": control["reason"]})
        if generated is None:
            report.append({"program": program.id, "no_generated_bug": len(singles)})
        else:
            rows.append(generated)
        passed = evaluate(sandbox, program, program.buggy)
        if passed is None or all(passed):
            report.append({"program": program.id, "hand_written_rejected": "no failing input"})
            continue
        built, info, control = admit(docker, program, program.buggy, HAND_WRITTEN, passed)
        if control["admitted"]:
            rows.append(row(built, info, control, "hand_written", "original hand-written bug"))
        else:
            report.append({"program": program.id, "hand_written_rejected": control["reason"]})
    return rows, report


def quota_hours(client, logfile):
    quota = json.loads(client.call(["quota", "--format", "json"], logfile))
    return float(next(r["remaining"] for r in quota if r["resource"] == "GPU")[:-1])


def run_arms(client, root, frozen, tasks, decision, prefix, title):
    """Each frozen arm once, greedy, in its own private notebook (two at a time).

    The notebook refuses unless the mounted adapter's hash equals the frozen one. A failed
    arm is recorded, never retried. Returns {arm: outcome}.
    """
    runtime = combined_runtime(root / "runtime-combined.py", "rlfirst_checked_eval_runtime.py")
    code = shipped_code()

    def run_arm(arm):
        spec = frozen["arms"][arm]
        payload = {
            "kind": f"rlfirst_{prefix}_eval",
            "arm": arm,
            "model": LARGE,
            "dtype": "fp16" if decision["use_fp16"] else "bf16",
            "code": code,
            "tasks": tasks,
            "batch_size": decision["batch_size"],
            "adapter_glob": spec["adapter_glob"],
            "adapter_sha256": spec["adapter_sha256"],
        }
        kernel = f"alwaysacing/repairlab-{prefix}-{arm}"
        try:
            _, version = submit(
                client,
                root / "arms" / arm,
                f"{prefix}-{arm}",
                f"# {title} ({arm})\n\nGreedy two-turn episodes with public cases only; "
                "hidden grading happens elsewhere.",
                payload,
                runtime,
                "run_checked_eval",
                "repairlab-rlfirst-eval-*",
                kernel_metadata(
                    kernel,
                    kernel.split("/")[1].replace("-", " ").title(),
                    "RepairLab_Eval.ipynb",
                    [spec["kernel"]] if spec["kernel"] else [],
                ),
            )
            return arm, {"status": "downloaded", "kernel": kernel, "version": version}
        except BaseException as exc:  # noqa: BLE001 - recorded for a human decision
            return arm, {"status": "failed", "kernel": kernel, "error": str(exc)[:500]}

    with ThreadPoolExecutor(2) as pool:
        return dict(pool.map(run_arm, ARMS))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--frozen", type=Path, required=True)
    parser.add_argument("--preflight", type=Path, required=True)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    args = parser.parse_args()
    if REGISTRY.exists():
        raise SystemExit("Sealed test 3 was already opened: " + str(REGISTRY))
    frozen = json.loads(args.frozen.read_text())
    now = protocol()
    changed = [k for k, v in now.items() if frozen.get(k) != v]
    if changed:
        raise SystemExit("Changed since the freeze: " + ", ".join(changed))
    decision = json.loads((args.preflight / "preflight-decision.json").read_text())
    client = Kaggle(args.kaggle, args.token_file)
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    if quota_hours(client, root / "quota-check.json") < MIN_QUOTA_HOURS:
        raise SystemExit(f"Less than {MIN_QUOTA_HOURS} GPU hours left; open after the reset")

    rows, report = sealed_tasks(args.pool, LocalExecutor(), DockerExecutor())
    groups = {g: sum(r["set"] == g for r in rows) for g in ("generated", "hand_written")}
    write_json(root / "sealed-tasks.json", {"tasks": rows})
    write_json(root / "admission.json", {"counts": groups, "report": report, "seed": SEALED_SEED})
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
        "counts": groups,
        "rule": "Opened once after every adapter and rule was frozen; never reopen.",
    }
    with REGISTRY.open("x") as handle:
        handle.write(json.dumps(entry, indent=2) + "\n")
    write_json(
        root / "experiment-plan.json",
        {
            "kind": "rlfirst_final_test_3",
            "frozen": frozen,
            "episodes": len(ARMS) * len(tasks),
            "decoding": "greedy, two turns, Hugging Face fp16 (as the development evaluations)",
            "no_retries_or_best_of": True,
            "checkpoint_selection_on_this_split": False,
        },
    )
    for name in ("repairlab", "scripts", "tests"):
        shutil.copytree(
            ROOT / name,
            root / "source-snapshot" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    outcomes = run_arms(
        client, root, frozen, tasks, decision, "final3", "RepairLab sealed test 3"
    )
    client.call(["quota", "--format", "json"], root / "quota-after.json")
    failed = [a for a, o in outcomes.items() if o["status"] != "downloaded"]
    status = "failed_partway_needs_human_decision" if failed else "downloaded_needs_grading"
    write_json(REGISTRY, {**entry, "status": status, "outcomes": outcomes})
    write_json(root / "status.json", {"status": status, "outcomes": outcomes})
    print("FINAL TEST 3", status, json.dumps(groups), flush=True)


if __name__ == "__main__":
    main()
