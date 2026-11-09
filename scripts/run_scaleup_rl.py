"""Run the fixed 1.7B RL cycles: sample on Kaggle, grade locally in Docker, update on Kaggle.

The start adapter is chosen by the plan's rule 1 from the saved development comparison
of the two supervised variants, not by hand. Each cycle's programs follow the plan's
program rule from the saved reports of earlier cycles. ``--resume`` keeps completed
cycles and renames an unfinished one aside. ``--cycles`` and ``--programs`` exist for
scripted dry runs; the real run uses the fixed configuration.
"""

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kaggle_client import Kaggle  # noqa: E402
from run_rl_training import kernel_metadata, submit  # noqa: E402

from repairlab.execution import DockerExecutor  # noqa: E402
from repairlab.experiments import source_fingerprint, write_json  # noqa: E402
from repairlab.rl.programs import rl_programs  # noqa: E402
from repairlab.scaleup import fingerprint  # noqa: E402
from repairlab.scaleup.freeze import chosen_variant  # noqa: E402
from repairlab.scaleup.rl import (  # noqa: E402
    CONFIG,
    PROTOCOL,
    UPDATE_PROTOCOL,
    SizedRolloutCycle,
    kernel_for_cycle,
    pass_rates,
    select_programs,
    update_payload,
)

ROLLOUT_KERNEL = "alwaysacing/repairlab-scaleup-rl-rollouts"


def run_cycle(client, root, cycle, policy, tasks, samples):
    folder = root / "cycles" / str(cycle)
    rollouts = SizedRolloutCycle(
        DockerExecutor(), folder / "rollouts", cycle, policy, tasks, samples
    )
    write_json(
        folder / "rollouts" / "experiment-plan.json",
        {
            "task_ids": [t.id for t in tasks],
            "samples_per_program": samples,
            "episodes": samples * len(tasks),
            "selection_rule": CONFIG["program_rule"],
            "success_rule": "Explicit submit within budget; all public and private checks must pass.",
            "score_meaning": "Sampled RL training reward, not an evaluation score.",
        },
    )
    while True:
        batch = rollouts.export()
        if batch is None:
            break
        if batch["round"] > 8:
            raise RuntimeError("Unexpected ninth round")
        print(
            "Cycle", cycle, "round", batch["round"], "requests", len(batch["requests"]), flush=True
        )
        extracted, _ = submit(
            client,
            folder / "rollouts" / "rounds" / str(batch["round"]),
            "scaleup-rollout",
            "# RepairLab 1.7B RL rollouts\n\nSamples one next action per unfinished RL training "
            "attempt from one saved adapter. Model output is never executed on this GPU. Only "
            "public RL-program conversations and checkpoint hashes are embedded, compressed to "
            "fit Kaggle's 1 MB notebook limit. No training occurs.",
            batch,
            "scaleup_rollout_runtime.py",
            "run_scaleup_rollout_round",
            "repairlab-scaleup-rollout-*",
            kernel_metadata(
                ROLLOUT_KERNEL,
                "RepairLab Scaleup RL Rollouts",
                "RepairLab_Scaleup_RL_Rollouts.ipynb",
                [policy["kernel_source"]],
            ),
        )
        metadata = json.loads((extracted / "model-config.json").read_text())
        outputs = [
            json.loads(line) for line in (extracted / "responses.jsonl").read_text().splitlines()
        ]
        rollouts.apply(batch["batch_sha256"], metadata, outputs)
    result = rollouts.finish()
    turns = result.pop("training_turns")
    write_json(folder / "cycle-report.json", {k: v for k, v in result.items() if k != "rows"})
    print("Cycle", cycle, "mean reward", result["mean_reward"], "turns", len(turns), flush=True)
    payload = update_payload(cycle, policy, turns)
    write_json(folder / "update-payload.json", payload)
    if not payload["turns"]:
        write_json(folder / "cycle-adapter.json", {**policy, "updated": False})
        return policy
    kernel = kernel_for_cycle(cycle)
    extracted, version = submit(
        client,
        folder / "update",
        "scaleup-update",
        "# RepairLab 1.7B RL update\n\nOne group-relative policy update from graded RL-program "
        "attempts: sampled token ids, prompts, and advantages only, compressed to fit Kaggle's "
        "1 MB notebook limit. No credentials, paid inference, or execution of model-generated "
        "code.",
        payload,
        "scaleup_update_runtime.py",
        "run_scaleup_update",
        "repairlab-scaleup-update-*",
        kernel_metadata(
            kernel,
            "RepairLab Scaleup RL Cycle " + str(cycle),
            "RepairLab_Scaleup_RL_Update.ipynb",
            [policy["kernel_source"]],
        ),
    )
    evidence = json.loads((extracted / "weight-evidence.json").read_text())
    if evidence["initial_adapter_sha256"] != policy["tensor_sha256"]:
        raise RuntimeError("Update did not start from the sampled policy")
    updated = {
        "kernel_source": kernel,
        "upstream_version": version,
        "file_sha256": hashlib.sha256(
            (extracted / "adapter/adapter_model.safetensors").read_bytes()
        ).hexdigest(),
        "tensor_sha256": evidence["final_adapter_sha256"],
        "base_sha256": evidence["base_final_sha256"],
    }
    write_json(folder / "cycle-adapter.json", {**updated, "updated": True})
    return updated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    parser.add_argument("--plain-run", type=Path, required=True)
    parser.add_argument("--notes-run", type=Path, required=True)
    parser.add_argument("--choice-run", type=Path, required=True)
    parser.add_argument("--cycles", type=int, default=CONFIG["cycles"])
    parser.add_argument("--programs", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if CONFIG["max_sequence_tokens"] is None or CONFIG["sampling_square_cap"] is None:
        raise SystemExit("The preflight rule has not set the RL memory limits")
    client = Kaggle(args.kaggle, args.token_file)
    dry_run = args.programs is not None or args.cycles != CONFIG["cycles"]
    variant, counts = chosen_variant(args.choice_run, allow_partial=dry_run)
    start_run = {"plain": args.plain_run, "notes": args.notes_run}[variant]
    start = json.loads((start_run / "scaleup-adapter.json").read_text())
    tasks = rl_programs()[: args.programs]
    samples = CONFIG["samples_per_program"]
    root = args.output
    plan = {
        "protocol": PROTOCOL,
        "update_protocol": UPDATE_PROTOCOL,
        "config": CONFIG,
        "cycles": args.cycles,
        "uses_fixed_configuration": args.cycles == CONFIG["cycles"]
        and len(tasks) == len(rl_programs()),
        "start_variant": variant,
        "start_rule": "docs/scaleup-plan-20261105.md, rule 1",
        "development_successes": counts,
        "start": start,
        "task_ids": [t.id for t in tasks],
        "samples_per_program": samples,
        "rollout_kernel": ROLLOUT_KERNEL,
        "cycle_kernels": [kernel_for_cycle(c) for c in range(1, args.cycles + 1)],
        "split": "rl_training",
        "development_or_final_tasks_used": 0,
        "source_sha256": source_fingerprint(),
        "scaleup_sha256": fingerprint(),
    }
    if args.resume:
        if json.loads((root / "experiment-plan.json").read_text()) != plan:
            raise RuntimeError("Resumed run differs from its saved plan")
        attempt = 1
        while (root / f"source-snapshot-resume-{attempt}").exists():
            attempt += 1
        snapshot = root / f"source-snapshot-resume-{attempt}"
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "experiment-plan.json", plan)
        snapshot = root / "source-snapshot"
    for name in ("repairlab", "scripts", "tests"):
        shutil.copytree(
            ROOT / name, snapshot / name, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
    if not args.resume:
        client.require_gpu_quota(root / "quota-before.json")
    write_json(root / "status.json", {"status": "running"})
    policy, history, cycle = start, [], 0
    try:
        for cycle in range(1, args.cycles + 1):
            folder = root / "cycles" / str(cycle)
            if (folder / "cycle-adapter.json").exists():
                saved = json.loads((folder / "cycle-adapter.json").read_text())
                policy = {k: v for k, v in saved.items() if k != "updated"}
                report = json.loads((folder / "cycle-report.json").read_text())
                history.append(pass_rates(report["episodes"]))
                continue
            if folder.exists():
                attempt = 1
                while (root / "cycles" / f"{cycle}-abandoned-{attempt}").exists():
                    attempt += 1
                folder.rename(root / "cycles" / f"{cycle}-abandoned-{attempt}")
            chosen = select_programs(tasks, history)
            policy = run_cycle(client, root, cycle, policy, chosen, samples)
            report = json.loads((folder / "cycle-report.json").read_text())
            history.append(pass_rates(report["episodes"]))
        write_json(root / "rl-adapter.json", policy)
        client.call(["quota", "--format", "json"], root / "quota-after.json")
        write_json(root / "status.json", {"status": "completed", "cycles": args.cycles})
        print("SCALEUP RL COMPLETE", policy["tensor_sha256"], flush=True)
    except BaseException as exc:
        write_json(
            root / "status.json", {"status": "needs_diagnosis", "error": str(exc), "cycle": cycle}
        )
        raise


if __name__ == "__main__":
    main()
