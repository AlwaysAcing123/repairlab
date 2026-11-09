"""Run the fixed RL cycles: sample on Kaggle, grade locally in Docker, update on Kaggle.

Each cycle's update runs in its own private notebook, so the next cycle attaches that
exact adapter by hash. ``--resume`` keeps completed cycles and renames an unfinished
one aside; nothing saved is overwritten. ``--cycles`` and ``--programs`` exist for
scripted dry runs; the real run uses the fixed configuration.
"""

import argparse
import ast
import hashlib
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from kaggle_client import Kaggle, build_notebook  # noqa: E402

from repairlab.curriculum import fingerprint as curriculum_fingerprint  # noqa: E402
from repairlab.execution import DockerExecutor  # noqa: E402
from repairlab.experiments import source_fingerprint, write_json  # noqa: E402
from repairlab.rl import fingerprint  # noqa: E402
from repairlab.rl.programs import rl_programs  # noqa: E402
from repairlab.rl.rollouts import (  # noqa: E402
    CONFIG,
    PROTOCOL,
    START,
    UPDATE_PROTOCOL,
    RolloutCycle,
    kernel_for_cycle,
    update_payload,
)

ROLLOUT_KERNEL = "alwaysacing/repairlab-native-rl-rollouts"


def kernel_metadata(kernel, title, code_file, sources):
    return {
        "id": kernel,
        "title": title,
        "code_file": code_file,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": sources,
        "model_sources": [],
    }


def submit(client, folder, name, heading, payload, runtime, entry, pattern, metadata):
    if client.token in json.dumps(payload):
        raise RuntimeError("Credential would enter the notebook")
    upload = folder / "upload"
    upload.mkdir(parents=True, exist_ok=False)
    notebook = build_notebook(
        upload / metadata["code_file"],
        name,
        heading,
        payload,
        runtime,
        entry,
        pattern,
        compress=True,
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile(ast.parse(cell["source"]), "rl-client", "exec")
    extracted, archive, version = client.run(folder, notebook, metadata)
    write_json(
        folder / "round-verification.json",
        {
            "server_private": True,
            "server_sources_match": True,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "worker_complete": True,
            "credentials_uploaded": False,
            "kernel_version": version,
        },
    )
    return extracted, version


def run_cycle(client, root, cycle, policy, tasks, samples):
    folder = root / "cycles" / str(cycle)
    rollouts = RolloutCycle(DockerExecutor(), folder / "rollouts", cycle, policy, tasks, samples)
    write_json(
        folder / "rollouts" / "experiment-plan.json",
        {
            "task_ids": [t.id for t in tasks],
            "samples_per_program": samples,
            "episodes": samples * len(tasks),
            "selection_rule": "Every RL program, every cycle.",
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
        round_folder = folder / "rollouts" / "rounds" / str(batch["round"])
        print(
            "Cycle", cycle, "round", batch["round"], "requests", len(batch["requests"]), flush=True
        )
        extracted, _ = submit(
            client,
            round_folder,
            "rl-rollout",
            "# RepairLab RL rollouts\n\nSamples one next action per unfinished RL training "
            "attempt from one saved adapter. Model output is never executed on this GPU. Only "
            "public RL-program conversations and checkpoint hashes are embedded, compressed to "
            "fit Kaggle's 1 MB notebook limit. No training occurs.",
            batch,
            "rl_rollout_runtime.py",
            "run_rl_rollout_round",
            "repairlab-rl-rollout-*",
            kernel_metadata(
                ROLLOUT_KERNEL,
                "RepairLab Native RL Rollouts",
                "RepairLab_RL_Rollouts.ipynb",
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
    payload = update_payload(cycle, policy, policy["base_sha256"], turns)
    write_json(folder / "update-payload.json", payload)
    if not turns:
        write_json(folder / "cycle-adapter.json", {**policy, "updated": False})
        return policy
    kernel = kernel_for_cycle(cycle)
    extracted, version = submit(
        client,
        folder / "update",
        "rl-update",
        "# RepairLab RL update\n\nOne group-relative policy update from graded RL-program "
        "attempts: sampled token ids, prompts, and advantages only, compressed to fit "
        "Kaggle's 1 MB notebook limit. No credentials, paid inference, or execution of "
        "model-generated code.",
        payload,
        "rl_training_runtime.py",
        "run_rl_update",
        "repairlab-rl-update-*",
        kernel_metadata(
            kernel,
            "RepairLab Native RL Cycle " + str(cycle),
            "RepairLab_RL_Update.ipynb",
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
    parser.add_argument("--cycles", type=int, default=CONFIG["cycles"])
    parser.add_argument("--programs", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    client = Kaggle(args.kaggle, args.token_file)
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
        "start": START,
        "task_ids": [t.id for t in tasks],
        "samples_per_program": samples,
        "rollout_kernel": ROLLOUT_KERNEL,
        "cycle_kernels": [kernel_for_cycle(c) for c in range(1, args.cycles + 1)],
        "split": "rl_training",
        "development_or_final_tasks_used": 0,
        "source_sha256": source_fingerprint(),
        "curriculum_sha256": curriculum_fingerprint(),
        "rl_sha256": fingerprint(),
    }
    if args.resume:
        if json.loads((root / "experiment-plan.json").read_text()) != plan:
            raise RuntimeError("Resumed run differs from its saved plan")
        attempt = 1
        while (root / f"source-snapshot-resume-{attempt}").exists():
            attempt += 1
        for name in ("repairlab", "scripts", "tests"):
            shutil.copytree(
                ROOT / name,
                root / f"source-snapshot-resume-{attempt}" / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
    else:
        root.mkdir(parents=True, exist_ok=False)
        write_json(root / "experiment-plan.json", plan)
        for name in ("repairlab", "scripts", "tests"):
            shutil.copytree(
                ROOT / name,
                root / "source-snapshot" / name,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
            )
        client.require_gpu_quota(root / "quota-before.json")
    write_json(root / "status.json", {"status": "running"})
    policy, cycle = START, 0
    try:
        for cycle in range(1, args.cycles + 1):
            folder = root / "cycles" / str(cycle)
            if (folder / "cycle-adapter.json").exists():
                saved = json.loads((folder / "cycle-adapter.json").read_text())
                policy = {k: v for k, v in saved.items() if k != "updated"}
                continue
            if folder.exists():
                attempt = 1
                while (root / "cycles" / f"{cycle}-abandoned-{attempt}").exists():
                    attempt += 1
                folder.rename(root / "cycles" / f"{cycle}-abandoned-{attempt}")
            policy = run_cycle(client, root, cycle, policy, tasks, samples)
        write_json(root / "rl-adapter.json", policy)
        client.call(["quota", "--format", "json"], root / "quota-after.json")
        write_json(root / "status.json", {"status": "completed", "cycles": args.cycles})
        print("RL TRAINING COMPLETE", policy["tensor_sha256"], flush=True)
    except BaseException as exc:
        write_json(
            root / "status.json", {"status": "needs_diagnosis", "error": str(exc), "cycle": cycle}
        )
        raise


if __name__ == "__main__":
    main()
