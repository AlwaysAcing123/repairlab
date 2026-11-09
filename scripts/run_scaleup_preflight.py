"""Measure Qwen3-1.7B memory and speed on a private Kaggle T4, then apply the plan's rule.

No task is attempted. The payload holds only the longest supervised training samples
of each variant; RL and sampling sizes use their tokens repeated to length.
"""

import argparse
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
from repairlab.native_tools import tools  # noqa: E402
from repairlab.scaleup import fingerprint  # noqa: E402
from repairlab.scaleup.models import LARGE  # noqa: E402
from repairlab.scaleup.supervised import CONFIG, VARIANTS, variant_samples  # noqa: E402
from repairlab.tasks import canonical  # noqa: E402

KERNEL = "alwaysacing/repairlab-scaleup-preflight"
LIMIT_GIB = 13.0
RL_LENGTHS = [4096, 6144, 8192]
SQUARE_CAPS = [60_000_000, 40_000_000, 25_000_000]


def decide(results):
    """The plan's preflight rule, applied to saved measurements only."""
    rows = results["measurements"]

    def fits(row):
        return row["ok"] and row["peak_gib"] < LIMIT_GIB

    supervised = [r for r in rows if r["name"] == "supervised_step_checkpointed"]
    supervised_fits = bool(supervised) and all(fits(r) for r in supervised)
    rl = [r["tokens"] for r in rows if r["name"] == "rl_step_checkpointed" and fits(r)]
    caps = [r["square_cap"] for r in rows if r["name"] == "sampling_batch" and fits(r)]
    return {
        "limit_gib": LIMIT_GIB,
        "supervised_fits": supervised_fits,
        "rl_max_sequence_tokens": max(rl) if rl else None,
        "sampling_square_cap": max(caps) if caps else None,
        "gradient_checkpointing": True,
        "base_sha256": results["base_sha256"],
        "proceed": supervised_fits and bool(rl) and bool(caps),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    args = parser.parse_args()
    client = Kaggle(args.kaggle, args.token_file)
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    executor = DockerExecutor()
    samples = {}
    for variant in VARIANTS:
        ranked = sorted(
            variant_samples(executor, variant),
            key=lambda s: len(canonical(s["prompt"] + s["completion"])),
        )
        samples[variant] = ranked[-2:]
    payload = {
        "kind": "preflight",
        "model": LARGE,
        "config": CONFIG,
        "tools": tools(),
        "samples": samples,
        "rl_lengths": RL_LENGTHS,
        "square_caps": SQUARE_CAPS,
        "long_prompt": 8000,
        "greedy_prompt": 2000,
    }
    write_json(
        root / "experiment-plan.json",
        {
            "kind": "memory_and_speed_preflight_not_a_result",
            "payload": payload,
            "limit_gib": LIMIT_GIB,
            "rule": "docs/scaleup-plan-20261105.md, Preflight rule",
            "source_sha256": source_fingerprint(),
            "scaleup_sha256": fingerprint(),
        },
    )
    for name in ("repairlab", "scripts", "tests"):
        shutil.copytree(
            ROOT / name,
            root / "source-snapshot" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    client.require_gpu_quota(root / "quota-before.json")
    try:
        extracted, _ = submit(
            client,
            root / "run",
            "preflight",
            "# RepairLab Qwen3-1.7B preflight\n\nMeasures GPU memory and time for the largest "
            "training, RL, and sampling steps the plan allows. No task is attempted and no "
            "model is saved. The payload holds only public training samples.",
            payload,
            "scaleup_preflight_runtime.py",
            "run_preflight",
            "repairlab-preflight-*",
            kernel_metadata(KERNEL, "RepairLab Scaleup Preflight", "RepairLab_Preflight.ipynb", []),
        )
        results = json.loads((extracted / "preflight.json").read_text())
        decision = decide(results)
        write_json(root / "preflight-decision.json", decision)
        client.call(["quota", "--format", "json"], root / "quota-after.json")
        write_json(root / "status.json", {"status": "completed", "proceed": decision["proceed"]})
        print("PREFLIGHT", json.dumps(decision), flush=True)
    except BaseException as exc:
        write_json(root / "status.json", {"status": "needs_diagnosis", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
