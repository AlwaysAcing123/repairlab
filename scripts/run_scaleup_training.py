"""Train one 1.7B supervised variant, plain or notes, on a private Kaggle GPU."""

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
from repairlab.native_tools import tools  # noqa: E402
from repairlab.scaleup import fingerprint  # noqa: E402
from repairlab.scaleup.supervised import CONFIG, PROTOCOL, VARIANTS, bundle  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    parser.add_argument("--variant", choices=VARIANTS, required=True)
    args = parser.parse_args()
    client = Kaggle(args.kaggle, args.token_file)
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    data = bundle(DockerExecutor(), args.variant)
    write_json(root / "bundle.json", data)
    write_json(
        root / "experiment-plan.json",
        {
            "protocol": PROTOCOL,
            "variant": args.variant,
            "bundle_sha256": data["bundle_sha256"],
            "config": CONFIG,
            "start": "new LoRA adapter on the untrained base model",
            "training_samples": len(data["train"]),
            "training_programs": len({s["program_id"] for s in data["train"]}),
            "checkpoint_rule": "Final step fixed before training; intermediate checkpoints are "
            "recovery evidence only.",
            "new_training": True,
            "reinforcement_learning": False,
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
    kernel = "alwaysacing/repairlab-scaleup-sft-" + args.variant
    client.require_gpu_quota(root / "quota-before.json")
    try:
        extracted, version = submit(
            client,
            root / "run",
            "sft",
            "# RepairLab Qwen3-1.7B supervised training\n\n"
            "Trains a new LoRA adapter on next-action repair demonstrations from authored "
            "training programs, variant `" + args.variant + "`. Only assistant completions are "
            "supervised. Wrong repairs appear only in history. No RL, paid inference, account "
            "credentials, or execution of model-generated code. The public training data is "
            "compressed to fit Kaggle's 1 MB notebook limit; the run saves it as bundle.json.",
            {"kind": "scaleup_sft", "bundle": data, "tools": tools()},
            "scaleup_training_runtime.py",
            "run_scaleup_training",
            "repairlab-scaleup-sft-*",
            kernel_metadata(
                kernel,
                "RepairLab Scaleup SFT " + args.variant.capitalize(),
                "RepairLab_Scaleup_SFT.ipynb",
                [],
            ),
        )
        evidence = json.loads((extracted / "weight-evidence.json").read_text())
        write_json(
            root / "scaleup-adapter.json",
            {
                "kernel_source": kernel,
                "upstream_version": version,
                "file_sha256": hashlib.sha256(
                    (extracted / "adapter/adapter_model.safetensors").read_bytes()
                ).hexdigest(),
                "tensor_sha256": evidence["final_adapter_sha256"],
                "base_sha256": evidence["base_final_sha256"],
            },
        )
        client.call(["quota", "--format", "json"], root / "quota-after.json")
        write_json(root / "status.json", {"status": "completed"})
        print("SCALEUP TRAINING COMPLETE", evidence["final_adapter_sha256"], flush=True)
    except BaseException as exc:
        write_json(root / "status.json", {"status": "needs_diagnosis", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
