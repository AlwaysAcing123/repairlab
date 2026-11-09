"""Train the recovery adapter from the frozen first adapter on a private Kaggle GPU."""

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

from repairlab.curriculum import fingerprint  # noqa: E402
from repairlab.curriculum.recovery import CONFIG, PROTOCOL, WARM_START, bundle  # noqa: E402
from repairlab.execution import DockerExecutor  # noqa: E402
from repairlab.experiments import source_fingerprint, write_json  # noqa: E402
from repairlab.native_tools import tools  # noqa: E402

KERNEL = "alwaysacing/repairlab-native-recovery-training"
ADAPTER = (
    "artifacts/native-supervised-warmup-20261030-v1/extracted/adapter/adapter_model.safetensors"
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-file", type=Path, required=True)
    parser.add_argument("--kaggle", type=Path, required=True)
    parser.add_argument("--kernel", default=KERNEL)
    args = parser.parse_args()
    if hashlib.sha256((ROOT / ADAPTER).read_bytes()).hexdigest() != WARM_START["file_sha256"]:
        raise RuntimeError("Local saved checkpoint differs from the frozen warm-start adapter")
    client = Kaggle(args.kaggle, args.token_file)
    root = args.output
    root.mkdir(parents=True, exist_ok=False)
    data = bundle(DockerExecutor())
    write_json(root / "bundle.json", data)
    write_json(
        root / "experiment-plan.json",
        {
            "protocol": PROTOCOL,
            "bundle_sha256": data["bundle_sha256"],
            "config": CONFIG,
            "warm_start": WARM_START,
            "training_samples": len(data["train"]),
            "training_programs": len({s["program_id"] for s in data["train"]}),
            "checkpoint_rule": "Final step fixed before training; intermediate checkpoints are recovery evidence only.",
            "new_training": True,
            "reinforcement_learning": False,
            "source_sha256": source_fingerprint(),
            "curriculum_sha256": fingerprint(),
        },
    )
    for name in ("repairlab", "scripts", "tests"):
        shutil.copytree(
            ROOT / name,
            root / "source-snapshot" / name,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
        )
    upload = root / "upload"
    upload.mkdir()
    payload = {"bundle": data, "tools": tools()}
    if client.token in json.dumps(payload):
        raise RuntimeError("Credential would enter the notebook")
    notebook = build_notebook(
        upload / "RepairLab_Recovery_Training.ipynb",
        "recovery",
        "# RepairLab supervised recovery training\n\n"
        "Continues the frozen first adapter on next-action recovery demonstrations from new "
        "training programs: after a failed public test, revise, retest, then submit. Only assistant "
        "completions are supervised. Wrong repairs appear only in history. No RL, paid inference, "
        "account credentials, or model-generated-code execution. The training data is public "
        "and compressed to fit Kaggle's 1 MB notebook limit; the run saves it back as bundle.json.",
        payload,
        "recovery_training_runtime.py",
        "run_recovery_training",
        "repairlab-native-recovery-*",
        compress=True,
    )
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            compile(ast.parse(cell["source"]), "training-client", "exec")
    metadata = {
        "id": args.kernel,
        "title": "RepairLab Native Recovery Training",
        "code_file": "RepairLab_Recovery_Training.ipynb",
        "language": "python",
        "kernel_type": "notebook",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [WARM_START["kernel_source"]],
        "model_sources": [],
    }
    client.require_gpu_quota(root / "quota-before.json")
    try:
        extracted, archive, version = client.run(root, notebook, metadata)
        evidence = json.loads((extracted / "weight-evidence.json").read_text())
        write_json(
            root / "recovery-adapter.json",
            {
                "kernel_source": args.kernel,
                "upstream_version": version,
                "file_sha256": hashlib.sha256(
                    (extracted / "adapter/adapter_model.safetensors").read_bytes()
                ).hexdigest(),
                "tensor_sha256": evidence["final_adapter_sha256"],
                "base_sha256": evidence["base_final_sha256"],
            },
        )
        write_json(
            root / "run-verification.json",
            {
                "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "server_private": True,
                "server_sources_match": True,
                "worker_complete": True,
                "credentials_uploaded": False,
            },
        )
        client.call(["quota", "--format", "json"], root / "quota-after.json")
        write_json(root / "status.json", {"status": "completed"})
        print("RECOVERY TRAINING COMPLETE", evidence["final_adapter_sha256"], flush=True)
    except BaseException as exc:
        write_json(root / "status.json", {"status": "needs_diagnosis", "error": str(exc)})
        raise


if __name__ == "__main__":
    main()
