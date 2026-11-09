#!/usr/bin/env python3
"""Stand-in for the Kaggle CLI so runners can be exercised end to end without a GPU.

Every response it emits is scripted from authored references. Output from this file
is a plumbing check and must never be saved or reported as a model result.
"""

import ast
import base64
import gzip
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repairlab.rl.programs import rl_programs  # noqa: E402
from repairlab.tasks import pilot_tasks  # noqa: E402

STATE = Path(os.environ["FAKE_KAGGLE_STATE"])
TEMPLATE = "fake chat template"
ENVIRONMENT = {
    "seed": 7,
    "gpu": "Tesla T4",
    "dtype": "torch.bfloat16",
    "decoding": {"do_sample": False, "num_beams": 1, "enable_thinking": False},
    "packages": {
        "torch": "2.11.0+cu128",
        "transformers": "5.2.0",
        "peft": "0.18.1",
        "trl": "1.8.0",
        "safetensors": "0.8.0",
        "tokenizers": "0.22.2",
        "huggingface-hub": "1.29.0",
    },
}


def chosen_action(request):
    """Scripted arms: `original` keeps calling test wrongly; adapters follow the reference.

    RL samples alternate: odd sample arms repair, even ones submit the bug, so groups
    mix rewards. Tasks without an authored reference here are read and submitted.
    """
    known = {t.id: t for t in pilot_tasks() + rl_programs()}
    task = known.get(request["episode_id"].split("--")[0])
    arm, decision = request["condition"], request["decision"]
    if task is None or (arm[1:].isdigit() and int(arm[1:]) % 2 == 0):
        return [("read", {}), ("submit", {})][decision - 1]
    seeded = decision == 1 and any(m["role"] == "tool" for m in request["messages"])
    if not seeded and decision > 1:
        seeded = sum(m["role"] == "tool" for m in request["messages"]) >= decision + 2
    if arm == "original":
        return "test", {"public_tests": []}
    if arm == "supervised" and seeded:
        return "submit", {}
    plan = [("read", {}), ("edit", {"source": task.reference}), ("test", {}), ("submit", {})]
    return (plan[1:] if seeded else plan)[decision - 1]


def response(request, sampled=False):
    tool, arguments = chosen_action(request)
    extra = {"completion_ids": list(range(40)), "prompt_sha256": "0" * 64} if sampled else {}
    return {
        **extra,
        "episode_id": request["episode_id"],
        "condition": request["condition"],
        "decision": request["decision"],
        "raw": "<tool_call>\n"
        + json.dumps({"name": tool, "arguments": arguments})
        + "\n</tool_call><|im_end|>",
        "input_tokens": 700,
        "output_tokens": 40,
        "parsed": {
            "role": "assistant",
            "content": "",
            "tool_calls": [
                {"type": "function", "function": {"name": tool, "arguments": arguments}}
            ],
        },
        "elapsed_seconds": 0.01,
    }


def round_metadata(batch):
    from hashlib import sha256

    metadata = {
        **ENVIRONMENT,
        "protocol": batch["protocol"],
        "round": batch["round"],
        "batch_sha256": batch["batch_sha256"],
        "model": batch["model"],
        "model_revision": batch["model_revision"],
        "budget": batch["budget"],
        "chat_template_sha256": sha256(TEMPLATE.encode()).hexdigest(),
        "elapsed_seconds": 1.0,
        "peak_gpu_bytes": 1,
    }
    if batch.get("kind") == "rl_rollout":
        metadata.update(decoding=batch["sampling"], seed=batch["seed"])
    if "arms" in batch:
        metadata["arms"] = {}
        for name, adapter in batch["arms"].items():
            evidence = {
                "base_initial_sha256": batch["base_sha256"],
                "base_final_sha256": batch["base_sha256"],
            }
            if adapter:
                evidence.update(
                    adapter_file_sha256=adapter["file_sha256"],
                    adapter_initial_sha256=adapter["tensor_sha256"],
                    adapter_final_sha256=adapter["tensor_sha256"],
                )
            metadata["arms"][name] = evidence
    else:
        adapter = batch["adapter"]
        for name in ("original", "adapted_base"):
            metadata[name + "_initial_sha256"] = adapter["base_sha256"]
            metadata[name + "_final_sha256"] = adapter["base_sha256"]
        metadata.update(
            adapter_file_sha256=adapter["file_sha256"],
            adapter_initial_sha256=adapter["tensor_sha256"],
            adapter_final_sha256=adapter["tensor_sha256"],
        )
    return metadata


def embedded_payload(notebook):
    compressed = "import base64,gzip,json\nPAYLOAD=json.loads(gzip.decompress(base64.b64decode("
    for cell in notebook["cells"]:
        if cell["source"].startswith(compressed):
            literal = cell["source"][len(compressed) : cell["source"].index(")")]
            return json.loads(gzip.decompress(base64.b64decode(ast.literal_eval(literal))))
        for name in ("BATCH", "PAYLOAD"):
            prefix = "import json\n" + name + "=json.loads("
            if cell["source"].startswith(prefix):
                literal = cell["source"][len(prefix) : cell["source"].rindex(")")]
                return json.loads(ast.literal_eval(literal))
    raise SystemExit("No embedded payload")


def sized_metadata(batch):
    from hashlib import sha256

    metadata = {
        **ENVIRONMENT,
        "decoding": batch["decoding"],
        "seed": batch["seed"],
        "protocol": batch["protocol"],
        "round": batch["round"],
        "batch_sha256": batch["batch_sha256"],
        "budget": batch["budget"],
        "chat_template_sha256": sha256(TEMPLATE.encode()).hexdigest(),
        "models": {
            key: {
                "model": spec["model"],
                "model_revision": spec["model_revision"],
                "base_initial_sha256": spec["base_sha256"],
                "base_final_sha256": spec["base_sha256"],
            }
            for key, spec in batch["models"].items()
        },
        "arms": {},
        "elapsed_seconds": 1.0,
        "peak_gpu_bytes": 1,
    }
    for name, arm in batch["arms"].items():
        adapter = arm["adapter"]
        metadata["arms"][name] = {
            "model_key": arm["model_key"],
            "adapter_file_sha256": adapter["file_sha256"] if adapter else None,
        }
        if adapter:
            metadata["arms"][name].update(
                adapter_initial_sha256=adapter["tensor_sha256"],
                adapter_final_sha256=adapter["tensor_sha256"],
            )
    return metadata


def preflight_results(payload):
    rows = [
        {"name": "supervised_step_checkpointed", "tokens": 1700, "ok": True, "peak_gib": 8.0},
        *(
            {"name": "rl_step_checkpointed", "tokens": n, "ok": True, "peak_gib": 9.0}
            for n in payload["rl_lengths"]
        ),
        *(
            {"name": "sampling_batch", "square_cap": c, "ok": True, "peak_gib": 10.0}
            for c in payload["square_caps"]
        ),
    ]
    return {"base_sha256": "f" * 64, "measurements": rows}


def write_outputs(version, payload):
    out = STATE / f"output-{version}"
    out.mkdir(parents=True)
    if payload.get("kind") == "preflight":
        (out / "preflight.json").write_text(json.dumps(preflight_results(payload)))
    elif payload.get("kind") in ("sized_eval", "scaleup_rollout"):
        sampled = payload["kind"] == "scaleup_rollout"
        (out / "chat_template.jinja").write_text(TEMPLATE)
        (out / "model-config.json").write_text(json.dumps(sized_metadata(payload)))
        (out / "responses.jsonl").write_text(
            "".join(json.dumps(response(r, sampled)) + "\n" for r in payload["requests"])
        )
    elif payload.get("kind") == "scaleup_sft":
        from hashlib import sha256

        variant = payload["bundle"]["variant"].encode()
        (out / "adapter").mkdir()
        (out / "adapter/adapter_model.safetensors").write_bytes(b"fake 1.7b adapter " + variant)
        (out / "weight-evidence.json").write_text(
            json.dumps(
                {
                    "final_adapter_sha256": sha256(b"fake 1.7b " + variant).hexdigest(),
                    "base_final_sha256": payload["bundle"]["base_sha256"],
                }
            )
        )
    elif payload.get("kind") in ("rl_update", "scaleup_update"):
        from hashlib import sha256

        (out / "adapter").mkdir()
        (out / "adapter/adapter_model.safetensors").write_bytes(
            b"fake rl adapter " + str(payload["cycle"]).encode()
        )
        (out / "adapter/adapter_config.json").write_text("{}")
        base = payload["base_sha256"]
        evidence = {
            "optimizer_steps": payload["config"]["optimizer_steps_per_cycle"],
            "turns": len(payload["turns"]),
            "episodes": len({t["episode_id"] for t in payload["turns"]}),
            "base_initial_sha256": base,
            "base_final_sha256": base,
            "initial_adapter_sha256": payload["policy"]["tensor_sha256"],
            "final_adapter_sha256": sha256(b"fake rl %d" % payload["cycle"]).hexdigest(),
            "adapter_l2_change": 0.1,
            "prompts_matched_sampled_tokens": len(payload["turns"]),
        }
        (out / "weight-evidence.json").write_text(json.dumps(evidence))
        (out / "training-log.json").write_text("[]")
        (out / "model-config.json").write_text(
            json.dumps(
                {
                    "payload_sha256": payload["payload_sha256"],
                    "config": payload["config"],
                    "cycle": payload["cycle"],
                }
            )
        )
    elif "bundle" in payload:
        (out / "adapter").mkdir()
        (out / "adapter/adapter_model.safetensors").write_bytes(b"fake adapter " + b"0" * 32)
        (out / "weight-evidence.json").write_text(
            json.dumps(
                {
                    "final_adapter_sha256": "e" * 64,
                    "base_final_sha256": payload["bundle"]["warm_start"]["base_sha256"],
                }
            )
        )
    else:
        (out / "chat_template.jinja").write_text(TEMPLATE)
        (out / "model-config.json").write_text(json.dumps(round_metadata(payload)))
        (out / "responses.jsonl").write_text(
            "".join(
                json.dumps(response(r, payload.get("kind") == "rl_rollout")) + "\n"
                for r in payload["requests"]
            )
        )
    return out


def next_version():
    path = STATE / "version"
    version = int(path.read_text()) + 1 if path.exists() else 1
    path.write_text(str(version))
    return version


def main():
    STATE.mkdir(parents=True, exist_ok=True)
    arguments = sys.argv[1:]
    if arguments[0] == "quota":
        print(json.dumps([{"resource": "GPU", "remaining": "22.00h"}]))
        return
    command = arguments[1]
    if command == "push":
        upload = Path(arguments[arguments.index("-p") + 1])
        version = next_version()
        saved = STATE / f"pushed-{version}"
        shutil.copytree(upload, saved)
        notebook = json.loads(next(saved.glob("*.ipynb")).read_text())
        write_outputs(version, embedded_payload(notebook))
        print(f"Kernel version {version} successfully pushed.")
    elif command == "pull":
        version = arguments[2].rsplit("/", 1)[1]
        target = Path(arguments[arguments.index("-p") + 1])
        shutil.copytree(STATE / f"pushed-{version}", target)
        metadata = json.loads((target / "kernel-metadata.json").read_text())
        (target / "kernel-metadata.json").write_text(json.dumps({**metadata, "is_private": True}))
    elif command == "status":
        print(arguments[2], 'has status "KernelWorkerStatus.COMPLETE"')
    elif command == "output":
        version = arguments[2].rsplit("/", 1)[1]
        target = Path(arguments[arguments.index("-p") + 1])
        target.mkdir(parents=True, exist_ok=True)
        source = STATE / f"output-{version}"
        with zipfile.ZipFile(target / "outputs.zip", "w") as archive:
            for path in sorted(source.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(source))
    else:
        raise SystemExit("Unsupported fake command: " + command)


if __name__ == "__main__":
    main()
