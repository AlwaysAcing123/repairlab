"""Standalone inference-only client; loads the saved adapter from a private kernel source."""


def run_native_round(batch):
    import hashlib
    import importlib.metadata
    import json
    import random
    import shutil
    import time
    import zipfile
    from datetime import datetime, timezone
    from pathlib import Path

    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.chat_template_utils import qwen3_schema

    output = Path(
        "repairlab-native-pilot-round-"
        + str(batch["round"])
        + "-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(exist_ok=False)

    def save(name, value):
        (output / name).write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")

    def parameter_hash(values):
        digest = hashlib.sha256()
        for name, value in values.items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def adapter_hash(model):
        values = get_peft_model_state_dict(model)
        digest = hashlib.sha256()
        for name in sorted(values):
            digest.update(name.encode() + b"\0")
            digest.update(
                values[name].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            )
        return digest.hexdigest()

    expected = batch["adapter"]
    candidates = []
    for path in Path("/kaggle/input").rglob("adapter_model.safetensors"):
        if hashlib.sha256(path.read_bytes()).hexdigest() == expected["file_sha256"]:
            candidates.append(path.parent)
    if candidates:
        adapter_directory = sorted(candidates)[0]
    else:
        adapter_directory = Path("/kaggle/working/loaded-frozen-adapter")
        adapter_directory.mkdir(exist_ok=True)
        found = False
        for path in Path("/kaggle/input").rglob("*.zip"):
            with zipfile.ZipFile(path) as archive:
                if "adapter/adapter_model.safetensors" not in archive.namelist():
                    continue
                values = archive.read("adapter/adapter_model.safetensors")
                if hashlib.sha256(values).hexdigest() != expected["file_sha256"]:
                    continue
                (adapter_directory / "adapter_model.safetensors").write_bytes(values)
                (adapter_directory / "adapter_config.json").write_bytes(
                    archive.read("adapter/adapter_config.json")
                )
                found = True
                break
        if not found:
            raise RuntimeError(
                "Exact saved adapter not found in the attached private kernel outputs"
            )
    actual_file_hash = hashlib.sha256(
        (adapter_directory / "adapter_model.safetensors").read_bytes()
    ).hexdigest()
    assert actual_file_hash == expected["file_sha256"]
    assert (
        json.loads((adapter_directory / "adapter_config.json").read_text())["revision"]
        == batch["model_revision"]
    )
    random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(
        batch["model"], revision=batch["model_revision"], trust_remote_code=False
    )
    tokenizer.response_schema = qwen3_schema
    (output / "chat_template.jinja").write_text(tokenizer.chat_template)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

    def load_base():
        return (
            AutoModelForCausalLM.from_pretrained(
                batch["model"],
                revision=batch["model_revision"],
                dtype=dtype,
                trust_remote_code=False,
            )
            .to("cuda:0")
            .eval()
            .requires_grad_(False)
        )

    original = load_base()
    adapted_base = load_base()
    original_parameters = dict(original.named_parameters())
    adapted_parameters = dict(adapted_base.named_parameters())
    original_hash, adapted_hash = (
        parameter_hash(original_parameters),
        parameter_hash(adapted_parameters),
    )
    assert original_hash == adapted_hash == expected["base_sha256"]
    supervised = (
        PeftModel.from_pretrained(adapted_base, adapter_directory, is_trainable=False)
        .to("cuda:0")
        .eval()
        .requires_grad_(False)
    )
    initial_adapter_hash = adapter_hash(supervised)
    assert initial_adapter_hash == expected["tensor_sha256"]
    models = {"original": original, "supervised": supervised}
    metadata = {
        "protocol": batch["protocol"],
        "round": batch["round"],
        "batch_sha256": batch["batch_sha256"],
        "model": batch["model"],
        "model_revision": batch["model_revision"],
        "budget": batch["budget"],
        "seed": 7,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "decoding": {"do_sample": False, "num_beams": 1, "enable_thinking": False},
        "original_initial_sha256": original_hash,
        "adapted_base_initial_sha256": adapted_hash,
        "adapter_file_sha256": actual_file_hash,
        "adapter_initial_sha256": initial_adapter_hash,
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "transformers",
                "peft",
                "trl",
                "safetensors",
                "tokenizers",
                "huggingface-hub",
            )
        },
    }
    save("model-config.json", metadata)
    print(
        "Original and saved-adapter models loaded; round",
        batch["round"],
        "requests",
        len(batch["requests"]),
        flush=True,
    )
    started = time.monotonic()
    with (output / "responses.jsonl").open("w") as log:
        for work in batch["requests"]:
            row = {
                "episode_id": work["episode_id"],
                "condition": work["condition"],
                "decision": work["decision"],
            }
            begin = time.monotonic()
            try:
                inputs = tokenizer.apply_chat_template(
                    work["messages"],
                    tools=batch["tools"],
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                    add_generation_prompt=True,
                    enable_thinking=False,
                ).to("cuda:0")
                length = int(inputs["input_ids"].shape[-1])
                allowance = work["max_new_tokens"]
                row["input_tokens"] = length
                if allowance <= 0:
                    row.update(stop="output_token_budget", output_tokens=0)
                elif length + allowance > work["max_context_tokens"]:
                    row.update(stop="context_budget", output_tokens=0)
                else:
                    with torch.inference_mode():
                        generated = models[work["condition"]].generate(
                            **inputs,
                            do_sample=False,
                            num_beams=1,
                            max_new_tokens=allowance,
                            use_cache=True,
                            pad_token_id=tokenizer.eos_token_id,
                        )
                    torch.cuda.synchronize()
                    completion = generated[0, length:]
                    raw = tokenizer.decode(completion, skip_special_tokens=False)
                    row.update(raw=raw, output_tokens=int(completion.shape[0]))
                    try:
                        row["parsed"] = tokenizer.parse_response(raw)
                    except Exception as exc:
                        row.update(
                            parsed=None, parse_error=type(exc).__name__ + ": " + str(exc)[:500]
                        )
            except Exception as exc:
                row["generation_error"] = type(exc).__name__ + ": " + str(exc)[:500]
            row["elapsed_seconds"] = time.monotonic() - begin
            log.write(json.dumps(row, ensure_ascii=False) + "\n")
            log.flush()
            print(work["episode_id"], "decision", work["decision"], "saved", flush=True)
            if row.get("generation_error"):
                raise RuntimeError("Model generation failed; inspect saved output")
    metadata.update(
        original_final_sha256=parameter_hash(original_parameters),
        adapted_base_final_sha256=parameter_hash(adapted_parameters),
        adapter_final_sha256=adapter_hash(supervised),
        elapsed_seconds=time.monotonic() - started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(0),
    )
    assert (
        metadata["original_final_sha256"]
        == metadata["adapted_base_final_sha256"]
        == expected["base_sha256"]
    )
    assert metadata["adapter_final_sha256"] == expected["tensor_sha256"]
    save("model-config.json", metadata)
    archive = shutil.make_archive(str(output), "zip", output)
    print("NATIVE REPAIR ROUND COMPLETE", batch["round"], "Archive:", archive, flush=True)
