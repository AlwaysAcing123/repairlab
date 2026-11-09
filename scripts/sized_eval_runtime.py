"""Standalone greedy inference client for arms on more than one base model.

Model output is never executed here. Each base model is loaded once and fingerprinted
around the round. Its adapters attach to it by exact file hash under the arm's name;
an untrained arm runs that base with every adapter disabled. PEFT refuses module names
containing dots, such as `0.6b-rl`, so adapters are named by arm position. Requests are
decoded one at a time, unpadded, as in every earlier comparison.
"""


def run_sized_round(batch):
    import hashlib
    import importlib.metadata
    import json
    import random
    import shutil
    import time
    import zipfile
    from contextlib import nullcontext
    from datetime import datetime, timezone
    from pathlib import Path

    import torch
    from peft import PeftModel, get_peft_model_state_dict
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.chat_template_utils import qwen3_schema

    output = Path(
        "repairlab-sized-round-"
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

    def adapter_hash(model, name):
        values = get_peft_model_state_dict(model, adapter_name=adapter_names[name])
        digest = hashlib.sha256()
        for key in sorted(values):
            digest.update(key.encode() + b"\0")
            digest.update(
                values[key].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            )
        return digest.hexdigest()

    def locate(file_sha256, name):
        for path in sorted(Path("/kaggle/input").rglob("adapter_model.safetensors")):
            if hashlib.sha256(path.read_bytes()).hexdigest() == file_sha256:
                return path.parent
        directory = Path("/kaggle/working/loaded-" + name)
        directory.mkdir(exist_ok=True)
        for path in Path("/kaggle/input").rglob("*.zip"):
            with zipfile.ZipFile(path) as archive:
                if "adapter/adapter_model.safetensors" not in archive.namelist():
                    continue
                values = archive.read("adapter/adapter_model.safetensors")
                if hashlib.sha256(values).hexdigest() != file_sha256:
                    continue
                (directory / "adapter_model.safetensors").write_bytes(values)
                (directory / "adapter_config.json").write_bytes(
                    archive.read("adapter/adapter_config.json")
                )
                return directory
        raise RuntimeError("Exact saved adapter not found: " + name)

    random.seed(batch["seed"])
    torch.manual_seed(batch["seed"])
    torch.cuda.manual_seed_all(batch["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    tokenizers, templates, bases, parameters, wrapped = {}, set(), {}, {}, {}
    model_evidence, arm_evidence = {}, {}
    adapter_names = {name: "arm" + str(i) for i, name in enumerate(batch["arms"])}
    for key, spec in batch["models"].items():
        tokenizer = AutoTokenizer.from_pretrained(
            spec["model"], revision=spec["model_revision"], trust_remote_code=False
        )
        tokenizer.response_schema = qwen3_schema
        tokenizers[key] = tokenizer
        templates.add(tokenizer.chat_template)
        base = (
            AutoModelForCausalLM.from_pretrained(
                spec["model"], revision=spec["model_revision"], dtype=dtype, trust_remote_code=False
            )
            .to("cuda:0")
            .eval()
            .requires_grad_(False)
        )
        parameters[key] = dict(base.named_parameters())
        model_evidence[key] = {
            "model": spec["model"],
            "model_revision": spec["model_revision"],
            "base_initial_sha256": parameter_hash(parameters[key]),
        }
        assert model_evidence[key]["base_initial_sha256"] == spec["base_sha256"]
        bases[key] = base
    assert len(templates) == 1, "Every base model must share one chat template"
    template = templates.pop()
    (output / "chat_template.jinja").write_text(template)
    for name, arm in batch["arms"].items():
        key, adapter = arm["model_key"], arm["adapter"]
        arm_evidence[name] = {"model_key": key, "adapter_file_sha256": None}
        if adapter is None:
            continue
        directory = locate(adapter["file_sha256"], name)
        actual = hashlib.sha256((directory / "adapter_model.safetensors").read_bytes()).hexdigest()
        assert actual == adapter["file_sha256"]
        config = json.loads((directory / "adapter_config.json").read_text())
        assert config["revision"] == batch["models"][key]["model_revision"]
        if key in wrapped:
            # PEFT 0.18 creates a later adapter in the base's bfloat16, loads the saved
            # float32 weights into it, and only then casts it to float32, rounding them.
            # Loading again into the float32 copy restores the exact saved weights.
            for _ in range(2):
                wrapped[key].load_adapter(
                    directory, adapter_name=adapter_names[name], is_trainable=False
                )
        else:
            wrapped[key] = PeftModel.from_pretrained(
                bases[key], directory, adapter_name=adapter_names[name], is_trainable=False
            )
        wrapped[key].to("cuda:0").eval().requires_grad_(False)
        arm_evidence[name].update(adapter_file_sha256=actual)
    for name, arm in batch["arms"].items():
        if arm["adapter"] is not None:
            digest = adapter_hash(wrapped[arm["model_key"]], name)
            arm_evidence[name]["adapter_initial_sha256"] = digest
            assert digest == arm["adapter"]["tensor_sha256"]

    def runner(name):
        """The module that generates for an arm, and the adapter context it needs."""
        key, adapter = batch["arms"][name]["model_key"], batch["arms"][name]["adapter"]
        if key not in wrapped:
            return bases[key], nullcontext()
        if adapter is None:
            return wrapped[key], wrapped[key].disable_adapter()
        wrapped[key].set_adapter(adapter_names[name])
        return wrapped[key], nullcontext()

    metadata = {
        "protocol": batch["protocol"],
        "round": batch["round"],
        "batch_sha256": batch["batch_sha256"],
        "budget": batch["budget"],
        "seed": batch["seed"],
        "gpu": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "decoding": batch["decoding"],
        "models": model_evidence,
        "arms": arm_evidence,
        "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
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
    print("Models loaded; round", batch["round"], "requests", len(batch["requests"]), flush=True)
    started = time.monotonic()
    with (output / "responses.jsonl").open("w") as log:
        for work in batch["requests"]:
            row = {
                "episode_id": work["episode_id"],
                "condition": work["condition"],
                "decision": work["decision"],
            }
            tokenizer = tokenizers[batch["arms"][work["condition"]]["model_key"]]
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
                    model, context = runner(work["condition"])
                    with context, torch.inference_mode():
                        generated = model.generate(
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
    for key in batch["models"]:
        model_evidence[key]["base_final_sha256"] = parameter_hash(parameters[key])
        assert model_evidence[key]["base_final_sha256"] == batch["models"][key]["base_sha256"]
    for name, arm in batch["arms"].items():
        if arm["adapter"] is not None:
            digest = adapter_hash(wrapped[arm["model_key"]], name)
            arm_evidence[name]["adapter_final_sha256"] = digest
            assert digest == arm["adapter"]["tensor_sha256"]
    metadata.update(
        elapsed_seconds=time.monotonic() - started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(0),
    )
    save("model-config.json", metadata)
    archive = shutil.make_archive(str(output), "zip", output)
    print("SIZED ROUND COMPLETE", batch["round"], "Archive:", archive, flush=True)
