"""Standalone GPU runtime for a 1.7B supervised adapter, embedded verbatim in its notebook.

Starts a new LoRA adapter on the untrained base and trains it on next-action samples
with gradient checkpointing, which recomputes activations in the backward pass instead
of storing them and leaves gradients unchanged. Micro-batch one avoids padding. No
checkpoint is selected by evaluation. No credential or RL is used.
"""


def run_scaleup_training(payload):
    import hashlib
    import importlib.metadata
    import json
    import math
    import random
    import shutil
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    import torch
    from peft import LoraConfig, get_peft_model, get_peft_model_state_dict
    from safetensors.torch import save_file
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from trl.chat_template_utils import qwen3_schema

    bundle, tools = payload["bundle"], payload["tools"]
    config = bundle["config"]
    output = Path(
        "repairlab-scaleup-sft-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(exist_ok=False)

    def save(name, value):
        (output / name).write_text(
            json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
        )

    def phase(name):
        save("status.json", {"status": "running", "phase": name})
        print("PHASE", name, flush=True)

    def parameter_hash(values):
        digest = hashlib.sha256()
        for name, value in values.items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def adapter_values(model):
        return {
            k: v.detach().cpu().contiguous().clone()
            for k, v in get_peft_model_state_dict(model).items()
        }

    def adapter_hash(values):
        digest = hashlib.sha256()
        for name in sorted(values):
            digest.update(name.encode() + b"\0")
            digest.update(
                values[name].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            )
        return digest.hexdigest()

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    save("bundle.json", bundle)
    phase("loading")
    tokenizer = AutoTokenizer.from_pretrained(
        bundle["model"], revision=bundle["model_revision"], trust_remote_code=False
    )
    tokenizer.response_schema = qwen3_schema
    (output / "chat_template.jinja").write_text(tokenizer.chat_template)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = (
        AutoModelForCausalLM.from_pretrained(
            bundle["model"],
            revision=bundle["model_revision"],
            dtype=dtype,
            trust_remote_code=False,
        )
        .to("cuda:0")
        .eval()
        .requires_grad_(False)
    )
    base_parameters = dict(base.named_parameters())
    base_initial = parameter_hash(base_parameters)
    assert base_initial == bundle["base_sha256"]
    metadata = {
        "protocol": bundle["protocol"],
        "variant": bundle["variant"],
        "model": bundle["model"],
        "model_revision": bundle["model_revision"],
        "bundle_sha256": bundle["bundle_sha256"],
        "config": config,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "attention": base.config._attn_implementation,
        "base_initial_sha256": base_initial,
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "packages": {
            p: importlib.metadata.version(p)
            for p in (
                "torch",
                "transformers",
                "trl",
                "peft",
                "safetensors",
                "tokenizers",
                "huggingface-hub",
            )
        },
    }
    save("model-config.json", metadata)
    phase("tokenization_preflight")
    train = bundle["train"]
    encoded, token_stats = [], []
    for sample in train:
        options = dict(tools=tools, tokenize=False, enable_thinking=False)
        prefix = tokenizer.apply_chat_template(
            sample["prompt"], add_generation_prompt=True, **options
        )
        full = tokenizer.apply_chat_template(
            sample["prompt"] + sample["completion"], add_generation_prompt=False, **options
        )
        if not full.startswith(prefix):
            raise RuntimeError("Training and inference template prefixes differ")
        prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
        ids = tokenizer(full, add_special_tokens=False)["input_ids"]
        if ids[: len(prefix_ids)] != prefix_ids or len(ids) <= len(prefix_ids):
            raise RuntimeError("Unstable completion-token boundary")
        if len(ids) > config["max_sequence_tokens"]:
            raise RuntimeError("Training example exceeds fixed context budget")
        parsed = tokenizer.parse_response(full[len(prefix) :])
        target = sample["completion"][0]
        if parsed.get("tool_calls") != target["tool_calls"]:
            raise RuntimeError("Teacher native action did not round-trip")
        if (parsed.get("content") or "").strip() != target["content"].strip():
            raise RuntimeError("Teacher note did not round-trip")
        encoded.append(
            {"input_ids": ids, "labels": [-100] * len(prefix_ids) + ids[len(prefix_ids) :]}
        )
        token_stats.append(
            {
                "sample_id": sample["sample_id"],
                "prefix_tokens": len(prefix_ids),
                "completion_tokens": len(ids) - len(prefix_ids),
                "total_tokens": len(ids),
            }
        )
    save("tokenization-preflight.json", token_stats)
    save(
        "tokenized-training.json",
        [{"sample_id": s["sample_id"], **row} for s, row in zip(train, encoded)],
    )
    phase("attaching_new_lora")
    model = get_peft_model(
        base,
        LoraConfig(
            revision=bundle["model_revision"],
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["target_modules"],
            task_type="CAUSAL_LM",
        ),
    ).to("cuda:0")
    initial_adapter = adapter_values(model)
    assert all(
        torch.count_nonzero(value) == 0
        for name, value in initial_adapter.items()
        if "lora_B" in name
    )
    initial_adapter_hash = adapter_hash(initial_adapter)
    save_file(initial_adapter, str(output / "initial_adapter.safetensors"))
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    assert all(not p.requires_grad for p in base_parameters.values())
    optimizer = torch.optim.AdamW(
        list(trainable.values()), lr=config["learning_rate"], weight_decay=0.0
    )
    if config["gradient_checkpointing"]:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    model.config.use_cache = False
    generator = random.Random(config["seed"])
    order = list(range(len(encoded)))
    generator.shuffle(order)
    cursor, logs, batches = 0, [], []
    torch.cuda.reset_peak_memory_stats(0)
    started = time.monotonic()
    phase("supervised_training")
    for step in range(config["optimizer_steps"]):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro in range(config["gradient_accumulation"]):
            if cursor == len(order):
                generator.shuffle(order)
                cursor = 0
            index = order[cursor]
            cursor += 1
            ids = torch.tensor([encoded[index]["input_ids"]], dtype=torch.long, device="cuda:0")
            labels = torch.tensor([encoded[index]["labels"]], dtype=torch.long, device="cuda:0")
            with torch.autocast("cuda", dtype=dtype):
                loss = model(input_ids=ids, labels=labels, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite supervised loss")
            losses.append(float(loss.detach()))
            (loss / config["gradient_accumulation"]).backward()
            batches.append(
                {
                    "optimizer_step": step + 1,
                    "micro_batch": micro + 1,
                    "sample_ids": [train[index]["sample_id"]],
                }
            )
        norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), 1.0)
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite gradient norm")
        optimizer.step()
        logs.append(
            {
                "optimizer_step": step + 1,
                "loss": sum(losses) / len(losses),
                "gradient_norm": float(norm.detach()),
                "elapsed_seconds": time.monotonic() - started,
            }
        )
        save("training-log.json", logs)
        if (step + 1) % 10 == 0:
            print("Optimizer step", step + 1, "loss", logs[-1]["loss"], flush=True)
        if step + 1 in config["checkpoint_steps"]:
            model.save_pretrained(output / f"checkpoint-{step + 1}", safe_serialization=True)
    save("training-batches.json", batches)
    model.save_pretrained(output / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(output / "adapter")
    final_adapter = adapter_values(model)
    final_adapter_hash = adapter_hash(final_adapter)
    delta = math.sqrt(
        sum(
            float((final_adapter[n].double() - initial_adapter[n].double()).square().sum())
            for n in initial_adapter
        )
    )
    assert delta > 0 and final_adapter_hash != initial_adapter_hash
    base_final = parameter_hash(base_parameters)
    assert base_final == base_initial
    evidence = {
        "optimizer_steps": len(logs),
        "base_initial_sha256": base_initial,
        "base_final_sha256": base_final,
        "base_weights_unchanged": True,
        "initial_adapter_sha256": initial_adapter_hash,
        "final_adapter_sha256": final_adapter_hash,
        "adapter_l2_change": delta,
        "trained_sample_ids": sorted({s for b in batches for s in b["sample_ids"]}),
        "supervised_loss_only_on_assistant_completion": True,
    }
    save("weight-evidence.json", evidence)
    metadata.update(
        peak_gpu_bytes=torch.cuda.max_memory_allocated(0),
        elapsed_seconds=time.monotonic() - started,
    )
    save("model-config.json", metadata)
    save(
        "status.json",
        {"status": "completed", "optimizer_steps": len(logs), "training_method": "supervised"},
    )
    archive = shutil.make_archive(str(output), "zip", output)
    print("SCALEUP TRAINING COMPLETE", json.dumps(evidence), "Archive:", archive, flush=True)
