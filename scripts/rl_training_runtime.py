"""Standalone GPU runtime for one group-relative RL update, embedded in its notebook.

Every prompt is re-rendered and must hash to the token ids the rollout sampled from,
and every completion is scored on its exact sampled token ids. Old log-probabilities
come from the unchanged policy before the first step. Only LoRA weights train; the
base model is fingerprinted before and after. No model output is executed here.
"""


def run_rl_update(payload):
    import hashlib
    import importlib.metadata
    import json
    import math
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

    config, policy, turns = payload["config"], payload["policy"], payload["turns"]
    output = Path(
        "repairlab-rl-update-"
        + str(payload["cycle"])
        + "-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(exist_ok=False)
    scale = 1024.0

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

    def locate(file_sha256):
        for path in sorted(Path("/kaggle/input").rglob("adapter_model.safetensors")):
            if hashlib.sha256(path.read_bytes()).hexdigest() == file_sha256:
                return path.parent
        directory = Path("/kaggle/working/loaded-policy")
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
        raise RuntimeError("Exact policy adapter not found in the attached kernel outputs")

    random.seed(config["seed"] + payload["cycle"])
    torch.manual_seed(config["seed"] + payload["cycle"])
    torch.cuda.manual_seed_all(config["seed"] + payload["cycle"])
    torch.backends.cuda.matmul.allow_tf32 = False
    phase("loading")
    tokenizer = AutoTokenizer.from_pretrained(
        payload["model"], revision=payload["model_revision"], trust_remote_code=False
    )
    tokenizer.response_schema = qwen3_schema
    (output / "chat_template.jinja").write_text(tokenizer.chat_template)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = (
        AutoModelForCausalLM.from_pretrained(
            payload["model"],
            revision=payload["model_revision"],
            dtype=dtype,
            trust_remote_code=False,
        )
        .to("cuda:0")
        .eval()
        .requires_grad_(False)
    )
    base_parameters = dict(base.named_parameters())
    base_initial = parameter_hash(base_parameters)
    assert base_initial == payload["base_sha256"]
    metadata = {
        "protocol": payload["protocol"],
        "cycle": payload["cycle"],
        "payload_sha256": payload["payload_sha256"],
        "model": payload["model"],
        "model_revision": payload["model_revision"],
        "config": config,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "base_initial_sha256": base_initial,
        "loss_scale": scale,
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
    phase("prompt_check")
    encoded = []
    for turn in turns:
        prompt = tokenizer.apply_chat_template(
            turn["messages"],
            tools=payload["tools"],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )["input_ids"][0].tolist()
        if hashlib.sha256(json.dumps(prompt).encode()).hexdigest() != turn["prompt_sha256"]:
            raise RuntimeError("Re-rendered prompt differs from the sampled prompt")
        ids = prompt + turn["completion_ids"]
        if len(ids) > config["max_sequence_tokens"]:
            raise RuntimeError("Training turn exceeds the fixed sequence budget")
        encoded.append({"ids": ids, "completion": len(turn["completion_ids"])})
    phase("attaching_policy")
    model = PeftModel.from_pretrained(base, locate(policy["file_sha256"]), is_trainable=True)
    model = model.to("cuda:0")
    assert model.peft_config["default"].revision == payload["model_revision"]
    assert model.peft_config["default"].lora_dropout == 0.0
    initial_adapter = adapter_values(model)
    initial_hash = adapter_hash(initial_adapter)
    assert initial_hash == policy["tensor_sha256"]
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    assert all(not p.requires_grad for p in base_parameters.values())
    model.train()
    model.config.use_cache = False

    def token_logprobs(index):
        ids = torch.tensor([encoded[index]["ids"]], dtype=torch.long, device="cuda:0")
        count = encoded[index]["completion"]
        with torch.autocast("cuda", dtype=dtype):
            logits = model(input_ids=ids, use_cache=False, logits_to_keep=count + 1).logits
        logits = logits[0, :-1].float()
        targets = ids[0, -count:]
        return torch.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)

    phase("old_logprobs")
    with torch.no_grad():
        old = [token_logprobs(i).detach() for i in range(len(encoded))]
    if not all(bool(torch.isfinite(o).all()) for o in old):
        raise RuntimeError("Nonfinite policy log-probability")
    episodes = sorted({t["episode_id"] for t in turns})
    random.Random(config["seed"] + payload["cycle"]).shuffle(episodes)
    steps = config["optimizer_steps_per_cycle"]
    minibatches = [set(episodes[k::steps]) for k in range(steps)]
    optimizer = torch.optim.AdamW(
        list(trainable.values()), lr=config["learning_rate"], weight_decay=0.0
    )
    epsilon, logs = config["clip_ratio"], []
    started = time.monotonic()
    phase("rl_update")
    for step, members in enumerate(minibatches):
        indices = [i for i, t in enumerate(turns) if t["episode_id"] in members]
        if not indices:
            continue
        tokens = sum(encoded[i]["completion"] for i in indices)
        optimizer.zero_grad(set_to_none=True)
        total = clipped = ratio_sum = 0.0
        for i in indices:
            new = token_logprobs(i)
            ratio = torch.exp(new - old[i])
            advantage = turns[i]["advantage"]
            objective = torch.minimum(
                ratio * advantage, torch.clamp(ratio, 1 - epsilon, 1 + epsilon) * advantage
            )
            loss = -objective.sum() / tokens
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite RL loss")
            (loss * scale).backward()
            total += float(loss.detach())
            ratio_sum += float(ratio.detach().sum())
            clipped += float(((ratio - 1).abs() > epsilon).float().sum())
        for parameter in trainable.values():
            if parameter.grad is not None:
                parameter.grad.div_(scale)
        norm = torch.nn.utils.clip_grad_norm_(list(trainable.values()), config["gradient_clip"])
        if not torch.isfinite(norm):
            raise RuntimeError("Nonfinite gradient norm")
        optimizer.step()
        logs.append(
            {
                "optimizer_step": step + 1,
                "episodes": len(members),
                "turns": len(indices),
                "completion_tokens": tokens,
                "loss": total,
                "mean_ratio": ratio_sum / tokens,
                "clipped_fraction": clipped / tokens,
                "gradient_norm": float(norm.detach()),
            }
        )
        save("training-log.json", logs)
        print("Optimizer step", step + 1, json.dumps(logs[-1]), flush=True)
    model.save_pretrained(output / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(output / "adapter")
    final_adapter = adapter_values(model)
    final_hash = adapter_hash(final_adapter)
    delta = math.sqrt(
        sum(
            float((final_adapter[n].double() - initial_adapter[n].double()).square().sum())
            for n in initial_adapter
        )
    )
    assert delta > 0 and final_hash != initial_hash
    base_final = parameter_hash(base_parameters)
    assert base_final == base_initial
    evidence = {
        "optimizer_steps": len(logs),
        "turns": len(turns),
        "episodes": len(episodes),
        "base_initial_sha256": base_initial,
        "base_final_sha256": base_final,
        "base_weights_unchanged": True,
        "initial_adapter_sha256": initial_hash,
        "final_adapter_sha256": final_hash,
        "adapter_l2_change": delta,
        "prompts_matched_sampled_tokens": len(turns),
    }
    save("weight-evidence.json", evidence)
    metadata.update(
        peak_gpu_bytes=torch.cuda.max_memory_allocated(0),
        elapsed_seconds=time.monotonic() - started,
    )
    save("model-config.json", metadata)
    save("status.json", {"status": "completed", "optimizer_steps": len(logs)})
    archive = shutil.make_archive(str(output), "zip", output)
    print("RL UPDATE COMPLETE", json.dumps(evidence), "Archive:", archive, flush=True)
