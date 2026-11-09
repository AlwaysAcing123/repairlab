"""Standalone memory and speed preflight for Qwen3-1.7B, embedded in its notebook.

No task is attempted and nothing is saved as a model. It measures peak GPU memory and
time for the worst training, RL, and sampling steps the plan allows, and records the
base-parameter fingerprint later runs must match. Out-of-memory is caught and recorded.
"""


def run_preflight(payload):
    import hashlib
    import importlib.metadata
    import json
    import math
    import shutil
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = Path("repairlab-preflight-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    model_spec, config = payload["model"], payload["config"]
    results = {"measurements": []}

    def save():
        (output / "preflight.json").write_text(json.dumps(results, indent=2) + "\n")

    def parameter_hash(values):
        digest = hashlib.sha256()
        for name, value in values.items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def measure(name, function, **details):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        torch.cuda.synchronize()
        begin = time.monotonic()
        row = {"name": name, **details}
        try:
            row.update(function() or {})
            torch.cuda.synchronize()
            row.update(ok=True, oom=False)
        except torch.OutOfMemoryError as exc:
            row.update(ok=False, oom=True, error=str(exc)[:300])
        row.update(
            seconds=time.monotonic() - begin,
            peak_gib=torch.cuda.max_memory_allocated(0) / 2**30,
        )
        results["measurements"].append(row)
        save()
        print(json.dumps(row), flush=True)
        torch.cuda.empty_cache()

    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["model"], revision=model_spec["model_revision"], trust_remote_code=False
    )
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = (
        AutoModelForCausalLM.from_pretrained(
            model_spec["model"],
            revision=model_spec["model_revision"],
            dtype=dtype,
            trust_remote_code=False,
        )
        .to("cuda:0")
        .eval()
        .requires_grad_(False)
    )
    base_parameters = dict(base.named_parameters())
    results.update(
        model=model_spec["model"],
        model_revision=model_spec["model_revision"],
        base_sha256=parameter_hash(base_parameters),
        parameters=sum(p.numel() for p in base_parameters.values()),
        gpu=torch.cuda.get_device_name(0),
        capability=list(torch.cuda.get_device_capability(0)),
        total_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
        dtype=str(dtype),
        attention=base.config._attn_implementation,
        chat_template_sha256=hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        packages={
            p: importlib.metadata.version(p)
            for p in ("torch", "transformers", "peft", "trl", "tokenizers")
        },
    )
    save()
    model = get_peft_model(
        base,
        LoraConfig(
            revision=model_spec["model_revision"],
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["target_modules"],
            task_type="CAUSAL_LM",
        ),
    ).to("cuda:0")
    model.config.use_cache = False
    encoded = {}
    for variant, samples in payload["samples"].items():
        for sample in samples:
            options = dict(tools=payload["tools"], tokenize=False, enable_thinking=False)
            prefix = tokenizer.apply_chat_template(
                sample["prompt"], add_generation_prompt=True, **options
            )
            full = tokenizer.apply_chat_template(
                sample["prompt"] + sample["completion"], add_generation_prompt=False, **options
            )
            assert full.startswith(prefix)
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            ids = tokenizer(full, add_special_tokens=False)["input_ids"]
            encoded[sample["sample_id"]] = (variant, ids, len(prefix_ids))
    longest = max(encoded.values(), key=lambda item: len(item[1]))[1]

    def supervised_step(ids, prefix):
        def step():
            tensor = torch.tensor([ids], dtype=torch.long, device="cuda:0")
            labels = torch.tensor([[-100] * prefix + ids[prefix:]], device="cuda:0")
            with torch.autocast("cuda", dtype=dtype):
                loss = model(input_ids=tensor, labels=labels, use_cache=False).loss
            loss.backward()
            model.zero_grad(set_to_none=True)
            return {"loss_finite": bool(torch.isfinite(loss))}

        return step

    def tiled(length):
        return (longest * math.ceil(length / len(longest)))[:length]

    def rl_step(length, completion=512):
        def step():
            ids = torch.tensor([tiled(length)], dtype=torch.long, device="cuda:0")
            with torch.autocast("cuda", dtype=dtype):
                logits = model(input_ids=ids, use_cache=False, logits_to_keep=completion + 1).logits
            logits = logits[0, :-1].float()
            targets = ids[0, -completion:]
            values = torch.log_softmax(logits, dim=-1).gather(1, targets[:, None]).squeeze(1)
            (-values.mean()).backward()
            model.zero_grad(set_to_none=True)
            return {"finite": bool(torch.isfinite(values).all())}

        return step

    def sampling(batch, length, new_tokens=512):
        def step():
            ids = torch.tensor([tiled(length)] * batch, dtype=torch.long, device="cuda:0")
            with torch.inference_mode():
                generated = model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    do_sample=True,
                    temperature=1.0,
                    top_p=1.0,
                    top_k=0,
                    max_new_tokens=new_tokens,
                    min_new_tokens=new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            return {"generated_tokens": int(generated.shape[1] - length) * batch}

        return step

    def greedy(length, new_tokens=256):
        def step():
            ids = torch.tensor([tiled(length)], dtype=torch.long, device="cuda:0")
            with torch.inference_mode():
                model.generate(
                    input_ids=ids,
                    attention_mask=torch.ones_like(ids),
                    do_sample=False,
                    num_beams=1,
                    max_new_tokens=new_tokens,
                    min_new_tokens=new_tokens,
                    use_cache=True,
                    pad_token_id=tokenizer.eos_token_id,
                )
            return {"new_tokens": new_tokens}

        return step

    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    for sample_id, (variant, ids, prefix) in encoded.items():
        measure(
            "supervised_step_checkpointed",
            supervised_step(ids, prefix),
            sample_id=sample_id,
            variant=variant,
            tokens=len(ids),
        )
    for length in payload["rl_lengths"]:
        measure("rl_step_checkpointed", rl_step(length), tokens=length)
    model.eval()
    for cap in payload["square_caps"]:
        batch = 16
        length = int(math.sqrt(cap / batch)) - 512
        measure(
            "sampling_batch", sampling(batch, length), square_cap=cap, batch=batch, tokens=length
        )
    measure(
        "sampling_single_long", sampling(1, payload["long_prompt"]), tokens=payload["long_prompt"]
    )
    measure("greedy_decode", greedy(payload["greedy_prompt"]), tokens=payload["greedy_prompt"])
    model.gradient_checkpointing_disable()
    model.train()
    variant, ids, prefix = max(encoded.values(), key=lambda item: len(item[1]))
    measure(
        "supervised_step_without_checkpointing",
        supervised_step(ids, prefix),
        variant=variant,
        tokens=len(ids),
    )
    assert parameter_hash(base_parameters) == results["base_sha256"]
    results["base_unchanged"] = True
    save()
    archive = shutil.make_archive(str(output), "zip", output)
    print("PREFLIGHT COMPLETE", results["base_sha256"], "Archive:", archive, flush=True)
