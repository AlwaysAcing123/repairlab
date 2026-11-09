"""Standalone sampling client for RL rollouts, embedded verbatim in its notebook.

Model output is never executed here. One adapter is loaded by exact file hash and
shared by every sample arm; its fingerprints are checked before and after the round.
Requests are sampled in left-padded batches seeded from the batch hash, and each
completion keeps its exact token ids so the update can score the same tokens. Padded
attention grows with the square of the longest prompt, so batches are capped by that
square: long histories sample in small batches and short ones in batches of 16.
"""


def run_rl_rollout_round(batch):
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
        "repairlab-rl-rollout-"
        + str(batch["cycle"])
        + "-"
        + str(batch["round"])
        + "-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(exist_ok=False)
    sampling = batch["sampling"]
    batch_size, square_cap = 16, 60_000_000

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

    random.seed(batch["seed"])
    torch.manual_seed(batch["seed"])
    torch.cuda.manual_seed_all(batch["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(
        batch["model"], revision=batch["model_revision"], trust_remote_code=False
    )
    tokenizer.response_schema = qwen3_schema
    (output / "chat_template.jinja").write_text(tokenizer.chat_template)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    base = (
        AutoModelForCausalLM.from_pretrained(
            batch["model"], revision=batch["model_revision"], dtype=dtype, trust_remote_code=False
        )
        .to("cuda:0")
        .eval()
        .requires_grad_(False)
    )
    parameters = dict(base.named_parameters())
    base_initial = parameter_hash(parameters)
    assert base_initial == batch["base_sha256"]
    policies = {a["file_sha256"] for a in batch["arms"].values()}
    assert len(policies) == 1, "Every sample arm must share one adapter"
    policy = next(iter(batch["arms"].values()))
    directory = locate(policy["file_sha256"])
    file_hash = hashlib.sha256((directory / "adapter_model.safetensors").read_bytes()).hexdigest()
    assert file_hash == policy["file_sha256"]
    assert (
        json.loads((directory / "adapter_config.json").read_text())["revision"]
        == batch["model_revision"]
    )
    model = (
        PeftModel.from_pretrained(base, directory, is_trainable=False)
        .to("cuda:0")
        .eval()
        .requires_grad_(False)
    )
    adapter_initial = adapter_hash(model)
    assert adapter_initial == policy["tensor_sha256"]
    stop_ids = model.generation_config.eos_token_id
    stop_ids = set(stop_ids if isinstance(stop_ids, list) else [stop_ids])
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else min(stop_ids)
    metadata = {
        "protocol": batch["protocol"],
        "round": batch["round"],
        "cycle": batch["cycle"],
        "batch_sha256": batch["batch_sha256"],
        "model": batch["model"],
        "model_revision": batch["model_revision"],
        "budget": batch["budget"],
        "seed": batch["seed"],
        "gpu": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "decoding": sampling,
        "batching": {"max_batch": batch_size, "square_cap": square_cap, "padding": "left"},
        "stop_token_ids": sorted(stop_ids),
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
    print("Policy loaded; cycle", batch["cycle"], "round", batch["round"], flush=True)
    rows, active = {}, []
    for work in batch["requests"]:
        row = {
            "episode_id": work["episode_id"],
            "condition": work["condition"],
            "decision": work["decision"],
        }
        ids = tokenizer.apply_chat_template(
            work["messages"],
            tools=batch["tools"],
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            add_generation_prompt=True,
            enable_thinking=False,
        )["input_ids"][0].tolist()
        row["input_tokens"] = len(ids)
        row["prompt_sha256"] = hashlib.sha256(json.dumps(ids).encode()).hexdigest()
        if work["max_new_tokens"] <= 0:
            row.update(stop="output_token_budget", output_tokens=0)
        elif len(ids) + work["max_new_tokens"] > work["max_context_tokens"]:
            row.update(stop="context_budget", output_tokens=0)
        else:
            active.append((work, ids))
        rows[work["episode_id"]] = row
    active.sort(key=lambda item: len(item[1]))
    chunks, chunk = [], []
    for item in active:
        longest = max([len(i) for _, i in chunk + [item]]) + 512
        if chunk and (len(chunk) == batch_size or (len(chunk) + 1) * longest**2 > square_cap):
            chunks.append(chunk)
            chunk = []
        chunk.append(item)
    if chunk:
        chunks.append(chunk)
    started = time.monotonic()
    for index, chunk in enumerate(chunks):
        seed = int(
            hashlib.sha256((batch["batch_sha256"] + ":" + str(index)).encode()).hexdigest()[:8], 16
        )
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        width = max(len(ids) for _, ids in chunk)
        input_ids = torch.tensor(
            [[pad_id] * (width - len(ids)) + ids for _, ids in chunk], device="cuda:0"
        )
        attention = torch.tensor(
            [[0] * (width - len(ids)) + [1] * len(ids) for _, ids in chunk], device="cuda:0"
        )
        allowance = max(work["max_new_tokens"] for work, _ in chunk)
        begin = time.monotonic()
        with torch.inference_mode():
            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention,
                do_sample=sampling["do_sample"],
                temperature=sampling["temperature"],
                top_p=sampling["top_p"],
                top_k=sampling["top_k"],
                num_beams=sampling["num_beams"],
                max_new_tokens=allowance,
                use_cache=True,
                pad_token_id=pad_id,
            )
        torch.cuda.synchronize()
        elapsed = time.monotonic() - begin
        for position, (work, _) in enumerate(chunk):
            completion = generated[position, width:].tolist()[: work["max_new_tokens"]]
            for cut, token in enumerate(completion):
                if token in stop_ids:
                    completion = completion[: cut + 1]
                    break
            raw = tokenizer.decode(completion, skip_special_tokens=False)
            row = rows[work["episode_id"]]
            row.update(
                raw=raw,
                output_tokens=len(completion),
                completion_ids=completion,
                sample_batch=index,
                sample_seed=seed,
                elapsed_seconds=elapsed,
            )
            try:
                row["parsed"] = tokenizer.parse_response(raw)
            except Exception as exc:
                row.update(parsed=None, parse_error=type(exc).__name__ + ": " + str(exc)[:500])
        del generated, input_ids, attention
        torch.cuda.empty_cache()
        print("Sampled batch", index + 1, "of", len(chunks), "size", len(chunk), flush=True)
    with (output / "responses.jsonl").open("w") as log:
        for work in batch["requests"]:
            log.write(json.dumps(rows[work["episode_id"]], ensure_ascii=False) + "\n")
    base_final = parameter_hash(parameters)
    adapter_final = adapter_hash(model)
    assert base_final == batch["base_sha256"] and adapter_final == policy["tensor_sha256"]
    metadata["arms"] = {
        name: {
            "base_initial_sha256": base_initial,
            "base_final_sha256": base_final,
            "adapter_file_sha256": file_hash,
            "adapter_initial_sha256": adapter_initial,
            "adapter_final_sha256": adapter_final,
        }
        for name in batch["arms"]
    }
    metadata.update(
        elapsed_seconds=time.monotonic() - started,
        peak_gpu_bytes=torch.cuda.max_memory_allocated(0),
        sample_batches=len(chunks),
    )
    save("model-config.json", metadata)
    archive = shutil.make_archive(str(output), "zip", output)
    print("RL ROLLOUT ROUND COMPLETE", batch["round"], "Archive:", archive, flush=True)
