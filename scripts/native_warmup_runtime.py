"""Standalone GPU runtime embedded verbatim in the private warm-up notebook.

No account credential, local executor, model-output execution, or RL is used here.
"""


def run(payload):
    import ast
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
        "repairlab-native-warmup-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(exist_ok=False)

    def save(name, value):
        path = output / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")

    def phase(name):
        save("status.json", {"status": "running", "phase": name})
        print("PHASE", name, flush=True)

    def parameter_hash(values):
        digest = hashlib.sha256()
        for name, value in values.items():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    def adapter_hash(values):
        digest = hashlib.sha256()
        for name in sorted(values):
            digest.update(name.encode() + b"\0")
            digest.update(
                values[name].detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            )
        return digest.hexdigest()

    def adapter_values(model):
        return {
            k: v.detach().cpu().contiguous().clone()
            for k, v in get_peft_model_state_dict(model).items()
        }

    def validate(assistant):
        calls = assistant.get("tool_calls") if isinstance(assistant, dict) else None
        if not isinstance(calls, list) or len(calls) != 1:
            raise ValueError("Exactly one native tool call required")
        function = calls[0]["function"]
        name, arguments = function["name"], function["arguments"]
        if name not in {"read", "test", "edit", "submit"} or not isinstance(arguments, dict):
            raise ValueError("Invalid native function or argument type")
        if set(arguments) != ({"source"} if name == "edit" else set()):
            raise ValueError("Invalid native argument fields")
        if name == "edit":
            source = arguments["source"]
            if not isinstance(source, str) or len(source.encode()) > 8192:
                raise ValueError("Invalid source argument")
            tree = ast.parse(source)
            if (
                len(tree.body) != 1
                or not isinstance(tree.body[0], ast.FunctionDef)
                or tree.body[0].name != "solve"
            ):
                raise ValueError("Return one solve function")
            function_node = tree.body[0]
            if (
                function_node.decorator_list
                or function_node.args.defaults
                or any(item is not None for item in function_node.args.kw_defaults)
            ):
                raise ValueError("No decorators or defaults allowed")
            if function_node.returns or any(
                arg.annotation
                for arg in (
                    function_node.args.posonlyargs
                    + function_node.args.args
                    + function_node.args.kwonlyargs
                )
            ):
                raise ValueError("No annotations allowed")
        return {"tool": name, **arguments}

    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    torch.cuda.manual_seed_all(config["seed"])
    torch.backends.cuda.matmul.allow_tf32 = False
    save("bundle.json", bundle)
    phase("loading")
    tokenizer = AutoTokenizer.from_pretrained(
        payload["model"], revision=payload["model_revision"], trust_remote_code=False
    )
    tokenizer.response_schema = qwen3_schema
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    (output / "chat_template.jinja").write_text(tokenizer.chat_template)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = (
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
    base_parameters = dict(model.named_parameters())
    initial_base_hash = parameter_hash(base_parameters)
    assert initial_base_hash == payload["expected_original_base_sha256"], (
        "Original base weights differ from the native diagnostic"
    )
    metadata = {
        "protocol": bundle["protocol"],
        "model": payload["model"],
        "model_revision": payload["model_revision"],
        "bundle_sha256": bundle["bundle_sha256"],
        "config": config,
        "gpu": torch.cuda.get_device_name(0),
        "dtype": str(dtype),
        "base_initial_sha256": initial_base_hash,
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
    assert len(train) == 64 and all(r["split"] == "train" for r in train)
    assert {r["group_id"] for r in train}.isdisjoint({r["group_id"] for r in bundle["development"]})
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
        if parsed.get("tool_calls") != sample["completion"][0]["tool_calls"]:
            raise RuntimeError("Teacher native action did not round-trip")
        labels = [-100] * len(prefix_ids) + ids[len(prefix_ids) :]
        assert all(v == -100 for v in labels[: len(prefix_ids)])
        assert labels[len(prefix_ids) :] == ids[len(prefix_ids) :]
        encoded.append({"input_ids": ids, "labels": labels})
        token_stats.append(
            {
                "sample_id": sample["sample_id"],
                "program_id": sample["program_id"],
                "prefix_tokens": len(prefix_ids),
                "completion_tokens": len(ids) - len(prefix_ids),
                "total_tokens": len(ids),
                "teacher_roundtrip": True,
                "prompt_masked": True,
            }
        )
    save("tokenization-preflight.json", token_stats)
    save(
        "tokenized-training.json",
        [{"sample_id": sample["sample_id"], **row} for sample, row in zip(train, encoded)],
    )

    def evaluate(label):
        model.eval()
        rows = []
        for category, fixtures in (
            ("historical_diagnostic", bundle["historical_diagnostic"]),
            ("separate_programs", bundle["development"]),
        ):
            for probe in fixtures:
                inputs = tokenizer.apply_chat_template(
                    probe["messages"],
                    tools=tools,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    tokenize=True,
                    return_dict=True,
                    return_tensors="pt",
                ).to("cuda:0")
                length = int(inputs["input_ids"].shape[-1])
                if length + config["max_eval_output_tokens"] > config["max_eval_context_tokens"]:
                    raise RuntimeError("Evaluation prompt exceeds context budget")
                begin = time.monotonic()
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs,
                        do_sample=False,
                        num_beams=1,
                        max_new_tokens=config["max_eval_output_tokens"],
                        use_cache=True,
                        pad_token_id=tokenizer.eos_token_id,
                    )
                torch.cuda.synchronize()
                completion = generated[0, length:]
                raw = tokenizer.decode(completion, skip_special_tokens=False)
                row = {
                    "category": category,
                    "probe_id": probe["probe_id"],
                    "messages": probe["messages"],
                    "raw": raw,
                    "input_tokens": length,
                    "output_tokens": int(completion.shape[0]),
                    "elapsed_seconds": time.monotonic() - begin,
                }
                try:
                    parsed = tokenizer.parse_response(raw)
                    action = validate(parsed)
                    row.update(
                        parsed=parsed,
                        action=action,
                        valid_native_call=True,
                        requested_tool_matched=action["tool"] == probe["expected_tool"],
                        requested_source_matched=(
                            action.get("source", "").strip() == probe["expected_source"].strip()
                            if probe["expected_source"] is not None
                            else None
                        ),
                    )
                except (ValueError, TypeError, KeyError, AttributeError) as exc:
                    row.update(
                        valid_native_call=False,
                        requested_tool_matched=False,
                        error=type(exc).__name__ + ": " + str(exc)[:400],
                    )
                rows.append(row)
                save("evaluation/" + label + "/" + probe["probe_id"] + ".json", row)
        summary = {
            category: {
                "probes": sum(r["category"] == category for r in rows),
                "valid_native_calls": sum(
                    r["category"] == category and r["valid_native_call"] for r in rows
                ),
                "requested_tools_matched": sum(
                    r["category"] == category and r["requested_tool_matched"] for r in rows
                ),
                "source_matches": sum(
                    r["category"] == category and r.get("requested_source_matched") is True
                    for r in rows
                ),
            }
            for category in ("historical_diagnostic", "separate_programs")
        }
        save("evaluation/" + label + "/summary.json", summary)
        print(label, json.dumps(summary), flush=True)
        return summary

    started = time.monotonic()
    phase("original_model_evaluation")
    original = evaluate("original")
    assert parameter_hash(base_parameters) == initial_base_hash
    phase("attaching_lora")
    model = get_peft_model(
        model,
        LoraConfig(
            revision=payload["model_revision"],
            r=config["lora_rank"],
            lora_alpha=config["lora_alpha"],
            lora_dropout=config["lora_dropout"],
            target_modules=config["target_modules"],
            task_type="CAUSAL_LM",
        ),
    )
    initial_adapter = adapter_values(model)
    assert all(
        torch.count_nonzero(value) == 0
        for name, value in initial_adapter.items()
        if "lora_B" in name
    )
    save_file(initial_adapter, str(output / "initial_adapter.safetensors"))
    initial_adapter_hash = adapter_hash(initial_adapter)
    trainable = {n: p for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)
    assert all(not p.requires_grad for p in base_parameters.values())
    save(
        "trainable-parameters.json",
        {
            "names": list(trainable),
            "trainable_parameters": sum(p.numel() for p in trainable.values()),
            "initial_adapter_sha256": initial_adapter_hash,
        },
    )
    optimizer = torch.optim.AdamW(
        list(trainable.values()), lr=config["learning_rate"], weight_decay=0.0
    )
    model.train()
    model.config.use_cache = False
    generator = random.Random(config["seed"])
    order = list(range(len(encoded)))
    generator.shuffle(order)
    cursor, logs, batches = 0, [], []
    phase("supervised_training")
    for step in range(config["optimizer_steps"]):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for micro in range(config["gradient_accumulation"]):
            indices = []
            for _ in range(config["micro_batch_size"]):
                if cursor == len(order):
                    generator.shuffle(order)
                    cursor = 0
                indices.append(order[cursor])
                cursor += 1
            features = [encoded[i] for i in indices]
            length = max(len(r["input_ids"]) for r in features)
            inputs = torch.tensor(
                [
                    r["input_ids"] + [tokenizer.pad_token_id] * (length - len(r["input_ids"]))
                    for r in features
                ],
                dtype=torch.long,
                device="cuda:0",
            )
            labels = torch.tensor(
                [r["labels"] + [-100] * (length - len(r["labels"])) for r in features],
                dtype=torch.long,
                device="cuda:0",
            )
            attention = torch.tensor(
                [
                    [1] * len(r["input_ids"]) + [0] * (length - len(r["input_ids"]))
                    for r in features
                ],
                dtype=torch.long,
                device="cuda:0",
            )
            with torch.autocast("cuda", dtype=dtype):
                loss = model(
                    input_ids=inputs, attention_mask=attention, labels=labels, use_cache=False
                ).loss
            if not torch.isfinite(loss):
                raise RuntimeError("Nonfinite supervised loss")
            losses.append(float(loss.detach()))
            (loss / config["gradient_accumulation"]).backward()
            batches.append(
                {
                    "optimizer_step": step + 1,
                    "micro_batch": micro + 1,
                    "sample_ids": [train[i]["sample_id"] for i in indices],
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
            }
        )
        save("training-log.json", logs)
        if (step + 1) % 10 == 0:
            print("Optimizer step", step + 1, "loss", logs[-1]["loss"], flush=True)
        if step + 1 == 30:
            model.save_pretrained(output / "checkpoint-30", safe_serialization=True)
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
    final_base_hash = parameter_hash(base_parameters)
    assert final_base_hash == initial_base_hash
    phase("trained_adapter_evaluation")
    model.eval().requires_grad_(False)
    trained = evaluate("supervised")
    assert adapter_hash(adapter_values(model)) == final_adapter_hash
    evidence = {
        "optimizer_steps": len(logs),
        "base_initial_sha256": initial_base_hash,
        "base_final_sha256": final_base_hash,
        "base_weights_unchanged": True,
        "initial_adapter_sha256": initial_adapter_hash,
        "final_adapter_sha256": final_adapter_hash,
        "adapter_l2_change": delta,
        "adapter_weights_changed": True,
        "evaluation_weights_unchanged": True,
        "trained_sample_ids": sorted({s for b in batches for s in b["sample_ids"]}),
        "supervised_loss_only_on_assistant_completion": True,
    }
    save("weight-evidence.json", evidence)
    comparison = {
        "original": original,
        "supervised": trained,
        "optimizer_steps": len(logs),
        "weight_evidence": evidence,
        "training": "supervised next-action LoRA; no RL",
        "final_test_result": False,
        "repair_benchmark_accuracy_measured": False,
    }
    save("comparison.json", comparison)
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
    print("WARMUP COMPLETE", json.dumps(comparison), "Archive:", archive, flush=True)
    return comparison
