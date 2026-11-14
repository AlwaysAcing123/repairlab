"""Supervised arm of the RL-first phase (plan, Arms item 4). Appended to rlfirst_runtime.py.

The launcher embeds ``rlfirst_runtime.py`` followed by this file, so the supervised arm
uses the RL arms' own code for loading (``_load``), the LoRA settings (``_policy``), the
prompt (``_render`` of ``first_prompt``), and the update (``_update`` with every
advantage 1: the summed log-probability of the target tokens divided by the batch's total
target tokens). The target is the reference function in one ```python block, ended by the
same stop token the RL replies end with.

One pass over the chosen tasks in a seeded order, 8 tasks per optimizer step, learning
rate 1e-4 (constant), gradient clip 1.0, AdamW without weight decay: the 1.7B phase's
supervised settings. With fp16, a step whose gradients overflow is redone on the same
batch at the gradient scaler's lowered scale, so every batch gives exactly one optimizer
step. No sampling, grading, or evaluation happens here.
"""


def run_sft(payload):
    import hashlib
    import json
    import random
    import shutil
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    import torch

    began = time.monotonic()
    settings = payload["sft"]
    output = Path("repairlab-rlfirst-sft-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(parents=True, exist_ok=False)
    _install_code(payload)
    from repairlab.rlfirst.episode import first_prompt, parse

    tasks = _tasks(payload)
    references = payload["references"]
    tokenizer, base = _load(payload["model"], payload["dtype"])
    dtype = torch.float16 if payload["dtype"] == "fp16" else torch.bfloat16
    stop = tokenizer.convert_tokens_to_ids("<|im_end|>")
    examples = []
    for row, task in zip(payload["tasks"], tasks):
        reference = references[row["program"]]
        target = "```python\n" + reference + "```"
        if parse(target, task["signature"])["source"] != reference:
            raise RuntimeError("Target does not parse back to the reference: " + row["task_id"])
        prompt = _render(
            tokenizer,
            [{"role": "user", "content": first_prompt(task, task["bug_public_result"])}],
        )
        prompt_ids = tokenizer(prompt)["input_ids"]
        ids = tokenizer(target, add_special_tokens=False)["input_ids"] + [stop]
        if len(prompt_ids) + len(ids) > settings["max_sequence_tokens"]:
            raise RuntimeError("Example longer than the RL arms' limit: " + row["task_id"])
        examples.append(
            {
                "task_id": row["task_id"],
                "prompt_ids": prompt_ids,
                "ids": ids,
                "target_sha256": hashlib.sha256(target.encode()).hexdigest(),
            }
        )
    torch.manual_seed(payload["seed"])
    policy = _policy(base)
    policy.eval()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=settings["lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=payload["dtype"] == "fp16")
    order = list(range(len(examples)))
    random.Random(payload["seed"]).shuffle(order)
    steps = []
    for start in range(0, len(order), settings["batch_tasks"]):
        chosen = [examples[i] for i in order[start : start + settings["batch_tasks"]]]
        items = [(e["prompt_ids"], e["ids"], 1.0) for e in chosen]
        attempts = []
        for _ in range(settings["max_overflow_retries"] + 1):
            update = _update(policy, optimizer, scaler, items, settings, dtype)
            attempts.append(update)
            if update["finite_gradients"]:
                break
        else:
            raise RuntimeError("Gradients overflowed on every retry")
        row = {
            "step": len(steps) + 1,
            "task_ids": [e["task_id"] for e in chosen],
            "target_tokens": update["trained_tokens"],
            "mean_target_nll": update["loss"],
            "update": update,
            "overflow_retries": len(attempts) - 1,
            "elapsed_seconds": round(time.monotonic() - began, 1),
        }
        steps.append(row)
        print(json.dumps({k: row[k] for k in ("step", "target_tokens", "overflow_retries")}
                         | {"loss": update["loss"], "grad_norm": update["grad_norm"]}), flush=True)
    keep = output / "state"
    keep.mkdir()
    policy.save_pretrained(keep / "adapter")
    log = {
        "arm": "supervised",
        "examples": [{k: e[k] for k in ("task_id", "target_sha256")} for e in examples],
        "order": [examples[i]["task_id"] for i in order],
        "steps": steps,
        "optimizer_steps": len(steps),
        "seconds": round(time.monotonic() - began),
        "peak_gib": torch.cuda.max_memory_allocated(0) / 2**30,
    }
    (output / "log.json").write_text(json.dumps(log, indent=2) + "\n")
    archive = shutil.make_archive(str(output), "zip", output)
    print("SFT COMPLETE", len(steps), "steps", "Archive:", archive, flush=True)
