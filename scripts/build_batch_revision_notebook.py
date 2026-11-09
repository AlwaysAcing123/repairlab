"""Public-data-only batch inference notebook. No credentials or network callbacks."""

import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repairlab.revision import MODEL, REVISION  # noqa: E402


def build(destination, batch):
    cells = []

    def add(kind, text):
        cell = {"cell_type": kind, "metadata": {}, "source": textwrap.dedent(text).strip() + "\n"}
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cell["id"] = f"batch-{len(cells):02d}"
        cells.append(cell)

    add(
        "markdown",
        """
    # RepairLab: one paired revision round

    Original Qwen3-0.6B, no training. This batch generates ONE next action per
    unfinished episode. Only public task conversations are embedded. There are no
    credentials, callback endpoints, reference repairs, hidden cases, or grader.
    Model-generated code is never executed on this GPU. Downloaded outputs are
    applied locally before the next round is generated. Keep the notebook private.
    """,
    )
    add(
        "code",
        """
    import os, sys, subprocess
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==5.2.0"])
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime")
    """,
    )
    add(
        "code",
        "import json\nBATCH = json.loads("
        + repr(json.dumps(batch, ensure_ascii=False))
        + ")\nMODEL = "
        + repr(MODEL)
        + "\nREVISION = "
        + repr(REVISION),
    )
    add(
        "code",
        """
    import hashlib, importlib.metadata, random, time
    from pathlib import Path
    from datetime import datetime, timezone
    from transformers import AutoModelForCausalLM, AutoTokenizer
    random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    output = Path("repairlab-round-" + str(BATCH["round"]) + "-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=dtype, trust_remote_code=False
    ).to("cuda:0").eval().requires_grad_(False)
    def weight_hash():
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()
    metadata = {
        "model": MODEL, "model_revision": REVISION, "protocol": BATCH["protocol"], "budget": BATCH["budget"],
        "batch_sha256": BATCH["batch_sha256"], "round": BATCH["round"], "seed": 7,
        "decoding": {"do_sample": False, "num_beams": 1, "enable_thinking": False},
        "initial_weight_sha256": weight_hash(), "gpu": torch.cuda.get_device_name(0), "dtype": str(dtype),
        "chat_template_sha256": hashlib.sha256(tokenizer.chat_template.encode()).hexdigest(),
        "packages": {p: importlib.metadata.version(p) for p in
                     ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors")}}
    (output / "model-config.json").write_text(json.dumps(metadata, indent=2))
    print("Loaded original model; round", BATCH["round"], "requests", len(BATCH["requests"]), flush=True)
    """,
    )
    add(
        "code",
        """
    started = time.monotonic()
    with (output / "responses.jsonl").open("w") as log:
        for work in BATCH["requests"]:
            row = {"episode_id": work["episode_id"], "decision": work["decision"]}
            begin = time.monotonic()
            try:
                prompt = tokenizer.apply_chat_template(work["messages"], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False)
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
                length = int(inputs.input_ids.shape[1])
                allowance = work["max_new_tokens"]
                row["input_tokens"] = length
                if allowance <= 0:
                    row.update(stop="output_token_budget", output_tokens=0)
                elif length + allowance > work["max_context_tokens"]:
                    row.update(stop="context_budget", output_tokens=0)
                else:
                    with torch.inference_mode():
                        generated = model.generate(**inputs, do_sample=False, num_beams=1,
                            num_return_sequences=1, max_new_tokens=allowance,
                            pad_token_id=tokenizer.eos_token_id)
                    torch.cuda.synchronize()
                    completion = generated[0, length:]
                    row.update(raw=tokenizer.decode(completion, skip_special_tokens=True),
                               output_tokens=int(completion.shape[0]))
            except Exception as exc:
                row["generation_error"] = type(exc).__name__ + ": " + str(exc)[:500]
            row["elapsed_seconds"] = time.monotonic() - begin
            log.write(json.dumps(row, ensure_ascii=False) + "\\n")
            log.flush()
            print(work["episode_id"], "decision", work["decision"], "saved", flush=True)
            if row.get("generation_error"):
                raise RuntimeError("Generation failed; outputs saved for diagnosis")
    metadata.update(final_weight_sha256=weight_hash(), elapsed_seconds=time.monotonic()-started,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated(0))
    assert metadata["initial_weight_sha256"] == metadata["final_weight_sha256"]
    (output / "model-config.json").write_text(json.dumps(metadata, indent=2))
    import shutil
    archive = shutil.make_archive(str(output), "zip", output)
    print("Results archive:", archive, flush=True)
    """,
    )
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
            "kaggle": {"isInternetEnabled": True, "isGpuEnabled": True},
        },
        "cells": cells,
    }
    Path(destination).write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n")
    return notebook
