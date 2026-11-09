"""Build the public-protocol GPU client; no tasks or grading material are embedded.

An ephemeral coordinator bearer credential is embedded in a PRIVATE submission only.
The Kaggle account token is never part of this file or its notebook output.
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repairlab.revision import BUDGET, MODEL, PROTOCOL, REVISION  # noqa: E402


def build(destination, endpoint, token):
    if not endpoint.startswith("https://") or len(token) < 32:
        raise ValueError("HTTPS endpoint and a strong ephemeral coordinator token required")
    cells = []

    def markdown(text):
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": textwrap.dedent(text).strip() + "\n",
            }
        )

    def code(text):
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "source": textwrap.dedent(text).strip() + "\n",
                "execution_count": None,
                "outputs": [],
            }
        )

    markdown("""
    # RepairLab: paired revision-loop experiment

    Original Qwen3-0.6B, no training. 24 development tasks in each of two conditions:
    execution feedback available versus withheld. Eight actions and 4,096 output tokens
    per episode, at most 512 tokens per call, same prompt and decoding in both conditions.
    The pair order alternates by task. Final correctness checks are performed locally
    and are never sent back into the model conversation.

    This PRIVATE notebook contains a short-lived credential for a narrowly scoped local
    coordinator. It contains NO Kaggle account credential, task references, or grader.
    Do not publish it while the coordinator is active. Shut down the temporary link after
    the run. Model output is never executed in this GPU environment.
    """)
    code("""
    import os
    import sys
    import subprocess
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==5.2.0"])
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A GPU is required")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    """)
    code(
        "import json\n"
        + "ENDPOINT = "
        + repr(endpoint.rstrip("/"))
        + "\n"
        + "COORDINATOR_TOKEN = "
        + repr(token)
        + "\n"
        + "MODEL = "
        + repr(MODEL)
        + "\nREVISION = "
        + repr(REVISION)
        + "\nPROTOCOL = "
        + repr(PROTOCOL)
        + "\nBUDGET = json.loads("
        + repr(json.dumps(BUDGET))
        + ")\n"
    )
    code("""
    import hashlib
    import importlib.metadata
    from pathlib import Path
    import random
    import time
    from urllib.request import Request, urlopen
    from urllib.error import HTTPError, URLError
    from datetime import datetime, timezone
    from transformers import AutoModelForCausalLM, AutoTokenizer

    output = Path("repairlab-revision-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)

    def request(path, data=None):
        payload = None if data is None else json.dumps(data, allow_nan=False).encode()
        headers = {"Authorization": "Bearer " + COORDINATOR_TOKEN,
                   "Content-Type": "application/json"}
        # Retry the IDENTICAL action ID and bytes only; host processing is idempotent.
        for attempt in range(3):
            try:
                response = urlopen(Request(ENDPOINT + path, data=payload, headers=headers), timeout=75)
                with response:
                    return json.load(response)
            except HTTPError as exc:
                if exc.code not in (502, 504) or attempt == 2:
                    raise RuntimeError("Coordinator HTTP " + str(exc.code)) from None
            except (URLError, TimeoutError):
                if attempt == 2:
                    raise RuntimeError("Coordinator unreachable after three identical requests") from None
            time.sleep(1)

    def weight_hash(model):
        digest = hashlib.sha256()
        for name, parameter in model.named_parameters():
            digest.update(name.encode())
            digest.update(parameter.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()

    random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.backends.cuda.matmul.allow_tf32 = False
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, dtype=dtype, trust_remote_code=False
    ).to("cuda:0").eval().requires_grad_(False)
    metadata = {
        "model": MODEL, "model_revision": REVISION, "protocol": PROTOCOL, "budget": BUDGET,
        "seed": 7, "decoding": {"do_sample": False, "enable_thinking": False, "num_beams": 1},
        "gpu": torch.cuda.get_device_name(0), "dtype": str(dtype),
        "initial_weight_sha256": weight_hash(model), "adapter": None,
        "packages": {p: importlib.metadata.version(p) for p in
                     ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors")},
    }
    (output / "model-config.json").write_text(json.dumps(metadata, indent=2))
    request("/metadata", metadata)
    print("Model loaded and registered. No weights will be updated.", flush=True)
    """)
    code("""
    started = time.monotonic()
    calls, episodes = 0, 0
    with (output / "model-calls.jsonl").open("w") as log:
        while True:
            work = request("/work")
            if work["done"]:
                break
            ident = work["episode_id"] + ":" + str(work["decision"])
            payload = {"request_id": ident}
            prompt = tokenizer.apply_chat_template(
                work["messages"], tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
            length = int(inputs.input_ids.shape[1])
            allowance = work["max_new_tokens"]
            if allowance <= 0:
                payload.update(stop="output_token_budget", input_tokens=length)
            elif length + allowance > work["max_context_tokens"]:
                payload.update(stop="context_budget", input_tokens=length)
            else:
                begin = time.monotonic()
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, do_sample=False, num_beams=1, num_return_sequences=1,
                        max_new_tokens=allowance, pad_token_id=tokenizer.eos_token_id
                    )
                torch.cuda.synchronize()
                completion = generated[0, length:]
                payload.update(raw=tokenizer.decode(completion, skip_special_tokens=True),
                               input_tokens=length, output_tokens=int(completion.shape[0]),
                               elapsed_seconds=time.monotonic() - begin)
            row = {"episode_id": work["episode_id"], "decision": work["decision"],
                   "messages": work["messages"], "payload": payload}
            log.write(json.dumps(row, ensure_ascii=False) + "\\n")
            log.flush()
            result = request("/action", payload)
            calls += 1
            if result["episode_done"]:
                episodes += 1
            print(work["episode_id"], "decision", work["decision"],
                  "episode complete" if result["episode_done"] else "continuing", flush=True)
    final = {"final_weight_sha256": weight_hash(model), "calls": calls, "episodes": episodes,
             "elapsed_seconds": time.monotonic() - started,
             "peak_gpu_bytes": torch.cuda.max_memory_allocated(0)}
    assert final["final_weight_sha256"] == metadata["initial_weight_sha256"]
    response = request("/finish", final)
    (output / "completion.json").write_text(json.dumps(final, indent=2))
    print("Completed", response["episodes"], "episodes; weights unchanged.", flush=True)
    """)
    code("""
    import shutil
    archive = shutil.make_archive(str(output), "zip", output)
    print("Results archive:", archive, flush=True)
    """)
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
    for index, cell in enumerate(cells):
        cell["id"] = f"revision-{index:02d}"
    Path(destination).write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n")
    return notebook


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--token-file", required=True, type=Path)
    args = parser.parse_args()
    build(args.output, args.endpoint, args.token_file.read_text().strip())
    print("Private revision notebook built; bearer credential omitted from log.")


if __name__ == "__main__":
    main()
