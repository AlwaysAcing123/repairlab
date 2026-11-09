"""Build a private, uploadable GPU notebook containing ONLY public pilot inputs.

Inference runs on Kaggle/Colab; generated patches are graded locally in Docker.
No candidate code executes in the notebook that holds the GPU or its credentials.
"""

import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repairlab.tasks import canonical, manifest, pilot_tasks  # noqa: E402


def build(humanevalfix=False):
    if humanevalfix:
        from repairlab.humanevalfix import PROTOCOL, fingerprint, public_tasks

        tasks = public_tasks()
        dataset_hash = fingerprint()
    else:
        tasks = [t.public() for t in pilot_tasks()]
        dataset_hash = manifest()["manifest_sha256"]
    cells = []

    def markdown(value):
        cells.append(
            {
                "cell_type": "markdown",
                "metadata": {},
                "source": textwrap.dedent(value).strip() + "\n",
            }
        )

    def code(value):
        cells.append(
            {
                "cell_type": "code",
                "metadata": {},
                "execution_count": None,
                "outputs": [],
                "source": textwrap.dedent(value).strip() + "\n",
            }
        )

    markdown("""
    # RepairLab: original-model feasibility baseline

    Generate one proposed repair for each of **24 development problems**. This is
    a **single-response baseline without execution feedback**, not an interactive
    agent evaluation, training run, or final held-out result. No weights are trained.

    Qwen3-0.6B is the first feasibility candidate because this project's predecessor
    already ran it on a free GPU. It is not a claim that this is the best coding model.
    Near-zero success would tell us to reconsider difficulty or the starting model.

    **Kaggle:** import this notebook, keep it private, enable Internet, select one GPU.
    **Colab:** upload the notebook, select a GPU runtime. Run cells in order.
    Check your live free quota before starting; this notebook does not read it.
    The run has 24 prompts, one response per prompt, and at most 1,024 new tokens
    per response. It does not use paid inference or require an API key.

    Only public task data and a version fingerprint are embedded. There are no
    private cases, reference repairs, or grader in this notebook. Model-generated
    Python is saved as text and NEVER executed here. Download the output archive
    and evaluate it locally with the command at the end.
    """)
    code("""
    import os
    import sys
    import subprocess
    if sys.version_info < (3, 10):
        raise RuntimeError("Use a Python 3.10+ GPU runtime")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("Select a GPU runtime before continuing")
    print("GPU:", torch.cuda.get_device_name(0))
    subprocess.check_call([sys.executable, "-m", "pip", "install", "transformers==5.2.0"])
    """)
    public = canonical(tasks)
    code(
        "import json\nTASKS = json.loads("
        + repr(public)
        + ")\n"
        + "MANIFEST_SHA256 = "
        + repr(dataset_hash)
        + "\n"
        + f"assert len(TASKS) == {len(tasks)}\nprint('Public tasks:', len(TASKS))"
    )
    code("""
    import importlib.metadata
    import os
    from pathlib import Path
    import random
    import time
    from datetime import datetime, timezone
    from transformers import AutoModelForCausalLM, AutoTokenizer

    MODEL = "Qwen/Qwen3-0.6B"
    REVISION = "c1899de289a04d12100db370d81485cdf75e47ca"
    SEED = 7
    MAX_NEW_TOKENS = 1024
    SYSTEM = (
        "Repair the Python function to satisfy its specification. "
        "Return ONLY the complete corrected Python source, exactly one function "
        "named solve. Preserve its signature. Do not include Markdown, decorators, "
        "annotations, default arguments, examples, or explanations. Inputs and "
        "outputs are JSON values. Preserve output types and do not mutate inputs. "
        "You cannot execute code or use tools. You get one response."
    )
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cuda.matmul.allow_tf32 = False
    OUTPUT = Path("repairlab-baseline-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    OUTPUT.mkdir(exist_ok=False)
    (OUTPUT / "raw").mkdir()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    metadata = {
        "protocol": "single-response-v1", "split": "development", "model": MODEL,
        "model_revision": REVISION, "adapter": None, "seed": SEED,
        "manifest_sha256": MANIFEST_SHA256, "system_message": SYSTEM,
        "decoding": {"do_sample": False, "num_beams": 1, "num_return_sequences": 1,
                     "max_new_tokens": MAX_NEW_TOKENS, "enable_thinking": False,
                     "max_input_tokens": 3072},
        "source_extraction": "strip-whitespace-and-one-enclosing-code-fence-v1",
        "gpu": torch.cuda.get_device_name(0), "dtype": str(dtype),
        "packages": {p: importlib.metadata.version(p) for p in
                     ("torch", "transformers", "tokenizers", "huggingface-hub", "safetensors")},
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (OUTPUT / "model-config.json").write_text(json.dumps(metadata, indent=2))
    tokenizer = AutoTokenizer.from_pretrained(MODEL, revision=REVISION, trust_remote_code=False)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL, revision=REVISION, torch_dtype=dtype, trust_remote_code=False
    ).to("cuda:0").eval()
    print("Loaded", MODEL, REVISION, "Output:", OUTPUT)
    """)
    code("""
    # No exec/eval/import of model output. All patches remain text until local grading.
    def extract_source(raw):
        text = raw.strip()
        lines = text.splitlines()
        if len(lines) >= 3 and lines[0] in ("```python", "```") and lines[-1] == "```":
            return "\\n".join(lines[1:-1]) + "\\n"
        return text + "\\n"

    started = time.monotonic()
    with (OUTPUT / "predictions.jsonl").open("w") as predictions:
        for index, task in enumerate(TASKS):
            begin = time.monotonic()
            row = {"task_id": task["task_id"]}
            try:
                messages = [{"role": "system", "content": SYSTEM},
                            {"role": "user", "content": json.dumps(task, ensure_ascii=False)}]
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
                if inputs.input_ids.shape[1] > 3072:
                    raise ValueError("Input exceeds the fixed 3072-token budget")
                with torch.inference_mode():
                    generated = model.generate(
                        **inputs, do_sample=False, num_beams=1, num_return_sequences=1,
                        max_new_tokens=MAX_NEW_TOKENS, pad_token_id=tokenizer.eos_token_id
                    )
                completion = generated[0, inputs.input_ids.shape[1]:]
                raw = tokenizer.decode(completion, skip_special_tokens=True)
                filename = task["task_id"].replace("/", "-") + ".txt"
                (OUTPUT / "raw" / filename).write_text(raw)
                row.update(source=extract_source(raw), raw_output=raw,
                           input_tokens=int(inputs.input_ids.shape[1]),
                           output_tokens=int(completion.shape[0]))
            except Exception as exc:
                row["generation_error"] = type(exc).__name__ + ": " + str(exc)[:1000]
            row["elapsed_seconds"] = time.monotonic() - begin
            predictions.write(json.dumps(row, ensure_ascii=False) + "\\n")
            predictions.flush()
            print(index + 1, "/", len(TASKS), task["task_id"],
                  "ERROR" if "generation_error" in row else "saved; not yet graded", flush=True)
            if row.get("generation_error", "").startswith("OutOfMemoryError"):
                print("Stopping after GPU memory failure; do not retry unchanged settings.")
                break
    (OUTPUT / "runtime.json").write_text(json.dumps({
        "elapsed_seconds": time.monotonic() - started,
        "peak_gpu_bytes": torch.cuda.max_memory_allocated(0),
        "note": "Inference only. No repair execution, correctness scores, or weight updates."
    }, indent=2))
    """)
    code("""
    import shutil
    archive = shutil.make_archive(str(OUTPUT), "zip", OUTPUT)
    print("Download:", archive)
    try:
        from google.colab import files
        files.download(archive)
    except ImportError:
        from IPython.display import FileLink, display
        display(FileLink(archive))
    """)
    markdown("""
    ## Grade locally

    Download and extract the archive. From the local `repairlab` directory, with
    Docker running:

    ```sh
    python3 -B -m repairlab score-patches \\
      --predictions /absolute/path/to/extracted/predictions.jsonl \\
      --metadata /absolute/path/to/extracted/model-config.json \\
      --output artifacts/baseline-001
    ```

    All 24 tasks remain in the denominator, including missing, malformed, and
    failed generations. Infrastructure failures stop scoring and must be diagnosed.
    Scores on these development tasks guide our next experiment; they are not a
    final test result. Do not train on them and later call their scores held-out.
    """)
    if humanevalfix:
        instruction = (
            "Repair the buggy Python function to satisfy its specification and examples. "
            "Return ONLY the complete corrected Python source, including necessary "
            "standard-library imports. Preserve the supplied entry-point name and signature. "
            "Do not include Markdown, tests, or explanations. You cannot execute code or use tools. "
            "You get one response."
        )
        for cell in cells:
            value = cell["source"].replace(
                "24 development problems", "164 external HumanEvalFix problems"
            )
            value = value.replace("24 prompts", "164 prompts").replace(
                "All 24 tasks", "All 164 tasks"
            )
            value = value.replace("repairlab score-patches", "repairlab external-score")
            if "SYSTEM = (" in value:
                start = value.index("SYSTEM = (")
                end = value.index("random.seed(SEED)", start)
                value = value[:start] + f"SYSTEM = {instruction!r}\n" + value[end:]
                value = value.replace(
                    '"protocol": "single-response-v1", "split": "development"',
                    f'"protocol": {PROTOCOL!r}, "split": "external-public-test"',
                )
            cell["source"] = value
        markdown("""
        This is **RepairLab's HumanEvalFix Python adaptation**, using docstrings AND
        public examples, complete-source responses, and the original assertion tests.
        Prompting, extraction, helper imports, and execution limits differ from the
        authors' harness. These scores are not a reproduction of published leaderboard
        results. The local external assert runner is not a reward-hacking-resistant RL
        oracle. Do not train on these public test tasks.
        """)
    notebook = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "accelerator": "GPU",
        },
        "cells": cells,
    }
    for index, cell in enumerate(cells):
        cell["id"] = f"repairlab-{index:02d}"
    filename = "RepairLab_HumanEvalFix.ipynb" if humanevalfix else "RepairLab_Baseline.ipynb"
    path = ROOT / "notebooks" / filename
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n")
    print(path)
    return notebook


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--humanevalfix", action="store_true")
    build(parser.parse_args().humanevalfix)
