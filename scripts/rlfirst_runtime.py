"""Standalone RL-first runtime (preflight part A and the base sweep), embedded in its notebook.

The payload carries the repairlab modules needed for episodes and in-session grading,
the model spec, and training-pool tasks only (with their hidden cases). It never carries
a development, sealed, or HumanEvalFix task. Nothing here is saved as a model.
"""


def _install_code(payload):
    import sys
    from pathlib import Path

    root = Path("repairlab-code").resolve()
    for relative, text in payload["code"].items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)
    sys.path.insert(0, str(root))
    return root


def _tasks(payload):
    """Rebuild task dicts from programs (cases stored once) and per-task indices."""
    tasks = []
    for row in payload["tasks"]:
        program = payload["programs"][row["program"]]
        cases = [{"args": a, "expected": e} for a, e in program["cases"]]
        public_index = set(row["public_index"])
        tasks.append(
            {
                "task_id": row["task_id"],
                "kind": row["kind"],
                "specification": program["specification"],
                "signature": program["signature"],
                "buggy": row["buggy"],
                "public": [cases[i] for i in row["public_index"]],
                "hidden": [c for i, c in enumerate(cases) if i not in public_index],
                "bug_public_result": row["bug_public_result"],
            }
        )
    return tasks


def _environment(executor):
    import importlib.metadata
    import os
    import platform
    import subprocess

    import torch

    packages = {}
    for name in ("torch", "transformers", "peft", "trl", "tokenizers", "accelerate"):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    try:
        user = subprocess.run(["id"], capture_output=True, text=True, timeout=10).stdout.strip()
    except OSError:
        user = None
    return {
        "python": platform.python_version(),
        "root": os.geteuid() == 0,
        "id": user,
        "cpus": os.cpu_count(),
        "gpu": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "total_gib": torch.cuda.get_device_properties(0).total_memory / 2**30,
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "packages": packages,
        "sandbox": executor.metadata(),
    }


def _canary_search(canary, executor):
    """Can the unprivileged candidate user find payload text on disk? Lists readable hits."""
    import subprocess

    if not executor.root:
        return {"checked": False, "reason": "session is not root; no privilege drop"}
    script = (
        "import os,sys\n"
        "hits=[]\n"
        "for top in ('/kaggle','/tmp','/root','/home','/opt/conda/etc','/etc','/var/tmp'):\n"
        "    for base,dirs,files in os.walk(top):\n"
        "        dirs[:]=[d for d in dirs if d not in ('proc','sys','lib','site-packages')]\n"
        "        for f in files:\n"
        "            p=os.path.join(base,f)\n"
        "            try:\n"
        "                if os.path.getsize(p)>50_000_000: continue\n"
        "                with open(p,'rb') as h: data=h.read()\n"
        "            except Exception: continue\n"
        "            if sys.argv[1].encode() in data: hits.append(p)\n"
        "print('\\n'.join(hits))\n"
    )

    def drop():
        import os

        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)

    try:
        result = subprocess.run(
            ["python3", "-c", script, canary],
            capture_output=True,
            text=True,
            timeout=600,
            preexec_fn=drop,
        )
        hits = [line for line in result.stdout.splitlines() if line]
        return {
            "checked": True,
            "readable_paths_with_payload_text": hits,
            "returncode": result.returncode,
        }
    except Exception as exc:  # noqa: BLE001 - recorded, not fatal
        return {"checked": False, "reason": repr(exc)[:300]}


def _load(model_spec, dtype_name):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[dtype_name]
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec["model"], revision=model_spec["model_revision"], trust_remote_code=False
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = (
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
    return tokenizer, model


def _render(tokenizer, conversation):
    return tokenizer.apply_chat_template(
        conversation, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )


VLLM_SERVER = r"""
import json, sys
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

cfg = json.loads(sys.argv[1])
llm = LLM(
    model=cfg["model"], revision=cfg["revision"], dtype="half", enable_lora=cfg["lora"],
    max_lora_rank=16, max_loras=1, max_model_len=4096, seed=cfg["seed"],
    gpu_memory_utilization=cfg["utilization"], enable_sleep_mode=cfg["sleep"],
    generation_config="vllm",
)
TAG = "@@REPAIRLAB@@"
print(TAG + json.dumps({"ok": True, "ready": True}), flush=True)
for line in sys.stdin:
    request = json.loads(line)
    try:
        op = request["op"]
        if op == "exit":
            break
        if op == "sleep":
            llm.sleep(level=1)
            reply = {"ok": True}
        elif op == "wake":
            llm.wake_up()
            reply = {"ok": True}
        else:
            params = []
            for i, _ in enumerate(request["prompts"]):
                if request["sample"]:
                    params.append(SamplingParams(
                        temperature=1.0, top_p=1.0, max_tokens=request["max_tokens"],
                        seed=request["seed"] + i, stop_token_ids=request["stops"]))
                else:
                    params.append(SamplingParams(
                        temperature=0.0, max_tokens=request["max_tokens"],
                        stop_token_ids=request["stops"]))
            lora = request.get("lora")
            lora = LoRARequest(lora["name"], lora["id"], lora["path"]) if lora else None
            outputs = llm.generate(request["prompts"], params, lora_request=lora, use_tqdm=False)
            reply = {"ok": True, "outputs": [
                {"prompt_ids": list(o.prompt_token_ids), "ids": list(o.outputs[0].token_ids),
                 "finish": o.outputs[0].finish_reason,
                 "stop_reason": o.outputs[0].stop_reason if isinstance(o.outputs[0].stop_reason, int) else None}
                for o in outputs]}
    except Exception as exc:
        reply = {"ok": False, "error": repr(exc)[:2000]}
    print(TAG + json.dumps(reply), flush=True)
"""


def _vllm_python():
    """A separate environment with vLLM (Addendum 3): uv, since the session's venv lacks pip."""
    import subprocess
    import sys
    from pathlib import Path

    env = Path("/tmp/vllm-env")
    python = env / "bin" / "python"
    if not python.exists():
        for command in (
            [sys.executable, "-m", "pip", "install", "--quiet", "uv"],
            [sys.executable, "-m", "uv", "venv", "--quiet", "--python", sys.executable, str(env)],
            [
                sys.executable,
                "-m",
                "uv",
                "pip",
                "install",
                "--quiet",
                "--python",
                str(python),
                "vllm==0.33.0",
            ],
        ):
            subprocess.run(command, check=True, capture_output=True, text=True, timeout=1800)
    return str(python)


class VLLMSampler:
    """A vLLM process in its own environment, driven by JSON lines. Used by ``_generate``."""

    is_vllm = True

    def __init__(self, model_spec, utilization, sleep_mode, lora, seed, log_path):
        import json
        import subprocess

        self.log = open(log_path, "a")
        self.process = subprocess.Popen(
            [
                _vllm_python(),
                "-c",
                VLLM_SERVER,
                json.dumps(
                    {
                        "model": model_spec["model"],
                        "revision": model_spec["model_revision"],
                        "utilization": utilization,
                        "sleep": sleep_mode,
                        "lora": lora,
                        "seed": seed,
                    }
                ),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self.log,
            text=True,
            bufsize=1,
        )
        self.lora = None
        self.asleep = False
        self._read()

    def _read(self):
        import json

        while True:
            line = self.process.stdout.readline()
            if not line:
                raise RuntimeError("vLLM process exited; see its log")
            if line.startswith("@@REPAIRLAB@@"):
                reply = json.loads(line[len("@@REPAIRLAB@@") :])
                if not reply["ok"]:
                    raise RuntimeError("vLLM error: " + reply["error"])
                return reply
            self.log.write(line)

    def call(self, request):
        import json

        self.process.stdin.write(json.dumps(request) + "\n")
        self.process.stdin.flush()
        return self._read()

    def sleep(self):
        if not self.asleep:
            self.call({"op": "sleep"})
            self.asleep = True

    def wake(self):
        if self.asleep:
            self.call({"op": "wake"})
            self.asleep = False

    def use_adapter(self, path, number):
        self.lora = {"name": f"adapter-{number}", "id": number, "path": str(path)}

    def close(self):
        try:
            self.call({"op": "exit"})
        except Exception:  # noqa: BLE001 - shutting down
            pass
        self.process.wait(timeout=120)
        self.log.close()

    def replies(self, tokenizer, conversations, sample, max_new_tokens, seed, keep_ids, stops):
        self.wake()
        texts = [_render(tokenizer, c) for c in conversations]
        outputs = self.call(
            {
                "op": "generate",
                "prompts": texts,
                "sample": sample,
                "max_tokens": max_new_tokens,
                "seed": seed,
                "stops": sorted(stops),
                "lora": self.lora,
            }
        )["outputs"]
        result = []
        for output in outputs:
            ids = list(output["ids"])
            finished = output["finish"] == "stop"
            if finished and (not ids or ids[-1] not in stops):
                ids.append(
                    output["stop_reason"]
                    if output["stop_reason"] in stops
                    else tokenizer.eos_token_id
                )
            reply = {
                "text": tokenizer.decode(ids, skip_special_tokens=True),
                "tokens": len(ids),
                "finished": finished,
            }
            if keep_ids:
                reply["prompt_ids"] = output["prompt_ids"]
                reply["ids"] = ids
            result.append(reply)
        return result


def _stops_for(model_spec, tokenizer):
    from transformers import GenerationConfig

    config = GenerationConfig.from_pretrained(
        model_spec["model"], revision=model_spec["model_revision"]
    )
    ids = config.eos_token_id
    return set(ids if isinstance(ids, list) else [ids]) | {tokenizer.eos_token_id}


def _sampler(payload, tokenizer, output, lora, sleep, utilization):
    sampler = VLLMSampler(
        payload["model"], utilization, sleep, lora, payload["seed"], output / "vllm.log"
    )
    sampler.stops = _stops_for(payload["model"], tokenizer)
    return sampler


def _stop_ids(tokenizer, model):
    configured = model.generation_config.eos_token_id
    ids = set(configured if isinstance(configured, list) else [configured])
    return ids | {tokenizer.eos_token_id}


def _generate(
    tokenizer, model, conversations, sample, batch_size, max_new_tokens, seed, keep_ids=False
):
    """Replies for chat conversations; batches sorted by length to limit padding.

    A reply's tokens run up to and including the first stop token; padding after it is
    dropped. With ``keep_ids`` the exact prompt and reply token ids are kept for training.
    """
    import torch

    if getattr(model, "is_vllm", False):
        return model.replies(
            tokenizer, conversations, sample, max_new_tokens, seed, keep_ids, model.stops
        )
    stops = _stop_ids(tokenizer, model)
    texts = [_render(tokenizer, c) for c in conversations]
    lengths = [len(tokenizer(t)["input_ids"]) for t in texts]
    order = sorted(range(len(texts)), key=lambda i: lengths[i])
    replies = [None] * len(texts)
    dtype = model.get_input_embeddings().weight.dtype
    options = dict(max_new_tokens=max_new_tokens, pad_token_id=tokenizer.pad_token_id)
    if sample:
        options.update(do_sample=True, temperature=1.0, top_p=1.0, top_k=0)
    else:
        options.update(do_sample=False, temperature=None, top_p=None, top_k=None)

    def run(chunk, chunk_seed):
        batch = tokenizer([texts[i] for i in chunk], return_tensors="pt", padding=True).to("cuda:0")
        torch.manual_seed(chunk_seed)
        output, failed = None, False
        try:
            with torch.no_grad(), torch.autocast("cuda", dtype=dtype):
                output = model.generate(**batch, **options)
        except torch.OutOfMemoryError:
            failed = True
        if failed:
            # Outside the handler, so the failed attempt's tensors are already released.
            del batch
            torch.cuda.empty_cache()
            if len(chunk) == 1:
                raise RuntimeError("One sequence does not fit in GPU memory")
            half = len(chunk) // 2
            OOM_SPLITS.append(len(chunk))
            run(chunk[:half], chunk_seed * 2 + 1)
            run(chunk[half:], chunk_seed * 2 + 2)
            return
        width = batch["input_ids"].shape[1]
        for row, index in enumerate(chunk):
            tokens = output[row, width:].tolist()
            end = next((i + 1 for i, t in enumerate(tokens) if t in stops), len(tokens))
            ids = tokens[:end]
            reply = {
                "text": tokenizer.decode(ids, skip_special_tokens=True),
                "tokens": len(ids),
                "finished": end < len(tokens) or (bool(ids) and ids[-1] in stops),
            }
            if keep_ids:
                mask = batch["attention_mask"][row].bool()
                reply["prompt_ids"] = batch["input_ids"][row][mask].tolist()
                reply["ids"] = ids
            replies[index] = reply
        del output, batch
        torch.cuda.empty_cache()

    # Batches are bounded by sequences and by total tokens (prompt plus reply budget), so
    # long retry prompts never share a batch size meant for first prompts.
    chunks, current = [], []
    for index in order:
        cost = lengths[index] + max_new_tokens
        if current and (
            len(current) >= batch_size or (len(current) + 1) * cost > GENERATION_TOKEN_BUDGET
        ):
            chunks.append(current)
            current = []
        current.append(index)
    if current:
        chunks.append(current)
    for number, chunk in enumerate(chunks):
        run(chunk, seed + number)
    return replies


OOM_SPLITS = []  # generation batches halved after running out of memory
# Hugging Face generation: at most this many prompt-plus-reply tokens per batch. Preflight
# part A peaked at 9.7 GiB for 64 first prompts (about 64 x 1,400 tokens).
GENERATION_TOKEN_BUDGET = 90_000
NOTEBOOK_COPIES = ("/kaggle/src/script.ipynb", "/kaggle/working/__notebook__.ipynb")


def _harden():
    """Make the notebook copies (which hold the payload) readable by root only.

    Repeated before every grading batch, since the executor may rewrite its copy.
    """
    import glob
    import os

    if os.geteuid() != 0:
        return []
    done = []
    for path in (
        set(NOTEBOOK_COPIES)
        | set(glob.glob("/kaggle/src/*"))
        | set(glob.glob("/kaggle/working/*.ipynb"))
    ):
        try:
            os.chmod(path, 0o600)
            done.append(path)
        except OSError:
            pass
    return done


def _readable_by_candidate(paths):
    """Which of these paths the unprivileged candidate user can open."""
    import subprocess

    def drop():
        import os

        os.setgroups([])
        os.setgid(65534)
        os.setuid(65534)

    script = (
        "import sys\nfor p in sys.argv[1:]:\n    try:\n        open(p,'rb').read(1)\n"
        "        print(p)\n    except Exception:\n        pass\n"
    )
    result = subprocess.run(
        ["python3", "-c", script, *paths],
        capture_output=True,
        text=True,
        timeout=60,
        preexec_fn=drop,
    )
    return [line for line in result.stdout.splitlines() if line]


def _grade_all(executor, jobs, workers=8):
    """jobs: (task, source or None). Returns grade dicts (None where nothing was submitted)."""
    from concurrent.futures import ThreadPoolExecutor

    from repairlab.rlfirst.episode import grade

    if executor.root:
        _harden()

    def one(job):
        task, source = job
        return None if source is None else grade(executor, task, source)

    with ThreadPoolExecutor(workers) as pool:
        return list(pool.map(one, jobs))


def play(tokenizer, model, executor, tasks, samples, sample, batch_size, seed, log, keep_ids=False):
    """Two-turn episodes for every task x sample. Returns one record per episode."""
    import time

    from repairlab.rlfirst.episode import (
        MAX_NEW_TOKENS,
        first_prompt,
        needs_retry,
        outcome,
        parse,
        retry_prompt,
    )

    episodes = [
        {
            "task": t,
            "sample": s,
            "conversation": [{"role": "user", "content": first_prompt(t, t["bug_public_result"])}],
            "attempts": [],
        }
        for t in tasks
        for s in range(samples)
    ]
    timing = {}
    for turn in (1, 2):
        active = [e for e in episodes if turn == 1 or needs_retry(e["attempts"][-1])]
        if not active:
            break
        if turn == 2:
            for e in active:
                last = e["attempts"][-1]
                public_result = last["grade"]["public_result"] if last["grade"] else {}
                e["conversation"] = e["conversation"] + [
                    {"role": "assistant", "content": last["reply"]},
                    {
                        "role": "user",
                        "content": retry_prompt(last["parsed"], public_result, e["task"]),
                    },
                ]
        began = time.monotonic()
        replies = _generate(
            tokenizer,
            model,
            [e["conversation"] for e in active],
            sample,
            batch_size,
            MAX_NEW_TOKENS,
            seed + 1000 * turn,
            keep_ids,
        )
        generated = time.monotonic() - began
        parsed = [parse(r["text"], e["task"]["signature"]) for r, e in zip(replies, active)]
        began = time.monotonic()
        grades = _grade_all(executor, [(e["task"], p["source"]) for e, p in zip(active, parsed)])
        graded = time.monotonic() - began
        for e, r, p, g in zip(active, replies, parsed, grades):
            e["attempts"].append(
                {
                    "reply": r["text"],
                    "tokens": r["tokens"],
                    "finished": r["finished"],
                    "parsed": p,
                    "grade": g,
                    "prompt_ids": r.get("prompt_ids"),
                    "ids": r.get("ids"),
                }
            )
        timing[f"turn{turn}"] = {
            "episodes": len(active),
            "generate_seconds": round(generated, 1),
            "grade_seconds": round(graded, 1),
            "new_tokens": sum(r["tokens"] for r in replies),
        }
        log(f"turn {turn}: {timing[f'turn{turn}']}")
    records = []
    for e in episodes:
        records.append(
            {
                "task_id": e["task"]["task_id"],
                "kind": e["task"]["kind"],
                "sample": e["sample"],
                "attempts": [
                    {
                        "reply": a["reply"],
                        "tokens": a["tokens"],
                        "finished": a["finished"],
                        "valid": a["parsed"]["source"] is not None,
                        "reason": a["parsed"]["reason"],
                        "status": a["grade"]["status"] if a["grade"] else None,
                        "public_passed": bool(a["grade"] and a["grade"]["public_passed"]),
                        "hidden_passed": bool(a["grade"] and a["grade"]["hidden_passed"]),
                        "public_result": a["grade"]["public_result"] if a["grade"] else None,
                        **({"prompt_ids": a["prompt_ids"], "ids": a["ids"]} if keep_ids else {}),
                    }
                    for a in e["attempts"]
                ],
                **outcome(e["attempts"]),
            }
        )
    return records, timing


def _summary(records):
    n = len(records) or 1
    return {
        "episodes": len(records),
        "hardened": sum(r["hardened_reward"] for r in records),
        "weak": sum(r["weak_reward"] for r in records),
        "false_pass": sum(r["false_pass"] for r in records),
        "valid_first": sum(r["valid_first"] for r in records),
        "valid_first_rate": sum(r["valid_first"] for r in records) / n,
        "public_pass_first": sum(r["attempts"][0]["public_passed"] for r in records),
        "retry_identical": sum(r["retry_identical"] for r in records),
    }


def _vllm_check(payload, tasks, output, tokenizer, log):
    """Install vLLM (session if it keeps torch, else a separate venv) and measure it."""
    import json
    import subprocess
    import sys
    import time
    from pathlib import Path

    result = {}
    began = time.monotonic()
    report = Path("/tmp/vllm-dry-run.json")
    dry = subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--dry-run",
            "--quiet",
            "--report",
            str(report),
            "vllm",
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    replaces_torch = None
    if dry.returncode == 0 and report.exists():
        plan = json.loads(report.read_text())
        replaces_torch = any(
            item["metadata"]["name"].lower() == "torch" for item in plan.get("install", [])
        )
    result["dry_run_ok"] = dry.returncode == 0
    result["would_replace_session_torch"] = replaces_torch
    python = sys.executable
    install_command = [python, "-m", "pip", "install", "--quiet", "vllm"]
    if replaces_torch is not False:
        # The session's venv module lacks ensurepip; uv makes the separate environment.
        env = Path("/tmp/vllm-env")
        steps = [
            [sys.executable, "-m", "pip", "install", "--quiet", "uv"],
            [sys.executable, "-m", "uv", "venv", "--quiet", "--python", sys.executable, str(env)],
        ]
        for command in steps:
            made = subprocess.run(command, capture_output=True, text=True, timeout=900)
            if made.returncode:
                result["venv_error"] = (made.stdout + made.stderr)[-800:]
                return result
        python = str(env / "bin" / "python")
        install_command = [
            sys.executable,
            "-m",
            "uv",
            "pip",
            "install",
            "--quiet",
            "--python",
            python,
            "vllm",
        ]
        result["environment"] = "separate_venv"
    else:
        result["environment"] = "session"
    install = subprocess.run(install_command, capture_output=True, text=True, timeout=1800)
    result["install_ok"] = install.returncode == 0
    result["install_seconds"] = round(time.monotonic() - began)
    if install.returncode:
        result["install_error"] = (install.stdout + install.stderr)[-1500:]
        return result
    version = subprocess.run(
        [python, "-c", "import vllm,torch;print(vllm.__version__, torch.__version__)"],
        capture_output=True,
        text=True,
    )
    result["versions"] = version.stdout.strip() or version.stderr[-500:]
    prompts = [_render(tokenizer, [{"role": "user", "content": t["first_prompt"]}]) for t in tasks]
    job = {
        "model": payload["model"]["model"],
        "revision": payload["model"]["model_revision"],
        "adapter": str(Path(output / "fresh-adapter").resolve()),
        "prompts": prompts,
        "samples": payload["speed"]["samples_per_prompt"],
        "max_tokens": 768,
    }
    (Path("/tmp") / "vllm-job.json").write_text(json.dumps(job))
    script = r"""
import json, time
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
job = json.load(open("/tmp/vllm-job.json"))
out = {}
t = time.monotonic()
llm = LLM(model=job["model"], revision=job["revision"], dtype="half", enable_lora=True,
          max_lora_rank=16, max_model_len=4096, gpu_memory_utilization=0.85, seed=7)
out["load_seconds"] = time.monotonic() - t
lora = LoRARequest("fresh", 1, job["adapter"])
t = time.monotonic()
greedy = llm.generate(job["prompts"], SamplingParams(temperature=0.0, max_tokens=job["max_tokens"]), lora_request=lora)
out["greedy_seconds"] = time.monotonic() - t
out["greedy"] = [g.outputs[0].text for g in greedy]
t = time.monotonic()
sampled = llm.generate(job["prompts"], SamplingParams(n=job["samples"], temperature=1.0, top_p=1.0, max_tokens=job["max_tokens"], seed=11), lora_request=lora)
out["sample_seconds"] = time.monotonic() - t
out["sample_tokens"] = sum(len(o.token_ids) for g in sampled for o in g.outputs)
out["sample_sequences"] = sum(len(g.outputs) for g in sampled)
json.dump(out, open("/tmp/vllm-out.json", "w"))
"""
    began = time.monotonic()
    run = subprocess.run([python, "-c", script], capture_output=True, text=True, timeout=3600)
    result["run_ok"] = run.returncode == 0
    result["run_seconds"] = round(time.monotonic() - began)
    if run.returncode:
        result["run_error"] = (run.stdout + run.stderr)[-3000:]
        return result
    measured = json.loads(Path("/tmp/vllm-out.json").read_text())
    result.update({k: v for k, v in measured.items() if k != "greedy"})
    result["greedy_replies"] = measured["greedy"]
    log(f"vllm: {json.dumps({k: v for k, v in result.items() if k != 'greedy_replies'})[:800]}")
    return result


def run_preflight(payload):
    import gc
    import json
    import shutil
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    import torch

    output = Path(
        "repairlab-rlfirst-preflight-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(exist_ok=False)
    _install_code(payload)
    from repairlab.rlfirst.episode import first_prompt, parse
    from repairlab.rlfirst.sandbox import LocalExecutor

    results = {"kind": "rlfirst_preflight_part_a_not_a_result"}

    def save():
        (output / "preflight.json").write_text(json.dumps(results, indent=2) + "\n")

    def log(line):
        print(line, flush=True)

    executor = LocalExecutor()
    results["environment"] = _environment(executor)
    save()
    tasks = _tasks(payload)
    for t in tasks:
        t["first_prompt"] = first_prompt(t, t["bug_public_result"])

    # In-session controls: the bug fails a public case; a public-only shortcut is not needed here.
    began = time.monotonic()
    controls = _grade_all(executor, [(t, t["buggy"]) for t in tasks])
    results["in_session_controls"] = {
        "bug_graded": sum(g["status"] == "graded" for g in controls),
        "bug_fails_public": sum(not g["public_passed"] for g in controls),
        "tasks": len(tasks),
        "seconds": round(time.monotonic() - began, 2),
    }
    references = _grade_all(executor, [(t, payload["references"][t["task_id"]]) for t in tasks])
    results["in_session_controls"]["reference_passes"] = sum(
        g["public_passed"] and g["hidden_passed"] for g in references
    )
    save()
    results["canary"] = _canary_search(payload["canary"], executor)
    save()
    log(json.dumps(results["in_session_controls"]))

    for dtype in ("bf16", "fp16"):
        tokenizer, model = _load(payload["model"], dtype)
        row = {"dtype": dtype}
        texts = [
            _render(tokenizer, [{"role": "user", "content": t["first_prompt"]}]) for t in tasks
        ]
        finite = True
        with torch.no_grad():
            for start in range(0, len(texts), 8):
                batch = tokenizer(texts[start : start + 8], return_tensors="pt", padding=True).to(
                    "cuda:0"
                )
                logits = model(**batch).logits
                mask = batch["attention_mask"].bool()
                finite = finite and bool(torch.isfinite(logits[mask]).all())
        row["all_logits_finite"] = finite
        torch.cuda.reset_peak_memory_stats(0)
        began = time.monotonic()
        replies = _generate(
            tokenizer,
            model,
            [[{"role": "user", "content": t["first_prompt"]}] for t in tasks],
            False,
            len(tasks),
            768,
            7,
        )
        row["greedy_seconds"] = round(time.monotonic() - began, 1)
        row["greedy_peak_gib"] = torch.cuda.max_memory_allocated(0) / 2**30
        parsed = [parse(r["text"], t["signature"]) for r, t in zip(replies, tasks)]
        grades = _grade_all(executor, [(t, p["source"]) for t, p in zip(tasks, parsed)])
        row["valid_replies"] = sum(p["source"] is not None for p in parsed)
        row["public_passes"] = sum(bool(g and g["public_passed"]) for g in grades)
        row["hardened_passes"] = sum(
            bool(g and g["public_passed"] and g["hidden_passed"]) for g in grades
        )
        row["new_tokens"] = sum(r["tokens"] for r in replies)
        row["replies"] = [r["text"] for r in replies]
        results.setdefault("dtype_check", []).append(row)
        save()
        log(json.dumps({k: v for k, v in row.items() if k != "replies"}))
        if dtype == "fp16":
            speed = payload["speed"]
            for batch_size in speed["batch_sizes"]:
                torch.cuda.reset_peak_memory_stats(0)
                chosen = tasks[: max(1, batch_size // speed["samples_per_prompt"])]
                conversations = [
                    [{"role": "user", "content": t["first_prompt"]}]
                    for t in chosen
                    for _ in range(speed["samples_per_prompt"])
                ][:batch_size]
                began = time.monotonic()
                try:
                    out = _generate(tokenizer, model, conversations, True, batch_size, 768, 11)
                    ok, error = True, None
                except torch.OutOfMemoryError as exc:
                    out, ok, error = [], False, str(exc)[:200]
                seconds = time.monotonic() - began
                measured = {
                    "batch_size": batch_size,
                    "ok": ok,
                    "error": error,
                    "seconds": round(seconds, 1),
                    "sequences": len(out),
                    "new_tokens": sum(r["tokens"] for r in out),
                    "peak_gib": torch.cuda.max_memory_allocated(0) / 2**30,
                }
                measured["sequences_per_hour"] = round(len(out) / seconds * 3600) if ok else 0
                results.setdefault("hf_sampling", []).append(measured)
                save()
                log(json.dumps(measured))
            # A fresh rank-16 adapter on every projection, for vLLM's LoRA path.
            from peft import LoraConfig, get_peft_model

            peft_model = get_peft_model(
                model,
                LoraConfig(
                    r=16,
                    lora_alpha=32,
                    target_modules=[
                        "q_proj",
                        "k_proj",
                        "v_proj",
                        "o_proj",
                        "gate_proj",
                        "up_proj",
                        "down_proj",
                    ],
                    lora_dropout=0.0,
                    task_type="CAUSAL_LM",
                ),
            )
            peft_model.save_pretrained(output / "fresh-adapter")
            del peft_model
        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Grading speed on the greedy replies, one at a time and with eight workers.
    replies = results["dtype_check"][-1]["replies"]
    jobs = [(t, parse(r, t["signature"])["source"] or t["buggy"]) for t, r in zip(tasks, replies)]
    began = time.monotonic()
    for job in jobs[:16]:
        _grade_all(executor, [job], workers=1)
    single = (time.monotonic() - began) / 16
    began = time.monotonic()
    _grade_all(executor, jobs * 4, workers=8)
    parallel = (time.monotonic() - began) / (len(jobs) * 4)
    results["grading_speed"] = {
        "seconds_each_single": round(single, 3),
        "seconds_each_parallel8": round(parallel, 3),
    }
    save()

    try:
        results["vllm"] = _vllm_check(payload, tasks, output, tokenizer, log)
        if results["vllm"].get("greedy_replies"):
            parsed = [
                parse(r, t["signature"]) for r, t in zip(results["vllm"]["greedy_replies"], tasks)
            ]
            grades = _grade_all(executor, [(t, p["source"]) for t, p in zip(tasks, parsed)])
            results["vllm"]["valid_replies"] = sum(p["source"] is not None for p in parsed)
            results["vllm"]["public_passes"] = sum(bool(g and g["public_passed"]) for g in grades)
    except Exception as exc:  # noqa: BLE001 - vLLM is optional; record and continue
        results["vllm"] = {"error": repr(exc)[:1500]}
    save()
    shutil.rmtree(output / "fresh-adapter", ignore_errors=True)
    archive = shutil.make_archive(str(output), "zip", output)
    print("PREFLIGHT COMPLETE", "Archive:", archive, flush=True)


def run_sweep(payload):
    """Base sweep (addendum 1): the untrained model, 8 sampled episodes per training task."""
    import json
    import shutil
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    output = Path(
        "repairlab-rlfirst-sweep-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(exist_ok=False)
    _install_code(payload)
    from repairlab.rlfirst.sandbox import LocalExecutor

    executor = LocalExecutor()
    meta = {
        "kind": "rlfirst_base_sweep",
        "shard": payload["shard"],
        "environment": _environment(executor),
    }
    tasks = _tasks(payload)
    if payload.get("sampler") == "vllm":
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            payload["model"]["model"], revision=payload["model"]["model_revision"]
        )
        model = _sampler(payload, tokenizer, output, lora=False, sleep=False, utilization=0.85)
    else:
        tokenizer, model = _load(payload["model"], payload["dtype"])
    meta["sampler"] = payload.get("sampler", "hf")
    began = time.monotonic()
    records = []
    chunk = payload["tasks_per_chunk"]
    for start in range(0, len(tasks), chunk):
        part = tasks[start : start + chunk]
        rows, timing = play(
            tokenizer,
            model,
            executor,
            part,
            payload["samples"],
            True,
            payload["batch_size"],
            payload["seed"] + start * 10007,
            lambda line: print(line, flush=True),
        )
        records.extend(rows)
        meta.setdefault("timing", []).append({"tasks": len(part), **timing})
        print(
            f"tasks {start + len(part)}/{len(tasks)} after {time.monotonic() - began:.0f}s",
            flush=True,
        )
        (output / "sweep.json").write_text(json.dumps({"meta": meta, "records": records}) + "\n")
    meta["seconds"] = round(time.monotonic() - began)
    if getattr(model, "is_vllm", False):
        model.close()
    meta["summary"] = _summary(records)
    meta["oom_splits"] = list(OOM_SPLITS)
    (output / "sweep.json").write_text(json.dumps({"meta": meta, "records": records}) + "\n")
    archive = shutil.make_archive(str(output), "zip", output)
    print("SWEEP COMPLETE", json.dumps(meta["summary"]), "Archive:", archive, flush=True)


LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def _policy(model, adapter_dir=None):
    """Rank-16 LoRA on every attention and MLP projection; adapter weights kept in fp32."""
    from peft import LoraConfig, PeftModel, get_peft_model

    if adapter_dir is not None:
        policy = PeftModel.from_pretrained(model, adapter_dir, is_trainable=True)
    else:
        policy = get_peft_model(
            model,
            LoraConfig(
                r=16,
                lora_alpha=32,
                lora_dropout=0.0,
                target_modules=LORA_TARGETS,
                task_type="CAUSAL_LM",
            ),
        )
    for parameter in policy.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    # PEFT would otherwise cast every activation to the adapter's fp32, which ran out of
    # memory at batch 64 (part B, first attempt). Forwards run under autocast instead.
    from peft.tuners.tuners_utils import BaseTunerLayer

    for module in policy.modules():
        if isinstance(module, BaseTunerLayer):
            module.cast_input_dtype_enabled = False
    inner = policy.base_model.model
    inner.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    inner.config.use_cache = True
    return policy


def _token_logprobs(policy, items, dtype):
    """Summed log-probabilities of each item's reply tokens given its prompt (one micro-batch)."""
    import torch
    import torch.nn.functional as F

    inner = policy.base_model.model
    width = max(len(p) + len(r) for p, r, _ in items)
    pad = 0
    input_ids = torch.full((len(items), width), pad, dtype=torch.long)
    attention = torch.zeros((len(items), width), dtype=torch.long)
    for row, (prompt, reply, _) in enumerate(items):
        sequence = prompt + reply
        input_ids[row, : len(sequence)] = torch.tensor(sequence)
        attention[row, : len(sequence)] = 1
    input_ids, attention = input_ids.to("cuda:0"), attention.to("cuda:0")
    with torch.autocast("cuda", dtype=dtype):
        hidden = inner.model(input_ids=input_ids, attention_mask=attention, use_cache=False)
        hidden = hidden.last_hidden_state
    sums = []
    for row, (prompt, reply, _) in enumerate(items):
        # Position t predicts token t + 1: reply tokens sit at len(prompt) .. end.
        start = len(prompt) - 1
        states = hidden[row, start : start + len(reply)]
        targets = torch.tensor(reply, device="cuda:0")
        total = 0.0
        for begin in range(0, len(reply), 256):
            with torch.autocast("cuda", dtype=dtype):
                logits = inner.lm_head(states[begin : begin + 256])
            total = total - F.cross_entropy(
                logits.float(), targets[begin : begin + 256], reduction="sum"
            )
        sums.append(total)
    return torch.stack(sums)


def _accumulate(policy, scaler, batch, total_tokens, dtype, trainable, splits):
    """Backward one micro-batch; if it runs out of memory, restore the gradients and halve it.

    The loss is a sum over items divided by the step's total reply tokens, so splitting a
    micro-batch leaves the step's gradient unchanged. Gradients are restored from a copy
    because an out-of-memory error during backward can leave some of them half-added.
    (Added after the weak arm's first chunk ran out of memory in step 13; Addendum 7.)
    """
    import torch

    saved = [None if p.grad is None else p.grad.detach().to("cpu", copy=True) for p in trainable]
    logp = loss = advantages = None
    try:
        logp = _token_logprobs(policy, batch, dtype)
        advantages = torch.tensor([a for _, _, a in batch], device=logp.device, dtype=torch.float32)
        loss = -(advantages * logp).sum() / total_tokens
        scaler.scale(loss).backward()
        return float(loss.detach())
    except torch.OutOfMemoryError:
        if len(batch) == 1:
            raise
    logp = loss = advantages = None
    for p, g in zip(trainable, saved):
        p.grad = None if g is None else g.to(p.device)
    saved = None
    torch.cuda.empty_cache()
    splits.append(len(batch))
    half = len(batch) // 2
    first = _accumulate(policy, scaler, batch[:half], total_tokens, dtype, trainable, splits)
    return first + _accumulate(policy, scaler, batch[half:], total_tokens, dtype, trainable, splits)


def _update(policy, optimizer, scaler, items, settings, dtype):
    """One on-policy step: loss = -sum(A * logp(reply tokens)) / total reply tokens."""
    import torch

    policy.train()
    optimizer.zero_grad(set_to_none=True)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    total_tokens = sum(len(r) for _, r, _ in items)
    order = sorted(items, key=lambda x: len(x[0]) + len(x[1]))
    batches, current = [], []
    for item in order:
        length = len(item[0]) + len(item[1])
        if current and (len(current) + 1) * length > settings["micro_batch_tokens"]:
            batches.append(current)
            current = []
        current.append(item)
    if current:
        batches.append(current)
    loss_value, splits = 0.0, []
    for batch in batches:
        loss_value += _accumulate(policy, scaler, batch, total_tokens, dtype, trainable, splits)
    scaler.unscale_(optimizer)
    norm = torch.nn.utils.clip_grad_norm_(trainable, settings["clip"])
    finite = bool(torch.isfinite(norm))
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)
    policy.eval()
    return {
        "loss": loss_value,
        "grad_norm": float(norm) if finite else None,
        "finite_gradients": finite,
        "micro_batches": len(batches),
        "oom_splits": splits,
        "trained_tokens": total_tokens,
        "scale": float(scaler.get_scale()) if scaler.is_enabled() else None,
    }


def run_rl(payload):
    """One chunk (at most ``chunk_seconds``) of RL from the base model or a saved state."""
    import gzip
    import json
    import random
    import shutil
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    import torch

    began = time.monotonic()
    settings = payload["rl"]
    arm = payload["arm"]
    reward_key = {"hardened": "hardened_reward", "weak": "weak_reward"}[arm]
    output = Path(
        f"repairlab-rlfirst-rl-{arm}-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    (output / "steps").mkdir(parents=True, exist_ok=False)
    _install_code(payload)
    from repairlab.rlfirst.sandbox import LocalExecutor

    executor = LocalExecutor()
    tasks = _tasks(payload)
    by_id = {t["task_id"]: t for t in tasks}
    tokenizer, base = _load(payload["model"], payload["dtype"])
    dtype = torch.float16 if payload["dtype"] == "fp16" else torch.bfloat16
    resume = None
    if payload.get("resume_glob"):
        found = sorted(Path("/kaggle/input").glob(payload["resume_glob"]))
        if len(found) != 1:
            raise RuntimeError(f"Expected one resume state, found {len(found)}")
        resume = found[0]
    policy = _policy(base, str(resume / "adapter") if resume else None)
    policy.eval()
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=settings["lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=payload["dtype"] == "fp16")
    if resume:
        saved = torch.load(resume / "optimizer.pt", map_location="cuda:0", weights_only=False)
        optimizer.load_state_dict(saved["optimizer"])
        scaler.load_state_dict(saved["scaler"])
        state = json.loads((resume / "state.json").read_text())
        rng = random.Random()
        rng.setstate(tuple(tuple(x) if isinstance(x, list) else x for x in state["python_rng"]))
        torch.set_rng_state(torch.load(resume / "rng.pt", weights_only=False)["cpu"])
        torch.cuda.set_rng_state(torch.load(resume / "rng.pt", weights_only=False)["cuda"])
    else:
        rng = random.Random(payload["seed"])
        torch.manual_seed(payload["seed"])
        state = {
            "step": 0,
            "order": [],
            "pointer": 0,
            "retired": [],
            "collapse_streak": 0,
            "collapsed": False,
            "epoch": 0,
        }
    retired = set(state["retired"])
    sampler = None
    if payload.get("sampler") == "vllm":
        torch.cuda.empty_cache()
        sampler = _sampler(
            payload,
            tokenizer,
            output,
            lora=True,
            sleep=True,
            utilization=settings["vllm_utilization"],
        )
    adapters = Path("/tmp/repairlab-adapters")

    def publish():
        """The current adapter, saved where the sampler reads it (id = steps taken + 1)."""
        if sampler is not None:
            path = adapters / f"step-{state['step']}"
            policy.save_pretrained(path)
            sampler.use_adapter(path, state["step"] + 1)

    publish()

    def draw(count):
        picked = []
        while len(picked) < count:
            if state["pointer"] >= len(state["order"]):
                state["order"] = [t["task_id"] for t in tasks if t["task_id"] not in retired]
                rng.shuffle(state["order"])
                state["pointer"] = 0
                state["epoch"] += 1
                if not state["order"]:
                    break
            task_id = state["order"][state["pointer"]]
            state["pointer"] += 1
            if task_id not in retired:
                picked.append(by_id[task_id])
        return picked

    log = {"arm": arm, "environment": _environment(executor), "steps": []}
    step_seconds = []
    stop_reason = "chunk_time"
    while state["step"] < settings["steps_total"]:
        estimate = max(step_seconds) if step_seconds else settings["first_step_estimate"]
        if time.monotonic() - began + estimate > settings["chunk_seconds"]:
            break
        step_began = time.monotonic()
        torch.cuda.reset_peak_memory_stats(0)
        groups, dropped, draws, episodes = [], [], 0, []
        while len(groups) < settings["groups"] and draws < settings["max_draws"]:
            part = draw(min(settings["draw_chunk"], settings["max_draws"] - draws))
            if not part:
                break
            draws += len(part)
            rows, _ = play(
                tokenizer,
                sampler or policy,
                executor,
                part,
                settings["samples"],
                True,
                payload["batch_size"],
                payload["seed"] * 7919 + state["step"] * 1_000_003 + draws * 10_007,
                lambda line: None,
                keep_ids=True,
            )
            episodes.extend(rows)
            for task in part:
                group = [r for r in rows if r["task_id"] == task["task_id"]]
                rewards = [r[reward_key] for r in group]
                if len(set(rewards)) == 1:
                    dropped.append(task["task_id"])
                    if rewards[0] == 1:
                        retired.add(task["task_id"])
                elif len(groups) < settings["groups"]:
                    groups.append(group)
                else:
                    dropped.append(task["task_id"])
        items, skipped_long = [], 0
        for group in groups:
            mean = sum(r[reward_key] for r in group) / len(group)
            for record in group:
                record["advantage"] = record[reward_key] - mean
                for attempt in record["attempts"]:
                    if (
                        len(attempt["prompt_ids"]) + len(attempt["ids"])
                        > settings["max_sequence_tokens"]
                    ):
                        skipped_long += 1
                        continue
                    items.append((attempt["prompt_ids"], attempt["ids"], record["advantage"]))
        if sampler is not None:
            sampler.sleep()
            torch.cuda.empty_cache()
        update = _update(policy, optimizer, scaler, items, settings, dtype) if items else None
        attempts = [a for r in episodes for a in r["attempts"]]
        valid_rate = sum(a["valid"] for a in attempts) / (len(attempts) or 1)
        state["collapse_streak"] = (
            state["collapse_streak"] + 1 if valid_rate < settings["collapse_threshold"] else 0
        )
        state["step"] += 1
        torch.cuda.empty_cache()
        publish()
        trained_ids = {id(r) for g in groups for r in g}
        for record in episodes:
            if id(record) not in trained_ids:
                for attempt in record["attempts"]:
                    attempt.pop("prompt_ids", None)
                    attempt.pop("ids", None)
        row = {
            "step": state["step"],
            "draws": draws,
            "informative_groups": len(groups),
            "dropped_groups": len(dropped),
            "retired_total": len(retired),
            "episodes": len(episodes),
            "valid_reply_rate": valid_rate,
            "mean_reward": sum(r[reward_key] for r in episodes) / (len(episodes) or 1),
            "hardened_rate": sum(r["hardened_reward"] for r in episodes) / (len(episodes) or 1),
            "false_passes": sum(r["false_pass"] for r in episodes),
            "skipped_long_turns": skipped_long,
            "update": update,
            "seconds": round(time.monotonic() - step_began, 1),
            "peak_gib": torch.cuda.max_memory_allocated(0) / 2**30,
        }
        step_seconds.append(row["seconds"])
        log["steps"].append(row)
        with gzip.open(output / "steps" / f"step-{state['step']:04d}.json.gz", "wt") as handle:
            json.dump(
                {
                    "summary": row,
                    "groups": [[r["task_id"] for r in g][:1] for g in groups],
                    "episodes": episodes,
                },
                handle,
            )
        print(json.dumps(row), flush=True)
        if state["collapse_streak"] >= 3:
            state["collapsed"] = True
            stop_reason = "collapsed"
            break
    else:
        stop_reason = "steps_complete"
    if sampler is not None:
        sampler.close()
    state["retired"] = sorted(retired)
    state["python_rng"] = rng.getstate()
    keep = output / "state"
    keep.mkdir()
    policy.save_pretrained(keep / "adapter")
    torch.save(
        {"optimizer": optimizer.state_dict(), "scaler": scaler.state_dict()}, keep / "optimizer.pt"
    )
    torch.save({"cpu": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state()}, keep / "rng.pt")
    (keep / "state.json").write_text(json.dumps(state) + "\n")
    log["oom_splits"] = list(OOM_SPLITS)
    log.update(
        stop_reason=stop_reason, seconds=round(time.monotonic() - began), state_step=state["step"]
    )
    (output / "log.json").write_text(json.dumps(log, indent=2) + "\n")
    archive = shutil.make_archive(str(output), "zip", output)
    print("RL CHUNK COMPLETE", stop_reason, state["step"], "Archive:", archive, flush=True)


def run_part_b(payload):
    """Preflight item 4: one full RL step at batch size, timing sampling and updates."""
    import json
    import random
    import shutil
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    import torch

    output = Path(
        "repairlab-rlfirst-partb-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(exist_ok=False)
    _install_code(payload)
    from repairlab.rlfirst.sandbox import LocalExecutor

    settings = payload["rl"]
    executor = LocalExecutor()
    tasks = _tasks(payload)
    tokenizer, base = _load(payload["model"], payload["dtype"])
    dtype = torch.float16 if payload["dtype"] == "fp16" else torch.bfloat16
    policy = _policy(base)
    policy.eval()
    sampler = None
    if payload.get("sampler") == "vllm":
        torch.cuda.empty_cache()
        sampler = _sampler(
            payload,
            tokenizer,
            output,
            lora=True,
            sleep=True,
            utilization=settings["vllm_utilization"],
        )
        policy.save_pretrained("/tmp/repairlab-adapters/step-0")
        sampler.use_adapter("/tmp/repairlab-adapters/step-0", 1)
    results = {"environment": _environment(executor), "dtype": payload["dtype"]}
    rng = random.Random(payload["seed"])
    rng.shuffle(tasks)
    began = time.monotonic()
    torch.cuda.reset_peak_memory_stats(0)
    groups, draws, episodes = [], 0, []
    while len(groups) < settings["groups"] and draws < settings["max_draws"]:
        part = tasks[draws : draws + settings["draw_chunk"]]
        draws += len(part)
        rows, timing = play(
            tokenizer,
            sampler or policy,
            executor,
            part,
            settings["samples"],
            True,
            payload["batch_size"],
            payload["seed"] + draws,
            lambda line: None,
            keep_ids=True,
        )
        episodes.extend(rows)
        for task in part:
            group = [r for r in rows if r["task_id"] == task["task_id"]]
            if len({r["hardened_reward"] for r in group}) > 1 and len(groups) < settings["groups"]:
                groups.append(group)
        print(json.dumps({"draws": draws, "groups": len(groups), **timing}), flush=True)
    results["sampling"] = {
        "seconds": round(time.monotonic() - began, 1),
        "draws": draws,
        "informative_groups": len(groups),
        "episodes": len(episodes),
        "peak_gib": torch.cuda.max_memory_allocated(0) / 2**30,
    }
    if sampler is not None:
        sampler.sleep()
        torch.cuda.empty_cache()
    items = []
    for group in groups:
        mean = sum(r["hardened_reward"] for r in group) / len(group)
        for record in group:
            for attempt in record["attempts"]:
                if (
                    len(attempt["prompt_ids"]) + len(attempt["ids"])
                    <= settings["max_sequence_tokens"]
                ):
                    items.append(
                        (attempt["prompt_ids"], attempt["ids"], record["hardened_reward"] - mean)
                    )
    results["update_items"] = len(items)
    results["update_tokens"] = sum(len(p) + len(r) for p, r, _ in items)
    trainable = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=settings["lr"], weight_decay=0.0)
    scaler = torch.amp.GradScaler("cuda", enabled=payload["dtype"] == "fp16")
    results["updates"] = []
    for budget in payload["micro_budgets"]:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(0)
        started = time.monotonic()
        row = {"micro_batch_tokens": budget}
        try:
            row.update(
                _update(
                    policy,
                    optimizer,
                    scaler,
                    items,
                    {**settings, "micro_batch_tokens": budget},
                    dtype,
                )
            )
            row["ok"] = True
        except torch.OutOfMemoryError as exc:
            row.update(ok=False, error=str(exc)[:200])
            optimizer.zero_grad(set_to_none=True)
        row["seconds"] = round(time.monotonic() - started, 1)
        row["peak_gib"] = torch.cuda.max_memory_allocated(0) / 2**30
        results["updates"].append(row)
        print(json.dumps(row), flush=True)
        (output / "partb.json").write_text(json.dumps(results, indent=2) + "\n")
    if sampler is not None:
        # The updated adapter must reach the sampler: save, swap, wake, and sample again.
        torch.cuda.empty_cache()
        policy.save_pretrained("/tmp/repairlab-adapters/step-1")
        sampler.use_adapter("/tmp/repairlab-adapters/step-1", 2)
        started = time.monotonic()
        rows, timing = play(
            tokenizer,
            sampler,
            executor,
            tasks[draws : draws + 4],
            settings["samples"],
            True,
            payload["batch_size"],
            payload["seed"] + 999_983,
            lambda line: None,
            keep_ids=True,
        )
        attempts = [a for r in rows for a in r["attempts"]]
        results["after_update_sampling"] = {
            "seconds": round(time.monotonic() - started, 1),
            "episodes": len(rows),
            "valid_reply_rate": sum(a["valid"] for a in attempts) / (len(attempts) or 1),
            "hardened_rate": sum(r["hardened_reward"] for r in rows) / (len(rows) or 1),
        }
        sampler.close()
    results["sampler"] = payload.get("sampler", "hf")
    results["oom_splits"] = list(OOM_SPLITS)
    (output / "partb.json").write_text(json.dumps(results, indent=2) + "\n")
    archive = shutil.make_archive(str(output), "zip", output)
    print("PART B COMPLETE", "Archive:", archive, flush=True)


def run_eval(payload):
    """Greedy two-turn episodes on evaluation tasks. Only public cases are in this session.

    Hidden grading happens on the laptop in Docker; the in-session hidden result is empty
    by construction and is never used.
    """
    import json
    import shutil
    import time
    from datetime import datetime, timezone
    from pathlib import Path

    output = Path(
        f"repairlab-rlfirst-eval-{payload['arm']}-"
        + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(exist_ok=False)
    _install_code(payload)
    from repairlab.rlfirst.sandbox import LocalExecutor

    executor = LocalExecutor()
    tasks = payload["tasks"]
    for task in tasks:
        if task["hidden"]:
            raise RuntimeError("Hidden cases must never reach an evaluation session")
    tokenizer, model = _load(payload["model"], payload["dtype"])
    if payload.get("adapter_glob"):
        from peft import PeftModel

        found = sorted(Path("/kaggle/input").glob(payload["adapter_glob"]))
        if len(found) != 1:
            raise RuntimeError(f"Expected one adapter, found {len(found)}")
        model = PeftModel.from_pretrained(model, str(found[0])).eval()
    began = time.monotonic()
    records, timing = play(
        tokenizer,
        model,
        executor,
        tasks,
        1,
        False,
        payload["batch_size"],
        0,
        lambda line: print(line, flush=True),
    )
    meta = {
        "arm": payload["arm"],
        "environment": _environment(executor),
        "timing": timing,
        "seconds": round(time.monotonic() - began),
        "oom_splits": list(OOM_SPLITS),
    }
    (output / "eval.json").write_text(json.dumps({"meta": meta, "records": records}) + "\n")
    archive = shutil.make_archive(str(output), "zip", output)
    print("EVAL COMPLETE", payload["arm"], len(records), "Archive:", archive, flush=True)


def run_vllm_check(payload):
    """Addendum 3: harden the notebook copies, re-check the canary, and retry vLLM."""
    import gc
    import json
    import shutil
    from datetime import datetime, timezone
    from pathlib import Path

    import torch

    output = Path(
        "repairlab-rlfirst-vllm-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    )
    output.mkdir(exist_ok=False)
    _install_code(payload)
    from repairlab.rlfirst.episode import first_prompt, parse
    from repairlab.rlfirst.sandbox import LocalExecutor

    executor = LocalExecutor()
    results = {"kind": "rlfirst_vllm_check_not_a_result", "environment": _environment(executor)}
    results["hardened_paths"] = _harden()
    results["readable_after_hardening"] = _readable_by_candidate(list(NOTEBOOK_COPIES))
    results["canary_after_hardening"] = _canary_search(payload["canary"], executor)
    tasks = _tasks(payload)
    for t in tasks:
        t["first_prompt"] = first_prompt(t, t["bug_public_result"])
    tokenizer, model = _load(payload["model"], "fp16")
    from peft import LoraConfig, get_peft_model

    adapter = get_peft_model(
        model,
        LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.0,
            target_modules=LORA_TARGETS,
            task_type="CAUSAL_LM",
        ),
    )
    adapter.save_pretrained(output / "fresh-adapter")
    del adapter, model
    gc.collect()
    torch.cuda.empty_cache()
    try:
        results["vllm"] = _vllm_check(
            payload, tasks, output, tokenizer, lambda line: print(line, flush=True)
        )
    except Exception as exc:  # noqa: BLE001 - recorded, decided on the laptop
        results["vllm"] = {"error": repr(exc)[:1500]}
    replies = results["vllm"].get("greedy_replies")
    if replies:
        parsed = [parse(r, t["signature"]) for r, t in zip(replies, tasks)]
        grades = _grade_all(executor, [(t, p["source"]) for t, p in zip(tasks, parsed)])
        results["vllm"]["valid_replies"] = sum(p["source"] is not None for p in parsed)
        results["vllm"]["public_passes"] = sum(bool(g and g["public_passed"]) for g in grades)
    shutil.rmtree(output / "fresh-adapter", ignore_errors=True)
    (output / "vllm-check.json").write_text(json.dumps(results, indent=2) + "\n")
    archive = shutil.make_archive(str(output), "zip", output)
    print("VLLM CHECK COMPLETE", "Archive:", archive, flush=True)
