"""Build a finite native-tool interface diagnostic; public synthetic probes only."""

import argparse
import json
import sys
import textwrap
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repairlab import native_tools  # noqa: E402
from repairlab.revision import MODEL, REVISION  # noqa: E402


def build(destination):
    cells = []

    def add(kind, text):
        cell = {
            "cell_type": kind,
            "metadata": {},
            "source": textwrap.dedent(text).strip() + "\n",
            "id": f"native-diagnostic-{len(cells):02d}",
        }
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)

    add(
        "markdown",
        """
    # RepairLab native-tool diagnostic

    Original Qwen3-0.6B; 12 synthetic interface probes, four serializer/parser
    round-trips, one response per probe, at most 256 new tokens each. No training,
    repair benchmark scoring, model-generated-code execution, or private evaluator.
    Standard Qwen tool definitions, native chat template, and TRL Qwen response schema
    are used. No examples are added beyond the synthetic probe's explicit history.
    All native raw outputs and parsed responses are saved, including failures.

    Readiness gate: all four contract round-trips and all 12 probe tool calls must
    parse and match their requested tool before launching a larger native repair run.
    This is a development gate, not a general tool-use or code-repair benchmark.
    """,
    )
    add(
        "code",
        """
    import os,sys,subprocess
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
    subprocess.check_call([sys.executable,"-m","pip","install","transformers==5.2.0","trl==1.8.0"])
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU required")
    """,
    )
    add(
        "code",
        "import json\nMODEL = "
        + repr(MODEL)
        + "\nREVISION = "
        + repr(REVISION)
        + "\nTOOLS = json.loads("
        + repr(json.dumps(native_tools.tools()))
        + ")"
        + "\nPROBES = json.loads("
        + repr(json.dumps(native_tools.probes()))
        + ")"
        + "\nEXAMPLES = json.loads("
        + repr(json.dumps(native_tools.roundtrip_examples()))
        + ")"
        + "\nPROTOCOL = "
        + repr(native_tools.PROTOCOL),
    )
    add(
        "code",
        """
    import hashlib,importlib.metadata,random,time
    from pathlib import Path
    from datetime import datetime,timezone
    from transformers import AutoTokenizer,AutoModelForCausalLM
    from trl.chat_template_utils import qwen3_schema
    output = Path("repairlab-native-diagnostic-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
    output.mkdir(exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(MODEL,revision=REVISION,trust_remote_code=False)
    tokenizer.response_schema = qwen3_schema
    template = tokenizer.chat_template
    (output/"chat_template.jinja").write_text(template)
    def validate(assistant):
        calls = assistant.get("tool_calls") if isinstance(assistant,dict) else None
        if not isinstance(calls,list) or len(calls)!=1:
            raise ValueError("Exactly one native tool call required")
        function = calls[0]["function"]
        name,args = function["name"],function["arguments"]
        if name not in {"read","test","edit","submit"} or not isinstance(args,dict):
            raise ValueError("Unknown tool or invalid arguments")
        if set(args)!=({"source"} if name=="edit" else set()):
            raise ValueError("Unexpected argument keys")
        if name=="edit" and not isinstance(args["source"],str):
            raise ValueError("Source must be a string")
        return {"tool":name,**args}
    roundtrips=[]
    for example in EXAMPLES:
        history=[{"role":"system","content":"Use the supplied tools."},
                 {"role":"user","content":"Call the requested tool."}]
        options=dict(tools=TOOLS,tokenize=False,enable_thinking=False)
        prefix=tokenizer.apply_chat_template(history,add_generation_prompt=True,**options)
        full=tokenizer.apply_chat_template(history+[example],add_generation_prompt=False,**options)
        record={"tool":example["tool_calls"][0]["function"]["name"],"prefix_preserved":full.startswith(prefix)}
        try:
            if not record["prefix_preserved"]:
                raise ValueError("Template prefix mismatch")
            raw=full[len(prefix):]
            parsed=tokenizer.parse_response(raw)
            record.update(raw=raw,parsed=parsed,action=validate(parsed))
            record["matched"]=parsed.get("tool_calls")==example["tool_calls"]
        except Exception as exc:
            record.update(matched=False,error=type(exc).__name__+": "+str(exc)[:500])
        roundtrips.append(record)
    (output/"roundtrips.json").write_text(json.dumps(roundtrips,indent=2))
    if not all(r["matched"] for r in roundtrips):
        import shutil
        shutil.make_archive(str(output),"zip",output)
        raise RuntimeError("Native serialization/parser contract failed; diagnose before model generation")
    random.seed(7)
    torch.manual_seed(7)
    torch.cuda.manual_seed_all(7)
    torch.backends.cuda.matmul.allow_tf32=False
    dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    model=AutoModelForCausalLM.from_pretrained(MODEL,revision=REVISION,dtype=dtype,
        trust_remote_code=False).to("cuda:0").eval().requires_grad_(False)
    def weight_hash():
        digest=hashlib.sha256()
        for name,value in model.named_parameters():
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes())
        return digest.hexdigest()
    metadata={"protocol":PROTOCOL,"model":MODEL,"model_revision":REVISION,
        "seed":7,"max_new_tokens":256,"max_context_tokens":4096,"thinking":False,
        "do_sample":False,"initial_weight_sha256":weight_hash(),
        "gpu":torch.cuda.get_device_name(0),"dtype":str(dtype),
        "chat_template_sha256":hashlib.sha256(template.encode()).hexdigest(),
        "packages":{p:importlib.metadata.version(p) for p in
            ("torch","transformers","trl","tokenizers","huggingface-hub","safetensors")}}
    (output/"model-config.json").write_text(json.dumps(metadata,indent=2))
    print("Four native round-trips passed; original model loaded.",flush=True)
    """,
    )
    add(
        "code",
        """
    started=time.monotonic()
    rows=[]
    with (output/"responses.jsonl").open("w") as log:
        for probe in PROBES:
            row={"probe_id":probe["probe_id"],"messages":probe["messages"]}
            inputs=tokenizer.apply_chat_template(probe["messages"],tools=TOOLS,
                add_generation_prompt=True,enable_thinking=False,tokenize=True,
                return_dict=True,return_tensors="pt").to(model.device)
            length=int(inputs["input_ids"].shape[-1])
            if length+256>4096:
                raise RuntimeError("Probe exceeds frozen context budget")
            begin=time.monotonic()
            with torch.inference_mode():
                generated=model.generate(**inputs,do_sample=False,max_new_tokens=256,
                    num_beams=1,pad_token_id=tokenizer.eos_token_id,use_cache=True)
            torch.cuda.synchronize()
            completion=generated[0,length:]
            raw=tokenizer.decode(completion,skip_special_tokens=False)
            row.update(raw=raw,input_tokens=length,output_tokens=int(completion.shape[-1]),
                       elapsed_seconds=time.monotonic()-begin)
            try:
                parsed=tokenizer.parse_response(raw)
                action=validate(parsed)
                row.update(parsed=parsed,action=action,valid_native_call=True,
                           requested_tool_matched=action["tool"]==probe["expected_tool"])
                row["requested_source_matched"]=(action.get("source","").strip()==probe["expected_source"].strip()
                    if probe["expected_source"] is not None else None)
            except Exception as exc:
                row.update(valid_native_call=False,requested_tool_matched=False,
                           parse_error=type(exc).__name__+": "+str(exc)[:500])
            rows.append(row)
            log.write(json.dumps(row,ensure_ascii=False)+"\\n")
            log.flush()
            print(probe["probe_id"],"native",row["valid_native_call"],
                  "requested tool",row["requested_tool_matched"],flush=True)
    metadata.update(final_weight_sha256=weight_hash(),elapsed_seconds=time.monotonic()-started,
                    peak_gpu_bytes=torch.cuda.max_memory_allocated(0))
    assert metadata["initial_weight_sha256"]==metadata["final_weight_sha256"]
    (output/"model-config.json").write_text(json.dumps(metadata,indent=2))
    report={"probes":len(rows),"valid_native_calls":sum(r["valid_native_call"] for r in rows),
        "requested_tools_matched":sum(r["requested_tool_matched"] for r in rows),
        "roundtrip_passes":sum(r["matched"] for r in roundtrips),"weights_unchanged":True,
        "ready_for_small_repair_pilot":all(r["valid_native_call"] and r["requested_tool_matched"] for r in rows),
        "repair_accuracy_measured":False,"training_performed":False}
    (output/"report.json").write_text(json.dumps(report,indent=2))
    import shutil
    archive=shutil.make_archive(str(output),"zip",output)
    print("DIAGNOSTIC COMPLETE",json.dumps(report),"Archive:",archive,flush=True)
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    build(parser.parse_args().output)
