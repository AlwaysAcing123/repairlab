"""Audit a saved 1.7B supervised run: targets, notes, masks, tensors, and separation.

Needs numpy, safetensors, and tokenizers, like the earlier training audits.
"""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_native_diagnostic import raw_native_action  # noqa: E402

from repairlab.curriculum.final_test import final_tasks  # noqa: E402
from repairlab.curriculum.training_programs import training_programs  # noqa: E402
from repairlab.execution import DockerExecutor  # noqa: E402
from repairlab.experiments import write_json  # noqa: E402
from repairlab.final2.programs import final2_tasks  # noqa: E402
from repairlab.native_tools import parsed_action  # noqa: E402
from repairlab.scaleup.models import LARGE  # noqa: E402
from repairlab.scaleup.supervised import bundle  # noqa: E402
from repairlab.tasks import canonical, pilot_tasks, sha256  # noqa: E402


def verify(root):
    import numpy as np
    from safetensors.numpy import load_file
    from tokenizers import Tokenizer

    root = Path(root)
    output = root / "run" / "extracted"
    plan = json.loads((root / "experiment-plan.json").read_text())
    recorded = json.loads((output / "bundle.json").read_text())
    metadata = json.loads((output / "model-config.json").read_text())
    evidence = json.loads((output / "weight-evidence.json").read_text())
    assert json.loads((output / "status.json").read_text())["status"] == "completed"
    variant = plan["variant"]
    rebuilt = bundle(DockerExecutor(), variant)
    assert recorded == rebuilt == json.loads((root / "bundle.json").read_text())
    assert recorded["bundle_sha256"] == plan["bundle_sha256"] == metadata["bundle_sha256"]
    assert (
        sha256(canonical({k: v for k, v in recorded.items() if k != "bundle_sha256"}))
        == plan["bundle_sha256"]
    )
    assert recorded["model"] == LARGE["model"] and recorded["base_sha256"] == LARGE["base_sha256"]
    config = recorded["config"]
    assert config["gradient_checkpointing"] is True
    train = {r["sample_id"]: r for r in recorded["train"]}
    assert len(train) == len(recorded["train"]) == 480
    programs = {p.task.id: p for p in training_programs()}
    notes = 0
    for sample in train.values():
        completion = sample["completion"][0]
        function = completion["tool_calls"][0]["function"]
        if function["name"] == "edit":
            assert function["arguments"]["source"] == programs[sample["program_id"]].task.reference
            if variant == "notes":
                assert completion["content"].startswith(("Failing case: ", "Cause: "))
                notes += 1
        if variant == "plain" or function["name"] != "edit":
            assert completion["content"] == ""
    text = canonical(recorded["train"])
    for task in pilot_tasks() + final_tasks() + final2_tasks():
        assert json.dumps(task.reference)[1:-1] not in text
    tokenized = json.loads((output / "tokenized-training.json").read_text())
    stats = {
        r["sample_id"]: r for r in json.loads((output / "tokenization-preflight.json").read_text())
    }
    tokenizer = Tokenizer.from_file(str(output / "adapter/tokenizer.json"))
    assert len(tokenized) == len(stats) == len(train)
    for row in tokenized:
        stat = stats[row["sample_id"]]
        prefix = stat["prefix_tokens"]
        assert len(row["input_ids"]) == len(row["labels"]) == stat["total_tokens"]
        assert stat["total_tokens"] <= config["max_sequence_tokens"]
        assert all(value == -100 for value in row["labels"][:prefix])
        assert row["labels"][prefix:] == row["input_ids"][prefix:]
        decoded = tokenizer.decode(row["labels"][prefix:], skip_special_tokens=False)
        completion = train[row["sample_id"]]["completion"][0]
        assert raw_native_action(decoded) == parsed_action(completion)
        assert decoded.split("<tool_call>")[0].strip() == completion["content"].strip()
    logs = json.loads((output / "training-log.json").read_text())
    batches = json.loads((output / "training-batches.json").read_text())
    steps = config["optimizer_steps"]
    assert [r["optimizer_step"] for r in logs] == list(range(1, steps + 1))
    assert all(math.isfinite(r["loss"]) and math.isfinite(r["gradient_norm"]) for r in logs)
    assert len(batches) == steps * config["gradient_accumulation"]
    observed = Counter(sample for row in batches for sample in row["sample_ids"])
    assert set(observed) <= set(train) and set(observed) == set(evidence["trained_sample_ids"])
    initial = load_file(str(output / "initial_adapter.safetensors"))
    final = load_file(str(output / "adapter/adapter_model.safetensors"))
    assert set(initial) == set(final) and all("lora_" in name for name in final)
    assert all(not initial[k].any() for k in initial if "lora_B" in k)

    def tensor_hash(values):
        digest = hashlib.sha256()
        for name in sorted(values):
            digest.update(name.encode() + b"\0")
            digest.update(values[name].tobytes(order="C"))
        return digest.hexdigest()

    assert tensor_hash(initial) == evidence["initial_adapter_sha256"]
    assert tensor_hash(final) == evidence["final_adapter_sha256"]
    assert evidence["initial_adapter_sha256"] != evidence["final_adapter_sha256"]
    delta = math.sqrt(
        sum(
            float(np.square(final[k].astype(np.float64) - initial[k].astype(np.float64)).sum())
            for k in initial
        )
    )
    assert delta > 0 and math.isclose(delta, evidence["adapter_l2_change"], rel_tol=1e-10)
    assert (
        evidence["base_initial_sha256"] == evidence["base_final_sha256"] == recorded["base_sha256"]
    )
    adapter = json.loads((root / "scaleup-adapter.json").read_text())
    assert adapter["tensor_sha256"] == evidence["final_adapter_sha256"]
    assert adapter["base_sha256"] == recorded["base_sha256"]
    assert (
        adapter["file_sha256"]
        == hashlib.sha256((output / "adapter/adapter_model.safetensors").read_bytes()).hexdigest()
    )
    summary = {
        "verified": True,
        "variant": variant,
        "optimizer_steps": steps,
        "unique_samples": len(train),
        "unique_samples_presented": len(observed),
        "edit_targets_with_notes": notes,
        "wrong_repairs_as_targets": 0,
        "evaluation_solutions_in_training": 0,
        "adapter_l2_change": delta,
        "first_loss": logs[0]["loss"],
        "final_loss": logs[-1]["loss"],
        "longest_sequence_tokens": max(s["total_tokens"] for s in stats.values()),
        "peak_gpu_bytes": metadata["peak_gpu_bytes"],
        "elapsed_seconds": metadata["elapsed_seconds"],
    }
    write_json(root / "independent-verification.json", summary)
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    verify(parser.parse_args().path)
