"""Independent saved-tensor, completion-mask, data-boundary and evaluation audit."""

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_native_diagnostic import raw_native_action  # noqa: E402

from repairlab.experiments import source_fingerprint, write_json  # noqa: E402
from repairlab.native_tools import parsed_action  # noqa: E402
from repairlab.tasks import canonical, sha256  # noqa: E402
from repairlab.tool_warmup import bundle  # noqa: E402


def verify(root):
    import numpy as np
    from safetensors.numpy import load_file
    from tokenizers import Tokenizer

    root = Path(root)
    output = root / "extracted"
    plan = json.loads((root / "experiment-plan.json").read_text())
    recorded = json.loads((output / "bundle.json").read_text())
    metadata = json.loads((output / "model-config.json").read_text())
    comparison = json.loads((output / "comparison.json").read_text())
    evidence = json.loads((output / "weight-evidence.json").read_text())
    status = json.loads((output / "status.json").read_text())
    assert status["status"] == "completed"
    assert plan["source_sha256"] == source_fingerprint()
    assert recorded == bundle() == json.loads((root / "bundle.json").read_text())
    assert recorded["bundle_sha256"] == plan["bundle_sha256"] == metadata["bundle_sha256"]
    assert (
        sha256(canonical({k: v for k, v in recorded.items() if k != "bundle_sha256"}))
        == plan["bundle_sha256"]
    )
    train = {r["sample_id"]: r for r in recorded["train"]}
    assert len(train) == 64
    assert {r["group_id"] for r in train.values()}.isdisjoint(
        {r["group_id"] for r in recorded["development"]}
    )
    tokenized = json.loads((output / "tokenized-training.json").read_text())
    stats = {
        r["sample_id"]: r for r in json.loads((output / "tokenization-preflight.json").read_text())
    }
    tokenizer = Tokenizer.from_file(str(output / "adapter/tokenizer.json"))
    assert len(tokenized) == len(stats) == len(train) == 64
    for row in tokenized:
        stat = stats[row["sample_id"]]
        prefix = stat["prefix_tokens"]
        assert stat["teacher_roundtrip"] and stat["prompt_masked"]
        assert len(row["input_ids"]) == len(row["labels"]) == stat["total_tokens"]
        assert all(value == -100 for value in row["labels"][:prefix])
        assert row["labels"][prefix:] == row["input_ids"][prefix:]
        assert stat["completion_tokens"] == len(row["labels"]) - prefix > 0
        decoded = tokenizer.decode(row["labels"][prefix:], skip_special_tokens=False)
        assert raw_native_action(decoded) == parsed_action(train[row["sample_id"]]["completion"][0])
    logs = json.loads((output / "training-log.json").read_text())
    batches = json.loads((output / "training-batches.json").read_text())
    assert [r["optimizer_step"] for r in logs] == list(range(1, 61))
    assert all(math.isfinite(r["loss"]) and math.isfinite(r["gradient_norm"]) for r in logs)
    assert len(batches) == 60 * recorded["config"]["gradient_accumulation"]
    observed_samples = Counter(sample for row in batches for sample in row["sample_ids"])
    assert set(observed_samples) == set(train) == set(evidence["trained_sample_ids"])
    assert sum(observed_samples.values()) == 60 * 4 * 2
    for row in batches:
        assert len(row["sample_ids"]) == 2
        assert all(train[ident]["split"] == "train" for ident in row["sample_ids"])
    initial = load_file(str(output / "initial_adapter.safetensors"))
    final = load_file(str(output / "adapter/adapter_model.safetensors"))
    assert set(initial) == set(final)
    assert all("lora_" in name for name in final)

    def tensor_hash(values):
        digest = hashlib.sha256()
        for name in sorted(values):
            digest.update(name.encode() + b"\0")
            digest.update(values[name].tobytes(order="C"))
        return digest.hexdigest()

    assert tensor_hash(initial) == evidence["initial_adapter_sha256"]
    assert tensor_hash(final) == evidence["final_adapter_sha256"]
    assert evidence["initial_adapter_sha256"] != evidence["final_adapter_sha256"]
    assert all(np.count_nonzero(v) == 0 for k, v in initial.items() if "lora_B" in k)
    delta = math.sqrt(
        sum(
            float(np.square(final[k].astype(np.float64) - initial[k].astype(np.float64)).sum())
            for k in initial
        )
    )
    assert delta > 0 and math.isclose(
        delta, evidence["adapter_l2_change"], rel_tol=1e-10, abs_tol=1e-12
    )
    assert (
        evidence["base_initial_sha256"]
        == evidence["base_final_sha256"]
        == plan["expected_original_base_sha256"]
    )
    assert evidence["evaluation_weights_unchanged"] and evidence["base_weights_unchanged"]
    adapter_config = json.loads((output / "adapter/adapter_config.json").read_text())
    assert adapter_config["r"] == 8 and set(adapter_config["target_modules"]) == {
        "q_proj",
        "v_proj",
    }
    assert adapter_config["revision"] == plan["model_revision"]
    fixture_groups = {
        "historical_diagnostic": recorded["historical_diagnostic"],
        "separate_programs": recorded["development"],
    }
    verified = {}
    details = []
    for label in ("original", "supervised"):
        verified[label] = {}
        for category, fixtures in fixture_groups.items():
            results = []
            for fixture in fixtures:
                row = json.loads(
                    (output / "evaluation" / label / (fixture["probe_id"] + ".json")).read_text()
                )
                assert row["messages"] == fixture["messages"] and row["category"] == category
                assert 0 < row["output_tokens"] <= 256 and row["input_tokens"] + 256 <= 4096
                try:
                    action = raw_native_action(row["raw"])
                    valid = True
                except (ValueError, TypeError, KeyError):
                    action, valid = None, False
                assert valid == row["valid_native_call"], (label, fixture["probe_id"])
                if valid:
                    assert action == row["action"] == parsed_action(row["parsed"])
                matched = valid and action["tool"] == fixture["expected_tool"]
                source_matched = (
                    valid and action.get("source", "").strip() == fixture["expected_source"].strip()
                    if fixture["expected_source"] is not None
                    else None
                )
                assert matched == row["requested_tool_matched"]
                results.append(
                    {"valid": valid, "matched": matched, "source_matched": source_matched}
                )
                details.append(
                    "<details><summary>"
                    + escape(label + " · " + fixture["probe_id"])
                    + " · "
                    + ("PASS" if matched else "FAIL")
                    + "</summary><pre>"
                    + escape(row["raw"])
                    + "</pre></details>"
                )
            summary = {
                "probes": len(fixtures),
                "valid_native_calls": sum(r["valid"] for r in results),
                "requested_tools_matched": sum(r["matched"] for r in results),
                "source_matches": sum(r["source_matched"] is True for r in results),
            }
            assert summary == comparison[label][category]
            verified[label][category] = summary
    assert json.loads((root / "server-v1/kernel-metadata.json").read_text())["is_private"] is True
    local = json.loads((root / "upload/RepairLab_Native_Supervised_Warmup.ipynb").read_text())
    remote = json.loads(next((root / "server-v1").glob("*.ipynb")).read_text())
    assert [c["source"] for c in local["cells"]] == [c["source"] for c in remote["cells"]]
    summary = {
        "verified": True,
        "optimizer_steps": 60,
        "training_examples": 64,
        "sample_presentations": 480,
        "completion_masks_and_targets_verified": True,
        "training_evaluation_groups_disjoint": True,
        "base_weights_unchanged": True,
        "saved_adapter_tensors_changed": True,
        "adapter_tensors": len(final),
        "adapter_l2_change": delta,
        "adapter_hashes_match_actual_files": True,
        "evaluation_raw_outputs_verified": True,
        "private_notebook_source_matches": True,
        "scores": verified,
        "training_method": "supervised LoRA; no RL",
        "repair_accuracy_measured": False,
    }
    write_json(root / "independent-verification.json", summary)
    html = """<!doctype html><meta charset="utf-8"><title>RepairLab supervised warm-up</title>
<style>body{font:16px system-ui;max-width:950px;margin:48px auto;padding:0 24px;color:#183139;background:#f5f7f4}
details{background:white;padding:18px;margin:12px 0;border:1px solid #d7dfd9;border-radius:8px}
summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow:auto;font-size:13px}table{border-collapse:collapse;width:100%}td,th{padding:10px;border:1px solid #cbd7d0}</style>
<small>ACTUAL SUPERVISED TRAINING · 60 LORA STEPS · DEVELOPMENT INTERFACE CHECKS · NO RL</small>
<h1>Native tool-use warm-up</h1><p>Base weights stayed frozen. Saved adapter tensors changed;
all training masks, target calls, and before/after scores were independently audited.
These are instructed tool-use probes, not autonomous repair benchmark scores.</p>"""
    html += "<table><tr><th>Evaluation</th><th>Original</th><th>Supervised adapter</th></tr>"
    for category, fixtures in fixture_groups.items():
        html += (
            "<tr><td>"
            + escape(category.replace("_", " "))
            + "</td><td>"
            + str(verified["original"][category]["requested_tools_matched"])
            + "/"
            + str(len(fixtures))
            + "</td><td>"
            + str(verified["supervised"][category]["requested_tools_matched"])
            + "/"
            + str(len(fixtures))
            + "</td></tr>"
        )
    html += (
        "</table><p>Adapter L2 parameter change: "
        + str(round(delta, 6))
        + ". No checkpoint was selected based on these evaluation scores.</p>"
    )
    (root / "results.html").write_text(html + "".join(details))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    verify(parser.parse_args().path)
