"""Audit native tool probes against saved raw XML/JSON and the frozen probe inputs."""

import argparse
import hashlib
import json
import re
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repairlab.execution import reject_constant, unique_keys  # noqa: E402
from repairlab.experiments import source_fingerprint, write_json  # noqa: E402
from repairlab.native_tools import parsed_action, probes, roundtrip_examples  # noqa: E402


def raw_native_action(raw):
    blocks = re.findall(r"<tool_call>(.*?)</tool_call>", raw, flags=re.S)
    if len(blocks) != 1:
        raise ValueError("Expected one complete native tool_call block")
    function = json.loads(blocks[0], parse_constant=reject_constant, object_pairs_hook=unique_keys)
    if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
        raise ValueError("Unexpected native function object")
    return parsed_action({"tool_calls": [{"type": "function", "function": function}]})


def verify(root):
    root = Path(root)
    plan = json.loads((root / "experiment-plan.json").read_text())
    output = root / "extracted"
    metadata = json.loads((output / "model-config.json").read_text())
    report = json.loads((output / "report.json").read_text())
    roundtrips = json.loads((output / "roundtrips.json").read_text())
    rows = [json.loads(line) for line in (output / "responses.jsonl").read_text().splitlines()]
    assert len(rows) == len(probes()) == 12 and len({r["probe_id"] for r in rows}) == 12
    assert plan["probes"] == probes() and plan["source_sha256"] == source_fingerprint()
    assert metadata["model_revision"] == plan["model_revision"]
    assert metadata["initial_weight_sha256"] == metadata["final_weight_sha256"]
    assert (
        hashlib.sha256((output / "chat_template.jinja").read_bytes()).hexdigest()
        == metadata["chat_template_sha256"]
    )
    assert len(roundtrips) == 4
    for actual, expected in zip(roundtrips, roundtrip_examples()):
        assert actual["prefix_preserved"] and actual["matched"]
        assert actual["parsed"]["tool_calls"] == expected["tool_calls"]
        assert raw_native_action(actual["raw"]) == parsed_action(expected) == actual["action"]
    by_id = {p["probe_id"]: p for p in probes()}
    results = []
    details = []
    for row in rows:
        probe = by_id[row["probe_id"]]
        assert row["messages"] == probe["messages"]
        assert 0 < row["output_tokens"] <= 256
        assert row["input_tokens"] + 256 <= 4096
        try:
            action = raw_native_action(row["raw"])
            valid = True
        except (ValueError, TypeError, KeyError):
            action, valid = None, False
        # The native parser and independent raw-block extraction must agree.
        assert valid == row["valid_native_call"], row["probe_id"]
        if valid:
            assert action == parsed_action(row["parsed"]) == row["action"]
        matched = valid and action["tool"] == probe["expected_tool"]
        assert matched == row["requested_tool_matched"]
        source_matched = (
            valid and action.get("source", "").strip() == probe["expected_source"].strip()
            if probe["expected_source"] is not None
            else None
        )
        results.append(
            {
                "probe_id": row["probe_id"],
                "valid_native_call": valid,
                "requested_tool_matched": matched,
                "requested_source_matched": source_matched,
            }
        )
        details.append(
            "<details><summary>"
            + escape(row["probe_id"])
            + " · "
            + ("PASS" if matched else "FAIL")
            + "</summary><h3>Prompt</h3><pre>"
            + escape(json.dumps(probe["messages"], indent=2))
            + "</pre><h3>Actual native output</h3><pre>"
            + escape(row["raw"])
            + "</pre><h3>Parsed action</h3><pre>"
            + escape(json.dumps(action, indent=2))
            + "</pre></details>"
        )
    assert report["valid_native_calls"] == sum(r["valid_native_call"] for r in results)
    assert report["requested_tools_matched"] == sum(r["requested_tool_matched"] for r in results)
    ready = all(r["valid_native_call"] and r["requested_tool_matched"] for r in results)
    assert ready == report["ready_for_small_repair_pilot"]
    server = json.loads((root / "server-v1/kernel-metadata.json").read_text())
    assert server["is_private"] is True
    local_notebook = json.loads(
        (root / "upload/RepairLab_Native_Tool_Diagnostic.ipynb").read_text()
    )
    remote_notebook = json.loads(next((root / "server-v1").glob("*.ipynb")).read_text())
    assert [c["source"] for c in local_notebook["cells"]] == [
        c["source"] for c in remote_notebook["cells"]
    ]
    summary = {
        "verified": True,
        "roundtrips": 4,
        "probes": 12,
        "valid_native_calls": report["valid_native_calls"],
        "requested_tools_matched": report["requested_tools_matched"],
        "explicit_source_matches": sum(r["requested_source_matched"] is True for r in results),
        "source_argument_probes": sum(r["requested_source_matched"] is not None for r in results),
        "weights_unchanged": True,
        "source_and_probes_unchanged": True,
        "server_private_and_source_matches": True,
        "native_parser_and_independent_raw_extraction_agree": True,
        "ready_for_small_repair_pilot": ready,
        "repair_accuracy_measured": False,
        "rows": results,
    }
    write_json(root / "independent-verification.json", summary)
    html = """<!doctype html><meta charset="utf-8"><title>RepairLab native-tool diagnostic</title>
<style>body{font:16px system-ui;max-width:950px;margin:48px auto;padding:0 24px;color:#183139;background:#f5f7f4}
details{background:white;padding:18px;margin:12px 0;border:1px solid #d7dfd9;border-radius:8px}
summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;overflow:auto;font-size:13px}</style>
<small>ORIGINAL MODEL · INTERFACE DIAGNOSTIC · NO TRAINING · NOT A REPAIR SCORE</small>
<h1>Native tool-calling diagnostic</h1>"""
    html += (
        "<p>"
        + str(report["valid_native_calls"])
        + "/12 valid native calls; "
        + str(report["requested_tools_matched"])
        + "/12 requested tools matched. Four serializer/parser round-trips passed.</p>"
    )
    html += "<p>These synthetic probes check interface behavior. They do not establish autonomous debugging or generalization.</p>"
    (root / "results.html").write_text(html + "".join(details))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    verify(parser.parse_args().path)
