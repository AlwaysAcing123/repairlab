"""Audit frozen native repair evidence and render actual model trajectories.

Accepts the six-task pilot and the 18-task expansion. Writes its verification and
HTML into the run directory, so point it at a copy to re-audit a finished run.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_native_diagnostic import raw_native_action  # noqa: E402

from repairlab.curriculum.splits import remaining_development_tasks  # noqa: E402
from repairlab.execution import DockerExecutor, validate_source  # noqa: E402
from repairlab.experiments import source_fingerprint, write_json  # noqa: E402
from repairlab.native_repair import (  # noqa: E402
    BUDGET,
    MODEL,
    PROTOCOL,
    REVISION,
    SYSTEM,
    TASK_IDS,
)
from repairlab.native_tools import parsed_action, tools  # noqa: E402
from repairlab.tasks import canonical, manifest, pilot_tasks, sha256  # noqa: E402

PILOT_ROOT = ROOT / "artifacts/native-repair-pilot-20261030-v1"
ENVIRONMENT_KEYS = ("gpu", "dtype", "packages", "seed", "decoding", "chat_template_sha256")


def load(path):
    return json.loads(path.read_text())


def verify(root, regrade=False):
    root = Path(root)
    config = load(root / "config.json")
    plan = load(root / "experiment-plan.json")
    report = load(root / "report.json")
    assert load(root / "status.json")["status"] == "completed"
    task_ids = plan["tasks"]
    assert task_ids == config["task_ids"]
    assert task_ids in (list(TASK_IDS), [t.id for t in remaining_development_tasks()])
    assert config["manifest"] == manifest()
    assert config["source_sha256"] == source_fingerprint() == plan["source_sha256"]
    frozen = {
        p.name: sha256(p.read_text())
        for p in sorted((root / "source-snapshot/repairlab").glob("*.py"))
    }
    assert sha256(canonical(frozen)) == config["source_sha256"]
    assert config["budget"] == BUDGET and config["system_message"] == SYSTEM
    assert config["model"] == MODEL and config["model_revision"] == REVISION
    assert config["tools"] == tools() and config["new_training"] is False
    tasks = {t.id: t for t in pilot_tasks()}
    records = {
        t + "--" + arm: load(root / "episodes" / arm / (t + ".json"))
        for t in task_ids
        for arm in ("original", "supervised")
    }
    histories = {
        ident: [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": canonical(tasks[record["task"]["task_id"]].public())},
        ]
        for ident, record in records.items()
    }
    used_tokens, decisions = Counter(), Counter()
    finished = set()
    environment = None
    generated_calls = valid_native = 0
    adapter = config["adapter"]
    warmup = ROOT / "artifacts/native-supervised-warmup-20261030-v1/extracted"
    warmup_weights = load(warmup / "weight-evidence.json")
    assert (
        adapter["file_sha256"]
        == hashlib.sha256((warmup / "adapter/adapter_model.safetensors").read_bytes()).hexdigest()
    )
    assert adapter["tensor_sha256"] == warmup_weights["final_adapter_sha256"]
    assert (
        adapter["base_sha256"]
        == warmup_weights["base_initial_sha256"]
        == warmup_weights["base_final_sha256"]
    )
    for round_number in range(1, report["rounds"] + 1):
        folder = root / "rounds" / str(round_number)
        batch = load(folder / "public-requests.json")
        assert batch["batch_sha256"] == sha256(
            canonical({k: v for k, v in batch.items() if k != "batch_sha256"})
        )
        assert batch["adapter"] == adapter and batch["tools"] == tools()
        assert batch["model_revision"] == REVISION and batch["protocol"] == PROTOCOL
        metadata = load(folder / "extracted/model-config.json")
        assert metadata["batch_sha256"] == batch["batch_sha256"]
        assert metadata["round"] == batch["round"] == round_number
        assert metadata["model"] == MODEL and metadata["model_revision"] == REVISION
        assert metadata["protocol"] == PROTOCOL and metadata["budget"] == BUDGET
        assert metadata["adapter_file_sha256"] == adapter["file_sha256"]
        for name in ("original", "adapted_base"):
            assert (
                metadata[name + "_initial_sha256"]
                == metadata[name + "_final_sha256"]
                == adapter["base_sha256"]
            )
        assert (
            metadata["adapter_initial_sha256"]
            == metadata["adapter_final_sha256"]
            == adapter["tensor_sha256"]
        )
        env = {k: metadata[k] for k in ENVIRONMENT_KEYS}
        assert env["seed"] == 7 and env["decoding"] == {
            "do_sample": False,
            "num_beams": 1,
            "enable_thinking": False,
        }
        assert environment is None or env == environment
        environment = env
        assert (
            hashlib.sha256((folder / "extracted/chat_template.jinja").read_bytes()).hexdigest()
            == env["chat_template_sha256"]
        )
        local = load(folder / "upload/RepairLab_Native_Repair_Round.ipynb")
        remote = load(next((folder / "server").glob("*.ipynb")))
        assert [c["source"] for c in local["cells"]] == [c["source"] for c in remote["cells"]]
        server = load(folder / "server/kernel-metadata.json")
        assert server["is_private"] is True
        assert server["kernel_sources"] == [adapter["kernel_source"]]
        verification = load(folder / "round-verification.json")
        assert all(
            verification[k] is True
            for k in ("server_private", "server_sources_match", "worker_complete")
        )
        archives = list((folder / "download").glob("*.zip"))
        assert len(archives) == 1
        assert (
            hashlib.sha256(archives[0].read_bytes()).hexdigest() == verification["archive_sha256"]
        )
        responses = [
            json.loads(line)
            for line in (folder / "extracted/responses.jsonl").read_text().splitlines()
        ]
        assert len(responses) == len(batch["requests"]) == verification["response_count"]
        by_id = {r["episode_id"]: r for r in responses}
        assert len(by_id) == len(responses)
        assert {r["episode_id"] for r in batch["requests"]} == set(records) - finished
        for request in batch["requests"]:
            ident = request["episode_id"]
            record, response = records[ident], by_id[ident]
            decisions[ident] += 1
            assert request["messages"] == histories[ident]
            assert request["max_new_tokens"] == min(512, 4096 - used_tokens[ident])
            assert request["max_context_tokens"] == 12288
            assert response["decision"] == request["decision"] == decisions[ident]
            assert response["condition"] == request["condition"] == record["condition"]
            saved = record["model_calls"][decisions[ident] - 1]
            assert saved["messages"] == histories[ident]
            assert {k: saved[k] for k in response} == response
            assert not response.get("generation_error")
            if response.get("stop"):
                assert response["output_tokens"] == 0
                if response["stop"] == "context_budget":
                    assert response["input_tokens"] + request["max_new_tokens"] > 12288
                else:
                    assert (
                        response["stop"] == "output_token_budget" and request["max_new_tokens"] <= 0
                    )
                finished.add(ident)
                continue
            generated_calls += 1
            used_tokens[ident] += response["output_tokens"]
            assert 0 < response["output_tokens"] <= request["max_new_tokens"]
            assert response["input_tokens"] + request["max_new_tokens"] <= 12288
            event = record["events"][decisions[ident] - 1]
            try:
                parsed = parsed_action(response.get("parsed"))
            except (ValueError, TypeError, KeyError, AttributeError) as exc:
                parsed = None
                assert saved["native_error"] == str(exc)
            try:
                raw = raw_native_action(response["raw"])
            except (ValueError, TypeError, KeyError):
                raw = None
            assert parsed == raw, ident
            observation = event["observation"]
            if parsed is None:
                assert event["action"] == {"invalid_native_output": response["raw"][:8192]}
                histories[ident] += [
                    {"role": "assistant", "content": response["raw"]},
                    {
                        "role": "user",
                        "content": "Native action rejected: "
                        + saved["native_error"]
                        + ". Use exactly one valid native tool call. "
                        + canonical(observation),
                    },
                ]
            else:
                valid_native += 1
                assert saved["native_error"] is None and event["action"] == parsed
                if parsed["tool"] == "test":
                    task = tasks[record["task"]["task_id"]]
                    assert observation["total_cases"] == len(task.public_cases)
                    assert [c["args"] for c in observation["cases"]] == [
                        c.args for c in task.public_cases
                    ]
                    assert [c["expected"] for c in observation["cases"]] == [
                        c.expected for c in task.public_cases
                    ]
                elif parsed["tool"] == "submit":
                    assert observation == {"submitted": True}
                histories[ident] += [
                    response["parsed"],
                    {
                        "role": "tool",
                        "name": parsed["tool"],
                        "content": canonical(
                            {k: v for k, v in observation.items() if k != "elapsed_seconds"}
                        ),
                    },
                ]
            if decisions[ident] == len(record["model_calls"]):
                finished.add(ident)
    assert len(finished) == 2 * len(task_ids)
    pilot_config = load(PILOT_ROOT / "config.json")
    for key in ("system_message", "budget", "tools", "adapter", "protocol", "model"):
        assert config[key] == pilot_config[key], key
    assert config["model_revision"] == pilot_config["model_revision"]
    assert config["source_sha256"] == pilot_config["source_sha256"]
    pilot_metadata = load(PILOT_ROOT / "rounds/8/extracted/model-config.json")
    pilot_environment = {k: pilot_metadata[k] for k in ENVIRONMENT_KEYS}
    environment_matches_pilot = environment == pilot_environment
    regrades = []
    executor = DockerExecutor() if regrade else None
    rows, detail_html = [], []
    for ident, record in records.items():
        task = tasks[record["task"]["task_id"]]
        assert record["task"] == task.public() and record["task_sha256"] == task.fingerprint()
        source = task.buggy
        effective_edits = 0
        for index, event in enumerate(record["events"], 1):
            assert event["index"] == index
            action = event["action"]
            if action.get("tool") == "edit" and "error" not in event["observation"]:
                validate_source(action["source"], task.signature)
                effective_edits += int(source != action["source"])
                source = action["source"]
            if action.get("tool") == "read":
                assert event["observation"] == {"source": source}
            assert event["source_sha256"] == sha256(source)
        assert source == record["final_source"] and sha256(source) == record["final_source_sha256"]
        assert len(record["events"]) == record["actions"] <= 8
        assert record["test_calls"] == sum(
            e["action"].get("tool") == "test" for e in record["events"]
        )
        assert record["edits"] == sum(e["action"].get("tool") == "edit" for e in record["events"])
        assert record["invalid_actions"] == sum(
            "error" in e["observation"] for e in record["events"]
        )
        assert len(record["model_calls"]) == decisions[ident]
        assert used_tokens[ident] <= 4096
        result = record["result"]
        if result["termination"] == "submitted":
            assert record["events"][-1]["action"] == {"tool": "submit"}
            assert result["success"] == (result["public"]["passed"] and result["private"]["passed"])
            assert result["public"]["status"] == result["private"]["status"] == "graded"
            if regrade:
                public = executor.run(source, task.public_cases)
                private = executor.run(source, task.private_cases)
                assert public["status"] == private["status"] == "graded"
                assert (public["passed"] and private["passed"]) == result["success"]
                regrades.append({"episode_id": ident, "public": public, "private": private})
        else:
            assert result["success"] is False
            if result["termination"] == "action_budget":
                assert record["actions"] == 8
        row = {
            "episode_id": ident,
            "task_id": task.id,
            "condition": record["condition"],
            "success": result["success"],
            "termination": result["termination"],
            "changed_source": source != task.buggy,
            "effective_edits": effective_edits,
            "actions": record["actions"],
            "test_calls": record["test_calls"],
            "invalid_actions": record["invalid_actions"],
            "output_tokens": used_tokens[ident],
        }
        rows.append(row)
        event_details = "".join(
            "<details><summary>Action "
            + str(e["index"])
            + " · "
            + escape(e["action"].get("tool", "invalid"))
            + "</summary><h4>Actual model output</h4><pre>"
            + escape(record["model_calls"][e["index"] - 1]["raw"])
            + "</pre><h4>Applied diff</h4><pre>"
            + escape(e.get("diff", "No source change"))
            + "</pre><h4>Tool result</h4><pre>"
            + escape(json.dumps(e["observation"], indent=2))
            + "</pre></details>"
            for e in record["events"]
        )
        detail_html.append(
            "<details class='episode'><summary>"
            + escape(ident)
            + " · "
            + ("PASS" if result["success"] else "FAIL")
            + " · "
            + escape(result["termination"])
            + "</summary><p>"
            + escape(task.specification)
            + "</p><h3>Starting function</h3><pre>"
            + escape(task.buggy)
            + "</pre>"
            + event_details
            + "<h3>Final candidate</h3><pre>"
            + escape(source)
            + "</pre><h3>Independent final grade</h3><pre>"
            + escape(json.dumps(result, indent=2))
            + "</pre></details>"
        )
    for arm in ("original", "supervised"):
        selected = [r for r in rows if r["condition"] == arm]
        for key in ("successes", "actions", "test_calls", "invalid_actions", "output_tokens"):
            assert report["arms"][arm][key] == sum(
                r["success" if key == "successes" else key] for r in selected
            )
    pairs = [
        {arm: records[t + "--" + arm]["result"]["success"] for arm in ("original", "supervised")}
        for t in task_ids
    ]
    assert report["paired"] == {
        "both_pass": sum(all(p.values()) for p in pairs),
        "adapter_only": sum(p["supervised"] and not p["original"] for p in pairs),
        "original_only": sum(p["original"] and not p["supervised"] for p in pairs),
        "both_fail": sum(not any(p.values()) for p in pairs),
    }
    summary = {
        "verified": True,
        "episodes": 2 * len(task_ids),
        "tasks": len(task_ids),
        "generated_calls": generated_calls,
        "valid_native_calls": valid_native,
        "invalid_native_calls": generated_calls - valid_native,
        "source_and_task_fingerprints_match": True,
        "public_only_conversations_verified": True,
        "native_raw_and_parsed_actions_agree": True,
        "identical_environment_across_rounds": True,
        "environment_matches_pilot": environment_matches_pilot,
        "environment_differences_from_pilot": {
            k: {"pilot": pilot_environment[k], "this_run": environment[k]}
            for k in ENVIRONMENT_KEYS
            if environment[k] != pilot_environment[k]
        },
        "server_private_and_sources_match": True,
        "saved_adapter_reloaded_and_unchanged": True,
        "both_base_models_unchanged": True,
        "fresh_submitted_regrades": regrades,
        "arms": report["arms"],
        "paired": report["paired"],
        "rows": rows,
    }
    write_json(root / "independent-verification.json", summary)
    table_rows = "".join(
        "<tr><td>"
        + escape(t)
        + "</td>"
        + "".join(
            "<td>" + ("PASS" if records[t + "--" + a]["result"]["success"] else "FAIL") + "</td>"
            for a in ("original", "supervised")
        )
        + "</tr>"
        for t in task_ids
    )
    count = len(task_ids)
    html = """<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>RepairLab autonomous repair pilot</title><style>
body{font:16px system-ui;max-width:1000px;margin:48px auto;padding:0 24px;background:#f5f7f4;color:#183139}
h1{font-size:34px}table{border-collapse:collapse;width:100%;background:white;margin:24px 0}th,td{text-align:left;padding:12px;border-bottom:1px solid #d7dfd9}
details{background:white;padding:16px;margin:12px 0;border:1px solid #d7dfd9;border-radius:8px}summary{cursor:pointer;font-weight:600}
pre{white-space:pre-wrap;overflow:auto;font-size:13px}small{color:#50636a}</style></head><body>
<small>ACTUAL MODEL RUNS · """
    html += (
        str(count)
        + """ DEVELOPMENT TASKS · SAVED SUPERVISED ADAPTER · NO NEW TRAINING OR RL</small>
<h1>Can the model repair a bug autonomously?</h1>"""
    )
    html += (
        "<p>Original: <b>"
        + str(report["arms"]["original"]["successes"])
        + "/"
        + str(count)
        + "</b>. Saved adapter: <b>"
        + str(report["arms"]["supervised"]["successes"])
        + "/"
        + str(count)
        + "</b>.</p>"
    )
    html += "<p>Both used the same native read, test, edit, and submit tools; public feedback; eight actions; 512 output tokens per call and 4,096 per episode. Passing requires explicit submission and all independent final checks. Private grades were never fed back.</p>"
    html += (
        "<p>This small synthetic development comparison measures end-to-end behavior. "
        + str(count)
        + " tasks, one seed, and reused development problems do not establish broad generalization or a statistically reliable training gain.</p>"
    )
    html += (
        "<table><thead><tr><th>Task</th><th>Original</th><th>Saved adapter</th></tr></thead><tbody>"
        + table_rows
        + "</tbody></table>"
    )
    html += (
        "<p>"
        + str(valid_native)
        + "/"
        + str(generated_calls)
        + " generated responses were valid native actions. All checkpoint, model, source, budget, and conversation checks passed.</p>"
    )
    html += (
        "<h2>Saved trajectories</h2><p>Expand an episode to inspect actual output, applied edits, public feedback, and the local final grade. No reasoning has been invented.</p>"
        + "".join(detail_html)
        + "</body></html>"
    )
    (root / "results.html").write_text(html)
    print(
        json.dumps(
            {k: v for k, v in summary.items() if k not in {"fresh_submitted_regrades", "rows"}},
            indent=2,
        )
    )
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--regrade", action="store_true")
    args = parser.parse_args()
    verify(args.path, args.regrade)
