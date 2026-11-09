"""Audit a matched comparison: conversations, checkpoints, budgets, and fresh grades.

Covers development comparisons, RL rollout cycles, and the one-time final test, which
share one episode engine. Writes its verification and HTML into the run directory.
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

from repairlab.curriculum.comparison import (  # noqa: E402
    ENVIRONMENT_KEYS,
    PROTOCOL,
    SEEDED_ACTIONS,
)
from repairlab.curriculum.final_test import final_tasks  # noqa: E402
from repairlab.curriculum.transcript import initial_messages, tool_message  # noqa: E402
from repairlab.execution import DockerExecutor, validate_source  # noqa: E402
from repairlab.experiments import write_json  # noqa: E402
from repairlab.native_repair import BUDGET, MODEL, REVISION, SYSTEM  # noqa: E402
from repairlab.native_tools import assistant_call, parsed_action, tools  # noqa: E402
from repairlab.rl.programs import rl_programs  # noqa: E402
from repairlab.rl.rollouts import CONFIG as RL_CONFIG  # noqa: E402
from repairlab.rl.rollouts import PROTOCOL as RL_PROTOCOL  # noqa: E402
from repairlab.rl.rollouts import SAMPLING  # noqa: E402
from repairlab.tasks import canonical, pilot_tasks, sha256  # noqa: E402

SPLITS = {
    "development": (PROTOCOL, pilot_tasks),
    "final_test": (PROTOCOL, final_tasks),
    "rl_training": (RL_PROTOCOL, rl_programs),
}


def load(path):
    return json.loads(path.read_text())


def scripted_history(task, events):
    messages = initial_messages(task)
    for event in events:
        action = dict(event["action"])
        tool = action.pop("tool")
        messages += [assistant_call(tool, action), tool_message(tool, event["observation"])]
    return messages


def verify(root, regrade=False):
    root = Path(root)
    config, plan = load(root / "config.json"), load(root / "experiment-plan.json")
    report = load(root / "report.json")
    assert load(root / "status.json")["status"] == "completed"
    split = config["split"]
    protocol, source = SPLITS[split]
    assert config["protocol"] == protocol and config["model"] == MODEL
    assert config["model_revision"] == REVISION and config["budget"] == BUDGET
    assert config["system_message"] == SYSTEM and config["tools"] == tools()
    assert config["new_training"] is False
    available = [t.id for t in source()]
    assert plan["task_ids"] == config["task_ids"]
    if split == "rl_training":
        assert config["task_ids"] == available[: len(config["task_ids"])]
        assert not set(config["task_ids"]) & {t.id for t in pilot_tasks()}
    else:
        assert config["task_ids"] == available
    if split != "final_test":
        assert not set(config["task_ids"]) & {t.id for t in final_tasks()}
    tasks = {t.id: t for t in source()}
    arms = config["arms"]
    seeded = config["seeded_actions"]
    assert seeded == (SEEDED_ACTIONS if config["seeded_failure"] else 0)
    records = {
        t + "--" + arm: load(root / "episodes" / arm / (t + ".json"))
        for t in config["task_ids"]
        for arm in arms
    }
    assert set(records) == set(config["ordered_jobs"])
    histories = {
        ident: scripted_history(tasks[r["task"]["task_id"]], r["events"][:seeded])
        for ident, r in records.items()
    }
    used, decisions = Counter(), Counter()
    finished, environment = set(), None
    generated = valid = 0
    for number in range(1, report["rounds"] + 1):
        folder = root / "rounds" / str(number)
        batch = load(folder / "public-requests.json")
        assert batch["batch_sha256"] == sha256(
            canonical({k: v for k, v in batch.items() if k != "batch_sha256"})
        )
        assert batch["arms"] == arms and batch["tools"] == tools()
        metadata = load(folder / "extracted/model-config.json")
        assert metadata["batch_sha256"] == batch["batch_sha256"] and metadata["round"] == number
        assert metadata["model_revision"] == REVISION and metadata["budget"] == BUDGET
        for name, adapter in arms.items():
            evidence = metadata["arms"][name]
            assert (
                evidence["base_initial_sha256"]
                == evidence["base_final_sha256"]
                == config["base_sha256"]
            )
            if adapter:
                assert evidence["adapter_file_sha256"] == adapter["file_sha256"]
                assert (
                    evidence["adapter_initial_sha256"]
                    == evidence["adapter_final_sha256"]
                    == adapter["tensor_sha256"]
                )
        current = {k: metadata[k] for k in ENVIRONMENT_KEYS}
        if split == "rl_training":
            assert current["seed"] == RL_CONFIG["seed"] and current["decoding"] == SAMPLING
            assert batch["sampling"] == SAMPLING and batch["cycle"] == config["cycle"]
        else:
            assert current["seed"] == 7 and current["decoding"] == {
                "do_sample": False,
                "num_beams": 1,
                "enable_thinking": False,
            }
        assert environment is None or current == environment
        environment = current
        assert (
            hashlib.sha256((folder / "extracted/chat_template.jinja").read_bytes()).hexdigest()
            == current["chat_template_sha256"]
        )
        local = load(next((folder / "upload").glob("*.ipynb")))
        remote = load(next((folder / "server").glob("*.ipynb")))
        assert [c["source"] for c in local["cells"]] == [c["source"] for c in remote["cells"]]
        assert load(folder / "server/kernel-metadata.json")["is_private"] is True
        round_check = load(folder / "round-verification.json")
        assert all(
            round_check[k] is True
            for k in ("server_private", "server_sources_match", "worker_complete")
        )
        archives = list((folder / "download").glob("*.zip"))
        assert len(archives) == 1
        assert hashlib.sha256(archives[0].read_bytes()).hexdigest() == round_check["archive_sha256"]
        rows = [
            json.loads(line)
            for line in (folder / "extracted/responses.jsonl").read_text().splitlines()
        ]
        by_id = {r["episode_id"]: r for r in rows}
        assert len(by_id) == len(rows) == len(batch["requests"])
        assert {r["episode_id"] for r in batch["requests"]} == set(records) - finished
        for request in batch["requests"]:
            ident = request["episode_id"]
            record, response = records[ident], by_id[ident]
            decisions[ident] += 1
            assert request["messages"] == histories[ident]
            assert request["max_new_tokens"] == min(512, 4096 - used[ident])
            assert response["decision"] == request["decision"] == decisions[ident]
            saved = record["model_calls"][decisions[ident] - 1]
            assert {k: saved[k] for k in response} == response
            if response.get("stop"):
                finished.add(ident)
                continue
            generated += 1
            used[ident] += response["output_tokens"]
            assert 0 < response["output_tokens"] <= request["max_new_tokens"]
            if split == "rl_training":
                assert len(response["completion_ids"]) == response["output_tokens"]
            event = record["events"][seeded + decisions[ident] - 1]
            observation = event["observation"]
            try:
                parsed = parsed_action(response.get("parsed"))
            except (ValueError, TypeError, KeyError, AttributeError):
                parsed = None
            try:
                raw = raw_native_action(response["raw"])
            except (ValueError, TypeError, KeyError):
                raw = None
            assert parsed == raw, ident
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
                valid += 1
                assert event["action"] == parsed and saved["native_error"] is None
                if parsed["tool"] == "submit":
                    assert observation == {"submitted": True}
                histories[ident] += [
                    response["parsed"],
                    tool_message(parsed["tool"], observation),
                ]
            if decisions[ident] == len(record["model_calls"]):
                finished.add(ident)
    assert finished == set(records)
    executor = DockerExecutor() if regrade else None
    regrades, rows, details = [], [], []
    for ident, record in records.items():
        task = tasks[record["task"]["task_id"]]
        assert record["task"] == task.public() and record["task_sha256"] == task.fingerprint()
        source = task.buggy
        for index, event in enumerate(record["events"], 1):
            assert event["index"] == index
            action = event["action"]
            if action.get("tool") == "edit" and "error" not in event["observation"]:
                validate_source(action["source"], task.signature)
                source = action["source"]
            assert event["source_sha256"] == sha256(source)
        assert source == record["final_source"] and len(record["events"]) <= 8
        assert used[ident] <= 4096
        result = record["result"]
        if result["termination"] == "submitted":
            assert result["success"] == (result["public"]["passed"] and result["private"]["passed"])
            if regrade:
                public = executor.run(source, task.public_cases)
                private = executor.run(source, task.private_cases)
                assert public["status"] == private["status"] == "graded"
                assert (public["passed"] and private["passed"]) == result["success"]
                regrades.append({"episode_id": ident, "public": public, "private": private})
        else:
            assert result["success"] is False
        rows.append(
            {
                "episode_id": ident,
                "task_id": task.id,
                "condition": record["condition"],
                "success": result["success"],
                "termination": result["termination"],
            }
        )
        details.append(
            "<details><summary>"
            + escape(ident)
            + " · "
            + ("PASS" if result["success"] else "FAIL")
            + " · "
            + escape(result["termination"])
            + "</summary><h3>Final candidate</h3><pre>"
            + escape(source)
            + "</pre><h3>Actions</h3><pre>"
            + escape(
                json.dumps(
                    [
                        {
                            "index": e["index"],
                            "scripted": e["index"] <= seeded,
                            "action": e["action"],
                            "observation": e["observation"],
                        }
                        for e in record["events"]
                    ],
                    indent=2,
                    ensure_ascii=False,
                )
            )
            + "</pre></details>"
        )
    for arm in arms:
        selected = [r for r in rows if r["condition"] == arm]
        assert report["arms"][arm]["successes"] == sum(r["success"] for r in selected)
    summary = {
        "verified": True,
        "seeded_failure": config["seeded_failure"],
        "episodes": len(records),
        "tasks": len(config["task_ids"]),
        "generated_calls": generated,
        "valid_native_calls": valid,
        "invalid_native_calls": generated - valid,
        "identical_environment_across_rounds": True,
        "checkpoints_reloaded_and_unchanged": True,
        "public_only_conversations_verified": True,
        "split": split,
        "sealed_final_tasks_used": len(config["task_ids"]) if split == "final_test" else 0,
        "fresh_submitted_regrades": regrades,
        "arms": report["arms"],
        "rows": rows,
    }
    write_json(root / "independent-verification.json", summary)
    label = {
        "development": "ACTUAL AUTONOMOUS ATTEMPTS · DEVELOPMENT TASKS",
        "final_test": "ACTUAL AUTONOMOUS ATTEMPTS · SEALED FINAL TEST, OPENED ONCE",
        "rl_training": "SAMPLED RL TRAINING ATTEMPTS · NOT AN EVALUATION SCORE",
    }[split]
    if config["seeded_failure"]:
        label = "SEEDED RECOVERY PROBE · NOT AN AUTONOMOUS REPAIR SCORE"
    table = "".join(
        "<tr><td>"
        + escape(t)
        + "</td>"
        + "".join(
            "<td>" + ("PASS" if records[t + "--" + a]["result"]["success"] else "FAIL") + "</td>"
            for a in arms
        )
        + "</tr>"
        for t in config["task_ids"]
    )
    html = (
        "<!doctype html><html><head><meta charset='utf-8'><title>RepairLab comparison</title><style>"
        "body{font:16px system-ui;max-width:1000px;margin:48px auto;padding:0 24px;color:#183139;background:#f5f7f4}"
        "table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:10px;border-bottom:1px solid #d7dfd9}"
        "details{background:white;padding:14px;margin:10px 0;border:1px solid #d7dfd9;border-radius:8px}pre{white-space:pre-wrap;font-size:13px}"
        "</style></head><body><small>"
        + label
        + "</small><h1>Matched checkpoint comparison</h1><table><thead><tr><th>Task</th>"
        + "".join("<th>" + escape(a) + "</th>" for a in arms)
        + "</tr></thead><tbody>"
        + table
        + "</tbody></table><p>"
        + " · ".join(
            escape(a)
            + ": "
            + str(report["arms"][a]["successes"])
            + "/"
            + str(len(config["task_ids"]))
            for a in arms
        )
        + "</p>"
        + "".join(details)
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
