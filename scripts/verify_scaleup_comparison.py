"""Audit a comparison across model sizes: conversations, checkpoints, budgets, grades.

Covers development comparisons, 1.7B RL rollout cycles, the one-time second final
test, and HumanEvalFix native, which share one episode engine. ``--rerun`` checks that
an arm reproduced the saved trajectories of an earlier run exactly. Writes its
verification and HTML into the run directory.
"""

import argparse
import hashlib
import json
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_matched_comparison import scripted_history  # noqa: E402
from verify_native_diagnostic import raw_native_action  # noqa: E402

from repairlab.curriculum.comparison import ENVIRONMENT_KEYS, SEEDED_ACTIONS  # noqa: E402
from repairlab.curriculum.transcript import tool_message  # noqa: E402
from repairlab.execution import DockerExecutor, validate_source  # noqa: E402
from repairlab.experiments import write_json  # noqa: E402
from repairlab.final2.programs import final2_tasks  # noqa: E402
from repairlab.native_repair import BUDGET, SYSTEM  # noqa: E402
from repairlab.native_tools import parsed_action, tools  # noqa: E402
from repairlab.scaleup.comparison import (  # noqa: E402
    GREEDY,
    PROTOCOL,
    debugging_metrics,
    split_tasks,
)
from repairlab.scaleup.models import MODELS  # noqa: E402
from repairlab.tasks import canonical, sha256  # noqa: E402

LABELS = {
    "development": "ACTUAL AUTONOMOUS ATTEMPTS · DEVELOPMENT TASKS",
    "final_test_2": "ACTUAL AUTONOMOUS ATTEMPTS · SEALED SECOND FINAL TEST, OPENED ONCE",
    "humanevalfix_native": "ACTUAL AUTONOMOUS ATTEMPTS · HUMANEVALFIX NATIVE, RUN ONCE",
    "rl_training": "SAMPLED RL TRAINING ATTEMPTS · NOT AN EVALUATION SCORE",
}


def load(path):
    return json.loads(path.read_text())


def strict_raw_action(raw):
    """The independent parse, also refusing stray tool tags.

    A sampled 1.7B turn once wrote a valid test call followed by an invented tool
    response closed with a second ``</tool_call>``. The native parser rejects such
    text; the earlier regex parser accepted the first block, so it is tightened here.
    """
    if raw.count("<tool_call>") != 1 or raw.count("</tool_call>") != 1 or "<tool_response>" in raw:
        raise ValueError("Stray native tool tags")
    return raw_native_action(raw)


def verify(root, regrade=False, tasks=None, protocol=PROTOCOL, decoding=GREEDY):
    """``tasks`` and ``protocol`` are given by the RL audit, whose task list is per cycle."""
    root = Path(root)
    config, report = load(root / "config.json"), load(root / "report.json")
    assert load(root / "status.json")["status"] == "completed"
    split = config["split"]
    assert config["protocol"] == protocol and config["budget"] == BUDGET
    assert config["system_message"] == SYSTEM and config["tools"] == tools()
    assert config["new_training"] is False
    for key, model in config["models"].items():
        assert model == MODELS[key], key
    if tasks is None:
        available = split_tasks(split)
        assert config["task_ids"] == [t.id for t in available][: len(config["task_ids"])]
        assert config["full_task_list"] == (len(config["task_ids"]) == len(available))
        tasks = available
    tasks = {t.id: t for t in tasks}
    if split != "final_test_2":
        sealed = {t.fingerprint() for t in final2_tasks()}
        assert not {tasks[i].fingerprint() for i in config["task_ids"]} & sealed
    else:
        assert config["full_task_list"] and not config["seeded_failure"]
    plan = load(root / "experiment-plan.json") if (root / "experiment-plan.json").exists() else {}
    if "task_ids" in plan:
        assert plan["task_ids"] == config["task_ids"]
    arms, seeded = config["arms"], config["seeded_actions"]
    assert seeded == (SEEDED_ACTIONS if config["seeded_failure"] else 0)
    for spec in arms.values():
        assert spec["model_key"] in config["models"]
        if spec["adapter"]:
            assert (
                spec["adapter"]["base_sha256"] == config["models"][spec["model_key"]]["base_sha256"]
            )
    records = {
        t + "--" + a: load(root / "episodes" / a / (t + ".json"))
        for t in config["task_ids"]
        for a in arms
    }
    assert set(records) == set(config["ordered_jobs"])
    histories = {
        ident: scripted_history(tasks[r["task"]["task_id"]], r["events"][:seeded])
        for ident, r in records.items()
    }
    used, decisions, finished = {i: 0 for i in records}, {i: 0 for i in records}, set()
    environment, generated, valid = None, 0, 0
    for number in range(1, report["rounds"] + 1):
        folder = root / "rounds" / str(number)
        batch = load(folder / "public-requests.json")
        assert batch["batch_sha256"] == sha256(
            canonical({k: v for k, v in batch.items() if k != "batch_sha256"})
        )
        assert batch["arms"] == arms and batch["tools"] == tools()
        assert batch["models"] == config["models"]
        metadata = load(folder / "extracted/model-config.json")
        assert metadata["batch_sha256"] == batch["batch_sha256"] and metadata["round"] == number
        assert metadata["budget"] == BUDGET
        for key, model in config["models"].items():
            evidence = metadata["models"][key]
            assert evidence["model_revision"] == model["model_revision"]
            assert evidence["base_initial_sha256"] == evidence["base_final_sha256"]
            assert evidence["base_final_sha256"] == model["base_sha256"]
        for name, spec in arms.items():
            evidence, adapter = metadata["arms"][name], spec["adapter"]
            assert evidence["model_key"] == spec["model_key"]
            if adapter:
                assert evidence["adapter_file_sha256"] == adapter["file_sha256"]
                assert (
                    evidence["adapter_initial_sha256"]
                    == evidence["adapter_final_sha256"]
                    == adapter["tensor_sha256"]
                )
            else:
                assert evidence["adapter_file_sha256"] is None
        current = {k: metadata[k] for k in ENVIRONMENT_KEYS}
        assert current["decoding"] == decoding == batch["decoding"]
        assert current["seed"] == batch["seed"]
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
        check = load(folder / "round-verification.json")
        assert all(check[k] is True for k in ("server_private", "server_sources_match"))
        assert check["worker_complete"] is True
        archives = list((folder / "download").glob("*.zip"))
        assert len(archives) == 1
        assert hashlib.sha256(archives[0].read_bytes()).hexdigest() == check["archive_sha256"]
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
            assert request["messages"] == histories[ident], ident
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
                raw = strict_raw_action(response["raw"])
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
                histories[ident] += [response["parsed"], tool_message(parsed["tool"], observation)]
            if decisions[ident] == len(record["model_calls"]):
                finished.add(ident)
    assert finished == set(records)
    executor = DockerExecutor() if regrade else None
    regrades, rows, details, infrastructure = [], [], [], []
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
                saved = (result["public"]["status"], result["private"]["status"])
                fresh = (public["status"], private["status"])
                if fresh != saved and "timeout" in saved and "timeout" not in fresh:
                    # The host slept mid-grade once, turning a normal grade into a timeout.
                    # Reported, never silently corrected: the saved grade is the score.
                    infrastructure.append(
                        {
                            "episode_id": ident,
                            "saved_statuses": saved,
                            "regraded_statuses": fresh,
                            "regrade_passes": public["passed"] and private["passed"],
                        }
                    )
                else:
                    # A timeout or resource limit is a failing grade and must recur.
                    assert fresh == saved
                    assert (public["passed"] and private["passed"]) == result["success"]
                regrades.append({"episode_id": ident, "public": public, "private": private})
        else:
            assert result["success"] is False
        debugging = debugging_metrics(task, record)
        rows.append(
            {
                "episode_id": ident,
                "task_id": task.id,
                "condition": record["condition"],
                "success": result["success"],
                "termination": result["termination"],
                "debugging": debugging,
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
            + "</pre><h3>Model turns and observations</h3><pre>"
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
            + "</pre><h3>Raw model text</h3><pre>"
            + escape("\n---\n".join(c.get("raw", "[stopped]") for c in record["model_calls"]))
            + "</pre></details>"
        )
    for name in arms:
        selected = [r for r in rows if r["condition"] == name]
        saved = report["arms"][name]
        assert saved["successes"] == sum(r["success"] for r in selected)
        if split != "rl_training":
            assert saved["repeated_edits"] == sum(
                r["debugging"]["repeated_edits"] for r in selected
            )
            assert saved["real_recoveries"] == sum(
                r["debugging"]["real_recovery"] for r in selected
            )
    summary = {
        "verified": True,
        "split": split,
        "seeded_failure": config["seeded_failure"],
        "full_task_list": config["full_task_list"],
        "episodes": len(records),
        "tasks": len(config["task_ids"]),
        "generated_calls": generated,
        "valid_native_calls": valid,
        "invalid_native_calls": generated - valid,
        "identical_environment_across_rounds": True,
        "checkpoints_reloaded_and_unchanged": True,
        "public_only_conversations_verified": True,
        "sealed_final_2_tasks_used": len(config["task_ids"]) if split == "final_test_2" else 0,
        "fresh_submitted_regrades": regrades,
        "infrastructure_timeouts": infrastructure,
        "arms": {
            name: {k: v for k, v in arm.items() if not isinstance(v, dict)}
            for name, arm in report["arms"].items()
        },
        "rows": rows,
    }
    write_json(root / "independent-verification.json", summary)
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
        + LABELS[split]
        + "</small><h1>Matched comparison across model sizes</h1><table><thead><tr><th>Task</th>"
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


def reproduced(root, arm, earlier):
    """Compare an arm's trajectories with the same arm's saved runs, task by task.

    ``earlier`` is a list of (run directory, arm name there). A task matches when every
    model turn's raw text and every action, observation, and source hash is identical.
    Grading times are not compared.
    """

    def trajectory(record):
        return [c.get("raw") for c in record["model_calls"]], [
            (
                e["action"],
                {k: v for k, v in e["observation"].items() if k != "elapsed_seconds"},
                e["source_sha256"],
            )
            for e in record["events"]
        ]

    root = Path(root)
    config = load(root / "config.json")
    matched, compared, differing = 0, 0, []
    for run, name in earlier:
        for path in sorted((Path(run) / "episodes" / name).glob("*.json")):
            task_id = path.stem
            if task_id not in config["task_ids"]:
                continue
            old, new = load(path), load(root / "episodes" / arm / path.name)
            compared += 1
            same = trajectory(old) == trajectory(new)
            if same:
                matched += 1
            else:
                differing.append(task_id)
    return {"arm": arm, "compared": compared, "identical": matched, "differing": differing}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--regrade", action="store_true")
    parser.add_argument(
        "--rerun",
        action="append",
        default=[],
        help="ARM=RUN:OLDARM[,RUN:OLDARM]; checks the arm reproduced those saved episodes",
    )
    args = parser.parse_args()
    verify(args.path, args.regrade)
    checks = []
    for item in args.rerun:
        arm, sources = item.split("=", 1)
        earlier = [tuple(s.rsplit(":", 1)) for s in sources.split(",")]
        checks.append(reproduced(args.path, arm, earlier))
    if checks:
        write_json(args.path / "rerun-verification.json", checks)
        print(json.dumps(checks, indent=2))
