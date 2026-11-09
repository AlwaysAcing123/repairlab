"""Audit saved batch outputs, public conversation histories, budgets and final counts."""

import argparse
import hashlib
import json
import sys
from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repairlab.batch_revision import parse_batch_action  # noqa: E402
from repairlab.experiments import source_fingerprint, write_json  # noqa: E402
from repairlab.revision import BUDGET, SYSTEM  # noqa: E402
from repairlab.tasks import canonical, manifest, pilot_tasks, sha256  # noqa: E402


def verify(root):
    root = Path(root)
    config = json.loads((root / "config.json").read_text())
    report = json.loads((root / "report.json").read_text())
    assert config["source_sha256"] == source_fingerprint()
    assert config["manifest"] == manifest()
    assert config["budget"] == BUDGET
    responses = {}
    environments = []
    rounds = sorted((root / "rounds").iterdir(), key=lambda p: int(p.name))
    for folder in rounds:
        batch = json.loads((folder / "public-requests.json").read_text())
        assert (
            sha256(canonical({k: v for k, v in batch.items() if k != "batch_sha256"}))
            == batch["batch_sha256"]
        )
        metadata = json.loads((folder / "extracted/model-config.json").read_text())
        assert metadata["batch_sha256"] == batch["batch_sha256"]
        assert metadata["initial_weight_sha256"] == metadata["final_weight_sha256"]
        environments.append(
            {
                k: metadata.get(k)
                for k in (
                    "initial_weight_sha256",
                    "model",
                    "model_revision",
                    "dtype",
                    "gpu",
                    "packages",
                    "seed",
                    "decoding",
                    "chat_template_sha256",
                )
            }
        )
        requests = {r["episode_id"]: r for r in batch["requests"]}
        outputs = [
            json.loads(line)
            for line in (folder / "extracted/responses.jsonl").read_text().splitlines()
        ]
        assert len(outputs) == len(requests)
        assert len({o["episode_id"] for o in outputs}) == len(outputs)
        assert {o["episode_id"] for o in outputs} == set(requests)
        for output in outputs:
            key = (output["episode_id"], output["decision"])
            assert key not in responses
            assert output["decision"] == requests[output["episode_id"]]["decision"]
            responses[key] = (requests[output["episode_id"]], output)
        remote_meta = json.loads((folder / "server/kernel-metadata.json").read_text())
        assert remote_meta["is_private"] is True
        archive = next((folder / "download").glob("*.zip"))
        verification = json.loads((folder / "round-verification.json").read_text())
        assert verification["archive_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
        assert '"KernelWorkerStatus.COMPLETE"' in (folder / "worker-status.txt").read_text()
    assert all(environment == environments[0] for environment in environments)
    rows, details, calls_seen = [], [], 0
    diagnostics = {
        arm: {
            "valid_test_calls": 0,
            "valid_edit_calls": 0,
            "invalid_actions": 0,
            "unchanged_final_sources": 0,
        }
        for arm in ("without_feedback", "with_feedback")
    }
    for task in pilot_tasks():
        for arm in ("without_feedback", "with_feedback"):
            ident = task.id + "--" + arm
            episode = json.loads((root / "episodes" / arm / (task.id + ".json")).read_text())
            assert episode["task"] == task.public()
            assert episode["task_sha256"] == task.fingerprint()
            assert episode["final_source_sha256"] == sha256(episode["final_source"])
            assert episode["max_actions"] == BUDGET["max_actions"]
            diagnostics[arm]["unchanged_final_sources"] += (
                episode["final_source"].strip() == task.buggy.strip()
            )
            assert len(episode["events"]) <= BUDGET["max_actions"]
            conversation = [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": canonical(task.public())},
            ]
            total_output = 0
            for decision, call in enumerate(episode["model_calls"], 1):
                request, output = responses[(ident, decision)]
                assert call["messages"] == request["messages"] == conversation
                assert {k: v for k, v in call.items() if k != "messages"} == output
                assert request["max_new_tokens"] == min(
                    BUDGET["max_call_output_tokens"],
                    BUDGET["max_episode_output_tokens"] - total_output,
                )
                total_output += call.get("output_tokens", 0)
                if call.get("stop"):
                    assert episode["result"]["termination"] == call["stop"]
                    continue
                assert call["output_tokens"] <= request["max_new_tokens"]
                assert (
                    call["input_tokens"] + request["max_new_tokens"] <= BUDGET["max_context_tokens"]
                )
                event = episode["events"][decision - 1]
                assert event["action"] == parse_batch_action(
                    call["raw"], config.get("allow_json_fences", False)
                )
                action, observation = event["action"], event["observation"]
                diagnostics[arm]["invalid_actions"] += "error" in observation
                if (
                    isinstance(action, dict)
                    and action.get("tool") == "edit"
                    and "error" not in observation
                ):
                    diagnostics[arm]["valid_edit_calls"] += 1
                if (
                    isinstance(action, dict)
                    and action.get("tool") == "submit"
                    and "error" not in observation
                ):
                    assert observation == {"submitted": True}
                if (
                    isinstance(action, dict)
                    and action.get("tool") == "test"
                    and "error" not in observation
                ):
                    diagnostics[arm]["valid_test_calls"] += 1
                    if arm == "without_feedback":
                        assert observation.get("feedback") == "unavailable"
                    else:
                        assert observation["total_cases"] == len(task.public_cases)
                        if observation.get("cases"):
                            assert [r["args"] for r in observation["cases"]] == [
                                c.args for c in task.public_cases
                            ]
                            assert [r["expected"] for r in observation["cases"]] == [
                                c.expected for c in task.public_cases
                            ]
                conversation.extend(
                    [
                        {"role": "assistant", "content": call["raw"]},
                        {"role": "user", "content": "Tool result: " + canonical(observation)},
                    ]
                )
            assert total_output <= BUDGET["max_episode_output_tokens"]
            result = episode["result"]
            success = (
                result["termination"] == "submitted"
                and result["public"]["passed"]
                and result["private"]["passed"]
            )
            assert success == result["success"] and int(success) == result["reward"]
            row = next(r for r in report["rows"] if r["episode_id"] == ident)
            assert row["success"] == success and row["actions"] == len(episode["events"])
            calls_seen += len(episode["model_calls"])
            rows.append(row)
            content = "".join(
                "<h4>Decision "
                + str(index + 1)
                + "</h4><pre>"
                + escape(call.get("raw", call.get("stop", "")))
                + "</pre>"
                + (
                    "<h4>Public tool observation</h4><pre>"
                    + escape(json.dumps(episode["events"][index]["observation"], indent=2))
                    + "</pre>"
                    if index < len(episode["events"])
                    else ""
                )
                for index, call in enumerate(episode["model_calls"])
            )
            details.append(
                "<details><summary>"
                + escape(ident)
                + " · "
                + ("PASS" if success else escape(result["termination"]))
                + "</summary><p>"
                + escape(task.specification)
                + "</p>"
                + content
                + "<h4>Final source</h4><pre>"
                + escape(episode["final_source"])
                + "</pre><h4>Independent grade</h4><pre>"
                + escape(json.dumps(result, indent=2))
                + "</pre></details>"
            )
    assert calls_seen == len(responses)
    for arm in ("without_feedback", "with_feedback"):
        selected = [r for r in rows if r["condition"] == arm]
        assert len(selected) == 24
        assert sum(r["success"] for r in selected) == report["arms"][arm]["successes"]
    summary = {
        "verified": True,
        "episodes": len(rows),
        "model_calls": calls_seen,
        "rounds": len(rounds),
        "source_and_task_versions_match": True,
        "all_initial_pair_prompts_identical": True,
        "all_archived_responses_match_traces": True,
        "all_conversation_histories_match_public_observations": True,
        "hidden_grade_feedback_absent": True,
        "budgets_respected": True,
        "same_original_model_weights_and_environment_across_rounds": True,
        "private_notebooks": True,
        "arms": report["arms"],
        "paired": report["paired"],
        "diagnostics": diagnostics,
    }
    write_json(root / "independent-verification.json", summary)
    intro = """<!doctype html><meta charset="utf-8"><title>RepairLab paired revision results</title>
<style>body{font:16px system-ui;max-width:960px;margin:48px auto;padding:0 24px;color:#183139;background:#f5f7f4}
details{background:white;padding:18px;margin:12px 0;border:1px solid #d7dfd9;border-radius:8px}
summary{cursor:pointer;font-weight:600}pre{white-space:pre-wrap;font-size:13px;overflow:auto}small{color:#53676b}</style>
<small>SAVED ORIGINAL-MODEL INTERACTIONS · DEVELOPMENT TASKS · NO TRAINING</small>
<h1>Paired revision-loop experiment</h1><p>Same model, prompts, action budget, and token limits.
Public execution feedback is available in one condition and withheld in the other.</p>"""
    intro += (
        "<pre>"
        + escape(json.dumps({"arms": report["arms"], "paired": report["paired"]}, indent=2))
        + "</pre>"
    )
    (root / "results.html").write_text(intro + "".join(details))
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    verify(parser.parse_args().path)
