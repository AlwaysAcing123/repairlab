"""Admission controls, scripted demonstration, and scoring of real model outputs."""

import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

from .episode import INTERFACE_VERSION, Episode
from .tasks import canonical, manifest, pilot_tasks, sha256


def source_fingerprint():
    return sha256(
        canonical(
            {p.name: sha256(p.read_text()) for p in sorted(Path(__file__).parent.glob("*.py"))}
        )
    )


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def new_run(destination, kind, executor):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    config = {
        "kind": kind,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "interface_version": INTERFACE_VERSION,
        "source_sha256": source_fingerprint(),
        "host_python": platform.python_version(),
        "executor": executor.metadata(),
        "manifest": manifest(),
        "split": "development",
        "final_test_tasks": 0,
    }
    write_json(destination / "config.json", config)
    write_json(destination / "status.json", {"status": "running", "kind": kind})
    return destination, config


def admission(executor, destination):
    destination, config = new_run(destination, "authored_controls_not_model_results", executor)
    rows = []
    for task in pilot_tasks():
        cases = task.public_cases + task.private_cases
        controls = {}
        for name, candidate, expected in (
            ("broken", task.buggy, False),
            ("reference", task.reference, True),
            ("alternative", task.alternative, True),
            ("public_shortcut", task.shortcut(), False),
        ):
            actual = executor.run(candidate, cases)
            controls[name] = {
                "expected_pass": expected,
                "actual": actual,
                "matched": actual["status"] == "graded" and actual["passed"] is expected,
            }
        shortcut_public = executor.run(task.shortcut(), task.public_cases)
        row = {
            "task_id": task.id,
            "controls": controls,
            "shortcut_passes_public": shortcut_public["passed"],
            "all_matched": all(c["matched"] for c in controls.values())
            and shortcut_public["passed"],
        }
        rows.append(row)
        write_json(destination / "tasks" / f"{task.id}.json", row)
        print(f"{task.id}: {'PASS' if row['all_matched'] else 'FAIL'}", flush=True)
    report = {
        "kind": config["kind"],
        "tasks": len(rows),
        "controls": len(rows) * 4,
        "matched_controls": sum(c["matched"] for r in rows for c in r["controls"].values()),
        "shortcut_public_checks_passed": sum(r["shortcut_passes_public"] for r in rows),
        "all_matched": all(r["all_matched"] for r in rows),
        "model_episodes": 0,
        "source_unchanged": source_fingerprint() == config["source_sha256"],
        "manifest_sha256": config["manifest"]["manifest_sha256"],
        "rows": rows,
    }
    write_json(destination / "report.json", report)
    return report


def demo(executor, task, destination):
    destination, _ = new_run(destination, "scripted_reference_replay_not_a_model", executor)
    episode = Episode(task, executor)
    for action in (
        {"tool": "test"},
        {"tool": "edit", "source": task.reference},
        {"tool": "test"},
        {"tool": "submit"},
    ):
        episode.step(action)
    record = {"actor": "scripted_reference", **episode.record()}
    write_json(destination / "episode.json", record)
    # A small portable HTML replay without a frontend dependency or invented reasoning.
    from html import escape

    events = "".join(
        f"<article><h2>{event['index']}. {escape(event['action']['tool'])}</h2>"
        f"<pre>{escape(event.get('diff', ''))}</pre>"
        f"<pre>{escape(json.dumps(event['observation'], indent=2, ensure_ascii=False))}</pre></article>"
        for event in record["events"]
    )
    html = """<!doctype html><meta charset="utf-8"><title>RepairLab replay</title>
<style>body{font:16px system-ui;max-width:900px;margin:40px auto;padding:0 24px;
background:#f8f8f5;color:#182b31}h1{font-size:32px}article{background:white;padding:16px;
margin:20px 0;border:1px solid #d8deda;border-radius:8px}pre{overflow:auto;white-space:pre-wrap;
font-size:13px}small{color:#50636a}</style>"""
    html += (
        "<small>SCRIPTED REFERENCE · SAVED REPLAY · NO MODEL INVOLVED</small>"
        f"<h1>{escape(task.id)}</h1><p>{escape(task.specification)}</p>"
        f"<h2>Starting function</h2><pre>{escape(task.buggy)}</pre>"
        + events
        + f"<h2>Independent grade: {'passed' if record['result']['success'] else 'failed'}</h2>"
        + f"<p>{len(task.public_cases)} public and {len(task.private_cases)} private checks. "
        + escape(task.domain)
        + " Passing is limited to this checked domain.</p>"
    )
    (destination / "replay.html").write_text(html)
    return record


def score_patches(executor, predictions_path, metadata_path, destination):
    """One model response per task, no execution feedback. Never silently drop failures."""
    metadata = json.loads(Path(metadata_path).read_text())
    required = {"model", "model_revision", "decoding", "seed", "protocol", "manifest_sha256"}
    if not required <= metadata.keys() or metadata["protocol"] != "single-response-v1":
        raise ValueError("Missing model metadata or wrong protocol; see notebook and README")
    tasks = pilot_tasks()
    if metadata["manifest_sha256"] != manifest(tasks)["manifest_sha256"]:
        raise ValueError("Predictions target a different version of the dataset")
    predictions = {}
    for line in Path(predictions_path).read_text().splitlines():
        row = json.loads(line)
        ident = row.get("task_id")
        if ident in predictions:
            raise ValueError("Duplicate prediction: " + str(ident))
        if ident not in {t.id for t in tasks}:
            raise ValueError("Unknown task: " + str(ident))
        predictions[ident] = row
    destination, config = new_run(destination, "single_response_model_evaluation", executor)
    write_json(destination / "model-config.json", metadata)
    rows = []
    for task in tasks:
        prediction = predictions.get(task.id)
        episode = Episode(task, executor, max_actions=2)
        if prediction is None:
            episode.stop("missing_prediction")
        elif prediction.get("generation_error"):
            episode.stop("generation_error")
        else:
            episode.step({"tool": "edit", "source": prediction.get("source")})
            if "error" in episode.events[-1]["observation"]:
                episode.stop("invalid_patch")
            else:
                episode.step({"tool": "submit"})
        record = {"actor": "model", "model_output": prediction, **episode.record()}
        write_json(destination / "episodes" / f"{task.id}.json", record)
        rows.append(
            {
                "task_id": task.id,
                "family": task.family,
                "success": record["result"]["success"],
                "termination": record["result"]["termination"],
                "episode": f"episodes/{task.id}.json",
            }
        )
        print(
            f"{task.id}: {'PASS' if rows[-1]['success'] else rows[-1]['termination']}", flush=True
        )
    report = {
        "kind": config["kind"],
        "split": "development",
        "tasks": len(tasks),
        "successes": sum(r["success"] for r in rows),
        "success_rate": sum(r["success"] for r in rows) / len(tasks),
        "missing_predictions": len(tasks) - len(predictions),
        "termination_counts": dict(Counter(r["termination"] for r in rows)),
        "by_family": {
            f: {
                "tasks": sum(r["family"] == f for r in rows),
                "successes": sum(r["family"] == f and r["success"] for r in rows),
            }
            for f in sorted({t.family for t in tasks})
        },
        "source_unchanged": source_fingerprint() == config["source_sha256"],
        "predictions_sha256": sha256(Path(predictions_path).read_text()),
        "rows": rows,
    }
    write_json(destination / "report.json", report)
    return report
