"""Versioned HumanEvalFix Python adaptation using docstrings and public examples.

This is a local adaptation, not a reproduction of the authors' leaderboard protocol.
The external assert runner must NOT be reused as a secure RL reward oracle.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

from .experiments import source_fingerprint, write_json
from .tasks import canonical, sha256

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "humanevalfix"
PROTOCOL = "humanevalfix-docs-examples-single-response-v1"


def load():
    snapshot = json.loads((DATA / "snapshot.json").read_text())
    payload = (DATA / "humanevalpack-python.jsonl").read_text()
    if sha256(payload) != snapshot["file_sha256"]:
        raise ValueError("HumanEvalFix snapshot checksum mismatch")
    rows = [json.loads(line) for line in payload.splitlines() if line.strip()]
    if len(rows) != 164 or len({r["task_id"] for r in rows}) != 164:
        raise ValueError("Unexpected HumanEvalFix task count")
    return rows, snapshot


def public_tasks():
    rows, _ = load()
    return [
        {
            "task_id": r["task_id"],
            "specification": r["docstring"],
            "source": r["declaration"] + r["buggy_solution"],
            "public_tests_source": r["example_test"],
            "entry_point": r["entry_point"],
        }
        for r in rows
    ]


def fingerprint():
    return sha256(canonical({"snapshot": load()[1], "protocol": PROTOCOL}))


def start(destination, executor, kind):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(
        destination / "config.json",
        {
            "kind": kind,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "protocol": PROTOCOL,
            "snapshot": load()[1],
            "source_sha256": source_fingerprint(),
            "executor": executor.metadata(),
            "verification": "external_assert_suite",
            "limitations": "External tests execute with candidate code. Not a secure RL oracle.",
        },
    )
    write_json(destination / "status.json", {"status": "running", "kind": kind})
    return destination


def control_matches(result, expected):
    return result["passed"] is expected and (
        result["status"] == "graded"
        or (not expected and result["status"] in {"timeout", "resource_limit", "execution_error"})
    )


def admit(executor, destination):
    destination = start(destination, executor, "external_authored_controls_not_model_results")
    rows, _ = load()
    results = []
    for row in rows:
        controls = {}
        for name, field, expected in (
            ("reference", "canonical_solution", True),
            ("broken", "buggy_solution", False),
        ):
            result = executor.run_suite(row["declaration"] + row[field], row["test"])
            controls[name] = {
                "expected_pass": expected,
                "actual": result,
                "matched": control_matches(result, expected),
            }
        results.append({"task_id": row["task_id"], "controls": controls})
        print(row["task_id"], {k: v["matched"] for k, v in controls.items()}, flush=True)
    report = {
        "tasks": len(rows),
        "model_episodes": 0,
        "reference_passes": sum(r["controls"]["reference"]["actual"]["passed"] for r in results),
        "broken_rejected": sum(not r["controls"]["broken"]["actual"]["passed"] for r in results),
        "all_matched": all(c["matched"] for r in results for c in r["controls"].values()),
        "source_unchanged": source_fingerprint()
        == json.loads((destination / "config.json").read_text())["source_sha256"],
        "rows": results,
    }
    write_json(destination / "report.json", report)
    return report


def score(executor, predictions, metadata, destination):
    metadata_path = Path(metadata)
    metadata = json.loads(metadata_path.read_text())
    if metadata.get("protocol") != PROTOCOL or metadata.get("manifest_sha256") != fingerprint():
        raise ValueError("Wrong external dataset version or protocol")
    rows, _ = load()
    by_id = {r["task_id"]: r for r in rows}
    outputs = {}
    for line in Path(predictions).read_text().splitlines():
        prediction = json.loads(line)
        ident = prediction.get("task_id")
        if ident not in by_id or ident in outputs:
            raise ValueError("Unknown or duplicate external prediction")
        outputs[ident] = prediction
    destination = start(destination, executor, "external_single_response_model_evaluation")
    write_json(destination / "model-config.json", metadata)
    results = []
    for row in rows:
        prediction = outputs.get(row["task_id"])
        if prediction is None or prediction.get("generation_error"):
            result = {"passed": False, "status": "missing_or_failed_generation"}
        else:
            try:
                result = executor.run_suite(prediction.get("source"), row["test"])
            except ValueError:
                result = {"passed": False, "status": "invalid_patch"}
        record = {"task_id": row["task_id"], "result": result, "model_output": prediction}
        results.append(record)
        write_json(destination / "episodes" / (row["task_id"].replace("/", "-") + ".json"), record)
        print(row["task_id"], result["passed"], flush=True)
    report = {
        "tasks": len(rows),
        "successes": sum(r["result"]["passed"] for r in results),
        "protocol": PROTOCOL,
        "missing_predictions": len(rows) - len(outputs),
        "predictions_sha256": sha256(Path(predictions).read_text()),
        "source_unchanged": source_fingerprint()
        == json.loads((destination / "config.json").read_text())["source_sha256"],
        "rows": results,
    }
    write_json(destination / "report.json", report)
    return report
