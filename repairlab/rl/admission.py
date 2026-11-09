"""Admission controls and separation checks for the RL programs. No model is involved.

The RL programs form their own split, so the shared leakage check compares them with
the earlier training programs too: an RL program must be unseen, not just held apart
from development and the sealed final test.
"""

from pathlib import Path

from ..curriculum.final_test import final_tasks
from ..curriculum.splits import (
    REVIEWED_NEAR_PAIRS,
    _task_entry,
    entries,
    leakage_report,
    reject_final,
)
from ..experiments import write_json
from ..tasks import canonical, pilot_tasks, sha256
from ..tool_warmup import programs as warmup_programs
from . import fingerprint
from .programs import rl_programs

SPLIT = "rl_training"
# Hand-reviewed November 4, 2026. The source pairs are one-line comparisons of three
# arguments or a list accumulator, which anonymize to nearly the same tokens while
# computing different functions with different fixes. `positive-batch`/`zip-code`
# agree only because both return False for almost every sampled input. Draft program
# `strong-pin` shared the length-and-every-item check of sealed `ipv4-octets`, whose
# bug is a missing length check, so it was replaced. A new near pair fails tests.
REVIEWED_RL_NEAR_PAIRS = frozenset(
    frozenset(pair)
    for pair in (
        ("adjacent-differences", "matrix-diagonal"),
        ("cache-expiry", "can-afford"),
        ("cache-expiry", "within-tolerance"),
        ("closed-window", "can-afford"),
        ("closed-window", "within-tolerance"),
        ("flatten-once", "clip-negatives"),
        ("positive-batch", "zip-code"),
        ("quiet-hours", "within-tolerance"),
        ("retry-decision", "can-afford"),
        ("retry-decision", "within-tolerance"),
    )
)


def rl_entries():
    return [_task_entry(t, "rl-" + t.id, SPLIT) for t in rl_programs()]


def reject_evaluation(tasks):
    """Refuse development and sealed tasks in RL training, even under another ID."""
    reject_final(tasks)
    held = set()
    for task in pilot_tasks():
        held |= {task.id, task.specification, task.buggy, task.reference}
    for task in tasks:
        if held & {task.id, task.specification, task.buggy, task.reference}:
            raise ValueError("Development task in an RL training path: " + task.id)


def rl_leakage_report():
    report = leakage_report(entries() + rl_entries())
    reviewed = REVIEWED_NEAR_PAIRS | REVIEWED_RL_NEAR_PAIRS
    report["unreviewed_near_pairs"] = [
        n
        for n in report["near_pairs_for_review"]
        if frozenset((n["left"], n["right"])) not in reviewed
    ]
    return report


def rl_manifest():
    body = {
        "split": SPLIT,
        "tasks": [
            {
                "task_id": t.id,
                "group_id": "rl-" + t.id,
                "family": t.family,
                "bug_category": t.bug_category,
                "public_cases": len(t.public_cases),
                "private_cases": len(t.private_cases),
                "task_sha256": t.fingerprint(),
            }
            for t in rl_programs()
        ],
        "earlier_programs_compared": len(entries()),
        "warmup_programs": len(warmup_programs()),
        "final_test_model_evaluations": 0,
    }
    return {**body, "manifest_sha256": sha256(canonical(body))}


def admit(executor, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    tasks = rl_programs()
    reject_evaluation(tasks)
    rows = []
    for task in tasks:
        cases = task.public_cases + task.private_cases
        controls = {}
        for name, candidate, passes in (
            ("broken", task.buggy, False),
            ("reference", task.reference, True),
            ("alternative", task.alternative, True),
            ("public_shortcut", task.shortcut(), False),
        ):
            actual = executor.run(candidate, cases)
            controls[name] = {
                "expected_pass": passes,
                "actual": actual,
                "matched": actual["status"] == "graded" and actual["passed"] is passes,
            }
        checks = {
            "shortcut_passes_public": executor.run(task.shortcut(), task.public_cases)["passed"],
            "broken_fails_public": not executor.run(task.buggy, task.public_cases)["passed"],
        }
        matched = all(c["matched"] for c in controls.values()) and all(checks.values())
        rows.append({"task_id": task.id, "controls": controls, **checks, "all_matched": matched})
        print(f"{SPLIT}/{task.id}: {'PASS' if matched else 'FAIL'}", flush=True)
    leakage = rl_leakage_report()
    report = {
        "kind": "authored_controls_not_model_results",
        "tasks": len(rows),
        "all_matched": all(r["all_matched"] for r in rows),
        "controls": sum(len(r["controls"]) for r in rows),
        "matched_controls": sum(c["matched"] for r in rows for c in r["controls"].values()),
        "model_episodes": 0,
        "rl_sha256": fingerprint(),
        "manifest_sha256": rl_manifest()["manifest_sha256"],
        "sealed_final_tasks": len(final_tasks()),
        "leakage": leakage,
        "rows": rows,
    }
    write_json(destination / "rl-manifest.json", rl_manifest())
    write_json(destination / "report.json", report)
    return report
