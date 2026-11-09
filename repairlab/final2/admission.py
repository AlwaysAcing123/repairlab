"""Admission controls and separation checks for the second sealed test. No model is involved.

The new tasks must be unseen relative to every earlier program, including the opened
first final test and the RL programs, so the leakage check compares them with all 120.
Earlier programs are compared under one label, which skips their already-reviewed
pairs among themselves.
"""

from dataclasses import replace
from pathlib import Path

from ..curriculum.splits import _task_entry, entries, leakage_report
from ..experiments import write_json
from ..rl.admission import rl_entries
from ..tasks import canonical, sha256
from . import fingerprint
from .programs import final2_tasks

SPLIT = "final_test_2"
# Hand-reviewed November 5, 2026. Behavior pairs are string or list checks that agree
# only because both return False on at least 94% of the inputs both accept. Source pairs
# share a one-line skeleton such as `all(... for x in xs)`, a set operation, `a // b + 1`,
# a stepped `range`, or `range(a, b + 1)`, and compute different functions with
# different fixes. Two drafts were replaced instead: `never-overdrawn` had the running
# balance algorithm of training program `balanced-parens`, and `column-sums` nearly
# copied the code and wording of opened final-test task `transpose-rows`. A new near
# pair fails tests until someone reads it and adds it here.
REVIEWED_FINAL2_NEAR_PAIRS = frozenset(
    frozenset(pair)
    for pair in (
        ("balanced-parens", "semver-parts"),
        ("chunk-items", "every-nth"),
        ("chunk-items", "window-sums"),
        ("dev-quotient", "seat-row"),
        ("email-shape", "chess-square"),
        ("email-shape", "semver-parts"),
        ("granted-permissions", "index-in-range"),
        ("granted-permissions", "only-in-one"),
        ("hex-color-code", "semver-parts"),
        ("ipv4-octets", "chess-square"),
        ("positive-batch", "chess-square"),
        ("quorum-majority", "seat-row"),
        ("remove-at-index", "ends-match"),
        ("remove-at-index", "every-nth"),
        ("retry-delays", "hour-labels"),
        ("same-letter-counts", "only-in-one"),
        ("uniform-values", "semver-parts"),
        ("zip-code", "chess-square"),
    )
)


def final2_entries():
    return [_task_entry(t, "f2-" + t.id, SPLIT) for t in final2_tasks()]


def reject_final2(tasks):
    """Refuse any model-facing use of a second-test task, even under a different ID."""
    sealed = set()
    for task in final2_tasks():
        sealed |= {task.id, task.specification, task.buggy, task.reference}
    for task in tasks:
        if sealed & {task.id, task.specification, task.buggy, task.reference}:
            raise ValueError("Sealed second-test task in a model-facing path: " + task.id)


def final2_leakage_report():
    earlier = [replace(e, split="earlier") for e in entries() + rl_entries()]
    report = leakage_report(earlier + final2_entries())
    report["earlier_programs_compared"] = len(earlier)
    report["unreviewed_near_pairs"] = [
        n
        for n in report["near_pairs_for_review"]
        if frozenset((n["left"], n["right"])) not in REVIEWED_FINAL2_NEAR_PAIRS
    ]
    return report


def final2_manifest():
    body = {
        "split": SPLIT,
        "tasks": [
            {
                "task_id": t.id,
                "group_id": "f2-" + t.id,
                "family": t.family,
                "bug_category": t.bug_category,
                "public_cases": len(t.public_cases),
                "private_cases": len(t.private_cases),
                "task_sha256": t.fingerprint(),
            }
            for t in final2_tasks()
        ],
        "earlier_programs_compared": len(entries() + rl_entries()),
        "final_test_model_evaluations": 0,
    }
    return {**body, "manifest_sha256": sha256(canonical(body))}


def admit(executor, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for task in final2_tasks():
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
    leakage = final2_leakage_report()
    report = {
        "kind": "authored_controls_not_model_results",
        "tasks": len(rows),
        "all_matched": all(r["all_matched"] for r in rows),
        "controls": sum(len(r["controls"]) for r in rows),
        "matched_controls": sum(c["matched"] for r in rows for c in r["controls"].values()),
        "model_episodes": 0,
        "final2_sha256": fingerprint(),
        "manifest_sha256": final2_manifest()["manifest_sha256"],
        "leakage": leakage,
        "rows": rows,
    }
    write_json(destination / "final2-manifest.json", final2_manifest())
    write_json(destination / "report.json", report)
    return report
