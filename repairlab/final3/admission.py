"""Admission controls and separation checks for the third sealed test. No model is involved.

The new tasks must be unseen relative to every earlier program: 24 pilot, 30 curriculum
training, 24 first final-test, 30 RL, and 48 second final-test programs, plus the 12
legacy warm-up programs, 168 entries in all. Earlier programs are compared under one
label, which skips their already-reviewed pairs among themselves. A second check
compares the new tasks with every converted HumanEvalFix task, dropped ones included.
"""

from dataclasses import replace
from pathlib import Path

from ..curriculum.splits import _task_entry, entries, leakage_report
from ..experiments import write_json
from ..final2.admission import final2_entries
from ..rl.admission import rl_entries
from ..scaleup.humanevalfix_native import converted_tasks, native_entries
from ..tasks import canonical, sha256
from . import fingerprint
from .programs import final3_tasks

SPLIT = "final_test_3"
# Hand-reviewed November 9, 2026. The first report flagged 37 pairs. Real closeness was
# fixed in the drafts instead: one-line bugs that anonymized to an earlier one-liner
# (time-before, title-words, code-window, expand-tabs, merge-ready, and add-padded's zip),
# two domains too weak to separate the task from an earlier check (subsequence-of from
# substring membership, even-shares from a length test), and `highest-altitude`, a
# running maximum of prefix sums too close to development task `running-totals`, which
# became `cycle-length`. Every pair left is a behavior match on inputs where one side
# barely runs, listed below with the reason. A new near pair fails tests until read.
REVIEWED_FINAL3_NEAR_PAIRS = frozenset(
    frozenset(pair)
    for pair in (
        # Comparable only when the second argument is empty: the gcd loop and the
        # door-state event loop never run and return the first argument unchanged, as
        # add-counters and ledger-transfers then do. Every other input errors on one side.
        ("euclid-gcd", "add-counters"),
        ("euclid-gcd", "ledger-transfers"),
        ("door-state", "add-counters"),
        ("door-state", "ledger-transfers"),
        ("humanevalfix-13", "add-counters"),
        ("humanevalfix-13", "ledger-transfers"),
        # Comparable only on page-slice inputs with offset > limit, where the guessing loop
        # never starts and both return []; every other input errors on one side.
        ("page-slice", "binary-guesses"),
        # Comparable only on string lists where unbroken-days stops at an exact duplicate
        # or gets [], answering as unique-usernames does; it errors on all other strings.
        ("unique-usernames", "unbroken-days"),
        # Comparable only on the earlier checks' strings, which these list checks also
        # iterate, and both sides return False on at least 96% of them.
        ("semver-parts", "mountain-shape"),
        ("semver-parts", "price-tiers"),
        ("semver-parts", "unbroken-days"),
        ("chess-square", "price-tiers"),
    )
)


def final3_entries():
    return [_task_entry(t, "f3-" + t.id, SPLIT) for t in final3_tasks()]


def earlier_entries():
    return [replace(e, split="earlier") for e in entries() + rl_entries() + final2_entries()]


def reject_final3(tasks):
    """Refuse any model-facing use of a third-test task, even under a different ID."""
    sealed = set()
    for task in final3_tasks():
        sealed |= {task.id, task.specification, task.buggy, task.reference, task.alternative}
    for task in tasks:
        if sealed & {task.id, task.specification, task.buggy, task.reference, task.alternative}:
            raise ValueError("Sealed third-test task in a model-facing path: " + task.id)


def _mark_reviewed(report):
    report["unreviewed_near_pairs"] = [
        n
        for n in report["near_pairs_for_review"]
        if frozenset((n["left"], n["right"])) not in REVIEWED_FINAL3_NEAR_PAIRS
    ]
    return report


def final3_leakage_report():
    earlier = earlier_entries()
    report = leakage_report(earlier + final3_entries())
    report["earlier_programs_compared"] = len(earlier)
    return _mark_reviewed(report)


def final3_humanevalfix_report():
    """The new tasks against every converted HumanEvalFix task, by the same checks."""
    external = native_entries(converted_tasks())
    report = leakage_report(external + final3_entries())
    report["humanevalfix_tasks_compared"] = len(external)
    return _mark_reviewed(report)


def final3_manifest():
    body = {
        "split": SPLIT,
        "tasks": [
            {
                "task_id": t.id,
                "group_id": "f3-" + t.id,
                "family": t.family,
                "bug_category": t.bug_category,
                "public_cases": len(t.public_cases),
                "private_cases": len(t.private_cases),
                "task_sha256": t.fingerprint(),
            }
            for t in final3_tasks()
        ],
        "earlier_programs_compared": len(earlier_entries()),
        "final_test_model_evaluations": 0,
    }
    return {**body, "manifest_sha256": sha256(canonical(body))}


def admit(executor, destination):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    rows = []
    for task in final3_tasks():
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
    leakage = final3_leakage_report()
    external = final3_humanevalfix_report()
    report = {
        "kind": "authored_controls_not_model_results",
        "tasks": len(rows),
        "all_matched": all(r["all_matched"] for r in rows),
        "controls": sum(len(r["controls"]) for r in rows),
        "matched_controls": sum(c["matched"] for r in rows for c in r["controls"].values()),
        "model_episodes": 0,
        "executor": executor.metadata(),
        "final3_sha256": fingerprint(),
        "manifest_sha256": final3_manifest()["manifest_sha256"],
        "leakage": leakage,
        "humanevalfix_separation": external,
        "rows": rows,
    }
    write_json(destination / "final3-manifest.json", final3_manifest())
    write_json(destination / "report.json", report)
    return report
