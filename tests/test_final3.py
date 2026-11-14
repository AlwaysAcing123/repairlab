import json
import signal
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from test_rl import run_in_process

from repairlab.curriculum.final_test import final_tasks
from repairlab.curriculum.splits import REVIEWED_NEAR_PAIRS
from repairlab.curriculum.training_programs import training_programs
from repairlab.execution import validate_source
from repairlab.final2.admission import REVIEWED_FINAL2_NEAR_PAIRS
from repairlab.final2.programs import final2_tasks
from repairlab.final3.admission import (
    REVIEWED_FINAL3_NEAR_PAIRS,
    final3_humanevalfix_report,
    final3_leakage_report,
    final3_manifest,
    reject_final3,
)
from repairlab.final3.programs import final3_tasks
from repairlab.rl.admission import REVIEWED_RL_NEAR_PAIRS
from repairlab.rl.programs import rl_programs
from repairlab.scaleup.humanevalfix_native import converted_tasks
from repairlab.tasks import canonical, pilot_tasks
from repairlab.tool_warmup import programs as warmup_programs

ROOT = Path(__file__).resolve().parents[1]
CATEGORIES = {
    "spec_misreading",
    "wrong_algorithm",
    "missing_edge_case",
    "boundary",
    "wrong_variable",
    "wrong_init",
    "statement_order",
    "missing_step",
    "wrong_scope",
    "wrong_operation",
    "aliasing",
    "inverted_logic",
    "wrong_return",
}


def earlier_tasks():
    return (
        pilot_tasks()
        + [p.task for p in training_programs()]
        + final_tasks()
        + rl_programs()
        + final2_tasks()
    )


class _Timeout(Exception):
    pass


def outcomes(source, cases, seconds=2.0):
    """Authored sources only: each case's result, or 'timeout' if it does not finish."""

    def expired(_signal, _frame):
        raise _Timeout

    namespace = {}
    exec(source, namespace)
    previous = signal.signal(signal.SIGALRM, expired)
    results = []
    try:
        for case in cases:
            args = json.loads(canonical(case.args))
            try:
                try:
                    signal.setitimer(signal.ITIMER_REAL, seconds)
                    results.append(canonical(namespace["solve"](*args)))
                finally:
                    signal.setitimer(signal.ITIMER_REAL, 0)
            except _Timeout:
                results.append("timeout")
            except Exception:
                results.append("error")
    finally:
        signal.signal(signal.SIGALRM, previous)
    return results


class Final3Tests(unittest.TestCase):
    def test_ninety_six_balanced_tasks_with_new_ids(self):
        tasks = final3_tasks()
        self.assertEqual(len(tasks), 96)
        self.assertEqual(
            Counter(t.family for t in tasks),
            {"validation": 32, "collections": 32, "control_flow": 32},
        )
        ids = [t.id for t in tasks]
        self.assertEqual(len(set(ids)), 96)
        earlier = {t.id for t in earlier_tasks()} | {p.id for p in warmup_programs()}
        earlier |= {t.id for t in converted_tasks()}
        self.assertFalse(set(ids) & earlier)

    def test_bug_categories_are_varied_and_none_dominates(self):
        counts = Counter(t.bug_category for t in final3_tasks())
        self.assertEqual(set(counts), CATEGORIES)
        self.assertLessEqual(max(counts.values()), 96 // 5)
        handwritten = counts["spec_misreading"] + counts["wrong_algorithm"]
        self.assertGreaterEqual(handwritten + counts["missing_edge_case"], 30)

    def test_examples_and_domains_are_small_and_finite(self):
        for task in final3_tasks():
            self.assertTrue(3 <= len(task.public_cases) <= 5, task.id)
            self.assertTrue(20 <= len(task.private_cases) <= 1500, task.id)
            self.assertTrue(task.domain.strip(), task.id)
            self.assertTrue(task.specification.strip(), task.id)

    def test_every_source_keeps_the_signature(self):
        for task in final3_tasks():
            for source in (task.reference, task.alternative, task.buggy, task.shortcut()):
                validate_source(source, task.signature)
            self.assertNotEqual(task.buggy, task.reference, task.id)
            self.assertNotEqual(task.alternative, task.reference, task.id)

    def test_authored_controls_hold(self):
        for task in final3_tasks():
            cases = task.public_cases + task.private_cases
            self.assertFalse(run_in_process(task.buggy, cases), task.id)
            self.assertFalse(run_in_process(task.buggy, task.public_cases), task.id)
            self.assertTrue(run_in_process(task.reference, cases), task.id)
            self.assertTrue(run_in_process(task.alternative, cases), task.id)
            self.assertTrue(run_in_process(task.shortcut(), task.public_cases), task.id)
            self.assertFalse(run_in_process(task.shortcut(), cases), task.id)

    def test_buggy_versions_terminate_on_every_input(self):
        for task in final3_tasks():
            results = outcomes(task.buggy, task.public_cases + task.private_cases)
            self.assertNotIn("timeout", results, task.id)

    def test_leakage_has_no_failures_and_every_near_pair_was_reviewed(self):
        report = final3_leakage_report()
        self.assertEqual(report["hard_failures"], [])
        self.assertEqual(report["unreviewed_near_pairs"], [])
        self.assertEqual(report["earlier_programs_compared"], 168)
        self.assertEqual(report["cross_split_pairs_compared"], 168 * 96)
        external = final3_humanevalfix_report()
        self.assertEqual(external["hard_failures"], [])
        self.assertEqual(external["unreviewed_near_pairs"], [])
        self.assertEqual(external["humanevalfix_tasks_compared"], len(converted_tasks()))
        found = {
            frozenset((n["left"], n["right"]))
            for n in report["near_pairs_for_review"] + external["near_pairs_for_review"]
        }
        self.assertEqual(found, set(REVIEWED_FINAL3_NEAR_PAIRS))

    def test_reviewed_pairs_each_name_one_third_test_task(self):
        ids = {t.id for t in final3_tasks()}
        for pair in REVIEWED_FINAL3_NEAR_PAIRS:
            self.assertEqual(len(pair & ids), 1, sorted(pair))
        earlier = REVIEWED_NEAR_PAIRS | REVIEWED_RL_NEAR_PAIRS | REVIEWED_FINAL2_NEAR_PAIRS
        self.assertFalse(REVIEWED_FINAL3_NEAR_PAIRS & earlier)

    def test_reject_final3_by_id_and_by_content(self):
        for sealed in final3_tasks()[::12]:
            with self.assertRaises(ValueError):
                reject_final3([sealed])
            with self.assertRaises(ValueError):
                reject_final3([replace(sealed, id="innocent-name")])
            with self.assertRaises(ValueError):
                reject_final3([replace(sealed, id="other", specification="Fix it.")])
        reject_final3(earlier_tasks())
        reject_final3(converted_tasks())

    def test_manifest_is_stable_and_records_no_evaluation(self):
        first = final3_manifest()
        self.assertEqual(first, final3_manifest())
        self.assertEqual(first["final_test_model_evaluations"], 0)
        self.assertEqual(first["earlier_programs_compared"], 168)
        self.assertEqual(len(first["tasks"]), 96)

    def test_no_script_or_notebook_carries_the_third_test(self):
        # Scripts may name the package, to leave it out of a source snapshot or to call
        # reject_final3, but none may load its tasks. Admission runs through
        # repairlab.final3.admission.admit; a later opening or audit script that needs the
        # tasks must be added here deliberately.
        # The one-time opening of this test (plan Addendum 9) builds its tasks on the laptop.
        allowed = {"run_final_test_3.py"}
        loaders = ("final3_tasks", "final3.programs", "final3_entries", "final3_manifest")
        for path in sorted((ROOT / "scripts").glob("*.py")):
            if path.name in allowed:
                continue
            text = path.read_text()
            for loader in loaders:
                self.assertNotIn(loader, text, path.name)
        sealed = [t.specification for t in final3_tasks()] + [t.buggy for t in final3_tasks()]
        for path in sorted((ROOT / "notebooks").rglob("*")):
            if path.is_file():
                text = path.read_text(errors="ignore")
                self.assertFalse(any(s in text for s in sealed), path.name)

    def test_package_is_listed_for_installation(self):
        self.assertIn('"repairlab.final3"', (ROOT / "pyproject.toml").read_text())


if __name__ == "__main__":
    unittest.main()
