import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from test_rl import run_in_process

from repairlab.curriculum.final_test import final_tasks
from repairlab.curriculum.training_programs import training_programs
from repairlab.final2.admission import (
    REVIEWED_FINAL2_NEAR_PAIRS,
    final2_leakage_report,
    final2_manifest,
    reject_final2,
)
from repairlab.final2.programs import final2_tasks
from repairlab.rl.admission import REVIEWED_RL_NEAR_PAIRS
from repairlab.rl.programs import rl_programs
from repairlab.tasks import pilot_tasks
from repairlab.tool_warmup import programs as warmup_programs

ROOT = Path(__file__).resolve().parents[1]


def earlier_tasks():
    return pilot_tasks() + [p.task for p in training_programs()] + final_tasks() + rl_programs()


class Final2Tests(unittest.TestCase):
    def test_forty_eight_balanced_tasks_with_new_ids(self):
        tasks = final2_tasks()
        self.assertEqual(len(tasks), 48)
        self.assertEqual(set(Counter(t.family for t in tasks).values()), {16})
        self.assertEqual(set(Counter(t.bug_category for t in tasks).values()), {12})
        ids = [t.id for t in tasks]
        self.assertEqual(len(set(ids)), 48)
        earlier = {t.id for t in earlier_tasks()} | {p.id for p in warmup_programs()}
        self.assertFalse(set(ids) & earlier)

    def test_authored_controls_hold(self):
        for task in final2_tasks():
            cases = task.public_cases + task.private_cases
            self.assertFalse(run_in_process(task.buggy, cases), task.id)
            self.assertFalse(run_in_process(task.buggy, task.public_cases), task.id)
            self.assertTrue(run_in_process(task.reference, cases), task.id)
            self.assertTrue(run_in_process(task.alternative, cases), task.id)
            self.assertTrue(run_in_process(task.shortcut(), task.public_cases), task.id)
            self.assertFalse(run_in_process(task.shortcut(), cases), task.id)

    def test_leakage_has_no_failures_and_every_near_pair_was_reviewed(self):
        report = final2_leakage_report()
        self.assertEqual(report["hard_failures"], [])
        self.assertEqual(report["unreviewed_near_pairs"], [])
        self.assertEqual(report["earlier_programs_compared"], 120)
        self.assertEqual(report["cross_split_pairs_compared"], 120 * 48)
        found = {frozenset((n["left"], n["right"])) for n in report["near_pairs_for_review"]}
        self.assertEqual(found, set(REVIEWED_FINAL2_NEAR_PAIRS))
        self.assertFalse(REVIEWED_FINAL2_NEAR_PAIRS & REVIEWED_RL_NEAR_PAIRS)

    def test_reject_final2_by_id_and_by_content(self):
        sealed = final2_tasks()[0]
        with self.assertRaises(ValueError):
            reject_final2([sealed])
        with self.assertRaises(ValueError):
            reject_final2([replace(sealed, id="innocent-name")])
        reject_final2(earlier_tasks())

    def test_manifest_is_stable_and_records_no_evaluation(self):
        first = final2_manifest()
        self.assertEqual(first, final2_manifest())
        self.assertEqual(first["final_test_model_evaluations"], 0)
        self.assertEqual(len(first["tasks"]), 48)

    def test_no_model_facing_script_imports_the_second_test(self):
        # Admission, the one-time opening, and two audits that check separation. Every
        # other script, including every notebook builder and runtime, must not.
        allowed = {
            "admit_final2.py",
            "run_final_test_2.py",
            "verify_scaleup_comparison.py",
            "verify_scaleup_training.py",
        }
        for path in sorted((ROOT / "scripts").glob("*.py")):
            if path.name in allowed:
                continue
            self.assertNotIn("final2", path.read_text(), path.name)


if __name__ == "__main__":
    unittest.main()
