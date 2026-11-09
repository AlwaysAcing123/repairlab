"""Host-side invariants; no model-generated code executes in these tests."""

import json
import tempfile
import unittest
from pathlib import Path

from repairlab.episode import Episode, parse_action
from repairlab.execution import validate_source
from repairlab.experiments import score_patches
from repairlab.tasks import canonical, manifest, pilot_tasks


class StubExecutor:
    def __init__(self):
        self.calls = []

    def run(self, source, cases, *, reveal=False):
        self.calls.append((source, cases, reveal))
        return {
            "passed": True,
            "passed_cases": len(cases),
            "total_cases": len(cases),
            "status": "graded",
            **({"cases": [c.record() for c in cases]} if reveal else {}),
        }

    def metadata(self):
        return {"backend": "stub-unit-test-only"}


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tasks = pilot_tasks()

    def setUp(self):
        self.executor = StubExecutor()
        self.episode = Episode(self.tasks[0], self.executor)

    def test_manifest_does_not_invent_train_or_final_splits(self):
        data = manifest(self.tasks)
        self.assertEqual(len(data["tasks"]), 24)
        self.assertEqual(data["training_tasks"], 0)
        self.assertEqual(data["final_test_tasks"], 0)
        self.assertEqual({r["split"] for r in data["tasks"]}, {"development"})
        self.assertEqual(len({r["group_id"] for r in data["tasks"]}), 24)
        self.assertEqual(data, manifest(pilot_tasks()))

    def test_public_export_excludes_protected_material(self):
        for task in self.tasks:
            public = task.public()
            self.assertEqual(set(public), {"task_id", "source", "specification", "public_tests"})
            self.assertNotIn(task.reference, canonical(public))
            self.assertTrue(task.private_cases)
            self.assertFalse(
                {canonical(c.args) for c in task.public_cases}
                & {canonical(c.args) for c in task.private_cases}
            )

    def test_invalid_actions_consume_budget_and_do_not_change_source(self):
        episode = Episode(self.tasks[0], self.executor, max_actions=2)
        original = episode.source
        episode.step({"tool": "edit", "source": "not Python"})
        episode.step({"tool": "shell", "command": "anything"})
        self.assertTrue(episode.done)
        self.assertEqual(episode.source, original)
        self.assertEqual(episode.result["termination"], "action_budget")
        self.assertEqual(episode.result["reward"], 0)
        self.assertEqual(episode.record()["invalid_actions"], 2)

    def test_submit_on_last_action_is_allowed_and_final(self):
        episode = Episode(self.tasks[0], self.executor, max_actions=1)
        observation = episode.step({"tool": "submit"})
        self.assertEqual(observation, {"submitted": True})
        self.assertTrue(episode.result["success"])
        self.assertEqual(episode.result["termination"], "submitted")
        with self.assertRaises(ValueError):
            episode.step({"tool": "edit", "source": self.tasks[0].reference})

    def test_private_results_never_enter_model_observations(self):
        public = self.episode.step({"tool": "test"})
        self.assertEqual(public["total_cases"], len(self.tasks[0].public_cases))
        observation = self.episode.step({"tool": "submit"})
        self.assertEqual(observation, {"submitted": True})
        self.assertEqual(self.executor.calls[-1][1], self.tasks[0].private_cases)
        self.assertFalse(self.executor.calls[-1][2])

    def test_extra_fields_cannot_select_hidden_tests(self):
        observation = self.episode.step({"tool": "test", "private": True})
        self.assertIn("error", observation)
        self.assertEqual(self.executor.calls, [])

    def test_edit_preserves_function_signature(self):
        original = self.episode.source
        response = self.episode.step({"tool": "edit", "source": "def solve(x):\n    return x\n"})
        self.assertIn("error", response)
        self.assertEqual(self.episode.source, original)

    def test_json_parser_rejects_duplicate_keys_and_nonfinite_numbers(self):
        for raw in ('{"tool":"read","tool":"submit"}', '{"tool": NaN}', "```json\n{}\n```"):
            self.assertIn("invalid_model_output", parse_action(raw))

    def test_source_cannot_add_top_level_executable_statements(self):
        for raw in (
            "import os\ndef solve(x):\n    return x\n",
            "print('hi')",
            "@decorator\ndef solve(x):\n    return x\n",
            "def solve(x=print('hi')):\n    return x\n",
        ):
            with self.assertRaises(ValueError):
                validate_source(raw)

    def test_scoring_keeps_missing_and_invalid_outputs_in_denominator(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "predictions.jsonl").write_text(
                json.dumps({"task_id": self.tasks[0].id, "source": "broken text"}) + "\n"
            )
            (root / "metadata.json").write_text(
                json.dumps(
                    {
                        "model": "unit-test-stub",
                        "model_revision": "stub",
                        "seed": 7,
                        "decoding": {},
                        "protocol": "single-response-v1",
                        "manifest_sha256": manifest()["manifest_sha256"],
                    }
                )
            )
            report = score_patches(
                self.executor, root / "predictions.jsonl", root / "metadata.json", root / "run"
            )
            self.assertEqual(report["tasks"], 24)
            self.assertEqual(report["missing_predictions"], 23)
            self.assertEqual(report["termination_counts"]["invalid_patch"], 1)
            self.assertEqual(report["successes"], 0)
            self.assertEqual(self.executor.calls, [])


if __name__ == "__main__":
    unittest.main()
