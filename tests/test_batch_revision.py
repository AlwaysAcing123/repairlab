"""Staged experiment invariants without GPU inference or generated-code execution."""

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from test_revision import Executor

from repairlab.batch_revision import BatchRevision, parse_batch_action
from repairlab.revision import BUDGET, MODEL, PROTOCOL, REVISION
from repairlab.tasks import pilot_tasks


def metadata(batch):
    return {
        "model": MODEL,
        "model_revision": REVISION,
        "protocol": PROTOCOL,
        "budget": BUDGET,
        "initial_weight_sha256": "a" * 64,
        "final_weight_sha256": "a" * 64,
        "batch_sha256": batch["batch_sha256"],
        "round": batch["round"],
        "gpu": "test",
        "dtype": "test",
        "packages": {},
        "seed": 7,
        "decoding": {"do_sample": False},
    }


def responses(batch, action):
    return [
        {
            "episode_id": r["episode_id"],
            "decision": r["decision"],
            "input_tokens": 100,
            "output_tokens": 20,
            "raw": action,
        }
        for r in batch["requests"]
    ]


class BatchRevisionTests(unittest.TestCase):
    def test_fence_compatibility_changes_only_one_enclosing_wrapper(self):
        action = {"tool": "edit", "source": "def solve(x):\n    return x\n"}
        raw = json.dumps(action)
        fenced = "```json\n" + raw + "\n```"
        self.assertEqual(parse_batch_action(fenced, True), action)
        self.assertEqual(parse_batch_action(raw, True), action)
        self.assertIn("invalid_model_output", parse_batch_action(fenced, False))
        for malformed in (
            "Here it is:\n" + fenced,
            fenced + "\nextra",
            "```json\n{}\n{}\n```",
            '```json\n{"tool":"read","tool":"test"}\n```',
        ):
            self.assertIn("invalid_model_output", parse_batch_action(malformed, True))

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.executor = Executor()
        self.run = BatchRevision(self.executor, Path(self.temp.name) / "run", pilot_tasks()[:1])

    def apply(self, action):
        batch = self.run.export()
        self.run.apply(batch["batch_sha256"], metadata(batch), responses(batch, action))
        return batch

    def test_public_export_is_identical_before_condition_feedback(self):
        batch = self.run.export()
        left, right = batch["requests"]
        self.assertEqual(left["messages"], right["messages"])
        self.assertEqual(left["max_new_tokens"], right["max_new_tokens"])
        self.assertNotIn("private_cases", json.dumps(batch))
        self.assertEqual(self.run.export(), batch)

    def test_only_feedback_condition_executes_public_tests(self):
        self.apply('{"tool":"test"}')
        self.assertEqual(len(self.executor.calls), 1)
        next_batch = self.run.export()
        by_id = {r["episode_id"]: r for r in next_batch["requests"]}
        self.assertIn(
            "unavailable", by_id["quantity-range--without_feedback"]["messages"][-1]["content"]
        )
        self.assertIn(
            "passed_cases", by_id["quantity-range--with_feedback"]["messages"][-1]["content"]
        )

    def test_batch_cannot_be_applied_twice(self):
        batch = self.apply('{"tool":"submit"}')
        before = len(self.executor.calls)
        with self.assertRaises(ValueError):
            self.run.apply(
                batch["batch_sha256"], metadata(batch), responses(batch, '{"tool":"submit"}')
            )
        self.assertEqual(len(self.executor.calls), before)
        self.assertIsNone(self.run.export())
        self.assertTrue(self.run.finish()["weights_unchanged"])

    def test_missing_or_invalid_response_does_not_partially_advance_batch(self):
        batch = self.run.export()
        rows = responses(batch, '{"tool":"submit"}')
        rows[1]["output_tokens"] = 513
        with self.assertRaises(ValueError):
            self.run.apply(batch["batch_sha256"], metadata(batch), rows)
        self.assertEqual(self.executor.calls, [])
        self.assertTrue(all(not s["episode"].done for s in self.run.states.values()))

    def test_eight_invalid_rounds_have_no_automatic_submission(self):
        for _ in range(8):
            self.apply("not JSON")
        report = self.run.finish()
        self.assertEqual(report["paired"]["both_fail"], 1)
        self.assertEqual(report["arms"]["with_feedback"]["invalid_actions"], 8)
        self.assertEqual(self.executor.calls, [])

    def test_model_parameter_changes_are_rejected(self):
        batch = self.run.export()
        config = {**metadata(batch), "final_weight_sha256": "b" * 64}
        with self.assertRaises(ValueError):
            self.run.apply(batch["batch_sha256"], config, responses(batch, '{"tool":"submit"}'))
        self.assertEqual(self.executor.calls, [])

    def test_notebook_contains_no_credentials_callbacks_or_code_execution(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "batch_builder", root / "scripts" / "build_batch_revision_notebook.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            notebook = module.build(Path(folder) / "round.ipynb", self.run.export())
        text = json.dumps(notebook)
        self.assertNotIn("KGAT_", text)
        self.assertNotIn("COORDINATOR_TOKEN", text)
        self.assertNotIn("ENDPOINT", text)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse(cell["source"])
            compile(tree, "batch-client", "exec")
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, {"eval", "exec", "compile"})


if __name__ == "__main__":
    unittest.main()
