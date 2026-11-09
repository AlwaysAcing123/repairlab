"""Matched native repair histories, checkpoint guards, and budget enforcement."""

import json
import sys
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from repairlab.native_repair import BUDGET, PROTOCOL, REVISION, NativeRepairPilot, selected_tasks
from repairlab.native_tools import assistant_call

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from build_native_repair_notebook import build  # noqa: E402
from test_revision import Executor  # noqa: E402

ADAPTER = {"file_sha256": "a" * 64, "tensor_sha256": "b" * 64, "base_sha256": "c" * 64}


class NativeRepairTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.executor = Executor()
        self.pilot = NativeRepairPilot(
            self.executor, Path(self.temp.name) / "run", ADAPTER, selected_tasks()[:1]
        )

    def responses(self, tool="read", arguments=None):
        batch = self.pilot.export()
        metadata = {
            "protocol": PROTOCOL,
            "model_revision": REVISION,
            "budget": BUDGET,
            "batch_sha256": batch["batch_sha256"],
            "adapter_file_sha256": ADAPTER["file_sha256"],
            "adapter_initial_sha256": ADAPTER["tensor_sha256"],
            "adapter_final_sha256": ADAPTER["tensor_sha256"],
        }
        for name in ("original", "adapted_base"):
            for stage in ("initial", "final"):
                metadata[name + "_" + stage + "_sha256"] = ADAPTER["base_sha256"]
        rows = [
            {
                "episode_id": r["episode_id"],
                "condition": r["condition"],
                "decision": r["decision"],
                "raw": "unit-test-only",
                "parsed": assistant_call(tool, arguments or {}),
                "input_tokens": 100,
                "output_tokens": 20,
            }
            for r in batch["requests"]
        ]
        return batch, metadata, rows

    def apply(self, tool="read", arguments=None):
        batch, metadata, rows = self.responses(tool, arguments)
        self.pilot.apply(batch["batch_sha256"], metadata, rows)
        return batch, metadata, rows

    def test_selection_balanced_and_initial_prompts_identical(self):
        self.assertEqual(
            Counter(t.family for t in selected_tasks()),
            {"validation": 2, "collections": 2, "control_flow": 2},
        )
        batch = self.pilot.export()
        self.assertEqual(batch["requests"][0]["messages"], batch["requests"][1]["messages"])
        self.assertEqual(batch, self.pilot.export())

    def test_real_public_feedback_and_native_history(self):
        self.apply("test")
        self.assertEqual(len(self.executor.calls), 2)
        self.assertTrue(all(c[2] for c in self.executor.calls))
        messages = self.pilot.export()["requests"][0]["messages"]
        self.assertEqual(messages[-2], assistant_call("test", {}))
        self.assertEqual(messages[-1]["role"], "tool")
        self.assertNotIn("elapsed_seconds", messages[-1]["content"])
        self.assertNotIn("private", messages[-1]["content"])

    def test_edit_changes_candidate_and_submit_does_not_reveal_grade(self):
        task = selected_tasks()[0]
        self.apply("edit", {"source": task.reference})
        self.apply("submit")
        for state in self.pilot.states.values():
            self.assertEqual(state["episode"].source, task.reference)
            self.assertEqual(json.loads(state["messages"][-1]["content"]), {"submitted": True})
        self.assertEqual(self.executor.calls[-1][1], task.private_cases)
        self.assertEqual(self.pilot.finish()["paired"]["both_pass"], 1)

    def test_bad_calls_consume_actions_and_never_autosubmit(self):
        for _ in range(8):
            self.apply("test", {"public_tests": []})
        report = self.pilot.finish()
        self.assertEqual(self.executor.calls, [])
        for arm in report["arms"].values():
            self.assertEqual(arm["invalid_actions"], 8)
            self.assertEqual(arm["termination_counts"], {"action_budget": 1})
            self.assertEqual(arm["successes"], 0)

    def test_replayed_batch_cannot_repeat_execution(self):
        batch, metadata, rows = self.apply("test")
        with self.assertRaises(ValueError):
            self.pilot.apply(batch["batch_sha256"], metadata, rows)
        self.assertEqual(len(self.executor.calls), 2)

    def test_whole_batch_checked_before_mutations(self):
        batch, metadata, rows = self.responses("test")
        rows[-1]["output_tokens"] = 513
        with self.assertRaises(ValueError):
            self.pilot.apply(batch["batch_sha256"], metadata, rows)
        self.assertEqual(self.executor.calls, [])
        self.assertTrue(all(not s["episode"].events for s in self.pilot.states.values()))

    def test_changed_adapter_or_base_rejected(self):
        batch, metadata, rows = self.responses()
        for key in (
            "adapter_file_sha256",
            "adapter_initial_sha256",
            "adapter_final_sha256",
            "original_initial_sha256",
            "original_final_sha256",
            "adapted_base_initial_sha256",
            "adapted_base_final_sha256",
        ):
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.pilot.apply(batch["batch_sha256"], {**metadata, key: "wrong"}, rows)

    def test_notebook_only_contains_public_conversations_and_inference(self):
        notebook = build(Path(self.temp.name) / "round.ipynb", self.pilot.export())
        source = "\n".join(c["source"] for c in notebook["cells"])
        for forbidden in (
            "KAGGLE_API_TOKEN",
            "private_cases",
            "task.reference",
            ".backward(",
            "optimizer.step(",
        ):
            self.assertNotIn(forbidden, source)
        for cell in notebook["cells"]:
            if cell["cell_type"] == "code":
                compile(cell["source"], "notebook-cell", "exec")
        self.assertIn("PeftModel.from_pretrained", source)
        self.assertIn('actual_file_hash == expected["file_sha256"]', source)


if __name__ == "__main__":
    unittest.main()
