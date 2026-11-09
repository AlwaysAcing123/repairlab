"""Warm-up data boundaries and standalone notebook validity, without GPU training."""

import ast
import importlib.util
import json
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from repairlab.native_tools import parsed_action, probes
from repairlab.tasks import canonical, pilot_tasks
from repairlab.tool_warmup import CONFIG, bundle, demonstration_data, development_probes, programs


class WarmupTests(unittest.TestCase):
    def test_program_groups_and_evaluation_prompts_do_not_cross_training_boundary(self):
        training = demonstration_data()
        development = development_probes()
        self.assertEqual(len(training), 64)
        self.assertEqual(len(development), 16)
        self.assertEqual(len({r["program_id"] for r in training}), 8)
        self.assertEqual(len({r["program_id"] for r in development}), 4)
        self.assertTrue(
            {r["group_id"] for r in training}.isdisjoint({r["group_id"] for r in development})
        )
        self.assertTrue(
            {r["program_id"] for r in training}.isdisjoint({t.id for t in pilot_tasks()})
        )
        self.assertTrue(
            {canonical(r["prompt"]) for r in training}.isdisjoint(
                {canonical(r["messages"]) for r in development + probes()}
            )
        )
        self.assertTrue(all(r["split"] == "train" for r in training))
        self.assertEqual(bundle(), bundle())

    def test_teacher_targets_are_valid_single_native_actions(self):
        counts = Counter()
        for sample in demonstration_data():
            self.assertEqual(len(sample["completion"]), 1)
            action = parsed_action(sample["completion"][0])
            counts[action["tool"]] += 1
            self.assertEqual(sample["prompt"][0]["role"], "system")
            if action["tool"] == "submit" and "trajectory" in sample["sample_id"]:
                history = sample["prompt"]
                self.assertEqual(history[-1]["role"], "tool")
                self.assertTrue(json.loads(history[-1]["content"])["passed"])
        self.assertEqual(dict(counts), {"read": 16, "test": 24, "edit": 8, "submit": 16})

    def test_training_programs_are_new_and_all_have_golden_examples(self):
        fixtures = programs()
        self.assertEqual(len(fixtures), 12)
        old = {t.reference for t in pilot_tasks()}
        for task in fixtures:
            self.assertTrue(task.cases)
            self.assertEqual(len(task.public_cases), 2)
            self.assertNotEqual(task.reference, task.buggy)
            self.assertNotIn(task.reference, old)
            self.assertEqual(ast.parse(task.reference).body[0].name, "solve")

    def test_notebook_contains_finite_training_and_no_execution_of_model_text(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "warmup_builder", root / "scripts" / "build_native_warmup.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            notebook = module.build(Path(folder) / "warmup.ipynb")
        text = json.dumps(notebook)
        self.assertNotIn("KGAT_", text)
        self.assertNotIn("COORDINATOR_TOKEN", text)
        self.assertEqual(CONFIG["optimizer_steps"], 60)
        self.assertIn("labels = [-100]", text)
        self.assertIn("initial_adapter.safetensors", text)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse(cell["source"])
            compile(tree, "warmup-cell", "exec")
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, {"exec", "eval", "compile"})


if __name__ == "__main__":
    unittest.main()
