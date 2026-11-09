"""External adapter contracts, without executing downloaded programs on the host."""

import ast
import json
import unittest
from pathlib import Path

from repairlab.humanevalfix import PROTOCOL, control_matches, fingerprint, load, public_tasks


class HumanEvalFixTests(unittest.TestCase):
    def test_broken_program_resource_failures_are_expected_control_outcomes(self):
        self.assertTrue(control_matches({"passed": False, "status": "resource_limit"}, False))
        self.assertFalse(control_matches({"passed": False, "status": "resource_limit"}, True))
        self.assertFalse(control_matches({"passed": False, "status": "invalid_output"}, False))

    def test_snapshot_and_public_boundary(self):
        rows, metadata = load()
        self.assertEqual(len(rows), 164)
        self.assertEqual(len(metadata["revision"]), 40)
        self.assertEqual(len(fingerprint()), 64)
        public = public_tasks()
        for row, original in zip(public, rows):
            self.assertEqual(
                set(row),
                {"task_id", "specification", "source", "public_tests_source", "entry_point"},
            )
            self.assertEqual(row["public_tests_source"], original["example_test"])
            self.assertEqual(row["source"], original["declaration"] + original["buggy_solution"])
            self.assertNotIn("canonical_solution", row)
            self.assertNotIn("test", row)

    def test_external_notebook_uses_correct_protocol_and_only_public_payload(self):
        root = Path(__file__).resolve().parents[1]
        data = json.loads((root / "notebooks" / "RepairLab_HumanEvalFix.ipynb").read_text())
        tasks = None
        protocol = None
        for cell in data["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse(cell["source"])
            compile(tree, "external-notebook-cell", "exec")
            for node in tree.body:
                if not isinstance(node, ast.Assign):
                    continue
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                if "TASKS" in names:
                    tasks = json.loads(node.value.args[0].value)
                if "metadata" in names:
                    protocol = next(
                        value.value
                        for key, value in zip(node.value.keys, node.value.values)
                        if key.value == "protocol"
                    )
        self.assertEqual(tasks, public_tasks())
        self.assertEqual(protocol, PROTOCOL)


if __name__ == "__main__":
    unittest.main()
