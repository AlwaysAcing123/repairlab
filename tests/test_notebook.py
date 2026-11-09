"""Check the distributable notebook without downloading or running a model."""

import ast
import json
import unittest
from pathlib import Path

from repairlab.tasks import pilot_tasks


class NotebookTests(unittest.TestCase):
    def test_source_extraction_preserves_real_newlines(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "RepairLab_Baseline.ipynb"
        notebook = json.loads(path.read_text())
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            for node in ast.parse(cell["source"]).body:
                if isinstance(node, ast.FunctionDef) and node.name == "extract_source":
                    namespace = {}
                    # Execute only the trusted formatter definition, never a generated patch.
                    exec(
                        compile(ast.Module(body=[node], type_ignores=[]), "formatter", "exec"),
                        namespace,
                    )
                    function = namespace["extract_source"]
                    source = "def solve(x):\n    return x\n"
                    self.assertEqual(function(source), source)
                    self.assertEqual(function("```python\n" + source + "```"), source)
                    return
        self.fail("Source formatter missing")

    def test_public_payload_and_cell_syntax(self):
        path = Path(__file__).resolve().parents[1] / "notebooks" / "RepairLab_Baseline.ipynb"
        notebook = json.loads(path.read_text())
        self.assertEqual(notebook["nbformat"], 4)
        tasks = None
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse(cell["source"])
            compile(tree, "notebook-cell", "exec")
            for statement in tree.body:
                if isinstance(statement, ast.Assign) and any(
                    isinstance(target, ast.Name) and target.id == "TASKS"
                    for target in statement.targets
                ):
                    tasks = json.loads(statement.value.args[0].value)
        self.assertEqual(tasks, [task.public() for task in pilot_tasks()])
        self.assertTrue(all(cell.get("outputs", []) == [] for cell in notebook["cells"]))


if __name__ == "__main__":
    unittest.main()
