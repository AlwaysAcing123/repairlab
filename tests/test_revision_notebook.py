"""Validate the trusted inference client without loading a model or credentials."""

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


class RevisionNotebookTests(unittest.TestCase):
    def test_cell_syntax_and_no_task_or_grader_bundle(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "revision_builder", root / "scripts" / "build_revision_notebook.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            data = module.build(
                Path(folder) / "client.ipynb", "https://example.invalid", "dummy" * 12
            )
        for cell in data["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse(cell["source"])
            compile(tree, "revision-client", "exec")
            # Model text is transported to local execution, never exec/eval'd here.
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, {"exec", "eval", "compile"})
        text = json.dumps(data)
        self.assertNotIn("private_cases", text)
        self.assertNotIn("canonical_solution", text)
        self.assertNotIn("KGAT_", text)
        self.assertIn("request_id", text)


if __name__ == "__main__":
    unittest.main()
