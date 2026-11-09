"""Native tool-schema validation and public-only diagnostic packaging."""

import ast
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from repairlab.native_tools import assistant_call, parsed_action, probes, roundtrip_examples, tools


class NativeToolsTests(unittest.TestCase):
    def test_all_tools_and_complete_source_arguments_round_trip_locally(self):
        self.assertEqual(
            {t["function"]["name"] for t in tools()}, {"read", "test", "edit", "submit"}
        )
        actions = [parsed_action(example) for example in roundtrip_examples()]
        self.assertEqual([a["tool"] for a in actions], ["read", "test", "edit", "submit"])
        self.assertIn("\n    return", actions[2]["source"])

    def test_invalid_native_responses_are_not_repaired_or_silently_selected(self):
        invalid = [
            None,
            {"content": '{"tool":"read"}'},
            {"tool_calls": []},
            {"tool_calls": assistant_call("read", {})["tool_calls"] * 2},
            assistant_call("shell", {}),
            assistant_call("test", {"private": True}),
            assistant_call("read", "{}"),
            assistant_call("edit", {"source": "print('not a function')"}),
        ]
        for value in invalid:
            with self.assertRaises(ValueError):
                parsed_action(value)

    def test_twelve_synthetic_probes_and_native_tool_result_roles(self):
        fixtures = probes()
        self.assertEqual(len(fixtures), 12)
        self.assertEqual(len({p["probe_id"] for p in fixtures}), 12)
        self.assertEqual(sum(any(m["role"] == "tool" for m in p["messages"]) for p in fixtures), 4)
        for probe in fixtures:
            self.assertEqual(probe["messages"][0]["role"], "system")
            self.assertEqual(probe["messages"][-1]["role"], "user")
            self.assertIn(probe["expected_tool"], {"read", "test", "edit", "submit"})

    def test_notebook_syntax_and_no_generated_code_execution(self):
        root = Path(__file__).resolve().parents[1]
        spec = importlib.util.spec_from_file_location(
            "native_builder", root / "scripts" / "build_native_diagnostic.py"
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as folder:
            notebook = module.build(Path(folder) / "diagnostic.ipynb")
        text = json.dumps(notebook)
        self.assertNotIn("KGAT_", text)
        self.assertNotIn("COORDINATOR_TOKEN", text)
        self.assertNotIn("private_cases", text)
        self.assertIn("skip_special_tokens=False", text)
        self.assertIn("qwen3_schema", text)
        for cell in notebook["cells"]:
            if cell["cell_type"] != "code":
                continue
            tree = ast.parse(cell["source"])
            compile(tree, "native-diagnostic-cell", "exec")
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertNotIn(node.func.id, {"exec", "eval", "compile"})


if __name__ == "__main__":
    unittest.main()
