"""The supervised arm's launcher helpers (plan, Addendum 8). No GPU or model needed."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from run_rlfirst_sft import choose, combined_runtime  # noqa: E402

from repairlab.rlfirst.episode import parse  # noqa: E402

POOL = ROOT / "artifacts" / "rlfirst-pool-v3" / "pool.json"
SWEEP = ROOT / "artifacts" / "rlfirst-sweep-v1" / "sweep-verification.json"


def rl_rows():
    retired = set(json.loads(SWEEP.read_text())["retired_before_rl"])
    return [r for r in json.loads(POOL.read_text())["tasks"] if r["task_id"] not in retired]


@unittest.skipUnless(POOL.exists() and SWEEP.exists(), "pool and sweep artifacts needed")
class SupervisedLauncher(unittest.TestCase):
    def test_choice_is_seeded_and_covers_the_list(self):
        rows = rl_rows()
        self.assertEqual(choose(rows, 50, 7), choose(rows, 50, 7))
        self.assertEqual(len(choose(rows, 50, 7)), 50)
        self.assertNotEqual(
            {r["task_id"] for r in choose(rows, 50, 7)}, {r["task_id"] for r in choose(rows, 50, 8)}
        )
        self.assertEqual(len(choose(rows, len(rows), 7)), len(rows))
        self.assertEqual(len(choose(rows, len(rows) + 5, 7)), len(rows))

    def test_every_target_parses_back_to_its_reference(self):
        for row in rl_rows():
            target = "```python\n" + row["reference"] + "```"
            self.assertEqual(parse(target, row["signature"])["source"], row["reference"])

    def test_combined_runtime_has_both_parts(self):
        with tempfile.TemporaryDirectory() as folder:
            relative = combined_runtime(Path(folder) / "runtime.py")
            text = (ROOT / "scripts" / relative).read_text()
        rl = (ROOT / "scripts" / "rlfirst_runtime.py").read_text()
        self.assertTrue(text.startswith(rl))
        self.assertIn("def run_sft(payload)", text)
        self.assertIn("def run_rl(payload)", text)


if __name__ == "__main__":
    unittest.main()
