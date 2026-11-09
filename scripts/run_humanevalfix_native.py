"""Run HumanEvalFix native exactly once: `0.6b-rl`, `1.7b-base`, and the final 1.7B arm.

Reported only; no checkpoint is chosen on it. Opens like the second final test: the
frozen record is required and the registry entry is written before any upload.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_final_test_2 import open_once, parser_for  # noqa: E402

from repairlab.scaleup.humanevalfix_native import SPLIT, native_tasks  # noqa: E402

REGISTRY = ROOT / "artifacts" / "humanevalfix-native-registry.json"


def main():
    args = parser_for(__doc__, "--open-humanevalfix-native").parse_args()
    frozen = json.loads(args.frozen.read_text())
    open_once(
        args,
        args.registry or REGISTRY,
        frozen,
        ["0.6b-rl", "1.7b-base", frozen["final_1.7b"]],
        native_tasks(),
        SPLIT,
        "alwaysacing/repairlab-humanevalfix-native",
        "RepairLab HumanEvalFix Native",
        "Autonomous repair score on converted HumanEvalFix tasks; reported only.",
    )


if __name__ == "__main__":
    main()
