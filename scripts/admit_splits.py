"""Run authored controls and leakage checks for the new splits. No model is involved."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repairlab.curriculum.splits import admit  # noqa: E402
from repairlab.execution import DockerExecutor  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    report = admit(DockerExecutor(), parser.parse_args().output)
    print(
        json.dumps(
            {
                **{k: report[k] for k in ("tasks", "matched_controls", "controls", "all_matched")},
                "hard_leakage_failures": report["leakage"]["hard_failures"],
                "unreviewed_near_pairs": report["leakage"]["unreviewed_near_pairs"],
            },
            indent=2,
        )
    )
    if (
        not report["all_matched"]
        or report["leakage"]["hard_failures"]
        or report["leakage"]["unreviewed_near_pairs"]
    ):
        raise SystemExit("Split admission failed")


if __name__ == "__main__":
    main()
