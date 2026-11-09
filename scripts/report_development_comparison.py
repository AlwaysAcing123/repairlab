"""Report the six-task pilot and 18-task expansion separately, then descriptively together.

Reads saved evidence only. Never writes into either run directory.
"""

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repairlab.experiments import write_json  # noqa: E402
from repairlab.tasks import pilot_tasks  # noqa: E402

ARMS = ("original", "supervised")


def exact_sign_test(wins, losses):
    """Two-sided exact binomial test on discordant pairs."""
    total = wins + losses
    if total == 0:
        return 1.0
    tail = sum(math.comb(total, k) for k in range(min(wins, losses) + 1)) / 2**total
    return min(1.0, 2 * tail)


def summarize(rows):
    pairs = {}
    for row in rows:
        pairs.setdefault(row["task_id"], {})[row["condition"]] = row["success"]
    wins = sum(p["supervised"] and not p["original"] for p in pairs.values())
    losses = sum(p["original"] and not p["supervised"] for p in pairs.values())
    return {
        "tasks": len(pairs),
        "original_successes": sum(p["original"] for p in pairs.values()),
        "supervised_successes": sum(p["supervised"] for p in pairs.values()),
        "adapter_only": wins,
        "original_only": losses,
        "both_pass": sum(all(p.values()) for p in pairs.values()),
        "both_fail": sum(not any(p.values()) for p in pairs.values()),
        "exact_sign_test_two_sided": exact_sign_test(wins, losses),
    }


def by_field(rows, field):
    groups = {}
    for row in rows:
        groups.setdefault(row[field], []).append(row)
    return {name: summarize(selected) for name, selected in sorted(groups.items())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", type=Path, required=True)
    parser.add_argument("--expansion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pilot = json.loads((args.pilot / "report.json").read_text())
    expansion = json.loads((args.expansion / "report.json").read_text())
    categories = {t.id: t.bug_category for t in pilot_tasks()}
    rows = [
        {**r, "bug_category": categories[r["task_id"]]} for r in pilot["rows"] + expansion["rows"]
    ]
    assert len({r["task_id"] for r in rows}) == 24 and len(rows) == 48
    report = {
        "split": "development",
        "note": "The pilot and expansion are reported separately. The combined row is descriptive; these tasks are not a held-out test.",
        "pilot": summarize(pilot["rows"]),
        "expansion": summarize(expansion["rows"]),
        "combined_descriptive": summarize(rows),
        "combined_by_family": by_field(rows, "family"),
        "combined_by_bug_category": by_field(rows, "bug_category"),
        "expansion_terminations": {
            arm: expansion["arms"][arm]["termination_counts"] for arm in ARMS
        },
        "expansion_invalid_actions": {
            arm: expansion["arms"][arm]["invalid_actions"] for arm in ARMS
        },
        "expansion_failures": [
            {k: r[k] for k in ("task_id", "condition", "termination", "edits", "test_calls")}
            for r in expansion["rows"]
            if not r["success"]
        ],
    }
    write_json(args.output, report)
    print(
        json.dumps({k: report[k] for k in ("pilot", "expansion", "combined_descriptive")}, indent=2)
    )


if __name__ == "__main__":
    main()
