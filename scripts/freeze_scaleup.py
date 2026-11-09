"""Freeze every checkpoint and choice before the second final test opens.

Recomputes rule 1 (supervised variant) and rule 2 (final 1.7B checkpoint) from the
saved, audited development comparisons, checks that every adapter's run was audited,
and writes one record that refuses to be overwritten.
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from run_scaleup_comparison import arm_specs  # noqa: E402

from repairlab.experiments import source_fingerprint  # noqa: E402
from repairlab.scaleup import fingerprint  # noqa: E402
from repairlab.scaleup.freeze import chosen_variant, final_checkpoint  # noqa: E402

FROZEN = ROOT / "artifacts" / "scaleup-frozen.json"
FINAL_ARMS = ("0.6b-rl", "1.7b-base", "1.7b-plain", "1.7b-notes", "1.7b-rl")


def audited(path, name):
    return json.loads((Path(path) / name).read_text())["verified"] is True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plain-run", type=Path, required=True)
    parser.add_argument("--notes-run", type=Path, required=True)
    parser.add_argument("--choice-run", type=Path, required=True)
    parser.add_argument("--rl-run", type=Path, required=True)
    parser.add_argument("--rl-choice-run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=FROZEN)
    parser.add_argument("--allow-partial", action="store_true", help="scripted dry runs only")
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit("Checkpoints are already frozen: " + str(args.output))
    for run, name in (
        (args.plain_run, "independent-verification.json"),
        (args.notes_run, "independent-verification.json"),
        (args.rl_run, "rl-verification.json"),
    ):
        if not audited(run, name):
            raise SystemExit("Run was not audited: " + str(run))
    variant, variant_counts = chosen_variant(args.choice_run, args.allow_partial)
    rl_plan = json.loads((args.rl_run / "experiment-plan.json").read_text())
    if rl_plan["start_variant"] != variant:
        raise SystemExit("RL did not start from the variant rule 1 chose")
    final, final_counts = final_checkpoint(args.rl_choice_run, variant, args.allow_partial)
    arms = arm_specs(FINAL_ARMS, args.plain_run, args.notes_run, args.rl_run)
    record = {
        "frozen_at": datetime.now(timezone.utc).isoformat(),
        "rules": "docs/scaleup-plan-20261105.md",
        "arms": arms,
        "chosen_variant": variant,
        "variant_development_successes": variant_counts,
        "final_1.7b": final,
        "final_development_successes": final_counts,
        "comparator": "0.6b-rl",
        "runs": {
            "plain": str(args.plain_run),
            "notes": str(args.notes_run),
            "variant_choice": str(args.choice_run),
            "rl": str(args.rl_run),
            "final_choice": str(args.rl_choice_run),
        },
        "full_development_lists": not args.allow_partial,
        "source_sha256": source_fingerprint(),
        "scaleup_sha256": fingerprint(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as handle:
        handle.write(json.dumps(record, indent=2) + "\n")
    print(json.dumps({k: record[k] for k in ("chosen_variant", "final_1.7b")}, indent=2))


if __name__ == "__main__":
    main()
