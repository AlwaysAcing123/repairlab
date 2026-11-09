"""The plan's decision rules, applied to saved and audited development reports only.

Rule 1 picks the supervised variant RL starts from; rule 2 picks the final 1.7B
checkpoint. Both read a comparison's report and refuse one that was not audited,
was a dry run, or lacks the arms the rule compares.
"""

import json
from pathlib import Path


def _successes(run, arms):
    run = Path(run)
    report = json.loads((run / "report.json").read_text())
    audit = json.loads((run / "independent-verification.json").read_text())
    if report["split"] != "development" or audit["verified"] is not True:
        raise ValueError("Rules use audited development comparisons only: " + str(run))
    if set(arms) - set(report["arms"]):
        raise ValueError("The comparison lacks the arms the rule compares")
    counts = {name: report["arms"][name]["successes"] for name in arms}
    if audit["arms"] and {n: audit["arms"][n]["successes"] for n in arms} != counts:
        raise ValueError("Report and audit disagree")
    return counts, report["full_task_list"]


def chosen_variant(run, allow_partial=False):
    """Rule 1: more development successes wins; a tie goes to notes."""
    counts, full = _successes(run, ("1.7b-plain", "1.7b-notes"))
    if not full and not allow_partial:
        raise ValueError("Rule 1 needs every development task")
    variant = "notes" if counts["1.7b-notes"] >= counts["1.7b-plain"] else "plain"
    return variant, counts


def final_checkpoint(run, variant, allow_partial=False):
    """Rule 2: RL only if it solves more development tasks than its starting variant."""
    start = "1.7b-" + variant
    counts, full = _successes(run, (start, "1.7b-rl"))
    if not full and not allow_partial:
        raise ValueError("Rule 2 needs every development task")
    return ("1.7b-rl" if counts["1.7b-rl"] > counts[start] else start), counts
