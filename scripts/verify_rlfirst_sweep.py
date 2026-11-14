"""Audit the RL-first base sweep on the laptop and summarize it (plan addendum 1).

Every reply is parsed again with the same code, and every distinct (task, submitted
source) pair is regraded in the pinned Docker grader. Public results shown to the model
before a retry, both rewards, false passes, and identical retries are recomputed. Any
difference is reported, never corrected. Then the sweep is summarized: base solve rate by
kind, informative groups (preflight item 6's rule), and tasks retired before RL.

Usage: python3 -B scripts/verify_rlfirst_sweep.py --pool artifacts/rlfirst-pool-v2 \
  --sweep artifacts/rlfirst-sweep-v1 [--workers 6]
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from repairlab.execution import DockerExecutor  # noqa: E402
from repairlab.experiments import write_json  # noqa: E402
from repairlab.rlfirst.episode import grade, outcome, parse  # noqa: E402

INFORMATIVE_SHARE = 0.25


def load_records(sweep):
    records = []
    for path in sorted((sweep / "shards").glob("*/extracted/**/sweep.json")):
        records.extend(json.loads(path.read_text())["records"])
    return records


def comparable(public_result):
    if not public_result or public_result.get("status") != "graded":
        return {"status": (public_result or {}).get("status")}
    return {
        "status": "graded",
        "cases": [
            {k: c.get(k) for k in ("value", "error", "mutated_input", "passed")}
            for c in public_result["cases"]
        ],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--sweep", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--destination", type=Path, default=None)
    args = parser.parse_args()
    destination = args.destination or args.sweep
    tasks = {t["task_id"]: t for t in json.loads((args.pool / "pool.json").read_text())["tasks"]}
    plan = json.loads((args.sweep / "experiment-plan.json").read_text())
    planned = [task_id for shard in plan["shards"] for task_id in shard]
    records = load_records(args.sweep)
    docker = DockerExecutor()

    pairs = {}
    for record in records:
        task = tasks[record["task_id"]]
        for attempt in record["attempts"]:
            source = parse(attempt["reply"], task["signature"])["source"]
            if source is not None:
                pairs[(record["task_id"], source)] = None
    keys = list(pairs)
    with ThreadPoolExecutor(args.workers) as pool:
        for key, graded in zip(keys, pool.map(lambda k: grade(docker, tasks[k[0]], k[1]), keys)):
            pairs[key] = graded

    mismatches = []
    for record in records:
        task = tasks[record["task_id"]]
        attempts = []
        for turn, attempt in enumerate(record["attempts"], 1):
            parsed = parse(attempt["reply"], task["signature"])
            graded = pairs.get((record["task_id"], parsed["source"])) if parsed["source"] else None
            attempts.append({"parsed": parsed, "grade": graded})
            where = {"task_id": record["task_id"], "sample": record["sample"], "turn": turn}
            if (parsed["source"] is not None) != attempt["valid"] or parsed["reason"] != attempt[
                "reason"
            ]:
                mismatches.append({**where, "field": "parse"})
            if graded is not None:
                for field in ("public_passed", "hidden_passed"):
                    if graded[field] != attempt[field]:
                        mismatches.append({**where, "field": field})
                if turn < len(record["attempts"]) and comparable(
                    graded["public_result"]
                ) != comparable(attempt["public_result"]):
                    mismatches.append({**where, "field": "public_result_shown"})
        recomputed = outcome(attempts)
        for field, value in recomputed.items():
            if record[field] != value:
                mismatches.append(
                    {"task_id": record["task_id"], "sample": record["sample"], "field": field}
                )

    by_task = defaultdict(list)
    for record in records:
        by_task[record["task_id"]].append(record)
    complete = {t: rs for t, rs in by_task.items() if len(rs) == 8}
    solved = {t: sum(r["hardened_reward"] for r in rs) for t, rs in complete.items()}
    informative = [t for t, k in solved.items() if 0 < k < 8]
    retired = sorted(t for t, k in solved.items() if k == 8)
    never = sorted(t for t, k in solved.items() if k == 0)
    by_kind = defaultdict(lambda: Counter())
    for record in records:
        row = by_kind[record["kind"]]
        row["episodes"] += 1
        row["hardened"] += record["hardened_reward"]
        row["weak"] += record["weak_reward"]
        row["false_pass"] += record["false_pass"]
        row["valid_first"] += record["valid_first"]
    share = len(informative) / len(complete) if complete else 0.0
    mean_solve = sum(solved.values()) / (8 * len(solved)) if solved else 0.0
    adjustment = None
    if share < INFORMATIVE_SHARE:
        adjustment = "raise two-bug share" if mean_solve > 0.5 else "lower two-bug share"
    episodes = len(records) or 1
    replies = [a for r in records for a in r["attempts"]]
    valid_rate = sum(a["valid"] for a in replies) / (len(replies) or 1)
    report = {
        "kind": "rlfirst_base_sweep_audit_and_summary",
        "planned_tasks": len(planned),
        "swept_tasks": len(by_task),
        "complete_tasks": len(complete),
        "missing_tasks": sorted(set(planned) - set(by_task)),
        "episodes": len(records),
        "distinct_submissions_regraded": len(pairs),
        "mismatches": mismatches,
        "mismatch_count": len(mismatches),
        "hardened_rate": sum(r["hardened_reward"] for r in records) / episodes,
        "weak_rate": sum(r["weak_reward"] for r in records) / episodes,
        "false_passes": sum(r["false_pass"] for r in records),
        "valid_first_rate": sum(r["valid_first"] for r in records) / episodes,
        "retry_identical": sum(r["retry_identical"] for r in records),
        "informative_tasks": len(informative),
        "informative_share": share,
        "item6_adjustment": adjustment,
        "valid_reply_rate": valid_rate,
        "collapse_threshold": 0.5 if valid_rate >= 0.5 else 0.5 * valid_rate,
        "retired_before_rl": retired,
        "never_solved": len(never),
        "by_kind": {k: dict(v) for k, v in sorted(by_kind.items())},
        "solved_counts": dict(sorted(Counter(solved.values()).items())),
    }
    write_json(destination / "sweep-verification.json", report)
    print(
        json.dumps(
            {
                k: report[k]
                for k in ("episodes", "mismatch_count", "hardened_rate", "informative_share")
            }
        )
    )


if __name__ == "__main__":
    main()
