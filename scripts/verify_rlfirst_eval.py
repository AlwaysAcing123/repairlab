"""Grade an RL-first development evaluation in Docker and audit what the model was shown.

For every episode: the reply is parsed again; every submitted source is graded in the
pinned Docker grader on public and hidden cases; the public results shown before a retry
are recomputed and compared. Writes results by set and by kind. Differences are
reported, never corrected.

Usage: python3 -B scripts/verify_rlfirst_eval.py --pool artifacts/rlfirst-pool-v2 \
  --run artifacts/rlfirst-dev-base-v1 [--destination <copy>]
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
from repairlab.rlfirst.episode import grade, parse, public_cases  # noqa: E402
from repairlab.tasks import canonical  # noqa: E402

SETS = ("dev-seen", "dev-unseen")


def shown(public_result):
    if not public_result or public_result.get("status") != "graded":
        return {"status": (public_result or {}).get("status")}
    keys = ("value", "error", "mutated_input", "passed")
    return {
        "status": "graded",
        "cases": [{k: c.get(k) for k in keys} for c in public_result["cases"]],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    destination = args.destination or args.run
    tasks, split = {}, {}
    for name in SETS:
        for row in json.loads((args.pool / f"{name}.json").read_text())["tasks"]:
            tasks[row["task_id"]] = row
            split[row["task_id"]] = name
    records = json.loads(next((args.run / "run").rglob("eval.json")).read_text())["records"]
    if sorted(r["task_id"] for r in records) != sorted(tasks):
        raise SystemExit("Episodes do not match the development sets")
    docker = DockerExecutor()
    jobs = {}
    for record in records:
        task = tasks[record["task_id"]]
        for attempt in record["attempts"]:
            source = parse(attempt["reply"], task["signature"])["source"]
            if source is not None:
                jobs[(record["task_id"], source)] = None
    keys = list(jobs)
    with ThreadPoolExecutor(args.workers) as pool:
        for key, result in zip(keys, pool.map(lambda k: grade(docker, tasks[k[0]], k[1]), keys)):
            jobs[key] = result
    mismatches, rows = [], []
    for record in records:
        task = tasks[record["task_id"]]
        attempts = record["attempts"]
        for turn, attempt in enumerate(attempts, 1):
            parsed = parse(attempt["reply"], task["signature"])
            if (parsed["source"] is not None) != attempt["valid"]:
                mismatches.append({"task_id": record["task_id"], "turn": turn, "field": "parse"})
            graded = jobs.get((record["task_id"], parsed["source"])) if parsed["source"] else None
            if graded is not None and turn < len(attempts):
                if shown(graded["public_result"]) != shown(attempt["public_result"]):
                    mismatches.append(
                        {"task_id": record["task_id"], "turn": turn, "field": "public_result_shown"}
                    )
            if graded is not None and graded["public_passed"] != attempt["public_passed"]:
                mismatches.append(
                    {"task_id": record["task_id"], "turn": turn, "field": "public_passed"}
                )
        final = parse(attempts[-1]["reply"], task["signature"])["source"]
        graded = jobs.get((record["task_id"], final)) if final else None
        solved = bool(graded and graded["public_passed"] and graded["hidden_passed"])
        false_pass = bool(graded and graded["public_passed"] and not graded["hidden_passed"])
        first = parse(attempts[0]["reply"], task["signature"])["source"]
        rows.append(
            {
                "task_id": record["task_id"],
                "set": split[record["task_id"]],
                "kind": task["kind"],
                "solved": solved,
                "false_pass": false_pass,
                "turns": len(attempts),
                "valid_first": first is not None,
                "retry_identical": len(attempts) == 2 and first is not None and first == final,
                "unchanged_bug": final is not None
                and canonical(final.strip()) == canonical(task["buggy"].strip()),
            }
        )
    summary = {}
    for name in SETS:
        part = [r for r in rows if r["set"] == name]
        by_kind = defaultdict(Counter)
        for r in part:
            by_kind[r["kind"]]["tasks"] += 1
            by_kind[r["kind"]]["solved"] += r["solved"]
        summary[name] = {
            "tasks": len(part),
            "solved": sum(r["solved"] for r in part),
            "false_passes": sum(r["false_pass"] for r in part),
            "valid_first": sum(r["valid_first"] for r in part),
            "retried": sum(r["turns"] == 2 for r in part),
            "retry_identical": sum(r["retry_identical"] for r in part),
            "by_kind": {k: dict(v) for k, v in sorted(by_kind.items())},
        }
    write_json(
        destination / "eval-results.json",
        {
            "kind": "rlfirst_development_results",
            "summary": summary,
            "mismatches": mismatches,
            "rows": rows,
            "public_cases_checked": sum(len(public_cases(t)) for t in tasks.values()),
        },
    )
    print(
        json.dumps({k: {"solved": v["solved"], "tasks": v["tasks"]} for k, v in summary.items()}),
        "mismatches",
        len(mismatches),
    )


if __name__ == "__main__":
    main()
