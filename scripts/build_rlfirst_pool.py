"""Build and admit the RL-first generated-bug pool and development sets. No model is involved.

Generation and the first behavior check use the local process sandbox; every selected
task's controls then run in the pinned Docker grader, which decides admission:

- the reference passes every case (once per program);
- the bug fails at least one public case (so the model sees a real failing test);
- the public-answer shortcut passes every public case and fails a hidden one;
- the bug's per-case verdicts in Docker match the sandbox's (else the task is dropped
  and the mismatch reported).

Docker's revealed public results become the first prompt's test lines.

Usage: python3 -B scripts/build_rlfirst_pool.py --output artifacts/rlfirst-pool-v2
"""

import argparse
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from repairlab.execution import DockerExecutor  # noqa: E402
from repairlab.experiments import write_json  # noqa: E402
from repairlab.rlfirst import fingerprint  # noqa: E402
from repairlab.rlfirst.pool import build, record, summary  # noqa: E402
from repairlab.rlfirst.sandbox import LocalExecutor  # noqa: E402
from repairlab.tasks import canonical, sha256  # noqa: E402


def controls(docker, task, sandbox_passed):
    public = list(task.public_cases)
    hidden = list(task.private_cases)
    bug = docker.run(task.buggy, public + hidden, reveal=True)
    shortcut = docker.run(task.shortcut(), public + hidden, reveal=True)
    row = {"task_id": task.id, "bug_status": bug["status"], "shortcut_status": shortcut["status"]}
    if bug["status"] != "graded" or shortcut["status"] != "graded":
        return {**row, "admitted": False, "reason": "did not finish in Docker"}
    flags = [c["passed"] for c in bug["cases"]]
    order = {canonical(c.args): i for i, c in enumerate(public + hidden)}
    expected_flags = [None] * len(flags)
    for args_key, passed in sandbox_passed.items():
        if args_key in order:
            expected_flags[order[args_key]] = passed
    bug_public_fails = not all(flags[: len(public)])
    short_public = all(c["passed"] for c in shortcut["cases"][: len(public)])
    short_hidden = all(c["passed"] for c in shortcut["cases"][len(public) :])
    agrees = flags == expected_flags
    admitted = bug_public_fails and short_public and not short_hidden and agrees
    reason = None
    if not admitted:
        reason = (
            "sandbox and Docker disagree"
            if not agrees
            else "bug passes every public case"
            if not bug_public_fails
            else "shortcut control failed"
        )
    public_rows = [
        {k: v for k, v in c.items() if k not in ("args", "expected")}
        for c in bug["cases"][: len(public)]
    ]
    return {
        **row,
        "admitted": admitted,
        "reason": reason,
        "bug_public_result": {"status": "graded", "cases": public_rows},
        "bug_failed_cases": flags.count(False),
        "total_cases": len(flags),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    sandbox = LocalExecutor()
    docker = DockerExecutor()
    write_json(
        out / "config.json",
        {
            "kind": "rlfirst_pool_admission_not_model_results",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "rlfirst_fingerprint": fingerprint(),
            "host_python": platform.python_version(),
            "sandbox": sandbox.metadata(),
            "docker": docker.metadata(),
            "plan": "docs/rl-first-plan-20261107.md (Addendum 1)",
        },
    )
    write_json(out / "status.json", {"status": "running"})
    selected = build(sandbox, log=lambda line: print(line, flush=True))
    print(f"selection done in {time.monotonic() - started:.0f}s; running Docker controls")

    # Sandbox verdicts per task, recomputed in the same fixed order for comparison.
    def sandbox_flags(task):
        cases = list(task.public_cases) + list(task.private_cases)
        result = sandbox.run(task.buggy, cases, reveal=True)
        return {canonical(c.args): r["passed"] for c, r in zip(cases, result.get("cases", []))}

    everything = selected["pool"] + selected["dev_seen"] + selected["dev_unseen"]
    with ThreadPoolExecutor(args.workers) as pool:
        flags = list(pool.map(sandbox_flags, everything))
        rows = list(pool.map(lambda pair: controls(docker, *pair), zip(everything, flags)))
    by_id = {row["task_id"]: row for row in rows}

    references = {}
    for task in everything:
        references.setdefault(task.reference, task)
    with ThreadPoolExecutor(args.workers) as pool:
        reference_rows = list(
            pool.map(
                lambda t: (
                    t.id.split("~")[0],
                    docker.run(t.reference, list(t.public_cases) + list(t.private_cases)),
                ),
                references.values(),
            )
        )
    bad_references = [name for name, r in reference_rows if not r["passed"]]
    if bad_references:
        raise SystemExit("Reference control failed: " + ", ".join(bad_references))

    def keep(tasks, split):
        kept = []
        for task in tasks:
            row = by_id[task.id]
            if row["admitted"]:
                kept.append(
                    {
                        **record(task),
                        **selected["meta"][task.id],
                        "split": split,
                        "bug_public_result": row["bug_public_result"],
                    }
                )
        return kept

    pool_rows = keep(selected["pool"], "training_pool")
    seen_rows = keep(selected["dev_seen"], "dev_seen")
    unseen_rows = keep(selected["dev_unseen"], "dev_unseen")
    dropped = [row for row in rows if not row["admitted"]]
    write_json(out / "pool.json", {"tasks": pool_rows})
    write_json(out / "dev-seen.json", {"tasks": seen_rows})
    write_json(out / "dev-unseen.json", {"tasks": unseen_rows})
    write_json(
        out / "controls.json",
        {
            "references_checked": len(reference_rows),
            "references_failed": bad_references,
            "tasks_checked": len(rows),
            "dropped": [{k: row[k] for k in ("task_id", "reason")} for row in dropped],
            "rows": [{k: v for k, v in row.items() if k != "bug_public_result"} for row in rows],
        },
    )
    report = summary(selected)
    report["after_docker_controls"] = {
        "pool": len(pool_rows),
        "dev_seen": len(seen_rows),
        "dev_unseen": len(unseen_rows),
        "dropped": len(dropped),
    }
    report["pool_by_kind_admitted"] = dict(
        sorted(
            {
                k: sum(r["kind"] == k for r in pool_rows) for k in {r["kind"] for r in pool_rows}
            }.items()
        )
    )
    report["pool_sha256"] = sha256(canonical([r["task_sha256"] for r in pool_rows]))
    report["elapsed_seconds"] = round(time.monotonic() - started)
    write_json(out / "summary.json", report)
    write_json(out / "status.json", {"status": "completed"})
    print(canonical(report["after_docker_controls"]), canonical(report["pool_by_kind_admitted"]))


if __name__ == "__main__":
    main()
