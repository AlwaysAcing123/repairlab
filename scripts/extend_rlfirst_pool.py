"""Add the new hand-written bugs (plan addendum 2, item 2) to the pool. No model is involved.

Each bug from ``repairlab/rlfirst/handwritten.py`` gets the same admission as a generated
bug: it keeps the signature, always finishes, changes some answer, is not a duplicate,
and gets a witness public case. Docker then runs the same controls as the pool build. The
new pool is the old pool's tasks plus the admitted bugs; development sets are copied
unchanged. Every kind must stay at or under 15% of the new pool.

Usage: python3 -B scripts/extend_rlfirst_pool.py --base artifacts/rlfirst-pool-v2 \
  --output artifacts/rlfirst-pool-v3
"""

import argparse
import json
import platform
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from build_rlfirst_pool import controls  # noqa: E402

from repairlab.curriculum.splits import normalized_tokens  # noqa: E402
from repairlab.execution import DockerExecutor, validate_source  # noqa: E402
from repairlab.experiments import write_json  # noqa: E402
from repairlab.rlfirst import fingerprint  # noqa: E402
from repairlab.rlfirst.handwritten import HANDWRITTEN  # noqa: E402
from repairlab.rlfirst.mutate import HAND_WRITTEN  # noqa: E402
from repairlab.rlfirst.pool import (  # noqa: E402
    KIND_SHARE,
    _ident,
    evaluate,
    make_task,
    record,
    reject_evaluation_material,
    training_sources,
)
from repairlab.rlfirst.sandbox import LocalExecutor  # noqa: E402
from repairlab.tasks import canonical, sha256  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    sandbox, docker = LocalExecutor(), DockerExecutor()
    write_json(
        out / "config.json",
        {
            "kind": "rlfirst_pool_extension_not_model_results",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "base": str(args.base),
            "base_pool_sha256": sha256((args.base / "pool.json").read_text()),
            "rlfirst_fingerprint": fingerprint(),
            "host_python": platform.python_version(),
            "sandbox": sandbox.metadata(),
            "docker": docker.metadata(),
            "plan": "docs/rl-first-plan-20261107.md (Addendum 2, item 2)",
        },
    )
    base_rows = json.loads((args.base / "pool.json").read_text())["tasks"]
    sources = {t.id: (split, t) for split, t in training_sources()}
    seen = {tuple(normalized_tokens(r["buggy"])) for r in base_rows}
    seen |= {tuple(normalized_tokens(t.reference)) for _, t in sources.values()}
    seen |= {tuple(normalized_tokens(t.buggy)) for _, t in sources.values()}
    added, rejected = [], []
    for program_id, entry in sorted(HANDWRITTEN.items()):
        split, task = sources[program_id]
        text = entry["source"]
        reason = None
        try:
            validate_source(text, task.signature)
        except ValueError as exc:
            reason = f"invalid source: {exc}"
        if reason is None and tuple(normalized_tokens(text)) in seen:
            reason = "duplicate after anonymizing names"
        passed = None if reason else evaluate(sandbox, task, text)
        if reason is None and passed is None:
            reason = "does not finish within limits"
        if reason is None and all(passed):
            reason = "same answers as the reference on every input"
        if reason:
            rejected.append({"program": program_id, "reason": reason})
            continue
        built, info = make_task(task, _ident(task.id, text), HAND_WRITTEN, text, passed, split)
        reject_evaluation_material([built])
        flags = {
            canonical(c.args): ok
            for c, ok in zip(list(task.public_cases) + list(task.private_cases), passed)
        }
        row = controls(docker, built, flags)
        if not row["admitted"]:
            rejected.append({"program": program_id, "reason": row["reason"]})
            continue
        seen.add(tuple(normalized_tokens(text)))
        added.append(
            {
                **record(built),
                **info,
                "description": entry["description"],
                "kind_detail": entry["kind_detail"],
                "written_for": "addendum 2",
                "split": "training_pool",
                "bug_public_result": row["bug_public_result"],
            }
        )
    rows = base_rows + added
    counts = Counter(r["kind"] for r in rows)
    cap = KIND_SHARE * len(rows)
    over = {k: n for k, n in counts.items() if n > cap}
    if over:
        raise SystemExit(f"Kind cap exceeded: {over} of {len(rows)}")
    write_json(out / "pool.json", {"tasks": rows})
    for name in ("dev-seen.json", "dev-unseen.json"):
        shutil.copyfile(args.base / name, out / name)
    summary = json.loads((args.base / "summary.json").read_text())
    summary.update(
        extended_from=str(args.base),
        added_hand_written=len(added),
        rejected_hand_written=rejected,
        pool_tasks=len(rows),
        pool_by_kind_admitted=dict(sorted(counts.items())),
        pool_sha256=sha256(canonical([r["task_sha256"] for r in rows])),
    )
    write_json(out / "summary.json", summary)
    write_json(out / "status.json", {"status": "completed"})
    print(
        "added", len(added), "rejected", rejected, "pool", len(rows), dict(sorted(counts.items()))
    )


if __name__ == "__main__":
    main()
