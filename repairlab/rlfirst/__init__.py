"""RL-first phase (plan: docs/rl-first-plan-20261107.md): generated bugs, two-turn episodes,
in-session grading, and the base sweep.

Kept outside every earlier package so the fingerprints that earlier runs recorded stay
meaningful. Use ``fingerprint`` below for this package.
"""

from pathlib import Path

from ..tasks import canonical, sha256


def fingerprint():
    # Imported here so the Kaggle runtime can ship this package without any sealed module.
    from ..final2 import fingerprint as final2_fingerprint

    files = {p.name: sha256(p.read_text()) for p in sorted(Path(__file__).parent.glob("*.py"))}
    worker = Path(__file__).parent.parent / "worker.py"
    return sha256(
        canonical(
            {
                "final2": final2_fingerprint(),
                "worker": sha256(worker.read_text()),
                "rlfirst": files,
            }
        )
    )
