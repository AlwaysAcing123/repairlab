"""Reinforcement learning on separate, unseen training programs.

Kept outside ``repairlab/curriculum/`` so that package's fingerprint, recorded by every
earlier run, stays meaningful. Use ``fingerprint`` below for this package.
"""

from pathlib import Path

from ..curriculum import fingerprint as curriculum_fingerprint
from ..tasks import canonical, sha256


def fingerprint():
    files = {p.name: sha256(p.read_text()) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return sha256(canonical({"curriculum": curriculum_fingerprint(), "rl": files}))
