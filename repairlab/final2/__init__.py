"""A second sealed final test, authored after the first was opened on November 5, 2026.

Kept outside ``repairlab/curriculum/`` and ``repairlab/rl/`` so the fingerprints that
earlier runs recorded stay meaningful. Use ``fingerprint`` below for this package.
"""

from pathlib import Path

from ..rl import fingerprint as rl_fingerprint
from ..tasks import canonical, sha256


def fingerprint():
    files = {p.name: sha256(p.read_text()) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return sha256(canonical({"rl": rl_fingerprint(), "final2": files}))
