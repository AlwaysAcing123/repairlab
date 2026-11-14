"""A third sealed final test, authored for the RL-first phase before any of its GPU jobs.

Kept outside every earlier package so the fingerprints that earlier runs recorded stay
meaningful. Use ``fingerprint`` below for this package. It chains the scale-up package,
whose fingerprint covers the second sealed test, the RL programs, and the curriculum.
"""

from pathlib import Path

from ..scaleup import fingerprint as scaleup_fingerprint
from ..tasks import canonical, sha256


def fingerprint():
    files = {p.name: sha256(p.read_text()) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return sha256(canonical({"scaleup": scaleup_fingerprint(), "final3": files}))
