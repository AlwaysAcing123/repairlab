"""The Qwen3-1.7B pipeline: supervised variants, RL, and the one-time second final test.

Kept outside earlier packages so the fingerprints their runs recorded stay meaningful.
Use ``fingerprint`` below for this package. The plan is docs/scaleup-plan-20261105.md.
"""

from pathlib import Path

from ..final2 import fingerprint as final2_fingerprint
from ..tasks import canonical, sha256


def fingerprint():
    files = {p.name: sha256(p.read_text()) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return sha256(canonical({"final2": final2_fingerprint(), "scaleup": files}))
