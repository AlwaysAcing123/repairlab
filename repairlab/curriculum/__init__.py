"""Separated datasets and later experiments, kept out of top-level ``repairlab/*.py``.

``repairlab.experiments.source_fingerprint`` hashes only the top-level modules, and
the saved pilot and warm-up audits require that hash to match exactly. Keeping new
code here leaves those audits valid. Use ``fingerprint`` below for this package.
"""

from pathlib import Path

from ..experiments import source_fingerprint
from ..tasks import canonical, sha256


def fingerprint():
    files = {p.name: sha256(p.read_text()) for p in sorted(Path(__file__).parent.glob("*.py"))}
    return sha256(canonical({"core": source_fingerprint(), "curriculum": files}))
