"""Convert HumanEvalFix to the native format and run its Docker controls. No model runs."""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from repairlab.execution import DockerExecutor  # noqa: E402
from repairlab.scaleup.humanevalfix_native import admit  # noqa: E402

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    report = admit(DockerExecutor(), parser.parse_args().output)
    print(json.dumps({k: report[k] for k in ("converted", "dropped", "scored_tasks")}, indent=2))
