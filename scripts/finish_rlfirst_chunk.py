"""Finish the laptop side of an RL chunk whose notebook completed but whose download was refused.

For a chunk where Kaggle reported COMPLETE and the output archive was downloaded, but the
laptop's archive check stopped the launcher (the first hardened chunk: its 140 MB
optimizer file exceeded the old 64 MB per-file limit; Addendum 7). This does what the
launcher would have done next, and nothing else:
- checks again that the server notebook is private and matches the uploaded source;
- extracts the archive with the client's path and size checks;
- writes ``round-verification.json`` and ``chunk-result.json`` as the launcher does.
Afterwards, run the launcher again with ``--resume`` to start the next chunk.

Usage: python3 -B scripts/finish_rlfirst_chunk.py --run artifacts/rlfirst-rl-hardened-v1 --chunk 1
"""

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from kaggle_client import extract_output  # noqa: E402

from repairlab.experiments import write_json  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--chunk", type=int, required=True)
    parser.add_argument("--reason", required=True)
    args = parser.parse_args()
    folder = args.run / "chunks" / str(args.chunk)
    if (folder / "chunk-result.json").exists() or (folder / "extracted").exists():
        raise SystemExit("This chunk was already finished")
    status = (folder / "worker-status.txt").read_text()
    if '"KernelWorkerStatus.COMPLETE"' not in status:
        raise SystemExit("The notebook did not complete: " + status.strip())
    response = (folder / "submission-response.txt").read_text()
    version = int(re.search(r"Kernel version (\d+) successfully pushed", response).group(1))
    upload = json.loads((folder / "upload" / "kernel-metadata.json").read_text())
    server = json.loads((folder / "server" / "kernel-metadata.json").read_text())
    if server.get("is_private") is not True:
        raise SystemExit("Server did not confirm a private notebook")
    local = json.loads((folder / "upload" / upload["code_file"]).read_text())
    remote = json.loads(next((folder / "server").glob("*.ipynb")).read_text())
    if [c["source"] for c in remote["cells"]] != [c["source"] for c in local["cells"]]:
        raise SystemExit("Submitted notebook source mismatch")
    extracted, archive = extract_output(folder)
    write_json(
        folder / "round-verification.json",
        {
            "server_private": True,
            "server_sources_match": True,
            "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
            "worker_complete": True,
            "credentials_uploaded": False,
            "kernel_version": version,
            "finished_by": "scripts/finish_rlfirst_chunk.py",
            "reason": args.reason,
        },
    )
    log = json.loads(next(extracted.rglob("log.json")).read_text())
    result = {
        "chunk": args.chunk,
        "kernel": upload["id"],
        "version": version,
        "stop_reason": log["stop_reason"],
        "state_step": log["state_step"],
        "steps_this_chunk": len(log["steps"]),
    }
    if result["steps_this_chunk"] == 0:
        raise SystemExit("The chunk made no progress; diagnose before continuing")
    write_json(folder / "chunk-result.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
