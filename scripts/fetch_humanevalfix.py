"""Fetch an immutable, licensed snapshot from the benchmark authors' repository."""

import hashlib
import json
from pathlib import Path
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
REPOSITORY = "bigcode-project/octopack"
RELATIVE = "evaluation/create/humaneval-x/data/python/data/humanevalpack.jsonl"


def fetch(url):
    request = Request(url, headers={"User-Agent": "RepairLab-research-pilot"})
    with urlopen(request, timeout=30) as response:
        return response.read()


def main():
    destination = ROOT / "data" / "humanevalfix"
    if destination.exists():
        raise SystemExit("Snapshot already exists; preserve it instead of overwriting")
    revision = json.loads(fetch(f"https://api.github.com/repos/{REPOSITORY}/commits/main"))["sha"]
    url = f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/{RELATIVE}"
    payload = fetch(url)
    rows = [json.loads(line) for line in payload.decode().splitlines() if line.strip()]
    if len(rows) != 164 or len({row["task_id"] for row in rows}) != 164:
        raise ValueError("Unexpected benchmark snapshot; inspect before adapting")
    license_bytes = fetch(f"https://raw.githubusercontent.com/{REPOSITORY}/{revision}/LICENSE")
    destination.mkdir(parents=True)
    (destination / "humanevalpack-python.jsonl").write_bytes(payload)
    (destination / "LICENSE").write_bytes(license_bytes)
    metadata = {
        "repository": REPOSITORY,
        "revision": revision,
        "source_url": url,
        "file_sha256": hashlib.sha256(payload).hexdigest(),
        "tasks": len(rows),
        "license": "MIT",
        "use": "external evaluation; not training",
    }
    (destination / "snapshot.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
