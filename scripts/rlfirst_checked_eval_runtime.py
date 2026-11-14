"""Adapter check before an evaluation. Appended to rlfirst_runtime.py, like the supervised trainer.

The frozen record (``scripts/freeze_rlfirst.py``) names each trained arm's adapter by the
SHA-256 of its weights file. This refuses to evaluate unless the adapter mounted from the
kernel source is byte-identical, then runs ``run_eval`` unchanged. The base arm has no
adapter and no hash.
"""


def run_checked_eval(payload):
    import hashlib
    from pathlib import Path

    expected = payload.get("adapter_sha256")
    if payload.get("adapter_glob"):
        found = sorted(Path("/kaggle/input").glob(payload["adapter_glob"]))
        if len(found) != 1:
            raise RuntimeError(f"Expected one adapter, found {len(found)}")
        weights = found[0] / "adapter_model.safetensors"
        digest = hashlib.sha256(weights.read_bytes()).hexdigest()
        if digest != expected:
            raise RuntimeError("The mounted adapter is not the frozen one")
    elif expected is not None:
        raise RuntimeError("A frozen adapter hash was given without an adapter")
    print("ADAPTER CHECK PASSED", payload.get("arm"), expected, flush=True)
    run_eval(payload)
