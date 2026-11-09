"""Model revisions, the frozen 0.6B comparator, and the 48-task development set.

Both model revisions have byte-identical chat templates and tokenizers, so a prompt
renders to the same token ids for every arm.
"""

from ..curriculum.final_test import final_tasks
from ..revision import MODEL, REVISION
from ..rl.rollouts import START as RECOVERY_06B
from ..tasks import pilot_tasks

SMALL = {
    "model": MODEL,
    "model_revision": REVISION,
    "base_sha256": RECOVERY_06B["base_sha256"],
}
LARGE = {
    "model": "Qwen/Qwen3-1.7B",
    "model_revision": "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
    # Parameter fingerprint as loaded in bfloat16 (artifacts/scaleup-preflight-v1).
    "base_sha256": "9131bb923a64aa985b93f8c52b864fb973ffecb7a54f3624c7dfb83d96e63b04",
}
MODELS = {"0.6b": SMALL, "1.7b": LARGE}
# The end of the 0.6B pipeline, frozen November 5 (artifacts/native-rl-training-v1).
RL_06B = {
    "kernel_source": "alwaysacing/repairlab-native-rl-cycle-6",
    "upstream_version": 1,
    "file_sha256": "179592c22a530e43bb8bc6d12e0baea34ed298b75ff07f5881ce1626fc576571",
    "tensor_sha256": "45e360fad2d75f2e5a63c925729f8801716cfd88b3fd85e15eea2b3b95e4b379",
    "base_sha256": SMALL["base_sha256"],
}


def arm(model_key, adapter=None):
    if adapter is not None and adapter["base_sha256"] != MODELS[model_key]["base_sha256"]:
        raise ValueError("Adapter was trained on a different base model")
    return {"model_key": model_key, "adapter": adapter}


def development_tasks():
    """The 24 pilot tasks, then the 24 tasks of the first final test, opened Nov 5."""
    return pilot_tasks() + final_tasks()
