"""RL for Qwen3-1.7B: 8 sampled attempts per program, on programs nearest a 50% pass rate.

The update is the 0.6B method unchanged: group-relative advantage, clipped importance
ratio, four optimizer steps per cycle, no KL penalty. Cycle 1 samples every RL program.
Each later cycle samples the 15 programs whose most recent pass rate is nearest one half,
because a program whose attempts all pass or all fail gives no learning signal.
"""

from ..native_tools import tools
from ..rl.admission import reject_evaluation
from ..rl.rollouts import CONFIG as SMALL_CONFIG
from ..rl.rollouts import SAMPLING, group_advantages
from ..tasks import canonical, sha256
from . import fingerprint
from .comparison import SizedComparison
from .models import LARGE, arm

PROTOCOL = "sized-rl-rollouts-v1"
UPDATE_PROTOCOL = "sized-rl-group-update-v1"
CONFIG = {
    **SMALL_CONFIG,
    "samples_per_program": 8,
    "programs_per_cycle": 15,
    "program_rule": "Cycle 1 samples every RL program. Each later cycle samples the 15 "
    "programs whose most recent pass rate is nearest 0.5; ties keep program order.",
    # Set by the preflight rule (artifacts/scaleup-preflight-v1): 6,144 tokens ran out of
    # memory. A longer turn is left out of the update and counted, never truncated.
    "max_sequence_tokens": 4096,
    "sampling_square_cap": 60_000_000,
    "gradient_checkpointing": True,
}


def kernel_for_cycle(cycle):
    return "alwaysacing/repairlab-scaleup-rl-cycle-" + str(cycle)


def select_programs(tasks, history):
    """The programs for the next cycle, from the pass rates of the cycles before it."""
    tasks = list(tasks)
    if not history:
        return tasks
    latest = {}
    for rates in history:
        latest.update(rates)
    position = {t.id: i for i, t in enumerate(tasks)}
    ranked = sorted(tasks, key=lambda t: (abs(latest[t.id] - 0.5), position[t.id]))
    chosen = {t.id for t in ranked[: CONFIG["programs_per_cycle"]]}
    return [t for t in tasks if t.id in chosen]


def pass_rates(episodes):
    groups = {}
    for row in episodes.values():
        groups.setdefault(row["task_id"], []).append(row["reward"])
    return {task_id: sum(r) / len(r) for task_id, r in groups.items()}


class SizedRolloutCycle(SizedComparison):
    """One cycle of sampled attempts; each sample index is an arm with the same adapter."""

    protocol = PROTOCOL

    def __init__(self, executor, destination, cycle, policy, tasks, samples):
        tasks = list(tasks)
        reject_evaluation(tasks)
        self.cycle, self.policy = cycle, policy
        arms = {"s" + str(k): arm("1.7b", policy) for k in range(1, samples + 1)}
        super().__init__(executor, destination, arms, tasks, split="rl_training")

    def _extra_config(self):
        return {
            "cycle": self.cycle,
            "policy": self.policy,
            "decoding": SAMPLING,
            "sampling_square_cap": CONFIG["sampling_square_cap"],
            "rl_config": CONFIG,
        }

    def _extra_batch(self):
        return {
            "kind": "scaleup_rollout",
            "cycle": self.cycle,
            "decoding": SAMPLING,
            "seed": CONFIG["seed"],
            "square_cap": CONFIG["sampling_square_cap"],
        }

    def _check_outputs(self, outputs):
        super()._check_outputs(outputs)
        for row in outputs:
            if row.get("stop"):
                continue
            ids = row.get("completion_ids")
            if (
                not isinstance(ids, list)
                or len(ids) != row["output_tokens"]
                or not all(type(i) is int and i >= 0 for i in ids)
                or not isinstance(row.get("prompt_sha256"), str)
            ):
                raise ValueError("Missing or inconsistent sampled token record")

    def finish(self):
        report = super().finish()
        if fingerprint() != self.config["scaleup_sha256"]:
            raise ValueError("Source changed during the cycle")
        return {**report, **group_advantages(self.states)}


def trainable(turns):
    """Turns that fit the preflight's sequence budget, and the IDs of those that do not."""
    limit = CONFIG["max_sequence_tokens"]
    kept = [t for t in turns if t["input_tokens"] + len(t["completion_ids"]) <= limit]
    return kept, [t["turn_id"] for t in turns if t not in kept]


def update_payload(cycle, policy, turns):
    turns, skipped = trainable(turns)
    body = {
        "kind": "scaleup_update",
        "protocol": UPDATE_PROTOCOL,
        "cycle": cycle,
        "config": CONFIG,
        "policy": policy,
        "model": LARGE["model"],
        "model_revision": LARGE["model_revision"],
        "base_sha256": LARGE["base_sha256"],
        "tools": tools(),
        "turns": turns,
        "skipped_long_turns": skipped,
    }
    return {**body, "payload_sha256": sha256(canonical(body))}
