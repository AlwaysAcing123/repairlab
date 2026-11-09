"""Sampled rollouts on RL programs and the group-relative update they feed.

Each cycle samples several attempts per program from the current adapter under the
evaluation tools and budgets, grades them locally with the real executor, and scores
each attempt against the mean of its own program's attempts. Only attempts that beat
or trail their group are trained on, and every generated token of such an attempt
shares its advantage. The configuration is fixed before any RL result exists.
"""

from ..curriculum.comparison import MatchedComparison
from ..native_tools import tools
from ..revision import MODEL, REVISION
from ..tasks import canonical, sha256
from . import fingerprint
from .admission import SPLIT, reject_evaluation

PROTOCOL = "native-rl-rollouts-v1"
UPDATE_PROTOCOL = "native-rl-group-update-v1"
START = {
    "kernel_source": "alwaysacing/repairlab-native-recovery-training",
    "upstream_version": 1,
    "file_sha256": "d7e7194d9623137a60d170dc3fdebe3957d2454f8352a08ea455addfab192853",
    "tensor_sha256": "84c8b61f2f195b12643934f31e7dd0c26d9da71ad89ffde5b041f33233c444d1",
    "base_sha256": "2d4bad117c2ae98f4a69740ac05d48c237237d84711349a396c5e98866553204",
}
SAMPLING = {
    "do_sample": True,
    "temperature": 1.0,
    "top_p": 1.0,
    "top_k": 0,
    "num_beams": 1,
    "enable_thinking": False,
}
CONFIG = {
    "cycles": 6,
    "samples_per_program": 4,
    "seed": 7,
    "reward": "1 for an explicit submit passing every public and private check, else 0",
    "advantage": "reward minus the mean reward of the same program's samples this cycle",
    "learning_rate": 2e-05,
    "optimizer": "AdamW, weight decay 0, state reset each cycle",
    "optimizer_steps_per_cycle": 4,
    "clip_ratio": 0.2,
    "gradient_clip": 1.0,
    "kl_penalty": 0.0,
    "max_sequence_tokens": 8192,
    "loss": "token mean of the clipped importance-weighted advantage over each minibatch",
    "checkpoint_rule": "The adapter after the final cycle is the RL checkpoint; "
    "earlier cycles are evidence only.",
}


def kernel_for_cycle(cycle):
    return "alwaysacing/repairlab-native-rl-cycle-" + str(cycle)


class RolloutCycle(MatchedComparison):
    """One cycle of sampled attempts; each sample index is an arm with the same adapter."""

    protocol = PROTOCOL

    def __init__(self, executor, destination, cycle, policy, tasks, samples):
        tasks = list(tasks)
        reject_evaluation(tasks)
        self.cycle, self.policy = cycle, policy
        arms = {"s" + str(k): policy for k in range(1, samples + 1)}
        super().__init__(executor, destination, arms, tasks, split=SPLIT)

    def _extra_config(self):
        return {
            "cycle": self.cycle,
            "policy": self.policy,
            "sampling": SAMPLING,
            "seed": CONFIG["seed"],
            "rl_sha256": fingerprint(),
        }

    def _extra_batch(self):
        return {
            "kind": "rl_rollout",
            "cycle": self.cycle,
            "sampling": SAMPLING,
            "seed": CONFIG["seed"],
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
        if fingerprint() != self.config["rl_sha256"]:
            raise ValueError("RL source changed during the cycle")
        return {**report, **group_advantages(self.states)}


def group_advantages(states):
    """Rewards, group-relative advantages, and the training turns they select."""
    groups = {}
    for ident, state in states.items():
        groups.setdefault(state["task"].id, []).append(ident)
    episodes, turns = {}, []
    for task_id, idents in groups.items():
        rewards = [states[i]["episode"].result["reward"] for i in idents]
        mean = sum(rewards) / len(rewards)
        for ident, reward in zip(idents, rewards):
            advantage = reward - mean
            episodes[ident] = {"task_id": task_id, "reward": reward, "advantage": advantage}
            if advantage == 0:
                continue
            for call in states[ident]["model_calls"]:
                if call.get("stop"):
                    continue
                turns.append(
                    {
                        "turn_id": ident + "--" + str(call["decision"]),
                        "episode_id": ident,
                        "messages": call["messages"],
                        "input_tokens": call["input_tokens"],
                        "prompt_sha256": call["prompt_sha256"],
                        "completion_ids": call["completion_ids"],
                        "advantage": advantage,
                    }
                )
    rewards = [e["reward"] for e in episodes.values()]
    return {
        "episodes": episodes,
        "mean_reward": sum(rewards) / len(rewards),
        "programs": len(groups),
        "informative_programs": sum(
            len({episodes[i]["reward"] for i in idents}) > 1 for idents in groups.values()
        ),
        "training_turns": turns,
    }


def update_payload(cycle, policy, base_sha256, turns):
    body = {
        "kind": "rl_update",
        "protocol": UPDATE_PROTOCOL,
        "cycle": cycle,
        "config": CONFIG,
        "policy": policy,
        "model": MODEL,
        "model_revision": REVISION,
        "base_sha256": base_sha256,
        "tools": tools(),
        "turns": turns,
    }
    return {**body, "payload_sha256": sha256(canonical(body))}
