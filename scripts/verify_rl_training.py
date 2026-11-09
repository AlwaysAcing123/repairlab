"""Audit an RL run: every cycle's rollouts, rewards, advantages, update, and adapter chain.

Each cycle's rollouts get the full comparison audit. Rewards, advantages, and the exact
training payload are recomputed from saved episodes, the uploaded notebook must embed
that payload, and every cycle must start from the previous cycle's saved adapter.
With ``--tokenizer`` every sampled token record must decode to its saved raw text.
Writes its verification into the run directory, so re-audit only a copy.
"""

import argparse
import ast
import base64
import gzip
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from verify_matched_comparison import verify as verify_rollouts  # noqa: E402

from repairlab.experiments import write_json  # noqa: E402
from repairlab.rl.admission import reject_evaluation  # noqa: E402
from repairlab.rl.programs import rl_programs  # noqa: E402
from repairlab.rl.rollouts import CONFIG, START, kernel_for_cycle, update_payload  # noqa: E402
from repairlab.tasks import canonical, sha256  # noqa: E402

COMPRESSED = "import base64,gzip,json\nPAYLOAD=json.loads(gzip.decompress(base64.b64decode("


def load(path):
    return json.loads(Path(path).read_text())


def embedded(notebook):
    for cell in notebook["cells"]:
        if cell["source"].startswith(COMPRESSED):
            literal = cell["source"][len(COMPRESSED) : cell["source"].index(")")]
            return json.loads(gzip.decompress(base64.b64decode(ast.literal_eval(literal))))
    raise AssertionError("No compressed payload in notebook")


def recompute(rollouts):
    config = load(rollouts / "config.json")
    groups = {}
    for ident in config["ordered_jobs"]:
        task_id, arm = ident.split("--")
        groups.setdefault(task_id, []).append(
            (ident, load(rollouts / "episodes" / arm / (task_id + ".json")))
        )
    episodes, turns = {}, []
    for task_id, members in groups.items():
        rewards = [record["result"]["reward"] for _, record in members]
        assert all(
            r == int(record["result"]["success"]) for r, (_, record) in zip(rewards, members)
        )
        mean = sum(rewards) / len(rewards)
        for (ident, record), reward in zip(members, rewards):
            advantage = reward - mean
            episodes[ident] = {"task_id": task_id, "reward": reward, "advantage": advantage}
            if advantage == 0:
                continue
            for call in record["model_calls"]:
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
    return episodes, turns


def check_tokens(rollouts, tokenizer):
    config, checked = load(rollouts / "config.json"), 0
    for ident in config["ordered_jobs"]:
        task_id, arm = ident.split("--")
        for call in load(rollouts / "episodes" / arm / (task_id + ".json"))["model_calls"]:
            if call.get("stop"):
                continue
            assert (
                tokenizer.decode(call["completion_ids"], skip_special_tokens=False) == call["raw"]
            )
            checked += 1
    return checked


def verify(root, regrade=False, tokenizer=None):
    root = Path(root)
    plan = load(root / "experiment-plan.json")
    assert load(root / "status.json")["status"] == "completed"
    assert plan["config"] == CONFIG and plan["start"] == START and plan["split"] == "rl_training"
    tasks = rl_programs()[: len(plan["task_ids"])]
    assert plan["task_ids"] == [t.id for t in tasks]
    reject_evaluation(tasks)
    policy, cycles, decoded = START, [], 0
    for cycle in range(1, plan["cycles"] + 1):
        folder = root / "cycles" / str(cycle)
        rollouts = folder / "rollouts"
        config = load(rollouts / "config.json")
        assert config["cycle"] == cycle and config["policy"] == policy
        assert all(adapter == policy for adapter in config["arms"].values())
        assert len(config["arms"]) == plan["samples_per_program"]
        audit = verify_rollouts(rollouts, regrade)
        episodes, turns = recompute(rollouts)
        report = load(folder / "cycle-report.json")
        assert report["episodes"] == episodes and report["programs"] == len(tasks)
        assert report["mean_reward"] == sum(e["reward"] for e in episodes.values()) / len(episodes)
        payload = load(folder / "update-payload.json")
        assert payload == update_payload(cycle, policy, policy["base_sha256"], turns)
        assert payload["payload_sha256"] == sha256(
            canonical({k: v for k, v in payload.items() if k != "payload_sha256"})
        )
        if tokenizer is not None:
            decoded += check_tokens(rollouts, tokenizer)
        saved = load(folder / "cycle-adapter.json")
        update = {"turns": len(turns), "updated": saved["updated"]}
        if not turns:
            assert saved == {**policy, "updated": False}
        else:
            update_folder = folder / "update"
            local = load(next((update_folder / "upload").glob("*.ipynb")))
            remote = load(next((update_folder / "server").glob("*.ipynb")))
            assert [c["source"] for c in local["cells"]] == [c["source"] for c in remote["cells"]]
            assert embedded(local) == payload
            assert load(update_folder / "server/kernel-metadata.json")["is_private"] is True
            check = load(update_folder / "round-verification.json")
            archives = list((update_folder / "download").glob("*.zip"))
            assert len(archives) == 1
            assert hashlib.sha256(archives[0].read_bytes()).hexdigest() == check["archive_sha256"]
            extracted = update_folder / "extracted"
            metadata = load(extracted / "model-config.json")
            evidence = load(extracted / "weight-evidence.json")
            assert metadata["payload_sha256"] == payload["payload_sha256"]
            assert metadata["config"] == CONFIG and metadata["cycle"] == cycle
            assert evidence["initial_adapter_sha256"] == policy["tensor_sha256"]
            assert evidence["base_initial_sha256"] == evidence["base_final_sha256"]
            assert evidence["base_final_sha256"] == policy["base_sha256"]
            assert evidence["prompts_matched_sampled_tokens"] == len(turns)
            assert evidence["turns"] == len(turns)
            assert evidence["optimizer_steps"] <= CONFIG["optimizer_steps_per_cycle"]
            assert saved["kernel_source"] == kernel_for_cycle(cycle) and saved["updated"] is True
            assert saved["upstream_version"] == check["kernel_version"]
            assert saved["tensor_sha256"] == evidence["final_adapter_sha256"]
            assert saved["base_sha256"] == policy["base_sha256"]
            assert (
                hashlib.sha256(
                    (extracted / "adapter/adapter_model.safetensors").read_bytes()
                ).hexdigest()
                == saved["file_sha256"]
            )
            update.update(
                optimizer_steps=evidence["optimizer_steps"],
                adapter_l2_change=evidence["adapter_l2_change"],
                training_log=load(extracted / "training-log.json"),
            )
        policy = {k: v for k, v in saved.items() if k != "updated"}
        cycles.append(
            {
                "cycle": cycle,
                "mean_reward": report["mean_reward"],
                "successes": sum(e["reward"] for e in episodes.values()),
                "episodes": len(episodes),
                "informative_programs": report["informative_programs"],
                "invalid_native_calls": audit["invalid_native_calls"],
                "fresh_submitted_regrades": len(audit["fresh_submitted_regrades"]),
                **update,
            }
        )
    assert load(root / "rl-adapter.json") == policy
    summary = {
        "verified": True,
        "cycles": cycles,
        "final_policy": policy,
        "uses_fixed_configuration": plan["uses_fixed_configuration"],
        "development_or_final_tasks_used": 0,
        "sampled_turns_decoded": decoded if tokenizer is not None else None,
    }
    write_json(root / "rl-verification.json", summary)
    print(json.dumps({k: v for k, v in summary.items() if k != "cycles"}, indent=2))
    for row in cycles:
        print(json.dumps({k: v for k, v in row.items() if k != "training_log"}))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--regrade", action="store_true")
    parser.add_argument("--tokenizer", type=Path, help="tokenizer.json saved with an adapter")
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer:
        from tokenizers import Tokenizer

        tokenizer = Tokenizer.from_file(str(args.tokenizer))
    verify(args.path, args.regrade, tokenizer)
