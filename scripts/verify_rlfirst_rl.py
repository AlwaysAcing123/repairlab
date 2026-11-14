"""Audit an RL-first training run (plan, Audits) on the laptop.

For every saved step:
- every reply is parsed again, and every distinct (task, submitted source) pair is
  regraded in the pinned Docker grader, so the rewards, false passes, and the public
  results shown before a retry are recomputed;
- informative groups and advantages (reward minus group mean) are recomputed;
- trained episodes' token ids are decoded and compared with the saved reply text, and the
  prompt ids with the conversation rebuilt from the task (needs ``transformers``; run
  with a Python that has it, or pass ``--skip-tokens``);
- every task must be a training-pool task;
- every submission is scanned for file, process, network, and introspection calls.
Differences are reported, never corrected.

Usage: <python with transformers> -B scripts/verify_rlfirst_rl.py --pool artifacts/rlfirst-pool-v3 \
  --run artifacts/rlfirst-rl-hardened-v1 [--destination <copy>] [--skip-tokens]
"""

import argparse
import ast
import gzip
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from repairlab.execution import DockerExecutor  # noqa: E402
from repairlab.experiments import write_json  # noqa: E402
from repairlab.rlfirst.episode import first_prompt, grade, outcome, parse, retry_prompt  # noqa: E402

SUSPICIOUS_NAMES = {
    "open",
    "__import__",
    "eval",
    "exec",
    "compile",
    "globals",
    "locals",
    "vars",
    "getattr",
    "setattr",
    "__builtins__",
    "breakpoint",
    "input",
}
SUSPICIOUS_MODULES = {
    "os",
    "sys",
    "subprocess",
    "socket",
    "pathlib",
    "shutil",
    "io",
    "inspect",
    "gc",
}


def suspicious(source):
    """Names in a submission that could read files, spawn processes, or reach the grader."""
    found = set()
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Name) and (
            node.id in SUSPICIOUS_NAMES or node.id in SUSPICIOUS_MODULES
        ):
            found.add(node.id)
        if (
            isinstance(node, ast.Attribute)
            and node.attr.startswith("__")
            and node.attr != "__init__"
        ):
            found.add("." + node.attr)
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            found.add("import")
    return sorted(found)


def shown(public_result):
    if not public_result or public_result.get("status") != "graded":
        return {"status": (public_result or {}).get("status")}
    keys = ("value", "error", "mutated_input", "passed")
    return {
        "status": "graded",
        "cases": [{k: c.get(k) for k in keys} for c in public_result["cases"]],
    }


def check_groups(step, settings, reward_key, problems, repeats):
    """Rebuild the step's groups exactly as ``run_rl`` formed them and check advantages.

    Tasks are drawn in parts of ``draw_chunk``; ``play`` returns ``samples`` records per
    drawn task, in draw order. For each drawn task, in order, the runtime's group is every
    record of that task within the same part (so a task drawn twice in one part forms one
    group of both draws, considered twice). Uniform groups are dropped; the first
    ``groups`` informative ones are trained, with advantage = reward - group mean. A task
    may also be drawn again in a later part of the same step, after the task order is
    reshuffled; that forms a separate group. Both repeats are reported.
    """
    number = step["summary"]["step"]
    episodes = step["episodes"]
    samples = settings["samples"]
    sizes, drawn = [], 0
    while drawn < step["summary"]["draws"]:
        size = min(settings["draw_chunk"], settings["max_draws"] - drawn)
        sizes.append(size)
        drawn += size
    if sum(sizes) * samples != len(episodes) or drawn != step["summary"]["draws"]:
        problems.append({"step": number, "field": "draw_parts"})
        return
    groups, dropped, start, seen_in_step = [], 0, 0, {}
    expected = {}
    for size in sizes:
        part = episodes[start : start + size * samples]
        start += size * samples
        blocks = [part[i : i + samples] for i in range(0, len(part), samples)]
        counts = {}
        for block in blocks:
            task_id = block[0]["task_id"]
            if any(r["task_id"] != task_id for r in block):
                problems.append({"step": number, "field": "draw_block", "task_id": task_id})
                return
            counts[task_id] = counts.get(task_id, 0) + 1
        for task_id, count in counts.items():
            if count > 1:
                repeats["drawn_twice_in_one_draw"].append({"step": number, "task_id": task_id})
            if task_id in seen_in_step:
                repeats["drawn_twice_in_a_step"].append({"step": number, "task_id": task_id})
            seen_in_step[task_id] = True
        for block in blocks:
            task_id = block[0]["task_id"]
            group = [r for r in part if r["task_id"] == task_id]
            rewards = [r[reward_key] for r in group]
            if len(set(rewards)) == 1 or len(groups) >= settings["groups"]:
                dropped += 1
                continue
            groups.append(task_id)
            mean = sum(rewards) / len(rewards)
            for record in group:
                expected[id(record)] = record[reward_key] - mean
    if groups != [g[0] for g in step["groups"]]:
        problems.append({"step": number, "field": "group_order"})
    if len(groups) != step["summary"]["informative_groups"] or dropped != step["summary"][
        "dropped_groups"
    ]:
        problems.append({"step": number, "field": "group_counts"})
    for record in episodes:
        want = expected.get(id(record))
        have = record.get("advantage")
        if (want is None) != (have is None) or (
            want is not None and abs(have - want) > 1e-9
        ):
            problems.append(
                {"step": number, "task_id": record["task_id"], "sample": record["sample"], "field": "advantage"}
            )


def step_files(run):
    files = []
    for path in run.glob("chunks/*/extracted/**/steps/step-*.json.gz"):
        if "abandoned" in str(path.relative_to(run)):
            continue
        files.append(path)
    return sorted(files, key=lambda p: int(p.name.split("-")[1].split(".")[0]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pool", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--destination", type=Path)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--skip-tokens", action="store_true")
    args = parser.parse_args()
    destination = args.destination or args.run
    plan = json.loads((args.run / "experiment-plan.json").read_text())
    reward_key = {"hardened": "hardened_reward", "weak": "weak_reward"}[plan["arm"]]
    tasks = {r["task_id"]: r for r in json.loads((args.pool / "pool.json").read_text())["tasks"]}
    steps = [json.loads(gzip.open(p, "rt").read()) for p in step_files(args.run)]
    numbers = [s["summary"]["step"] for s in steps]
    problems = []
    if numbers != list(range(1, len(numbers) + 1)):
        problems.append({"field": "step_sequence", "steps": numbers})

    pairs = {}
    for step in steps:
        for record in step["episodes"]:
            if record["task_id"] not in tasks:
                problems.append({"field": "task_not_in_pool", "task_id": record["task_id"]})
                continue
            for attempt in record["attempts"]:
                source = parse(attempt["reply"], tasks[record["task_id"]]["signature"])["source"]
                if source is not None:
                    pairs[(record["task_id"], source)] = None
    keys = list(pairs)
    docker = DockerExecutor()
    with ThreadPoolExecutor(args.workers) as pool:
        for key, graded in zip(keys, pool.map(lambda k: grade(docker, tasks[k[0]], k[1]), keys)):
            pairs[key] = graded

    tokenizer = None
    if not args.skip_tokens:
        from transformers import AutoTokenizer

        from repairlab.scaleup.models import LARGE

        tokenizer = AutoTokenizer.from_pretrained(LARGE["model"], revision=LARGE["model_revision"])

    flagged, trained_checked = [], 0
    repeats = {"drawn_twice_in_a_step": [], "drawn_twice_in_one_draw": []}
    for step in steps:
        number = step["summary"]["step"]
        check_groups(step, plan["settings"], reward_key, problems, repeats)
        for record in step["episodes"]:
            task = tasks.get(record["task_id"])
            if task is None:
                continue
            attempts = []
            conversation = [
                {"role": "user", "content": first_prompt(task, task["bug_public_result"])}
            ]
            for turn, attempt in enumerate(record["attempts"], 1):
                parsed = parse(attempt["reply"], task["signature"])
                graded = (
                    pairs.get((record["task_id"], parsed["source"])) if parsed["source"] else None
                )
                attempts.append({"parsed": parsed, "grade": graded})
                where = {
                    "step": number,
                    "task_id": record["task_id"],
                    "sample": record["sample"],
                    "turn": turn,
                }
                if (parsed["source"] is not None) != attempt["valid"]:
                    problems.append({**where, "field": "parse"})
                if graded is not None:
                    for field in ("public_passed", "hidden_passed"):
                        if graded[field] != attempt[field]:
                            problems.append({**where, "field": field})
                    if turn < len(record["attempts"]) and shown(graded["public_result"]) != shown(
                        attempt["public_result"]
                    ):
                        problems.append({**where, "field": "public_result_shown"})
                if parsed["source"] is not None:
                    names = suspicious(parsed["source"])
                    if names:
                        flagged.append({**where, "names": names})
                if tokenizer is not None and "ids" in attempt:
                    trained_checked += 1
                    if (
                        tokenizer.decode(attempt["ids"], skip_special_tokens=True)
                        != attempt["reply"]
                    ):
                        problems.append({**where, "field": "reply_tokens"})
                    rendered = tokenizer.apply_chat_template(
                        conversation,
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False,
                    )
                    if tokenizer(rendered)["input_ids"] != attempt["prompt_ids"]:
                        problems.append({**where, "field": "prompt_tokens"})
                public_result = graded["public_result"] if graded else {}
                conversation = conversation + [
                    {"role": "assistant", "content": attempt["reply"]},
                    {"role": "user", "content": retry_prompt(parsed, public_result, task)},
                ]
            recomputed = outcome(attempts)
            for field, value in recomputed.items():
                if record[field] != value:
                    problems.append(
                        {
                            "step": number,
                            "task_id": record["task_id"],
                            "sample": record["sample"],
                            "field": field,
                        }
                    )

    report = {
        "kind": "rlfirst_rl_audit",
        "arm": plan["arm"],
        "steps": len(steps),
        "episodes": sum(len(s["episodes"]) for s in steps),
        "distinct_submissions_regraded": len(pairs),
        "trained_turns_token_checked": trained_checked,
        "token_check": not args.skip_tokens,
        "problems": problems,
        "problem_count": len(problems),
        "flagged_submissions": flagged,
        "flagged_count": len(flagged),
        "repeated_draws": repeats,
    }
    write_json(destination / "rl-verification.json", report)
    print(
        json.dumps({k: report[k] for k in ("steps", "episodes", "problem_count", "flagged_count")})
    )


if __name__ == "__main__":
    main()
