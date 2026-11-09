import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from test_curriculum import ScriptedExecutor

from repairlab.curriculum.comparison import MatchedComparison
from repairlab.curriculum.final_test import final_tasks
from repairlab.curriculum.splits import REVIEWED_NEAR_PAIRS
from repairlab.curriculum.training_programs import training_programs
from repairlab.rl.admission import REVIEWED_RL_NEAR_PAIRS, reject_evaluation, rl_leakage_report
from repairlab.rl.programs import rl_programs
from repairlab.rl.rollouts import CONFIG, SAMPLING, START, RolloutCycle, update_payload
from repairlab.tasks import canonical, pilot_tasks
from repairlab.tool_warmup import programs as warmup_programs

ROOT = Path(__file__).resolve().parents[1]


def run_in_process(source, cases):
    """Authored references and controls only; never model text."""
    namespace = {}
    exec(source, namespace)
    for case in cases:
        args = json.loads(canonical(case.args))
        before = canonical(args)
        try:
            value = namespace["solve"](*args)
        except Exception:
            return False
        if canonical(value) != canonical(case.expected) or canonical(args) != before:
            return False
    return True


class RLExecutor(ScriptedExecutor):
    def __init__(self):
        super().__init__()
        self.references |= {t.reference for t in rl_programs()}


def call(tool, arguments=None):
    arguments = arguments or {}
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": tool, "arguments": arguments}}],
    }


def sampled(request, tool, arguments=None):
    return {
        "episode_id": request["episode_id"],
        "condition": request["condition"],
        "decision": request["decision"],
        "raw": "<tool_call>" + tool + "</tool_call>",
        "parsed": call(tool, arguments),
        "input_tokens": 700,
        "output_tokens": 3,
        "completion_ids": [1, 2, 3],
        "prompt_sha256": "a" * 64,
    }


def metadata(batch):
    evidence = {
        "base_initial_sha256": START["base_sha256"],
        "base_final_sha256": START["base_sha256"],
        "adapter_file_sha256": START["file_sha256"],
        "adapter_initial_sha256": START["tensor_sha256"],
        "adapter_final_sha256": START["tensor_sha256"],
    }
    return {
        "protocol": batch["protocol"],
        "budget": batch["budget"],
        "model_revision": batch["model_revision"],
        "batch_sha256": batch["batch_sha256"],
        "arms": {name: evidence for name in batch["arms"]},
        "seed": CONFIG["seed"],
        "decoding": SAMPLING,
    }


class ProgramTests(unittest.TestCase):
    def test_thirty_balanced_programs_with_new_ids(self):
        programs = rl_programs()
        self.assertEqual(len(programs), 30)
        self.assertEqual(set(Counter(t.family for t in programs).values()), {10})
        earlier = {t.id for t in pilot_tasks() + final_tasks()}
        earlier |= {p.task.id for p in training_programs()} | {p.id for p in warmup_programs()}
        ids = [t.id for t in programs]
        self.assertEqual(len(set(ids)), 30)
        self.assertFalse(set(ids) & earlier)

    def test_authored_controls_hold(self):
        for task in rl_programs():
            cases = task.public_cases + task.private_cases
            self.assertFalse(run_in_process(task.buggy, cases), task.id)
            self.assertFalse(run_in_process(task.buggy, task.public_cases), task.id)
            self.assertTrue(run_in_process(task.reference, cases), task.id)
            self.assertTrue(run_in_process(task.alternative, cases), task.id)
            self.assertTrue(run_in_process(task.shortcut(), task.public_cases), task.id)
            self.assertFalse(run_in_process(task.shortcut(), cases), task.id)

    def test_separated_from_every_earlier_program(self):
        report = rl_leakage_report()
        self.assertEqual(report["hard_failures"], [])
        self.assertEqual(report["unreviewed_near_pairs"], [])
        rl_ids = {t.id for t in rl_programs()}
        found = {
            frozenset((n["left"], n["right"]))
            for n in report["near_pairs_for_review"]
            if {n["left"], n["right"]} & rl_ids
        }
        self.assertEqual(found, set(REVIEWED_RL_NEAR_PAIRS))
        self.assertFalse(REVIEWED_RL_NEAR_PAIRS & REVIEWED_NEAR_PAIRS)

    def test_rl_refuses_development_and_sealed_tasks(self):
        reject_evaluation(rl_programs())
        for task in (pilot_tasks()[0], final_tasks()[0]):
            with self.assertRaises(ValueError):
                reject_evaluation([task])
            with self.assertRaises(ValueError):
                reject_evaluation([replace(task, id="innocent-name")])


class RolloutTests(unittest.TestCase):
    def cycle(self, folder, actions):
        tasks = rl_programs()[:2]
        rollouts = RolloutCycle(RLExecutor(), Path(folder) / "cycle", 1, START, tasks, 2)
        reference = {t.id: t.reference for t in tasks}
        while True:
            batch = rollouts.export()
            if batch is None:
                return rollouts.finish()
            self.assertEqual(batch["kind"], "rl_rollout")
            self.assertEqual(batch["sampling"], SAMPLING)
            outputs = []
            for request in batch["requests"]:
                task_id = request["episode_id"].split("--")[0]
                plan = actions(request["condition"], reference[task_id])
                tool, arguments = plan[request["decision"] - 1]
                outputs.append(sampled(request, tool, arguments))
            rollouts.apply(batch["batch_sha256"], metadata(batch), outputs)

    def test_group_relative_advantages_select_only_mixed_groups(self):
        def actions(arm, reference):
            if arm == "s1":
                return [("edit", {"source": reference}), ("submit", {})]
            return [("submit", {})]

        with tempfile.TemporaryDirectory() as folder:
            result = self.cycle(folder, actions)
        self.assertEqual(result["mean_reward"], 0.5)
        self.assertEqual(result["informative_programs"], 2)
        advantages = {i: e["advantage"] for i, e in result["episodes"].items()}
        self.assertEqual(sorted(set(advantages.values())), [-0.5, 0.5])
        turns = result["training_turns"]
        self.assertEqual(len(turns), 2 * 2 + 2 * 1)
        first = next(t for t in turns if t["turn_id"].endswith("s1--1"))
        self.assertEqual(first["advantage"], 0.5)
        self.assertEqual(first["completion_ids"], [1, 2, 3])
        self.assertEqual([m["role"] for m in first["messages"]], ["system", "user"])
        payload = update_payload(1, START, START["base_sha256"], turns)
        self.assertEqual(payload["config"], CONFIG)
        self.assertEqual(payload, update_payload(1, START, START["base_sha256"], turns))

    def test_uniform_groups_give_no_training_turns(self):
        with tempfile.TemporaryDirectory() as folder:
            result = self.cycle(folder, lambda arm, reference: [("submit", {})])
        self.assertEqual(result["mean_reward"], 0)
        self.assertEqual(result["training_turns"], [])

    def test_sampled_token_records_are_required(self):
        with tempfile.TemporaryDirectory() as folder:
            rollouts = RolloutCycle(
                RLExecutor(), Path(folder) / "cycle", 1, START, rl_programs()[:1], 2
            )
            batch = rollouts.export()
            outputs = [sampled(r, "read") for r in batch["requests"]]
            outputs[0]["completion_ids"] = [1]
            with self.assertRaises(ValueError):
                rollouts.apply(batch["batch_sha256"], metadata(batch), outputs)

    def test_fixed_configuration(self):
        self.assertEqual(CONFIG["cycles"], 6)
        self.assertEqual(CONFIG["samples_per_program"], 4)
        self.assertEqual(SAMPLING["temperature"], 1.0)
        self.assertEqual(SAMPLING["top_k"], 0)


class FinalTestGuardTests(unittest.TestCase):
    def test_only_every_sealed_task_unseeded_may_open(self):
        adapter = {"base_sha256": START["base_sha256"]}
        with tempfile.TemporaryDirectory() as folder:
            for tasks, seeded in ((final_tasks()[:1], False), (final_tasks(), True)):
                with self.assertRaises(ValueError):
                    MatchedComparison(
                        ScriptedExecutor(),
                        Path(folder) / "run",
                        {"a": adapter},
                        tasks,
                        seeded,
                        split="final_test",
                    )
            with self.assertRaises(ValueError):
                MatchedComparison(
                    ScriptedExecutor(), Path(folder) / "dev", {"a": adapter}, final_tasks()
                )
            opened = MatchedComparison(
                ScriptedExecutor(),
                Path(folder) / "run",
                {"a": adapter},
                final_tasks(),
                split="final_test",
            )
        self.assertEqual(opened.config["split"], "final_test")

    def test_runner_refuses_a_second_opening(self):
        with tempfile.TemporaryDirectory() as folder:
            registry = Path(folder) / "registry.json"
            registry.write_text("{}")
            result = subprocess.run(
                [
                    sys.executable,
                    "-B",
                    str(ROOT / "scripts/run_final_test.py"),
                    "--output", str(Path(folder) / "run"),
                    "--token-file", str(registry),
                    "--kaggle", str(Path(folder) / "no-such-cli"),
                    "--recovery-run", folder,
                    "--rl-run", folder,
                    "--registry", str(registry),
                    "--open-sealed-final-test",
                ],
                capture_output=True,
                text=True,
            )  # fmt: skip
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("already opened", result.stderr)
            self.assertFalse((Path(folder) / "run").exists())


@unittest.skipUnless(os.environ.get("REPAIRLAB_DOCKER_TESTS"), "Docker dry run")
class RLDryRunTests(unittest.TestCase):
    """Real runners, Docker grading, and audits with a scripted Kaggle stand-in."""

    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)
        self.root = Path(self.folder.name)
        self.token = self.root / "token"
        self.token.write_text("unit-test-token")
        self.fake = self.root / "kaggle"
        self.fake.write_text(
            f'#!/bin/sh\nexec {sys.executable} -B {ROOT / "tests/fake_kaggle.py"} "$@"\n'
        )
        self.fake.chmod(0o755)
        self.environment = {**os.environ, "FAKE_KAGGLE_STATE": str(self.root / "state")}

    def script(self, name, *arguments):
        return subprocess.run(
            [sys.executable, "-B", str(ROOT / "scripts" / name), *map(str, arguments)],
            env=self.environment,
            capture_output=True,
            text=True,
            check=True,
        )

    def test_rl_cycles_then_four_arm_comparison_then_final_test(self):
        rl = self.root / "rl"
        self.script(
            "run_rl_training.py",
            "--output", rl,
            "--token-file", self.token,
            "--kaggle", self.fake,
            "--cycles", 2,
            "--programs", 3,
        )  # fmt: skip
        audit = self.script("verify_rl_training.py", rl, "--regrade")
        self.assertIn('"verified": true', audit.stdout)
        summary = json.loads((rl / "rl-verification.json").read_text())
        self.assertEqual([c["mean_reward"] for c in summary["cycles"]], [0.5, 0.5])
        self.assertTrue(all(c["updated"] for c in summary["cycles"]))
        second = json.loads((rl / "cycles/2/rollouts/config.json").read_text())["policy"]
        first = json.loads((rl / "cycles/1/cycle-adapter.json").read_text())
        self.assertEqual(second["tensor_sha256"], first["tensor_sha256"])
        self.assertNotIn(
            "unit-test-token", "".join(p.read_text(errors="ignore") for p in rl.rglob("*.json"))
        )
        recovery = self.root / "recovery"
        recovery.mkdir()
        (recovery / "recovery-adapter.json").write_text(json.dumps(START))
        comparison = self.root / "comparison"
        self.script(
            "run_matched_comparison.py",
            "--output", comparison,
            "--token-file", self.token,
            "--kaggle", self.fake,
            "--arms", "original,supervised,recovery,rl",
            "--recovery-run", recovery,
            "--rl-run", rl,
        )  # fmt: skip
        report = json.loads((comparison / "report.json").read_text())
        self.assertEqual(report["arms"]["rl"]["successes"], 24)
        audit = self.script("verify_matched_comparison.py", comparison, "--regrade")
        self.assertIn('"sealed_final_tasks_used": 0', audit.stdout)
        registry = self.root / "registry.json"
        final = self.root / "final"
        self.script(
            "run_final_test.py",
            "--output", final,
            "--token-file", self.token,
            "--kaggle", self.fake,
            "--recovery-run", recovery,
            "--rl-run", rl,
            "--registry", registry,
            "--open-sealed-final-test",
        )  # fmt: skip
        audit = self.script("verify_matched_comparison.py", final, "--regrade")
        self.assertIn('"sealed_final_tasks_used": 24', audit.stdout)
        self.assertEqual(json.loads(registry.read_text())["status"], "completed")
        with self.assertRaises(subprocess.CalledProcessError):
            self.script(
                "run_final_test.py",
                "--output", self.root / "again",
                "--token-file", self.token,
                "--kaggle", self.fake,
                "--recovery-run", recovery,
                "--rl-run", rl,
                "--registry", registry,
                "--open-sealed-final-test",
            )  # fmt: skip


if __name__ == "__main__":
    unittest.main()
