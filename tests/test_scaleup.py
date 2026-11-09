"""The 1.7B pipeline: notes, supervised variants, sized comparisons, RL rules, and seals."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock

from test_curriculum import ScriptedExecutor
from test_rl import RLExecutor, run_in_process

from repairlab.curriculum.final_test import final_tasks
from repairlab.curriculum.recovery import demonstrations
from repairlab.curriculum.training_programs import training_programs
from repairlab.final2.programs import final2_tasks
from repairlab.rl.programs import rl_programs
from repairlab.scaleup.comparison import SizedComparison, debugging_metrics
from repairlab.scaleup.freeze import chosen_variant, final_checkpoint
from repairlab.scaleup.humanevalfix_native import (
    DROPPED,
    Unconvertible,
    conversions,
    native_tasks,
    test_cases,
)
from repairlab.scaleup.models import LARGE, MODELS, RL_06B, arm, development_tasks
from repairlab.scaleup.notes import DIAGNOSES, failing_case, note
from repairlab.scaleup.rl import CONFIG, select_programs
from repairlab.scaleup.supervised import variant_samples
from repairlab.tasks import canonical, pilot_tasks, sha256

ROOT = Path(__file__).resolve().parents[1]


def large():
    """Arms on the 1.7B base, for tests that run before or without its measured hash."""
    return mock.patch.dict(LARGE, base_sha256=LARGE["base_sha256"] or "d" * 64)


class NoteTests(unittest.TestCase):
    def test_every_training_program_has_two_diagnoses(self):
        self.assertEqual(set(DIAGNOSES), {p.task.id for p in training_programs()})
        for texts in DIAGNOSES.values():
            self.assertEqual(len(texts), 4)
            self.assertTrue(all(t and not t.endswith(".") for t in texts))

    def test_failing_case_quotes_the_first_failure(self):
        observation = {
            "cases": [
                {"args": [1], "expected": 1, "value": 1, "passed": True},
                {"args": [[1, 2], "a"], "expected": [2], "value": [1], "passed": False},
                {"args": [3], "expected": 0, "value": 9, "passed": False},
            ]
        }
        self.assertEqual(failing_case(observation), 'solve([1,2], "a") returned [1], expected [2]')
        changed = {"cases": [{"args": [[1]], "value": [], "mutated_input": True, "passed": False}]}
        self.assertIn("but changed its input", failing_case(changed))
        raised = {"cases": [{"args": [0], "error": "ZeroDivisionError", "passed": False}]}
        self.assertEqual(failing_case(raised), "solve(0) raised ZeroDivisionError")

    def test_note_after_reading_has_no_failing_case(self):
        text = note("leap-year", {"source": "def solve(year):\n    pass\n"}, False)
        self.assertTrue(text.startswith("Cause: century years"))


class SupervisedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.plain = variant_samples(ScriptedExecutor(), "plain")
        cls.notes = variant_samples(ScriptedExecutor(), "notes")
        cls.programs = {p.task.id: p for p in training_programs()}

    def test_plain_is_exactly_the_recovery_data(self):
        self.assertEqual(self.plain, demonstrations(ScriptedExecutor()))

    def test_notes_change_only_the_text_of_repair_edits(self):
        self.assertEqual(len(self.notes), len(self.plain))
        for plain, noted in zip(self.plain, self.notes):
            program = self.programs[plain["program_id"]]
            for a, b in zip(
                plain["prompt"] + plain["completion"], noted["prompt"] + noted["completion"]
            ):
                if a == b:
                    continue
                self.assertEqual(
                    {k: v for k, v in b.items() if k != "content"},
                    {k: v for k, v in a.items() if k != "content"},
                )
                source = b["tool_calls"][0]["function"]["arguments"]["source"]
                self.assertEqual(source, program.task.reference)
                self.assertTrue(b["content"].startswith(("Failing case: ", "Cause: ")))

    def test_every_edit_target_has_a_note_and_a_history_note_matches_its_target(self):
        targets = {}
        for sample in self.notes:
            completion = sample["completion"][0]
            if completion["tool_calls"][0]["function"]["name"] == "edit":
                self.assertTrue(completion["content"])
                targets[canonical(sample["prompt"])] = completion["content"]
        self.assertEqual(len(targets), 120)
        for sample in self.notes:
            for index, message in enumerate(sample["prompt"]):
                if message.get("content") and message["role"] == "assistant":
                    self.assertEqual(
                        targets[canonical(sample["prompt"][:index])], message["content"]
                    )

    def test_wrong_repairs_carry_no_note_and_are_never_targets(self):
        wrong = {p.wrong for p in self.programs.values()}
        for sample in self.notes:
            for message in sample["prompt"] + sample["completion"]:
                for call in message.get("tool_calls") or []:
                    if call["function"]["arguments"].get("source") in wrong:
                        self.assertEqual(message["content"], "")
                        self.assertIn(message, sample["prompt"])

    def test_no_sealed_content(self):
        text = canonical(self.notes)
        for task in final2_tasks() + final_tasks() + pilot_tasks():
            self.assertNotIn(json.dumps(task.reference)[1:-1], text)


class ComparisonTests(unittest.TestCase):
    def test_development_is_the_pilot_then_the_opened_final_test(self):
        tasks = development_tasks()
        self.assertEqual(len(tasks), 48)
        self.assertEqual([t.id for t in tasks[24:]], [t.id for t in final_tasks()])

    def test_second_final_test_needs_every_task_unseeded(self):
        arms = {"a": arm("0.6b", RL_06B)}
        with tempfile.TemporaryDirectory() as folder:
            for tasks, seeded, split in (
                (final2_tasks()[:1], False, "final_test_2"),
                (final2_tasks(), True, "final_test_2"),
                (final2_tasks(), False, "development"),
                ([replace(final2_tasks()[0], id="renamed")], False, "rl_training"),
            ):
                with self.assertRaises(ValueError):
                    SizedComparison(
                        ScriptedExecutor(), Path(folder) / "x", arms, tasks, seeded, split
                    )
            opened = SizedComparison(
                ScriptedExecutor(), Path(folder) / "ok", arms, final2_tasks(), split="final_test_2"
            )
        self.assertTrue(opened.config["full_task_list"])

    def test_development_tasks_follow_the_fixed_order(self):
        arms = {"a": arm("0.6b", RL_06B)}
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                SizedComparison(
                    ScriptedExecutor(), Path(folder) / "x", arms, development_tasks()[1:3]
                )

    def test_adapters_must_match_their_base(self):
        with self.assertRaises(ValueError):
            arm("0.6b", {**RL_06B, "base_sha256": "e" * 64})
        with large(), tempfile.TemporaryDirectory() as folder:
            wrong = {"a": {"model_key": "1.7b", "adapter": RL_06B}}
            with self.assertRaises(ValueError):
                SizedComparison(ScriptedExecutor(), Path(folder) / "x", wrong, pilot_tasks()[:1])

    def test_metadata_must_name_every_base_and_adapter(self):
        with large(), tempfile.TemporaryDirectory() as folder:
            arms = {"small": arm("0.6b", RL_06B), "large": arm("1.7b")}
            run = SizedComparison(ScriptedExecutor(), Path(folder) / "run", arms, pilot_tasks()[:1])
            batch = run.export()
            self.assertEqual(set(batch["models"]), {"0.6b", "1.7b"})
            good = {
                "protocol": batch["protocol"],
                "budget": batch["budget"],
                "batch_sha256": batch["batch_sha256"],
                "models": {
                    k: {
                        "model_revision": m["model_revision"],
                        "base_initial_sha256": m["base_sha256"],
                        "base_final_sha256": m["base_sha256"],
                    }
                    for k, m in batch["models"].items()
                },
                "arms": {
                    "small": {
                        "model_key": "0.6b",
                        "adapter_file_sha256": RL_06B["file_sha256"],
                        "adapter_initial_sha256": RL_06B["tensor_sha256"],
                        "adapter_final_sha256": RL_06B["tensor_sha256"],
                    },
                    "large": {"model_key": "1.7b", "adapter_file_sha256": None},
                },
            }
            self.assertIsNotNone(run._check_metadata(batch["batch_sha256"], good))
            for broken in (
                {**good, "arms": {**good["arms"], "large": {"model_key": "0.6b"}}},
                {
                    **good,
                    "arms": {
                        **good["arms"],
                        "large": {"model_key": "1.7b", "adapter_file_sha256": "f" * 64},
                    },
                },
                {
                    **good,
                    "models": {
                        **good["models"],
                        "1.7b": {**good["models"]["1.7b"], "base_final_sha256": "0" * 64},
                    },
                },
            ):
                with self.assertRaises(ValueError):
                    run._check_metadata(batch["batch_sha256"], broken)

    def test_debugging_metrics_count_repeats_and_real_recoveries(self):
        task = pilot_tasks()[0]

        def edit(source):
            return {
                "action": {"tool": "edit", "source": source},
                "observation": {"edited": True},
                "source_sha256": sha256(source),
            }

        def test(source, passed):
            return {
                "action": {"tool": "test"},
                "observation": {"passed": passed},
                "source_sha256": sha256(source),
            }

        record = {
            "events": [edit("a"), test("a", False), edit("a"), edit("b"), test("b", True)],
            "result": {"success": True},
        }
        self.assertEqual(
            debugging_metrics(task, record),
            {"edits": 3, "later_edits": 2, "repeated_edits": 1, "real_recovery": True},
        )
        untouched = {"events": [test(task.buggy, False), edit("a")], "result": {"success": True}}
        self.assertFalse(debugging_metrics(task, untouched)["real_recovery"])


class RLRuleTests(unittest.TestCase):
    def test_first_cycle_takes_every_program_then_the_nearest_to_half(self):
        tasks = rl_programs()
        self.assertEqual(select_programs(tasks, []), tasks)
        rates = {t.id: (i % 9) / 8 for i, t in enumerate(tasks)}
        chosen = select_programs(tasks, [rates])
        self.assertEqual(len(chosen), CONFIG["programs_per_cycle"])
        self.assertEqual(chosen, [t for t in tasks if t in chosen])
        worst = max(abs(rates[t.id] - 0.5) for t in chosen)
        self.assertTrue(all(abs(rates[t.id] - 0.5) >= worst for t in tasks if t not in chosen))
        later = select_programs(tasks, [rates, {tasks[0].id: 0.5}])
        self.assertIn(tasks[0], later)

    def test_fixed_configuration(self):
        self.assertEqual(CONFIG["samples_per_program"], 8)
        self.assertEqual(CONFIG["cycles"], 6)
        self.assertEqual(CONFIG["optimizer_steps_per_cycle"], 4)
        self.assertTrue(CONFIG["gradient_checkpointing"])

    def test_rollouts_refuse_evaluation_tasks(self):
        from repairlab.scaleup.rl import SizedRolloutCycle

        with large(), tempfile.TemporaryDirectory() as folder:
            policy = {**RL_06B, "base_sha256": LARGE["base_sha256"]}
            for task in (pilot_tasks()[0], final_tasks()[0], final2_tasks()[0]):
                with self.assertRaises(ValueError):
                    SizedRolloutCycle(RLExecutor(), Path(folder) / "x", 1, policy, [task], 2)


class FreezeTests(unittest.TestCase):
    def comparison(self, folder, counts, full=True, verified=True):
        folder = Path(folder)
        folder.mkdir()
        arms = {name: {"successes": value} for name, value in counts.items()}
        (folder / "report.json").write_text(
            json.dumps({"split": "development", "arms": arms, "full_task_list": full})
        )
        (folder / "independent-verification.json").write_text(
            json.dumps({"verified": verified, "arms": arms})
        )
        return folder

    def test_rule_one_breaks_ties_toward_notes(self):
        with tempfile.TemporaryDirectory() as root:
            tie = self.comparison(Path(root) / "tie", {"1.7b-plain": 20, "1.7b-notes": 20})
            self.assertEqual(chosen_variant(tie)[0], "notes")
            plain = self.comparison(Path(root) / "p", {"1.7b-plain": 21, "1.7b-notes": 20})
            self.assertEqual(chosen_variant(plain)[0], "plain")

    def test_rule_two_needs_rl_strictly_ahead(self):
        with tempfile.TemporaryDirectory() as root:
            tie = self.comparison(Path(root) / "tie", {"1.7b-notes": 25, "1.7b-rl": 25})
            self.assertEqual(final_checkpoint(tie, "notes")[0], "1.7b-notes")
            ahead = self.comparison(Path(root) / "a", {"1.7b-notes": 25, "1.7b-rl": 26})
            self.assertEqual(final_checkpoint(ahead, "notes")[0], "1.7b-rl")

    def test_rules_refuse_partial_or_unaudited_runs(self):
        with tempfile.TemporaryDirectory() as root:
            counts = {"1.7b-plain": 1, "1.7b-notes": 2}
            partial = self.comparison(Path(root) / "p", counts, full=False)
            with self.assertRaises(ValueError):
                chosen_variant(partial)
            unaudited = self.comparison(Path(root) / "u", counts, verified=False)
            with self.assertRaises(ValueError):
                chosen_variant(unaudited)


class HumanEvalFixTests(unittest.TestCase):
    def test_test_cases_accept_only_literal_candidate_calls(self):
        text = "def check(f):\n    assert f([1, 2], 'a') == (1, 2)\n    assert not f(0)\n    assert f(3)\n    assert True\n"
        self.assertEqual(test_cases(text), [([[1, 2], "a"], [1, 2]), ([0], False), ([3], True)])
        for bad in (
            "def check(f):\n    assert abs(f(1) - 2) < 1e-6\n",
            "def check(f):\n    for i in range(3):\n        assert f(i) == i\n",
            "def check(f):\n    assert f(x) == 1\n",
            "def check(f):\n    assert f(1) == {1: 2}\n",
        ):
            with self.assertRaises(Unconvertible):
                test_cases(bad)

    def test_conversion_is_mechanical_and_complete(self):
        rows = conversions()
        self.assertEqual(len(rows), 164)
        converted = [t for _, t, _ in rows if t]
        self.assertEqual(len(converted), 135)
        self.assertTrue(all((t is None) != (r is None) for _, t, r in rows))
        for task in converted:
            self.assertTrue(task.buggy.startswith("def solve(" + task.signature + "):"))
            passes = run_in_process(task.reference, task.public_cases + task.private_cases)
            self.assertTrue(passes or task.id in DROPPED, task.id)
        self.assertLessEqual(set(DROPPED), {t.id for t in converted})
        self.assertEqual(len(native_tasks()), len(converted) - len(DROPPED))

    def test_no_overlap_with_any_sealed_or_training_id(self):
        ids = {t.id for t in native_tasks()}
        earlier = {t.id for t in final2_tasks() + final_tasks() + pilot_tasks() + rl_programs()}
        self.assertFalse(ids & earlier)


class OpeningTests(unittest.TestCase):
    def test_openers_refuse_a_second_opening(self):
        for script, flag in (
            ("run_final_test_2.py", "--open-sealed-final-test-2"),
            ("run_humanevalfix_native.py", "--open-humanevalfix-native"),
        ):
            with tempfile.TemporaryDirectory() as folder:
                registry = Path(folder) / "registry.json"
                registry.write_text("{}")
                frozen = Path(folder) / "frozen.json"
                frozen.write_text(json.dumps({"arms": {}, "final_1.7b": "1.7b-rl"}))
                result = subprocess.run(
                    [
                        sys.executable, "-B", str(ROOT / "scripts" / script),
                        "--output", str(Path(folder) / "run"),
                        "--token-file", str(registry),
                        "--kaggle", str(Path(folder) / "no-such-cli"),
                        "--frozen", str(frozen),
                        "--registry", str(registry),
                        flag,
                    ],
                    capture_output=True,
                    text=True,
                )  # fmt: skip
                self.assertNotEqual(result.returncode, 0)
                self.assertIn("Already opened", result.stderr)
                self.assertFalse((Path(folder) / "run").exists())

    def test_freeze_refuses_to_overwrite(self):
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder) / "frozen.json"
            output.write_text("{}")
            result = subprocess.run(
                [
                    sys.executable, "-B", str(ROOT / "scripts/freeze_scaleup.py"),
                    *[x for name in ("plain", "notes", "choice", "rl", "rl-choice") for x in (f"--{name}-run", folder)],
                    "--output", str(output),
                ],
                capture_output=True,
                text=True,
            )  # fmt: skip
            self.assertIn("already frozen", result.stderr)
            self.assertEqual(output.read_text(), "{}")


class SuspensionTests(unittest.TestCase):
    def test_only_a_timeout_is_graded_again_and_every_retry_is_logged(self):
        sys.path.insert(0, str(ROOT / "scripts"))
        from run_scaleup_comparison import SuspensionSafeExecutor

        class Flaky:
            def __init__(self, statuses):
                self.statuses = list(statuses)

            def metadata(self):
                return {"backend": "unit"}

            def run(self, source, cases, *, reveal=False):
                status = self.statuses.pop(0)
                return {"status": status, "passed": status == "graded"}

        with tempfile.TemporaryDirectory() as folder:
            log = Path(folder) / "retries.jsonl"
            once = SuspensionSafeExecutor(Flaky(["timeout", "graded"]), log)
            self.assertEqual(once.run("x", [])["status"], "graded")
            self.assertEqual(len(log.read_text().splitlines()), 1)
            real = SuspensionSafeExecutor(Flaky(["timeout"] * 3), log)
            self.assertEqual(real.run("x", [])["status"], "timeout")
            self.assertEqual(len(log.read_text().splitlines()), 3)
            graded = SuspensionSafeExecutor(Flaky(["graded"]), log)
            self.assertTrue(graded.run("x", [])["passed"])
            self.assertEqual(real.metadata()["timeout_attempts"], 3)


class ModelTests(unittest.TestCase):
    def test_both_revisions_are_pinned(self):
        self.assertEqual(MODELS["0.6b"]["base_sha256"], RL_06B["base_sha256"])
        self.assertEqual(LARGE["model_revision"], "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e")


@unittest.skipUnless(os.environ.get("REPAIRLAB_DOCKER_TESTS"), "Docker dry run")
class ScaleupDryRunTests(unittest.TestCase):
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

    def kaggle(self, name, *arguments):
        return self.script(name, "--token-file", self.token, "--kaggle", self.fake, *arguments)

    @unittest.skipIf(LARGE["base_sha256"] is None, "needs the preflight's base fingerprint")
    def test_whole_pipeline_through_both_openings(self):
        runs = {name: self.root / name for name in ("plain", "notes", "rl")}
        for variant in ("plain", "notes"):
            self.kaggle("run_scaleup_training.py", "--output", runs[variant], "--variant", variant)
        choice = self.root / "choice"
        self.kaggle(
            "run_scaleup_comparison.py", "--output", choice, "--arms", "1.7b-plain,1.7b-notes",
            "--plain-run", runs["plain"], "--notes-run", runs["notes"], "--tasks", 4,
        )  # fmt: skip
        audit = self.script("verify_scaleup_comparison.py", choice, "--regrade")
        self.assertIn('"verified": true', audit.stdout)
        self.kaggle(
            "run_scaleup_rl.py", "--output", runs["rl"], "--plain-run", runs["plain"],
            "--notes-run", runs["notes"], "--choice-run", choice, "--cycles", 2, "--programs", 3,
        )  # fmt: skip
        audit = self.script("verify_scaleup_rl.py", runs["rl"], "--regrade")
        self.assertIn('"verified": true', audit.stdout)
        summary = json.loads((runs["rl"] / "rl-verification.json").read_text())
        self.assertEqual(summary["start_variant"], "notes")
        final_choice = self.root / "final-choice"
        self.kaggle(
            "run_scaleup_comparison.py", "--output", final_choice, "--arms", "1.7b-notes,1.7b-rl",
            "--notes-run", runs["notes"], "--rl-run", runs["rl"], "--tasks", 4,
        )  # fmt: skip
        self.script("verify_scaleup_comparison.py", final_choice)
        for run in (runs["plain"], runs["notes"]):
            (run / "independent-verification.json").write_text('{"verified": true}')
        frozen = self.root / "frozen.json"
        self.script(
            "freeze_scaleup.py", "--plain-run", runs["plain"], "--notes-run", runs["notes"],
            "--choice-run", choice, "--rl-run", runs["rl"], "--rl-choice-run", final_choice,
            "--output", frozen, "--allow-partial",
        )  # fmt: skip
        record = json.loads(frozen.read_text())
        self.assertEqual(
            set(record["arms"]), {"0.6b-rl", "1.7b-base", "1.7b-plain", "1.7b-notes", "1.7b-rl"}
        )
        registry = self.root / "registry.json"
        final = self.root / "final"
        self.kaggle(
            "run_final_test_2.py", "--output", final, "--frozen", frozen,
            "--registry", registry, "--open-sealed-final-test-2",
        )  # fmt: skip
        audit = self.script("verify_scaleup_comparison.py", final)
        self.assertIn('"sealed_final_2_tasks_used": 48', audit.stdout)
        self.assertEqual(json.loads(registry.read_text())["status"], "completed")
        with self.assertRaises(subprocess.CalledProcessError):
            self.kaggle(
                "run_final_test_2.py", "--output", self.root / "again", "--frozen", frozen,
                "--registry", registry, "--open-sealed-final-test-2",
            )  # fmt: skip
        tokens = "".join(p.read_text(errors="ignore") for p in self.root.rglob("*.json"))
        self.assertNotIn("unit-test-token", tokens)


if __name__ == "__main__":
    unittest.main()
