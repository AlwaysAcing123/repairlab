"""Split separation, the final-test seal, demonstration targets, and comparison rules."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

from repairlab.curriculum import fingerprint
from repairlab.curriculum.comparison import MatchedComparison, recovery_metrics
from repairlab.curriculum.final_test import GROUPS, final_tasks
from repairlab.curriculum.recovery import bundle, demonstrations, trajectories
from repairlab.curriculum.splits import (
    REVIEWED_NEAR_PAIRS,
    Entry,
    entries,
    leakage_report,
    reject_final,
    remaining_development_tasks,
    split_manifest,
)
from repairlab.curriculum.training_programs import training_programs
from repairlab.curriculum.transcript import initial_messages, retained_repair
from repairlab.experiments import source_fingerprint
from repairlab.native_repair import SYSTEM, TASK_IDS
from repairlab.native_tools import assistant_call
from repairlab.tasks import Case, canonical, manifest, pilot_tasks
from repairlab.tool_warmup import programs as warmup_programs

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from kaggle_client import SOURCE_LIMIT_BYTES, Kaggle, build_notebook  # noqa: E402

PILOT = ROOT / "artifacts/native-repair-pilot-20261030-v1"
ADAPTER = {
    "kernel_source": "unit/test",
    "file_sha256": "a" * 64,
    "tensor_sha256": "b" * 64,
    "base_sha256": "c" * 64,
}


class ScriptedExecutor:
    """Passes only an exact reference; a real observation shape, no container."""

    def __init__(self):
        self.references = {t.reference for t in pilot_tasks()}
        self.references |= {p.task.reference for p in training_programs()}

    def metadata(self):
        return {"backend": "unit-test-only"}

    def run(self, source, cases, *, reveal=False):
        passed = source in self.references
        result = {
            "passed": passed,
            "status": "graded",
            "elapsed_seconds": 0.1,
            "passed_cases": len(cases) if passed else 0,
            "total_cases": len(cases),
        }
        if reveal:
            result["cases"] = [
                {**c.record(), "value": None, "mutated_input": False, "passed": passed}
                for c in cases
            ]
        return result


class SplitTests(unittest.TestCase):
    def test_earlier_audited_evidence_stays_valid(self):
        self.assertEqual(
            source_fingerprint(), json.loads((PILOT / "config.json").read_text())["source_sha256"]
        )
        self.assertEqual(manifest(), json.loads((PILOT / "config.json").read_text())["manifest"])

    def test_remaining_tasks_complete_the_development_set(self):
        remaining = remaining_development_tasks()
        self.assertEqual(len(remaining), 18)
        self.assertFalse({t.id for t in remaining} & set(TASK_IDS))
        self.assertEqual({t.id for t in remaining} | set(TASK_IDS), {t.id for t in pilot_tasks()})
        order = [t.id for t in pilot_tasks()]
        self.assertEqual([t.id for t in remaining], [i for i in order if i not in TASK_IDS])

    def test_split_sizes_and_balance(self):
        train, final = [p.task for p in training_programs()], final_tasks()
        self.assertEqual(
            Counter(t.family for t in train),
            Counter(validation=10, collections=10, control_flow=10),
        )
        self.assertEqual(
            Counter(t.family for t in final), Counter(validation=8, collections=8, control_flow=8)
        )
        self.assertEqual(len({p.group for p in training_programs()}), 30)
        self.assertEqual(len(set(GROUPS.values())), 24)

    def test_no_program_crosses_splits(self):
        report = leakage_report()
        self.assertEqual(report["hard_failures"], [])
        self.assertEqual(report["unreviewed_near_pairs"], [])

    def test_reviewed_near_pairs_are_not_stale(self):
        report = leakage_report()
        found = {frozenset((n["left"], n["right"])) for n in report["near_pairs_for_review"]}
        self.assertEqual(found, set(REVIEWED_NEAR_PAIRS))

    def test_leakage_check_catches_renamed_and_equivalent_programs(self):
        rows = {r.id: r for r in entries()}
        target = rows["stable-unique"]
        renamed = Entry(
            "leak-renamed",
            "leak-renamed",
            "training",
            "Keep the earliest copy of each integer.",
            "items",
            ("def solve(items):\n    return list(dict.fromkeys(items))\n",),
            ("def solve(items):\n    return items\n",),
            target.inputs,
        )
        equivalent = Entry(
            "leak-equivalent",
            "leak-equivalent",
            "training",
            "Unrelated wording about ordering.",
            "items",
            (
                "def solve(items):\n    out = []\n    seen = set()\n    for x in items:\n"
                "        if x not in seen:\n            seen.add(x)\n            out.append(x)\n"
                "    return out\n",
            ),
            ("def solve(items):\n    return items\n",),
            target.inputs,
        )
        report = leakage_report([target, renamed, equivalent])
        self.assertEqual(
            [(f["check"], f["right"]) for f in report["hard_failures"]],
            [("identical_normalized_solution", "leak-renamed")],
        )
        behavior = {
            n["right"] for n in report["near_pairs_for_review"] if n["check"] == "near_behavior"
        }
        self.assertEqual(behavior, {"leak-renamed", "leak-equivalent"})

    def test_group_cannot_cross_splits(self):
        rows = [r for r in entries() if r.id in {"warm-square", "stable-unique"}]
        moved = [replace(r, group="shared") for r in rows]
        self.assertEqual(leakage_report(moved)["hard_failures"][0]["check"], "group_crosses_splits")

    def test_manifest_is_stable_and_records_no_final_evaluation(self):
        first = split_manifest()
        self.assertEqual(first, split_manifest())
        self.assertEqual(first["final_test_model_evaluations"], 0)
        self.assertEqual(
            Counter(r["split"] for r in first["tasks"]),
            {"training": 30, "development": 24, "final_test": 24},
        )


class SealTests(unittest.TestCase):
    def test_reject_final_by_id_and_by_content(self):
        sealed = final_tasks()[0]
        with self.assertRaises(ValueError):
            reject_final([sealed])
        with self.assertRaises(ValueError):
            reject_final([replace(sealed, id="innocent-name")])
        reject_final(pilot_tasks())

    def test_comparison_refuses_sealed_tasks(self):
        with tempfile.TemporaryDirectory() as folder:
            with self.assertRaises(ValueError):
                MatchedComparison(
                    ScriptedExecutor(), Path(folder) / "run", {"a": ADAPTER}, final_tasks()[:1]
                )
            self.assertFalse((Path(folder) / "run").exists())

    def test_demonstrations_refuse_sealed_tasks(self):
        program = training_programs()[0]
        with self.assertRaises(ValueError):
            demonstrations(ScriptedExecutor(), [replace(program, task=final_tasks()[0])])

    def test_notebook_builders_do_not_import_sealed_tasks(self):
        for name in (
            "kaggle_client.py",
            "run_recovery_training.py",
            "run_matched_comparison.py",
            "recovery_training_runtime.py",
            "matched_eval_runtime.py",
        ):
            text = (ROOT / "scripts" / name).read_text()
            self.assertNotIn("final_test", text, name)
            self.assertNotIn("final_tasks", text, name)


class DemonstrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.programs = training_programs()
        cls.samples = demonstrations(ScriptedExecutor(), cls.programs)

    def test_sixteen_unique_samples_per_program(self):
        self.assertEqual(len(self.samples), 480)
        self.assertEqual(len({s["sample_id"] for s in self.samples}), 480)
        keys = [canonical(s["prompt"]) + canonical(s["completion"]) for s in self.samples]
        self.assertEqual(len(set(keys)), 480)

    def test_wrong_repairs_are_history_and_never_targets(self):
        references = {p.task.id: p.task.reference for p in self.programs}
        wrong = {p.wrong for p in self.programs}
        edits = 0
        for sample in self.samples:
            function = sample["completion"][0]["tool_calls"][0]["function"]
            if function["name"] == "edit":
                edits += 1
                self.assertEqual(function["arguments"]["source"], references[sample["program_id"]])
                self.assertNotIn(function["arguments"]["source"], wrong)
        self.assertEqual(edits, 4 * 30)
        recovering = [s for s in self.samples if s["trajectory"] == "recover-wrong"]
        self.assertTrue(
            any(
                canonical(p.wrong) in canonical(s["prompt"])
                for s in recovering
                for p in self.programs
            )
        )

    def test_prompts_match_autonomous_evaluation_format(self):
        tasks = {p.task.id: p.task for p in self.programs}
        for sample in self.samples:
            self.assertEqual(sample["prompt"][:2], initial_messages(tasks[sample["program_id"]]))
            self.assertEqual(sample["prompt"][0]["content"], SYSTEM)

    def test_observations_match_the_real_tool_contract(self):
        observed = 0
        for sample in self.samples:
            for message in sample["prompt"]:
                if message["role"] == "tool" and message["name"] == "test":
                    observed += 1
                    value = json.loads(message["content"])
                    self.assertNotIn("elapsed_seconds", value)
                    self.assertLessEqual(
                        {"passed", "status", "passed_cases", "total_cases", "cases"}, set(value)
                    )
                    self.assertIn("mutated_input", value["cases"][0])
        self.assertGreater(observed, 0)

    def test_failed_test_is_followed_by_revision_and_passed_test_by_submit(self):
        for sample in self.samples:
            tool = sample["completion"][0]["tool_calls"][0]["function"]["name"]
            last = sample["prompt"][-1]
            if last["role"] == "tool" and last["name"] == "test":
                expected = "submit" if json.loads(last["content"])["passed"] else "edit"
                self.assertEqual(tool, expected)

    def test_every_trajectory_fits_the_action_budget(self):
        for program in self.programs:
            for actions in trajectories(program).values():
                self.assertLessEqual(len(actions), 8)

    def test_bundle_is_deterministic_and_excludes_evaluation_solutions(self):
        first = bundle(ScriptedExecutor())
        self.assertEqual(first["bundle_sha256"], bundle(ScriptedExecutor())["bundle_sha256"])
        text = canonical(first["train"])
        for task in pilot_tasks() + final_tasks():
            self.assertNotIn(json.dumps(task.reference)[1:-1], text)

    def test_retained_repair_keeps_the_bug_and_stays_one_function(self):
        for task in pilot_tasks():
            self.assertTrue(retained_repair(task.buggy).startswith(task.buggy))
            compile(retained_repair(task.buggy), "retained", "exec")


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def run_arms(self, seeded=False, tasks=None):
        return MatchedComparison(
            ScriptedExecutor(),
            Path(self.folder.name) / "run",
            {"supervised": ADAPTER, "recovery": {**ADAPTER, "file_sha256": "d" * 64}},
            tasks or pilot_tasks()[:2],
            seeded,
        )

    def metadata(self, comparison, batch):
        arms = {}
        for name, adapter in comparison.arms.items():
            arms[name] = {
                "base_initial_sha256": "c" * 64,
                "base_final_sha256": "c" * 64,
                "adapter_file_sha256": adapter["file_sha256"],
                "adapter_initial_sha256": adapter["tensor_sha256"],
                "adapter_final_sha256": adapter["tensor_sha256"],
            }
        return {
            "protocol": batch["protocol"],
            "budget": batch["budget"],
            "model_revision": batch["model_revision"],
            "batch_sha256": batch["batch_sha256"],
            "arms": arms,
        }

    def step(self, comparison, tool, arguments=None):
        batch = comparison.export()
        rows = [
            {
                "episode_id": r["episode_id"],
                "condition": r["condition"],
                "decision": r["decision"],
                "raw": "unit-test-only",
                "parsed": assistant_call(tool, arguments or {}),
                "input_tokens": 10,
                "output_tokens": 5,
            }
            for r in batch["requests"]
        ]
        comparison.apply(batch["batch_sha256"], self.metadata(comparison, batch), rows)

    def test_arms_receive_identical_initial_prompts(self):
        requests = self.run_arms().export()["requests"]
        by_task = {}
        for request in requests:
            by_task.setdefault(request["episode_id"].split("--")[0], []).append(request["messages"])
        for pair in by_task.values():
            self.assertEqual(pair[0], pair[1])

    def test_seeded_mode_starts_after_a_real_failure_and_keeps_the_budget(self):
        comparison = self.run_arms(seeded=True)
        requests = comparison.export()["requests"]
        self.assertEqual([m["role"] for m in requests[0]["messages"]][-2:], ["assistant", "tool"])
        self.assertFalse(json.loads(requests[0]["messages"][-1]["content"])["passed"])
        for _ in range(5):
            self.step(comparison, "test")
        report = comparison.finish()
        for arm in report["arms"].values():
            self.assertEqual(arm["termination_counts"], {"action_budget": 2})

    def test_submit_after_seeded_failure_is_counted_as_unrecovered(self):
        comparison = self.run_arms(seeded=True)
        self.step(comparison, "submit")
        arm = comparison.finish()["arms"]["supervised"]
        self.assertEqual(arm["episodes_with_failed_test"], 2)
        self.assertEqual(arm["submitted_on_failure"], 2)
        self.assertEqual(arm["revised_after_failure"], 0)
        self.assertEqual(arm["recovered"], 0)

    def test_revision_after_failure_is_measured(self):
        task = pilot_tasks()[0]
        comparison = self.run_arms(seeded=True, tasks=[task])
        self.step(comparison, "edit", {"source": task.reference})
        self.step(comparison, "test")
        self.step(comparison, "submit")
        arm = comparison.finish()["arms"]["recovery"]
        self.assertEqual(arm["next_tool_after_first_failure"], {"edit": 1})
        self.assertEqual(
            [arm[k] for k in ("revised_after_failure", "retested_after_revision", "recovered")],
            [1, 1, 1],
        )
        self.assertEqual(arm["submitted_on_failure"], 0)

    def test_wrong_checkpoint_is_rejected(self):
        comparison = self.run_arms()
        batch = comparison.export()
        metadata = self.metadata(comparison, batch)
        metadata["arms"]["recovery"]["adapter_final_sha256"] = "wrong"
        with self.assertRaises(ValueError):
            comparison.apply(batch["batch_sha256"], metadata, [])

    def test_recovery_metrics_use_source_not_position(self):
        failed = {
            "action": {"tool": "test"},
            "observation": {"passed": False},
            "source_sha256": "x",
        }
        invalid = {
            "action": {"invalid_native_output": "?"},
            "observation": {"error": "bad"},
            "source_sha256": "x",
        }
        submit = {
            "action": {"tool": "submit"},
            "observation": {"submitted": True},
            "source_sha256": "x",
        }
        self.assertTrue(recovery_metrics([failed, invalid, submit])["submitted_on_failure"])
        self.assertEqual(recovery_metrics([submit]), {"saw_failure": False})


class FingerprintTests(unittest.TestCase):
    def test_curriculum_fingerprint_is_stable(self):
        self.assertEqual(fingerprint(), fingerprint())

    def test_warmup_groups_are_separate_from_new_training(self):
        self.assertFalse(
            {p.group for p in warmup_programs() if p.split == "development"}
            & {p.group for p in training_programs()}
        )

    def test_cases_are_records(self):
        self.assertEqual(Case([1], 2).record(), {"args": [1], "expected": 2})


class NotebookTests(unittest.TestCase):
    def test_compressed_payload_round_trips_within_the_size_limit(self):
        payload = {"train": [{"text": "the same public words é " * 40} for _ in range(5000)]}
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "notebook.ipynb"
            notebook = build_notebook(
                path,
                "x",
                "# x",
                payload,
                "recovery_training_runtime.py",
                "go",
                "x-*",
                compress=True,
            )
            scope = {}
            exec(notebook["cells"][2]["source"], scope)
            self.assertEqual(scope["PAYLOAD"], payload)
            self.assertLess(path.stat().st_size, SOURCE_LIMIT_BYTES)

    def test_oversized_notebook_is_rejected_before_upload(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / "upload").mkdir()
            (root / "upload/n.ipynb").write_bytes(b"x" * SOURCE_LIMIT_BYTES)
            (root / "token").write_text("unit-test-token")
            client = Kaggle(root / "no-such-cli", root / "token")
            with self.assertRaisesRegex(RuntimeError, "1 MB"):
                client.run(root, {"cells": []}, {"code_file": "n.ipynb"})

    def test_failed_download_is_logged_and_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cli = root / "cli"
            cli.write_text(
                '#!/bin/sh\nif [ ! -f "$0.ran" ]; then touch "$0.ran"; echo first; exit 1; fi\n'
                "echo second\n"
            )
            cli.chmod(0o755)
            (root / "token").write_text("unit-test-token")
            client = Kaggle(cli, root / "token")
            client._download("owner/kernel/1", root, pause_seconds=0)
            self.assertEqual((root / "download-failed-1.log").read_text(), "first\n")
            self.assertEqual((root / "download.log").read_text(), "second\n")
            (root / "cli.ran").unlink()
            cli.write_text("#!/bin/sh\nexit 1\n")
            with self.assertRaisesRegex(RuntimeError, "Kaggle command failed"):
                client._download("owner/kernel/1", root / "again", pause_seconds=0)

    def test_repeated_status_failures_keep_every_log(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            cli = root / "cli"
            cli.write_text(
                '#!/bin/sh\nn=$(cat "$0.count" 2>/dev/null || echo 0); echo $((n+1)) > "$0.count"\n'
                'if [ "$n" -lt 2 ]; then echo "lost DNS $n"; exit 1; fi\necho running\n'
            )
            cli.chmod(0o755)
            (root / "token").write_text("unit-test-token")
            client = Kaggle(cli, root / "token")
            log = root / "worker-status.txt"
            self.assertEqual(client._repeatable(["status"], log, 5, 0), "running\n")
            self.assertEqual((root / "worker-status-failed-1.txt").read_text(), "lost DNS 0\n")
            self.assertEqual((root / "worker-status-failed-2.txt").read_text(), "lost DNS 1\n")


@unittest.skipUnless(os.environ.get("REPAIRLAB_DOCKER_TESTS"), "Docker dry run")
class DryRunTests(unittest.TestCase):
    """Real runners, real Docker grading, and a scripted stand-in for the Kaggle CLI."""

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

    def test_expansion_runs_and_passes_the_frozen_audit(self):
        run = self.root / "expansion"
        self.script(
            "run_native_repair_pilot.py",
            "--task-set", "remaining",
            "--output", run,
            "--token-file", self.token,
            "--kaggle", self.fake,
        )  # fmt: skip
        report = json.loads((run / "report.json").read_text())
        self.assertEqual(report["tasks_per_condition"], 18)
        self.assertEqual(report["arms"]["supervised"]["successes"], 18)
        self.assertEqual(report["arms"]["original"]["successes"], 0)
        self.assertEqual(report["arms"]["original"]["termination_counts"], {"action_budget": 18})
        config = json.loads((run / "config.json").read_text())
        pilot = json.loads((PILOT / "config.json").read_text())
        for key in ("source_sha256", "system_message", "budget", "tools", "adapter", "manifest"):
            self.assertEqual(config[key], pilot[key], key)
        self.assertEqual(config["task_ids"], [t.id for t in remaining_development_tasks()])
        audit = self.script("verify_native_repair_pilot.py", run, "--regrade")
        self.assertIn('"verified": true', audit.stdout)
        self.assertIn('"environment_matches_pilot"', audit.stdout)
        self.assertNotIn(
            "unit-test-token", "".join(p.read_text(errors="ignore") for p in run.rglob("*.json"))
        )

    def test_recovery_training_then_seeded_comparison(self):
        training = self.root / "training"
        self.script(
            "run_recovery_training.py",
            "--output", training,
            "--token-file", self.token,
            "--kaggle", self.fake,
        )  # fmt: skip
        plan = json.loads((training / "experiment-plan.json").read_text())
        self.assertEqual(plan["training_samples"], 480)
        self.assertEqual(plan["training_programs"], 30)
        run = self.root / "seeded"
        self.script(
            "run_matched_comparison.py",
            "--output", run,
            "--token-file", self.token,
            "--kaggle", self.fake,
            "--recovery-run", training,
            "--seeded",
        )  # fmt: skip
        report = json.loads((run / "report.json").read_text())
        self.assertEqual(report["arms"]["supervised"]["submitted_on_failure"], 24)
        self.assertEqual(report["arms"]["recovery"]["recovered"], 24)
        audit = self.script("verify_matched_comparison.py", run, "--regrade")
        self.assertIn('"verified": true', audit.stdout)
        self.assertIn('"sealed_final_tasks_used": 0', audit.stdout)


if __name__ == "__main__":
    unittest.main()
