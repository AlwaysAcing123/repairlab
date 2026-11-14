import importlib.util
import os
import sys
import unittest
from pathlib import Path

from repairlab.rlfirst.episode import (
    first_prompt,
    needs_retry,
    outcome,
    parse,
    retry_prompt,
)
from repairlab.rlfirst.mutate import (
    ALL_KINDS,
    Edit,
    apply,
    candidates,
    combine,
    generate,
    overlapping,
)
from repairlab.rlfirst.sandbox import LocalExecutor
from repairlab.tasks import Case

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = """def solve(values, limit):
    total = 0
    seen = []
    for value in values:
        if value < 0:
            continue
        if value <= limit and value not in seen:
            seen.append(value)
            total += value
    result = list(seen)
    return total
"""


def load_script(name):
    sys.path.insert(0, str(ROOT / "scripts"))
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def task_dict(buggy="def solve(x):\n    return x + 2\n"):
    return {
        "task_id": "t~1",
        "kind": "boundary",
        "specification": "Return x plus one.",
        "signature": "x",
        "buggy": buggy,
        "public": [{"args": [1], "expected": 2}, {"args": [5], "expected": 6}],
        "hidden": [{"args": [n], "expected": n + 1} for n in range(-3, 4) if n != 1],
        "bug_public_result": {
            "status": "graded",
            "cases": [
                {"value": 3, "mutated_input": False, "passed": False},
                {"value": 7, "mutated_input": False, "passed": False},
            ],
        },
    }


class MutationTests(unittest.TestCase):
    def test_kinds_cover_plan_table(self):
        kinds = {m.kind for m in generate(REFERENCE)}
        for kind in (
            "boundary",
            "missing_edge_case",
            "wrong_variable",
            "wrong_init",
            "statement_order",
            "missing_step",
            "wrong_scope",
            "wrong_operation",
            "aliasing",
            "inverted_logic",
            "wrong_return",
        ):
            self.assertIn(kind, kinds)
        self.assertEqual(len(ALL_KINDS), 13)

    def test_mutations_are_local_and_valid(self):
        found = candidates(REFERENCE, "values, limit")
        self.assertGreater(len(found), 40)
        for text, mutation in found.items():
            self.assertNotEqual(text, REFERENCE)
            self.assertTrue(text.startswith("def solve(values, limit):\n"))
            self.assertEqual(apply(REFERENCE, mutation.edits), text)

    def test_specific_mistakes(self):
        found = {m.description: apply(REFERENCE, m.edits) for m in generate(REFERENCE)}
        self.assertIn("if value < limit and", found["<= -> <"])
        self.assertIn("result = seen\n", found["list(seen) copy removed"])
        self.assertNotIn("continue", found["guard deleted"])
        self.assertIn("total = 1\n", found["start 0 -> 1"])

    def test_two_bugs_need_distinct_kinds_and_disjoint_edits(self):
        found = generate(REFERENCE)
        boundary = next(m for m in found if m.kind == "boundary")
        init = next(m for m in found if m.kind == "wrong_init")
        combined = combine(REFERENCE, "values, limit", boundary, init)
        self.assertIsNotNone(combined)
        self.assertEqual(combined[0].kind, "two_bugs")
        self.assertIsNone(combine(REFERENCE, "values, limit", boundary, boundary))
        self.assertTrue(overlapping((Edit(1, 5, ""),), (Edit(4, 6, ""),)))
        self.assertFalse(overlapping((Edit(1, 5, ""),), (Edit(5, 6, ""),)))


class EpisodeTests(unittest.TestCase):
    def test_parse(self):
        good = "Here:\n```python\ndef solve(x):\n    return x + 1\n```"
        self.assertEqual(parse(good, "x")["source"], "def solve(x):\n    return x + 1\n")
        self.assertIsNone(parse("def solve(x): return x", "x")["source"])
        two = good + "\n```python\ndef solve(x):\n    return x\n```"
        self.assertIn("more than one", parse(two, "x")["reason"])
        self.assertIn("not Python", parse("```js\nlet a\n```", "x")["reason"])
        wrong = parse("```python\ndef solve(y):\n    return y\n```", "x")
        self.assertIn("signature", wrong["reason"])
        extra = parse("```python\nimport math\ndef solve(x):\n    return x\n```", "x")
        self.assertIsNone(extra["source"])

    def test_prompts_show_public_results(self):
        task = task_dict()
        text = first_prompt(task, task["bug_public_result"])
        self.assertIn("Return x plus one.", text)
        self.assertIn("- solve(1) returned 3; expected 2. [FAIL]", text)
        self.assertIn("```python\ndef solve(x):\n    return x + 2\n```", text)
        self.assertNotIn("-3", text)  # hidden inputs never shown
        bad = retry_prompt({"source": None, "reason": "no code."}, {}, task)
        self.assertIn("could not be used: no code.", bad)

    def test_outcome_rewards(self):
        def attempt(source, public, hidden):
            grade = None if source is None else {"public_passed": public, "hidden_passed": hidden}
            return {"parsed": {"source": source, "reason": None}, "grade": grade}

        self.assertEqual(outcome([attempt("a", True, True)])["hardened_reward"], 1)
        cheat = outcome([attempt("a", False, False), attempt("b", True, False)])
        self.assertEqual(
            (cheat["hardened_reward"], cheat["weak_reward"], cheat["false_pass"]), (0, 1, 1)
        )
        same = outcome([attempt("a", False, False), attempt("a", False, False)])
        self.assertEqual(same["retry_identical"], 1)
        self.assertTrue(needs_retry(attempt(None, False, False)))
        self.assertFalse(needs_retry(attempt("a", True, False)))


class SandboxTests(unittest.TestCase):
    def test_local_grading_matches_expectations(self):
        executor = LocalExecutor()
        cases = [Case([n], n + 1) for n in range(5)]
        self.assertTrue(executor.run("def solve(x):\n    return x + 1\n", cases)["passed"])
        self.assertFalse(executor.run("def solve(x):\n    return x\n", cases)["passed"])
        looping = executor.run("def solve(x):\n    while True:\n        pass\n", cases)
        self.assertEqual(looping["status"], "resource_limit")
        mutated = executor.run(
            "def solve(x):\n    x.append(1)\n    return 0\n", [Case([[]], 0)], reveal=True
        )
        self.assertFalse(mutated["passed"])
        self.assertTrue(mutated["cases"][0]["mutated_input"])

    @unittest.skipUnless(os.environ.get("REPAIRLAB_DOCKER_TESTS"), "Docker tests disabled")
    def test_local_agrees_with_docker(self):
        from repairlab.execution import DockerExecutor
        from repairlab.rl.programs import rl_programs

        task = rl_programs()[0]
        cases = list(task.public_cases + task.private_cases)
        local, docker = LocalExecutor(), DockerExecutor()
        for source in (task.reference, task.buggy, task.shortcut()):
            a, b = local.run(source, cases, reveal=True), docker.run(source, cases, reveal=True)
            self.assertEqual([c["passed"] for c in a["cases"]], [c["passed"] for c in b["cases"]])


class RuntimeTests(unittest.TestCase):
    def test_play_with_scripted_replies(self):
        runtime = load_script("rlfirst_runtime")
        task = task_dict()
        replies = iter(
            [
                # Turn 1: sample 0 fixes it; sample 1 hard-codes public cases; sample 2 is invalid.
                "```python\ndef solve(x):\n    return x + 1\n```",
                "```python\ndef solve(x):\n    if x == 1:\n        return 2\n    return 6\n```",
                "no code here",
                # Turn 2 only for sample 2: sample 1 passed the public cases and was submitted.
                "```python\ndef solve(x):\n    return x + 1\n```",
            ]
        )

        def scripted(tokenizer, model, conversations, *rest):
            return [{"text": next(replies), "tokens": 10, "finished": True} for _ in conversations]

        runtime._generate = scripted
        records, timing = runtime.play(
            None, None, LocalExecutor(), [task], 3, True, 8, 1, lambda line: None
        )
        by_sample = {r["sample"]: r for r in records}
        self.assertEqual(by_sample[0]["hardened_reward"], 1)
        self.assertEqual(by_sample[0]["turns"], 1)
        self.assertEqual((by_sample[1]["weak_reward"], by_sample[1]["hardened_reward"]), (1, 0))
        self.assertEqual(by_sample[1]["false_pass"], 1)
        self.assertEqual(by_sample[2]["valid_first"], 0)
        self.assertEqual(by_sample[2]["hardened_reward"], 1)
        self.assertEqual(timing["turn2"]["episodes"], 1)

    def test_preflight_decision_rule(self):
        launcher = load_script("run_rlfirst_preflight")
        base = {
            "dtype_check": [
                {
                    "dtype": "bf16",
                    "all_logits_finite": True,
                    "valid_replies": 40,
                    "public_passes": 10,
                },
                {
                    "dtype": "fp16",
                    "all_logits_finite": True,
                    "valid_replies": 41,
                    "public_passes": 12,
                },
            ],
            "hf_sampling": [{"ok": True, "batch_size": 64, "sequences_per_hour": 1000}],
            "vllm": {
                "run_ok": True,
                "sample_sequences": 384,
                "sample_seconds": 600,
                "valid_replies": 39,
                "public_passes": 11,
            },
            "in_session_controls": {"tasks": 48, "reference_passes": 48, "bug_fails_public": 48},
        }
        decision = launcher.decide(base)
        self.assertTrue(decision["use_fp16"])
        self.assertTrue(decision["use_vllm"])  # 2,304 per hour against 1,000
        self.assertTrue(decision["in_session_grading_ok"])
        base["dtype_check"][1]["public_passes"] = 13
        self.assertFalse(launcher.decide(base)["use_fp16"])
        base["dtype_check"][1]["public_passes"] = 12
        base["vllm"]["sample_seconds"] = 1000
        self.assertFalse(launcher.decide(base)["use_vllm"])


if __name__ == "__main__":
    unittest.main()


class CheatTests(unittest.TestCase):
    """The plan's injected cheat: a submission that reads answers from a file."""

    CHEAT = (
        "def solve(x):\n"
        "    return int(open(ANSWERS).read().split()[x + 3])\n"
    )

    def test_audit_flags_file_access(self):
        audit = load_script("verify_rlfirst_rl")
        self.assertIn("open", audit.suspicious(self.CHEAT))
        self.assertEqual(audit.suspicious("def solve(x):\n    return sorted(x)\n"), [])
        self.assertIn("import", audit.suspicious("def solve(x):\n    import os\n    return 1\n"))

    @unittest.skipUnless(os.environ.get("REPAIRLAB_DOCKER_TESTS"), "Docker tests disabled")
    def test_docker_regrade_catches_a_cheat_that_passes_in_session(self):
        import tempfile

        from repairlab.execution import DockerExecutor

        folder = Path(tempfile.mkdtemp())
        answers = folder / "answers.txt"
        answers.write_text(" ".join(str(n + 1) for n in range(-3, 4)))
        os.chmod(folder, 0o755)
        os.chmod(answers, 0o644)
        source = self.CHEAT.replace("ANSWERS", repr(str(answers)))
        cases = [Case([n], n + 1) for n in range(-3, 4)]
        self.assertTrue(LocalExecutor().run(source, cases)["passed"])
        self.assertFalse(DockerExecutor().run(source, cases)["passed"])
