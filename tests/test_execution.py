"""Opt-in integration checks against the real container boundary."""

import os
import unittest

from repairlab.execution import DockerExecutor
from repairlab.tasks import Case


@unittest.skipUnless(os.environ.get("REPAIRLAB_DOCKER_TESTS") == "1", "Opt in to Docker tests")
class ExecutionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.executor = DockerExecutor(timeout=5)

    def run_source(self, source, expected=1, args=None):
        return self.executor.run(
            source, (Case([] if args is None else args, expected),), reveal=True
        )

    def test_real_execution_and_type_sensitive_comparison(self):
        self.assertTrue(self.run_source("def solve():\n    return 1\n")["passed"])
        self.assertFalse(self.run_source("def solve():\n    return True\n")["passed"])

    def test_candidate_exception_is_a_failure(self):
        result = self.run_source("def solve():\n    raise ValueError('bad')\n")
        self.assertFalse(result["passed"])
        self.assertEqual(result["cases"][0]["error"], "ValueError")

    def test_input_mutation_is_rejected(self):
        result = self.run_source("def solve(xs):\n    xs.append(1)\n    return 1\n", args=[[]])
        self.assertFalse(result["passed"])
        self.assertTrue(result["cases"][0]["mutated_input"])

    def test_infinite_loop_is_bounded(self):
        result = self.run_source("def solve():\n    while True:\n        pass\n")
        self.assertFalse(result["passed"])
        self.assertIn(result["status"], {"timeout", "resource_limit", "execution_error"})

    def test_network_is_unavailable(self):
        code = (
            "def solve():\n    import socket\n    s = socket.socket()\n"
            "    s.settimeout(0.2)\n    try:\n        s.connect(('1.1.1.1', 53))\n"
            "        return True\n    except OSError:\n        return False\n"
            "    finally:\n        s.close()\n"
        )
        self.assertTrue(self.run_source(code, expected=False)["passed"])

    def test_host_files_are_unavailable(self):
        path = os.path.abspath(__file__)
        code = f"def solve():\n    import os\n    return os.path.exists({path!r})\n"
        self.assertTrue(self.run_source(code, expected=False)["passed"])

    def test_forged_pass_field_cannot_override_independent_comparison(self):
        code = (
            "def solve():\n    import os, sys\n"
            '    sys.__stdout__.write(\'{"results":[{"value":999,'
            '"mutated_input":false,"passed":true}]}\')\n'
            "    sys.__stdout__.flush()\n    os._exit(0)\n"
        )
        self.assertFalse(self.run_source(code)["passed"])

    def test_excessive_direct_output_is_bounded(self):
        code = "def solve():\n    import os\n    while True:\n        os.write(1, b'x' * 65536)\n"
        result = self.run_source(code)
        self.assertFalse(result["passed"])
        self.assertEqual(result["status"], "output_limit")

    def test_external_assert_suite_checks_both_pass_and_failure(self):
        tests = "assert solve(1) == 2\nassert solve(2) == 3\n"
        self.assertTrue(
            self.executor.run_suite("def solve(x):\n    return x + 1\n", tests)["passed"]
        )
        self.assertFalse(self.executor.run_suite("def solve(x):\n    return x\n", tests)["passed"])
        self.assertFalse(self.executor.run_suite("raise SystemExit(0)\n", tests)["passed"])


if __name__ == "__main__":
    unittest.main()
