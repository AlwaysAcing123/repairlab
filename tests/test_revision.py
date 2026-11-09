"""Control matching, idempotency, HTTP access, and no hidden-feedback leakage."""

import json
import tempfile
import threading
import unittest
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from repairlab.revision import BUDGET, MODEL, PROTOCOL, REVISION, RevisionExperiment, server
from repairlab.tasks import pilot_tasks


class Executor:
    def __init__(self):
        self.calls = []

    def metadata(self):
        return {"backend": "unit-test-only"}

    def run(self, source, cases, *, reveal=False):
        self.calls.append((source, cases, reveal))
        return {
            "passed": True,
            "status": "graded",
            "elapsed_seconds": 0.1,
            "passed_cases": len(cases),
            "total_cases": len(cases),
            **({"cases": [c.record() for c in cases]} if reveal else {}),
        }


def metadata():
    return {
        "model": MODEL,
        "model_revision": REVISION,
        "protocol": PROTOCOL,
        "budget": BUDGET,
        "initial_weight_sha256": "a" * 64,
    }


class RevisionTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.executor = Executor()
        self.experiment = RevisionExperiment(
            self.executor, Path(self.temp.name) / "run", pilot_tasks()[:1]
        )
        self.experiment.metadata(metadata())

    def action(self, raw, **extra):
        work = self.experiment.work()
        request = {
            "request_id": f"{work['episode_id']}:{work['decision']}",
            "raw": raw,
            "input_tokens": 100,
            "output_tokens": 20,
            **extra,
        }
        return request, self.experiment.action(request)

    def test_identical_prompts_and_budgets_in_both_conditions(self):
        first = self.experiment.work()
        self.action('{"tool":"submit"}')
        second = self.experiment.work()
        self.assertEqual(first["messages"], second["messages"])
        self.assertEqual(first["max_new_tokens"], second["max_new_tokens"])
        self.assertEqual(first["max_context_tokens"], second["max_context_tokens"])

    def test_withheld_feedback_does_not_execute_or_reveal_results(self):
        _, response = self.action('{"tool":"test"}')
        self.assertEqual(self.executor.calls, [])
        self.assertEqual(response["observation"]["feedback"], "unavailable")
        self.action('{"tool":"submit"}')
        calls_before = len(self.executor.calls)
        _, response = self.action('{"tool":"test"}')
        self.assertEqual(len(self.executor.calls), calls_before + 1)
        self.assertNotIn("elapsed_seconds", response["observation"])
        self.assertEqual(response["observation"]["total_cases"], len(pilot_tasks()[0].public_cases))

    def test_retry_cannot_duplicate_action_or_grading(self):
        request, first = self.action('{"tool":"submit"}')
        before = len(self.executor.calls)
        self.assertEqual(self.experiment.action(request), first)
        self.assertEqual(len(self.executor.calls), before)
        self.assertEqual(len(self.experiment.records), 1)
        with self.assertRaises(ValueError):
            self.experiment.action({**request, "raw": '{"tool":"read"}'})

    def test_submission_response_never_contains_private_grade(self):
        _, response = self.action('{"tool":"submit"}')
        self.assertEqual(response["observation"], {"submitted": True})
        self.assertNotIn("success", response)
        self.assertNotIn("private", response)
        task = pilot_tasks()[0]
        self.assertEqual(self.executor.calls[-1][1], task.private_cases)

    def test_token_budget_violation_rejected_without_action(self):
        work = self.experiment.work()
        with self.assertRaises(ValueError):
            self.experiment.action(
                {
                    "request_id": f"{work['episode_id']}:1",
                    "raw": "{}",
                    "input_tokens": 12000,
                    "output_tokens": 20,
                }
            )
        self.assertEqual(self.experiment.episode.events, [])

    def test_eight_invalid_actions_end_episode_without_autosubmitting(self):
        for _ in range(8):
            self.action("not JSON")
        self.assertEqual(self.experiment.records[0]["termination"], "action_budget")
        self.assertEqual(self.experiment.records[0]["invalid_actions"], 8)
        self.assertEqual(self.executor.calls, [])

    def test_finish_requires_all_episodes_and_unchanged_weights(self):
        with self.assertRaises(ValueError):
            self.experiment.finish({"final_weight_sha256": "a" * 64})
        self.action('{"tool":"submit"}')
        self.action('{"tool":"submit"}')
        with self.assertRaises(ValueError):
            self.experiment.finish({"final_weight_sha256": "b" * 64})
        self.assertEqual(self.experiment.finish({"final_weight_sha256": "a" * 64})["episodes"], 2)

    def test_http_authentication_and_route_boundary(self):
        httpd = server(self.experiment, "s" * 48)
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        endpoint = f"http://127.0.0.1:{httpd.server_port}"
        with self.assertRaises(HTTPError) as failure:
            urlopen(endpoint + "/work", timeout=3)
        self.assertEqual(failure.exception.code, 401)
        with self.assertRaises(HTTPError) as failure:
            urlopen(
                Request(endpoint + "/private", headers={"Authorization": "Bearer " + "s" * 48}),
                timeout=3,
            )
        self.assertEqual(failure.exception.code, 404)
        with urlopen(
            Request(endpoint + "/work", headers={"Authorization": "Bearer " + "s" * 48}), timeout=3
        ) as response:
            self.assertEqual(json.load(response)["decision"], 1)


if __name__ == "__main__":
    unittest.main()
