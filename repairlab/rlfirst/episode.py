"""The two-turn repair episode: prompts, reply parsing, grading, and rewards.

Turn 1 shows the specification, the buggy function, and its public test results. The
model replies with the whole corrected function in one Python code block. If the reply
passes every public case it is submitted; otherwise the model sees the new public
results (or why the reply could not be used) and gets one retry, which is submitted
whatever it does. Rewards: hardened = passes every public and hidden case; weak = passes
every public case. Both are computed for every attempt.

The same functions build prompts on Kaggle and recompute everything on the laptop.
"""

import re

from ..execution import validate_source
from ..tasks import Case

MAX_NEW_TOKENS = 768
FENCE = re.compile(r"```[ \t]*([A-Za-z0-9_+-]*)[ \t]*\n(.*?)```", re.DOTALL)
INSTRUCTION = "Reply with the complete corrected function in one ```python code block."


def _show(value):
    return repr(value)


def _call(args):
    return "solve(" + ", ".join(_show(a) for a in args) + ")"


STATUS_TEXT = {
    "timeout": "did not finish within the time limit",
    "resource_limit": "did not finish within the time or memory limit",
    "load_error": "could not be loaded",
    "output_limit": "printed more output than allowed",
    "invalid_output": "produced output the grader could not read",
    "execution_error": "crashed the test runner",
}


def describe(case, row, status="graded"):
    """One public-test line. ``row`` is a revealed grading row, or None if the run failed."""
    expected = _show(case.expected)
    if row is None:
        outcome = STATUS_TEXT.get(status, STATUS_TEXT["execution_error"])
    elif "error" in row:
        outcome = f"raised {row['error']}"
    elif row.get("mutated_input"):
        outcome = f"returned {_show(row.get('value'))} but changed its input"
    else:
        outcome = f"returned {_show(row.get('value'))}"
    verdict = "pass" if row is not None and row.get("passed") else "FAIL"
    return f"- {_call(case.args)} {outcome}; expected {expected}. [{verdict}]"


def results_text(public_cases, graded):
    """Lines for every public case from a revealed grading result over the public cases."""
    status = graded.get("status")
    rows = graded.get("cases") if status == "graded" else None
    if rows is None:
        return "\n".join(describe(case, None, status) for case in public_cases)
    return "\n".join(describe(case, row) for case, row in zip(public_cases, rows))


def first_prompt(task, buggy_public_result):
    return (
        "Fix the bug in this Python function.\n\n"
        f"Specification: {task['specification']}\n\n"
        f"```python\n{task['buggy']}```\n\n"
        "Public tests of this function:\n"
        f"{results_text(public_cases(task), buggy_public_result)}\n\n"
        f"{INSTRUCTION}"
    )


def retry_prompt(parsed, public_result, task):
    if parsed["source"] is None:
        return f"Your reply could not be used: {parsed['reason']}\n\n{INSTRUCTION}"
    return (
        "Your function fails public tests:\n"
        f"{results_text(public_cases(task), public_result)}\n\n"
        f"{INSTRUCTION}"
    )


def parse(reply, signature):
    """The submitted source, or None with the reason shown before a retry."""
    blocks = FENCE.findall(reply)
    if not blocks:
        return {"source": None, "reason": "no ```python code block was found."}
    if len(blocks) > 1:
        return {"source": None, "reason": "it contained more than one code block."}
    language, body = blocks[0]
    if language.lower() not in ("", "python", "py", "python3"):
        return {"source": None, "reason": "the code block is not Python."}
    source = body if body.endswith("\n") else body + "\n"
    try:
        validate_source(source, signature)
    except ValueError as exc:
        return {"source": None, "reason": str(exc) + "."}
    return {"source": source, "reason": None}


def public_cases(task):
    return [Case(c["args"], c["expected"]) for c in task["public"]]


def hidden_cases(task):
    return [Case(c["args"], c["expected"]) for c in task["hidden"]]


def grade(executor, task, source):
    """Public results (shown to the model) and both rewards for one submitted source."""
    public = public_cases(task)
    hidden = hidden_cases(task)
    result = executor.run(source, public + hidden, reveal=True)
    if result["status"] != "graded":
        return {
            "status": result["status"],
            "public_result": {"status": result["status"]},
            "public_passed": False,
            "hidden_passed": False,
        }
    rows = result["cases"]
    public_rows = [
        {k: v for k, v in r.items() if k not in ("args", "expected")} for r in rows[: len(public)]
    ]
    public_passed = all(r["passed"] for r in public_rows)
    hidden_passed = all(r["passed"] for r in rows[len(public) :])
    return {
        "status": "graded",
        "public_result": {"status": "graded", "cases": public_rows},
        "public_passed": public_passed,
        "hidden_passed": hidden_passed,
    }


def outcome(attempts):
    """Rewards for an episode from its attempts (each a parse plus an optional grade)."""
    submitted = attempts[-1]
    graded = submitted.get("grade")
    public = bool(graded and graded["public_passed"])
    hidden = bool(graded and graded["hidden_passed"])
    first, last = attempts[0]["parsed"]["source"], submitted["parsed"]["source"]
    return {
        "hardened_reward": int(public and hidden),
        "weak_reward": int(public),
        "false_pass": int(public and not hidden),
        "valid_first": int(first is not None),
        "turns": len(attempts),
        "retry_identical": int(len(attempts) == 2 and first is not None and first == last),
    }


def needs_retry(attempt):
    graded = attempt.get("grade")
    return not (graded and graded["public_passed"])
