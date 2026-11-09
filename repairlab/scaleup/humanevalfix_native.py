"""HumanEvalFix Python tasks converted mechanically to the native repair format.

Converted before any model ran, by a fixed rule: one function with at most ``typing``
imports, plain positional arguments, and tests made only of ``assert candidate(literal
arguments) == literal`` (or a bare or negated call, meaning True or False). The function
is renamed ``solve``, recursive calls included, and annotations are removed. Example
tests become public cases; the other hidden tests become private cases. Grading is the
host-side, type-sensitive output check of every other task here, which is stricter than
Python ``==``. Tasks failing a Docker control or matching a training program are listed
in ``DROPPED`` by the admission run and never scored. Reported only; decides nothing.
"""

import ast
import json
from dataclasses import replace
from pathlib import Path

from ..curriculum.final_test import final_tasks
from ..curriculum.splits import _task_entry, entries, leakage_report
from ..experiments import write_json
from ..humanevalfix import load
from ..rl.admission import rl_entries
from ..tasks import Case, Task, canonical, pilot_tasks
from ..tool_warmup import programs as warmup_programs
from . import fingerprint

SPLIT = "humanevalfix_native"
# From artifacts/humanevalfix-native-admission-v2, before any model run. Ten references
# fail the type-sensitive grader: they return tuples, which are not JSON, or change their
# input. Eight tasks match a training program, such as euclid-gcd or balanced-parens.
DROPPED = {
    "humanevalfix-13": "matches a training program: near_behavior",
    "humanevalfix-14": "matches a training program: near_source",
    "humanevalfix-20": "failed a Docker control",
    "humanevalfix-23": "matches a training program: near_source",
    "humanevalfix-35": "matches a training program: near_source",
    "humanevalfix-45": "matches a training program: near_source",
    "humanevalfix-54": "matches a training program: near_source",
    "humanevalfix-61": "matches a training program: near_behavior",
    "humanevalfix-70": "failed a Docker control",
    "humanevalfix-87": "failed a Docker control",
    "humanevalfix-107": "failed a Docker control",
    "humanevalfix-112": "failed a Docker control",
    "humanevalfix-120": "failed a Docker control",
    "humanevalfix-126": "matches a training program: near_behavior",
    "humanevalfix-136": "failed a Docker control",
    "humanevalfix-148": "failed a Docker control",
    "humanevalfix-149": "failed a Docker control",
    "humanevalfix-155": "failed a Docker control",
}


class Unconvertible(ValueError):
    pass


def _literal(node):
    try:
        value = ast.literal_eval(node)
    except (ValueError, TypeError, SyntaxError, MemoryError, RecursionError) as exc:
        raise Unconvertible("non-literal test value") from exc

    def check(item):
        if isinstance(item, bool) or item is None or isinstance(item, (int, str)):
            return
        if isinstance(item, float):
            if item != item or item in (float("inf"), float("-inf")):
                raise Unconvertible("non-finite float")
            return
        if isinstance(item, (list, tuple)):
            for element in item:
                check(element)
            return
        if isinstance(item, dict):
            for key, element in item.items():
                if not isinstance(key, str):
                    raise Unconvertible("dict key is not a string")
                check(element)
            return
        raise Unconvertible("value is not JSON: " + type(item).__name__)

    check(value)
    return json.loads(canonical(value))


def test_cases(text):
    """Every assertion in a HumanEvalFix check function, as (arguments, expected)."""
    if not text.strip():
        return []
    try:
        tree = ast.parse(text)
    except SyntaxError as exc:
        raise Unconvertible("test does not parse") from exc
    checks = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "check"]
    if len(checks) != 1 or len(checks[0].args.args) != 1:
        raise Unconvertible("no single check(candidate) function")
    candidate = checks[0].args.args[0].arg
    cases = []
    for statement in checks[0].body:
        if isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Constant):
            continue
        if not isinstance(statement, ast.Assert):
            raise Unconvertible("statement other than assert")
        test = statement.test
        if isinstance(test, ast.Constant) and test.value is True:
            continue
        if isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not):
            call, expected = test.operand, False
        elif isinstance(test, ast.Compare):
            if len(test.ops) != 1 or not isinstance(test.ops[0], (ast.Eq, ast.Is)):
                raise Unconvertible("comparison other than == or is")
            left, right = test.left, test.comparators[0]
            if isinstance(right, ast.Call):
                left, right = right, left
            call, expected = left, _literal(right)
        else:
            call, expected = test, True
        if (
            not isinstance(call, ast.Call)
            or not isinstance(call.func, ast.Name)
            or call.func.id != candidate
            or call.keywords
        ):
            raise Unconvertible("assertion is not a direct candidate call")
        cases.append(([_literal(a) for a in call.args], expected))
    return cases


class _Rename(ast.NodeTransformer):
    def __init__(self, name):
        self.name = name

    def visit_Name(self, node):
        if node.id == self.name:
            return ast.copy_location(ast.Name(id="solve", ctx=node.ctx), node)
        return node


def _source(declaration, body, entry):
    try:
        module = ast.parse(declaration + body)
    except SyntaxError as exc:
        raise Unconvertible("source does not parse") from exc
    functions = [n for n in module.body if isinstance(n, ast.FunctionDef)]
    if len(functions) != 1 or functions[0].name != entry:
        raise Unconvertible("declaration is not one function")
    imported = set()
    for node in module.body:
        if node is functions[0]:
            continue
        if not (isinstance(node, ast.ImportFrom) and node.module == "typing"):
            raise Unconvertible("module statement other than a typing import")
        imported |= {alias.asname or alias.name for alias in node.names}
    function = functions[0]
    arguments = function.args
    if (
        arguments.posonlyargs
        or arguments.vararg
        or arguments.kwonlyargs
        or arguments.kwarg
        or arguments.defaults
    ):
        raise Unconvertible("arguments other than plain positional ones")
    for argument in arguments.args:
        argument.annotation = None
    function.returns = None
    function.name = "solve"
    function = _Rename(entry).visit(function)
    for node in ast.walk(function):
        if isinstance(node, ast.AnnAssign) or (isinstance(node, ast.Name) and node.id in imported):
            raise Unconvertible("body depends on annotations or typing names")
    source = ast.unparse(ast.fix_missing_locations(function)) + "\n"
    return source, ", ".join(a.arg for a in arguments.args)


def convert(row):
    """One HumanEvalFix row as a native task; raises Unconvertible with the reason."""
    entry = row["entry_point"]
    reference, signature = _source(row["declaration"], row["canonical_solution"], entry)
    buggy, buggy_signature = _source(row["declaration"], row["buggy_solution"], entry)
    if signature != buggy_signature or reference == buggy:
        raise Unconvertible("buggy and reference do not differ only in the body")
    public, seen = [], {}
    for args, expected in test_cases(row["example_test"]):
        key = canonical(args)
        if seen.setdefault(key, canonical(expected)) != canonical(expected):
            raise Unconvertible("conflicting expected values")
        if key not in [canonical(c.args) for c in public]:
            public.append(Case(args, expected))
    private = []
    for args, expected in test_cases(row["test"]):
        key = canonical(args)
        if seen.setdefault(key, canonical(expected)) != canonical(expected):
            raise Unconvertible("conflicting expected values")
        if key in [canonical(c.args) for c in public + private]:
            continue
        private.append(Case(args, expected))
    if not public or not private:
        raise Unconvertible("needs at least one public and one private case")
    specification = row["docstring"].strip()
    if entry in specification:
        specification += f"\n(The function is named solve here; the examples call it {entry}.)"
    return Task(
        "humanevalfix-" + row["task_id"].split("/")[1],
        "humanevalfix",
        row["bug_type"],
        specification,
        signature,
        buggy,
        reference,
        reference,
        tuple(public),
        tuple(private),
        "HumanEvalFix example and hidden test inputs.",
    )


def conversions():
    """Every row: (task id, Task or None, reason it could not be converted)."""
    rows, _ = load()
    result = []
    for row in rows:
        ident = "humanevalfix-" + row["task_id"].split("/")[1]
        try:
            result.append((ident, convert(row), None))
        except Unconvertible as exc:
            result.append((ident, None, str(exc)))
    return result


def converted_tasks():
    return [task for _, task, _ in conversions() if task is not None]


def native_tasks():
    return [t for t in converted_tasks() if t.id not in DROPPED]


def native_entries(tasks=None):
    tasks = converted_tasks() if tasks is None else tasks
    return [_task_entry(t, "hef-" + t.id, SPLIT) for t in tasks]


def training_leakage_report(tasks=None):
    """Converted tasks against every program that entered training, by the usual checks."""
    evaluation = {t.id for t in pilot_tasks() + final_tasks()}
    evaluation |= {p.id for p in warmup_programs() if p.split != "train"}
    training = [
        replace(e, split="training") for e in entries() + rl_entries() if e.id not in evaluation
    ]
    report = leakage_report(training + native_entries(tasks))
    report["training_programs_compared"] = len(training)
    return report


def admit(executor, destination):
    """Docker controls and training separation for every converted task. No model runs.

    The reference must pass every case under normal grading. The buggy version must fail,
    and a timeout or resource limit counts as failing, as in the original external
    controls. The public-answer shortcut must pass the public cases and fail overall.
    """
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    rows, dropped = [], {}
    for ident, task, reason in conversions():
        if task is None:
            rows.append({"task_id": ident, "converted": False, "reason": reason})
            continue
        cases = task.public_cases + task.private_cases
        controls = {}
        for name, candidate, passes in (
            ("broken", task.buggy, False),
            ("reference", task.reference, True),
            ("public_shortcut", task.shortcut(), False),
        ):
            actual = executor.run(candidate, cases)
            graded = actual["status"] == "graded" or name == "broken"
            controls[name] = {
                "expected_pass": passes,
                "actual": actual,
                "matched": graded and actual["passed"] is passes,
            }
        shortcut_public = executor.run(task.shortcut(), task.public_cases)["passed"]
        matched = all(c["matched"] for c in controls.values()) and shortcut_public
        if not matched:
            dropped[task.id] = "failed a Docker control"
        rows.append(
            {
                "task_id": task.id,
                "converted": True,
                "controls": controls,
                "shortcut_passes_public": shortcut_public,
                "all_matched": matched,
            }
        )
        print(f"{SPLIT}/{task.id}: {'PASS' if matched else 'DROP'}", flush=True)
    kept = [t for t in converted_tasks() if t.id not in dropped]
    leakage = training_leakage_report(kept)
    native = {t.id for t in kept}
    for item in leakage["hard_failures"] + leakage["near_pairs_for_review"]:
        for side in ("left", "right", "id"):
            if item.get(side) in native:
                dropped.setdefault(item[side], "matches a training program: " + item["check"])
    report = {
        "kind": "external_conversion_controls_not_model_results",
        "rows_in_dataset": len(rows),
        "converted": sum(r["converted"] for r in rows),
        "dropped": dropped,
        "scored_tasks": sum(r["converted"] for r in rows) - len(dropped),
        "model_episodes": 0,
        "scaleup_sha256": fingerprint(),
        "leakage": leakage,
        "rows": rows,
    }
    write_json(destination / "report.json", report)
    return report
