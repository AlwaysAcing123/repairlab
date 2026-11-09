"""Program-separated splits, their admission controls, and mechanical leakage checks.

Development stays the 24 pilot tasks, which models have already been evaluated on.
Separation means no program, renamed copy, or mutation crosses a split. Shared broad
patterns such as "filter a list" are unavoidable and are not claimed to be absent.
"""

import builtins
import io
import json
import keyword
import signal
import tokenize
from collections import Counter
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path

from ..experiments import write_json
from ..native_repair import TASK_IDS
from ..tasks import canonical, pilot_tasks, sha256
from ..tool_warmup import programs as warmup_programs
from . import fingerprint
from .final_test import GROUPS, final_tasks
from .training_programs import training_programs

SPLIT_VERSION = "repair-splits-v1"
NEAR_SOURCE = 0.88
NEAR_SPEC = 0.6
NEAR_BEHAVIOR = 0.95
BEHAVIOR_SAMPLE = 120
MIN_COMPARABLE = 8
PRESERVED_NAMES = set(keyword.kwlist) | set(dir(builtins))
# Hand-reviewed October 30, 2026: every pair shares only a short token skeleton such as
# `return a op b` or a `set(values)` call, with different semantics and different fixes.
# A new near pair fails tests until someone reads it and adds it here.
REVIEWED_NEAR_PAIRS = frozenset(
    frozenset(pair)
    for pair in (
        ("allowed-status", "warm-add-pair"),
        ("allowed-status", "warm-multiply-pair"),
        ("closed-window", "quiet-hours"),
        ("consecutive-failures", "highest-score"),
        ("dev-distinct-count", "uniform-values"),
        ("dev-distinct-count", "warm-absolute"),
        ("dev-even", "leap-year"),
        ("dev-quotient", "warm-add-pair"),
        ("dev-quotient", "warm-multiply-pair"),
        ("drop-missing", "flatten-once"),
        ("email-shape", "required-fields"),
        ("email-shape", "warm-absolute"),
        ("email-shape", "warm-negate"),
        ("email-shape", "warm-reverse-text"),
        ("first-failure", "warm-absolute"),
        ("granted-permissions", "same-letter-counts"),
        ("payload-limit", "quorum-majority"),
        ("payload-limit", "warm-add-pair"),
        ("payload-limit", "warm-multiply-pair"),
        ("positive-total", "warm-absolute"),
        ("quiet-hours", "retry-decision"),
        ("remaining-budget", "warm-add-pair"),
        ("remaining-budget", "warm-multiply-pair"),
        ("required-fields", "warm-absolute"),
        ("stable-unique", "uniform-values"),
    )
)


@dataclass(frozen=True)
class Entry:
    id: str
    group: str
    split: str
    spec: str
    signature: str
    solutions: tuple
    faulty: tuple
    inputs: tuple


def remaining_development_tasks():
    return [t for t in pilot_tasks() if t.id not in TASK_IDS]


def reject_final(tasks):
    """Refuse any model-facing use of a sealed task, even under a different ID.

    Compares content directly: ``Task.fingerprint`` covers the ID, so a renamed copy
    would have a new fingerprint.
    """
    sealed = set()
    for task in final_tasks():
        sealed |= {task.id, task.specification, task.buggy, task.reference}
    for task in tasks:
        if sealed & {task.id, task.specification, task.buggy, task.reference}:
            raise ValueError("Sealed final-test task in a model-facing path: " + task.id)


def _task_entry(task, group, split, extra=()):
    return Entry(
        task.id,
        group,
        split,
        task.specification,
        task.signature,
        (task.reference, task.alternative),
        (task.buggy, *extra),
        tuple(c.args for c in task.public_cases + task.private_cases),
    )


def entries():
    rows = [_task_entry(p.task, p.group, "training", (p.wrong,)) for p in training_programs()]
    rows += [_task_entry(t, "dev-" + t.id, "development") for t in pilot_tasks()]
    rows += [_task_entry(t, GROUPS[t.id], "final_test") for t in final_tasks()]
    for program in warmup_programs():
        rows.append(
            Entry(
                program.id,
                program.group,
                "training" if program.split == "train" else "development",
                program.spec,
                program.signature,
                (program.reference,),
                (program.buggy,),
                tuple(c.args for c in program.cases),
            )
        )
    return rows


def normalized_tokens(text):
    """Source tokens with identifiers anonymized by first appearance."""
    names, result = {}, []
    for token in tokenize.generate_tokens(io.StringIO(text).readline):
        if token.type == tokenize.NAME:
            preserved = token.string in PRESERVED_NAMES
            result.append(
                token.string if preserved else names.setdefault(token.string, "v" + str(len(names)))
            )
        elif token.type not in (tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER):
            result.append(token.string)
    return result


def _words(text):
    return {w for w in "".join(c.lower() if c.isalnum() else " " for c in text).split()}


class _Timeout(Exception):
    pass


def _function(source):
    namespace = {}
    exec(source, namespace)  # Fixed authored references only; never model text.
    return namespace["solve"]


def _bounded_call(function, args):
    def expired(_signal, _frame):
        raise _Timeout

    previous = signal.signal(signal.SIGALRM, expired)
    try:
        # The timer is armed inside both blocks and the inner finally disarms it, so an
        # alarm landing before the call starts or after it ends is still caught below.
        try:
            signal.setitimer(signal.ITIMER_REAL, 0.02)
            return canonical(function(*json.loads(json.dumps(args))))
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    except Exception:
        return None
    finally:
        signal.signal(signal.SIGALRM, previous)


def _sample(inputs):
    step = max(1, len(inputs) // BEHAVIOR_SAMPLE)
    return list(inputs[::step])[:BEHAVIOR_SAMPLE]


def leakage_report(rows=None):
    """Compare every cross-split pair. Hard failures must be empty; near pairs need review.

    Only solution-to-solution identity is a hard failure: solutions are what a model
    could memorize. A trivial faulty line shared across splits is listed for review.
    """
    rows = entries() if rows is None else rows
    failures, near = [], []
    by_split = Counter(r.split for r in rows)
    ids, groups = {}, {}
    for row in rows:
        if row.id in ids:
            failures.append({"check": "duplicate_id", "id": row.id})
        ids[row.id] = row.split
        if groups.setdefault(row.group, row.split) != row.split:
            failures.append({"check": "group_crosses_splits", "group": row.group})
    solutions = {r.id: [normalized_tokens(s) for s in r.solutions] for r in rows}
    faulty = {r.id: [normalized_tokens(s) for s in r.faulty] for r in rows}
    functions = {r.id: _function(r.solutions[0]) for r in rows}
    samples = {r.id: _sample(r.inputs) for r in rows}
    compared = 0
    for index, left in enumerate(rows):
        for right in rows[index + 1 :]:
            if left.split == right.split:
                continue
            compared += 1
            pair = {"left": left.id, "right": right.id}
            if any(a == b for a in solutions[left.id] for b in solutions[right.id]):
                failures.append({"check": "identical_normalized_solution", **pair})
            best = 0.0
            for a in solutions[left.id] + faulty[left.id]:
                for b in solutions[right.id] + faulty[right.id]:
                    best = max(best, SequenceMatcher(None, a, b, autojunk=False).ratio())
            if best >= NEAR_SOURCE:
                near.append({"check": "near_source", "score": round(best, 3), **pair})
            words = _words(left.spec), _words(right.spec)
            jaccard = len(words[0] & words[1]) / len(words[0] | words[1])
            if jaccard >= NEAR_SPEC:
                near.append({"check": "near_specification", "score": round(jaccard, 3), **pair})
            if len(left.signature.split(",")) == len(right.signature.split(",")):
                inputs = {canonical(a): a for a in samples[left.id] + samples[right.id]}
                results = [
                    (_bounded_call(functions[left.id], a), _bounded_call(functions[right.id], a))
                    for a in inputs.values()
                ]
                comparable = [(a, b) for a, b in results if a is not None and b is not None]
                if len(comparable) >= MIN_COMPARABLE:
                    agreement = sum(a == b for a, b in comparable) / len(comparable)
                    varied = len({a for a, _ in comparable}) > 1
                    if varied and agreement >= NEAR_BEHAVIOR:
                        near.append(
                            {
                                "check": "near_behavior",
                                "score": round(agreement, 3),
                                "comparable_inputs": len(comparable),
                                **pair,
                            }
                        )
    return {
        "entries_by_split": dict(by_split),
        "cross_split_pairs_compared": compared,
        "hard_failures": failures,
        "near_pairs_for_review": near,
        "unreviewed_near_pairs": [
            n for n in near if frozenset((n["left"], n["right"])) not in REVIEWED_NEAR_PAIRS
        ],
        "thresholds": {
            "source_similarity": NEAR_SOURCE,
            "specification_jaccard": NEAR_SPEC,
            "behavior_agreement": NEAR_BEHAVIOR,
        },
    }


def split_manifest():
    def row(task, split, group):
        return {
            "task_id": task.id,
            "group_id": group,
            "split": split,
            "family": task.family,
            "bug_category": task.bug_category,
            "public_cases": len(task.public_cases),
            "private_cases": len(task.private_cases),
            "task_sha256": task.fingerprint(),
        }

    tasks = [row(p.task, "training", p.group) for p in training_programs()]
    tasks += [row(t, "development", "dev-" + t.id) for t in pilot_tasks()]
    tasks += [row(t, "final_test", GROUPS[t.id]) for t in final_tasks()]
    body = {
        "version": SPLIT_VERSION,
        "tasks": tasks,
        "legacy_warmup_programs": [
            {"program_id": p.id, "group_id": p.group, "split": p.split} for p in warmup_programs()
        ],
        "final_test_model_evaluations": 0,
    }
    return {**body, "manifest_sha256": sha256(canonical(body))}


def admit(executor, destination):
    """Author controls for the new splits. No model is involved."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    programs = {p.task.id: p for p in training_programs()}
    rows = []
    for split, tasks in (
        ("training", [p.task for p in programs.values()]),
        ("final_test", final_tasks()),
    ):
        for task in tasks:
            cases = task.public_cases + task.private_cases
            expected = [
                ("broken", task.buggy, False),
                ("reference", task.reference, True),
                ("alternative", task.alternative, True),
                ("public_shortcut", task.shortcut(), False),
            ]
            if split == "training":
                expected.append(("wrong_repair", programs[task.id].wrong, False))
            controls = {}
            for name, candidate, passes in expected:
                actual = executor.run(candidate, cases)
                controls[name] = {
                    "expected_pass": passes,
                    "actual": actual,
                    "matched": actual["status"] == "graded" and actual["passed"] is passes,
                }
            public_checks = {
                "shortcut_passes_public": executor.run(task.shortcut(), task.public_cases)[
                    "passed"
                ],
                "broken_fails_public": not executor.run(task.buggy, task.public_cases)["passed"],
            }
            if split == "training":
                public_checks["wrong_repair_fails_public"] = not executor.run(
                    programs[task.id].wrong, task.public_cases
                )["passed"]
            matched = all(c["matched"] for c in controls.values()) and all(public_checks.values())
            rows.append(
                {
                    "task_id": task.id,
                    "split": split,
                    "controls": controls,
                    **public_checks,
                    "all_matched": matched,
                }
            )
            print(f"{split}/{task.id}: {'PASS' if matched else 'FAIL'}", flush=True)
    leakage = leakage_report()
    report = {
        "kind": "authored_controls_not_model_results",
        "tasks": len(rows),
        "all_matched": all(r["all_matched"] for r in rows),
        "controls": sum(len(r["controls"]) for r in rows),
        "matched_controls": sum(c["matched"] for r in rows for c in r["controls"].values()),
        "model_episodes": 0,
        "curriculum_sha256": fingerprint(),
        "manifest_sha256": split_manifest()["manifest_sha256"],
        "leakage": leakage,
        "rows": rows,
    }
    write_json(destination / "split-manifest.json", split_manifest())
    write_json(destination / "report.json", report)
    return report
