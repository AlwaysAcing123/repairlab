"""Trusted, hand-authored pilot fixtures. Never copy this module into an agent workspace.

All fixtures are DEVELOPMENT material. No training or final test split exists yet.
References are authored code executed by the task builder, never model submissions.
"""

import hashlib
import json
import textwrap
from dataclasses import dataclass
from itertools import product

DATASET_VERSION = "backend-helpers-development-v1"


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def sha256(value):
    return hashlib.sha256(value.encode()).hexdigest()


def source(signature, body):
    return (
        f"def solve({signature}):\n" + textwrap.indent(textwrap.dedent(body).strip(), "    ") + "\n"
    )


@dataclass(frozen=True)
class Case:
    args: list
    expected: object

    def record(self):
        return {"args": self.args, "expected": self.expected}


@dataclass(frozen=True)
class Task:
    id: str
    family: str
    bug_category: str
    specification: str
    signature: str
    buggy: str
    reference: str
    alternative: str
    public_cases: tuple
    private_cases: tuple
    domain: str

    def public(self):
        return {
            "task_id": self.id,
            "specification": self.specification,
            "source": self.buggy,
            "public_tests": [case.record() for case in self.public_cases],
        }

    def fingerprint(self):
        return sha256(
            canonical(
                {
                    **self.public(),
                    "reference": self.reference,
                    "alternative": self.alternative,
                    "private_tests": [c.record() for c in self.private_cases],
                    "domain": self.domain,
                    "family": self.family,
                    "bug_category": self.bug_category,
                }
            )
        )

    def shortcut(self):
        """Deliberately overfit public inputs; the admission matrix must reject this."""
        names = [part.strip() for part in self.signature.split(",")]
        body = "\n".join(
            f"if [{', '.join(names)}] == {case.args!r}:\n    return {case.expected!r}"
            for case in self.public_cases
        )
        return source(self.signature, body + "\nreturn None")


def sequences(values, max_length=3):
    return [list(xs) for n in range(max_length + 1) for xs in product(values, repeat=n)]


def grid(*domains):
    return [list(args) for args in product(*domains)]


def task(
    ident, family, category, spec, signature, reference, buggy, alternative, public, inputs, domain
):
    reference_source = source(signature, reference)
    namespace = {}
    exec(reference_source, namespace)  # Only the fixed, trusted definitions below.
    function = namespace["solve"]
    public_cases = tuple(Case(args, expected) for args, expected in public)
    for case in public_cases:
        actual = function(*json.loads(canonical(case.args)))
        if canonical(actual) != canonical(case.expected):
            raise ValueError(f"Reference disagrees with hand-authored example: {ident}")
    seen = {canonical(c.args) for c in public_cases}
    private = []
    for args in inputs:
        key = canonical(args)
        if key in seen:
            continue
        seen.add(key)
        private.append(Case(args, function(*json.loads(key))))
    if not private:
        raise ValueError("A task must have independent private inputs")
    return Task(
        ident,
        family,
        category,
        spec,
        signature,
        source(signature, buggy),
        reference_source,
        source(signature, alternative),
        public_cases,
        tuple(private),
        domain,
    )


def pilot_tasks():
    small_lists = sequences([-2, -1, 0, 1, 2])
    flags = sequences([False, True], 4)
    tasks = [
        task(
            "quantity-range",
            "validation",
            "comparison",
            "For integer quantity and maximum >= 1, return whether 1 <= quantity <= maximum.",
            "quantity, maximum",
            "return 1 <= quantity <= maximum",
            "return 1 <= quantity < maximum",
            "return quantity > 0 and quantity < maximum + 1",
            [([3, 3], True), ([0, 3], False)],
            grid(range(-2, 8), range(1, 6)),
            "All quantity in [-2,7], maximum in [1,5].",
        ),
        task(
            "closed-window",
            "validation",
            "comparison",
            "For integers start <= end, return whether timestamp is in the CLOSED interval [start,end].",
            "timestamp, start, end",
            "return start <= timestamp <= end",
            "return start < timestamp <= end",
            "return not (timestamp < start or timestamp > end)",
            [([2, 2, 5], True), ([6, 2, 5], False)],
            [[t, a, b] for t, a, b in product(range(-2, 5), repeat=3) if a <= b],
            "All timestamp,start,end in [-2,4] with start <= end.",
        ),
        task(
            "payload-limit",
            "validation",
            "comparison",
            "Given nonnegative integer sizes, accept an empty or nonempty payload iff its size does not exceed the limit.",
            "size, limit",
            "return size <= limit",
            "return size < limit",
            "return not size > limit",
            [([4, 4], True), ([5, 4], False)],
            grid(range(7), range(7)),
            "All size,limit in [0,6].",
        ),
        task(
            "required-fields",
            "validation",
            "missing_condition",
            "Return True iff every element of the boolean list present is True. An empty list is valid.",
            "present",
            "return all(present)",
            "return any(present)",
            "return False not in present",
            [([[True, False]], False), ([[True]], True)],
            grid(flags),
            "All boolean lists of length 0 through 4.",
        ),
        task(
            "allowed-status",
            "validation",
            "comparison",
            "Return whether integer status equals any item in allowed (a list of integer status codes).",
            "status, allowed",
            "return status in allowed",
            "return status not in allowed",
            "return any(status == item for item in allowed)",
            [([200, [200, 201]], True), ([500, [200, 201]], False)],
            grid([200, 201, 400, 500], sequences([200, 201, 400], 3)),
            "All status in {200,201,400,500}, allowed lists over {200,201,400} of length 0..3.",
        ),
        task(
            "password-length",
            "validation",
            "comparison",
            "Return whether text has at least minimum characters. text is a string; minimum is a nonnegative integer.",
            "text, minimum",
            "return len(text) >= minimum",
            "return len(text) > minimum",
            "return not len(text) < minimum",
            [(["abc", 3], True), (["ab", 3], False)],
            grid(["", "a", "bb", "abc", "four", "éé", "🙂"], range(6)),
            "All text in {'','a','bb','abc','four','éé','🙂'}, minimum in [0,5].",
        ),
        task(
            "positive-batch",
            "validation",
            "missing_condition",
            "Accept a list of integer quantities iff it is nonempty and EVERY quantity is strictly positive.",
            "quantities",
            "return bool(quantities) and all(x > 0 for x in quantities)",
            "return all(x > 0 for x in quantities)",
            "return len(quantities) > 0 and not any(x <= 0 for x in quantities)",
            [([[]], False), ([[1, 2]], True), ([[1, 0]], False)],
            grid(small_lists),
            "All lists over {-2,-1,0,1,2} of length 0..3.",
        ),
        task(
            "optional-name",
            "validation",
            "missing_condition",
            "An optional name is valid if it is None, or a string containing at least one non-whitespace character.",
            "name",
            "return name is None or bool(name.strip())",
            "return bool(name) and bool(name.strip())",
            "if name is None:\n    return True\nreturn len(name.strip()) != 0",
            [([None], True), (["  "], False), (["Ada"], True)],
            grid([None, "", " ", "\t", "\n", "a", " a ", "é", "\ta\n"]),
            "All names in {None,'',' ','\\t','\\n','a',' a ','é','\\ta\\n'}.",
        ),
        task(
            "positive-total",
            "collections",
            "update_logic",
            "Return the sum of ONLY strictly positive integers in values; return 0 for an empty list.",
            "values",
            "return sum(x for x in values if x > 0)",
            "return sum(values)",
            "return sum(max(0, x) for x in values)",
            [([[-2, 1, 2]], 3), ([[]], 0)],
            grid(small_lists),
            "All integer lists over {-2,-1,0,1,2} of length 0..3.",
        ),
        task(
            "stable-unique",
            "collections",
            "update_logic",
            "Remove duplicate integers from values, preserving the order of their FIRST appearance.",
            "values",
            "return list(dict.fromkeys(values))",
            "return sorted(set(values))",
            "result = []\nfor value in values:\n    if value not in result:\n        result.append(value)\nreturn result",
            [([[2, 1, 2]], [2, 1]), ([[]], [])],
            grid(small_lists),
            "All integer lists over {-2,-1,0,1,2} of length 0..3.",
        ),
        task(
            "chunk-items",
            "collections",
            "off_by_one",
            "Split values into consecutive nonempty chunks of at most size elements. size >= 1. Keep any final partial chunk.",
            "values, size",
            "return [values[i:i + size] for i in range(0, len(values), size)]",
            "return [values[i:i + size] for i in range(0, len(values) - size + 1, size)]",
            "result = []\nremaining = list(values)\nwhile remaining:\n    result.append(remaining[:size])\n    remaining = remaining[size:]\nreturn result",
            [([[1, 2, 3], 2], [[1, 2], [3]]), ([[], 2], [])],
            grid(sequences([0, 1], 4), range(1, 5)),
            "All binary integer lists of length 0..4, size in [1,4].",
        ),
        task(
            "page-slice",
            "collections",
            "off_by_one",
            "Return values[offset:offset+limit]. offset and limit are nonnegative integers; preserve list order.",
            "values, offset, limit",
            "return values[offset:offset + limit]",
            "return values[offset:limit]",
            "return [value for i, value in enumerate(values) if offset <= i < offset + limit]",
            [([[0, 1, 2, 3], 2, 2], [2, 3]), ([[1], 0, 0], [])],
            grid(sequences([0, 1], 3), range(5), range(5)),
            "All binary integer lists of length 0..3, offset,limit in [0,4].",
        ),
        task(
            "first-match",
            "collections",
            "update_logic",
            "Return the index of the FIRST occurrence of target in values, or -1 if absent.",
            "values, target",
            "for i, value in enumerate(values):\n    if value == target:\n        return i\nreturn -1",
            "result = -1\nfor i, value in enumerate(values):\n    if value == target:\n        result = i\nreturn result",
            "return values.index(target) if target in values else -1",
            [([[1, 2, 1], 1], 0), ([[1, 2], 3], -1)],
            grid(sequences([0, 1, 2], 3), range(4)),
            "All lists over {0,1,2} of length 0..3, target in [0,3].",
        ),
        task(
            "drop-missing",
            "collections",
            "missing_condition",
            "Remove only None entries from values. Preserve zero, False, empty strings, and order.",
            "values",
            "return [x for x in values if x is not None]",
            "return [x for x in values if x]",
            "result = []\nfor value in values:\n    if value is None:\n        continue\n    result.append(value)\nreturn result",
            [([[None, 0, False, "", 1]], [0, False, "", 1])],
            grid(sequences([None, 0, False, "", 1], 3)),
            "All lists over {None,0,False,'',1} of length 0..3 (JSON types preserved).",
        ),
        task(
            "running-totals",
            "collections",
            "update_logic",
            "Return inclusive prefix sums: output[i] is sum(values[:i+1]). An empty input returns [].",
            "values",
            "result = []\ntotal = 0\nfor value in values:\n    total += value\n    result.append(total)\nreturn result",
            "result = []\ntotal = 0\nfor value in values:\n    total = value\n    result.append(total)\nreturn result",
            "return [sum(values[:i + 1]) for i in range(len(values))]",
            [([[1, 2, -1]], [1, 3, 2]), ([[]], [])],
            grid(small_lists),
            "All lists over {-2,-1,0,1,2} of length 0..3.",
        ),
        task(
            "bounded-queue",
            "collections",
            "off_by_one",
            "Append item to values and return the last capacity items. capacity >= 0; zero capacity returns []. Do not mutate values.",
            "values, item, capacity",
            "return (values + [item])[-capacity:] if capacity else []",
            "return (values + [item])[-capacity:]",
            "combined = values + [item]\nstart = max(0, len(combined) - capacity)\nreturn combined[start:]",
            [([[1, 2], 3, 0], []), ([[1, 2], 3, 2], [2, 3])],
            grid(sequences([0, 1], 3), [0, 1, 2], range(5)),
            "All binary integer lists length 0..3, item in [0,2], capacity in [0,4].",
        ),
        task(
            "retry-decision",
            "control_flow",
            "comparison",
            "Return True iff transient is True and attempts < maximum. attempts counts completed attempts; both counts are nonnegative integers.",
            "transient, attempts, maximum",
            "return transient and attempts < maximum",
            "return transient and attempts <= maximum",
            "if not transient:\n    return False\nreturn maximum > attempts",
            [([True, 3, 3], False), ([True, 2, 3], True), ([False, 0, 3], False)],
            grid([False, True], range(6), range(6)),
            "All booleans transient, attempts,maximum in [0,5].",
        ),
        task(
            "capped-backoff",
            "control_flow",
            "update_logic",
            "Return min(cap, base * 2**attempt). base,cap,attempt are nonnegative integers; attempt 0 uses base.",
            "base, attempt, cap",
            "return min(cap, base * 2 ** attempt)",
            "return max(cap, base * 2 ** attempt)",
            "delay = base\nfor _ in range(attempt):\n    delay *= 2\nreturn delay if delay < cap else cap",
            [([2, 2, 5], 5), ([2, 0, 5], 2)],
            grid(range(4), range(5), range(9)),
            "All base in [0,3], attempt in [0,4], cap in [0,8].",
        ),
        task(
            "cache-expiry",
            "control_flow",
            "comparison",
            "Return True iff the cache entry is expired: now >= created + ttl. All inputs are integers; ttl >= 0.",
            "now, created, ttl",
            "return now >= created + ttl",
            "return now > created + ttl",
            "return now - created >= ttl",
            [([5, 2, 3], True), ([4, 2, 3], False)],
            grid(range(-1, 7), range(-1, 5), range(5)),
            "All now in [-1,6], created in [-1,4], ttl in [0,4].",
        ),
        task(
            "next-scheduled-slot",
            "control_flow",
            "off_by_one",
            "Return the smallest multiple of interval that is >= now. now >= 0 and interval >= 1 are integers; a slot exactly at now is allowed.",
            "now, interval",
            "return ((now + interval - 1) // interval) * interval",
            "return (now // interval) * interval",
            "return now + (-now % interval)",
            [([5, 3], 6), ([6, 3], 6), ([0, 3], 0)],
            grid(range(13), range(1, 6)),
            "All now in [0,12], interval in [1,5].",
        ),
        task(
            "remaining-budget",
            "control_flow",
            "update_logic",
            "Return max(0, total-used). total and used are nonnegative integers.",
            "total, used",
            "return max(0, total - used)",
            "return total - used",
            "return total - min(total, used)",
            [([3, 5], 0), ([5, 3], 2)],
            grid(range(8), range(8)),
            "All total,used in [0,7].",
        ),
        task(
            "first-failure",
            "control_flow",
            "update_logic",
            "Given booleans outcomes, return how many initial True entries precede the first False, or len(outcomes) if none fail.",
            "outcomes",
            "count = 0\nfor ok in outcomes:\n    if not ok:\n        break\n    count += 1\nreturn count",
            "return sum(outcomes)",
            "return outcomes.index(False) if False in outcomes else len(outcomes)",
            [([[True, False, True]], 1), ([[]], 0)],
            grid(flags),
            "All boolean lists of length 0..4.",
        ),
        task(
            "consecutive-failures",
            "control_flow",
            "update_logic",
            "Return the number of consecutive False outcomes at the END of the boolean list. True means success and resets the count.",
            "outcomes",
            "count = 0\nfor ok in outcomes:\n    count = 0 if ok else count + 1\nreturn count",
            "count = 0\nfor ok in outcomes:\n    if not ok:\n        count += 1\nreturn count",
            "count = 0\nfor ok in reversed(outcomes):\n    if ok:\n        break\n    count += 1\nreturn count",
            [([[False, True, False]], 1), ([[True, True]], 0)],
            grid(flags),
            "All boolean lists of length 0..4.",
        ),
        task(
            "rate-window",
            "control_flow",
            "comparison",
            "Count integer event timestamps satisfying now-window < timestamp <= now. window >= 0. Exclude the lower boundary and future events.",
            "events, now, window",
            "return sum(now - window < t <= now for t in events)",
            "return sum(now - window <= t <= now for t in events)",
            "return len([t for t in events if t <= now and t > now - window])",
            [([[1, 2, 3, 4], 3, 2], 2), ([[], 3, 2], 0)],
            grid(sequences([-1, 0, 1, 2], 3), range(3), range(4)),
            "All lists over {-1,0,1,2} length 0..3, now in [0,2], window in [0,3].",
        ),
    ]
    if len({t.id for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs")
    return tasks


def manifest(tasks=None):
    tasks = pilot_tasks() if tasks is None else tasks
    rows = [
        {
            "task_id": t.id,
            "group_id": t.id,
            "split": "development",
            "family": t.family,
            "bug_category": t.bug_category,
            "public_cases": len(t.public_cases),
            "private_cases": len(t.private_cases),
            "checked_domain": t.domain,
            "task_sha256": t.fingerprint(),
        }
        for t in tasks
    ]
    body = {
        "dataset_version": DATASET_VERSION,
        "tasks": rows,
        "training_tasks": 0,
        "final_test_tasks": 0,
    }
    return {**body, "manifest_sha256": sha256(canonical(body))}
