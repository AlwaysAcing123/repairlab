"""Independent synthetic program groups for a bounded native-tool supervised pilot.

These teach next-action/tool formatting and a demonstrated workflow. They are not
the 24 backend-helper evaluation tasks or the old diagnostic probes.
"""

import json
from dataclasses import dataclass

from .native_tools import SYSTEM, assistant_call, probes
from .tasks import Case, canonical, sha256, source

PROTOCOL = "native-tool-supervised-warmup-v1"
CONFIG = {
    "optimizer_steps": 60,
    "micro_batch_size": 2,
    "gradient_accumulation": 4,
    "learning_rate": 0.0001,
    "max_sequence_tokens": 2048,
    "seed": 7,
    "lora_rank": 8,
    "lora_alpha": 16,
    "lora_dropout": 0.0,
    "target_modules": ["q_proj", "v_proj"],
    "max_eval_output_tokens": 256,
    "max_eval_context_tokens": 4096,
}


@dataclass(frozen=True)
class Program:
    id: str
    group: str
    split: str
    spec: str
    signature: str
    reference: str
    buggy: str
    cases: tuple
    public_cases: tuple

    def public(self):
        return {
            "task_id": self.id,
            "specification": self.spec,
            "source": self.buggy,
            "public_tests": [c.record() for c in self.public_cases],
        }


def program(ident, group, split, spec, signature, reference, buggy, examples, inputs):
    reference = source(signature, reference)
    namespace = {}
    exec(reference, namespace)  # Fixed authored references only; never model text.
    cases = tuple(Case(args, namespace["solve"](*json.loads(canonical(args)))) for args in inputs)
    public = tuple(Case(args, expected) for args, expected in examples)
    for case in public:
        assert canonical(namespace["solve"](*case.args)) == canonical(case.expected)
    return Program(
        ident, group, split, spec, signature, reference, source(signature, buggy), cases, public
    )


def programs():
    integers = [[x] for x in range(-5, 6)]
    pairs = [[a, b] for a in range(-3, 4) for b in range(-2, 3)]
    strings = [[s] for s in ("", "a", "ab", "cat", "hello", "éa")]
    return [
        program(
            "warm-square",
            "integer-powers",
            "train",
            "Return the square of integer number.",
            "number",
            "return number ** 2",
            "return number + number",
            [([3], 9), ([-2], 4)],
            integers,
        ),
        program(
            "warm-cube",
            "integer-powers",
            "train",
            "Return the cube of integer number.",
            "number",
            "return number ** 3",
            "return number ** 2",
            [([3], 27), ([-2], -8)],
            integers,
        ),
        program(
            "warm-absolute",
            "absolute-value",
            "train",
            "Return the absolute value of integer number.",
            "number",
            "return abs(number)",
            "return number",
            [([-3], 3), ([2], 2)],
            integers,
        ),
        program(
            "warm-negate",
            "integer-negation",
            "train",
            "Return the additive inverse of integer amount.",
            "amount",
            "return -amount",
            "return amount",
            [([4], -4), ([-3], 3)],
            integers,
        ),
        program(
            "warm-add-pair",
            "pair-addition",
            "train",
            "Return left plus right for two integers.",
            "left, right",
            "return left + right",
            "return left - right",
            [([2, 3], 5), ([1, -2], -1)],
            pairs,
        ),
        program(
            "warm-multiply-pair",
            "pair-multiplication",
            "train",
            "Return the product of two integers left and right.",
            "left, right",
            "return left * right",
            "return left + right",
            [([2, 3], 6), ([1, -2], -2)],
            pairs,
        ),
        program(
            "warm-reverse-text",
            "string-reversal",
            "train",
            "Return text with its characters in reverse order.",
            "text",
            "return text[::-1]",
            "return text",
            [(["cat"], "tac"), ([""], "")],
            strings,
        ),
        program(
            "warm-prefix-character",
            "string-prefix",
            "train",
            "Return the first character of text, or the empty string when text is empty.",
            "text",
            "return text[:1]",
            "return text[1:2]",
            [(["cat"], "c"), ([""], "")],
            strings,
        ),
        program(
            "dev-even",
            "integer-parity",
            "development",
            "Return True exactly when integer value is even.",
            "value",
            "return value % 2 == 0",
            "return value % 2 == 1",
            [([4], True), ([3], False)],
            integers,
        ),
        program(
            "dev-quotient",
            "integer-floor-division",
            "development",
            "Return numerator // denominator. Both are integers and denominator is positive.",
            "numerator, denominator",
            "return numerator // denominator",
            "return numerator % denominator",
            [([7, 3], 2), ([-1, 2], -1)],
            [[a, b] for a in range(-5, 6) for b in range(1, 5)],
        ),
        program(
            "dev-join-words",
            "string-joining",
            "development",
            "Join the strings in words with one hyphen between neighbors. An empty list returns an empty string.",
            "words",
            "return '-'.join(words)",
            "return ''.join(words)",
            [([["red", "blue"]], "red-blue"), ([[]], "")],
            [[w] for w in ([], ["a"], ["a", "b"], ["", "x"], ["a", "b", "c"])],
        ),
        program(
            "dev-distinct-count",
            "distinct-cardinality",
            "development",
            "Return the number of distinct integers appearing in values.",
            "values",
            "return len(set(values))",
            "return len(values)",
            [([[1, 1, 2]], 2), ([[]], 0)],
            [[v] for v in ([], [0], [1, 1], [1, 2, 1], [-1, 0, -1], [1, 2, 3])],
        ),
    ]


def demonstration_data():
    samples = []
    for task in programs():
        if task.split != "train":
            continue
        history = [
            {"role": "system", "content": SYSTEM},
            {
                "role": "user",
                "content": "Repair this training program using the tools. Inspect, test, edit, retest, then submit.\n"
                + canonical(task.public()),
            },
        ]
        bad_namespace, good_namespace = {}, {}
        exec(task.buggy, bad_namespace)  # Fixed authored program, not an agent output.
        exec(task.reference, good_namespace)

        def test_observation(function):
            rows = [
                {
                    "args": c.args,
                    "expected": c.expected,
                    "value": function(*json.loads(canonical(c.args))),
                    "passed": canonical(function(*json.loads(canonical(c.args))))
                    == canonical(c.expected),
                }
                for c in task.public_cases
            ]
            return {
                "passed": all(r["passed"] for r in rows),
                "passed_cases": sum(r["passed"] for r in rows),
                "total_cases": len(rows),
                "cases": rows,
            }

        trajectory = [
            ("read", {}, {"source": task.buggy}),
            ("test", {}, test_observation(bad_namespace["solve"])),
            ("edit", {"source": task.reference}, {"edited": True}),
            ("test", {}, test_observation(good_namespace["solve"])),
            ("submit", {}, {"submitted": True}),
        ]
        for index, (name, arguments, observation) in enumerate(trajectory):
            assistant = assistant_call(name, arguments)
            samples.append(
                {
                    "sample_id": task.id + "-trajectory-" + str(index),
                    "split": "train",
                    "program_id": task.id,
                    "group_id": task.group,
                    "prompt": json.loads(canonical(history)),
                    "completion": [assistant],
                }
            )
            history.extend(
                [assistant, {"role": "tool", "name": name, "content": canonical(observation)}]
            )
        for name, instruction in (
            ("read", "Fetch the current source for this training program using read."),
            ("test", "Execute the public checks for this training program using test."),
            ("submit", "Conclude this training attempt by calling submit."),
        ):
            samples.append(
                {
                    "sample_id": task.id + "-direct-" + name,
                    "split": "train",
                    "program_id": task.id,
                    "group_id": task.group,
                    "prompt": [
                        {"role": "system", "content": SYSTEM},
                        {"role": "user", "content": instruction + "\n" + canonical(task.public())},
                    ],
                    "completion": [assistant_call(name, {})],
                }
            )
    assert len(samples) == 64
    return samples


def development_probes():
    result = []
    for task in programs():
        if task.split != "development":
            continue
        base = [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": canonical(task.public())},
        ]
        for name, prompt, history in (
            ("read", "Inspect the implementation by calling read before changing it.", []),
            (
                "test",
                "Check the implementation now by calling test.",
                [
                    assistant_call("read", {}),
                    {"role": "tool", "name": "read", "content": canonical({"source": task.buggy})},
                ],
            ),
            (
                "edit",
                "Use edit to apply exactly the following replacement:\n" + task.reference,
                [
                    assistant_call("test", {}),
                    {"role": "tool", "name": "test", "content": canonical({"passed": False})},
                ],
            ),
            (
                "submit",
                "The current candidate is ready; submit it for final checking.",
                [
                    assistant_call("test", {}),
                    {"role": "tool", "name": "test", "content": canonical({"passed": True})},
                ],
            ),
        ):
            result.append(
                {
                    "probe_id": task.id + "-" + name,
                    "program_id": task.id,
                    "group_id": task.group,
                    "messages": base + history + [{"role": "user", "content": prompt}],
                    "expected_tool": name,
                    "expected_source": task.reference if name == "edit" else None,
                }
            )
    return result


def bundle():
    train = demonstration_data()
    development = development_probes()
    historical = probes()
    assert {r["group_id"] for r in train}.isdisjoint({r["group_id"] for r in development})
    assert {r["program_id"] for r in train}.isdisjoint({r["program_id"] for r in development})
    assert {canonical(r["prompt"]) for r in train}.isdisjoint(
        {canonical(r["messages"]) for r in development + historical}
    )
    body = {
        "protocol": PROTOCOL,
        "config": CONFIG,
        "train": train,
        "development": development,
        "historical_diagnostic": historical,
        "provenance": "New synthetic authored teacher workflows; no external-model teacher outputs or 24-task benchmark repairs used.",
    }
    return {**body, "bundle_sha256": sha256(canonical(body))}
