"""Native function-call contract and synthetic diagnostic probes, not repair scores."""

from .execution import validate_source
from .tasks import canonical

PROTOCOL = "qwen-native-tools-diagnostic-v1"
SYSTEM = """Use the supplied tools to carry out the user's request.
Call exactly one tool per assistant turn. Use the native function-call format, not
Markdown or a plain-text JSON action. When editing Python, provide the complete
function source in the source argument. Preserve its signature and use actual
newlines in source. Do not use annotations, decorators, or default arguments.
Only use information in the conversation. Do not claim a test or action happened
until a tool result is provided. This session is a tool-interface diagnostic.
"""


def tools():
    definitions = []
    for name, description in (
        ("read", "Read the current Python function source."),
        ("test", "Execute the permitted public tests and return their results."),
        ("edit", "Replace the current function with the complete Python source supplied."),
        ("submit", "Submit the current function for independent final evaluation."),
    ):
        parameters = {
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        }
        if name == "edit":
            parameters["properties"] = {
                "source": {"type": "string", "description": "Complete function source."}
            }
            parameters["required"] = ["source"]
        definitions.append(
            {
                "type": "function",
                "function": {"name": name, "description": description, "parameters": parameters},
            }
        )
    return definitions


def assistant_call(name, arguments):
    return {
        "role": "assistant",
        "content": "",
        "tool_calls": [{"type": "function", "function": {"name": name, "arguments": arguments}}],
    }


def parsed_action(assistant):
    """Validate an already natively parsed response; never infer missing calls."""
    if not isinstance(assistant, dict):
        raise ValueError("Native parser did not return an assistant object")
    calls = assistant.get("tool_calls")
    if not isinstance(calls, list) or len(calls) != 1:
        raise ValueError("Exactly one native tool call is required")
    function = calls[0].get("function") if isinstance(calls[0], dict) else None
    if not isinstance(function, dict) or not isinstance(function.get("arguments"), dict):
        raise ValueError("Native arguments must be an object")
    name, arguments = function.get("name"), function["arguments"]
    if name not in {"read", "test", "edit", "submit"}:
        raise ValueError("Unknown native tool")
    if set(arguments) != ({"source"} if name == "edit" else set()):
        raise ValueError("Unexpected native arguments")
    if name == "edit":
        validate_source(arguments["source"])
    return {"tool": name, **arguments}


def probes():
    result = []

    def add(ident, prompt, expected, source=None, history=None):
        result.append(
            {
                "probe_id": ident,
                "messages": [
                    {"role": "system", "content": SYSTEM},
                    *(history or []),
                    {"role": "user", "content": prompt},
                ],
                "expected_tool": expected,
                "expected_source": source,
            }
        )

    add("read-direct", "Read the current function using read.", "read")
    add(
        "read-before-debugging",
        "I want to inspect the current implementation. Use read first.",
        "read",
    )
    add("test-direct", "Run the public tests now using test.", "test")
    add(
        "test-after-failure-report",
        "An example was reported to fail. Execute test to get the actual public results.",
        "test",
    )
    for ident, signature, body in (
        ("edit-add-one", "x", "return x + 1"),
        ("edit-double", "value", "return value * 2"),
    ):
        source = f"def solve({signature}):\n    {body}\n"
        add(
            ident,
            "Use edit to replace the current function with exactly this source:\n" + source,
            "edit",
            source,
        )
    add("submit-direct", "Submit the current function now using submit.", "submit")
    add(
        "submit-after-user-confirmation",
        "The current candidate is ready. Call submit to end this attempt.",
        "submit",
    )
    read_history = [
        {"role": "user", "content": "Read the current function first."},
        assistant_call("read", {}),
        {
            "role": "tool",
            "name": "read",
            "content": canonical({"source": "def solve(x):\n    return x - 1\n"}),
        },
    ]
    add(
        "after-read-test",
        "Now run test to inspect the public results.",
        "test",
        history=read_history,
    )
    test_history = [
        {"role": "user", "content": "The function must return x+1. Run the public tests."},
        assistant_call("test", {}),
        {
            "role": "tool",
            "name": "test",
            "content": canonical(
                {
                    "passed": False,
                    "source": "def solve(x):\n    return x - 1\n",
                    "cases": [{"args": [2], "expected": 3, "value": 1, "passed": False}],
                }
            ),
        },
    ]
    add(
        "after-failing-test-edit",
        "Use edit to fix the function to return x+1.",
        "edit",
        "def solve(x):\n    return x + 1\n",
        history=test_history,
    )
    edited = "def solve(value):\n    return value * 2\n"
    edit_history = [
        {"role": "user", "content": "Change the function to double value."},
        assistant_call("edit", {"source": edited}),
        {"role": "tool", "name": "edit", "content": canonical({"edited": True})},
    ]
    add(
        "after-edit-test",
        "The edit has been applied. Use test to check it.",
        "test",
        history=edit_history,
    )
    success_history = [
        {"role": "user", "content": "Run the tests for the current candidate."},
        assistant_call("test", {}),
        {
            "role": "tool",
            "name": "test",
            "content": canonical({"passed": True, "passed_cases": 2, "total_cases": 2}),
        },
    ]
    add(
        "after-passing-test-submit",
        "Submit the current candidate now using submit.",
        "submit",
        history=success_history,
    )
    return result


def roundtrip_examples():
    return [
        assistant_call(
            name, {"source": "def solve(x):\n    return x + 1\n"} if name == "edit" else {}
        )
        for name in ("read", "test", "edit", "submit")
    ]
