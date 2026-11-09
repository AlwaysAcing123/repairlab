"""Container-only candidate worker with two explicitly different evaluation modes.

Custom mode receives inputs without expected answers; the host compares outputs.
External mode executes the original assertion suite with candidate code. This file
is the only host file mounted inside the container and has no project imports.
"""

import contextlib
import copy
import json
import resource
import sys


class DiscardOutput:
    def write(self, value):
        return len(value)

    def flush(self):
        pass


def main():
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (1048576, 1048576))
    request = json.load(sys.stdin)
    if request.get("mode") == "external_assert_suite":
        # The external adapter uses the authors' assert tests. It has a different
        # trust boundary from the custom tasks' host-owned expected-output oracle.
        namespace = {"__name__": "candidate"}
        sink = DiscardOutput()
        passed, error = False, None
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            try:
                exec(compile(request["source"], "/candidate.py", "exec"), namespace)
                exec(compile(request["tests"], "/tests.py", "exec"), namespace)
                passed = True
            except BaseException as exc:
                error = type(exc).__name__
        sys.__stdout__.write(json.dumps({"suite_passed": passed, "error": error}))
        return
    results = []
    namespace = {"__name__": "candidate"}
    sink = DiscardOutput()
    with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
        try:
            exec(compile(request["source"], "/candidate.py", "exec"), namespace)
            solve = namespace["solve"]
        except BaseException as exc:
            sys.__stdout__.write(json.dumps({"load_error": type(exc).__name__}))
            return
        for args in request["inputs"]:
            original = copy.deepcopy(args)
            try:
                value = solve(*args)
                # Reject tuples, non-string mapping keys, NaN, and other values
                # that JSON would silently change. Output types are part of the contract.
                validate_json(value)
                results.append({"value": value, "mutated_input": not equal(args, original)})
            except BaseException as exc:
                results.append({"error": type(exc).__name__})
    sys.__stdout__.write(json.dumps({"results": results}, allow_nan=False))


def validate_json(value):
    if value is None or type(value) in (str, bool, int):
        return
    if type(value) is float:
        json.dumps(value, allow_nan=False)
        return
    if type(value) is list:
        for item in value:
            validate_json(item)
        return
    if type(value) is dict and all(type(key) is str for key in value):
        for item in value.values():
            validate_json(item)
        return
    raise TypeError("Return a JSON value without implicit type conversions")


def equal(left, right):
    return json.dumps(left, sort_keys=True, allow_nan=False) == json.dumps(
        right, sort_keys=True, allow_nan=False
    )


if __name__ == "__main__":
    main()
