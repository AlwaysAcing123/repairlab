"""Hand-written diagnoses for the 30 supervised training programs, used only as notes.

Each program has a cause and fix for its original bug and for its plausible wrong
repair. A note is the text before a demonstrated edit; the edit itself is always the
reference. The failing case it quotes is read from the real test output, never typed.
"""

import json

# program id: (bug cause, bug fix, wrong-repair cause, wrong-repair fix)
DIAGNOSES = {
    "username-charset": (
        "any() accepts the text when only one character is allowed",
        "require every character with all(), keeping the nonempty check",
        "all() is True for an empty string, so dropping bool(text) accepts empty text",
        "keep bool(text) and use all()",
    ),
    "version-at-least": (
        "minor is compared even when major is already higher",
        "compare (major, minor) tuples, so minor only matters when the majors are equal",
        "the or lets a large minor pass even when major is lower",
        "compare (major, minor) tuples instead",
    ),
    "nondecreasing-series": (
        "< rejects equal neighbors, but equal items are allowed",
        "use <= instead of <",
        "zip(values, values[2:]) compares items two apart and skips neighbors",
        "pair each item with the next one using values[1:], comparing with <=",
    ),
    "balanced-parens": (
        "equal counts ignore order, so ')(' passes",
        "track the depth, fail when it goes negative, and require depth == 0 at the end",
        "the final depth is never checked, so an unclosed '(' passes",
        "return depth == 0 after the loop",
    ),
    "hex-color-code": (
        "the length is never checked, so a short code passes",
        "require len(text) == 7 as well as the '#'",
        "the first character is never checked, so a string without '#' passes",
        "also require text[0] == '#'",
    ),
    "half-open-overlap": (
        "<= counts intervals that only touch at an end as overlapping, but ends are excluded",
        "use strict < in both comparisons",
        "b_start <= a_end still counts intervals that only touch as overlapping",
        "make both comparisons strict",
    ),
    "granted-permissions": (
        "any() passes when only one required permission is granted",
        "require every item with all()",
        "bool(required) rejects an empty required list, which should be satisfied",
        "return all(item in granted for item in required) without the extra check",
    ),
    "case-palindrome": (
        "the comparison is case-sensitive",
        "lowercase the text before comparing it with its reverse",
        "only the first and last characters are compared",
        "compare the whole lowered text with lowered[::-1]",
    ),
    "leap-year": (
        "century years are treated as leap years",
        "exclude years divisible by 100 unless they are divisible by 400",
        "every century is excluded, but years divisible by 400 are leap years",
        "allow year % 400 == 0 as an exception",
    ),
    "quorum-majority": (
        ">= counts exactly half as a majority",
        "use votes * 2 > members",
        "votes > members // 2 + 1 demands one vote too many",
        "use votes * 2 > members",
    ),
    "word-frequency": (
        "each count is reset to 1 instead of incremented",
        "add one to the previous count with counts.get(word, 0) + 1",
        "counts.get(word, 1) starts at 1 and then adds 1, one too many",
        "start from 0: counts.get(word, 0) + 1",
    ),
    "flatten-once": (
        "only the first item of each group is kept",
        "include every item of every group in order",
        "prepending each group reverses the group order",
        "append each group's items in order",
    ),
    "rotate-left": (
        "count is not wrapped, so a count longer than the list does nothing",
        "reduce count with count %= len(values), returning [] for an empty list",
        "values[:count] + values[count:] just rebuilds the original list",
        "put the tail first: values[count:] + values[:count] after count %= len(values)",
    ),
    "adjacent-differences": (
        "the range starts at 0, so values[0] - values[-1] adds a wrapped difference",
        "pair neighbors with zip(values, values[1:]) and return b - a",
        "a - b subtracts in the wrong order",
        "return b - a for each neighboring pair",
    ),
    "highest-score": (
        "best starts at 0, so all-negative scores return 0",
        "return max(scores), or None when scores is empty",
        "best still starts at 0, so all-negative scores return 0",
        "use max(scores) when scores is nonempty",
    ),
    "merge-sorted": (
        "items left over in either list after the loop are dropped",
        "append left[i:] and right[j:] to the result",
        "leftover items from right are still dropped",
        "append right[j:] as well as left[i:]",
    ),
    "nth-from-end": (
        "len(values) - n - 1 is one position too far from the end",
        "use values[-n] when 1 <= n <= len(values)",
        "n = 0 is accepted, and values[-0] is the first item",
        "also require n >= 1",
    ),
    "interleave-lists": (
        "zip stops at the shorter list and drops the rest of the longer one",
        "after the shared part, append the remaining items of both lists",
        "only the leftovers of left are appended, so extra items of right are lost",
        "append the remaining items of right too",
    ),
    "longest-equal-run": (
        "the last run is never compared with best, so a run at the end is ignored",
        "update best = max(best, current) after every item",
        "a new run restarts at 0 instead of 1, so every run is counted one short",
        "start each new run at 1 and update best after every item",
    ),
    "remove-at-index": (
        "pop() removes the item from the caller's list",
        "build a new list with values[:index] + values[index + 1:]",
        "values[:index] + values[index:] keeps the item",
        "skip it with values[index + 1:]",
    ),
    "fizz-label": (
        "the multiple-of-3 check runs first, so multiples of 15 return 'Fizz'",
        "check n % 15 first, then 3, then 5",
        "the 3 and 5 labels are swapped",
        "return 'Fizz' for multiples of 3 and 'Buzz' for multiples of 5",
    ),
    "shipping-tier": (
        "< excludes the boundary weights 1000 and 5000 from their tiers",
        "use <= for both limits",
        "weight < 5000 still sends exactly 5000 grams to the top tier",
        "use weight <= 5000",
    ),
    "collatz-steps": (
        "steps starts at 1, so every count is one too high",
        "start steps at 0",
        "the loop stops at 2, one step before reaching 1",
        "loop while n != 1",
    ),
    "digit-sum": (
        "the loop stops while one digit is left, so the leading digit is never added",
        "loop while n > 0",
        "n // 10 adds the remaining number, not the last digit",
        "add n % 10 each time",
    ),
    "euclid-gcd": (
        "a is overwritten before a % b is computed, so b becomes 0 at once",
        "update both together: a, b = b, a % b",
        "after the loop b is always 0; the answer is in a",
        "return a",
    ),
    "token-bucket": (
        "the cap is applied after spending, so tokens above capacity are spent first",
        "cap tokens + refill at capacity before checking and spending cost",
        "when cost is too high it returns 0 instead of leaving the tokens unchanged",
        "spend cost only if level >= cost, otherwise return level",
    ),
    "door-state": (
        "an extra transition lets 'open' open a locked door",
        "remove ('locked', 'open') from the moves",
        "the 'unlock' transition from locked to closed is missing",
        "include ('locked', 'unlock'): 'closed'",
    ),
    "clamped-elevator": (
        "the floor is never clamped at 0, so it goes negative",
        "clamp each move with max(0, min(top, floor + move))",
        "clamping once at the end ignores that each move stops at the limits",
        "clamp after every move",
    ),
    "progressive-fee": (
        "above 10 units every unit is charged 2, not just the extra ones",
        "charge 3 for the first 10 units plus 2 for each unit above 10",
        "2 * amount charges the first units twice",
        "charge 2 only for max(0, amount - 10)",
    ),
    "days-until-weekday": (
        "target - today goes negative when the target is earlier in the week",
        "wrap around with (target - today) % 7",
        "abs() counts backwards when the target is earlier in the week",
        "use (target - today) % 7",
    ),
}


def _value(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def failing_case(observation):
    """The first failing public case of a real test observation, in words."""
    for case in observation.get("cases", []):
        if case["passed"]:
            continue
        call = "solve(" + ", ".join(_value(a) for a in case["args"]) + ")"
        if "error" in case:
            return call + " raised " + case["error"]
        if case.get("mutated_input"):
            return call + " returned " + _value(case["value"]) + " but changed its input"
        return (
            call + " returned " + _value(case["value"]) + ", expected " + _value(case["expected"])
        )
    raise ValueError("No failing case to quote")


def note(program_id, observation, after_wrong):
    """The line before a demonstrated fix: what failed, why, and what to change."""
    bug_cause, bug_fix, wrong_cause, wrong_fix = DIAGNOSES[program_id]
    cause, fix = (wrong_cause, wrong_fix) if after_wrong else (bug_cause, bug_fix)
    text = "Cause: " + cause + ". Fix: " + fix + "."
    if "cases" in observation:
        text = "Failing case: " + failing_case(observation) + ". " + text
    return text
