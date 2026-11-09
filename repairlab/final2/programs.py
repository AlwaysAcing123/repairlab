"""SEALED second final-test tasks. No model has seen these and none may until choices freeze.

Never export this module's content to a notebook, training bundle, or prompt.
``admission.reject_final2`` enforces that for model-facing paths; tests pin it. Sixteen
tasks each of validation, collections, and control flow, and twelve of each bug category.
"""

from itertools import combinations_with_replacement, product

from ..tasks import grid, sequences, task


def _strings(alphabet, max_length):
    return ["".join(xs) for xs in sequences(list(alphabet), max_length)]


def _valid_program(program):
    depth = 0
    for op in program:
        if op == "add":
            if depth < 2:
                return False
            depth -= 1
        elif op == "dup":
            if depth < 1:
                return False
            depth += 1
        else:
            depth += 1
    return True


def final2_tasks():
    intervals = [[s, e] for s in range(4) for e in range(s, 5)]
    interval_lists = [
        [list(i) for i in xs]
        for n in range(4)
        for xs in product(intervals, repeat=n)
        if all(a[0] <= b[0] for a, b in zip(xs, xs[1:]))
    ]
    sorted_lists = [list(xs) for n in range(5) for xs in combinations_with_replacement(range(4), n)]
    mappings = [
        {k: v for k, v in zip("abcd", choice) if v} for choice in product(["", "x", "y"], repeat=4)
    ]
    pairs = [["a", 1], ["a", 2], ["b", 1], ["b", 3]]
    lines = ["x", "  x", "    x", "        x"]
    return [
        # Validation
        task(
            "semver-parts",
            "validation",
            "missing_condition",
            "Return True iff text is three groups of one or more ASCII digits separated by "
            "single dots, like '1.20.3'.",
            "text",
            'parts = text.split(".")\n'
            'return len(parts) == 3 and all(p and all(c in "0123456789" for c in p) for p in parts)',
            'parts = text.split(".")\n'
            'return len(parts) == 3 and all(all(c in "0123456789" for c in p) for p in parts)',
            "count = 0\ndigits = 0\n"
            'for c in text + ".":\n    if c == ".":\n        if digits == 0:\n'
            "            return False\n        count += 1\n        digits = 0\n"
            '    elif c in "0123456789":\n        digits += 1\n    else:\n        return False\n'
            "return count == 3",
            [(["1.20.3"], True), (["1..3"], False), (["1.2"], False)],
            grid(_strings("10.x", 5) + ["12.0.7", "1.2.3.4", "305.1.22"]),
            "All strings over {1,0,.,x} of length 0..5, plus three longer listed strings.",
        ),
        task(
            "chess-square",
            "validation",
            "missing_condition",
            "Return True iff text names a chessboard square: exactly two characters, a letter "
            "from 'a' through 'h' followed by a digit from '1' through '8'.",
            "text",
            'return len(text) == 2 and text[0] in "abcdefgh" and text[1] in "12345678"',
            'return len(text) > 1 and text[0] in "abcdefgh" and text[1] in "12345678"',
            'return text in [c + d for c in "abcdefgh" for d in "12345678"]',
            [(["e4"], True), (["e45"], False), (["i1"], False)],
            grid(_strings("ahi189A", 3)),
            "All strings over {a,h,i,1,8,9,A} of length 0..3.",
        ),
        task(
            "price-format",
            "validation",
            "missing_condition",
            "Return True iff text is a price written as one or more ASCII digits, a dot, and "
            "exactly two ASCII digits, like '12.50'.",
            "text",
            'whole, dot, cents = text.partition(".")\n'
            'return dot == "." and len(whole) > 0 and len(cents) == 2 and all(\n'
            '    c in "0123456789" for c in whole + cents\n)',
            'whole, dot, cents = text.partition(".")\n'
            'return dot == "." and len(cents) == 2 and all(\n'
            '    c in "0123456789" for c in whole + cents\n)',
            'if len(text) < 4 or text[-3] != ".":\n    return False\n'
            'return all(c in "0123456789" for i, c in enumerate(text) if i != len(text) - 3)',
            [(["12.50"], True), ([".50"], False), (["3.5"], False)],
            grid(
                [
                    w + s + c
                    for w in ["", "0", "7", "12", "305", "x"]
                    for s in [".", ",", ""]
                    for c in ["", "5", "50", "99", "505", "5x"]
                ]
            ),
            "Every whole part in {'',0,7,12,305,x}, separator in {.,comma,''}, and cents "
            "part in {'',5,50,99,505,5x}.",
        ),
        task(
            "alternating-signs",
            "validation",
            "missing_condition",
            "Return True iff no item of values is zero and no two neighboring items have the "
            "same sign. Shorter lists qualify when they contain no zero.",
            "values",
            "if any(v == 0 for v in values):\n    return False\n"
            "return all((a > 0) != (b > 0) for a, b in zip(values, values[1:]))",
            "return all((a > 0) != (b > 0) for a, b in zip(values, values[1:]))",
            "previous = 0\nfor v in values:\n    if v == 0 or v * previous > 0:\n"
            "        return False\n    previous = v\nreturn True",
            [([[1, -2, 3]], True), ([[1, 0]], False), ([[2, 3]], False)],
            grid(sequences([-2, -1, 0, 1, 2], 4)),
            "All lists over {-2,-1,0,1,2} of length 0..4.",
        ),
        task(
            "exam-pass",
            "validation",
            "comparison",
            "Return True iff a student passes: score is at least 50, and attended is more "
            "than 75 percent of sessions, that is, attended * 100 > sessions * 75. sessions >= 1.",
            "score, attended, sessions",
            "return score >= 50 and attended * 100 > sessions * 75",
            "return score > 50 and attended * 100 > sessions * 75",
            "if score < 50:\n    return False\nreturn attended / sessions > 0.75",
            [([50, 8, 10], True), ([49, 10, 10], False), ([90, 7, 10], False)],
            [
                [score, attended, sessions]
                for score in [0, 49, 50, 51, 100]
                for sessions in [1, 4, 8]
                for attended in range(sessions + 1)
            ],
            "score in {0,49,50,51,100}, sessions in {1,4,8}, every attended from 0 to sessions.",
        ),
        task(
            "in-rectangle",
            "validation",
            "comparison",
            "Return True iff point (x, y) lies inside or on the border of the rectangle "
            "spanning left to right horizontally and bottom to top vertically. left <= right "
            "and bottom <= top.",
            "x, y, left, bottom, right, top",
            "return left <= x <= right and bottom <= y <= top",
            "return left <= x < right and bottom <= y < top",
            "return min(max(x, left), right) == x and min(max(y, bottom), top) == y",
            [([1, 1, 0, 0, 2, 2], True), ([2, 2, 0, 0, 2, 2], True), ([3, 1, 0, 0, 2, 2], False)],
            [
                [x, y, left, bottom, right, top]
                for x in range(-1, 4)
                for y in range(-1, 4)
                for left, right in ((0, 0), (0, 2), (1, 3))
                for bottom, top in ((0, 0), (0, 2), (1, 3))
            ],
            "x and y in [-1,3]; each axis spans [0,0], [0,2], or [1,3].",
        ),
        task(
            "even-split",
            "validation",
            "comparison",
            "Return True iff total items can be shared equally among groups groups with each "
            "group getting at least minimum items. groups >= 1.",
            "total, groups, minimum",
            "return total % groups == 0 and total // groups >= minimum",
            "return total % groups == 0 and total // groups > minimum",
            "share, rest = divmod(total, groups)\nreturn rest == 0 and share >= minimum",
            [([12, 3, 4], True), ([12, 5, 1], False), ([12, 4, 4], False)],
            grid(range(13), [1, 2, 3, 4], [0, 1, 3]),
            "total in [0,12], groups in [1,4], minimum in {0,1,3}.",
        ),
        task(
            "heart-rate-zone",
            "validation",
            "comparison",
            "Return True iff bpm is in the training zone for age: from 50 through 85 percent "
            "of 220 - age, inclusive. Compare exactly with integers: 100 * bpm must be from "
            "50 * (220 - age) through 85 * (220 - age).",
            "bpm, age",
            "limit = 220 - age\nreturn 50 * limit <= 100 * bpm <= 85 * limit",
            "limit = 220 - age\nreturn 50 * limit <= 100 * bpm < 85 * limit",
            "low = 50 * (220 - age)\nhigh = 85 * (220 - age)\n"
            "return not (100 * bpm < low or 100 * bpm > high)",
            [([170, 20], True), ([100, 20], True), ([171, 20], False)],
            grid(range(55, 206), [0, 20, 40, 60, 100]),
            "bpm in [55,205], age in {0,20,40,60,100}.",
        ),
        task(
            "check-digit",
            "validation",
            "off_by_one",
            "code is a string of at least two ASCII digits. Return True iff its last digit "
            "equals the sum of all its other digits modulo 10.",
            "code",
            "return sum(int(c) for c in code[:-1]) % 10 == int(code[-1])",
            "return sum(int(c) for c in code[:-2]) % 10 == int(code[-1])",
            "total = 0\nfor i in range(len(code) - 1):\n"
            '    total += "0123456789".index(code[i])\n'
            'return total % 10 == "0123456789".index(code[-1])',
            [(["123"], True), (["124"], False), (["909"], True)],
            grid(["".join(xs) for n in (2, 3, 4) for xs in product("0137", repeat=n)]),
            "All strings over {0,1,3,7} of length 2..4.",
        ),
        task(
            "ends-match",
            "validation",
            "off_by_one",
            "Return True iff the first n characters of text are the same as its last n "
            "characters. 0 <= n <= len(text).",
            "text, n",
            "return text[:n] == text[len(text) - n :]",
            "return text[:n] == text[-n:]",
            "return all(text[i] == text[len(text) - n + i] for i in range(n))",
            [(["abcab", 2], True), (["abc", 0], True), (["abca", 2], False)],
            [[t, n] for t in _strings("ab", 4) for n in range(len(t) + 1)],
            "Every string over {a,b} of length 0..4 with every n from 0 to its length.",
        ),
        task(
            "lucky-ticket",
            "validation",
            "off_by_one",
            "code is a string of ASCII digits with even length. Return True iff the digits of "
            "its first half and of its second half have equal sums.",
            "code",
            "half = len(code) // 2\n"
            "return sum(map(int, code[:half])) == sum(map(int, code[half:]))",
            "half = len(code) // 2\n"
            "return sum(map(int, code[:half])) == sum(map(int, code[half + 1 :]))",
            "balance = 0\nfor i, c in enumerate(code):\n"
            "    balance += int(c) if i < len(code) // 2 else -int(c)\nreturn balance == 0",
            [(["1230"], True), (["12"], False), (["3003"], True)],
            grid(["".join(xs) for n in (0, 2, 4) for xs in product("0123", repeat=n)]),
            "All strings over {0,1,2,3} of length 0, 2, or 4.",
        ),
        task(
            "index-in-range",
            "validation",
            "off_by_one",
            "Return True iff every integer in indexes is a valid Python index into a list of "
            "length size, counting negative indexes from the end: -size <= i < size. size >= 0.",
            "indexes, size",
            "return all(-size <= i < size for i in indexes)",
            "return all(1 - size <= i < size for i in indexes)",
            "for i in indexes:\n    if i < 0:\n        i += size\n    if not 0 <= i < size:\n"
            "        return False\nreturn True",
            [([[0, -3], 3], True), ([[3], 3], False), ([[-4], 3], False)],
            grid(sequences([-4, -3, -1, 0, 2, 3], 2), [0, 1, 3]),
            "All index lists over {-4,-3,-1,0,2,3} of length 0..2; size in {0,1,3}.",
        ),
        task(
            "luhn-checksum",
            "validation",
            "update_logic",
            "number is a nonempty string of ASCII digits. Return True iff it passes the Luhn "
            "check: starting from the rightmost digit, double every second digit, subtract 9 "
            "from any doubled result above 9, and the total of all digits must be divisible by 10.",
            "number",
            "total = 0\ndouble = False\nfor c in reversed(number):\n    d = int(c)\n"
            "    if double:\n        d *= 2\n        if d > 9:\n            d -= 9\n"
            "    total += d\n    double = not double\nreturn total % 10 == 0",
            "total = 0\ndouble = False\nfor c in reversed(number):\n    d = int(c)\n"
            "    if double:\n        d *= 2\n        if d > 9:\n            d -= 9\n"
            "    total += d\n    double = True\nreturn total % 10 == 0",
            "digits = [int(c) for c in number][::-1]\n"
            "doubled = [sum(divmod(2 * d, 10)) for d in digits[1::2]]\n"
            "return (sum(digits[::2]) + sum(doubled)) % 10 == 0",
            [(["79927398713"], True), (["79927398710"], False), (["18"], True)],
            grid(["".join(xs) for n in range(1, 5) for xs in product("0159", repeat=n)]),
            "All strings over {0,1,5,9} of length 1..4.",
        ),
        task(
            "arithmetic-steps",
            "validation",
            "update_logic",
            "Return True iff each item of values is exactly step more than the item before "
            "it. Lists with fewer than two items qualify.",
            "values, step",
            "for i in range(1, len(values)):\n    if values[i] != values[i - 1] + step:\n"
            "        return False\nreturn True",
            "for i in range(1, len(values)):\n    if values[i] != values[0] + step:\n"
            "        return False\nreturn True",
            "if not values:\n    return True\n"
            "return values == [values[0] + i * step for i in range(len(values))]",
            [([[1, 3, 5], 2], True), ([[1, 3, 4], 2], False), ([[7], 5], True)],
            grid(sequences([0, 1, 2, 3], 3), [0, 1, 2]),
            "All lists over {0,1,2,3} of length 0..3; step in {0,1,2}.",
        ),
        task(
            "consistent-indentation",
            "validation",
            "update_logic",
            "lines is a list of strings; a line's indentation is its number of leading spaces. "
            "Return True iff every indentation is a multiple of 4 and at most 4 more than the "
            "previous line's. The first line may be indented at most 4.",
            "lines",
            "previous = 0\nfor line in lines:\n"
            '    indent = len(line) - len(line.lstrip(" "))\n'
            "    if indent % 4 or indent > previous + 4:\n        return False\n"
            "    previous = indent\nreturn True",
            "previous = 0\nfor line in lines:\n"
            '    indent = len(line) - len(line.lstrip(" "))\n'
            "    if indent % 4 or indent > previous + 4:\n        return False\nreturn True",
            'indents = [len(line) - len(line.lstrip(" ")) for line in lines]\n'
            "for before, after in zip([0] + indents, indents):\n"
            "    if after % 4 != 0 or after - before > 4:\n        return False\nreturn True",
            [
                ([["a", "    b", "        c"]], True),
                ([["a", "        b"]], False),
                ([["  a"]], False),
            ],
            grid(sequences(lines, 4)),
            "All lists of length 0..4 of 'x' indented by 0, 2, 4, or 8 spaces.",
        ),
        task(
            "drawdown-limit",
            "validation",
            "update_logic",
            "Return True iff prices never fall more than limit below the highest price seen "
            "so far.",
            "prices, limit",
            "peak = None\nfor price in prices:\n    if peak is None or price > peak:\n"
            "        peak = price\n    if peak - price > limit:\n        return False\n"
            "return True",
            "peak = None\nfor price in prices:\n"
            "    if peak is not None and peak - price > limit:\n        return False\n"
            "    peak = price\nreturn True",
            "for i, price in enumerate(prices):\n"
            "    if prices[:i] and max(prices[:i]) - price > limit:\n        return False\n"
            "return True",
            [([[10, 8, 6], 3], False), ([[10, 8, 9], 3], True), ([[], 0], True)],
            grid(sequences([0, 2, 5, 9], 4), [0, 1, 3, 6]),
            "All lists over {0,2,5,9} of length 0..4; limit in {0,1,3,6}.",
        ),
        # Collections
        task(
            "safe-average",
            "collections",
            "missing_condition",
            "Return the average of values rounded down, sum(values) // len(values), or 0 for "
            "an empty list.",
            "values",
            "if not values:\n    return 0\nreturn sum(values) // len(values)",
            "return sum(values) // len(values)",
            "total = count = 0\nfor v in values:\n    total += v\n    count += 1\n"
            "return total // count if count else 0",
            [([[2, 4]], 3), ([[]], 0), ([[1, 2]], 1)],
            grid(sequences([-3, 0, 1, 4, 7], 3)),
            "All lists over {-3,0,1,4,7} of length 0..3.",
        ),
        task(
            "only-in-one",
            "collections",
            "missing_condition",
            "Return the sorted list of distinct integers that appear in exactly one of left "
            "and right.",
            "left, right",
            "return sorted(set(left) ^ set(right))",
            "return sorted(set(left) - set(right))",
            "result = []\nfor v in left + right:\n"
            "    if (v in left) != (v in right) and v not in result:\n        result.append(v)\n"
            "return sorted(result)",
            [([[1, 2, 3], [2, 3, 4]], [1, 4]), ([[1, 1], []], [1]), ([[], [5]], [5])],
            grid(sequences([0, 1, 2], 3), sequences([1, 2, 3], 2)),
            "left: lists over {0,1,2} of length 0..3; right: lists over {1,2,3} of length 0..2.",
        ),
        task(
            "lookup-with-default",
            "collections",
            "missing_condition",
            "Return [mapping[key] for each key in keys], using default for any key missing "
            "from mapping.",
            "mapping, keys, default",
            "return [mapping[k] if k in mapping else default for k in keys]",
            "return [mapping[k] for k in keys if k in mapping]",
            "return [mapping.get(k, default) for k in keys]",
            [
                ([{"a": 1}, ["a", "b"], 0], [1, 0]),
                ([{}, [], 5], []),
                ([{"x": 2}, ["x"], 9], [2]),
            ],
            grid(
                [{}, {"a": 1}, {"a": 1, "b": 0}, {"b": 2, "c": 3}],
                sequences(["a", "b", "c"], 2),
                [0, -1],
            ),
            "Four listed mappings, key lists over {a,b,c} of length 0..2, default in {0,-1}.",
        ),
        task(
            "trimmed-nonempty",
            "collections",
            "missing_condition",
            "Return the strings of items with leading and trailing spaces removed, dropping "
            "any that are empty after trimming, in order.",
            "items",
            'return [s.strip(" ") for s in items if s.strip(" ")]',
            'return [s.strip(" ") for s in items if s]',
            'result = []\nfor s in items:\n    trimmed = s.strip(" ")\n    if trimmed != "":\n'
            "        result.append(trimmed)\nreturn result",
            [([[" a ", "", "b"]], ["a", "b"]), ([["  "]], []), ([[]], [])],
            grid(sequences(["", " ", "a", " b ", "a b"], 3)),
            "All lists of length 0..3 over five listed strings.",
        ),
        task(
            "local-peaks",
            "collections",
            "comparison",
            "Return the indexes of items strictly greater than each neighbor they have, in "
            "order. The first and last items have one neighbor; a single item is a peak.",
            "values",
            "peaks = []\nfor i, v in enumerate(values):\n"
            "    left = i == 0 or v > values[i - 1]\n"
            "    right = i == len(values) - 1 or v > values[i + 1]\n"
            "    if left and right:\n        peaks.append(i)\nreturn peaks",
            "peaks = []\nfor i, v in enumerate(values):\n"
            "    left = i == 0 or v >= values[i - 1]\n"
            "    right = i == len(values) - 1 or v >= values[i + 1]\n"
            "    if left and right:\n        peaks.append(i)\nreturn peaks",
            "padded = [None] + values + [None]\nreturn [\n    i - 1\n"
            "    for i in range(1, len(padded) - 1)\n"
            "    if (padded[i - 1] is None or padded[i] > padded[i - 1])\n"
            "    and (padded[i + 1] is None or padded[i] > padded[i + 1])\n]",
            [([[1, 3, 2]], [1]), ([[2, 2, 1]], []), ([[5]], [0])],
            grid(sequences([0, 1, 2], 4)),
            "All lists over {0,1,2} of length 0..4.",
        ),
        task(
            "insert-position",
            "collections",
            "comparison",
            "values is sorted in nondecreasing order. Return the index at which target would "
            "be inserted to keep it sorted, placing it after any items equal to target.",
            "values, target",
            "count = 0\nfor v in values:\n    if v <= target:\n        count += 1\nreturn count",
            "count = 0\nfor v in values:\n    if v < target:\n        count += 1\nreturn count",
            "low, high = 0, len(values)\nwhile low < high:\n    middle = (low + high) // 2\n"
            "    if values[middle] <= target:\n        low = middle + 1\n    else:\n"
            "        high = middle\nreturn low",
            [([[1, 2, 2, 3], 2], 3), ([[], 5], 0), ([[1, 3], 2], 1)],
            grid(sorted_lists, range(-1, 5)),
            "All nondecreasing lists over {0,1,2,3} of length 0..4; target in [-1,4].",
        ),
        task(
            "closest-value",
            "collections",
            "comparison",
            "values is a nonempty list of integers. Return the item closest to target; when "
            "two items are equally close, return the smaller one.",
            "values, target",
            "return min(values, key=lambda v: (abs(v - target), v))",
            "best = values[0]\nfor v in values:\n"
            "    if abs(v - target) <= abs(best - target):\n        best = v\nreturn best",
            "best = values[0]\nfor v in values[1:]:\n"
            "    distance, current = abs(v - target), abs(best - target)\n"
            "    if distance < current or (distance == current and v < best):\n"
            "        best = v\nreturn best",
            [([[1, 4, 6], 5], 4), ([[3], 0], 3), ([[10, 2], 1], 2)],
            grid([xs for xs in sequences([-2, 0, 1, 3], 3) if xs], range(-3, 5)),
            "All nonempty lists over {-2,0,1,3} of length 1..3; target in [-3,4].",
        ),
        task(
            "top-scorers",
            "collections",
            "comparison",
            "Return the indexes of every item of scores equal to its largest value, in order. "
            "An empty list returns [].",
            "scores",
            "best, result = None, []\nfor i, s in enumerate(scores):\n"
            "    if best is None or s > best:\n        best, result = s, [i]\n"
            "    elif s == best:\n        result.append(i)\nreturn result",
            "best, result = None, []\nfor i, s in enumerate(scores):\n"
            "    if best is None or s >= best:\n        best, result = s, [i]\n"
            "    elif s == best:\n        result.append(i)\nreturn result",
            "if not scores:\n    return []\nbest = max(scores)\n"
            "return [i for i, s in enumerate(scores) if s == best]",
            [([[3, 1, 3]], [0, 2]), ([[]], []), ([[2, 5]], [1])],
            grid(sequences([-1, 0, 2, 3], 4)),
            "All lists over {-1,0,2,3} of length 0..4.",
        ),
        task(
            "every-nth",
            "collections",
            "off_by_one",
            "Return every nth item of values starting with the first: values[0], values[n], "
            "values[2n], and so on. n >= 1.",
            "values, n",
            "return [values[i] for i in range(0, len(values), n)]",
            "return [values[i] for i in range(n - 1, len(values), n)]",
            "return [v for i, v in enumerate(values) if i % n == 0]",
            [([[1, 2, 3, 4, 5], 2], [1, 3, 5]), ([[], 3], []), ([[7], 1], [7])],
            grid(sequences([0, 1, 2], 4), range(1, 5)),
            "All lists over {0,1,2} of length 0..4; n in [1,4].",
        ),
        task(
            "window-sums",
            "collections",
            "off_by_one",
            "Return the sum of each run of size consecutive items of values, in order. "
            "size >= 1; return [] when size exceeds the length.",
            "values, size",
            "return [sum(values[i : i + size]) for i in range(len(values) - size + 1)]",
            "return [sum(values[i : i + size]) for i in range(len(values) - size)]",
            "result = []\ntotal = 0\nfor i, v in enumerate(values):\n    total += v\n"
            "    if i >= size:\n        total -= values[i - size]\n"
            "    if i >= size - 1:\n        result.append(total)\nreturn result",
            [([[1, 2, 3], 2], [3, 5]), ([[1], 2], []), ([[4, 5], 1], [4, 5])],
            grid(sequences([-1, 0, 2, 5], 4), range(1, 5)),
            "All lists over {-1,0,2,5} of length 0..4; size in [1,4].",
        ),
        task(
            "histogram-buckets",
            "collections",
            "off_by_one",
            "values are nonnegative integers. Count them into buckets of size width: bucket i "
            "counts items from i * width up to but not including (i + 1) * width, for every i "
            "from 0 through max(values) // width. An empty list returns [].",
            "values, width",
            "if not values:\n    return []\ncounts = [0] * (max(values) // width + 1)\n"
            "for v in values:\n    counts[v // width] += 1\nreturn counts",
            "if not values:\n    return []\ncounts = [0] * (max(values) // width)\n"
            "for v in values:\n    counts[v // width] += 1\nreturn counts",
            "if not values:\n    return []\nreturn [\n"
            "    sum(1 for v in values if v // width == i) for i in range(max(values) // width + 1)\n]",
            [([[0, 1, 5], 2], [2, 0, 1]), ([[], 3], []), ([[3], 3], [0, 1])],
            grid(sequences([0, 1, 3, 4, 7], 3), [1, 2, 3, 5]),
            "All lists over {0,1,3,4,7} of length 0..3; width in {1,2,3,5}.",
        ),
        task(
            "join-with-separator",
            "collections",
            "off_by_one",
            "Return a new list with sep inserted between each pair of neighboring items of values.",
            "values, sep",
            "result = []\nfor i, v in enumerate(values):\n    if i > 0:\n"
            "        result.append(sep)\n    result.append(v)\nreturn result",
            "result = []\nfor i in range(len(values) - 1):\n    result += [values[i], sep]\n"
            "return result",
            "result = []\nfor v in values:\n    result += [v, sep]\nreturn result[:-1]",
            [([[1, 2, 3], 0], [1, 0, 2, 0, 3]), ([[], 0], []), ([[5], 9], [5])],
            grid(sequences([1, 2, 3], 4), [0, 9]),
            "All lists over {1,2,3} of length 0..4; sep in {0,9}.",
        ),
        task(
            "merge-intervals",
            "collections",
            "update_logic",
            "intervals is a list of [start, end] pairs with start <= end, sorted by start. "
            "Merge every interval that overlaps or touches the one before it (its start <= the "
            "merged end so far) and return the merged intervals in order.",
            "intervals",
            "merged = []\nfor start, end in intervals:\n"
            "    if merged and start <= merged[-1][1]:\n"
            "        merged[-1][1] = max(merged[-1][1], end)\n    else:\n"
            "        merged.append([start, end])\nreturn merged",
            "merged = []\nfor start, end in intervals:\n"
            "    if merged and start <= merged[-1][1]:\n        merged[-1][1] = end\n"
            "    else:\n        merged.append([start, end])\nreturn merged",
            "result = []\nfor start, end in intervals:\n"
            "    if result and start <= result[-1][1]:\n"
            "        result[-1] = [result[-1][0], max(result[-1][1], end)]\n    else:\n"
            "        result.append([start, end])\nreturn result",
            [
                ([[[1, 5], [2, 3]]], [[1, 5]]),
                ([[[1, 2], [2, 4], [6, 7]]], [[1, 4], [6, 7]]),
                ([[]], []),
            ],
            grid(interval_lists),
            "All start-sorted lists of 0..3 intervals [s,e] with 0 <= s <= 3 and s <= e <= 4.",
        ),
        task(
            "invert-mapping",
            "collections",
            "update_logic",
            "mapping maps strings to strings. Return a dict mapping each value to the sorted "
            "list of keys that map to it.",
            "mapping",
            "inverted = {}\nfor key, value in mapping.items():\n"
            "    inverted.setdefault(value, []).append(key)\n"
            "return {value: sorted(keys) for value, keys in inverted.items()}",
            "inverted = {}\nfor key, value in mapping.items():\n    inverted[value] = [key]\n"
            "return {value: sorted(keys) for value, keys in inverted.items()}",
            "return {v: sorted(k for k in mapping if mapping[k] == v) for v in set(mapping.values())}",
            [
                ([{"a": "x", "b": "y", "c": "x"}], {"x": ["a", "c"], "y": ["b"]}),
                ([{}], {}),
                ([{"k": "v"}], {"v": ["k"]}),
            ],
            grid(mappings),
            "Every mapping from a subset of keys {a,b,c,d} to values in {x,y}.",
        ),
        task(
            "last-value-per-key",
            "collections",
            "update_logic",
            "records is a list of [key, value] pairs. Return [key, value] pairs keeping each "
            "key's LAST value, ordered by each key's first appearance.",
            "records",
            "latest = {}\nfor key, value in records:\n    latest[key] = value\n"
            "return [[key, value] for key, value in latest.items()]",
            "latest = {}\nfor key, value in records:\n    if key not in latest:\n"
            "        latest[key] = value\nreturn [[key, value] for key, value in latest.items()]",
            "order = []\nfor key, _ in records:\n    if key not in order:\n"
            "        order.append(key)\n"
            "return [[key, [v for k, v in records if k == key][-1]] for key in order]",
            [
                ([[["a", 1], ["b", 2], ["a", 3]]], [["a", 3], ["b", 2]]),
                ([[]], []),
                ([[["x", 1]]], [["x", 1]]),
            ],
            grid([[list(p) for p in xs] for xs in sequences(pairs, 4)]),
            "All lists of 0..4 pairs from [a,1], [a,2], [b,1], [b,3].",
        ),
        task(
            "move-zeros-to-end",
            "collections",
            "update_logic",
            "Return a new list with every 0 of values moved to the end, keeping the other "
            "integers in their original order. Do not mutate values.",
            "values",
            "result = [0] * len(values)\nposition = 0\nfor v in values:\n    if v != 0:\n"
            "        result[position] = v\n        position += 1\nreturn result",
            "result = [0] * len(values)\nposition = 0\nfor v in values:\n    if v != 0:\n"
            "        result[position] = v\n    position += 1\nreturn result",
            "return [v for v in values if v != 0] + [0] * values.count(0)",
            [([[0, 1, 0, 2]], [1, 2, 0, 0]), ([[]], []), ([[3, 0]], [3, 0])],
            grid(sequences([0, 1, -2, 3], 4)),
            "All lists over {0,1,-2,3} of length 0..4.",
        ),
        # Control flow
        task(
            "rock-paper-scissors",
            "control_flow",
            "missing_condition",
            "Return 'first', 'second', or 'draw' for one round of rock-paper-scissors, where "
            "first and second are each 'rock', 'paper', or 'scissors'. Rock beats scissors, "
            "scissors beats paper, and paper beats rock.",
            "first, second",
            'beats = {"rock": "scissors", "scissors": "paper", "paper": "rock"}\n'
            'if first == second:\n    return "draw"\n'
            'return "first" if beats[first] == second else "second"',
            'beats = {"rock": "scissors", "scissors": "paper", "paper": "rock"}\n'
            'return "first" if beats[first] == second else "second"',
            'order = ["rock", "paper", "scissors"]\n'
            "difference = (order.index(first) - order.index(second)) % 3\n"
            'return ["draw", "first", "second"][difference]',
            [
                (["rock", "scissors"], "first"),
                (["rock", "paper"], "second"),
                (["paper", "paper"], "draw"),
            ],
            grid(["rock", "paper", "scissors"], ["rock", "paper", "scissors"]),
            "All nine move pairs.",
        ),
        task(
            "ordinal-suffix",
            "control_flow",
            "missing_condition",
            "Return n followed by its English ordinal suffix, for integer n >= 1: 1st, 2nd, "
            "3rd, 4th, 11th, 12th, 13th, 21st, 112th, and so on.",
            "n",
            'if n % 100 in (11, 12, 13):\n    return str(n) + "th"\n'
            'return str(n) + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")',
            'return str(n) + {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")',
            "last_two = n % 100\nlast = n % 10\n"
            "if 11 <= last_two <= 13 or last == 0 or last >= 4:\n"
            '    suffix = "th"\nelse:\n    suffix = ["", "st", "nd", "rd"][last]\n'
            "return str(n) + suffix",
            [([1], "1st"), ([12], "12th"), ([23], "23rd")],
            grid(range(1, 260)),
            "n in [1,259].",
        ),
        task(
            "season-of-month",
            "control_flow",
            "missing_condition",
            "Return the season for month, an integer from 1 through 12: 'winter' for December, "
            "January, and February; 'spring' for March through May; 'summer' for June through "
            "August; and 'autumn' for September through November.",
            "month",
            'if month == 12 or month <= 2:\n    return "winter"\n'
            'if month <= 5:\n    return "spring"\nif month <= 8:\n    return "summer"\n'
            'return "autumn"',
            'if month <= 2:\n    return "winter"\n'
            'if month <= 5:\n    return "spring"\nif month <= 8:\n    return "summer"\n'
            'return "autumn"',
            'return ["winter", "spring", "summer", "autumn"][(month % 12) // 3]',
            [([1], "winter"), ([12], "winter"), ([7], "summer")],
            grid(range(1, 13)),
            "month in [1,12].",
        ),
        task(
            "life-cell",
            "control_flow",
            "missing_condition",
            "In Conway's Game of Life, return whether a cell is alive in the next generation: "
            "a live cell survives with 2 or 3 live neighbors, and a dead cell becomes alive "
            "with exactly 3. neighbors is from 0 through 8.",
            "alive, neighbors",
            "if alive:\n    return neighbors in (2, 3)\nreturn neighbors == 3",
            "return neighbors in (2, 3)",
            "return neighbors == 3 or (alive and neighbors == 2)",
            [([True, 2], True), ([False, 2], False), ([False, 3], True)],
            grid([True, False], range(9)),
            "alive in {True,False}, neighbors in [0,8].",
        ),
        task(
            "stop-at-limit",
            "control_flow",
            "comparison",
            "Add the nonnegative items of values in order, stopping before the running total "
            "would exceed limit. Return how many items were added.",
            "values, limit",
            "total = count = 0\nfor v in values:\n    if total + v > limit:\n        break\n"
            "    total += v\n    count += 1\nreturn count",
            "total = count = 0\nfor v in values:\n    if total + v >= limit:\n        break\n"
            "    total += v\n    count += 1\nreturn count",
            "for count in range(len(values) + 1):\n"
            "    if count == len(values) or sum(values[: count + 1]) > limit:\n"
            "        return count",
            [([[2, 3, 4], 5], 2), ([[], 3], 0), ([[6], 5], 0)],
            grid(sequences([0, 1, 2, 4], 4), [0, 2, 3, 5]),
            "All lists over {0,1,2,4} of length 0..4; limit in {0,2,3,5}.",
        ),
        task(
            "bounce-count",
            "control_flow",
            "comparison",
            "A ball dropped from height cm rebounds to half its previous height, rounded down, "
            "on each bounce. Return how many bounces reach at least minimum cm. height >= 0 and "
            "minimum >= 1.",
            "height, minimum",
            "count = 0\nheight //= 2\nwhile height >= minimum:\n    count += 1\n"
            "    height //= 2\nreturn count",
            "count = 0\nheight //= 2\nwhile height > minimum:\n    count += 1\n"
            "    height //= 2\nreturn count",
            "count = 0\nwhile height // 2 ** (count + 1) >= minimum:\n    count += 1\nreturn count",
            [([100, 25], 2), ([10, 20], 0), ([8, 1], 3)],
            grid([0, 1, 7, 8, 16, 40, 100, 129], [1, 2, 4, 5, 25, 50]),
            "height in {0,1,7,8,16,40,100,129}, minimum in {1,2,4,5,25,50}.",
        ),
        task(
            "tennis-game",
            "control_flow",
            "comparison",
            "first and second are the points each player has won in a tennis game. A player "
            "wins with at least 4 points and at least 2 more than the other. Return 'first' or "
            "'second' for the winner, or 'ongoing' if neither has won.",
            "first, second",
            'if first >= 4 and first - second >= 2:\n    return "first"\n'
            'if second >= 4 and second - first >= 2:\n    return "second"\n'
            'return "ongoing"',
            'if first >= 4 and first - second > 2:\n    return "first"\n'
            'if second >= 4 and second - first > 2:\n    return "second"\n'
            'return "ongoing"',
            "lead = first - second\nif max(first, second) < 4 or abs(lead) < 2:\n"
            '    return "ongoing"\nreturn "first" if lead > 0 else "second"',
            [([4, 2], "first"), ([4, 3], "ongoing"), ([3, 5], "second")],
            grid(range(8), range(8)),
            "Both scores in [0,7].",
        ),
        task(
            "snail-climb",
            "control_flow",
            "comparison",
            "A snail at the bottom of a well climbs up meters each day and slips down meters "
            "each night. Return the day on which it first reaches the top, a height of at "
            "least depth. up > down >= 0 and depth >= 1.",
            "up, down, depth",
            "height = day = 0\nwhile True:\n    day += 1\n    height += up\n"
            "    if height >= depth:\n        return day\n    height -= down",
            "height = day = 0\nwhile True:\n    day += 1\n    height += up\n"
            "    if height > depth:\n        return day\n    height -= down",
            "if up >= depth:\n    return 1\n"
            "return (depth - up + (up - down) - 1) // (up - down) + 1",
            [([3, 1, 5], 2), ([5, 0, 5], 1), ([2, 1, 4], 3)],
            [
                [up, down, depth]
                for up in range(1, 6)
                for down in range(up)
                for depth in range(1, 13)
            ],
            "up in [1,5], down in [0,up-1], depth in [1,12].",
        ),
        task(
            "round-robin-turn",
            "control_flow",
            "off_by_one",
            "Players numbered 0 through players - 1 take turns in increasing order, wrapping "
            "around, starting with player start. Return the player who takes turn number turn, "
            "counting turns from 1. players >= 1 and 0 <= start < players.",
            "players, start, turn",
            "return (start + turn - 1) % players",
            "return (start + turn) % players",
            "player = start\nfor _ in range(turn - 1):\n    player = (player + 1) % players\n"
            "return player",
            [([3, 0, 1], 0), ([3, 2, 2], 0), ([4, 1, 4], 0)],
            [[p, s, t] for p in range(1, 5) for s in range(p) for t in range(1, 9)],
            "players in [1,4], every start, turn in [1,8].",
        ),
        task(
            "seat-row",
            "control_flow",
            "off_by_one",
            "Seats are numbered from 1, with per_row seats in each row: row 1 holds seats 1 "
            "through per_row, row 2 the next per_row, and so on. Return the row number of "
            "seat. seat >= 1 and per_row >= 1.",
            "seat, per_row",
            "return (seat - 1) // per_row + 1",
            "return seat // per_row + 1",
            "row = 1\nwhile seat > per_row:\n    seat -= per_row\n    row += 1\nreturn row",
            [([1, 4], 1), ([4, 4], 1), ([5, 4], 2)],
            grid(range(1, 26), [1, 2, 3, 4, 5, 10]),
            "seat in [1,25], per_row in {1,2,3,4,5,10}.",
        ),
        task(
            "hour-labels",
            "control_flow",
            "off_by_one",
            "Return two-digit labels for every hour from start through end inclusive, such as "
            "'09' for 9. 0 <= start <= end <= 23.",
            "start, end",
            'return ["%02d" % h for h in range(start, end + 1)]',
            'return ["%02d" % h for h in range(start, end)]',
            "labels = []\nhour = start\nwhile hour <= end:\n"
            '    labels.append(("0" + str(hour))[-2:])\n    hour += 1\nreturn labels',
            [([8, 10], ["08", "09", "10"]), ([5, 5], ["05"]), ([0, 1], ["00", "01"])],
            [[s, e] for s in range(24) for e in range(s, 24)],
            "Every 0 <= start <= end <= 23.",
        ),
        task(
            "ordered-pairs",
            "control_flow",
            "off_by_one",
            "Return every pair [i, j] of integers with 0 <= i < j < n, ordered by i and then by j.",
            "n",
            "return [[i, j] for i in range(n) for j in range(i + 1, n)]",
            "return [[i, j] for i in range(n) for j in range(i, n)]",
            "pairs = []\nfor j in range(n):\n    for i in range(j):\n        pairs.append([i, j])\n"
            "return sorted(pairs)",
            [([3], [[0, 1], [0, 2], [1, 2]]), ([1], []), ([0], [])],
            grid(range(13)),
            "n in [0,12].",
        ),
        task(
            "dice-score",
            "control_flow",
            "update_logic",
            "Score die rolls in order: each roll adds its value to the score, except a roll of "
            "1, which resets the score to 0. Return the final score.",
            "rolls",
            "score = 0\nfor roll in rolls:\n    if roll == 1:\n        score = 0\n    else:\n"
            "        score += roll\nreturn score",
            "score = 0\nfor roll in rolls:\n    if roll == 1:\n        score = 0\n    else:\n"
            "        score = roll\nreturn score",
            "last_one = -1\nfor i, roll in enumerate(rolls):\n    if roll == 1:\n"
            "        last_one = i\nreturn sum(rolls[last_one + 1 :])",
            [([[3, 4]], 7), ([[3, 1, 2]], 2), ([[]], 0)],
            grid(sequences([1, 2, 5, 6], 4)),
            "All roll lists over {1,2,5,6} of length 0..4.",
        ),
        task(
            "yearly-interest",
            "control_flow",
            "update_logic",
            "Each year, interest of rate percent of the current balance, rounded down to a "
            "whole number, is added to the balance. Return the balance after years years. All "
            "inputs are nonnegative integers.",
            "balance, rate, years",
            "for _ in range(years):\n    balance += balance * rate // 100\nreturn balance",
            "start = balance\nfor _ in range(years):\n    balance += start * rate // 100\n"
            "return balance",
            "history = [balance]\nwhile len(history) <= years:\n"
            "    history.append(history[-1] + history[-1] * rate // 100)\nreturn history[-1]",
            [([100, 10, 2], 121), ([100, 0, 5], 100), ([50, 50, 1], 75)],
            grid([0, 1, 50, 99, 100, 1000], [0, 1, 5, 10, 50, 100], range(6)),
            "balance in {0,1,50,99,100,1000}, rate in {0,1,5,10,50,100}, years in [0,5].",
        ),
        task(
            "stack-machine",
            "control_flow",
            "update_logic",
            "Run program on an empty stack and return the final stack, bottom first. Each "
            "instruction is an integer, which is pushed; 'add', which pops the top two items "
            "and pushes their sum; or 'dup', which pushes a copy of the top item. Every "
            "program is valid.",
            "program",
            "stack = []\nfor op in program:\n"
            '    if op == "add":\n        stack.append(stack.pop() + stack.pop())\n'
            '    elif op == "dup":\n        stack.append(stack[-1])\n    else:\n'
            "        stack.append(op)\nreturn stack",
            "stack = []\nfor op in program:\n"
            '    if op == "add":\n        stack.append(stack.pop() + stack[-1])\n'
            '    elif op == "dup":\n        stack.append(stack[-1])\n    else:\n'
            "        stack.append(op)\nreturn stack",
            "stack = []\nfor op in program:\n"
            '    if op == "add":\n        stack[-2:] = [stack[-2] + stack[-1]]\n'
            '    elif op == "dup":\n        stack = stack + [stack[-1]]\n    else:\n'
            "        stack = stack + [op]\nreturn stack",
            [([[1, 2, "add"]], [3]), ([[4, "dup", "add"]], [8]), ([[]], [])],
            grid([p for p in sequences([1, 2, "add", "dup"], 5) if _valid_program(p)]),
            "Every valid program of length 0..5 over {1, 2, add, dup}.",
        ),
        task(
            "inventory-restock",
            "control_flow",
            "update_logic",
            "Apply orders to stock in order. An order removes its quantity when at least that "
            "much is in stock and is skipped otherwise. Whenever an order leaves stock below "
            "threshold, batch more units are added. Return the final stock.",
            "stock, orders, threshold, batch",
            "for quantity in orders:\n    if quantity <= stock:\n        stock -= quantity\n"
            "        if stock < threshold:\n            stock += batch\nreturn stock",
            "for quantity in orders:\n    if quantity <= stock:\n        stock -= quantity\n"
            "        if stock < threshold:\n            stock = batch\nreturn stock",
            "level = stock\nfor quantity in orders:\n    if quantity > level:\n        continue\n"
            "    level -= quantity\n    level += batch if level < threshold else 0\nreturn level",
            [([10, [4, 4], 3, 5], 7), ([5, [6], 2, 10], 5), ([3, [3], 1, 2], 2)],
            grid([0, 3, 10], sequences([1, 4, 6], 3), [0, 3], [2, 5]),
            "stock in {0,3,10}, order lists over {1,4,6} of length 0..3, threshold in {0,3}, "
            "batch in {2,5}.",
        ),
    ]
