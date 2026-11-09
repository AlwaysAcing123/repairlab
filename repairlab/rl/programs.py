"""Thirty authored RL TRAINING programs, unseen by every earlier model and training run.

Reinforcement learning samples the model's own attempts here and rewards only passing
submissions, so no demonstration or wrong repair is needed. Each program is its own
group and must stay separate from training, development, and sealed final-test tasks.
"""

from itertools import product

from ..tasks import grid, sequences, task

LETTERS = "abcdefghijklmnopqrstuvwxyz"


def rl_programs():
    small_lists = sequences([-2, -1, 0, 1, 2])
    squares = (
        [[]]
        + [[[a]] for a in (0, 5)]
        + [[[a, b], [c, d]] for a, b, c, d in product((0, 1, 2), repeat=4)]
        + [[[a, 0, 0], [0, b, 0], [0, 0, c]] for a, b, c in product((1, 2), repeat=3)]
    )
    headings = [0, 1, 90, 179, 180, 181, 270, 359]
    return [
        task(
            "no-triple-repeat",
            "validation",
            "off_by_one",
            "Return True iff text never contains the same character three times in a row.",
            "text",
            "return all(not text[i] == text[i + 1] == text[i + 2] for i in range(len(text) - 2))",
            "return all(not text[i] == text[i + 1] == text[i + 2] for i in range(len(text) - 3))",
            "run = 0\nprevious = None\nfor c in text:\n    run = run + 1 if c == previous else 1\n"
            "    if run == 3:\n        return False\n    previous = c\nreturn True",
            [(["aab"], True), (["aaab"], False), (["baaa"], False)],
            grid(
                [
                    "",
                    "a",
                    "aa",
                    "aaa",
                    "aaaa",
                    "abab",
                    "abba",
                    "abbb",
                    "bbba",
                    "aabbaa",
                    "aabbbaa",
                    "xyz",
                    "zzzy",
                    "yzzz",
                ]
            ),
            "Fourteen listed strings with runs at the start, middle, and end.",
        ),
        task(
            "hostname-label",
            "validation",
            "missing_condition",
            "Return True iff label is a valid DNS label: 1 to 63 characters, each a lowercase "
            "ASCII letter, an ASCII digit, or a hyphen, and it neither starts nor ends with a hyphen.",
            "label",
            f'allowed = "{LETTERS}0123456789-"\n'
            "if not 1 <= len(label) <= 63 or not all(c in allowed for c in label):\n"
            "    return False\n"
            'return label[0] != "-" and label[-1] != "-"',
            f'allowed = "{LETTERS}0123456789-"\n'
            "if not 1 <= len(label) <= 63 or not all(c in allowed for c in label):\n"
            "    return False\n"
            'return label[0] != "-"',
            "if len(label) == 0 or len(label) > 63:\n    return False\n"
            'if label.startswith("-") or label.endswith("-"):\n    return False\n'
            'return all(c.isdigit() and c.isascii() or "a" <= c <= "z" or c == "-" for c in label)',
            [(["web-01"], True), (["web-"], False), (["-web"], False), (["a" * 64], False)],
            grid(
                [
                    "",
                    "a",
                    "-",
                    "a-b",
                    "a--b",
                    "web",
                    "Web",
                    "w_b",
                    "9lives",
                    "x" * 63,
                    "x" * 64,
                    "ab-",
                    "-ab",
                    "\u00e9",
                    "a.b",
                    "\u0663a",
                ]
            ),
            "Sixteen listed labels, including boundary lengths, edge hyphens, and non-ASCII text.",
        ),
        task(
            "valid-clock-time",
            "validation",
            "comparison",
            "Return True iff hour and minute form a valid 24-hour clock time: hour from 0 "
            "through 23 and minute from 0 through 59.",
            "hour, minute",
            "return 0 <= hour <= 23 and 0 <= minute <= 59",
            "return 0 <= hour <= 24 and 0 <= minute <= 59",
            "return hour in range(24) and minute in range(60)",
            [([23, 59], True), ([24, 0], False), ([0, 0], True)],
            grid([-1, 0, 1, 12, 22, 23, 24, 25], [-1, 0, 30, 58, 59, 60, 61]),
            "hour in {-1,0,1,12,22,23,24,25}, minute in {-1,0,30,58,59,60,61}.",
        ),
        task(
            "within-tolerance",
            "validation",
            "missing_condition",
            "Return True iff measured differs from target by at most tolerance, in either "
            "direction. tolerance >= 0; all are integers.",
            "measured, target, tolerance",
            "return abs(measured - target) <= tolerance",
            "return measured - target <= tolerance",
            "return target - tolerance <= measured <= target + tolerance",
            [([12, 10, 2], True), ([7, 10, 2], False), ([5, 5, 0], True)],
            grid([-3, 0, 3, 7, 10, 12, 13], [0, 10], [0, 1, 2, 3]),
            "measured in {-3,0,3,7,10,12,13}, target in {0,10}, tolerance in [0,3].",
        ),
        task(
            "unique-usernames",
            "validation",
            "missing_condition",
            "Return True iff no username in names appears twice when letter case is ignored.",
            "names",
            "lowered = [name.lower() for name in names]\nreturn len(set(lowered)) == len(lowered)",
            "return len(set(names)) == len(names)",
            "seen = set()\nfor name in names:\n    key = name.lower()\n    if key in seen:\n"
            "        return False\n    seen.add(key)\nreturn True",
            [([["ann", "bob"]], True), ([["Ann", "ann"]], False), ([[]], True)],
            grid(sequences(["a", "A", "b", "Bb", "bB"])),
            "All lists over {'a','A','b','Bb','bB'} of length 0..3.",
        ),
        task(
            "password-classes",
            "validation",
            "comparison",
            "Return True iff text uses at least three of four character classes: lowercase "
            "ASCII letters, uppercase ASCII letters, ASCII digits, and any other character.",
            "text",
            "classes = [\n"
            '    any("a" <= c <= "z" for c in text),\n'
            '    any("A" <= c <= "Z" for c in text),\n'
            '    any("0" <= c <= "9" for c in text),\n'
            '    any(not ("a" <= c <= "z" or "A" <= c <= "Z" or "0" <= c <= "9") for c in text),\n'
            "]\n"
            "return sum(classes) >= 3",
            "classes = [\n"
            '    any("a" <= c <= "z" for c in text),\n'
            '    any("A" <= c <= "Z" for c in text),\n'
            '    any("0" <= c <= "9" for c in text),\n'
            '    any(not ("a" <= c <= "z" or "A" <= c <= "Z" or "0" <= c <= "9") for c in text),\n'
            "]\n"
            "return sum(classes) > 3",
            "lower = upper = digit = other = False\nfor c in text:\n"
            '    if "a" <= c <= "z":\n        lower = True\n'
            '    elif "A" <= c <= "Z":\n        upper = True\n'
            '    elif "0" <= c <= "9":\n        digit = True\n'
            "    else:\n        other = True\n"
            "return lower + upper + digit + other >= 3",
            [(["aB3"], True), (["ab3"], False), (["aB3!"], True)],
            grid(
                [
                    "",
                    "a",
                    "A",
                    "1",
                    "!",
                    "aA",
                    "a1",
                    "A!",
                    "aA1",
                    "aA!",
                    "a1!",
                    "A1!",
                    "aA1!",
                    "\u00e91A",
                    "abc",
                    "ABC1",
                    "a b",
                ]
            ),
            "Seventeen listed strings covering every class combination plus non-ASCII text.",
        ),
        task(
            "zip-code",
            "validation",
            "missing_condition",
            "Return True iff code is a US ZIP code: five ASCII digits, optionally followed by "
            "a hyphen and exactly four more ASCII digits.",
            "code",
            'digits = "0123456789"\nif len(code) == 5:\n    return all(c in digits for c in code)\n'
            'if len(code) == 10 and code[5] == "-":\n'
            "    return all(c in digits for c in code[:5] + code[6:])\nreturn False",
            'digits = "0123456789"\nif len(code) == 5:\n    return all(c in digits for c in code)\n'
            "if len(code) == 10:\n"
            "    return all(c in digits for c in code[:5] + code[6:])\nreturn False",
            'head, sep, tail = code.partition("-")\n'
            'if len(head) != 5 or not all(c in "0123456789" for c in head):\n    return False\n'
            "if not sep:\n    return True\n"
            'return len(tail) == 4 and all(c in "0123456789" for c in tail)',
            [(["15213"], True), (["15213-1234"], True), (["15213 1234"], False)],
            grid(
                [
                    "",
                    "1521",
                    "152134",
                    "abcde",
                    "15213-",
                    "15213-123",
                    "15213-12345",
                    "15213x1234",
                    "1521-31234",
                    "00000-0000",
                    "15213-12a4",
                    "\u06635213",
                    "12345-12-4",
                ]
            ),
            "Thirteen listed codes, including misplaced separators and non-ASCII digits.",
        ),
        task(
            "business-hours",
            "validation",
            "comparison",
            "weekday is 0 (Monday) through 6 (Sunday) and hour is 0 through 23. Return True "
            "iff weekday is Monday through Friday and 9 <= hour < 17.",
            "weekday, hour",
            "return weekday <= 4 and 9 <= hour < 17",
            "return weekday <= 4 and 9 <= hour <= 17",
            "return weekday in (0, 1, 2, 3, 4) and hour in range(9, 17)",
            [([0, 9], True), ([2, 17], False), ([5, 10], False)],
            grid(range(7), [0, 8, 9, 10, 16, 17, 18, 23]),
            "weekday in [0,6], hour in {0,8,9,10,16,17,18,23}.",
        ),
        task(
            "file-extension",
            "validation",
            "missing_condition",
            "Return True iff filename ends with one of the extensions in allowed, ignoring "
            "letter case. Each extension includes its dot, such as '.png'.",
            "filename, allowed",
            "return any(filename.lower().endswith(ext.lower()) for ext in allowed)",
            "return any(filename.endswith(ext) for ext in allowed)",
            "name = filename.lower()\nfor ext in allowed:\n    if name.endswith(ext.lower()):\n"
            "        return True\nreturn False",
            [
                (["photo.PNG", [".png"]], True),
                (["notes.txt", [".png", ".jpg"]], False),
                (["a.jpg", [".png", ".jpg"]], True),
            ],
            grid(
                ["", "a.png", "A.PNG", "a.Png", "png", "a.png.txt", "a.jpeg", "b.JPG"],
                [[], [".png"], [".PNG"], [".png", ".jpg"], [".JPG"]],
            ),
            "Eight listed filenames crossed with five extension lists.",
        ),
        task(
            "can-afford",
            "validation",
            "comparison",
            "Return True iff a purchase of price is allowed: the balance after paying, "
            "balance - price, may go negative but not below -overdraft. overdraft >= 0.",
            "price, balance, overdraft",
            "return balance - price >= -overdraft",
            "return balance - price > -overdraft",
            "return price <= balance + overdraft",
            [([5, 10, 0], True), ([15, 10, 5], True), ([16, 10, 5], False)],
            grid([0, 5, 10, 14, 15, 16, 20], [0, 10], [0, 5]),
            "price in {0,5,10,14,15,16,20}, balance in {0,10}, overdraft in {0,5}.",
        ),
        task(
            "split-by-parity",
            "collections",
            "update_logic",
            "Return [evens, odds]: the even integers of values and the odd integers of "
            "values, each list in original order.",
            "values",
            "evens = [x for x in values if x % 2 == 0]\n"
            "odds = [x for x in values if x % 2 != 0]\nreturn [evens, odds]",
            "evens, odds = [], []\nfor x in values:\n    if x % 2:\n        evens.append(x)\n"
            "    else:\n        odds.append(x)\nreturn [evens, odds]",
            "groups = [[], []]\nfor x in values:\n    groups[x % 2].append(x)\nreturn groups",
            [([[1, 2, 3]], [[2], [1, 3]]), ([[]], [[], []]), ([[-1, 4]], [[4], [-1]])],
            grid(small_lists),
            "All lists over {-2,-1,0,1,2} of length 0..3.",
        ),
        task(
            "collapse-repeats",
            "collections",
            "off_by_one",
            "Collapse each run of equal neighboring integers into one item, preserving order: "
            "[1, 1, 2, 1] becomes [1, 2, 1].",
            "values",
            "result = []\nfor x in values:\n    if not result or result[-1] != x:\n"
            "        result.append(x)\nreturn result",
            "result = []\nfor i in range(1, len(values)):\n    if values[i] != values[i - 1]:\n"
            "        result.append(values[i])\nreturn result",
            "return [x for i, x in enumerate(values) if i == 0 or values[i - 1] != x]",
            [([[1, 1, 2, 1]], [1, 2, 1]), ([[]], []), ([[3, 3]], [3])],
            grid(sequences([0, 1, 2], 4)),
            "All lists over {0,1,2} of length 0..4.",
        ),
        task(
            "top-k-largest",
            "collections",
            "missing_condition",
            "Return the k largest integers of values in descending order. k >= 0; when k "
            "exceeds the length, return every item in descending order.",
            "values, k",
            "return sorted(values, reverse=True)[:k]",
            "return sorted(values)[-k:][::-1]",
            "remaining = list(values)\nresult = []\nwhile remaining and len(result) < k:\n"
            "    best = max(remaining)\n    remaining.remove(best)\n    result.append(best)\n"
            "return result",
            [([[3, 1, 2], 2], [3, 2]), ([[3, 1, 2], 0], []), ([[5], 3], [5])],
            grid(sequences([-1, 0, 2]), [0, 1, 2, 4]),
            "All lists over {-1,0,2} of length 0..3, k in {0,1,2,4}.",
        ),
        task(
            "count-above-average",
            "collections",
            "comparison",
            "Return how many integers in values are strictly greater than their average. "
            "An empty list returns 0.",
            "values",
            "if not values:\n    return 0\ntotal = sum(values)\n"
            "return sum(1 for x in values if x * len(values) > total)",
            "if not values:\n    return 0\ntotal = sum(values)\n"
            "return sum(1 for x in values if x * len(values) >= total)",
            "if not values:\n    return 0\naverage = sum(values) / len(values)\n"
            "return len([x for x in values if x > average])",
            [([[1, 2, 3]], 1), ([[2, 2]], 0), ([[]], 0)],
            grid(small_lists),
            "All lists over {-2,-1,0,1,2} of length 0..3.",
        ),
        task(
            "zip-to-dict",
            "collections",
            "missing_condition",
            "Return a dict pairing keys[i] with values[i]. If a key repeats, keep its FIRST "
            "value. Extra items in the longer list are ignored.",
            "keys, values",
            "result = {}\nfor key, value in zip(keys, values):\n    if key not in result:\n"
            "        result[key] = value\nreturn result",
            "return dict(zip(keys, values))",
            "result = {}\nfor i in range(min(len(keys), len(values))):\n"
            "    result.setdefault(keys[i], values[i])\nreturn result",
            [
                ([["a", "b"], [1, 2]], {"a": 1, "b": 2}),
                ([["a", "a"], [1, 2]], {"a": 1}),
                ([["x"], []], {}),
            ],
            grid(sequences(["a", "b"]), sequences([1, 2], 2)),
            "All key lists over {'a','b'} of length 0..3 with value lists over {1,2} of length 0..2.",
        ),
        task(
            "matrix-diagonal",
            "collections",
            "off_by_one",
            "Return the main diagonal of the square integer matrix rows: [rows[0][0], "
            "rows[1][1], ...]. An empty matrix returns [].",
            "rows",
            "return [rows[i][i] for i in range(len(rows))]",
            "return [rows[i][i] for i in range(len(rows) - 1)]",
            "result = []\nfor index, row in enumerate(rows):\n    result.append(row[index])\n"
            "return result",
            [([[[1, 2], [3, 4]]], [1, 4]), ([[]], []), ([[[7]]], [7])],
            grid(squares),
            "Empty, two 1x1, all 81 2x2 over {0,1,2}, and eight 3x3 diagonal matrices.",
        ),
        task(
            "most-frequent",
            "collections",
            "update_logic",
            "Return the integer that appears most often in values, breaking ties in favor of "
            "the one that appears first. Return None for an empty list.",
            "values",
            "if not values:\n    return None\nreturn max(values, key=values.count)",
            "if not values:\n    return None\nreturn max(set(values), key=values.count)",
            "counts = {}\nfor x in values:\n    counts[x] = counts.get(x, 0) + 1\nbest = None\n"
            "for x in values:\n    if best is None or counts[x] > counts[best]:\n        best = x\n"
            "return best",
            [([[2, 1]], 2), ([[1, 2, 2]], 2), ([[]], None)],
            grid(sequences([-1, 0, 1, 2])),
            "All lists over {-1,0,1,2} of length 0..3.",
        ),
        task(
            "run-length-encode",
            "collections",
            "off_by_one",
            "Run-length encode values: return [item, count] for each run of equal "
            "neighboring integers, in order.",
            "values",
            "runs = []\nfor x in values:\n    if runs and runs[-1][0] == x:\n"
            "        runs[-1][1] += 1\n    else:\n        runs.append([x, 1])\nreturn runs",
            "runs = []\nfor x in values:\n    if runs and runs[-1][0] == x:\n"
            "        runs[-1][1] += 1\n    else:\n        runs.append([x, 0])\nreturn runs",
            "runs = []\ni = 0\nwhile i < len(values):\n    j = i\n"
            "    while j < len(values) and values[j] == values[i]:\n        j += 1\n"
            "    runs.append([values[i], j - i])\n    i = j\nreturn runs",
            [([[1, 1, 2]], [[1, 2], [2, 1]]), ([[]], []), ([[5]], [[5, 1]])],
            grid(sequences([0, 1, 2], 4)),
            "All lists over {0,1,2} of length 0..4.",
        ),
        task(
            "clip-negatives",
            "collections",
            "update_logic",
            "Return a copy of values with every negative integer replaced by 0. Do not "
            "mutate values.",
            "values",
            "return [x if x > 0 else 0 for x in values]",
            "for i, x in enumerate(values):\n    if x < 0:\n        values[i] = 0\nreturn values",
            "result = []\nfor x in values:\n    result.append(max(x, 0))\nreturn result",
            [([[-1, 2]], [0, 2]), ([[]], []), ([[3, 0]], [3, 0])],
            grid(small_lists),
            "All lists over {-2,-1,0,1,2} of length 0..3.",
        ),
        task(
            "index-of-max",
            "collections",
            "comparison",
            "Return the index of the largest integer in values, choosing the LAST such index "
            "on ties. Return -1 for an empty list.",
            "values",
            "best = -1\nfor i, x in enumerate(values):\n"
            "    if best == -1 or x >= values[best]:\n        best = i\nreturn best",
            "best = -1\nfor i, x in enumerate(values):\n"
            "    if best == -1 or x > values[best]:\n        best = i\nreturn best",
            "if not values:\n    return -1\ntop = max(values)\n"
            "return len(values) - 1 - values[::-1].index(top)",
            [([[1, 3, 2]], 1), ([[2, 2]], 1), ([[]], -1)],
            grid(sequences([-1, 0, 1, 2])),
            "All lists over {-1,0,1,2} of length 0..3.",
        ),
        task(
            "letter-grade",
            "control_flow",
            "comparison",
            "Return the letter grade for an integer score from 0 through 100: 'A' for 90 and "
            "above, 'B' for 80-89, 'C' for 70-79, 'D' for 60-69, otherwise 'F'.",
            "score",
            'if score >= 90:\n    return "A"\nif score >= 80:\n    return "B"\n'
            'if score >= 70:\n    return "C"\nif score >= 60:\n    return "D"\nreturn "F"',
            'if score >= 90:\n    return "A"\nif score > 80:\n    return "B"\n'
            'if score >= 70:\n    return "C"\nif score >= 60:\n    return "D"\nreturn "F"',
            'for floor, letter in ((90, "A"), (80, "B"), (70, "C"), (60, "D")):\n'
            '    if score >= floor:\n        return letter\nreturn "F"',
            [([90], "A"), ([80], "B"), ([59], "F")],
            grid([0, 59, 60, 61, 69, 70, 71, 79, 81, 89, 91, 99, 100]),
            "Thirteen scores around every grade boundary.",
        ),
        task(
            "countdown",
            "control_flow",
            "off_by_one",
            "Return the list [n, n - 1, ..., 1] for integer n >= 0; n = 0 returns [].",
            "n",
            "return list(range(n, 0, -1))",
            "return list(range(n, 1, -1))",
            "result = []\nwhile n > 0:\n    result.append(n)\n    n -= 1\nreturn result",
            [([3], [3, 2, 1]), ([0], []), ([1], [1])],
            grid(range(8)),
            "n in [0,7].",
        ),
        task(
            "sum-of-multiples",
            "control_flow",
            "off_by_one",
            "Return the sum of the positive integers below limit that are multiples of 3 or "
            "5. limit >= 0.",
            "limit",
            "return sum(x for x in range(1, limit) if x % 3 == 0 or x % 5 == 0)",
            "return sum(x for x in range(1, limit + 1) if x % 3 == 0 or x % 5 == 0)",
            "total = 0\nfor x in range(limit):\n    if x % 3 == 0 or x % 5 == 0:\n"
            "        total += x\nreturn total",
            [([10], 23), ([1], 0), ([6], 8)],
            grid(range(31)),
            "limit in [0,30].",
        ),
        task(
            "binary-length",
            "control_flow",
            "off_by_one",
            "Return how many digits the binary representation of integer n >= 0 has. Zero "
            "has one digit.",
            "n",
            "digits = 1\nwhile n >= 2:\n    n //= 2\n    digits += 1\nreturn digits",
            "digits = 1\nwhile n > 2:\n    n //= 2\n    digits += 1\nreturn digits",
            "return max(1, n.bit_length())",
            [([0], 1), ([2], 2), ([5], 3)],
            grid(range(70)),
            "n in [0,69].",
        ),
        task(
            "parking-fee",
            "control_flow",
            "update_logic",
            "Return the parking fee for minutes >= 0 parked: 30 minutes or less is free; "
            "otherwise charge 2 for every started hour of the whole stay, never more than 20.",
            "minutes",
            "if minutes <= 30:\n    return 0\nhours = (minutes + 59) // 60\n"
            "return min(20, 2 * hours)",
            "if minutes <= 30:\n    return 0\nhours = minutes // 60\nreturn min(20, 2 * hours)",
            "if minutes <= 30:\n    return 0\nhours = minutes // 60\nif minutes % 60:\n"
            "    hours += 1\nreturn 2 * hours if hours < 10 else 20",
            [([30], 0), ([31], 2), ([61], 4), ([900], 20)],
            grid([0, 1, 29, 59, 60, 119, 120, 121, 539, 540, 541, 600, 601, 1440]),
            "Fourteen durations around the free period, hour, and cap boundaries.",
        ),
        task(
            "traffic-light",
            "control_flow",
            "missing_condition",
            "Return the next traffic-light state: 'green' becomes 'yellow', 'yellow' becomes "
            "'red', and 'red' becomes 'green'. During an emergency every state becomes 'red'.",
            "state, emergency",
            'if emergency:\n    return "red"\n'
            'return {"green": "yellow", "yellow": "red", "red": "green"}[state]',
            'return {"green": "yellow", "yellow": "red", "red": "green"}[state]',
            'order = ["green", "yellow", "red"]\nif emergency:\n    return "red"\n'
            "return order[(order.index(state) + 1) % 3]",
            [(["green", False], "yellow"), (["red", False], "green"), (["red", True], "red")],
            grid(["green", "yellow", "red"], [False, True]),
            "Every state with and without an emergency.",
        ),
        task(
            "bus-riders",
            "control_flow",
            "update_logic",
            "A bus starts empty. Each stop is [boarding, leaving]: up to leaving riders get "
            "off, never leaving fewer than zero, then boarding riders get on. Return the "
            "largest number on board after any stop, or 0 with no stops.",
            "stops",
            "riders = most = 0\nfor boarding, leaving in stops:\n"
            "    riders = max(0, riders - leaving) + boarding\n    most = max(most, riders)\n"
            "return most",
            "riders = most = 0\nfor boarding, leaving in stops:\n"
            "    riders = max(0, riders - leaving + boarding)\n    most = max(most, riders)\n"
            "return most",
            "riders = 0\nhistory = [0]\nfor stop in stops:\n    riders -= min(riders, stop[1])\n"
            "    riders += stop[0]\n    history.append(riders)\nreturn max(history)",
            [([[[3, 0], [1, 2]]], 3), ([[[2, 3]]], 2), ([[]], 0)],
            grid(sequences([[0, 0], [1, 0], [0, 1], [2, 1], [1, 3]])),
            "All stop lists over five [boarding, leaving] pairs of length 0..3.",
        ),
        task(
            "turn-direction",
            "control_flow",
            "comparison",
            "Headings are integer degrees 0 through 359. Return 'none' if target equals "
            "heading, otherwise 'right' (clockwise) or 'left' for the shorter turn. A turn of "
            "exactly 180 degrees goes 'right'.",
            "heading, target",
            'turn = (target - heading) % 360\nif turn == 0:\n    return "none"\n'
            'return "right" if turn <= 180 else "left"',
            'turn = (target - heading) % 360\nif turn == 0:\n    return "none"\n'
            'return "right" if turn < 180 else "left"',
            'if heading == target:\n    return "none"\nclockwise = (target - heading + 360) % 360\n'
            'counter = 360 - clockwise\nreturn "right" if clockwise <= counter else "left"',
            [([0, 90], "right"), ([0, 180], "right"), ([90, 0], "left"), ([10, 10], "none")],
            grid(headings, headings),
            "heading and target in {0,1,90,179,180,181,270,359}.",
        ),
        task(
            "retry-delays",
            "control_flow",
            "off_by_one",
            "Return the waiting time before each of count retries, where retry k, counting "
            "from 1, waits base * k. count >= 0.",
            "base, count",
            "return [base * k for k in range(1, count + 1)]",
            "return [base * k for k in range(count)]",
            "delays = []\nfor k in range(count):\n    delays.append(base * (k + 1))\nreturn delays",
            [([2, 3], [2, 4, 6]), ([5, 0], []), ([1, 1], [1])],
            grid([0, 1, 3, 10], range(6)),
            "base in {0,1,3,10}, count in [0,5].",
        ),
        task(
            "streak-points",
            "control_flow",
            "missing_condition",
            "days lists booleans, True when a goal was met. Each True day earns 10 points, "
            "plus 5 more when the day before it was also True. The first day has no day before it.",
            "days",
            "points = 0\nfor i, met in enumerate(days):\n    if met:\n        points += 10\n"
            "        if i > 0 and days[i - 1]:\n            points += 5\nreturn points",
            "points = 0\nfor i, met in enumerate(days):\n    if met:\n        points += 10\n"
            "        if days[i - 1]:\n            points += 5\nreturn points",
            "points = 0\nprevious = False\nfor met in days:\n    if met:\n"
            "        points += 15 if previous else 10\n    previous = met\nreturn points",
            [([[True, True]], 25), ([[True, False, True]], 20), ([[]], 0)],
            grid(sequences([False, True], 4)),
            "All boolean lists of length 0..4.",
        ),
    ]
