"""SEALED third final-test tasks. No model has seen these and none may until the opening.

Never export this module's content to a notebook, training bundle, prompt, or artifact.
``admission.reject_final3`` enforces that for model-facing paths; tests pin it. Thirty-two
tasks each of validation, collections, and control flow, with hand-written bugs of
thirteen kinds, none above 20% of the set. Written November 9, 2026 by an agent that saw
no model output of the RL-first phase.
"""

from itertools import combinations_with_replacement, permutations, product

from ..tasks import grid, sequences, task


def _strings(alphabet, max_length):
    return ["".join(xs) for xs in sequences(list(alphabet), max_length)]


def _dicts(keys, values):
    """Every dict over keys where each key is absent or holds one of values."""
    return [
        {k: v for k, v in zip(keys, choice) if v is not None}
        for choice in product([None] + list(values), repeat=len(keys))
    ]


def _bit_grids(shapes, cells):
    return [
        [[cells[(mask >> (r * width + c)) & 1] for c in range(width)] for r in range(height)]
        for height, width in shapes
        for mask in range(1 << (height * width))
    ]


def _validation():
    nested = [0, 1, [], [0], [1], [0, 0], [[]], [[0]], [0, [0]], [[0], 0], [[0, 0]], [0, 0, 0]]
    nested.append([[], []])
    deep = [0, [], [0], [[]], [0, 0], [[0]], [0, [0]], [[], [0]], [[[]]], [[[0]], 0]]
    deep += [[0, [[0, [0]]]], [[[], []], []]]
    clock_days = [[m, d] for m in (1, 6, 12) for d in (1, 15, 28)]
    times = ["0:00", "7:45", "9:05", "09:05", "9:30", "10:15", "12:00", "13:59", "23:59"]
    fields = ["*", "", "0", "5", "7", "12", "23", "59", "60", "0,59", "5,60", "1,,2", "*,5"]
    fields += ["7,7", "a", "1,12"]
    statuses = ["pending", "paid", "shipped", "delivered", "cancelled"]
    emails = ["a@spam.com", "a@SPAM.COM", "a@x.spam.com", "a@notspam.com", "a@spam.com.evil"]
    emails += ["a@b@spam.com", "a@good.org", "a@x.good.org", "a@org"]
    sudoku_rows = [[1, 2, 3, 4], [3, 4, 1, 2], [0, 0, 0, 0], [2, 1, 0, 0], [1, 1, 0, 0]]
    budgets = _dicts(["food", "fun"], [0, 3, 6])
    return [
        task(
            "knight-move",
            "validation",
            "wrong_algorithm",
            "Board squares are given as column and row numbers from 0 through 7. Return True "
            "iff a knight standing on (x1, y1) can reach (x2, y2) in a single move: two "
            "squares along one axis and one square along the other.",
            "x1, y1, x2, y2",
            """
            dx = abs(x1 - x2)
            dy = abs(y1 - y2)
            return sorted([dx, dy]) == [1, 2]
            """,
            """
            return abs(x1 - x2) + abs(y1 - y2) == 3
            """,
            """
            return (x1 - x2) ** 2 + (y1 - y2) ** 2 == 5
            """,
            [
                ([0, 0, 1, 2], True),
                ([3, 4, 3, 7], False),
                ([7, 6, 5, 5], True),
                ([2, 2, 2, 2], False),
            ],
            [
                [x1, y1, x2, y2]
                for x1, y1 in ((0, 0), (3, 4), (7, 6), (2, 1))
                for x2 in range(8)
                for y2 in range(8)
            ],
            "Starting squares (0,0), (3,4), (7,6), and (2,1), each with every target square.",
        ),
        task(
            "subsequence-of",
            "validation",
            "wrong_algorithm",
            "Return True iff the characters of pattern appear in text in the same order, "
            "though not necessarily next to each other, using each character of text at most "
            "once. An empty pattern always qualifies.",
            "pattern, text",
            """
            matched = 0
            for c in text:
                if matched < len(pattern) and c == pattern[matched]:
                    matched += 1
            return matched == len(pattern)
            """,
            """
            positions = [text.find(c) for c in pattern]
            return -1 not in positions and positions == sorted(positions)
            """,
            """
            rest = text
            for c in pattern:
                k = rest.find(c)
                if k < 0:
                    return False
                rest = rest[k + 1 :]
            return True
            """,
            [
                (["ace", "abcde"], True),
                (["aa", "ab"], False),
                (["ba", "aba"], True),
                (["ca", "abc"], False),
            ],
            grid(
                _strings("ab", 3) + ["c", "ac", "ca", "acb"],
                _strings("ab", 5) + ["abc", "cab", "acbc", "bcab"],
            ),
            "Every pattern over {a,b} of length 0..3 plus c, ac, ca, and acb, with every text "
            "over {a,b} of length 0..5 plus abc, cab, acbc, and bcab.",
        ),
        task(
            "one-edit-apart",
            "validation",
            "spec_misreading",
            "Return True iff second can be made from first with exactly one edit: inserting "
            "one character, deleting one character, or replacing one character with a "
            "different one.",
            "first, second",
            """
            if len(first) == len(second):
                return sum(a != b for a, b in zip(first, second)) == 1
            if abs(len(first) - len(second)) != 1:
                return False
            shorter, longer = sorted([first, second], key=len)
            return any(longer[:i] + longer[i + 1 :] == shorter for i in range(len(longer)))
            """,
            """
            if len(first) == len(second):
                return sum(a != b for a, b in zip(first, second)) <= 1
            if abs(len(first) - len(second)) != 1:
                return False
            shorter, longer = sorted([first, second], key=len)
            return any(longer[:i] + longer[i + 1 :] == shorter for i in range(len(longer)))
            """,
            """
            p = 0
            while p < len(first) and p < len(second) and first[p] == second[p]:
                p += 1
            s = 0
            while (
                s < len(first) - p
                and s < len(second) - p
                and first[len(first) - 1 - s] == second[len(second) - 1 - s]
            ):
                s += 1
            left = len(first) - p - s
            right = len(second) - p - s
            return left <= 1 and right <= 1 and left + right > 0
            """,
            [
                (["cat", "cart"], True),
                (["cat", "cat"], False),
                (["cat", "dog"], False),
                (["cat", "cut"], True),
            ],
            grid(_strings("ab", 4), _strings("ab", 4)),
            "Every pair of strings over {a,b} of length 0..4.",
        ),
        task(
            "mountain-shape",
            "validation",
            "missing_edge_case",
            "Return True iff heights strictly rises and then strictly falls: it has at least "
            "three items, its highest point is neither the first nor the last item, and no two "
            "neighbors are equal.",
            "heights",
            """
            i = 1
            while i < len(heights) and heights[i] > heights[i - 1]:
                i += 1
            if i == 1 or i == len(heights):
                return False
            while i < len(heights) and heights[i] < heights[i - 1]:
                i += 1
            return i == len(heights)
            """,
            """
            i = 1
            while i < len(heights) and heights[i] > heights[i - 1]:
                i += 1
            if i == 1:
                return False
            while i < len(heights) and heights[i] < heights[i - 1]:
                i += 1
            return i == len(heights)
            """,
            """
            if len(heights) < 3:
                return False
            peak = heights.index(max(heights))
            rising = all(a < b for a, b in zip(heights[:peak], heights[1 : peak + 1]))
            falling = all(a > b for a, b in zip(heights[peak:], heights[peak + 1 :]))
            return 0 < peak < len(heights) - 1 and rising and falling
            """,
            [
                ([[1, 3, 2]], True),
                ([[1, 2, 3]], False),
                ([[2, 2, 1]], False),
                ([[0, 2, 3, 1, 0]], True),
            ],
            grid(sequences([0, 1, 2, 3], 5)),
            "All lists over {0,1,2,3} of length 0..5.",
        ),
        task(
            "age-on-date",
            "validation",
            "missing_step",
            "birth and today are dates written as [year, month, day]. Return True iff a person "
            "born on birth has had at least their minimum-th birthday by today, counting a "
            "birthday that falls on today.",
            "birth, today, minimum",
            """
            age = today[0] - birth[0]
            if [today[1], today[2]] < [birth[1], birth[2]]:
                age -= 1
            return age >= minimum
            """,
            """
            age = today[0] - birth[0]
            return age >= minimum
            """,
            """
            return [birth[0] + minimum, birth[1], birth[2]] <= today
            """,
            [
                ([[2000, 6, 15], [2018, 6, 15], 18], True),
                ([[2000, 6, 15], [2018, 6, 14], 18], False),
                ([[2000, 12, 1], [2019, 1, 1], 18], True),
                ([[2001, 1, 1], [2018, 12, 31], 18], False),
            ],
            [
                [[2000] + born, [year] + day, minimum]
                for born in clock_days
                for year in (2016, 2017, 2018)
                for day in clock_days
                for minimum in (16, 17, 18)
            ],
            "Births in 2000 and dates in 2016..2018, each with month in {1,6,12} and day in "
            "{1,15,28}; minimum in {16,17,18}.",
        ),
        task(
            "code-window",
            "validation",
            "wrong_algorithm",
            "Time is split into consecutive windows of step seconds: window k runs from "
            "k * step through (k + 1) * step - 1. A login code issued at second issued is "
            "accepted at second now iff now falls in the same window as issued or in the "
            "window right after it. now >= issued >= 0 and step >= 1.",
            "issued, now, step",
            """
            issued_window = issued // step
            current_window = now // step
            return current_window - issued_window <= 1
            """,
            """
            window_end = issued + step
            grace_end = window_end + step
            return now < grace_end
            """,
            """
            deadline = (issued // step + 2) * step
            return now < deadline
            """,
            [([5, 9, 5], True), ([9, 15, 5], False), ([5, 14, 5], True), ([0, 4, 5], True)],
            [
                [issued, now, step]
                for step in (1, 2, 3, 5)
                for issued in range(12)
                for now in range(issued, issued + 12)
            ],
            "step in {1,2,3,5}, issued in [0,11], now from issued through issued + 11.",
        ),
        task(
            "matching-shape",
            "validation",
            "missing_step",
            "left and right are integers or lists nested to any depth. Return True iff they "
            "have the same shape: both are integers, or both are lists of the same length "
            "whose items have the same shape pairwise.",
            "left, right",
            """
            if isinstance(left, list) != isinstance(right, list):
                return False
            if not isinstance(left, list):
                return True
            return len(left) == len(right) and all(solve(a, b) for a, b in zip(left, right))
            """,
            """
            if isinstance(left, list) != isinstance(right, list):
                return False
            if not isinstance(left, list):
                return True
            return all(solve(a, b) for a, b in zip(left, right))
            """,
            """
            def outline(value):
                if isinstance(value, list):
                    return [outline(item) for item in value]
                return 0
            return outline(left) == outline(right)
            """,
            [
                ([[1, [2, 3]], [4, [5, 6]]], True),
                ([[1, 2], [1]], False),
                ([[[1]], [1]], False),
                ([[], []], True),
            ],
            grid(nested, nested),
            "Every pair from 13 listed integers and nested lists of depth at most 2.",
        ),
        task(
            "password-history",
            "validation",
            "spec_misreading",
            "history lists a user's earlier passwords from oldest to newest. Return True iff "
            "password differs from every one of the depth most recent entries of history, "
            "ignoring letter case. depth >= 0.",
            "password, history, depth",
            """
            recent = history[max(0, len(history) - depth) :]
            return all(password.lower() != old.lower() for old in recent)
            """,
            """
            recent = history[:depth]
            return all(password.lower() != old.lower() for old in recent)
            """,
            """
            for i in range(len(history) - 1, len(history) - 1 - depth, -1):
                if i >= 0 and history[i].lower() == password.lower():
                    return False
            return True
            """,
            [
                (["Blue", ["red", "blue", "green"], 2], False),
                (["red", ["red", "blue", "green"], 2], True),
                (["GREEN", ["red", "blue", "green"], 1], False),
                (["red", ["red"], 0], True),
            ],
            grid(["a", "A", "b", "c"], sequences(["a", "b", "B"], 3), range(4)),
            "password in {a,A,b,c}, history any list over {a,b,B} of length 0..3, depth in [0,3].",
        ),
        task(
            "balanced-journal",
            "validation",
            "missing_edge_case",
            "lines is a journal entry: a list of [account, amount] pairs. Return True iff the "
            "entry is valid: it has at least two lines, no amount is 0, and the amounts add up "
            "to 0.",
            "lines",
            """
            amounts = [amount for _, amount in lines]
            return len(amounts) >= 2 and 0 not in amounts and sum(amounts) == 0
            """,
            """
            amounts = [amount for _, amount in lines]
            return 0 not in amounts and sum(amounts) == 0
            """,
            """
            total = 0
            for line in lines:
                if line[1] == 0:
                    return False
                total += line[1]
            return total == 0 and len(lines) > 1
            """,
            [
                ([[["cash", 5], ["sales", -5]]], True),
                ([[]], False),
                ([[["cash", 0], ["sales", 0]]], False),
                ([[["cash", 5], ["sales", -4]]], False),
            ],
            grid(sequences([["cash", -2], ["cash", 1], ["bank", 2], ["bank", -1], ["fees", 0]], 3)),
            "All entries of 0..3 lines from five listed lines with amounts -2, 1, 2, -1, and 0.",
        ),
        task(
            "dependency-order",
            "validation",
            "missing_edge_case",
            "order lists distinct task names in run order. deps is a list of [first, then] "
            "pairs, each meaning first must run before then. Return True iff every pair whose "
            "two tasks both appear in order is respected; pairs that mention a task missing "
            "from order are ignored.",
            "order, deps",
            """
            position = {name: i for i, name in enumerate(order)}
            for first, then in deps:
                if first in position and then in position and position[first] > position[then]:
                    return False
            return True
            """,
            """
            position = {name: i for i, name in enumerate(order)}
            for first, then in deps:
                if position.get(first, -1) > position.get(then, -1):
                    return False
            return True
            """,
            """
            return not any(
                order.index(first) > order.index(then)
                for first, then in deps
                if first in order and then in order
            )
            """,
            [
                ([["a", "b", "c"], [["a", "c"]]], True),
                ([["a", "b"], [["b", "a"]]], False),
                ([["a", "b"], [["a", "z"]]], True),
                ([[], [["a", "b"]]], True),
            ],
            grid(
                [list(p) for n in range(4) for p in permutations("abc", n)],
                sequences([["a", "b"], ["b", "c"], ["c", "a"], ["a", "d"]], 2),
            ),
            "Every ordering of distinct tasks from {a,b,c} of length 0..3, with every list of "
            "0..2 pairs from [a,b], [b,c], [c,a], [a,d].",
        ),
        task(
            "grid-walk-inside",
            "validation",
            "wrong_variable",
            "A robot starts at column 0, row 0 of a grid with columns 0 through width - 1 and "
            "rows 0 through height - 1. Each move is 'U' (row + 1), 'D' (row - 1), 'R' (column "
            "+ 1), or 'L' (column - 1). Return True iff the robot stays on the grid after every "
            "move.",
            "moves, width, height",
            """
            x = y = 0
            for move in moves:
                if move == "L":
                    x -= 1
                elif move == "R":
                    x += 1
                elif move == "D":
                    y -= 1
                else:
                    y += 1
                if not (0 <= x < width and 0 <= y < height):
                    return False
            return True
            """,
            """
            x = y = 0
            for move in moves:
                if move == "L":
                    x -= 1
                elif move == "R":
                    x += 1
                elif move == "D":
                    y -= 1
                else:
                    y += 1
                if not (0 <= x < height and 0 <= y < width):
                    return False
            return True
            """,
            """
            steps = {"L": [-1, 0], "R": [1, 0], "D": [0, -1], "U": [0, 1]}
            column, row = 0, 0
            for move in moves:
                column += steps[move][0]
                row += steps[move][1]
                if column < 0 or row < 0 or column >= width or row >= height:
                    return False
            return True
            """,
            [
                (["RRU", 3, 2], True),
                (["RRR", 3, 2], False),
                (["UU", 3, 2], False),
                (["", 1, 1], True),
            ],
            grid(_strings("UDLR", 3), [1, 2, 3], [1, 2, 3]),
            "Every move string over {U,D,L,R} of length 0..3, width and height in [1,3].",
        ),
        task(
            "turn-counts",
            "validation",
            "spec_misreading",
            "board is a string of marks from a game in which X moves first and two players "
            "alternate placing marks: 'X', 'O', or '.' for an empty cell. Return True iff the "
            "numbers of X and O marks could occur in such a game: X has as many marks as O or "
            "exactly one more.",
            "board",
            """
            difference = board.count("X") - board.count("O")
            return difference == 0 or difference == 1
            """,
            """
            return abs(board.count("X") - board.count("O")) <= 1
            """,
            """
            balance = 0
            for cell in board:
                if cell == "X":
                    balance += 1
                elif cell == "O":
                    balance -= 1
            return 0 <= balance <= 1
            """,
            [(["XO.X"], True), (["OO.X"], False), (["XOO"], False), ([".."], True)],
            grid(_strings("XO.", 5)),
            "All strings over {X,O,.} of length 0..5.",
        ),
        task(
            "page-exists",
            "validation",
            "boundary",
            "Pages of per_page items are numbered from 1. A collection of total items has "
            "total / per_page pages rounded up, or a single page when it is empty. Return True "
            "iff page is an existing page number. per_page >= 1 and total >= 0.",
            "page, per_page, total",
            """
            pages = max(1, (total + per_page - 1) // per_page)
            return 1 <= page <= pages
            """,
            """
            pages = max(1, total // per_page)
            return 1 <= page <= pages
            """,
            """
            if page < 1:
                return False
            return page == 1 or (page - 1) * per_page < total
            """,
            [([3, 10, 25], True), ([4, 10, 25], False), ([1, 10, 0], True), ([0, 10, 5], False)],
            grid(range(-1, 7), [1, 3, 5, 10], [0, 1, 5, 9, 10, 11, 25]),
            "page in [-1,6], per_page in {1,3,5,10}, total in {0,1,5,9,10,11,25}.",
        ),
        task(
            "nesting-depth",
            "validation",
            "wrong_init",
            "value is an integer or a list nested to any depth. Its depth is 0 for an integer, "
            "and for a list it is 1 plus the largest depth among its items, so an empty list "
            "has depth 1. Return True iff the depth of value is at most limit.",
            "value, limit",
            """
            def depth(item):
                if not isinstance(item, list):
                    return 0
                return 1 + max([depth(x) for x in item], default=0)
            return depth(value) <= limit
            """,
            """
            deepest = 0
            stack = [[value, 0]]
            while stack:
                item, level = stack.pop()
                if isinstance(item, list):
                    deepest = max(deepest, level)
                    stack.extend([x, level + 1] for x in item)
            return deepest <= limit
            """,
            """
            level = 0
            frontier = [value]
            while any(isinstance(item, list) for item in frontier):
                level += 1
                frontier = [x for item in frontier if isinstance(item, list) for x in item]
            return level <= limit
            """,
            [([[1, [2]], 2], True), ([[1, [2]], 1], False), ([5, 0], True), ([[], 0], False)],
            grid(deep, range(4)),
            "12 listed integers and nested lists of depth 0 through 4, with limit in [0,3].",
        ),
        task(
            "time-before",
            "validation",
            "wrong_algorithm",
            "start and end are clock times written as hours, a colon, and two-digit minutes, "
            "with or without a leading zero on the hour, such as '9:05' or '09:05'. Return True "
            "iff start is strictly earlier in the day than end.",
            "start, end",
            """
            def minutes(text):
                hours, mins = text.split(":")
                return int(hours) * 60 + int(mins)
            return minutes(start) < minutes(end)
            """,
            """
            start_hour, start_minute = start.split(":")
            end_hour, end_minute = end.split(":")
            if start_hour != end_hour:
                return start_hour < end_hour
            return int(start_minute) < int(end_minute)
            """,
            """
            start_hour, start_minute = map(int, start.split(":"))
            end_hour, end_minute = map(int, end.split(":"))
            return [start_hour, start_minute] < [end_hour, end_minute]
            """,
            [
                (["9:30", "10:15"], True),
                (["10:15", "9:30"], False),
                (["8:05", "8:50"], True),
                (["09:30", "9:30"], False),
            ],
            grid(times, times),
            "Every pair from nine listed times between 0:00 and 23:59, padded and unpadded.",
        ),
        task(
            "even-shares",
            "validation",
            "missing_edge_case",
            "shares lists how many items each person received. Return True iff the items were "
            "split as evenly as possible: the shares add up to total and no two shares differ "
            "by more than 1. With no people, only a total of 0 qualifies.",
            "shares, total",
            """
            if not shares:
                return total == 0
            return sum(shares) == total and max(shares) - min(shares) <= 1
            """,
            """
            return sum(shares) == total and max(shares) - min(shares) <= 1
            """,
            """
            if len(shares) == 0:
                return total == 0
            low = total // len(shares)
            return sum(shares) == total and all(s == low or s == low + 1 for s in shares)
            """,
            [
                ([[3, 4, 3], 10], True),
                ([[2, 4, 4], 10], False),
                ([[], 0], True),
                ([[5, 5], 11], False),
            ],
            [
                [shares, sum(shares) + extra]
                for extra in (0, 1)
                for shares in sequences([1, 2, 3], 4)
            ],
            "All share lists over {1,2,3} of length 0..4, each with total equal to its sum and "
            "to its sum plus 1.",
        ),
        task(
            "parcel-fits",
            "validation",
            "spec_misreading",
            "box and slot each list three side lengths. The box may be turned so that any of "
            "its sides lines up with any side of the slot, keeping its edges parallel to the "
            "slot's. Return True iff the box fits: in some orientation each box side is at most "
            "the slot side it lines up with.",
            "box, slot",
            """
            small = sorted(box)
            large = sorted(slot)
            for i in range(3):
                if small[i] > large[i]:
                    return False
            return True
            """,
            """
            for i in range(3):
                if box[i] > slot[i]:
                    return False
            return True
            """,
            """
            pairs = zip(sorted(box), sorted(slot))
            return not any(side > room for side, room in pairs)
            """,
            [
                ([[2, 3, 1], [3, 2, 1]], True),
                ([[4, 1, 1], [3, 3, 3]], False),
                ([[1, 1, 1], [1, 1, 1]], True),
                ([[1, 3, 3], [2, 2, 3]], False),
            ],
            grid(
                [list(s) for s in product([1, 2, 3], repeat=3)],
                [list(s) for s in product([1, 2, 3], repeat=3)],
            ),
            "Every box and slot with sides in {1,2,3}.",
        ),
        task(
            "cron-field",
            "validation",
            "boundary",
            "field is one field of a schedule: either '*' or a comma-separated list of numbers "
            "written in ASCII digits. Return True iff field is '*' or every listed number is "
            "from low through high inclusive. An empty item, as in '1,,2', makes it invalid.",
            "field, low, high",
            """
            if field == "*":
                return True
            for part in field.split(","):
                if not part or any(c not in "0123456789" for c in part):
                    return False
                if not low <= int(part) <= high:
                    return False
            return True
            """,
            """
            if field == "*":
                return True
            for part in field.split(","):
                if not part or any(c not in "0123456789" for c in part):
                    return False
                if not low <= int(part) < high:
                    return False
            return True
            """,
            """
            if field == "*":
                return True
            numbers = []
            for part in field.split(","):
                if part == "" or not all("0" <= c <= "9" for c in part):
                    return False
                numbers.append(int(part))
            return min(numbers) >= low and max(numbers) <= high
            """,
            [
                (["*", 0, 59], True),
                (["0,30,59", 0, 59], True),
                (["1,,2", 0, 59], False),
                (["60", 0, 59], False),
            ],
            [[f, low, high] for f in fields for low, high in ((0, 59), (0, 23), (1, 12), (5, 7))],
            "16 listed fields, each with bounds (0,59), (0,23), (1,12), or (5,7).",
        ),
        task(
            "order-transition",
            "validation",
            "spec_misreading",
            "An order moves through the statuses 'pending', 'paid', 'shipped', and "
            "'delivered', one step at a time in that order. An order that is not yet delivered "
            "may instead become 'cancelled', and a cancelled order cannot change. Return True "
            "iff changing the status from current to target is allowed.",
            "current, target",
            """
            steps = ["pending", "paid", "shipped", "delivered"]
            if current not in steps[:3]:
                return False
            if target == "cancelled":
                return True
            return target in steps and steps.index(target) == steps.index(current) + 1
            """,
            """
            steps = ["pending", "paid", "shipped", "delivered"]
            if target == "cancelled":
                return current != "cancelled"
            if current not in steps[:3]:
                return False
            return target in steps and steps.index(target) == steps.index(current) + 1
            """,
            """
            allowed = {
                "pending": ["paid", "cancelled"],
                "paid": ["shipped", "cancelled"],
                "shipped": ["delivered", "cancelled"],
            }
            return target in allowed.get(current, [])
            """,
            [
                (["paid", "shipped"], True),
                (["delivered", "cancelled"], False),
                (["pending", "shipped"], False),
                (["shipped", "cancelled"], True),
            ],
            grid(statuses, statuses),
            "Every pair of the five statuses.",
        ),
        task(
            "feature-flags-mask",
            "validation",
            "wrong_operation",
            "flags, required, and forbidden are bit masks given as nonnegative integers. Return "
            "True iff every bit set in required is also set in flags and no bit set in "
            "forbidden is set in flags.",
            "flags, required, forbidden",
            """
            has_required = (flags & required) == required
            has_forbidden = (flags & forbidden) != 0
            return has_required and not has_forbidden
            """,
            """
            has_required = (flags & required) != 0
            has_forbidden = (flags & forbidden) != 0
            return has_required and not has_forbidden
            """,
            """
            for bit in range(8):
                mask = 1 << bit
                if required & mask and not flags & mask:
                    return False
                if forbidden & mask and flags & mask:
                    return False
            return True
            """,
            [([11, 3, 4], True), ([9, 3, 4], False), ([7, 1, 4], False), ([0, 0, 0], True)],
            grid(range(16), [0, 1, 3, 6], [0, 4, 8, 12]),
            "flags in [0,15], required in {0,1,3,6}, forbidden in {0,4,8,12}.",
        ),
        task(
            "price-tiers",
            "validation",
            "spec_misreading",
            "tiers is a price table: a list of [minimum_quantity, unit_price] rows. Return True "
            "iff the table is well formed: it is nonempty, the first row's minimum quantity is "
            "1, minimum quantities strictly increase from row to row, and unit prices never "
            "increase from row to row.",
            "tiers",
            """
            if not tiers or tiers[0][0] != 1:
                return False
            for previous, row in zip(tiers, tiers[1:]):
                if row[0] <= previous[0] or row[1] > previous[1]:
                    return False
            return True
            """,
            """
            if not tiers or tiers[0][0] != 1:
                return False
            for previous, row in zip(tiers, tiers[1:]):
                if row[0] <= previous[0] or row[1] >= previous[1]:
                    return False
            return True
            """,
            """
            quantities = [row[0] for row in tiers]
            prices = [row[1] for row in tiers]
            pairs = range(len(tiers) - 1)
            return (
                len(tiers) > 0
                and quantities[0] == 1
                and all(quantities[i] < quantities[i + 1] for i in pairs)
                and all(prices[i] >= prices[i + 1] for i in pairs)
            )
            """,
            [
                ([[[1, 10], [5, 8], [10, 8]]], True),
                ([[[1, 10], [5, 12]]], False),
                ([[[2, 10]]], False),
                ([[]], False),
            ],
            grid(sequences([[1, 10], [1, 8], [5, 8], [5, 10], [10, 8], [10, 6]], 3)),
            "All tables of 0..3 rows from six listed rows.",
        ),
        task(
            "unbroken-days",
            "validation",
            "spec_misreading",
            "days lists booked day numbers in any order. Return True iff no day is booked "
            "twice and the days form one unbroken run of consecutive integers. An empty list "
            "qualifies.",
            "days",
            """
            seen = set()
            for day in days:
                if day in seen:
                    return False
                seen.add(day)
            return not days or max(days) - min(days) + 1 == len(days)
            """,
            """
            return not days or max(days) - min(days) + 1 == len(days)
            """,
            """
            if not days:
                return True
            start = min(days)
            return sorted(days) == list(range(start, start + len(days)))
            """,
            [([[3, 1, 2]], True), ([[1, 1, 3]], False), ([[1, 3]], False), ([[]], True)],
            grid(sequences([0, 1, 2, 3, 4], 4)),
            "All lists over {0,1,2,3,4} of length 0..4.",
        ),
        task(
            "title-words",
            "validation",
            "wrong_algorithm",
            "Return True iff text is one or more words separated by single spaces, where every "
            "word is an uppercase ASCII letter followed by zero or more lowercase ASCII "
            "letters, as in 'Hello World'.",
            "text",
            """
            for word in text.split(" "):
                if not word or not "A" <= word[0] <= "Z":
                    return False
                if any(not "a" <= c <= "z" for c in word[1:]):
                    return False
            return True
            """,
            """
            words = text.split(" ")
            return all(word.istitle() for word in words)
            """,
            """
            if not text:
                return False
            upper = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            lower = "abcdefghijklmnopqrstuvwxyz"
            previous = " "
            for c in text:
                if previous == " ":
                    if c not in upper:
                        return False
                elif c != " " and c not in lower:
                    return False
                previous = c
            return previous != " "
            """,
            [
                (["Hello World"], True),
                (["Hello-World"], False),
                (["hello World"], False),
                (["Hi  There"], False),
            ],
            grid([" ".join(ws) for ws in sequences(["Ab", "A", "Bab", "ab", "A-B", ""], 3)]),
            "Every string made by joining 0..3 words from {Ab, A, Bab, ab, A-B, ''} with single "
            "spaces, so empty words give extra or edge spaces.",
        ),
        task(
            "cli-flag-rules",
            "validation",
            "inverted_logic",
            "args lists command-line flags. Return True iff the combination is allowed: "
            "'--quiet' and '--verbose' never appear together, and '--force' appears only when "
            "'--yes' also appears.",
            "args",
            """
            if "--quiet" in args and "--verbose" in args:
                return False
            if "--force" in args and "--yes" not in args:
                return False
            return True
            """,
            """
            if "--quiet" in args and "--verbose" in args:
                return False
            if "--force" in args and "--yes" in args:
                return False
            return True
            """,
            """
            flags = set(args)
            if {"--quiet", "--verbose"} <= flags:
                return False
            return "--yes" in flags or "--force" not in flags
            """,
            [
                ([["--force", "--yes"]], True),
                ([["--force"]], False),
                ([["--quiet", "--verbose"]], False),
                ([[]], True),
            ],
            grid(sequences(["--quiet", "--verbose", "--force", "--yes", "-n"], 3)),
            "All lists of length 0..3 over five flags.",
        ),
        task(
            "blocked-domain",
            "validation",
            "wrong_algorithm",
            "email contains at least one '@', and its domain is everything after the last '@'. "
            "Return True iff the domain is not blocked: it does not equal any entry of blocked "
            "and is not a subdomain of one, meaning it does not end with '.' followed by a "
            "blocked entry. Comparisons ignore letter case.",
            "email, blocked",
            """
            domain = email[email.rindex("@") + 1 :].lower()
            for entry in blocked:
                entry = entry.lower()
                if domain == entry or domain.endswith("." + entry):
                    return False
            return True
            """,
            """
            domain = email[email.rindex("@") + 1 :].lower()
            for entry in blocked:
                if domain.endswith(entry.lower()):
                    return False
            return True
            """,
            """
            labels = email.rsplit("@", 1)[1].lower().split(".")
            suffixes = {".".join(labels[i:]) for i in range(len(labels))}
            return not suffixes & {entry.lower() for entry in blocked}
            """,
            [
                (["ann@mail.spam.com", ["spam.com"]], False),
                (["bo@notspam.com", ["spam.com"]], True),
                (["cy@SPAM.com", ["spam.com"]], False),
                (["di@good.org", []], True),
            ],
            grid(emails, sequences(["spam.com", "Good.org", "org"], 2)),
            "Nine listed emails with every blocked list of 0..2 entries from {spam.com, "
            "Good.org, org}.",
        ),
        task(
            "rectangular-grid",
            "validation",
            "wrong_scope",
            "Return True iff rows is a nonempty list of nonempty lists that all have the same "
            "length.",
            "rows",
            """
            if not rows or not rows[0]:
                return False
            width = len(rows[0])
            for row in rows[1:]:
                if len(row) != width:
                    return False
            return True
            """,
            """
            if not rows or not rows[0]:
                return False
            width = len(rows[0])
            for row in rows[1:]:
                if len(row) != width:
                    return False
                return True
            return True
            """,
            """
            widths = {len(row) for row in rows}
            return len(widths) == 1 and 0 not in widths
            """,
            [
                ([[[1, 2], [3, 4]]], True),
                ([[[1, 2], [3, 4], [5]]], False),
                ([[]], False),
                ([[[]]], False),
            ],
            grid(sequences([[], [0], [1], [0, 1], [1, 1, 0]], 3)),
            "All lists of 0..3 rows from [], [0], [1], [0,1], and [1,1,0].",
        ),
        task(
            "mini-sudoku",
            "validation",
            "wrong_scope",
            "grid is a 4 by 4 puzzle: a list of 4 rows of 4 integers, where 0 marks an empty "
            "cell and 1 through 4 are filled in. Return True iff no nonzero number appears "
            "twice in any row or in any column.",
            "grid",
            """
            lines = [list(row) for row in grid] + [[row[c] for row in grid] for c in range(4)]
            for line in lines:
                seen = set()
                for value in line:
                    if value and value in seen:
                        return False
                    seen.add(value)
            return True
            """,
            """
            lines = [list(row) for row in grid] + [[row[c] for row in grid] for c in range(4)]
            seen = set()
            for line in lines:
                for value in line:
                    if value and value in seen:
                        return False
                    seen.add(value)
            return True
            """,
            """
            for i in range(4):
                row = [v for v in grid[i] if v != 0]
                column = [grid[r][i] for r in range(4) if grid[r][i] != 0]
                if len(set(row)) < len(row) or len(set(column)) < len(column):
                    return False
            return True
            """,
            [
                ([[[1, 2, 3, 4], [3, 4, 1, 2], [0, 0, 0, 0], [0, 0, 0, 0]]], True),
                ([[[1, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], False),
                ([[[1, 2, 3, 4], [1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]], False),
            ],
            grid([[list(r) for r in rows] for rows in product(sudoku_rows, repeat=4)]),
            "Every grid whose four rows are chosen, with repeats, from five listed rows.",
        ),
        task(
            "upload-problem",
            "validation",
            "statement_order",
            "Return the first problem with an upload, checking in this order: 'empty' if size "
            "is 0, 'too large' if size is more than limit, and 'no extension' if filename has "
            "no '.' after its first character. Return None if there is no problem.",
            "filename, size, limit",
            """
            if size == 0:
                return "empty"
            if size > limit:
                return "too large"
            if "." not in filename[1:]:
                return "no extension"
            return None
            """,
            """
            if size == 0:
                return "empty"
            if "." not in filename[1:]:
                return "no extension"
            if size > limit:
                return "too large"
            return None
            """,
            """
            checks = [
                ["empty", size == 0],
                ["too large", size > limit],
                ["no extension", filename.find(".", 1) == -1],
            ]
            for message, failed in checks:
                if failed:
                    return message
            return None
            """,
            [
                (["photo", 900, 500], "too large"),
                (["a.txt", 0, 500], "empty"),
                ([".env", 10, 500], "no extension"),
                (["a.txt", 10, 500], None),
            ],
            grid(
                ["a.txt", ".bashrc", "notes", "x.", "", "archive.tar.gz", "."],
                [0, 1, 500, 501, 900],
                [0, 500],
            ),
            "Seven listed filenames, size in {0,1,500,501,900}, limit in {0,500}.",
        ),
        task(
            "merge-ready",
            "validation",
            "inverted_logic",
            "Return True iff a pull request can be merged: it is not a draft, its checks "
            "passed, and it has at least required approvals.",
            "approvals, required, checks_passed, is_draft",
            """
            blocked = is_draft or not checks_passed
            enough = approvals >= required
            return enough and not blocked
            """,
            """
            blocked = is_draft and not checks_passed
            enough = approvals >= required
            return enough and not blocked
            """,
            """
            if is_draft:
                return False
            if not checks_passed:
                return False
            missing = required - approvals
            return missing <= 0
            """,
            [
                ([2, 2, True, False], True),
                ([3, 2, True, True], False),
                ([5, 1, False, False], False),
                ([0, 1, True, False], False),
            ],
            grid(range(4), range(4), [False, True], [False, True]),
            "approvals and required in [0,3], every combination of the two booleans.",
        ),
        task(
            "category-budgets",
            "validation",
            "wrong_variable",
            "spent and budget map category names to amounts. Return True iff no category in "
            "spent is over budget, where a category missing from budget has a budget of 0.",
            "spent, budget",
            """
            for category, amount in spent.items():
                if amount > budget.get(category, 0):
                    return False
            return True
            """,
            """
            for category in budget:
                if spent.get(category, 0) > budget[category]:
                    return False
            return True
            """,
            """
            return all(spent[c] <= (budget[c] if c in budget else 0) for c in spent)
            """,
            [
                ([{"food": 5}, {"food": 10}], True),
                ([{"food": 5, "fun": 1}, {"food": 10}], False),
                ([{"food": 12}, {"food": 10}], False),
                ([{}, {}], True),
            ],
            grid(budgets, budgets),
            "spent and budget each any dict with keys from {food, fun} and values in {0,3,6}.",
        ),
        task(
            "ship-placement",
            "validation",
            "boundary",
            "A ship length cells long is placed on a size by size board starting at row, col "
            "and extending to the right when horizontal is True, otherwise downward. Rows and "
            "columns are numbered from 0. Return True iff every cell of the ship is on the "
            "board.",
            "size, row, col, length, horizontal",
            """
            end_row = row if horizontal else row + length - 1
            end_col = col + length - 1 if horizontal else col
            return 0 <= row and 0 <= col and end_row < size and end_col < size
            """,
            """
            end_row = row if horizontal else row + length
            end_col = col + length if horizontal else col
            return 0 <= row and 0 <= col and end_row < size and end_col < size
            """,
            """
            if row < 0 or col < 0:
                return False
            if horizontal:
                return row < size and col + length <= size
            return col < size and row + length <= size
            """,
            [
                ([5, 0, 3, 2, True], True),
                ([5, 4, 0, 2, False], False),
                ([3, 0, 0, 3, False], True),
                ([5, -1, 0, 1, True], False),
            ],
            grid([3, 5], range(-1, 6), range(-1, 6), [1, 2, 3], [False, True]),
            "size in {3,5}, row and col in [-1,5], length in [1,3], both orientations.",
        ),
        task(
            "no-trailing-space",
            "validation",
            "wrong_operation",
            "Return True iff no string in lines ends with a space or a tab. Leading spaces and "
            "tabs are allowed.",
            "lines",
            r"""
            for line in lines:
                if line and line[-1] in " \t":
                    return False
            return True
            """,
            r"""
            return all(line.strip(" \t") == line for line in lines)
            """,
            r"""
            return not any(line.endswith((" ", "\t")) for line in lines)
            """,
            [
                ([["  indented", "ok"]], True),
                ([["x ", "y"]], False),
                ([["a\t"]], False),
                ([[]], True),
            ],
            grid(sequences(["", "a", " a", "a ", "\ta", "a\t", " "], 2)),
            "All lists of 0..2 lines from seven listed lines with and without leading or "
            "trailing whitespace.",
        ),
    ]


def _collections():
    counters = _dicts(["a", "b"], [0, 1, 2])
    stocks = _dicts(["a", "b"], [0, 1, 3])
    flat_options = [None, 1, {}, {"c": 2}, {"c": {"d": 3}}, {"c": 4, "d": {"e": 5}}]
    flat_inputs = [
        {k: v for k, v in zip("ab", choice) if v is not None}
        for choice in product(flat_options, repeat=2)
    ]
    dig_data = [
        {},
        {"a": 1},
        {"a": {"b": 2}},
        {"a": {"b": {"a": 3}}, "b": 0},
        {"b": {"a": None}},
        {"a": {"a": {"a": 4}}},
    ]
    spiral = [
        [[start + r * cols + c for c in range(cols)] for r in range(height)]
        for height in range(1, 5)
        for cols in range(1, 5)
        for start in (0, 10)
    ] + [[], [[]], [[], []]]
    cells = [
        [[0, 0], [0, 0]],
        [[1, 2, 3]],
        [[1], [2], [3]],
        [[7]],
        [[1, 2], [3, 4], [5, 6]],
        [[0, 1, 0], [1, 0, 1]],
    ]

    def fill(height, width, seed):
        return [[(seed + 2 * r + c) % 4 - 1 for c in range(width)] for r in range(height)]

    products = [
        [fill(n, m, sa), fill(m, p, sb)]
        for n in range(1, 4)
        for m in range(1, 4)
        for p in range(1, 4)
        for sa in (0, 1)
        for sb in (0, 3)
    ]
    return [
        task(
            "group-by-domain",
            "collections",
            "aliasing",
            "Each string in emails contains exactly one '@'. Return a dict mapping each email "
            "domain, the part after '@' in lowercase, to the list of local parts (the parts "
            "before '@') that use it, in order of appearance.",
            "emails",
            """
            groups = {}
            for email in emails:
                local, domain = email.split("@")
                groups.setdefault(domain.lower(), []).append(local)
            return groups
            """,
            """
            domains = [email.split("@")[1].lower() for email in emails]
            groups = dict.fromkeys(domains, [])
            for email, domain in zip(emails, domains):
                groups[domain].append(email.split("@")[0])
            return groups
            """,
            """
            groups = {}
            for email in emails:
                at = email.index("@")
                key = email[at + 1 :].lower()
                if key not in groups:
                    groups[key] = []
                groups[key].append(email[:at])
            return groups
            """,
            [
                (
                    [["ann@x.com", "bob@y.org", "cy@X.com"]],
                    {"x.com": ["ann", "cy"], "y.org": ["bob"]},
                ),
                ([[]], {}),
                ([["dee@z.io"]], {"z.io": ["dee"]}),
            ],
            grid(sequences(["a@x.io", "b@X.io", "c@y.io", "a@y.io"], 3)),
            "All lists of length 0..3 over four listed emails.",
        ),
        task(
            "pair-with-sum",
            "collections",
            "statement_order",
            "Return [i, j] for the pair of positions i < j with values[i] + values[j] == target "
            "that has the smallest j, and among those the smallest i. Return [] if there is no "
            "such pair.",
            "values, target",
            """
            first_seen = {}
            for j, value in enumerate(values):
                if target - value in first_seen:
                    return [first_seen[target - value], j]
                if value not in first_seen:
                    first_seen[value] = j
            return []
            """,
            """
            first_seen = {}
            for j, value in enumerate(values):
                if value not in first_seen:
                    first_seen[value] = j
                if target - value in first_seen:
                    return [first_seen[target - value], j]
            return []
            """,
            """
            for j in range(len(values)):
                for i in range(j):
                    if values[i] + values[j] == target:
                        return [i, j]
            return []
            """,
            [
                ([[2, 7, 11, 15], 9], [0, 1]),
                ([[3, 2, 4], 6], [1, 2]),
                ([[1, 2], 7], []),
                ([[3, 3], 6], [0, 1]),
            ],
            grid(sequences([1, 2, 3], 4), range(2, 7)),
            "All lists over {1,2,3} of length 0..4, target in [2,6].",
        ),
        task(
            "pack-batches",
            "collections",
            "missing_step",
            "Pack items into batches in order: an item joins the current batch if the batch's "
            "total size would stay at most capacity, and otherwise starts a new batch. An item "
            "larger than capacity gets a batch of its own. Return the list of batches.",
            "sizes, capacity",
            """
            batches = []
            current = []
            total = 0
            for size in sizes:
                if current and total + size > capacity:
                    batches.append(current)
                    current, total = [], 0
                current.append(size)
                total += size
            if current:
                batches.append(current)
            return batches
            """,
            """
            batches = []
            current = []
            total = 0
            for size in sizes:
                if current and total + size > capacity:
                    batches.append(current)
                    current, total = [], 0
                current.append(size)
                total += size
            return batches
            """,
            """
            batches = []
            for size in sizes:
                if batches and sum(batches[-1]) + size <= capacity:
                    batches[-1].append(size)
                else:
                    batches.append([size])
            return batches
            """,
            [
                ([[3, 4, 2, 5], 7], [[3, 4], [2, 5]]),
                ([[], 5], []),
                ([[9, 1], 5], [[9], [1]]),
                ([[1, 1, 1], 5], [[1, 1, 1]]),
            ],
            grid(sequences([1, 2, 3, 5], 4), [3, 5]),
            "All size lists over {1,2,3,5} of length 0..4, capacity in {3,5}.",
        ),
        task(
            "competition-rank",
            "collections",
            "spec_misreading",
            "Return the rank of each score, in input order. The highest score has rank 1, "
            "equal scores share a rank, and each score's rank is one more than the number of "
            "scores strictly higher than it, so ranks can skip, as in 1, 2, 2, 4.",
            "scores",
            """
            return [sum(other > score for other in scores) + 1 for score in scores]
            """,
            """
            distinct = sorted(set(scores), reverse=True)
            return [distinct.index(score) + 1 for score in scores]
            """,
            """
            ordered = sorted(scores, reverse=True)
            return [ordered.index(score) + 1 for score in scores]
            """,
            [
                ([[10, 20, 20, 5]], [3, 1, 1, 4]),
                ([[]], []),
                ([[7]], [1]),
                ([[3, 3, 3]], [1, 1, 1]),
            ],
            grid(sequences([0, 1, 2, 3], 4)),
            "All lists over {0,1,2,3} of length 0..4.",
        ),
        task(
            "dig-path",
            "collections",
            "missing_edge_case",
            "data is a dict whose values are integers, None, or more dicts of the same kind. "
            "Follow the keys in path one at a time, starting from data, and return the value "
            "reached. Return None if a key is missing or if a value that is not a dict is "
            "reached before path ends. An empty path returns data.",
            "data, path",
            """
            current = data
            for key in path:
                if not isinstance(current, dict) or key not in current:
                    return None
                current = current[key]
            return current
            """,
            """
            current = data
            for key in path:
                if key not in current:
                    return None
                current = current[key]
            return current
            """,
            """
            if not path:
                return data
            if isinstance(data, dict) and path[0] in data:
                return solve(data[path[0]], path[1:])
            return None
            """,
            [
                ([{"a": {"b": 1}}, ["a", "b"]], 1),
                ([{"a": 1}, ["a", "b"]], None),
                ([{"a": {}}, ["c"]], None),
                ([{"a": 2}, []], {"a": 2}),
            ],
            grid(dig_data, sequences(["a", "b"], 3)),
            "Six listed nested dicts, with path any list over {a,b} of length 0..3.",
        ),
        task(
            "add-counters",
            "collections",
            "aliasing",
            "left and right map strings to integer counts. Return a new dict containing every "
            "key of either one, where a key in both maps to the sum of its two counts. Do not "
            "change left or right.",
            "left, right",
            """
            result = dict(left)
            for key, count in right.items():
                result[key] = result.get(key, 0) + count
            return result
            """,
            """
            result = left
            for key, count in right.items():
                result[key] = result.get(key, 0) + count
            return result
            """,
            """
            keys = list(left) + [k for k in right if k not in left]
            return {k: left.get(k, 0) + right.get(k, 0) for k in keys}
            """,
            [
                ([{"a": 1}, {"a": 2, "b": 3}], {"a": 3, "b": 3}),
                ([{}, {}], {}),
                ([{"x": 4}, {}], {"x": 4}),
            ],
            grid(counters, counters),
            "left and right each any dict with keys from {a,b} and counts in {0,1,2}.",
        ),
        task(
            "replace-first-n",
            "collections",
            "wrong_scope",
            "Return a copy of values in which only the first limit occurrences of old are "
            "replaced by new; later occurrences stay. limit >= 0.",
            "values, old, new, limit",
            """
            result = []
            replaced = 0
            for value in values:
                if value == old and replaced < limit:
                    result.append(new)
                    replaced += 1
                else:
                    result.append(value)
            return result
            """,
            """
            result = []
            replaced = 0
            for value in values:
                if value == old and replaced < limit:
                    result.append(new)
                else:
                    result.append(value)
                replaced += 1
            return result
            """,
            """
            result = list(values)
            remaining = limit
            for i, value in enumerate(result):
                if remaining == 0:
                    break
                if value == old:
                    result[i] = new
                    remaining -= 1
            return result
            """,
            [
                ([[1, 2, 1, 1], 1, 9, 2], [9, 2, 9, 1]),
                ([[1, 1], 1, 0, 0], [1, 1]),
                ([[], 1, 2, 3], []),
                ([[3, 1], 1, 5, 1], [3, 5]),
            ],
            grid(sequences([0, 1, 2], 4), [0, 1], [7], range(4)),
            "All lists over {0,1,2} of length 0..4, old in {0,1}, new 7, limit in [0,3].",
        ),
        task(
            "spiral-order",
            "collections",
            "missing_edge_case",
            "rows is a rectangular matrix given as a list of equal-length rows. Return its "
            "items in clockwise spiral order: along the top row, down the right column, back "
            "along the bottom row, up the left column, and then the same for the part inside. "
            "A matrix with no items returns [].",
            "rows",
            """
            result = []
            top, bottom = 0, len(rows) - 1
            left, right = 0, (len(rows[0]) - 1 if rows else -1)
            while top <= bottom and left <= right:
                for c in range(left, right + 1):
                    result.append(rows[top][c])
                for r in range(top + 1, bottom + 1):
                    result.append(rows[r][right])
                if top < bottom:
                    for c in range(right - 1, left - 1, -1):
                        result.append(rows[bottom][c])
                if left < right:
                    for r in range(bottom - 1, top, -1):
                        result.append(rows[r][left])
                top, bottom, left, right = top + 1, bottom - 1, left + 1, right - 1
            return result
            """,
            """
            result = []
            top, bottom = 0, len(rows) - 1
            left, right = 0, (len(rows[0]) - 1 if rows else -1)
            while top <= bottom and left <= right:
                for c in range(left, right + 1):
                    result.append(rows[top][c])
                for r in range(top + 1, bottom + 1):
                    result.append(rows[r][right])
                for c in range(right - 1, left - 1, -1):
                    result.append(rows[bottom][c])
                for r in range(bottom - 1, top, -1):
                    result.append(rows[r][left])
                top, bottom, left, right = top + 1, bottom - 1, left + 1, right - 1
            return result
            """,
            """
            remaining = [list(row) for row in rows]
            result = []
            while remaining and remaining[0]:
                result.extend(remaining.pop(0))
                remaining = [list(column) for column in zip(*remaining)][::-1]
            return result
            """,
            [
                ([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]], [1, 2, 3, 6, 9, 8, 7, 4, 5]),
                ([[[1, 2, 3]]], [1, 2, 3]),
                ([[]], []),
                ([[[1], [2]]], [1, 2]),
            ],
            grid(spiral),
            "Every matrix of 1..4 rows and 1..4 columns numbered consecutively from 0 or 10, "
            "plus [], [[]], and [[],[]].",
        ),
        task(
            "roster-order",
            "collections",
            "spec_misreading",
            "people is a list of [name, age] pairs. Return them sorted by age from oldest to "
            "youngest, and alphabetically by name among people of the same age.",
            "people",
            """
            return sorted(people, key=lambda person: (-person[1], person[0]))
            """,
            """
            return sorted(people, key=lambda person: (person[1], person[0]), reverse=True)
            """,
            """
            by_name = sorted(people, key=lambda person: person[0])
            return sorted(by_name, key=lambda person: person[1], reverse=True)
            """,
            [
                ([[["ann", 30], ["bob", 25], ["cy", 30]]], [["ann", 30], ["cy", 30], ["bob", 25]]),
                ([[]], []),
                ([[["zed", 1], ["amy", 1]]], [["amy", 1], ["zed", 1]]),
            ],
            grid(sequences([["ann", 3], ["bob", 3], ["cy", 1], ["ann", 1]], 3)),
            "All lists of 0..3 people from four listed [name, age] pairs.",
        ),
        task(
            "stock-changes",
            "collections",
            "missing_edge_case",
            "before and after map item names to stock counts, and a missing item has count 0. "
            "Return a dict mapping every item whose count changed to after minus before. Items "
            "whose count did not change are left out.",
            "before, after",
            """
            changes = {}
            for item in set(before) | set(after):
                delta = after.get(item, 0) - before.get(item, 0)
                if delta:
                    changes[item] = delta
            return changes
            """,
            """
            changes = {}
            for item, count in after.items():
                delta = count - before.get(item, 0)
                if delta:
                    changes[item] = delta
            return changes
            """,
            """
            changes = dict(after)
            for item, count in before.items():
                changes[item] = changes.get(item, 0) - count
            return {item: delta for item, delta in changes.items() if delta != 0}
            """,
            [
                ([{"pen": 2, "ink": 1}, {"pen": 5}], {"pen": 3, "ink": -1}),
                ([{}, {"cup": 1}], {"cup": 1}),
                ([{"a": 1}, {"a": 1}], {}),
            ],
            grid(stocks, stocks),
            "before and after each any dict with keys from {a,b} and counts in {0,1,3}.",
        ),
        task(
            "flatten-keys",
            "collections",
            "wrong_variable",
            "data maps strings to integers or to dicts of the same kind, nested to any depth. "
            "Return a flat dict mapping the dotted path of each integer, such as 'a.b.c', to "
            "that integer. Empty nested dicts contribute nothing.",
            "data",
            """
            flat = {}
            stack = [["", data]]
            while stack:
                prefix, current = stack.pop()
                for key, value in current.items():
                    path = prefix + key
                    if isinstance(value, dict):
                        stack.append([path + ".", value])
                    else:
                        flat[path] = value
            return flat
            """,
            """
            flat = {}
            stack = [["", data]]
            while stack:
                prefix, current = stack.pop()
                for key, value in current.items():
                    path = prefix + key
                    if isinstance(value, dict):
                        stack.append([key + ".", value])
                    else:
                        flat[path] = value
            return flat
            """,
            """
            flat = {}
            for key, value in data.items():
                if isinstance(value, dict):
                    for inner, number in solve(value).items():
                        flat[key + "." + inner] = number
                else:
                    flat[key] = value
            return flat
            """,
            [
                ([{"a": {"b": {"c": 1}}, "d": 2}], {"a.b.c": 1, "d": 2}),
                ([{}], {}),
                ([{"x": {"y": 3}}], {"x.y": 3}),
            ],
            grid(flat_inputs),
            "Every dict whose keys a and b are each absent, an integer, or one of four listed "
            "nested dicts.",
        ),
        task(
            "wrap-words",
            "collections",
            "boundary",
            "Lay out words on lines of at most width characters. Words on a line are separated "
            "by one space; each word goes on the current line if it fits and otherwise starts a "
            "new line, and a word longer than width gets a line of its own. Return the list of "
            "lines.",
            "words, width",
            """
            lines = []
            for word in words:
                if lines and len(lines[-1]) + 1 + len(word) <= width:
                    lines[-1] += " " + word
                else:
                    lines.append(word)
            return lines
            """,
            """
            lines = []
            for word in words:
                if lines and len(lines[-1]) + 1 + len(word) < width:
                    lines[-1] += " " + word
                else:
                    lines.append(word)
            return lines
            """,
            """
            lines = []
            current = ""
            for word in words:
                if current and len(current) + len(word) < width:
                    current += " " + word
                else:
                    if current:
                        lines.append(current)
                    current = word
            if current:
                lines.append(current)
            return lines
            """,
            [
                ([["the", "quick", "fox"], 9], ["the quick", "fox"]),
                ([[], 5], []),
                ([["a", "b", "c"], 3], ["a b", "c"]),
                ([["longword", "x"], 4], ["longword", "x"]),
            ],
            grid(sequences(["a", "bb", "ccc"], 4), [1, 3, 5, 7]),
            "All lists over {a,bb,ccc} of length 0..4, width in {1,3,5,7}.",
        ),
        task(
            "shared-prefix",
            "collections",
            "missing_step",
            "Return the longest string that every string in words starts with. An empty list "
            "returns ''.",
            "words",
            """
            if not words:
                return ""
            prefix = words[0]
            for word in words[1:]:
                while not word.startswith(prefix):
                    prefix = prefix[:-1]
            return prefix
            """,
            """
            result = ""
            for chars in zip(*words):
                if len(set(chars)) == 1:
                    result += chars[0]
            return result
            """,
            """
            result = ""
            for chars in zip(*words):
                if len(set(chars)) != 1:
                    break
                result += chars[0]
            return result
            """,
            [
                ([["flower", "flow", "flight"]], "fl"),
                ([["abc", "axc"]], "a"),
                ([[]], ""),
                ([["solo"]], "solo"),
            ],
            grid(sequences(["ab", "abc", "ba", "a", "", "bab", "bb", "xbc"], 3)),
            "All lists of 0..3 strings from {ab, abc, ba, a, '', bab, bb, xbc}.",
        ),
        task(
            "trim-extremes",
            "collections",
            "aliasing",
            "Remove the k smallest and the k largest items from values and return the items "
            "that remain, sorted in increasing order. If none remain, return []. Do not change "
            "values.",
            "values, k",
            """
            ordered = sorted(values)
            return ordered[k : len(ordered) - k]
            """,
            """
            values.sort()
            return values[k : len(values) - k]
            """,
            """
            remaining = list(values)
            for _ in range(k):
                if len(remaining) < 2:
                    return []
                remaining.remove(min(remaining))
                remaining.remove(max(remaining))
            return sorted(remaining)
            """,
            [([[5, 1, 9, 3], 1], [3, 5]), ([[4, 2], 1], []), ([[2, 7, 1], 0], [1, 2, 7])],
            grid(sequences([0, 1, 2, 3], 4), range(3)),
            "All lists over {0,1,2,3} of length 0..4, k in [0,2].",
        ),
        task(
            "next-greater",
            "collections",
            "wrong_algorithm",
            "values are nonnegative integers. For each item, find the first later item that is "
            "strictly greater than it. Return those items in order, using -1 where there is "
            "none.",
            "values",
            """
            result = [-1] * len(values)
            waiting = []
            for i, value in enumerate(values):
                while waiting and values[waiting[-1]] < value:
                    result[waiting.pop()] = value
                waiting.append(i)
            return result
            """,
            """
            result = []
            for i, value in enumerate(values):
                rest = values[i + 1 :]
                if rest and max(rest) > value:
                    result.append(max(rest))
                else:
                    result.append(-1)
            return result
            """,
            """
            result = []
            for i, value in enumerate(values):
                larger = [v for v in values[i + 1 :] if v > value]
                result.append(larger[0] if larger else -1)
            return result
            """,
            [
                ([[2, 1, 3, 5]], [3, 3, 5, -1]),
                ([[]], []),
                ([[4, 4]], [-1, -1]),
                ([[1, 3, 2, 4]], [3, 4, 4, -1]),
            ],
            grid(sequences([0, 1, 2, 3], 5)),
            "All lists over {0,1,2,3} of length 0..5.",
        ),
        task(
            "neighbor-lists",
            "collections",
            "missing_step",
            "edges lists undirected connections as [a, b] pairs of node names. Return a dict "
            "mapping each node that appears in an edge to the sorted list of its distinct "
            "neighbors. An edge [a, a] makes a its own neighbor.",
            "edges",
            """
            graph = {}
            for a, b in edges:
                graph.setdefault(a, set()).add(b)
                graph.setdefault(b, set()).add(a)
            return {node: sorted(neighbors) for node, neighbors in graph.items()}
            """,
            """
            graph = {}
            for a, b in edges:
                graph.setdefault(a, set()).add(b)
            return {node: sorted(neighbors) for node, neighbors in graph.items()}
            """,
            """
            nodes = {node for edge in edges for node in edge}
            return {
                node: sorted({b for a, b in edges if a == node} | {a for a, b in edges if b == node})
                for node in nodes
            }
            """,
            [
                ([[["a", "b"], ["b", "c"]]], {"a": ["b"], "b": ["a", "c"], "c": ["b"]}),
                ([[]], {}),
                ([[["x", "y"], ["y", "x"]]], {"x": ["y"], "y": ["x"]}),
            ],
            grid(sequences([["a", "b"], ["b", "c"], ["c", "a"], ["a", "a"]], 3)),
            "All lists of 0..3 edges from [a,b], [b,c], [c,a], and [a,a].",
        ),
        task(
            "range-summary",
            "collections",
            "missing_edge_case",
            "values is a list of integers in strictly increasing order. Describe each maximal "
            "run of consecutive integers as a string: 'a' for a run of one number and 'a-b' "
            "for a run from a through b. Return the descriptions in order.",
            "values",
            """
            result = []
            i = 0
            while i < len(values):
                j = i
                while j + 1 < len(values) and values[j + 1] == values[j] + 1:
                    j += 1
                result.append(str(values[i]) if i == j else f"{values[i]}-{values[j]}")
                i = j + 1
            return result
            """,
            """
            result = []
            i = 0
            while i < len(values):
                j = i
                while j + 1 < len(values) and values[j + 1] == values[j] + 1:
                    j += 1
                result.append(f"{values[i]}-{values[j]}")
                i = j + 1
            return result
            """,
            """
            runs = []
            for value in values:
                if runs and runs[-1][1] + 1 == value:
                    runs[-1][1] = value
                else:
                    runs.append([value, value])
            return [str(a) if a == b else str(a) + "-" + str(b) for a, b in runs]
            """,
            [([[0, 1, 2, 4, 6, 7]], ["0-2", "4", "6-7"]), ([[]], []), ([[5]], ["5"])],
            grid([[v for v in range(7) if mask >> v & 1] for mask in range(128)]),
            "Every strictly increasing list of distinct integers from 0 through 6.",
        ),
        task(
            "split-evenly",
            "collections",
            "wrong_algorithm",
            "Split values into n consecutive parts whose lengths differ by at most one, with "
            "the longer parts first, and return the list of parts. Parts are empty when values "
            "has fewer than n items. n >= 1.",
            "values, n",
            """
            size, extra = divmod(len(values), n)
            parts = []
            start = 0
            for i in range(n):
                end = start + size + (1 if i < extra else 0)
                parts.append(values[start:end])
                start = end
            return parts
            """,
            """
            size = -(-len(values) // n)
            return [values[i * size : (i + 1) * size] for i in range(n)]
            """,
            """
            parts = []
            remaining = list(values)
            for left in range(n, 0, -1):
                take = -(-len(remaining) // left)
                parts.append(remaining[:take])
                remaining = remaining[take:]
            return parts
            """,
            [
                ([[1, 2, 3, 4], 3], [[1, 2], [3], [4]]),
                ([[1, 2, 3, 4, 5, 6, 7], 3], [[1, 2, 3], [4, 5], [6, 7]]),
                ([[], 2], [[], []]),
                ([[1, 2], 3], [[1], [2], []]),
            ],
            [[list(range(1, length + 1)), n] for length in range(13) for n in range(1, 6)],
            "values = [1, ..., length] for length in [0,12], n in [1,5].",
        ),
        task(
            "set-cell",
            "collections",
            "aliasing",
            "grid is a list of rows. Return a new grid equal to grid except that the cell at "
            "row, col holds value. Do not change grid or any of its rows.",
            "grid, row, col, value",
            """
            result = [list(line) for line in grid]
            result[row][col] = value
            return result
            """,
            """
            result = list(grid)
            result[row][col] = value
            return result
            """,
            """
            return [
                [value if [r, c] == [row, col] else cell for c, cell in enumerate(line)]
                for r, line in enumerate(grid)
            ]
            """,
            [
                ([[[0, 0], [0, 0]], 1, 0, 5], [[0, 0], [5, 0]]),
                ([[[7]], 0, 0, 7], [[7]]),
                ([[[1, 2, 3]], 0, 2, 0], [[1, 2, 0]]),
            ],
            [
                [g, r, c, v]
                for g in cells
                for r in range(len(g))
                for c in range(len(g[0]))
                for v in (0, 9)
            ],
            "Six listed grids, every cell of each, value in {0,9}.",
        ),
        task(
            "add-padded",
            "collections",
            "wrong_operation",
            "Return the item-by-item sums of left and right. When one list is longer, its extra "
            "items are added to 0, so the result is as long as the longer list.",
            "left, right",
            """
            result = []
            for i in range(max(len(left), len(right))):
                a = left[i] if i < len(left) else 0
                b = right[i] if i < len(right) else 0
                result.append(a + b)
            return result
            """,
            """
            result = []
            for i in range(min(len(left), len(right))):
                a = left[i] if i < len(left) else 0
                b = right[i] if i < len(right) else 0
                result.append(a + b)
            return result
            """,
            """
            longer, shorter = (left, right) if len(left) >= len(right) else (right, left)
            return [x + (shorter[i] if i < len(shorter) else 0) for i, x in enumerate(longer)]
            """,
            [([[1, 2, 3], [10, 20]], [11, 22, 3]), ([[], [4]], [4]), ([[1], [1]], [2])],
            grid(sequences([0, 1, 2], 3), sequences([1, 3], 3)),
            "left any list over {0,1,2} and right any list over {1,3}, each of length 0..3.",
        ),
        task(
            "matrix-product",
            "collections",
            "wrong_variable",
            "a is an n by m matrix and b is an m by p matrix, given as lists of rows, with n, "
            "m, and p at least 1. Return their matrix product: the n by p matrix whose item in "
            "row i, column j is the sum over k of a[i][k] * b[k][j].",
            "a, b",
            """
            inner = len(b)
            columns = len(b[0])
            return [
                [sum(a[i][k] * b[k][j] for k in range(inner)) for j in range(columns)]
                for i in range(len(a))
            ]
            """,
            """
            inner = len(b)
            columns = len(b[0])
            return [
                [sum(a[i][k] * b[k][i] for k in range(inner)) for j in range(columns)]
                for i in range(len(a))
            ]
            """,
            """
            columns = [list(column) for column in zip(*b)]
            return [[sum(x * y for x, y in zip(row, column)) for column in columns] for row in a]
            """,
            [
                ([[[1, 2], [3, 4]], [[5, 6], [7, 8]]], [[19, 22], [43, 50]]),
                ([[[2]], [[3]]], [[6]]),
                ([[[1, 0, 2]], [[1], [2], [3]]], [[7]]),
            ],
            products,
            "Every shape with n, m, and p in [1,3], each matrix filled by one of two fixed "
            "patterns of values from -1 through 2.",
        ),
        task(
            "cheapest-offer",
            "collections",
            "wrong_init",
            "offers is a list of [item, price] pairs with nonnegative prices. Return a dict "
            "mapping each offered item to its lowest price.",
            "offers",
            """
            best = {}
            for item, price in offers:
                if item not in best or price < best[item]:
                    best[item] = price
            return best
            """,
            """
            best = {}
            for item, price in offers:
                lowest = best.get(item, 0)
                best[item] = price if price < lowest else lowest
            return best
            """,
            """
            best = {}
            for item, price in sorted(offers, key=lambda offer: offer[1], reverse=True):
                best[item] = price
            return best
            """,
            [
                ([[["pen", 3], ["ink", 5], ["pen", 2]]], {"pen": 2, "ink": 5}),
                ([[]], {}),
                ([[["cup", 0]]], {"cup": 0}),
            ],
            grid(sequences([["pen", 3], ["pen", 0], ["ink", 5], ["cup", 4]], 3)),
            "All lists of 0..3 offers from four listed [item, price] pairs.",
        ),
        task(
            "peak-sessions",
            "collections",
            "spec_misreading",
            "sessions is a list of [start, end] pairs with start < end, and each session is "
            "active from start up to but not including end. Return the largest number of "
            "sessions active at the same moment. A session that ends at the moment another "
            "starts does not overlap it.",
            "sessions",
            """
            events = []
            for start, end in sessions:
                events.append([start, 1])
                events.append([end, -1])
            events.sort()
            active = peak = 0
            for _, change in events:
                active += change
                peak = max(peak, active)
            return peak
            """,
            """
            events = []
            for start, end in sessions:
                events.append([start, 1])
                events.append([end, -1])
            events.sort(key=lambda event: [event[0], -event[1]])
            active = peak = 0
            for _, change in events:
                active += change
                peak = max(peak, active)
            return peak
            """,
            """
            return max([sum(s <= t < e for s, e in sessions) for t, _ in sessions], default=0)
            """,
            [
                ([[[1, 3], [3, 5]]], 1),
                ([[[1, 4], [2, 5], [3, 6]]], 3),
                ([[]], 0),
                ([[[0, 2], [0, 2]]], 2),
            ],
            grid(sequences([[0, 2], [1, 3], [2, 4], [0, 4], [3, 4]], 3)),
            "All lists of 0..3 sessions from five listed intervals.",
        ),
        task(
            "transition-counts",
            "collections",
            "wrong_init",
            "states lists the states a system passed through, in order. Return a dict that "
            "maps 'a->b' to the number of times state a was immediately followed by state b.",
            "states",
            """
            counts = {}
            for a, b in zip(states, states[1:]):
                key = a + "->" + b
                counts[key] = counts.get(key, 0) + 1
            return counts
            """,
            """
            counts = {}
            previous = states[0] if states else None
            for state in states:
                if previous is not None:
                    key = f"{previous}->{state}"
                    counts[key] = counts.get(key, 0) + 1
                previous = state
            return counts
            """,
            """
            counts = {}
            previous = None
            for state in states:
                if previous is not None:
                    key = f"{previous}->{state}"
                    counts[key] = counts.get(key, 0) + 1
                previous = state
            return counts
            """,
            [([["a", "b", "a", "b"]], {"a->b": 2, "b->a": 1}), ([[]], {}), ([["x"]], {})],
            grid(sequences(["a", "b", "c"], 5)),
            "All lists over {a,b,c} of length 0..5.",
        ),
        task(
            "lru-order",
            "collections",
            "statement_order",
            "Simulate a least-recently-used cache holding at most capacity keys, capacity >= 1. "
            "For each key in accesses: if it is cached, it becomes the most recent; otherwise "
            "it is added as the most recent, first evicting the least recent key if the cache "
            "is full. Return the cached keys from least to most recent.",
            "accesses, capacity",
            """
            cache = []
            for key in accesses:
                if key in cache:
                    cache.remove(key)
                elif len(cache) == capacity:
                    cache.pop(0)
                cache.append(key)
            return cache
            """,
            """
            cache = []
            for key in accesses:
                if len(cache) == capacity:
                    cache.pop(0)
                if key in cache:
                    cache.remove(key)
                cache.append(key)
            return cache
            """,
            """
            last_used = {}
            for time, key in enumerate(accesses):
                last_used[key] = time
                if len(last_used) > capacity:
                    oldest = min(last_used, key=last_used.get)
                    del last_used[oldest]
            return sorted(last_used, key=last_used.get)
            """,
            [
                ([["a", "b", "b"], 2], ["a", "b"]),
                ([["a", "b", "c"], 2], ["b", "c"]),
                ([["x", "y", "x", "z"], 3], ["y", "x", "z"]),
                ([["a", "b", "a", "c"], 2], ["a", "c"]),
            ],
            grid(sequences(["a", "b", "c"], 5), [1, 2, 3]),
            "All access lists over {a,b,c} of length 0..5, capacity in [1,3].",
        ),
        task(
            "visible-posts",
            "collections",
            "inverted_logic",
            "posts is a list of [author, private] pairs. A viewer can see every post that is "
            "not private, plus their own private posts. Return the indexes of the posts viewer "
            "can see, in order.",
            "posts, viewer",
            """
            return [i for i, (author, private) in enumerate(posts) if not private or author == viewer]
            """,
            """
            return [i for i, (author, private) in enumerate(posts) if private or author == viewer]
            """,
            """
            visible = []
            for i in range(len(posts)):
                author, private = posts[i]
                if private and author != viewer:
                    continue
                visible.append(i)
            return visible
            """,
            [
                ([[["ann", False], ["bob", True], ["cy", True]], "cy"], [0, 2]),
                ([[], "ann"], []),
                ([[["ann", True]], "ann"], [0]),
                ([[["bob", False]], "ann"], [0]),
            ],
            grid(
                sequences([["ann", False], ["ann", True], ["bob", False], ["bob", True]], 3),
                ["ann", "bob", "zed"],
            ),
            "All lists of 0..3 posts from four listed pairs, viewer in {ann,bob,zed}.",
        ),
        task(
            "first-repeat",
            "collections",
            "wrong_algorithm",
            "Scan values from left to right and return the first item that is seen for the "
            "second time, or None if no item repeats.",
            "values",
            """
            seen = set()
            for value in values:
                if value in seen:
                    return value
                seen.add(value)
            return None
            """,
            """
            for i, value in enumerate(values):
                if value in values[i + 1 :]:
                    return value
            return None
            """,
            """
            for i in range(len(values)):
                if values[i] in values[:i]:
                    return values[i]
            return None
            """,
            [([[3, 1, 4, 1, 3]], 1), ([[1, 2, 3]], None), ([[]], None), ([[5, 5]], 5)],
            grid(sequences([0, 1, 2, 3], 5)),
            "All lists over {0,1,2,3} of length 0..5.",
        ),
        task(
            "cheapest-basket",
            "collections",
            "wrong_return",
            "baskets is a list of baskets, each a list of item prices. Return the index of the "
            "basket with the smallest total price, the earliest one on ties, or -1 when there "
            "are no baskets.",
            "baskets",
            """
            best = -1
            best_total = None
            for i, basket in enumerate(baskets):
                total = sum(basket)
                if best_total is None or total < best_total:
                    best, best_total = i, total
            return best
            """,
            """
            best = -1
            best_total = None
            for i, basket in enumerate(baskets):
                total = sum(basket)
                if best_total is None or total < best_total:
                    best, best_total = i, total
            return best_total
            """,
            """
            totals = [sum(basket) for basket in baskets]
            return totals.index(min(totals)) if totals else -1
            """,
            [([[[3, 4], [2, 2], [5]]], 1), ([[]], -1), ([[[1], [1]]], 0)],
            grid(sequences([[], [1], [2], [1, 1], [3, 0]], 3)),
            "All lists of 0..3 baskets from [], [1], [2], [1,1], and [3,0].",
        ),
        task(
            "strip-extension",
            "collections",
            "wrong_operation",
            "Return filename without its last extension: remove the last '.' and everything "
            "after it, unless that '.' is the first character, as in '.env'. A filename with "
            "no such '.' is returned unchanged.",
            "filename",
            """
            dot = filename.rfind(".")
            if dot <= 0:
                return filename
            return filename[:dot]
            """,
            """
            dot = filename.find(".")
            if dot <= 0:
                return filename
            return filename[:dot]
            """,
            """
            parts = filename.split(".")
            if len(parts) == 1 or filename.rindex(".") == 0:
                return filename
            return ".".join(parts[:-1])
            """,
            [
                (["report.pdf"], "report"),
                (["archive.tar.gz"], "archive.tar"),
                ([".env"], ".env"),
                (["notes"], "notes"),
            ],
            grid(_strings("a.", 5) + ["report.pdf", "archive.tar.gz", ".env.local", "x.y.z"]),
            "All strings over {a,.} of length 0..5, plus four listed filenames.",
        ),
        task(
            "context-window",
            "collections",
            "boundary",
            "Return the items of values within radius positions of index, that is, at "
            "positions index - radius through index + radius, keeping only positions that "
            "exist. 0 <= index < len(values) and radius >= 0.",
            "values, index, radius",
            """
            return values[max(0, index - radius) : index + radius + 1]
            """,
            """
            return values[max(0, index - radius) : index + radius]
            """,
            """
            start = index - radius if index > radius else 0
            window = []
            for position in range(start, min(len(values), index + radius + 1)):
                window.append(values[position])
            return window
            """,
            [
                ([[1, 2, 3, 4, 5], 2, 1], [2, 3, 4]),
                ([[1, 2, 3], 0, 1], [1, 2]),
                ([[1, 2, 3], 2, 5], [1, 2, 3]),
            ],
            [
                [list(range(10, 10 + n)), i, r]
                for n in range(1, 6)
                for i in range(n)
                for r in range(4)
            ],
            "values = [10, 11, ...] of length 1..5, every valid index, radius in [0,3].",
        ),
        task(
            "unique-visitors",
            "collections",
            "spec_misreading",
            "visits is a list of [user, page] pairs, one per page view. Return a dict mapping "
            "each visited page to the number of different users who viewed it.",
            "visits",
            """
            users = {}
            for user, page in visits:
                users.setdefault(page, set()).add(user)
            return {page: len(names) for page, names in users.items()}
            """,
            """
            users = {}
            for user, page in visits:
                users.setdefault(page, []).append(user)
            return {page: len(names) for page, names in users.items()}
            """,
            """
            pages = {page for _, page in visits}
            return {page: len({user for user, p in visits if p == page}) for page in pages}
            """,
            [
                (
                    [[["ann", "/a"], ["bob", "/a"], ["ann", "/a"], ["ann", "/b"]]],
                    {"/a": 2, "/b": 1},
                ),
                ([[]], {}),
                ([[["cy", "/c"]]], {"/c": 1}),
            ],
            grid(sequences([["ann", "/a"], ["bob", "/a"], ["ann", "/b"], ["bob", "/c"]], 3)),
            "All lists of 0..3 visits from four listed [user, page] pairs.",
        ),
        task(
            "product-of-others",
            "collections",
            "wrong_scope",
            "Return a list whose item i is the product of every item of values except the one "
            "at position i. The product of no items is 1.",
            "values",
            """
            result = []
            for i in range(len(values)):
                product = 1
                for j, value in enumerate(values):
                    if j != i:
                        product *= value
                result.append(product)
            return result
            """,
            """
            result = []
            product = 1
            for i in range(len(values)):
                for j, value in enumerate(values):
                    if j != i:
                        product *= value
                result.append(product)
            return result
            """,
            """
            n = len(values)
            before = [1] * n
            after = [1] * n
            for i in range(1, n):
                before[i] = before[i - 1] * values[i - 1]
            for i in range(n - 2, -1, -1):
                after[i] = after[i + 1] * values[i + 1]
            return [x * y for x, y in zip(before, after)]
            """,
            [([[1, 2, 3, 4]], [24, 12, 8, 6]), ([[]], []), ([[5]], [1]), ([[2, 0, 3]], [0, 6, 0])],
            grid(sequences([-1, 0, 1, 2, 3], 4)),
            "All lists over {-1,0,1,2,3} of length 0..4.",
        ),
    ]


def _control_flow():
    arrivals = [list(c) for n in range(5) for c in combinations_with_replacement([0, 1, 3, 6], n)]
    seconds = [0, 1, 9, 10, 59, 60, 61, 119, 120, 599, 3599, 3600, 3601, 3660, 7199, 86399]
    seconds = sorted(set(seconds + [90061] + list(range(0, 7300, 97))))
    durations = [
        h + m for h in ["", "0h", "1h", "2h", "12h"] for m in ["", "5m", "15m", "45m", "90m"]
    ]
    durations += ["1h1h", "5m5m", "10m1h", "3m2h4m"]
    alias_maps = [
        {k: v for k, v in zip("abc", choice) if v}
        for choice in product(["", "a", "b", "c", "d"], repeat=3)
    ]
    return [
        task(
            "grid-routes",
            "control_flow",
            "wrong_init",
            "A walker crosses a grid of rows by cols cells from the top-left cell [0, 0] to the "
            "bottom-right cell, stepping only right or down and never onto a cell listed in "
            "blocked. Return how many different routes exist; a blocked start or end cell "
            "gives 0.",
            "rows, cols, blocked",
            """
            ways = [[0] * cols for _ in range(rows)]
            for r in range(rows):
                for c in range(cols):
                    if [r, c] in blocked:
                        continue
                    if r == 0 and c == 0:
                        ways[r][c] = 1
                    else:
                        from_above = ways[r - 1][c] if r > 0 else 0
                        from_left = ways[r][c - 1] if c > 0 else 0
                        ways[r][c] = from_above + from_left
            return ways[rows - 1][cols - 1]
            """,
            """
            ways = [[1] * cols for _ in range(rows)]
            for r in range(1, rows):
                for c in range(1, cols):
                    if [r, c] in blocked:
                        ways[r][c] = 0
                    else:
                        ways[r][c] = ways[r - 1][c] + ways[r][c - 1]
            return ways[rows - 1][cols - 1]
            """,
            """
            def count(r, c):
                if r >= rows or c >= cols or [r, c] in blocked:
                    return 0
                if r == rows - 1 and c == cols - 1:
                    return 1
                return count(r + 1, c) + count(r, c + 1)
            return count(0, 0)
            """,
            [([3, 3, []], 6), ([3, 3, [[1, 1]]], 2), ([2, 3, [[0, 1]]], 1), ([1, 1, [[0, 0]]], 0)],
            grid(range(1, 4), range(1, 4), sequences([[0, 1], [1, 0], [1, 1], [2, 2]], 2)),
            "rows and cols in [1,3], blocked any list of 0..2 cells from [0,1], [1,0], [1,1], "
            "and [2,2].",
        ),
        task(
            "reachable-nodes",
            "control_flow",
            "wrong_init",
            "edges lists one-way links as [a, b] pairs. Return the sorted list of nodes that "
            "can be reached from start by following links, including start itself.",
            "edges, start",
            """
            seen = {start}
            frontier = [start]
            while frontier:
                node = frontier.pop()
                for a, b in edges:
                    if a == node and b not in seen:
                        seen.add(b)
                        frontier.append(b)
            return sorted(seen)
            """,
            """
            seen = set()
            frontier = [start]
            while frontier:
                node = frontier.pop()
                for a, b in edges:
                    if a == node and b not in seen:
                        seen.add(b)
                        frontier.append(b)
            return sorted(seen)
            """,
            """
            reached = {start}
            changed = True
            while changed:
                changed = False
                for a, b in edges:
                    if a in reached and b not in reached:
                        reached.add(b)
                        changed = True
            return sorted(reached)
            """,
            [
                ([[["a", "b"], ["b", "c"], ["d", "a"]], "a"], ["a", "b", "c"]),
                ([[], "x"], ["x"]),
                ([[["a", "b"], ["b", "a"]], "b"], ["a", "b"]),
            ],
            grid(sequences([["a", "b"], ["b", "c"], ["c", "a"], ["b", "d"]], 3), ["a", "c", "d"]),
            "All lists of 0..3 links from [a,b], [b,c], [c,a], and [b,d], start in {a,c,d}.",
        ),
        task(
            "payment-status",
            "control_flow",
            "boundary",
            "A bill due on day due may be paid up to grace days late without penalty. Return "
            "'early' if paid is before due, 'on time' if paid is from due through due + grace, "
            "and 'late' otherwise.",
            "due, paid, grace",
            """
            if paid < due:
                return "early"
            if paid <= due + grace:
                return "on time"
            return "late"
            """,
            """
            if paid < due:
                return "early"
            if paid < due + grace:
                return "on time"
            return "late"
            """,
            """
            delay = paid - due
            return "early" if delay < 0 else "on time" if delay <= grace else "late"
            """,
            [
                ([10, 8, 2], "early"),
                ([10, 12, 2], "on time"),
                ([10, 13, 2], "late"),
                ([10, 10, 0], "on time"),
            ],
            grid([5, 10], range(16), range(4)),
            "due in {5,10}, paid in [0,15], grace in [0,3].",
        ),
        task(
            "binary-guesses",
            "control_flow",
            "inverted_logic",
            "A player looks for target, which is between low and high inclusive, by guessing "
            "(low + high) // 2 each turn. After a guess that is too high, high becomes guess - "
            "1; after one that is too low, low becomes guess + 1. Return the list of guesses up "
            "to and including the correct one.",
            "target, low, high",
            """
            guesses = []
            while low <= high:
                guess = (low + high) // 2
                guesses.append(guess)
                if guess == target:
                    break
                if guess > target:
                    high = guess - 1
                else:
                    low = guess + 1
            return guesses
            """,
            """
            guesses = []
            while low <= high:
                guess = (low + high) // 2
                guesses.append(guess)
                if guess == target:
                    break
                if guess < target:
                    high = guess - 1
                else:
                    low = guess + 1
            return guesses
            """,
            """
            guess = (low + high) // 2
            if guess == target:
                return [guess]
            if guess > target:
                return [guess] + solve(target, low, guess - 1)
            return [guess] + solve(target, guess + 1, high)
            """,
            [
                ([3, 1, 10], [5, 2, 3]),
                ([7, 7, 7], [7]),
                ([1, 1, 4], [2, 1]),
                ([10, 1, 10], [5, 8, 9, 10]),
            ],
            [
                [target, low, high]
                for low in (0, 1, 5)
                for high in range(low, low + 11)
                for target in range(low, high + 1)
            ],
            "low in {0,1,5}, high from low through low + 10, every target from low to high.",
        ),
        task(
            "queue-waits",
            "control_flow",
            "statement_order",
            "One clerk serves customers in arrival order. arrivals lists their arrival times in "
            "nondecreasing order, and each customer takes service minutes. A customer is served "
            "as soon as they have arrived and the clerk is free. Return each customer's waiting "
            "time before being served.",
            "arrivals, service",
            """
            free_at = 0
            waits = []
            for arrival in arrivals:
                free_at = max(free_at, arrival)
                waits.append(free_at - arrival)
                free_at += service
            return waits
            """,
            """
            free_at = 0
            waits = []
            for arrival in arrivals:
                waits.append(free_at - arrival)
                free_at = max(free_at, arrival)
                free_at += service
            return waits
            """,
            """
            waits = []
            finish = None
            for arrival in arrivals:
                start = arrival if finish is None or finish <= arrival else finish
                waits.append(start - arrival)
                finish = start + service
            return waits
            """,
            [([[0, 1, 10], 3], [0, 2, 0]), ([[], 5], []), ([[2, 2, 2], 1], [0, 1, 2])],
            grid(arrivals, [1, 2, 4]),
            "Every nondecreasing arrival list of length 0..4 over {0,1,3,6}, service in {1,2,4}.",
        ),
        task(
            "first-open-slot",
            "control_flow",
            "wrong_scope",
            "booked lists busy periods as [start, end] pairs, each covering start up to but "
            "not including end. Return the earliest whole-number time t >= 0 such that a "
            "meeting from t up to t + length overlaps no busy period and t + length <= day_end, "
            "or -1 if there is none. length >= 1.",
            "booked, length, day_end",
            """
            for t in range(day_end - length + 1):
                if all(t + length <= start or end <= t for start, end in booked):
                    return t
            return -1
            """,
            """
            for t in range(day_end - length + 1):
                if all(t + length <= start or end <= t for start, end in booked):
                    return t
                return -1
            return -1
            """,
            """
            t = 0
            for start, end in sorted(booked):
                if start - t >= length:
                    break
                t = max(t, end)
            return t if t + length <= day_end else -1
            """,
            [
                ([[[0, 2]], 3, 8], 2),
                ([[], 2, 5], 0),
                ([[[1, 4]], 2, 5], -1),
                ([[[2, 3], [5, 6]], 2, 8], 0),
            ],
            grid(sequences([[0, 2], [1, 4], [5, 6], [3, 5]], 2), [1, 2, 3], [4, 6, 8]),
            "All lists of 0..2 periods from four listed ones, length in [1,3], day_end in {4,6,8}.",
        ),
        task(
            "turnstile",
            "control_flow",
            "missing_step",
            "A turnstile starts locked, and events is a list of 'coin' and 'push'. A coin "
            "unlocks it. A push on an unlocked turnstile lets one person through and locks it "
            "again; a push on a locked one sounds the alarm. Return [people through, alarms].",
            "events",
            """
            locked = True
            passes = alarms = 0
            for event in events:
                if event == "coin":
                    locked = False
                elif locked:
                    alarms += 1
                else:
                    passes += 1
                    locked = True
            return [passes, alarms]
            """,
            """
            locked = True
            passes = alarms = 0
            for event in events:
                if event == "coin":
                    locked = False
                elif locked:
                    alarms += 1
                else:
                    passes += 1
            return [passes, alarms]
            """,
            """
            state = "locked"
            counts = {"pass": 0, "alarm": 0}
            for event in events:
                if event == "coin":
                    state = "open"
                elif state == "open":
                    counts["pass"] += 1
                    state = "locked"
                else:
                    counts["alarm"] += 1
            return [counts["pass"], counts["alarm"]]
            """,
            [
                ([["coin", "push", "push"]], [1, 1]),
                ([[]], [0, 0]),
                ([["push", "coin", "coin", "push"]], [1, 1]),
            ],
            grid(sequences(["coin", "push"], 7)),
            "All event lists over {coin, push} of length 0..7.",
        ),
        task(
            "cursor-editor",
            "control_flow",
            "missing_edge_case",
            "Simulate a one-line text editor that starts empty with the cursor at position 0. "
            "['type', c] inserts character c before the cursor and moves the cursor after it; "
            "['left'] and ['right'] move the cursor one place but never past either end; "
            "['back'] deletes the character before the cursor, if there is one. Return the "
            "final text.",
            "commands",
            """
            text = ""
            cursor = 0
            for command in commands:
                if command[0] == "type":
                    text = text[:cursor] + command[1] + text[cursor:]
                    cursor += 1
                elif command[0] == "left":
                    cursor = max(0, cursor - 1)
                elif command[0] == "right":
                    cursor = min(len(text), cursor + 1)
                elif cursor > 0:
                    text = text[: cursor - 1] + text[cursor:]
                    cursor -= 1
            return text
            """,
            """
            text = ""
            cursor = 0
            for command in commands:
                if command[0] == "type":
                    text = text[:cursor] + command[1] + text[cursor:]
                    cursor += 1
                elif command[0] == "left":
                    cursor = max(0, cursor - 1)
                elif command[0] == "right":
                    cursor = min(len(text), cursor + 1)
                else:
                    text = text[: cursor - 1] + text[cursor:]
                    cursor -= 1
            return text
            """,
            """
            before, after = [], []
            for command in commands:
                if command[0] == "type":
                    before.append(command[1])
                elif command[0] == "left" and before:
                    after.append(before.pop())
                elif command[0] == "right" and after:
                    before.append(after.pop())
                elif command[0] == "back" and before:
                    before.pop()
            return "".join(before) + "".join(reversed(after))
            """,
            [
                ([[["type", "a"], ["type", "b"], ["left"], ["back"]]], "b"),
                ([[["type", "a"], ["type", "b"], ["left"], ["left"], ["back"]]], "ab"),
                ([[]], ""),
                ([[["type", "x"], ["right"], ["type", "y"]]], "xy"),
            ],
            grid(sequences([["type", "a"], ["type", "b"], ["left"], ["right"], ["back"]], 4)),
            "All lists of 0..4 commands from five listed commands.",
        ),
        task(
            "duration-minutes",
            "control_flow",
            "missing_step",
            "text is a duration made of groups of ASCII digits, each followed by 'h' for hours "
            "or 'm' for minutes, such as '2h15m', '45m', or '1h1h'. Return the total number of "
            "minutes.",
            "text",
            """
            total = 0
            number = 0
            for c in text:
                if c == "h":
                    total += number * 60
                    number = 0
                elif c == "m":
                    total += number
                    number = 0
                else:
                    number = number * 10 + int(c)
            return total
            """,
            """
            total = 0
            number = 0
            for c in text:
                if c == "h":
                    total += number * 60
                elif c == "m":
                    total += number
                    number = 0
                else:
                    number = number * 10 + int(c)
            return total
            """,
            """
            total = 0
            digits = ""
            for c in text:
                if c in "0123456789":
                    digits += c
                else:
                    total += int(digits) * (60 if c == "h" else 1)
                    digits = ""
            return total
            """,
            [(["2h15m"], 135), (["45m"], 45), ([""], 0), (["1h1h"], 120)],
            grid(durations),
            "Every hours part in {'',0h,1h,2h,12h} followed by every minutes part in "
            "{'',5m,15m,45m,90m}, plus four listed mixed strings.",
        ),
        task(
            "wildcard-match",
            "control_flow",
            "missing_edge_case",
            "Return True iff pattern matches all of text, where '?' in pattern matches any one "
            "character, '*' matches any sequence of characters including an empty one, and "
            "every other character matches itself.",
            "pattern, text",
            """
            match = [[False] * (len(text) + 1) for _ in range(len(pattern) + 1)]
            match[0][0] = True
            for i in range(1, len(pattern) + 1):
                if pattern[i - 1] == "*":
                    match[i][0] = match[i - 1][0]
            for i in range(1, len(pattern) + 1):
                for j in range(1, len(text) + 1):
                    p = pattern[i - 1]
                    if p == "*":
                        match[i][j] = match[i - 1][j] or match[i][j - 1]
                    elif p == "?" or p == text[j - 1]:
                        match[i][j] = match[i - 1][j - 1]
            return match[len(pattern)][len(text)]
            """,
            """
            match = [[False] * (len(text) + 1) for _ in range(len(pattern) + 1)]
            match[0][0] = True
            for i in range(1, len(pattern) + 1):
                for j in range(1, len(text) + 1):
                    p = pattern[i - 1]
                    if p == "*":
                        match[i][j] = match[i - 1][j] or match[i][j - 1]
                    elif p == "?" or p == text[j - 1]:
                        match[i][j] = match[i - 1][j - 1]
            return match[len(pattern)][len(text)]
            """,
            """
            if not pattern:
                return text == ""
            if pattern[0] == "*":
                return solve(pattern[1:], text) or (text != "" and solve(pattern, text[1:]))
            if text and pattern[0] in ("?", text[0]):
                return solve(pattern[1:], text[1:])
            return False
            """,
            [
                (["a*c", "abbc"], True),
                (["*", ""], True),
                (["a?c", "ac"], False),
                (["*b", "b"], True),
            ],
            grid(_strings("a?*", 3), _strings("ab", 3)),
            "Every pattern over {a,?,*} of length 0..3 with every text over {a,b} of length 0..3.",
        ),
        task(
            "expand-tabs",
            "control_flow",
            "wrong_algorithm",
            "Return line with each tab character replaced by spaces up to the next tab stop. "
            "Tab stops are every width columns, counting columns of the output from 0. width "
            ">= 1 and line contains no newlines.",
            "line, width",
            r"""
            out = ""
            for c in line:
                if c == "\t":
                    out += " " * (width - len(out) % width)
                else:
                    out += c
            return out
            """,
            r"""
            out = ""
            for c in line:
                out += " " * width if c == "\t" else c
            return out
            """,
            r"""
            pieces = line.split("\t")
            out = pieces[0]
            for piece in pieces[1:]:
                out += " " * (width - len(out) % width) + piece
            return out
            """,
            [
                (["a\tb", 4], "a   b"),
                (["ab\tc", 4], "ab  c"),
                (["\tx", 4], "    x"),
                (["abcd\te", 4], "abcd    e"),
            ],
            grid(_strings("a\t", 5), [1, 2, 4]),
            "All strings over {a,tab} of length 0..5, width in {1,2,4}.",
        ),
        task(
            "elapsed-label",
            "control_flow",
            "boundary",
            "Format a duration of seconds >= 0: under a minute as '45s', under an hour as "
            "'3m 05s', and otherwise as '2h 03m 05s'. Minutes and seconds that follow a larger "
            "unit always have two digits.",
            "seconds",
            """
            hours, rest = divmod(seconds, 3600)
            minutes, secs = divmod(rest, 60)
            if seconds < 60:
                return f"{secs}s"
            if seconds < 3600:
                return f"{minutes}m {secs:02d}s"
            return f"{hours}h {minutes:02d}m {secs:02d}s"
            """,
            """
            hours, rest = divmod(seconds, 3600)
            minutes, secs = divmod(rest, 60)
            if seconds <= 60:
                return f"{secs}s"
            if seconds < 3600:
                return f"{minutes}m {secs:02d}s"
            return f"{hours}h {minutes:02d}m {secs:02d}s"
            """,
            """
            if seconds < 60:
                return str(seconds) + "s"
            minutes = seconds // 60
            tail = str(seconds % 60).zfill(2) + "s"
            if minutes < 60:
                return str(minutes) + "m " + tail
            return str(minutes // 60) + "h " + str(minutes % 60).zfill(2) + "m " + tail
            """,
            [([45], "45s"), ([60], "1m 00s"), ([3725], "1h 02m 05s"), ([600], "10m 00s")],
            grid(seconds),
            "16 listed boundary values, 90061, and every multiple of 97 below 7300.",
        ),
        task(
            "rover-commands",
            "control_flow",
            "wrong_variable",
            "A rover starts at x = 0, y = 0 facing north. For each command, 'L' turns it 90 "
            "degrees left, 'R' turns it 90 degrees right, and 'F' moves it one unit forward; "
            "north is +y and east is +x. Return [x, y, heading], where heading is 'N', 'E', "
            "'S', or 'W'.",
            "commands",
            """
            x = y = 0
            facing = 0
            dx = [0, 1, 0, -1]
            dy = [1, 0, -1, 0]
            for c in commands:
                if c == "L":
                    facing = (facing - 1) % 4
                elif c == "R":
                    facing = (facing + 1) % 4
                else:
                    x += dx[facing]
                    y += dy[facing]
            return [x, y, "NESW"[facing]]
            """,
            """
            x = y = 0
            facing = 0
            dx = [0, 1, 0, -1]
            dy = [1, 0, -1, 0]
            for c in commands:
                if c == "L":
                    facing = (facing - 1) % 4
                elif c == "R":
                    facing = (facing + 1) % 4
                else:
                    x += dy[facing]
                    y += dx[facing]
            return [x, y, "NESW"[facing]]
            """,
            """
            x, y, heading = 0, 0, "N"
            left = {"N": "W", "W": "S", "S": "E", "E": "N"}
            right = {after: before for before, after in left.items()}
            for c in commands:
                if c == "L":
                    heading = left[heading]
                elif c == "R":
                    heading = right[heading]
                elif heading in "NS":
                    y += 1 if heading == "N" else -1
                else:
                    x += 1 if heading == "E" else -1
            return [x, y, heading]
            """,
            [
                (["FF"], [0, 2, "N"]),
                (["RFF"], [2, 0, "E"]),
                (["LLF"], [0, -1, "S"]),
                (["FRF"], [1, 1, "E"]),
            ],
            grid(_strings("LRF", 5)),
            "All command strings over {L,R,F} of length 0..5.",
        ),
        task(
            "speeding-fine",
            "control_flow",
            "wrong_scope",
            "Return the fine for driving at speed where the limit is limit: 0 when not over the "
            "limit, 100 when up to 10 over, 250 when 11 to 20 over, and 500 when more than 20 "
            "over. Every fine is doubled in a school zone.",
            "speed, limit, school_zone",
            """
            over = speed - limit
            if over <= 0:
                fine = 0
            elif over <= 10:
                fine = 100
            elif over <= 20:
                fine = 250
            else:
                fine = 500
            if school_zone:
                fine *= 2
            return fine
            """,
            """
            over = speed - limit
            if over <= 0:
                fine = 0
            elif over <= 10:
                fine = 100
            elif over <= 20:
                fine = 250
            else:
                fine = 500
                if school_zone:
                    fine *= 2
            return fine
            """,
            """
            over = speed - limit
            fine = 0
            for threshold, amount in [[0, 100], [10, 250], [20, 500]]:
                if over > threshold:
                    fine = amount
            return fine * (2 if school_zone else 1)
            """,
            [
                ([45, 40, True], 200),
                ([70, 40, True], 1000),
                ([40, 40, True], 0),
                ([55, 40, False], 250),
            ],
            grid(range(25, 76), [30, 40, 50], [False, True]),
            "speed in [25,75], limit in {30,40,50}, both school_zone values.",
        ),
        task(
            "resolve-alias",
            "control_flow",
            "wrong_return",
            "aliases maps names to other names. Starting from name, keep replacing the current "
            "name by its alias until reaching a name that has no alias, and return that name. "
            "If the replacements would go on forever because of a cycle, return None.",
            "name, aliases",
            """
            current = name
            for _ in range(len(aliases) + 1):
                if current not in aliases:
                    return current
                current = aliases[current]
            return None
            """,
            """
            current = name
            for _ in range(len(aliases) + 1):
                if current not in aliases:
                    return current
                current = aliases[current]
            return current
            """,
            """
            seen = set()
            while name in aliases:
                if name in seen:
                    return None
                seen.add(name)
                name = aliases[name]
            return name
            """,
            [
                (["a", {"a": "b", "b": "c"}], "c"),
                (["x", {}], "x"),
                (["a", {"a": "b", "b": "a"}], None),
                (["b", {"a": "b"}], "b"),
            ],
            grid(["a", "b", "c", "d"], alias_maps),
            "name in {a,b,c,d}; aliases maps each of a, b, c to nothing or one of a, b, c, d.",
        ),
        task(
            "elimination-circle",
            "control_flow",
            "boundary",
            "People numbered 1 through n stand in a circle. Starting from person 1, count k "
            "people around the circle; the k-th person counted leaves, and counting restarts "
            "with the next person. Return the number of the last person remaining. n >= 1 and "
            "k >= 1.",
            "n, k",
            """
            people = list(range(1, n + 1))
            index = 0
            while len(people) > 1:
                index = (index + k - 1) % len(people)
                people.pop(index)
            return people[0]
            """,
            """
            people = list(range(1, n + 1))
            index = 0
            while len(people) > 1:
                index = (index + k) % len(people)
                people.pop(index)
            return people[0]
            """,
            """
            survivor = 0
            for size in range(2, n + 1):
                survivor = (survivor + k) % size
            return survivor + 1
            """,
            [([5, 2], 3), ([7, 3], 4), ([1, 4], 1), ([4, 1], 4)],
            grid(range(1, 13), range(1, 6)),
            "n in [1,12], k in [1,5].",
        ),
        task(
            "cycle-length",
            "control_flow",
            "wrong_return",
            "next_of maps each node to the node it points to, and every node it mentions is a "
            "key. Starting from start, follow the pointers until some node is reached a second "
            "time. Return the length of the loop that the walk ends in: the number of steps it "
            "takes to get from that node back to itself.",
            "next_of, start",
            """
            first_step = {}
            current = start
            step = 0
            while current not in first_step:
                first_step[current] = step
                current = next_of[current]
                step += 1
            return step - first_step[current]
            """,
            """
            first_step = {}
            current = start
            step = 0
            while current not in first_step:
                first_step[current] = step
                current = next_of[current]
                step += 1
            return step
            """,
            """
            current = start
            for _ in range(len(next_of)):
                current = next_of[current]
            length = 1
            probe = next_of[current]
            while probe != current:
                probe = next_of[probe]
                length += 1
            return length
            """,
            [
                ([{"a": "b", "b": "c", "c": "b"}, "a"], 2),
                ([{"a": "a"}, "a"], 1),
                ([{"a": "b", "b": "a"}, "b"], 2),
                ([{"x": "y", "y": "z", "z": "z"}, "x"], 1),
            ],
            [
                [dict(zip("abc", targets)), start]
                for targets in product("abc", repeat=3)
                for start in "abc"
            ]
            + [[dict(zip("abcd", targets)), "a"] for targets in product("abcd", repeat=4)],
            "Every pointer map on {a,b,c} with every start, and every pointer map on {a,b,c,d} "
            "starting from a.",
        ),
        task(
            "ticket-line",
            "control_flow",
            "wrong_return",
            "People stand in a line, and person i wants tickets[i] tickets, at least 1 each. "
            "Every second the person at the front buys one ticket and, if they still want "
            "more, goes to the back of the line; otherwise they leave. Return how many seconds "
            "pass until person k has bought all of their tickets.",
            "tickets, k",
            """
            time = 0
            line = [[i, t] for i, t in enumerate(tickets)]
            while line:
                person, wanted = line.pop(0)
                time += 1
                if wanted > 1:
                    line.append([person, wanted - 1])
                elif person == k:
                    return time
            return time
            """,
            """
            time = 0
            line = [[i, t] for i, t in enumerate(tickets)]
            while line:
                person, wanted = line.pop(0)
                time += 1
                if person == k:
                    return time
                if wanted > 1:
                    line.append([person, wanted - 1])
            return time
            """,
            """
            mine = tickets[k]
            return sum(min(t, mine) if i <= k else min(t, mine - 1) for i, t in enumerate(tickets))
            """,
            [([[2, 3, 2], 2], 6), ([[5, 1, 1, 1], 0], 8), ([[1], 0], 1), ([[1, 2], 1], 3)],
            [
                [list(c), k]
                for n in range(1, 5)
                for c in product([1, 2, 3], repeat=n)
                for k in range(n)
            ],
            "Every ticket list over {1,2,3} of length 1..4, with every position k.",
        ),
        task(
            "undo-redo",
            "control_flow",
            "missing_step",
            "A counter starts at 0. ['add', n] adds n; ['undo'] reverts the most recent add "
            "that has not been undone, if any; ['redo'] reapplies the most recently undone add, "
            "if any. A new add discards everything that could have been redone. Return the "
            "final value.",
            "commands",
            """
            value = 0
            done, undone = [], []
            for command in commands:
                if command[0] == "add":
                    value += command[1]
                    done.append(command[1])
                    undone = []
                elif command[0] == "undo" and done:
                    amount = done.pop()
                    value -= amount
                    undone.append(amount)
                elif command[0] == "redo" and undone:
                    amount = undone.pop()
                    value += amount
                    done.append(amount)
            return value
            """,
            """
            value = 0
            done, undone = [], []
            for command in commands:
                if command[0] == "add":
                    value += command[1]
                    done.append(command[1])
                elif command[0] == "undo" and done:
                    amount = done.pop()
                    value -= amount
                    undone.append(amount)
                elif command[0] == "redo" and undone:
                    amount = undone.pop()
                    value += amount
                    done.append(amount)
            return value
            """,
            """
            history = []
            position = 0
            for command in commands:
                if command[0] == "add":
                    history = history[:position] + [command[1]]
                    position += 1
                elif command[0] == "undo":
                    position = max(0, position - 1)
                else:
                    position = min(len(history), position + 1)
            return sum(history[:position])
            """,
            [
                ([[["add", 5], ["undo"], ["add", 2], ["redo"]]], 2),
                ([[["add", 3], ["add", 4], ["undo"]]], 3),
                ([[["undo"], ["redo"]]], 0),
                ([[["add", 1], ["undo"], ["redo"]]], 1),
            ],
            grid(sequences([["add", 1], ["add", 10], ["undo"], ["redo"]], 4)),
            "All lists of 0..4 commands from add 1, add 10, undo, and redo.",
        ),
        task(
            "cinema-price",
            "control_flow",
            "spec_misreading",
            "Return the ticket price. Children under 12 and seniors aged 65 or older pay 6; "
            "otherwise students pay 9 on weekdays and everyone else pays 12. Days are numbered "
            "0 (Monday) through 6 (Sunday), and on Tuesday no ticket costs more than 7.",
            "age, day, student",
            """
            if age < 12 or age >= 65:
                price = 6
            elif student and day < 5:
                price = 9
            else:
                price = 12
            if day == 1:
                price = min(price, 7)
            return price
            """,
            """
            if age < 12 or age > 65:
                price = 6
            elif student and day < 5:
                price = 9
            else:
                price = 12
            if day == 1:
                price = min(price, 7)
            return price
            """,
            """
            prices = [12]
            if age < 12 or age > 64:
                prices.append(6)
            if student and day not in (5, 6):
                prices.append(9)
            if day == 1:
                prices.append(7)
            return min(prices)
            """,
            [
                ([65, 3, False], 6),
                ([30, 1, False], 7),
                ([20, 5, True], 12),
                ([20, 2, True], 9),
                ([8, 6, True], 6),
            ],
            grid([5, 11, 12, 30, 64, 65, 80], range(7), [False, True]),
            "age in {5,11,12,30,64,65,80}, every day, both student values.",
        ),
        task(
            "league-points",
            "control_flow",
            "missing_edge_case",
            "matches is a list of [home, away, home_goals, away_goals] results. A win is worth "
            "3 points, a draw 1 point to each team, and a loss 0. Return a dict mapping every "
            "team that played to its total points, including teams with 0 points.",
            "matches",
            """
            points = {}
            for home, away, home_goals, away_goals in matches:
                points.setdefault(home, 0)
                points.setdefault(away, 0)
                if home_goals > away_goals:
                    points[home] += 3
                elif home_goals < away_goals:
                    points[away] += 3
                else:
                    points[home] += 1
                    points[away] += 1
            return points
            """,
            """
            points = {}
            for home, away, home_goals, away_goals in matches:
                if home_goals > away_goals:
                    points[home] = points.get(home, 0) + 3
                elif home_goals < away_goals:
                    points[away] = points.get(away, 0) + 3
                else:
                    points[home] = points.get(home, 0) + 1
                    points[away] = points.get(away, 0) + 1
            return points
            """,
            """
            def score(team):
                total = 0
                for home, away, home_goals, away_goals in matches:
                    if team == home:
                        mine, theirs = home_goals, away_goals
                    elif team == away:
                        mine, theirs = away_goals, home_goals
                    else:
                        continue
                    total += 3 if mine > theirs else 1 if mine == theirs else 0
                return total
            teams = {match[0] for match in matches} | {match[1] for match in matches}
            return {team: score(team) for team in teams}
            """,
            [
                ([[["a", "b", 2, 1], ["b", "c", 0, 0]]], {"a": 3, "b": 1, "c": 1}),
                ([[["a", "b", 0, 3]]], {"a": 0, "b": 3}),
                ([[]], {}),
            ],
            grid(
                sequences(
                    [["a", "b", 1, 0], ["b", "a", 2, 2], ["a", "c", 0, 1], ["c", "b", 3, 1]], 3
                )
            ),
            "All lists of 0..3 results from four listed matches.",
        ),
        task(
            "next-alarm",
            "control_flow",
            "spec_misreading",
            "Times are minutes after midnight, 0 through 1439, and each alarm rings every day "
            "at its time. Return how many minutes after now the next alarm rings; an alarm set "
            "exactly at now next rings tomorrow, 1440 minutes later. Return None if there are "
            "no alarms.",
            "now, alarms",
            """
            best = None
            for alarm in alarms:
                wait = (alarm - now) % 1440
                if wait == 0:
                    wait = 1440
                if best is None or wait < best:
                    best = wait
            return best
            """,
            """
            best = None
            for alarm in alarms:
                wait = (alarm - now) % 1440
                if best is None or wait < best:
                    best = wait
            return best
            """,
            """
            return min([(alarm - now - 1) % 1440 + 1 for alarm in alarms], default=None)
            """,
            [([420, [450, 400]], 30), ([420, [420]], 1440), ([1439, [0]], 1), ([10, []], None)],
            grid([0, 1, 420, 719, 1439], sequences([0, 1, 420, 1000, 1439], 2)),
            "now in {0,1,420,719,1439}, alarms any list of 0..2 times from {0,1,420,1000,1439}.",
        ),
        task(
            "floor-label",
            "control_flow",
            "spec_misreading",
            "Return the elevator label for floor index n: 0 is 'G', a negative index -k is 'B' "
            "followed by k, and floors above ground are numbered 1, 2, 3, and so on, skipping "
            "the number 13, so index 13 is labeled '14' and index 14 is labeled '15'.",
            "n",
            """
            if n == 0:
                return "G"
            if n < 0:
                return "B" + str(-n)
            return str(n + 1 if n >= 13 else n)
            """,
            """
            if n == 0:
                return "G"
            if n < 0:
                return "B" + str(-n)
            if n == 13:
                return "14"
            return str(n)
            """,
            """
            if n <= 0:
                return "G" if n == 0 else "B%d" % -n
            label = 0
            for _ in range(n):
                label += 1
                if label == 13:
                    label += 1
            return str(label)
            """,
            [([0], "G"), ([-2], "B2"), ([12], "12"), ([14], "15"), ([13], "14")],
            grid(range(-5, 31)),
            "n in [-5,30].",
        ),
        task(
            "next-arrangement",
            "control_flow",
            "wrong_algorithm",
            "Return the next arrangement of values in lexicographic order: the smallest "
            "rearrangement of its items that is greater than values. If values is already the "
            "greatest arrangement, return its items sorted in increasing order. Do not change "
            "values.",
            "values",
            """
            items = list(values)
            i = len(items) - 2
            while i >= 0 and items[i] >= items[i + 1]:
                i -= 1
            if i < 0:
                return sorted(items)
            j = len(items) - 1
            while items[j] <= items[i]:
                j -= 1
            items[i], items[j] = items[j], items[i]
            items[i + 1 :] = reversed(items[i + 1 :])
            return items
            """,
            """
            items = list(values)
            i = len(items) - 2
            while i >= 0 and items[i] >= items[i + 1]:
                i -= 1
            if i < 0:
                return sorted(items)
            items[i], items[i + 1] = items[i + 1], items[i]
            items[i + 1 :] = sorted(items[i + 1 :])
            return items
            """,
            """
            from itertools import permutations
            arrangements = sorted(set(permutations(values)))
            position = arrangements.index(tuple(values))
            return list(arrangements[(position + 1) % len(arrangements)])
            """,
            [
                ([[1, 2, 3]], [1, 3, 2]),
                ([[1, 3, 2]], [2, 1, 3]),
                ([[3, 2, 1]], [1, 2, 3]),
                ([[1, 1, 5]], [1, 5, 1]),
            ],
            grid(sequences([1, 2, 3], 4)),
            "All lists over {1,2,3} of length 0..4.",
        ),
        task(
            "best-cargo",
            "control_flow",
            "wrong_algorithm",
            "items is a list of [weight, value] pairs, and each item can be taken at most once. "
            "Return the largest total value of items whose total weight is at most capacity.",
            "items, capacity",
            """
            best = [0] * (capacity + 1)
            for weight, value in items:
                for room in range(capacity, weight - 1, -1):
                    best[room] = max(best[room], best[room - weight] + value)
            return best[capacity]
            """,
            """
            best = [0] * (capacity + 1)
            for weight, value in items:
                for room in range(weight, capacity + 1):
                    best[room] = max(best[room], best[room - weight] + value)
            return best[capacity]
            """,
            """
            top = 0
            for mask in range(1 << len(items)):
                weight = value = 0
                for i, item in enumerate(items):
                    if mask >> i & 1:
                        weight += item[0]
                        value += item[1]
                if weight <= capacity:
                    top = max(top, value)
            return top
            """,
            [
                ([[[2, 3], [3, 4]], 4], 4),
                ([[[1, 1], [2, 5], [3, 6]], 5], 11),
                ([[], 3], 0),
                ([[[5, 9]], 4], 0),
            ],
            grid(sequences([[1, 1], [2, 3], [3, 4], [2, 2]], 3), [0, 2, 4, 6]),
            "All lists of 0..3 items from four listed pairs, capacity in {0,2,4,6}.",
        ),
        task(
            "land-regions",
            "control_flow",
            "wrong_algorithm",
            "grid is a list of equal-length rows of 0s and 1s. Return the number of islands: "
            "groups of 1 cells connected through their up, down, left, or right neighbors.",
            "grid",
            """
            seen = set()
            count = 0
            for r in range(len(grid)):
                for c in range(len(grid[r])):
                    if grid[r][c] != 1 or (r, c) in seen:
                        continue
                    count += 1
                    seen.add((r, c))
                    stack = [(r, c)]
                    while stack:
                        y, x = stack.pop()
                        for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                            inside = 0 <= ny < len(grid) and 0 <= nx < len(grid[ny])
                            if inside and grid[ny][nx] == 1 and (ny, nx) not in seen:
                                seen.add((ny, nx))
                                stack.append((ny, nx))
            return count
            """,
            """
            count = 0
            for r in range(len(grid)):
                for c in range(len(grid[r])):
                    if grid[r][c] == 1:
                        up = r > 0 and grid[r - 1][c] == 1
                        left = c > 0 and grid[r][c - 1] == 1
                        if not up and not left:
                            count += 1
            return count
            """,
            """
            land = {(r, c) for r, row in enumerate(grid) for c, v in enumerate(row) if v == 1}
            count = 0
            while land:
                count += 1
                frontier = [land.pop()]
                while frontier:
                    r, c = frontier.pop()
                    for cell in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                        if cell in land:
                            land.remove(cell)
                            frontier.append(cell)
            return count
            """,
            [([[[1, 1, 0], [0, 0, 1]]], 2), ([[[1, 0, 1], [1, 1, 1]]], 1), ([[[0, 0], [0, 0]]], 0)],
            grid(_bit_grids([(2, 3), (3, 2), (3, 3)], [0, 1])),
            "Every 2x3, 3x2, and 3x3 grid of 0s and 1s.",
        ),
        task(
            "progress-bar",
            "control_flow",
            "wrong_operation",
            "Draw a progress bar for done out of total steps, with total >= 1 and 0 <= done <= "
            "total: '[' then width cells, then ']'. The first done * width // total cells, "
            "rounded down, are '#' and the rest are '.'.",
            "done, total, width",
            """
            filled = done * width // total
            return "[" + "#" * filled + "." * (width - filled) + "]"
            """,
            """
            filled = round(done * width / total)
            return "[" + "#" * filled + "." * (width - filled) + "]"
            """,
            """
            cells = ["#" if (i + 1) * total <= done * width else "." for i in range(width)]
            return "[" + "".join(cells) + "]"
            """,
            [
                ([1, 3, 5], "[#....]"),
                ([3, 3, 4], "[####]"),
                ([0, 5, 3], "[...]"),
                ([1, 2, 4], "[##..]"),
            ],
            [[d, t, w] for t in (1, 3, 7) for d in range(t + 1) for w in (0, 4, 5, 10)],
            "total in {1,3,7}, every done from 0 to total, width in {0,4,5,10}.",
        ),
        task(
            "command-words",
            "control_flow",
            "wrong_operation",
            "line is a typed command whose words are separated by one or more spaces, possibly "
            "with spaces at either end. Return [command, args]: command is the first word in "
            "lowercase and args is the list of the remaining words. A line with no words "
            "returns ['', []].",
            "line",
            """
            words = line.split()
            if not words:
                return ["", []]
            return [words[0].lower(), words[1:]]
            """,
            """
            words = line.split(" ")
            if not words:
                return ["", []]
            return [words[0].lower(), words[1:]]
            """,
            """
            words = []
            current = ""
            for c in line + " ":
                if c == " ":
                    if current:
                        words.append(current)
                    current = ""
                else:
                    current += c
            if not words:
                return ["", []]
            return [words[0].lower(), words[1:]]
            """,
            [
                (["GO north"], ["go", ["north"]]),
                (["  look   around "], ["look", ["around"]]),
                ([""], ["", []]),
                (["   "], ["", []]),
            ],
            grid(_strings("aB ", 5)),
            "All strings over {a,B,space} of length 0..5.",
        ),
        task(
            "drop-blocks",
            "control_flow",
            "aliasing",
            "grid is a nonempty list of equal-length, nonempty rows of '#' (a block) and '.' "
            "(empty). Let every block fall straight down as far as it can within its column, "
            "and return the new grid. Do not change grid.",
            "grid",
            """
            height, width = len(grid), len(grid[0])
            result = [["."] * width for _ in range(height)]
            for c in range(width):
                blocks = sum(row[c] == "#" for row in grid)
                for r in range(height - blocks, height):
                    result[r][c] = "#"
            return result
            """,
            """
            height, width = len(grid), len(grid[0])
            for c in range(width):
                blocks = sum(row[c] == "#" for row in grid)
                for r in range(height):
                    grid[r][c] = "#" if r >= height - blocks else "."
            return grid
            """,
            """
            columns = [sorted(column, key=lambda cell: cell == "#") for column in zip(*grid)]
            return [list(row) for row in zip(*columns)]
            """,
            [
                ([[["#", "."], [".", "."]]], [[".", "."], ["#", "."]]),
                ([[["#", "#"], [".", "#"]]], [[".", "#"], ["#", "#"]]),
                ([[["."]]], [["."]]),
            ],
            grid(_bit_grids([(1, 2), (2, 2), (3, 2), (2, 3)], [".", "#"])),
            "Every 1x2, 2x2, 3x2, and 2x3 grid of '.' and '#'.",
        ),
        task(
            "retry-attempts",
            "control_flow",
            "inverted_logic",
            "statuses lists the HTTP status that each attempt of a request would get, in "
            "order. The client stops after a 200, after any status not in retryable, or when "
            "statuses runs out. Return how many attempts it makes.",
            "statuses, retryable",
            """
            attempts = 0
            for status in statuses:
                attempts += 1
                if status == 200 or status not in retryable:
                    break
            return attempts
            """,
            """
            attempts = 0
            for status in statuses:
                attempts += 1
                if status == 200 or status in retryable:
                    break
            return attempts
            """,
            """
            for i, status in enumerate(statuses):
                if status == 200 or status not in retryable:
                    return i + 1
            return len(statuses)
            """,
            [
                ([[503, 503, 200], [503]], 3),
                ([[404, 200], [503]], 1),
                ([[], [503]], 0),
                ([[503, 500], [503]], 2),
            ],
            grid(sequences([200, 404, 500, 503], 4), [[], [503], [500, 503]]),
            "All status lists over {200,404,500,503} of length 0..4, retryable in [], [503], "
            "and [500,503].",
        ),
        task(
            "ledger-transfers",
            "control_flow",
            "wrong_variable",
            "balances maps account names to amounts. Apply each [source, target, amount] "
            "transfer in order: it moves amount from source to target if source currently "
            "holds at least amount, and is skipped otherwise. An account missing from balances "
            "holds 0. Return the final balances as a new dict.",
            "balances, transfers",
            """
            result = dict(balances)
            for source, target, amount in transfers:
                if result.get(source, 0) >= amount:
                    result[source] = result.get(source, 0) - amount
                    result[target] = result.get(target, 0) + amount
            return result
            """,
            """
            result = dict(balances)
            for source, target, amount in transfers:
                if result.get(target, 0) >= amount:
                    result[source] = result.get(source, 0) - amount
                    result[target] = result.get(target, 0) + amount
            return result
            """,
            """
            result = {name: amount for name, amount in balances.items()}
            for source, target, amount in transfers:
                available = result[source] if source in result else 0
                if amount > available:
                    continue
                result[source] = available - amount
                result[target] = (result[target] if target in result else 0) + amount
            return result
            """,
            [
                ([{"a": 5, "b": 0}, [["a", "b", 3]]], {"a": 2, "b": 3}),
                ([{"a": 1}, [["a", "b", 2]]], {"a": 1}),
                ([{"a": 4, "b": 9}, [["a", "b", 4], ["b", "a", 13]]], {"a": 13, "b": 0}),
            ],
            [
                [{"a": x, "b": y}, transfers]
                for x in (0, 2, 5)
                for y in (0, 2, 5)
                for transfers in sequences([["a", "b", 2], ["b", "a", 3], ["a", "c", 1]], 2)
            ],
            "Balances a and b in {0,2,5}, with every list of 0..2 transfers from three listed "
            "ones.",
        ),
        task(
            "adler-checksum",
            "control_flow",
            "statement_order",
            "Compute a checksum of data, a list of byte values from 0 through 255. Start with "
            "a = 1 and b = 0; for each byte in order, set a to (a + byte) % 65521 and then b "
            "to (b + a) % 65521. Return b * 65536 + a.",
            "data",
            """
            a, b = 1, 0
            for byte in data:
                a = (a + byte) % 65521
                b = (b + a) % 65521
            return b * 65536 + a
            """,
            """
            a, b = 1, 0
            for byte in data:
                b = (b + a) % 65521
                a = (a + byte) % 65521
            return b * 65536 + a
            """,
            """
            n = len(data)
            a = (1 + sum(data)) % 65521
            b = (n + sum((n - i) * byte for i, byte in enumerate(data))) % 65521
            return b * 65536 + a
            """,
            [([[1, 2]], 393220), ([[]], 1), ([[255]], 16777472)],
            grid(sequences([0, 1, 97, 255], 4)),
            "All byte lists over {0,1,97,255} of length 0..4.",
        ),
    ]


def final3_tasks():
    tasks = _validation() + _collections() + _control_flow()
    if len({t.id for t in tasks}) != len(tasks):
        raise ValueError("Duplicate task IDs")
    return tasks
