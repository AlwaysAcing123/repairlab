"""Hand-written training bugs for the RL-first pool (plan addendum 2, item 2).

One new bug in each of 28 programs chosen by seed. Each is a whole function that a person
might plausibly write after reading the specification, not a single-token mutation:

- ``spec_misreading``: the wording is read differently, such as an inclusive or exclusive
  bound, a convention the specification overrides, duplicates or neighbours ignored, or a
  unit or format misunderstood.
- ``wrong_algorithm``: the specification is understood but the method is wrong, such as
  clamping once instead of per step, comparing only neighbours, or assuming sorted input.

Keys are training program ids from ``pool.training_sources()``. Every source keeps the
program's signature, finishes on every domain input, fails at least one of them, and
differs after name anonymization from the reference, the original bug and every bug
already in ``artifacts/rlfirst-pool-v2``. These bugs go through the same admission and
their own base sweep before entering the pool.
"""

HANDWRITTEN = {
    "allowed-status": {
        "kind_detail": "wrong_algorithm",
        "description": "Binary search over allowed, which assumes the list is sorted.",
        "source": (
            "def solve(status, allowed):\n"
            "    low, high = 0, len(allowed) - 1\n"
            "    while low <= high:\n"
            "        middle = (low + high) // 2\n"
            "        if allowed[middle] == status:\n"
            "            return True\n"
            "        if allowed[middle] < status:\n"
            "            low = middle + 1\n"
            "        else:\n"
            "            high = middle - 1\n"
            "    return False\n"
        ),
    },
    "balanced-parens": {
        "kind_detail": "wrong_algorithm",
        "description": (
            "Checks equal counts plus an opening first and closing last character "
            "instead of tracking the depth of every prefix."
        ),
        "source": (
            "def solve(text):\n"
            "    if not text:\n"
            "        return True\n"
            '    if text[0] != "(" or text[-1] != ")":\n'
            "        return False\n"
            '    return text.count("(") == text.count(")")\n'
        ),
    },
    "binary-length": {
        "kind_detail": "wrong_algorithm",
        "description": (
            "Doubles a power of two until it reaches n, which undercounts exact powers of two."
        ),
        "source": (
            "def solve(n):\n"
            "    digits = 1\n"
            "    power = 2\n"
            "    while power < n:\n"
            "        power *= 2\n"
            "        digits += 1\n"
            "    return digits\n"
        ),
    },
    "business-hours": {
        "kind_detail": "spec_misreading",
        "description": (
            "Numbers weekdays with Sunday as 0, so Monday through Friday become 1 through 5."
        ),
        "source": (
            "def solve(weekday, hour):\n"
            "    is_weekday = 1 <= weekday <= 5\n"
            "    in_office = 9 <= hour < 17\n"
            "    return is_weekday and in_office\n"
        ),
    },
    "can-afford": {
        "kind_detail": "spec_misreading",
        "description": (
            "Treats the balance and the overdraft as alternative ways to pay instead of adding them."
        ),
        "source": (
            "def solve(price, balance, overdraft):\n"
            "    if price <= balance:\n"
            "        return True\n"
            "    return price <= overdraft\n"
        ),
    },
    "clamped-elevator": {
        "kind_detail": "wrong_algorithm",
        "description": "Adds up all moves first and clamps only the final floor, not each move.",
        "source": (
            "def solve(moves, top):\n"
            "    target = sum(moves)\n"
            "    if target < 0:\n"
            "        return 0\n"
            "    if target > top:\n"
            "        return top\n"
            "    return target\n"
        ),
    },
    "clock-add": {
        "kind_detail": "wrong_algorithm",
        "description": (
            "Adds whole hours and leftover minutes separately, losing the carry when the "
            "minutes pass 60."
        ),
        "source": (
            "def solve(hour, minute, minutes):\n"
            "    new_minute = (minute + minutes) % 60\n"
            "    new_hour = (hour + minutes // 60) % 24\n"
            "    return [new_hour, new_minute]\n"
        ),
    },
    "collapse-repeats": {
        "kind_detail": "spec_misreading",
        "description": (
            "Removes every repeated value instead of only runs of equal neighbouring values."
        ),
        "source": (
            "def solve(values):\n"
            "    seen = []\n"
            "    for value in values:\n"
            "        if value not in seen:\n"
            "            seen.append(value)\n"
            "    return seen\n"
        ),
    },
    "countdown": {
        "kind_detail": "wrong_algorithm",
        "description": (
            "Starts the list with n and counts down while above 1, so n = 0 gives [0]."
        ),
        "source": (
            "def solve(n):\n"
            "    numbers = [n]\n"
            "    while numbers[-1] > 1:\n"
            "        numbers.append(numbers[-1] - 1)\n"
            "    return numbers\n"
        ),
    },
    "first-match": {
        "kind_detail": "spec_misreading",
        "description": "Returns a 1-based position instead of a 0-based index.",
        "source": (
            "def solve(values, target):\n"
            "    for position, value in enumerate(values, 1):\n"
            "        if value == target:\n"
            "            return position\n"
            "    return -1\n"
        ),
    },
    "half-open-overlap": {
        "kind_detail": "wrong_algorithm",
        "description": (
            "Checks only whether an endpoint of b falls inside a, missing b containing a."
        ),
        "source": (
            "def solve(a_start, a_end, b_start, b_end):\n"
            "    starts_inside = a_start <= b_start < a_end\n"
            "    ends_inside = a_start < b_end <= a_end\n"
            "    return starts_inside or ends_inside\n"
        ),
    },
    "hex-color-code": {
        "kind_detail": "spec_misreading",
        "description": "Also accepts the three-digit CSS shorthand such as '#abc'.",
        "source": (
            "def solve(text):\n"
            '    if len(text) not in (4, 7) or not text.startswith("#"):\n'
            "        return False\n"
            "    for c in text[1:]:\n"
            '        if c not in "0123456789abcdefABCDEF":\n'
            "            return False\n"
            "    return True\n"
        ),
    },
    "leap-year": {
        "kind_detail": "wrong_algorithm",
        "description": (
            "Rejects centuries before checking divisibility by 400, so 2000 is not a leap year."
        ),
        "source": (
            "def solve(year):\n"
            "    if year % 100 == 0:\n"
            "        return False\n"
            "    if year % 400 == 0:\n"
            "        return True\n"
            "    return year % 4 == 0\n"
        ),
    },
    "lower-median": {
        "kind_detail": "spec_misreading",
        "description": (
            "Averages the two middle items of an even-length list, rounding down, instead "
            "of taking the lower one."
        ),
        "source": (
            "def solve(values):\n"
            "    ordered = sorted(values)\n"
            "    n = len(ordered)\n"
            "    middle = n // 2\n"
            "    if n % 2 == 1:\n"
            "        return ordered[middle]\n"
            "    return (ordered[middle - 1] + ordered[middle]) // 2\n"
        ),
    },
    "no-triple-repeat": {
        "kind_detail": "spec_misreading",
        "description": (
            "Rejects any character that appears three times anywhere, ignoring 'in a row'."
        ),
        "source": (
            "def solve(text):\n"
            "    for c in set(text):\n"
            "        if text.count(c) >= 3:\n"
            "            return False\n"
            "    return True\n"
        ),
    },
    "optional-name": {
        "kind_detail": "spec_misreading",
        "description": "Treats only the space character as whitespace, so tabs and newlines pass.",
        "source": (
            "def solve(name):\n"
            "    if name is None:\n"
            "        return True\n"
            '    return len(name.replace(" ", "")) > 0\n'
        ),
    },
    "password-length": {
        "kind_detail": "spec_misreading",
        "description": "Measures the length in UTF-8 bytes instead of characters.",
        "source": (
            "def solve(text, minimum):\n    size = len(text.encode())\n    return size >= minimum\n"
        ),
    },
    "payload-limit": {
        "kind_detail": "spec_misreading",
        "description": "Treats a limit of 0 as meaning no limit at all.",
        "source": (
            "def solve(size, limit):\n"
            "    if limit == 0:\n"
            "        return True\n"
            "    return size <= limit\n"
        ),
    },
    "positive-batch": {
        "kind_detail": "wrong_algorithm",
        "description": (
            "Checks that the batch total is positive instead of every single quantity."
        ),
        "source": (
            "def solve(quantities):\n"
            "    total = 0\n"
            "    for quantity in quantities:\n"
            "        total += quantity\n"
            "    return total > 0\n"
        ),
    },
    "quantity-range": {
        "kind_detail": "spec_misreading",
        "description": "Only rules out negative quantities, so a quantity of 0 is accepted.",
        "source": (
            "def solve(quantity, maximum):\n"
            "    if quantity < 0:\n"
            "        return False\n"
            "    return quantity <= maximum\n"
        ),
    },
    "required-fields": {
        "kind_detail": "spec_misreading",
        "description": "Rejects an empty list, which the specification says is valid.",
        "source": (
            "def solve(present):\n"
            "    if len(present) == 0:\n"
            "        return False\n"
            "    for field in present:\n"
            "        if not field:\n"
            "            return False\n"
            "    return True\n"
        ),
    },
    "running-totals": {
        "kind_detail": "wrong_algorithm",
        "description": ("Adds each value to the previous value instead of to the previous total."),
        "source": (
            "def solve(values):\n"
            "    sums = []\n"
            "    for i, value in enumerate(values):\n"
            "        if i == 0:\n"
            "            sums.append(value)\n"
            "        else:\n"
            "            sums.append(values[i - 1] + value)\n"
            "    return sums\n"
        ),
    },
    "second-largest-distinct": {
        "kind_detail": "wrong_algorithm",
        "description": (
            "One pass tracking the top two values, which lets a repeat of the largest "
            "become the second largest."
        ),
        "source": (
            "def solve(values):\n"
            "    first = None\n"
            "    second = None\n"
            "    for x in values:\n"
            "        if first is None or x > first:\n"
            "            second = first\n"
            "            first = x\n"
            "        elif second is None or x > second:\n"
            "            second = x\n"
            "    return second\n"
        ),
    },
    "streak-points": {
        "kind_detail": "spec_misreading",
        "description": (
            "Makes the bonus grow by 5 for each extra day of a streak instead of a flat 5."
        ),
        "source": (
            "def solve(days):\n"
            "    points = 0\n"
            "    streak = 0\n"
            "    for met in days:\n"
            "        if met:\n"
            "            streak += 1\n"
            "            points += 10 + 5 * (streak - 1)\n"
            "        else:\n"
            "            streak = 0\n"
            "    return points\n"
        ),
    },
    "traffic-light": {
        "kind_detail": "spec_misreading",
        "description": (
            "In an emergency sends green through yellow first, as real signals do, "
            "instead of straight to red."
        ),
        "source": (
            "def solve(state, emergency):\n"
            "    if emergency:\n"
            '        return "yellow" if state == "green" else "red"\n'
            '    if state == "green":\n'
            '        return "yellow"\n'
            '    if state == "yellow":\n'
            '        return "red"\n'
            '    return "green"\n'
        ),
    },
    "triangle-sides": {
        "kind_detail": "wrong_algorithm",
        "description": "Checks only a + b > c, as if c were always the longest side.",
        "source": "def solve(a, b, c):\n    return c < a + b\n",
    },
    "valid-day-of-month": {
        "kind_detail": "wrong_algorithm",
        "description": (
            "Gives odd months 31 days and even months 30, missing the switch at August."
        ),
        "source": (
            "def solve(month, day):\n"
            "    if month == 2:\n"
            "        last = 28\n"
            "    elif month % 2 == 1:\n"
            "        last = 31\n"
            "    else:\n"
            "        last = 30\n"
            "    return 1 <= day <= last\n"
        ),
    },
    "word-frequency": {
        "kind_detail": "wrong_algorithm",
        "description": (
            "Counts runs of consecutive equal words, as if the list were grouped, so a "
            "word's count is its latest run length."
        ),
        "source": (
            "def solve(words):\n"
            "    counts = {}\n"
            "    previous = None\n"
            "    run = 0\n"
            "    for word in words:\n"
            "        if word == previous:\n"
            "            run += 1\n"
            "        else:\n"
            "            previous = word\n"
            "            run = 1\n"
            "        counts[word] = run\n"
            "    return counts\n"
        ),
    },
}
