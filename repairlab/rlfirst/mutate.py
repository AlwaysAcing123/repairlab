"""Program transformations that copy common real mistakes.

Inputs are trusted reference programs, never model output. Every mutation is a set of
text replacements on the reference source, located through the AST, so a bug keeps the
reference's formatting and differs from it only where the mistake is. Two single bugs
whose replacements do not overlap combine into a two-bug task.
"""

import ast
import re
from dataclasses import dataclass

from ..execution import validate_source

KINDS = (
    "boundary",
    "missing_edge_case",
    "wrong_variable",
    "wrong_init",
    "statement_order",
    "missing_step",
    "wrong_scope",
    "wrong_operation",
    "aliasing",
    "inverted_logic",
    "wrong_return",
)
HAND_WRITTEN = "hand_written"
TWO_BUGS = "two_bugs"
ALL_KINDS = KINDS + (HAND_WRITTEN, TWO_BUGS)

COMPARE_TEXT = {
    ast.Lt: "<",
    ast.LtE: "<=",
    ast.Gt: ">",
    ast.GtE: ">=",
    ast.Eq: "==",
    ast.NotEq: "!=",
    ast.In: "in",
    ast.NotIn: "not in",
    ast.Is: "is",
    ast.IsNot: "is not",
}
COMPARE_PATTERN = {
    ast.Lt: r"(?<![<>=!])<(?![<=])",
    ast.LtE: r"<=",
    ast.Gt: r"(?<![<>=!-])>(?![>=])",
    ast.GtE: r">=",
    ast.Eq: r"==",
    ast.NotEq: r"!=",
    ast.In: r"(?<![\w.])in(?!\w)",
    ast.NotIn: r"not\s+in",
    ast.Is: r"(?<![\w.])is(?!\w)",
    ast.IsNot: r"is\s+not",
}
BOUNDARY_SWAP = {ast.Lt: ast.LtE, ast.LtE: ast.Lt, ast.Gt: ast.GtE, ast.GtE: ast.Gt}
LOGIC_SWAP = {
    ast.Eq: ast.NotEq,
    ast.NotEq: ast.Eq,
    ast.In: ast.NotIn,
    ast.NotIn: ast.In,
    ast.Is: ast.IsNot,
    ast.IsNot: ast.Is,
}
BINOP_TEXT = {ast.Add: "+", ast.Sub: "-", ast.FloorDiv: "//", ast.Div: "/", ast.Mod: "%"}
BINOP_PATTERN = {
    ast.Add: r"\+",
    ast.Sub: r"(?<![eE])-",
    ast.FloorDiv: r"//",
    ast.Div: r"(?<!/)/(?![/=])",
    ast.Mod: r"%",
}
BINOP_SWAP = {
    ast.Add: (ast.Sub,),
    ast.Sub: (ast.Add,),
    ast.FloorDiv: (ast.Div, ast.Mod),
    ast.Div: (ast.FloorDiv,),
    ast.Mod: (ast.FloorDiv,),
}
FUNCTION_SWAP = {"min": ("max",), "max": ("min",), "any": ("all",), "all": ("any",)}
METHOD_SWAP = {
    "strip": ("lstrip", "rstrip"),
    "lstrip": ("rstrip", "strip"),
    "rstrip": ("lstrip", "strip"),
    "startswith": ("endswith",),
    "endswith": ("startswith",),
    "lower": ("upper",),
    "upper": ("lower",),
    "islower": ("isupper",),
    "isupper": ("islower",),
    "isalpha": ("isalnum",),
    "isalnum": ("isalpha",),
    "isdigit": ("isalnum",),
    "find": ("rfind",),
    "rfind": ("find",),
    "append": ("extend",),
    "extend": ("append",),
    "keys": ("values",),
    "values": ("keys",),
    "split": ("rsplit",),
}
SIMPLE_EXPR = (ast.Name, ast.Constant, ast.Call, ast.Attribute, ast.Subscript)


@dataclass(frozen=True)
class Edit:
    start: int
    end: int
    text: str


@dataclass(frozen=True)
class Mutation:
    kind: str
    description: str
    edits: tuple

    def span(self):
        return (min(e.start for e in self.edits), max(e.end for e in self.edits))


def apply(source, edits):
    result = source
    for edit in sorted(edits, key=lambda e: e.start, reverse=True):
        result = result[: edit.start] + edit.text + result[edit.end :]
    return result


def overlapping(first, second):
    return any(a.start < b.end and b.start < a.end for a in first for b in second)


class _Source:
    def __init__(self, text):
        if not text.isascii():
            raise ValueError("Mutation offsets assume ASCII source")
        self.text = text
        self.starts = [0]
        for line in text.splitlines(keepends=True):
            self.starts.append(self.starts[-1] + len(line))

    def offset(self, line, column):
        return self.starts[line - 1] + column

    def span(self, node):
        return self.offset(node.lineno, node.col_offset), self.offset(
            node.end_lineno, node.end_col_offset
        )

    def segment(self, node):
        start, end = self.span(node)
        return self.text[start:end]

    def lines(self, node):
        """Whole lines of a statement, from the line start to after its newline."""
        return self.starts[node.lineno - 1], self.starts[node.end_lineno]

    def indent(self, node):
        line = self.text[self.starts[node.lineno - 1] : self.starts[node.lineno]]
        return line[: len(line) - len(line.lstrip(" "))]


def _operator(src, left_end, right_start, pattern):
    """Locate an operator between two operand spans; None when it cannot be pinned."""
    between = src.text[left_end:right_start]
    matches = list(re.finditer(pattern, between))
    if len(matches) != 1:
        return None
    match = matches[0]
    return left_end + match.start(), left_end + match.end()


def _wrap(node, text):
    return text if isinstance(node, SIMPLE_EXPR) else f"({text})"


def _shift(src, node, delta):
    """Text for an integer expression moved by ``delta``, written as a person would."""
    text = src.segment(node)
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return str(node.value + delta)
    if (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, (ast.Add, ast.Sub))
        and isinstance(node.right, ast.Constant)
        and type(node.right.value) is int
    ):
        value = node.right.value if isinstance(node.op, ast.Add) else -node.right.value
        value += delta
        left = src.segment(node.left)
        if value == 0:
            return left
        return f"{left} + {value}" if value > 0 else f"{left} - {-value}"
    sign = "+" if delta > 0 else "-"
    return f"{_wrap(node, text)} {sign} {abs(delta)}"


def _statement_lists(tree):
    for node in ast.walk(tree):
        for field in ("body", "orelse", "finalbody"):
            block = getattr(node, field, None)
            if isinstance(block, list) and block and isinstance(block[0], ast.stmt):
                yield block


def _local_names(function):
    """Parameters and names the function body assigns, excluding comprehension variables."""
    names = [a.arg for a in function.args.args]
    comprehension_targets = set()
    for node in ast.walk(function):
        if isinstance(node, ast.comprehension):
            for name in ast.walk(node.target):
                if isinstance(name, ast.Name):
                    comprehension_targets.add(id(name))
    for node in ast.walk(function):
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Store)
            and id(node) not in comprehension_targets
            and node.id not in names
        ):
            names.append(node.id)
    return names


def _comprehension_scoped(function):
    """Name nodes that read a comprehension variable; swapping them breaks scope rules."""
    scoped = set()
    for node in ast.walk(function):
        if isinstance(node, (ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)):
            targets = {
                n.id
                for gen in node.generators
                for n in ast.walk(gen.target)
                if isinstance(n, ast.Name)
            }
            for name in ast.walk(node):
                if isinstance(name, ast.Name) and name.id in targets:
                    scoped.add(id(name))
    return scoped


def _delete_statement(src, block, index):
    statement = block[index]
    start, end = src.lines(statement)
    if len(block) == 1:
        return Edit(start, end, src.indent(statement) + "pass\n")
    return Edit(start, end, "")


def generate(reference):
    """All single mutations of a reference, as (kind, description, edits)."""
    src = _Source(reference)
    tree = ast.parse(reference)
    function = tree.body[0]
    found = []

    def add(kind, description, *edits):
        found.append(Mutation(kind, description, tuple(edits)))

    local_names = _local_names(function)
    scoped = _comprehension_scoped(function)

    for node in ast.walk(function):
        # Comparisons: boundary and inverted logic.
        if isinstance(node, ast.Compare):
            operands = [node.left, *node.comparators]
            for i, op in enumerate(node.ops):
                left_end = src.span(operands[i])[1]
                right_start = src.span(operands[i + 1])[0]
                location = _operator(src, left_end, right_start, COMPARE_PATTERN[type(op)])
                if location is None:
                    continue
                if type(op) in BOUNDARY_SWAP:
                    new = COMPARE_TEXT[BOUNDARY_SWAP[type(op)]]
                    add("boundary", f"{COMPARE_TEXT[type(op)]} -> {new}", Edit(*location, new))
                if type(op) in LOGIC_SWAP:
                    new = COMPARE_TEXT[LOGIC_SWAP[type(op)]]
                    add(
                        "inverted_logic",
                        f"{COMPARE_TEXT[type(op)]} -> {new}",
                        Edit(*location, new),
                    )
            for comparator in node.comparators:
                if isinstance(comparator, ast.Constant) and type(comparator.value) is int:
                    for delta in (1, -1):
                        add(
                            "boundary",
                            f"compared constant {comparator.value} -> {comparator.value + delta}",
                            Edit(*src.span(comparator), _shift(src, comparator, delta)),
                        )
        # range() stops and slice bounds: boundary.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id == "range" and 1 <= len(node.args) <= 3 and not node.keywords:
                stop = node.args[0] if len(node.args) == 1 else node.args[1]
                for delta in (1, -1):
                    add(
                        "boundary",
                        f"range stop {'+' if delta > 0 else '-'}1",
                        Edit(*src.span(stop), _shift(src, stop, delta)),
                    )
            if node.func.id in FUNCTION_SWAP and node.args:
                for new in FUNCTION_SWAP[node.func.id]:
                    add(
                        "wrong_operation",
                        f"{node.func.id} -> {new}",
                        Edit(*src.span(node.func), new),
                    )
            if node.func.id in ("list", "dict") and len(node.args) == 1 and not node.keywords:
                argument = node.args[0]
                if isinstance(argument, ast.Name):
                    add(
                        "aliasing",
                        f"{node.func.id}({argument.id}) copy removed",
                        Edit(*src.span(node), argument.id),
                    )
        if isinstance(node, ast.Slice):
            for bound in (node.lower, node.upper):
                if bound is None:
                    continue
                for delta in (1, -1):
                    add(
                        "boundary",
                        f"slice bound {'+' if delta > 0 else '-'}1",
                        Edit(*src.span(bound), _shift(src, bound, delta)),
                    )
        if (
            isinstance(node, ast.Subscript)
            and isinstance(node.slice, ast.Slice)
            and node.slice.lower is None
            and node.slice.upper is None
            and node.slice.step is None
            and isinstance(node.value, ast.Name)
        ):
            add(
                "aliasing",
                f"{node.value.id}[:] copy removed",
                Edit(*src.span(node), node.value.id),
            )
        # Method calls: wrong operation, aliasing.
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            method = node.func.attr
            end = src.span(node.func)[1]
            start = end - len(method)
            if src.text[start:end] == method:
                for new in METHOD_SWAP.get(method, ()):
                    add("wrong_operation", f".{method} -> .{new}", Edit(start, end, new))
            if method == "pop" and not node.keywords:
                call_start, call_end = src.span(node)
                if not node.args:
                    add("wrong_operation", ".pop() -> .pop(0)", Edit(call_end - 2, call_end, "(0)"))
                elif (
                    len(node.args) == 1
                    and isinstance(node.args[0], ast.Constant)
                    and node.args[0].value == 0
                ):
                    add(
                        "wrong_operation",
                        ".pop(0) -> .pop()",
                        Edit(src.span(node.args[0])[0], src.span(node.args[0])[1], ""),
                    )
            if method == "copy" and not node.args and isinstance(node.func.value, ast.Name):
                add(
                    "aliasing",
                    f"{node.func.value.id}.copy() removed",
                    Edit(*src.span(node), node.func.value.id),
                )
        # Arithmetic: wrong operation.
        if isinstance(node, ast.BinOp) and type(node.op) in BINOP_SWAP:
            location = _operator(
                src, src.span(node.left)[1], src.span(node.right)[0], BINOP_PATTERN[type(node.op)]
            )
            if location is not None:
                for new in BINOP_SWAP[type(node.op)]:
                    add(
                        "wrong_operation",
                        f"{BINOP_TEXT[type(node.op)]} -> {BINOP_TEXT[new]}",
                        Edit(*location, BINOP_TEXT[new]),
                    )
        if isinstance(node, ast.AugAssign) and isinstance(node.op, (ast.Add, ast.Sub)):
            location = _operator(
                src,
                src.span(node.target)[1],
                src.span(node.value)[0],
                r"\+=" if isinstance(node.op, ast.Add) else r"-=",
            )
            if location is not None:
                other = "-=" if isinstance(node.op, ast.Add) else "+="
                old = "+=" if isinstance(node.op, ast.Add) else "-="
                add("wrong_operation", f"{old} -> {other}", Edit(*location, other))
                add("wrong_operation", f"{old} -> =", Edit(*location, "="))
        # Boolean logic: inverted logic, missing edge case.
        if isinstance(node, ast.BoolOp):
            word = "and" if isinstance(node.op, ast.And) else "or"
            other = "or" if word == "and" else "and"
            locations = [
                _operator(
                    src,
                    src.span(a)[1],
                    src.span(b)[0],
                    rf"(?<!\w){word}(?!\w)",
                )
                for a, b in zip(node.values, node.values[1:])
            ]
            if all(locations):
                add(
                    "inverted_logic",
                    f"{word} -> {other}",
                    *(Edit(*location, other) for location in locations),
                )
            for i in range(len(node.values)):
                if i == 0:
                    start, end = src.span(node.values[0])[0], src.span(node.values[1])[0]
                else:
                    start, end = src.span(node.values[i - 1])[1], src.span(node.values[i])[1]
                remaining = [v for j, v in enumerate(node.values) if j != i]
                if len(node.values) == 2 and isinstance(remaining[0], ast.BoolOp):
                    continue
                add(
                    "missing_edge_case",
                    f"condition {i + 1} of {len(node.values)} dropped from {word}",
                    Edit(start, end, ""),
                )
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.Not):
            operand = src.segment(node.operand)
            add(
                "inverted_logic", "not dropped", Edit(*src.span(node), _wrap(node.operand, operand))
            )
        if isinstance(node, (ast.If, ast.While, ast.IfExp)) and not (
            isinstance(node.test, ast.UnaryOp) and isinstance(node.test.op, ast.Not)
        ):
            test = node.test
            text = src.segment(test)
            negated = f"not {text}" if isinstance(test, SIMPLE_EXPR) else f"not ({text})"
            add("inverted_logic", "condition negated", Edit(*src.span(test), negated))
        if isinstance(node, ast.IfExp):
            body, orelse = src.span(node.body), src.span(node.orelse)
            add(
                "inverted_logic",
                "conditional branches swapped",
                Edit(*body, src.text[orelse[0] : orelse[1]]),
                Edit(*orelse, src.text[body[0] : body[1]]),
            )
        # Initial values: wrong starting value.
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
        ):
            value = node.value
            span = src.span(value)
            if isinstance(value, ast.Constant):
                if type(value.value) is bool:
                    add("wrong_init", "boolean start flipped", Edit(*span, str(not value.value)))
                elif type(value.value) is int:
                    for delta in (1, -1):
                        add(
                            "wrong_init",
                            f"start {value.value} -> {value.value + delta}",
                            Edit(*span, str(value.value + delta)),
                        )
                elif value.value is None:
                    add("wrong_init", "start None -> 0", Edit(*span, "0"))
            elif (
                isinstance(value, ast.Call)
                and isinstance(value.func, ast.Name)
                and value.func.id == "float"
                and len(value.args) == 1
                and isinstance(value.args[0], ast.Constant)
                and value.args[0].value in ("inf", "-inf")
            ):
                add("wrong_init", "start infinity -> 0", Edit(*span, "0"))
            elif isinstance(value, ast.Subscript) and isinstance(value.slice, ast.Constant):
                add("wrong_init", "start from element -> 0", Edit(*span, "0"))
        # Returns: wrong return.
        if isinstance(node, ast.Return) and node.value is not None:
            value = node.value
            if isinstance(value, ast.Constant) and type(value.value) is bool:
                add(
                    "wrong_return",
                    "returned boolean flipped",
                    Edit(*src.span(value), str(not value.value)),
                )
            elif isinstance(value, ast.Name):
                for other in local_names:
                    if other != value.id:
                        add(
                            "wrong_return",
                            f"returns {other} instead of {value.id}",
                            Edit(*src.span(value), other),
                        )
        # Variables: wrong variable.
        if (
            isinstance(node, ast.Name)
            and isinstance(node.ctx, ast.Load)
            and node.id in local_names
            and id(node) not in scoped
        ):
            for other in local_names:
                if other != node.id:
                    add(
                        "wrong_variable",
                        f"{node.id} -> {other}",
                        Edit(*src.span(node), other),
                    )

    # Statement-level mutations.
    for block in _statement_lists(function):
        for i, statement in enumerate(block):
            guard = (
                isinstance(statement, ast.If)
                and not statement.orelse
                and isinstance(statement.body[-1], (ast.Return, ast.Raise, ast.Continue, ast.Break))
            )
            if guard:
                add("missing_edge_case", "guard deleted", _delete_statement(src, block, i))
            if isinstance(statement, ast.AugAssign) or (
                isinstance(statement, ast.Expr) and isinstance(statement.value, ast.Call)
            ):
                add("missing_step", "update deleted", _delete_statement(src, block, i))
            if (
                isinstance(statement, ast.Assign)
                and len(statement.targets) == 1
                and isinstance(statement.targets[0], ast.Name)
                and sum(
                    isinstance(n, ast.Name)
                    and isinstance(n.ctx, ast.Store)
                    and n.id == statement.targets[0].id
                    for n in ast.walk(function)
                )
                > 1
            ):
                add("missing_step", "reassignment deleted", _delete_statement(src, block, i))
            if i + 1 < len(block):
                first, second = statement, block[i + 1]
                if not isinstance(first, ast.Return) and not isinstance(second, ast.Return):
                    a_start, a_end = src.lines(first)
                    b_start, b_end = src.lines(second)
                    first_text, second_text = src.text[a_start:a_end], src.text[b_start:b_end]
                    if first_text.strip() != second_text.strip():
                        add(
                            "statement_order",
                            "adjacent statements swapped",
                            Edit(a_start, b_end, second_text + first_text),
                        )
            if isinstance(statement, (ast.For, ast.While)) and not statement.orelse:
                loop_start, loop_end = src.lines(statement)
                body = statement.body
                inner = src.indent(body[0])
                outer = src.indent(statement)
                if len(body) >= 2:
                    last_start, last_end = src.lines(body[-1])
                    moved = "".join(
                        outer + line[len(inner) :] if line.startswith(inner) else line
                        for line in src.text[last_start:last_end].splitlines(keepends=True)
                    )
                    add(
                        "wrong_scope",
                        "last loop statement moved after the loop",
                        Edit(last_start, loop_end, src.text[last_end:loop_end] + moved),
                    )
                if i + 1 < len(block):
                    after_start, after_end = src.lines(block[i + 1])
                    extra = inner[len(outer) :]
                    moved = "".join(
                        extra + line for line in src.text[after_start:after_end].splitlines(True)
                    )
                    add(
                        "wrong_scope",
                        "statement after the loop moved into it",
                        Edit(loop_start, after_end, src.text[loop_start:loop_end] + moved),
                    )
    return found


def candidates(reference, signature):
    """Valid, distinct single mutations of a reference, keyed by their source."""
    result = {}
    for mutation in generate(reference):
        text = apply(reference, mutation.edits)
        if text == reference or text in result:
            continue
        try:
            validate_source(text, signature)
        except ValueError:
            continue
        result[text] = mutation
    return result


def combine(reference, signature, first, second):
    """A two-bug mutation from two single bugs of different kinds, or None."""
    if first.kind == second.kind or overlapping(first.edits, second.edits):
        return None
    edits = tuple(sorted(first.edits + second.edits, key=lambda e: e.start))
    text = apply(reference, edits)
    try:
        validate_source(text, signature)
    except ValueError:
        return None
    return Mutation(TWO_BUGS, f"{first.description}; {second.description}", edits), text
