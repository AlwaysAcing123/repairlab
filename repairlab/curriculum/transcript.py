"""The exact conversation format of autonomous evaluation, shared with demonstrations."""

from ..native_repair import SYSTEM
from ..native_tools import assistant_call
from ..tasks import canonical


def initial_messages(task):
    return [
        {"role": "system", "content": SYSTEM},
        {"role": "user", "content": canonical(task.public())},
    ]


def tool_message(tool, observation):
    return {
        "role": "tool",
        "name": tool,
        "content": canonical({k: v for k, v in observation.items() if k != "elapsed_seconds"}),
    }


def scripted_step(episode, messages, tool, arguments=None):
    """Apply one authored action through the real episode; never a model action."""
    arguments = arguments or {}
    observation = episode.step({"tool": tool, **arguments})
    if "error" in observation:
        raise ValueError("Scripted action was rejected: " + observation["error"])
    messages.extend([assistant_call(tool, arguments), tool_message(tool, observation)])
    return observation


def retained_repair(source):
    """The observed failure shape: the faulty logic kept, with a redundant final line."""
    return source + source.splitlines(keepends=True)[-1]
