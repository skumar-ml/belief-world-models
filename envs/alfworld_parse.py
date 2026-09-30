"""Shared ALFWorld action parsing and observation cleanup.

Used by the eval env ([AlfWorldEnv]).
"""

import re


# Substring the underlying ALFWorld env returns when a parseable action is
# invalid / a no-op. Repeated occurrences signal the agent is stuck in a loop.
NOOP_MARKER = "Nothing happens"

# Exact format-error observation used by AlfWorldEnv.step on parse failure.
FORMAT_ERROR_OBS = "Observation: Error Input. Your input must contains 'Action: '"

# Generation stop strings from BaseAgent; also stripped if the model emits them.
STOP_WORDS = ["\nObservation:", "\nTask:", "\n---"]

_ACTION_RE = re.compile(r"Action:\s?(.*)", re.DOTALL)
_PUT_RE = re.compile(r"put\s+(.*)\s+[io]n\s+(.*)")


def process_ob(ob: str) -> str:
    # Strip the "You arrive at loc ..." prefix the env prepends after a goto.
    if ob.startswith("You arrive at loc "):
        ob = ob[ob.find(". ") + 2 :]
    return ob


def strip_reset_observation(obs: str) -> str:
    # Match AlfWorldTask.load_tasks: drop the first paragraph of the reset blob.
    parts = obs.split("\n\n")
    if len(parts) > 1:
        obs = "\n".join(parts[1:])
    return process_ob(obs)


def apply_stop_words(text: str, stop_words=None) -> str:
    # Truncate at the earliest stop marker so rollouts match vLLM stop behavior.
    if stop_words is None:
        stop_words = STOP_WORDS
    cut = len(text)
    for marker in stop_words:
        idx = text.find(marker)
        if idx != -1:
            cut = min(cut, idx)
    return text[:cut]


def parse_action(llm_output: str) -> str:
    # Extract the text after "Action:" from the ReAct/ReflAct output.
    llm_output = (llm_output or "").strip()
    matches = _ACTION_RE.findall(llm_output)
    if not matches:
        raise ValueError("LLM output does not contain 'Action:'")
    action = matches[0]
    # Normalize "put X in/on Y" to the exact form the env expects.
    put_action = _PUT_RE.findall(action)
    if put_action:
        action = f"put {put_action[0][0]} in/on {put_action[0][1]}"
    if action is None:
        raise ValueError("Parsed action is empty")
    return action
