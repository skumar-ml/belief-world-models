"""Shared BabyAI-Text action parsing and observation cleanup.

Mirrors BALROG's BabyAI-Text wrapper (balrog/environments/babyai_text):
the six language actions, description formatting (strip "You see "), and
the default fallback they would execute on an invalid string. We do NOT
execute that fallback — invalid / unparseable outputs stay no-step errors.
"""

import re


# Exact language actions BALROG exposes (order matches MiniGrid 0-5).
BABYAI_ACTION_SPACE = [
    "turn left",
    "turn right",
    "go forward",
    "pick up",
    "drop",
    "toggle",
]

# BALROG's default fallback if they accepted an invalid string. We never step it;
# it is documented here so a later WALL-E / oracle variant can reuse the same token.
BALROG_DEFAULT_ACTION = "go forward"

# Substring we inject when a parseable Action: is not one of the six commands.
NOOP_MARKER = "Nothing happens"

# Exact format-error observation used by BabyAIEnv.step on parse failure.
FORMAT_ERROR_OBS = "Observation: Error Input. Your input must contain 'Action: '"

# Exact invalid-action observation (parsed, but not in BABYAI_ACTION_SPACE).
INVALID_ACTION_OBS = "Observation: Nothing happens. That is not a valid action."

_ACTION_RE = re.compile(r"Action:\s?(.*)", re.DOTALL)


def format_text_obs(info: dict) -> str:
    """Turn MiniGrid info['descriptions'] into BALROG's text observation.

    BartekCupial/Minigrid puts BabyAI-Text sentences in info['descriptions'];
    BALROG strips the leading "You see " and joins with newlines.
    """
    descriptions = info.get("descriptions") or []
    return "\n".join(d.replace("You see ", "") for d in descriptions)


def parse_action(llm_output: str) -> str:
    """Extract the text after 'Action:' from a ReAct/ReflAct output."""
    llm_output = llm_output.strip()
    matches = _ACTION_RE.findall(llm_output)
    if not matches:
        raise ValueError("LLM output does not contain 'Action:'")
    action = matches[0].strip().split("\n", 1)[0].strip().lower()
    if not action:
        raise ValueError("Parsed action is empty")
    return action


def action_index(action: str) -> int:
    """Map a language action to the MiniGrid discrete index, or raise."""
    if action not in BABYAI_ACTION_SPACE:
        raise ValueError(f"Action {action!r} is not in the BabyAI action space")
    return BABYAI_ACTION_SPACE.index(action)
