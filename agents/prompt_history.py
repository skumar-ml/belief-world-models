"""Prompt-only history shaping for the eval loop.

`state.history` (and the JSON log) stays complete. This module builds the
message list sent to the API: BALROG-style last-N observations, plus optional
compaction of WALL-E imagination rejects.

BALROG's HistoryPromptBuilder keeps the instruction always and a deque of the
last `max_text_history` observations (default 16) with their actions. We do
the same: pin history[0] (instruction + mission + first FoV) and keep the last
`max_obs` user observations from the rest, with the assistant turns that
precede them.

WALL-E compaction: drop every [World model] reject pair except the latest
trailing one (the current decision point). Older rejects that were followed
by a real env step are omitted from the prompt.
"""

from typing import Any, Dict, List, Optional


WM_MARKER = "[World model]"


def _obs_body(msg: Dict[str, Any]) -> str:
    """Env/user turn body, without the 'Observation: ' prefix."""
    content = str(msg.get("content", "")).strip()
    if content.startswith("Observation:"):
        content = content[len("Observation:"):].strip()
    return content


def _is_walle_obs(msg: Dict[str, Any]) -> bool:
    """True when this user turn is a WALL-E imagination reject.

    Rejects are a whole observation that *starts* with [World model]. An always-on
    belief push is appended *after* a real env observation, so it must not match.
    """
    return msg.get("role") == "user" and _obs_body(msg).startswith(WM_MARKER)


def compact_walle_rejects(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Keep real env turns; at most one trailing WALL-E reject pair."""
    pairs: List[List[Dict[str, Any]]] = []
    i = 0
    while i < len(messages):
        nxt = messages[i + 1] if i + 1 < len(messages) else None
        if (
            nxt is not None
            and messages[i].get("role") == "assistant"
            and nxt.get("role") == "user"
        ):
            pairs.append([messages[i], nxt])
            i += 2
        else:
            pairs.append([messages[i]])
            i += 1

    # A real env observation clears any pending reject; only a trailing reject is kept.
    real: List[List[Dict[str, Any]]] = []
    trailing_walle = None
    for pair in pairs:
        if len(pair) == 2 and _is_walle_obs(pair[1]):
            trailing_walle = pair
        else:
            trailing_walle = None
            real.append(pair)

    out: List[Dict[str, Any]] = []
    for pair in real:
        out.extend(pair)
    if trailing_walle is not None:
        out.extend(trailing_walle)
    return out


def _window_tail(tail: List[Dict[str, Any]], max_obs: int) -> List[Dict[str, Any]]:
    """Keep the last `max_obs` user observations and the messages between them."""
    kept: List[Dict[str, Any]] = []
    n_obs = 0
    for msg in reversed(tail):
        kept.append(msg)
        if msg.get("role") == "user":
            n_obs += 1
            if n_obs >= max_obs:
                break
    kept.reverse()
    return kept


def build_prompt_messages(
    history: List[Dict[str, Any]],
    max_obs: Optional[int] = None,
    compact_walle: bool = False,
) -> List[Dict[str, Any]]:
    """API view of `history`. None/False options leave that stage unchanged."""
    if not history:
        return history
    header, tail = history[0], list(history[1:])
    if compact_walle:
        tail = compact_walle_rejects(tail)
    if max_obs is not None:
        tail = _window_tail(tail, max_obs)
    return [header] + tail
