import enum
from copy import deepcopy
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict


class State:
    """This should contains everything needed to continue the conversation.

    For example, the history of the conversation, the current task (success/failure) at each step, etc.
    """

    def __init__(
        self,
        reward: float = None,
        finished: bool = False,
        success: bool = False,
        terminate_reason: str = None,
    ):
        """
        The history should be a format like:
        [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Who won the world series in 2020?"},
            {"role": "assistant", "content": "The Los Angeles Dodgers won the World Series in 2020."},
            {"role": "user", "content": "Where was it played?"}
        ]
        """
        self.history: List[Dict[str, Any]] = []  # running list of role/content turns
        self.reward: float = reward
        self.finished: bool = finished            # episode over?
        self.success: bool = success              # goal achieved?
        self.terminate_reason: str = terminate_reason
        self.error: Optional[str] = None
        self.steps = 0                            # number of env steps taken
        # Per-task env step budget (SciWorld MPO max_steps, 10-120). Recorded so
        # AR-per-step can charge the full budget to episodes that broke out early
        # (max_steps / noop_loop). None for envs/runs that don't set it.
        self.max_steps = None
        # FoV-only: no-ops caused by interacting with an undiscovered receptacle
        # (the field-of-view gate), tracked separately from real env no-ops.
        self.gated_steps = 0
        # M3 WM-only: world-model queries the agent issued (do NOT count as env steps;
        # tracked separately so env-steps-to-goal stays comparable to the no-WM baseline).
        self.query_steps = 0
        # WALL-E-only: actions rejected "in imagination" by the rule gate. Do NOT count as
        # env steps (see envs/walle_alfworld_env.py); tracked separately like query_steps.
        self.walle_rejections = 0

    @classmethod
    def load_json(cls, json_dict: Dict[str, Any]):
        # Inverse of to_dict: all but the last entry is history; the last holds metadata.
        state = cls()
        state.history = json_dict[:-1]
        info = json_dict[-1]
        state.reward = info["reward"]
        state.finished = info["finished"]
        state.success = info["success"]
        state.terminate_reason = info["terminate_reason"]
        state.error = info["error"]
        state.steps = info["steps"]
        # Backward-compatible: pre-FoV / pre-WM runs won't have these keys.
        state.gated_steps = info.get("gated_steps", 0)
        state.query_steps = info.get("query_steps", 0)
        state.walle_rejections = info.get("walle_rejections", 0)
        state.max_steps = info.get("max_steps", None)
        return state

    @property
    def empty(self):
        # True before any turn has been added to history.
        return len(self.history) == 0

    def to_dict(self) -> Dict[str, Any]:
        # Serialize as history with a trailing metadata record appended.
        history = deepcopy(self.history)
        history.append({
            "steps": self.steps,
            "reward": self.reward,
            "finished": self.finished,
            "success": self.success,
            "terminate_reason": self.terminate_reason,
            "error": self.error,
            "gated_steps": self.gated_steps,
            "query_steps": self.query_steps,
            "walle_rejections": self.walle_rejections,
            "max_steps": self.max_steps,
        })
        return history
