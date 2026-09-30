"""Oracle WALL-E gate for BabyAI-Text: reject infeasible go-forward / pick-up.

Mirrors the ALFWorld / SciWorld WALL-E accounting: a rejected action is answered
in imagination with [World model] feedback, does not call env.step, and bumps
state.walle_rejections instead of state.steps. After walle_max_retries consecutive
rejections at one decision point the action is let through so the episode cannot
stall in imagination.

Validity is an oracle over MiniGrid state (see wm/walle/babyai_oracle.py), not
parse-from-text. Format errors and unknown verbs (except go-backward / go-left /
go-right aliases) still go through BabyAIEnv._reject and count as env steps.
"""

from typing import Tuple

from envs.babyai_env import BabyAIEnv
from envs.babyai_parse import BABYAI_ACTION_SPACE
from utils.datatypes import State
from wm.walle.babyai_oracle import check_babyai_action


class WalleOracleBabyAIEnv(BabyAIEnv):
    """BabyAI + oracle validity gate for go forward, pick up, and illegal move aliases."""

    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.walle_max_retries = kwargs.get("walle_max_retries", 5)
        self._consecutive_rejections = 0

    def reset(self) -> Tuple[str, State]:
        self._consecutive_rejections = 0
        return super().reset()

    def _emit_rejection(self, llm_output: str, feedback: str) -> Tuple[str, State]:
        """Record the rejected turn + imagination obs; do not step MiniGrid."""
        self.state.history.append({"role": "assistant", "content": llm_output})
        answer = f"[World model] {feedback}"
        observation = f"Observation: {answer}"
        self.state.history.append({"role": "user", "content": observation})
        self.state.walle_rejections += 1
        self._consecutive_rejections += 1
        return observation, self.state

    def step(self, llm_output: str) -> Tuple[str, State]:
        # Malformed output: parent's format-error path (counts as a step).
        try:
            action = self.parse_action(llm_output)
        except Exception:
            self._consecutive_rejections = 0
            return super().step(llm_output)

        # Retry cap: commit the no-op so the episode cannot wedge in imagination.
        if self._consecutive_rejections >= self.walle_max_retries:
            self._consecutive_rejections = 0
            return super().step(llm_output)

        # Gate in-space physics and illegal movement aliases before MiniGrid.
        feedback = check_babyai_action(self.env, action)
        if feedback is not None:
            return self._emit_rejection(llm_output, feedback)

        if action not in BABYAI_ACTION_SPACE:
            self._consecutive_rejections = 0
            return super().step(llm_output)

        # Allowed: record the agent turn, step MiniGrid, commit the observation.
        self.state.history.append({"role": "assistant", "content": llm_output})
        self._consecutive_rejections = 0
        observation, reward, done, info, terminated, truncated = self.conduct_action(
            action
        )
        return self.record_observation(
            observation, reward, done, info, terminated=terminated, truncated=truncated
        )
