"""Composed oracle WALL-E gate + placement-prior WM for BabyAI-Text.

Mirrors SciWorld (`WalleWMOracleSciWorldEnv`): the oracle gate governs physical
`go forward` / `pick up` and go-backward / go-left / go-right aliases, while `query` and unparseable
output pass through to `WMBabyAIEnv` (query short-circuit + belief fold).
Rejected actions never step MiniGrid and never fold into belief.

MRO: WalleWMOracleBabyAIEnv -> WalleOracleBabyAIEnv -> WMBabyAIEnv -> BabyAIEnv.
`record_observation` is the WM override. Accepted actions stash `_last_action`
before commit so the fold sees the real env body.
"""

from typing import Tuple

from envs.babyai_parse import BABYAI_ACTION_SPACE
from envs.walle_oracle_babyai_env import WalleOracleBabyAIEnv
from envs.wm_babyai_env import WMBabyAIEnv, _QUERY_RE
from utils.datatypes import State
from wm.walle.babyai_oracle import check_babyai_action


class WalleWMOracleBabyAIEnv(WalleOracleBabyAIEnv, WMBabyAIEnv):
    """Oracle validity gate composed with the 6x6 placement-prior belief WM."""

    def step(self, llm_output: str) -> Tuple[str, State]:
        # Query / format errors are not physical actions: WM owns those paths.
        try:
            action = self.parse_action(llm_output)
        except Exception:
            self._consecutive_rejections = 0
            return WMBabyAIEnv.step(self, llm_output)

        action_text = self._extract_action(llm_output)
        if action_text is not None and _QUERY_RE.match(action_text):
            self._consecutive_rejections = 0
            return WMBabyAIEnv.step(self, llm_output)

        # Retry cap: commit via the WM step so the episode cannot stall.
        if self._consecutive_rejections >= self.walle_max_retries:
            self._consecutive_rejections = 0
            return WMBabyAIEnv.step(self, llm_output)

        # Gate in-space physics and illegal movement aliases before MiniGrid.
        feedback = check_babyai_action(self.env, action)
        if feedback is not None:
            return self._emit_rejection(llm_output, feedback)

        if action not in BABYAI_ACTION_SPACE:
            self._consecutive_rejections = 0
            return WMBabyAIEnv.step(self, llm_output)

        # Accepted: real MiniGrid step. Clear the query-loop streak (WMBabyAIEnv.step
        # is bypassed here, so it cannot reset _consecutive_queries itself).
        self._last_action = action
        self.state.history.append({"role": "assistant", "content": llm_output})
        self._consecutive_rejections = 0
        self._consecutive_queries = 0
        observation, reward, done, info, terminated, truncated = self.conduct_action(
            action
        )
        return self.record_observation(
            observation, reward, done, info, terminated=terminated, truncated=truncated
        )
