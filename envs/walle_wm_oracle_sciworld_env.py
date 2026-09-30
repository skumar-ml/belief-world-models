"""Composed WALL-E parser-validity gate + belief-state WM for ScienceWorld.

Mirrors the ALFWorld composition (envs/walle_alfworld_env.py WalleWMAlfWorldEnv): the WALL-E
gate governs physical actions, while `query` and unparseable output pass through to the
belief-WM machinery (WMSciWorldEnv: query short-circuit + belief fold + optional push).

The gate here is the v2 reject-signal gate (see envs/walle_oracle_sciworld_env.py): validity
is discovered by stepping the real env via conduct_action() and checking for the
"No known action matches that input." no-op marker, NOT by membership in
getValidActionObjectCombinations(). A rejected action is a genuine engine no-op, so belief is
never polluted by it -- belief only folds ACCEPTED (real) observations, which is correct.

MRO: [WalleWMOracleSciWorldEnv, WalleOracleSciWorldEnv, WMSciWorldEnv, SciWorldEnv]. So
self.record_observation resolves to WMSciWorldEnv's override (belief fold + push), and the
inherited _emit_rejection / _consecutive_rejections come from WalleOracleSciWorldEnv.

Subtlety: WMSciWorldEnv.record_observation folds self._last_action, normally set inside
WMSciWorldEnv.step. When the gate commits an accepted action directly it bypasses that step,
so we stash self._last_action here before committing, or belief never folds accepted obs.
"""

import logging
from typing import Tuple

from envs.sciworld_env import _NOOP_MARKER
from envs.walle_oracle_sciworld_env import WalleOracleSciWorldEnv
from envs.wm_sciworld_env import WMSciWorldEnv, _QUERY_RE
from utils.datatypes import State


logger = logging.getLogger("agent_eval")


class WalleWMOracleSciWorldEnv(WalleOracleSciWorldEnv, WMSciWorldEnv):
    """Reject-signal validity gate composed with our belief-state WM.

    ``push_mode="none"`` is query + gate (``walle_oracle_wm``).
    ``push_mode="belief"`` is always-on push + gate (no query).
    """

    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        # walle_max_retries / _consecutive_rejections set by WalleOracleSciWorldEnv.__init__.

    def step(self, llm_output: str) -> Tuple[str, State]:
        # `query` and unparseable output are not physical actions: hand off to the belief-WM
        # step, which owns query short-circuit (query_steps) and belief fold/push on real steps.
        try:
            action_str = self.parse_action(llm_output)
        except Exception:
            self._consecutive_rejections = 0
            return WMSciWorldEnv.step(self, llm_output)

        action_text = self._extract_action(llm_output)
        if action_text is not None and _QUERY_RE.match(action_text):
            self._consecutive_rejections = 0
            return WMSciWorldEnv.step(self, llm_output)

        # Retry cap: after too many rejections at this decision point, commit via the belief-WM
        # step so the episode can't wedge in imagination (the action is a no-op, costs one step).
        if self._consecutive_rejections >= self.walle_max_retries:
            self._consecutive_rejections = 0
            return WMSciWorldEnv.step(self, llm_output)

        # Step the real env WITHOUT mutating self.state; the parser is the validity oracle.
        observation, reward, done, info = self.conduct_action(action_str)

        # Parser-invalid -> no-op on the engine; convert to a [World model] re-prompt, no commit.
        if _NOOP_MARKER in observation:
            return self._emit_rejection(llm_output, action_str)

        # Parser accepted: commit normally. Stash _last_action so WMSciWorldEnv.record_observation
        # folds it into belief, record the agent turn, then commit via the belief-WM override.
        self._last_action = action_str
        self.state.history.append({"role": "assistant", "content": llm_output})
        self._consecutive_rejections = 0
        return self.record_observation(observation, reward, done, info)
