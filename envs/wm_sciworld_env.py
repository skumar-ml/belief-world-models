"""WM-augmented SciWorld env (room-level belief).

Wraps `SciWorldEnv` with a decoupled room-level location world model. Two delivery
modes (`push_mode` in the task config):

  - `"none"` (default, permutation `wm`): the agent gets one extra action —
    `query <q>` — answered from belief folded from text observations. A `query`
    does NOT step the env and does NOT count as an env step; it increments
    `state.query_steps` instead.
  - `"belief"` (permutation `wm_push`): no `query` action. After the initial task
    and every real env step, the current belief is appended as a `[World model]`
    line (room, inventory, focus, and where-is on the goal target plus objects
    seeded from the task description).
"""

import logging
import re
from typing import Tuple

from envs.sciworld_env import SciWorldEnv
from utils.datatypes import State
from wm.sciworld_belief import SciWorldBeliefState
from wm.sciworld_goal import parse_goal, parse_stated_locations
from wm.sciworld_query import answer_query, format_push


logger = logging.getLogger("agent_eval")

_QUERY_RE = re.compile(r"^\s*query\b[:\s]*(.*)$", re.IGNORECASE | re.DOTALL)


class WMSciWorldEnv(SciWorldEnv):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.belief: SciWorldBeliefState = None
        self._track_belief = True
        # "none" = query-only (wm); "belief" = always-on push, no query (wm_push).
        self.push_mode = kwargs.get("push_mode", "none")
        self.max_consecutive_queries = 10
        self._consecutive_queries = 0
        self._last_action = None

    def _push_enabled(self) -> bool:
        return self.push_mode not in (None, "", "none")

    def _with_push(self, text: str) -> str:
        return f"{text}\n[World model] {format_push(self.belief)}"

    def reset(self) -> Tuple[str, State]:
        observation, state = super().reset()
        task_desc = self.env.getTaskDescription() if hasattr(self.env, "getTaskDescription") else ""
        try:
            self.belief = SciWorldBeliefState(
                parse_goal(task_desc),
                stated_locations=parse_stated_locations(task_desc),
                track_belief=self._track_belief,
            )
        except Exception as e:
            self.belief = None
            logger.warning(f"WM disabled for this task (goal unparseable): {e}")
        if self._push_enabled() and self.belief is not None:
            observation = self._with_push(observation)
            if state.history:
                state.history[-1]["content"] = observation
        return observation, state

    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append({"role": "assistant", "content": llm_output})

        action_text = self._extract_action(llm_output)
        q = _QUERY_RE.match(action_text) if action_text is not None else None

        # Push mode has no query action: fall through to the real env (invalid
        # `query` becomes a no-op like any other unrecognized verb).
        if q is not None and not self._push_enabled():
            if self.belief is None:
                answer = "World model is unavailable for this task."
            else:
                answer = answer_query(self.belief, q.group(1).strip())
            self.state.history.append({"role": "user", "content": f"Observation: {answer}"})
            self.state.query_steps += 1
            self._consecutive_queries += 1
            if self._consecutive_queries >= self.max_consecutive_queries:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "query_loop"
                if self.state.reward is None:
                    self.state.reward = 0
            return f"Observation: {answer}", self.state

        self._consecutive_queries = 0
        self.state.history.pop()
        try:
            self._last_action = self.parse_action(llm_output)
        except Exception:
            self._last_action = None
        return super().step(llm_output)

    def record_observation(self, observation, reward, done, info):
        if self.belief is not None and self._last_action is not None:
            try:
                self.belief.observe(self._last_action, observation)
            except Exception:
                pass
        self._last_action = None
        if self._push_enabled() and self.belief is not None:
            observation = self._with_push(observation)
        return super().record_observation(observation, reward, done, info)

    def _extract_action(self, llm_output: str):
        m = re.search(r"Action:\s?(.*)", (llm_output or "").strip(), re.DOTALL)
        return m.group(1).strip() if m else None


class DeterministicWMSciWorldEnv(WMSciWorldEnv):
    """Ablation: WM tracks only observed state (no room prior / stated-location mass)."""

    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self._track_belief = False
