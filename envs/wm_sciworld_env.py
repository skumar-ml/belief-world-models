"""WM-augmented SciWorld env (room-level belief, query-only).

Wraps `SciWorldEnv` with a decoupled room-level location world model. The agent gets
one extra action — `query <q>` — answered from belief folded from text observations.
Queries do not consume environment steps.
"""

import logging
import re
from typing import Tuple

from envs.sciworld_env import SciWorldEnv
from utils.datatypes import State
from wm.sciworld_belief import SciWorldBeliefState
from wm.sciworld_goal import parse_goal, parse_stated_locations
from wm.sciworld_query import answer_query


logger = logging.getLogger("agent_eval")

_QUERY_RE = re.compile(r"^\s*query\b[:\s]*(.*)$", re.IGNORECASE | re.DOTALL)


class WMSciWorldEnv(SciWorldEnv):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.belief: SciWorldBeliefState = None
        self.max_consecutive_queries = 10
        self._consecutive_queries = 0
        self._last_action = None

    def reset(self) -> Tuple[str, State]:
        observation, state = super().reset()
        task_desc = self.env.getTaskDescription() if hasattr(self.env, "getTaskDescription") else ""
        try:
            self.belief = SciWorldBeliefState(
                parse_goal(task_desc),
                stated_locations=parse_stated_locations(task_desc),
            )
        except Exception as e:
            self.belief = None
            logger.warning(f"WM disabled for this task (goal unparseable): {e}")
        return observation, state

    def step(self, llm_output: str) -> Tuple[str, State]:
        self.state.history.append({"role": "assistant", "content": llm_output})

        action_text = self._extract_action(llm_output)
        q = _QUERY_RE.match(action_text) if action_text is not None else None

        if q is not None:
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
        return super().record_observation(observation, reward, done, info)

    def _extract_action(self, llm_output: str):
        m = re.search(r"Action:\s?(.*)", llm_output.strip(), re.DOTALL)
        return m.group(1).strip() if m else None
