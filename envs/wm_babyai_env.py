"""WM-augmented BabyAI-Text env (6x6 placement prior, query-only).

Wraps `BabyAIEnv` with a decoupled parse-from-text world model. The agent gets
one extra action — `query <q>` — answered from the room-prior belief folded
from the same text observations the policy sees. Queries do not consume
environment steps.

No MiniGrid oracle: pose is odometry in the start-pose frame. Once wall rays
pin the known 6x6 interior, unlocated referents get a uniform categorical over
legal cells.
"""

import logging
import re
from typing import Tuple

from envs.babyai_env import BabyAIEnv
from utils.datatypes import State
from wm.babyai_goal import parse_goal
from wm.babyai_prior_belief import BabyAIPriorBeliefState
from wm.babyai_prior_query import answer_query


logger = logging.getLogger("agent_eval")

_QUERY_RE = re.compile(r"^\s*query\b[:\s]*(.*)$", re.IGNORECASE | re.DOTALL)


def _extract_first_obs(observation: str, mission: str) -> str:
    """Pull the first FoV body out of the zero-shot reset prompt."""
    marker = f"Your task is to: {mission}\n"
    idx = observation.find(marker)
    if idx < 0:
        return ""
    return observation[idx + len(marker) :].strip()


class WMBabyAIEnv(BabyAIEnv):
    """Query-only wrapper with the 6x6 placement-prior belief."""

    belief_cls = BabyAIPriorBeliefState
    answer_query_fn = staticmethod(answer_query)

    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.belief: BabyAIPriorBeliefState = None
        self._track_belief = True
        self.max_consecutive_queries = 10
        self._consecutive_queries = 0
        self._last_action = None

    def reset(self) -> Tuple[str, State]:
        observation, state = super().reset()
        self._consecutive_queries = 0
        self._last_action = None
        try:
            self.belief = self.belief_cls(
                parse_goal(self._mission or ""),
                track_belief=self._track_belief,
            )
            first = _extract_first_obs(observation, self._mission or "")
            if first:
                self.belief.observe("", first)
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
                answer = self.answer_query_fn(self.belief, q.group(1).strip())
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

    def record_observation(
        self,
        observation,
        reward,
        done,
        info=None,
        terminated=False,
        truncated=False,
    ):
        # Fold the raw env body (no 'Observation:' prefix) before the parent wraps it.
        if self.belief is not None and self._last_action is not None:
            try:
                self.belief.observe(self._last_action, observation)
            except Exception:
                pass
        self._last_action = None
        return super().record_observation(
            observation, reward, done, info, terminated=terminated, truncated=truncated
        )

    def _extract_action(self, llm_output: str):
        m = re.search(r"Action:\s?(.*)", llm_output.strip(), re.DOTALL)
        return m.group(1).strip() if m else None
