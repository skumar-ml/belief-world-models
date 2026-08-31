"""WM-augmented standard ALFWorld env.

Wraps `AlfWorldEnv` with a decoupled belief-state world model. The agent gets one
extra action — `query <q>` — answered by the world model from belief it has folded
from the same text observations the agent sees (parse-from-text only, no oracle).

A `query` does NOT step the underlying env and does NOT count as an env step; it
increments `state.query_steps` instead.

Supported queries (case-insensitive, after the `query` keyword):
  - "where is <object_type>"        -> ranked non-searched receptacles by P(target)
  - "what is in <receptacle>" / "what's in <receptacle>"
  - "have i searched <receptacle>" / "searched <receptacle>"
  - "state" / "status"              -> compact structured state dump
"""

import logging
import re
from typing import Tuple

from envs.alfworld_env import AlfWorldEnv
from utils.datatypes import State
from wm.belief import BeliefState
from wm.goal_parser import parse_goal
from wm.query import answer_query
from wm.scene import RECEP_BLOB_RE, parse_receptacles


logger = logging.getLogger("agent_eval")

_QUERY_RE = re.compile(r"^\s*query\b[:\s]*(.*)$", re.IGNORECASE | re.DOTALL)


class WMAlfWorldEnv(AlfWorldEnv):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.belief: BeliefState = None
        self._track_belief = True
        self.max_consecutive_queries = 10
        self._consecutive_queries = 0
        self._last_action = None

    def reset(self) -> Tuple[str, State]:
        observation, state = super().reset()
        raw = self.task.observation
        blob = RECEP_BLOB_RE.search(raw)
        receptacles = parse_receptacles(blob.group(1)) if blob else []
        goal_match = re.search(r"(Your task is to:.*)$", raw, re.DOTALL)
        goal_line = goal_match.group(1).strip() if goal_match else raw
        try:
            self.belief = BeliefState(
                parse_goal(goal_line), receptacles, track_belief=self._track_belief
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
                self.state.reward = 0
            return f"Observation: {answer}", self.state

        self._consecutive_queries = 0
        self.state.history.pop()
        try:
            self._last_action = self.parse_action(llm_output)
        except Exception:
            self._last_action = None
        return super().step(llm_output)

    def record_observation(self, observation, reward, done):
        if self.belief is not None and self._last_action is not None:
            try:
                self.belief.observe(self._last_action, observation)
            except Exception:
                pass
        self._last_action = None
        return super().record_observation(observation, reward, done)

    def _extract_action(self, llm_output: str):
        m = re.search(r"Action:\s?(.*)", llm_output.strip(), re.DOTALL)
        return m.group(1).strip() if m else None


class DeterministicWMAlfWorldEnv(WMAlfWorldEnv):
    """Ablation: WM tracks only deterministic state (no probabilistic belief)."""

    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self._track_belief = False
