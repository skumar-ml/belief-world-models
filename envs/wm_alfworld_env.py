"""WM-augmented standard ALFWorld env.

Wraps `AlfWorldEnv` with a decoupled belief-state world model. Two delivery modes
(`push_mode` in the task config):

  - `"none"` (default, permutation `wm`): the agent gets one extra action —
    `query <q>` — answered from belief folded from the same text observations the
    agent sees (parse-from-text only, no oracle). A `query` does NOT step the
    underlying env and does NOT count as an env step; it increments
    `state.query_steps` instead.
  - `"belief"` (permutation `wm_push`): no `query` action. After the initial task
    observation and every real env step, the current belief is appended as a
    `[World model]` line (location, inventory, and where-is on the goal target).

Supported queries when `push_mode` is `"none"` (case-insensitive, after `query`):
  - "where is <object_type>"        -> non-searched receptacles (ranked by P, or a
                                     random list with no probabilities)
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
from wm.query import answer_query, format_push
from wm.scene import RECEP_BLOB_RE, parse_receptacles


logger = logging.getLogger("agent_eval")

_QUERY_RE = re.compile(r"^\s*query\b[:\s]*(.*)$", re.IGNORECASE | re.DOTALL)


class WMAlfWorldEnv(AlfWorldEnv):
    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.belief: BeliefState = None
        self._track_belief = True
        self._report_probabilities = True
        # uniform keeps the affinity prior; zipf seeds it from the shared rank table.
        self._placement = kwargs.get("placement", "uniform")
        # "none" = query-only (wm); "belief" = always-on push, no query (wm_push).
        self.push_mode = kwargs.get("push_mode", "none")
        self.max_consecutive_queries = 10
        self._consecutive_queries = 0
        self._last_action = None

    def _push_enabled(self) -> bool:
        return self.push_mode not in (None, "", "none")

    def reset(self) -> Tuple[str, State]:
        observation, state = super().reset()
        raw = self.task.observation
        blob = RECEP_BLOB_RE.search(raw)
        receptacles = parse_receptacles(blob.group(1)) if blob else []
        goal_match = re.search(r"(Your task is to:.*)$", raw, re.DOTALL)
        goal_line = goal_match.group(1).strip() if goal_match else raw
        try:
            self.belief = BeliefState(
                parse_goal(goal_line),
                receptacles,
                track_belief=self._track_belief,
                report_probabilities=self._report_probabilities,
                placement=self._placement,
            )
        except Exception as e:
            self.belief = None
            logger.warning(f"WM disabled for this task (goal unparseable): {e}")
        if self._push_enabled() and self.belief is not None:
            observation = self._with_push(observation)
            if state.history:
                state.history[-1]["content"] = self._with_push(state.history[-1]["content"])
        return observation, state

    def _with_push(self, text: str) -> str:
        return f"{text}\n[World model] {format_push(self.belief)}"

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
        if self._push_enabled() and self.belief is not None:
            observation = self._with_push(observation)
        return super().record_observation(observation, reward, done)

    def _extract_action(self, llm_output: str):
        m = re.search(r"Action:\s?(.*)", (llm_output or "").strip(), re.DOTALL)
        return m.group(1).strip() if m else None


class DeterministicWMAlfWorldEnv(WMAlfWorldEnv):
    """Ablation: WM tracks only deterministic state (no probabilistic belief)."""

    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self._track_belief = False


class NoProbWMAlfWorldEnv(WMAlfWorldEnv):
    """Belief WM whose where-is answers omit probabilities.

    The candidate set is unchanged (allowed, not-yet-searched receptacles). The
    answer lists every candidate in random order, with no masses and no ranking.
    """

    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self._report_probabilities = False
