"""WALL-E baseline envs: rule-based action-validity gate on top of ALFWorld.

WALL-E 2.0 (arXiv 2504.15785) treats action feasibility as its "world model": a set of
learned rules `f(state, action) -> (feedback, success, suggestion)` flag infeasible
actions. Here we run WALL-E's *released* ALFWorld rule set (see wm/walle/SOURCE.md) as a
pre-action gate. Before an action reaches the real env, the gate parses the current
state + action and runs the rules; a rejected action is answered "in imagination" with
the rule's feedback/suggestion and never touches the env — the agent must propose a
different action. This is orthogonal to our belief-state WM (wm/belief.py, which answers
`query` about hidden current state); WalleWMAlfWorldEnv composes both.

Accounting (decision 2026-07-07): imagination rejections do NOT count as env steps; they
increment state.walle_rejections instead, so env-steps-to-goal stays comparable to the
no-WALL-E baseline (mirrors how WM `query` uses query_steps). Consecutive rejections at a
single decision point are capped by `walle_max_retries`; past the cap the action is let
through (act anyway) so the episode can't stall in imagination.
"""

import logging
import re
from typing import Tuple

from envs.alfworld_env import AlfWorldEnv
from envs.wm_alfworld_env import WMAlfWorldEnv
from utils.datatypes import State
from wm.walle import check_action, convert_action, load_rules, walle_state_transform


logger = logging.getLogger("agent_eval")


class _WalleGateMixin:
    """Shared WALL-E gate logic for the plain and WM-composed envs.

    Provides `_gate_init` (call from __init__) and `_run_walle_gate` (call at the top of
    step). The gate returns a verdict so each env decides how to delegate the accepted
    action to its own parent's step().
    """

    def _gate_init(self, **kwargs):
        self.walle_max_retries = kwargs.get("walle_max_retries", 5)
        self._rules = load_rules()
        # Consecutive rejections at the current decision point (reset on accept).
        self._consecutive_rejections = 0

    def _run_walle_gate(self, llm_output: str):
        """Classify the agent's output against the rule gate.

        Returns (verdict, action_str):
          - "passthrough": not parseable, or not a physical WALL-E verb (think/look/query)
            or the retry cap was hit -> caller should delegate to its parent step().
          - "accept": a physical action the rules allow -> delegate to parent step().
          - "reject": a physical action the rules block -> caller handles imagination reply.
        The reject payload (feedback/suggestion) is stashed on self._last_gate_result.
        """
        try:
            action_str = self.parse_action(llm_output)
        except Exception:
            return "passthrough", None  # let parent bad-step path handle it

        action_dict = convert_action(action_str)
        if action_dict is None:
            # think:/look/query and anything outside WALL-E's 9-verb schema: no gate.
            return "passthrough", action_str

        # Cap: if we've already rejected this decision point too many times, act anyway.
        if self._consecutive_rejections >= self.walle_max_retries:
            self._consecutive_rejections = 0
            return "passthrough", action_str

        state_dict = walle_state_transform(self.state.history)
        result = check_action(state_dict, action_dict, self._rules)
        if result["success"]:
            self._consecutive_rejections = 0
            return "accept", action_str
        self._last_gate_result = result
        return "reject", action_str

    def _emit_rejection(self, llm_output: str):
        """Record the rejected turn + an imagination observation; no env step.

        In push mode the belief line is appended after the rejection so the
        latest user turn still carries location / inventory / where-is. The
        rejection text itself still starts with ``[World model]``, which
        ``walle_state_transform`` uses to skip imagination turns.
        """
        result = self._last_gate_result
        self.state.history.append({"role": "assistant", "content": llm_output})
        suggestion = result.get("suggestion", "") or ""
        answer = f"[World model] {result['feedback']} {suggestion}".strip()
        observation = f"Observation: {answer}"
        if getattr(self, "_push_enabled", lambda: False)() and getattr(self, "belief", None) is not None:
            observation = self._with_push(observation)
        self.state.history.append({"role": "user", "content": observation})
        self.state.walle_rejections += 1
        self._consecutive_rejections += 1
        return observation, self.state


class WalleAlfWorldEnv(_WalleGateMixin, AlfWorldEnv):
    """Pure WALL-E baseline: action-validity gate, no belief/query WM."""

    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self._gate_init(**kwargs)

    def step(self, llm_output: str) -> Tuple[str, State]:
        verdict, _ = self._run_walle_gate(llm_output)
        if verdict == "reject":
            return self._emit_rejection(llm_output)
        # accept / passthrough: hand off to the standard env step unchanged.
        return AlfWorldEnv.step(self, llm_output)


class WalleWMAlfWorldEnv(_WalleGateMixin, WMAlfWorldEnv):
    """WALL-E gate composed with our belief-state WM (query and/or push).

    ``push_mode="none"`` is permutation ``wm_walle`` (query + gate).
    ``push_mode="belief"`` is permutation ``wm_push_walle`` (always-on push + gate,
    no query). The gate governs the 9 physical verbs. `query` and `think:` are not
    WALL-E verbs (convert_action -> None), so they pass through to WMAlfWorldEnv.step
    untouched. The belief only folds real env observations (accepted actions) —
    rejected actions never happened.
    """

    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self._gate_init(**kwargs)

    def step(self, llm_output: str) -> Tuple[str, State]:
        verdict, _ = self._run_walle_gate(llm_output)
        if verdict == "reject":
            return self._emit_rejection(llm_output)
        # accept / passthrough (incl. `query`): hand off to the WM env step.
        return WMAlfWorldEnv.step(self, llm_output)
