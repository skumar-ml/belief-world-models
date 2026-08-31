"""Phase 0a WALL-E baseline for ScienceWorld: a parser-validity oracle gate.

Mirrors the ALFWorld WALL-E gate (envs/walle_alfworld_env.py): reject actions the
ScienceWorld parser cannot execute "in imagination" (no env step consumed) and re-prompt
the agent. This is the parser-validity ceiling the learned WALL-E rules will later
approximate.

VALIDITY ORACLE = the parser itself, NOT getValidActionObjectCombinations().
An earlier version of this gate used pre-step membership in getValidActionObject-
Combinations(); that list UNDER-approximates the parser (it omits parseable forms like
`examine X`, `deactivate X`-when-already-off, `read X`), so the gate falsely rejected
valid actions and collapsed SR 2.5-4x. We verified on the live engine that:
  * a parser-invalid action returns the exact substring "No known action matches that
    input." AND does NOT advance the move counter (getNumMoves is unchanged) -- it is a
    genuine no-op on the engine, nothing to roll back;
  * ANY parsed action (even a semantically-refused one like `eat stove` -> "not edible",
    or `focus on <wrong object>`) advances the counter and is therefore parser-VALID.
So we step the real env via conduct_action() -- which does NOT mutate self.state -- inspect
the raw observation, and only commit it (record_observation, which bumps state.steps) when
the parser accepted it. A parse-fail is converted into a [World model] re-prompt instead.
This has ZERO false rejects by construction: the parser is ground truth.

Scope: parser-validity ONLY. A parser-valid but task-terminating action (e.g.
`focus on <wrong object>`) parses (moves +1) and passes through untouched -- that is
Phase 0b's concern, not this gate's.

See docs/walle_sciworld_scope.md (Part B.1, Phase 0a). Accounting matches the ALFWorld
gate: imagination rejections do NOT count as env steps; they bump state.walle_rejections.
A cap (walle_max_retries) lets the action through after repeated rejections at one
decision point so the episode can't stall in imagination.
"""

import logging
from typing import Tuple

from envs.sciworld_env import SciWorldEnv, _NOOP_MARKER
from utils.datatypes import State


logger = logging.getLogger("agent_eval")


class WalleOracleSciWorldEnv(SciWorldEnv):
    """SciWorld + parser-validity oracle gate (Phase 0a), using the parser as oracle."""

    def __init__(self, task, **kwargs):
        super().__init__(task, **kwargs)
        self.walle_max_retries = kwargs.get("walle_max_retries", 5)
        # Consecutive rejections at the current decision point (reset on accept/step).
        self._consecutive_rejections = 0

    def _emit_rejection(self, llm_output: str, action_str: str) -> Tuple[str, State]:
        """Record the rejected turn + an imagination observation; no env step committed.

        The real env was already stepped (conduct_action) but on a parser-invalid action
        that is a no-op (move counter unchanged, state unchanged), so nothing to undo. We
        simply do NOT call record_observation, so state.steps is not incremented.
        """
        self.state.history.append({"role": "assistant", "content": llm_output})
        answer = (
            f"[World model] The action '{action_str}' is not possible in the current "
            f"state (the parser does not recognize it here). Choose a different action."
        )
        self.state.history.append({"role": "user", "content": f"Observation: {answer}"})
        self.state.walle_rejections += 1
        self._consecutive_rejections += 1
        return f"Observation: {answer}", self.state

    def step(self, llm_output: str) -> Tuple[str, State]:
        # Malformed output (no "Action:"): defer to the parent's format-error path.
        try:
            action_str = self.parse_action(llm_output)
        except Exception:
            self._consecutive_rejections = 0
            return super().step(llm_output)

        # Retry cap: after too many rejections at this decision point, commit the action
        # anyway so the episode can't wedge in imagination. super().step re-parses and
        # records normally (the action is a no-op on the engine, so it costs one step).
        if self._consecutive_rejections >= self.walle_max_retries:
            self._consecutive_rejections = 0
            return super().step(llm_output)

        # Step the real env WITHOUT mutating self.state (conduct_action only calls
        # env.step and returns the tuple). The parser is the validity oracle.
        observation, reward, done, info = self.conduct_action(action_str)

        # Parser-invalid -> "No known action matches that input." and a no-op on the
        # engine. Convert to a [World model] re-prompt; do not commit (no state.steps bump).
        if _NOOP_MARKER in observation:
            return self._emit_rejection(llm_output, action_str)

        # Parser accepted (even if semantically refused): commit normally. Record the
        # agent turn, then let record_observation append the obs, bump steps, apply
        # termination/scoring -- exactly as the base env would.
        self.state.history.append({"role": "assistant", "content": llm_output})
        self._consecutive_rejections = 0
        return self.record_observation(observation, reward, done, info)
