"""The ScienceWorld environment object.

Structured like `AlfWorldEnv` (parse_action / conduct_action / step /
record_observation / reset) so a future belief-WM variant can override
`record_observation` exactly like `WMAlfWorldEnv` does — the WM seam is identical
across both environments.

The heavy ScienceWorld JVM server is constructed ONCE and shared across tasks
(passed in via env_config['env']); each task only calls env.load(...) in reset().
The score-scaling monkey-patch (utils.sciworld_score) must be applied before the
shared env is built.
"""

import re
import logging
from typing import Tuple

from scienceworld import ScienceWorldEnv

from envs import BaseEnv
from tasks import SciWorldTask
from prompt import prompt_with_icl
from utils.datatypes import State


logger = logging.getLogger("agent_eval")


# Substring ScienceWorld returns when a parseable action matches no valid command
# (the analogue of ALFWorld's "Nothing happens" no-op marker).
_NOOP_MARKER = "No known action matches that input"

# Success is defined as reaching this fraction of the partial score (0-100 scale).
# We deliberately do NOT trust ScienceWorld's own `done`/completion flag — prior
# work (WALL-E) found the env's completion signal unreliable — so success is derived
# purely from the (monotone, max-seen) score crossing this threshold.
_SUCCESS_SCORE_THRESHOLD = 70


class SciWorldEnv(BaseEnv):
    def __init__(
        self,
        task: SciWorldTask,
        env: ScienceWorldEnv = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: SciWorldTask = task
        # Shared JVM-backed env built once in eval_baselines.py.
        self.env: ScienceWorldEnv = env
        # ScienceWorld simplification flags passed to env.load(). NOTE: the shorthand
        # "easy" may NOT be combined with other tokens (the env then rejects the whole
        # string). Our default is the explicit expansion of "easy" PLUS openContainers,
        # so all containers are pre-opened. Electrical tasks drop "noElectricalAction"
        # (it is incompatible with electrical tasks -- the env crashes/rejects) so those
        # require real `connect` actions; handled per-task in _simplification_for().
        self.simplification = kwargs.get(
            "simplification",
            "noElectricalAction,openDoors,selfWateringFlowerPots,teleportAction,openContainers",
        )
        self.state = State()
        # Track consecutive malformed agent outputs (parse failures).
        self.bad_steps = 0
        self.max_bad_steps = 10
        # Track consecutive invalid-action no-ops; breaks out of stuck loops.
        self.noop_steps = 0
        self.max_noop_steps = 10
        # MPO: when set, inject task.workflow into the initial prompt (Instruction mode).
        self._mpo_workflow = kwargs.get("mpo_workflow", False)

    def parse_action(self, llm_output: str) -> str:
        # Extract the text after "Action:" from the ReAct/ReflAct output.
        llm_output = llm_output.strip()
        action = re.findall(r"Action:\s?(.*)", llm_output, re.DOTALL)[0]
        assert action is not None
        return action

    def conduct_action(self, action: str):
        # Step the shared env; monkey-patched step returns 0-100 score in info.
        observation, reward, done, info = self.env.step(action)
        return observation, reward, done, info

    def step(self, llm_output: str) -> Tuple[str, State]:
        # Record the agent's turn.
        self.state.history.append({"role": "assistant", "content": llm_output})

        # Try to parse and perform the action; handle malformed output.
        try:
            action = self.parse_action(llm_output)
        except Exception:
            observation = "Observation: Invalid format. Your input must contain 'Action: '"
            self.state.history.append({"role": "user", "content": observation})
            self.state.steps += 1
            self.bad_steps += 1
            self.noop_steps = 0
            # Not a real env step: do not append reward_trace. Queries and
            # WALL-E rejections already skip record_observation.
            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
            elif self.bad_steps >= self.max_bad_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "format_error_limit"
            return observation, self.state

        observation, reward, done, info = self.conduct_action(action)
        return self.record_observation(observation, reward, done, info)

    def record_observation(self, observation, reward, done, info):
        """Append an env observation to history and apply the termination rules.

        The WM extension point: a belief-WM variant overrides this to fold the
        observation into belief and push a state summary, mirroring WMAlfWorldEnv.
        `info['score']` is the monotone 0-100 score; we keep the max as state.reward.
        """
        observation = f"Observation: {observation}"
        self.state.history.append({"role": "user", "content": observation})

        # A parseable action executed; reset the parse-failure streak.
        self.bad_steps = 0
        # Track consecutive invalid-action no-ops; reset on any productive step.
        if _NOOP_MARKER in observation:
            self.noop_steps += 1
        else:
            self.noop_steps = 0

        # Keep the best (monotone) score seen as the episode reward -> AR.
        score = info.get("score", 0)
        if self.state.reward is None or score > self.state.reward:
            self.state.reward = score

        self.state.steps += 1
        # One running-max point per real env step (not queries / WALL-E rejects).
        self.state.append_reward_trace()

        # Termination is decoupled from the success METRIC. We do NOT force-stop the
        # moment the score crosses the threshold; the agent runs to a natural episode
        # end -- the env's own `done` flag (like MPO), the per-task step budget, or a
        # no-op loop. SUCCESS is then defined purely on the max score reaching the
        # threshold (>=70), independent of WHY the episode ended. This keeps our
        # score-based SR (WALL-E found the env `done` unreliable as a SUCCESS signal)
        # while still letting `done` act as a natural STOP signal (as MPO does).
        if done:
            self.state.finished = True
            self.state.terminate_reason = "env_done"
        elif self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.terminate_reason = "max_steps"
        elif self.noop_steps >= self.max_noop_steps:
            self.state.finished = True
            self.state.terminate_reason = "noop_loop"

        # Success metric: max score seen >= threshold, decided independently of the
        # termination reason. Set on every step so it reflects the running max; the
        # eval loop reads it at episode end.
        self.state.success = (
            self.state.reward is not None
            and self.state.reward >= _SUCCESS_SCORE_THRESHOLD
        )
        # A threshold-crossing success is still worth labelling as such for readouts,
        # without forcing an early stop.
        if self.state.finished and self.state.success and self.state.terminate_reason == "env_done":
            self.state.terminate_reason = "success"

        return observation, self.state

    def _simplification_for(self, sub_task_name: str) -> str:
        """Per-task simplification string. Electrical tasks (power-component /
        conductivity) are incompatible with `noElectricalAction` -- the env rejects or
        crashes -- so strip it for them; they then require real `connect` actions.
        All other tasks keep the configured flags verbatim."""
        flags = [f.strip() for f in self.simplification.split(",") if f.strip()]
        name = sub_task_name.lower()
        is_electrical = "power-component" in name or "conductivity" in name
        if is_electrical:
            flags = [f for f in flags if f != "noElectricalAction"]
        return ",".join(flags)

    def _select_icl(self):
        """Return the ICL conversation list to inject for this task.

        Format-agnostic so all ICL layouts work through the same path:
        - dict keyed by sub_task_name (per-task ICL): index by task.task_type first;
        - dict keyed by family (per-family ICL, rollback): fall back to task.icl_family;
        - list (legacy / WM ICL, e.g. sciworld_reflact_wm_icl.json): pass through
          unchanged. This keeps WMSciWorldEnv (which calls super().reset()) working
          on its still-flat ICL file with no change.
        prompt_with_icl expects a list of conversations, so we always return a list.
        """
        if isinstance(self.raw_icl, dict):
            # Per-task key (sub_task_name) wins; then family; then any exemplar.
            for key in (getattr(self.task, "task_type", None),
                        getattr(self.task, "icl_family", None)):
                if key in self.raw_icl:
                    return self.raw_icl[key]
            fallback = next(iter(self.raw_icl))
            logger.warning(
                "ICL keys %r/%r not in ICL dict; falling back to %r",
                getattr(self.task, "task_type", None),
                getattr(self.task, "icl_family", None), fallback,
            )
            return self.raw_icl[fallback]
        return self.raw_icl

    def reset(self) -> Tuple[str, State]:
        # Fresh state; load the specific task variation onto the shared env.
        self.state = State()
        # Prefer the per-task step budget (MPO max_steps.json) over the config
        # default, so each task gets its own horizon (10-120 steps).
        if getattr(self.task, "max_steps", None) is not None:
            self.max_steps = self.task.max_steps
        # Persist the per-task budget so AR-per-step can charge it to early break-outs.
        self.state.max_steps = self.max_steps
        self.env.load(
            self.task.sub_task_name,
            self.task.variation_idx,
            simplificationStr=self._simplification_for(self.task.sub_task_name),
            generateGoldPath=False,
        )
        obs, info = self.env.reset()
        cur_task = info["taskDesc"]

        # Build the initial prompt: instruction + ICL + task description.
        # _select_icl() picks this task's per-family exemplar from a dict ICL, or
        # passes a flat-list ICL through unchanged (WM path).
        logger.debug("SciWorld ICL family: %s", getattr(self.task, "icl_family", None))
        observation, messages = prompt_with_icl(
            instruction=self.instruction,
            raw_icl=self._select_icl(),
            cur_task=cur_task,
            icl_num=1,
            workflow=self.task.workflow if self._mpo_workflow else None,
        )
        if self.icl_format == "first":
            self.state.history.append({"role": "user", "content": observation})
        elif self.icl_format == "conversation":
            self.state.history = messages
        return observation, self.state
