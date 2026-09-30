"""BabyAI-Text environment wrapper (BALROG MixedTrainLocal, ReAct/ReflAct).

Structured like AlfWorldEnv / SciWorldEnv (parse_action / conduct_action /
step / record_observation / reset) so a later belief-WM variant can override
record_observation. Underlying simulator is BartekCupial/Minigrid's
BabyAI-MixedTrainLocal-v0 with BALROG's text-description formatting.

Invalid or unparseable actions do not step the env (unlike BALROG, which
falls back to 'go forward'). Horizon defaults to 64 (config / CLI); if both
are unset, MiniGrid's own max_steps is used (64 or 128). Prompting is
zero-shot (no ICL).
"""

from typing import Dict, Tuple

from envs import BaseEnv
from envs.babyai_parse import (
    BABYAI_ACTION_SPACE,
    FORMAT_ERROR_OBS,
    INVALID_ACTION_OBS,
    action_index,
    format_text_obs,
    parse_action as parse_babyai_action,
)
from tasks.babyai import ENV_ID, FAMILIES
from utils.datatypes import State


# One gymnasium env per family, shared across episodes (SciWorld-style).
_ENV_CACHE: Dict[str, object] = {}


def _import_minigrid():
    """Import gymnasium + the BALROG Minigrid fork, or fail with a clear hint."""
    try:
        import gymnasium as gym
        import minigrid
    except ImportError as e:
        raise ImportError(
            "BabyAI-Text requires gymnasium and the BartekCupial Minigrid fork "
            "(pip install 'minigrid @ git+https://github.com/BartekCupial/Minigrid.git')."
        ) from e
    # `import minigrid` already registers; only call this if MixedTrainLocal is missing.
    if "BabyAI-MixedTrainLocal-v0" not in gym.envs.registry:
        minigrid.register_minigrid_envs()
    return gym, minigrid


def make_family_env(family: str, num_dists: int = 0):
    """Construct MixedTrainLocal pinned to one BALROG family.

    BALROG resamples gym.make until action_kinds[0] matches the /goal suffix.
    After that, reset(seed=) stays in-family because action_kinds is a 1-list.
    """
    if family not in FAMILIES:
        raise ValueError(f"Unknown BabyAI family {family!r}; expected one of {FAMILIES}")
    gym, _minigrid = _import_minigrid()
    while True:
        env = gym.make(ENV_ID, num_dists=num_dists)
        kind = env.unwrapped.action_kinds[0].replace(" ", "_")
        if kind == family:
            return env


def get_family_env(family: str, num_dists: int = 0):
    """Return a cached env for this family, creating it on first use."""
    if family not in _ENV_CACHE:
        _ENV_CACHE[family] = make_family_env(family, num_dists=num_dists)
    return _ENV_CACHE[family]


class BabyAIEnv(BaseEnv):
    def __init__(self, task, **kwargs):
        super().__init__(**kwargs)
        self.task = task
        self.num_dists = int(kwargs.get("num_dists", 0))
        # Integer max_steps in env_config / --max_steps overrides the env horizon.
        # JSON null (or omitted + BaseEnv default) is replaced after reset().
        self._max_steps_override = kwargs.get("max_steps")
        self.env = get_family_env(task.family, num_dists=self.num_dists)
        self.state = State()
        self.bad_steps = 0
        self.max_bad_steps = 10
        self.noop_steps = 0
        self.max_noop_steps = 10
        self._last_obs_text = None
        self._mission = None

    def parse_action(self, llm_output: str) -> str:
        return parse_babyai_action(llm_output)

    def conduct_action(self, action: str):
        # Step the gymnasium env with the discrete MiniGrid index.
        action_int = action_index(action)
        obs, reward, terminated, truncated, info = self.env.step(action_int)
        text = format_text_obs(info)
        done = bool(terminated or truncated)
        return text, float(reward), done, info, bool(terminated), bool(truncated)

    def step(self, llm_output: str) -> Tuple[str, State]:
        # Record the agent's turn.
        self.state.history.append({"role": "assistant", "content": llm_output})

        try:
            action = self.parse_action(llm_output)
        except Exception:
            return self._reject(FORMAT_ERROR_OBS)

        if action not in BABYAI_ACTION_SPACE:
            return self._reject(INVALID_ACTION_OBS)

        observation, reward, done, info, terminated, truncated = self.conduct_action(action)
        return self.record_observation(
            observation, reward, done, info, terminated=terminated, truncated=truncated
        )

    def _reject(self, observation: str) -> Tuple[str, State]:
        """Format / invalid-action path: do not step the env."""
        self.state.history.append({"role": "user", "content": observation})
        self.state.steps += 1
        self.bad_steps += 1
        self.noop_steps = 0
        if self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
        elif self.bad_steps >= self.max_bad_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "format_error_limit"
        return observation, self.state

    def record_observation(
        self,
        observation,
        reward,
        done,
        info=None,
        terminated=False,
        truncated=False,
    ):
        """Append an env observation and apply termination / success rules."""
        observation = f"Observation: {observation}"
        self.state.history.append({"role": "user", "content": observation})

        self.bad_steps = 0
        # Unchanged view (e.g. walk into a wall) counts toward the no-op breakout.
        obs_body = observation
        if self._last_obs_text is not None and obs_body == self._last_obs_text:
            self.noop_steps += 1
        else:
            self.noop_steps = 0
        self._last_obs_text = obs_body

        if reward > 0:
            self.state.reward = reward
        elif self.state.reward is None:
            self.state.reward = reward

        self.state.steps += 1

        if reward > 0 or terminated:
            self.state.finished = True
            self.state.success = reward > 0
            self.state.terminate_reason = "success" if self.state.success else "env_done"
        elif truncated or self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
        elif self.noop_steps >= self.max_noop_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "noop_loop"

        return observation, self.state

    def reset(self) -> Tuple[str, State]:
        # Fresh state; reset the shared family env with the frozen seed.
        self.state = State()
        self.bad_steps = 0
        self.noop_steps = 0
        obs, info = self.env.reset(seed=self.task.seed)
        self._mission = obs.get("mission") or self.env.unwrapped.mission
        text = format_text_obs(info)
        # Default eval horizon is 64 (task config / --max_steps). MiniGrid's
        # own budget is 64 (goto/pickup) or 128 (open / putnext / seq).
        env_horizon = int(self.env.unwrapped.max_steps)
        if isinstance(self._max_steps_override, int):
            self.max_steps = self._max_steps_override
        else:
            self.max_steps = env_horizon
        self.state.max_steps = self.max_steps
        self._last_obs_text = None

        # Zero-shot: instruction + mission + first observation, no ICL exemplars.
        cur_task = f"Your task is to: {self._mission}\n{text}"
        observation = (
            f"{self.instruction}\n\n"
            f"Now, it's your turn and here is the task.\n{cur_task}"
        )
        self.state.history.append({"role": "user", "content": observation})
        return observation, self.state
