"""The alfworld environment object"""

import logging
from typing import Tuple

from envs import BaseEnv
from envs.alfworld_parse import NOOP_MARKER, parse_action as parse_alfworld_action, process_ob
from tasks import AlfWorldTask
from prompt import prompt_with_icl
from utils.datatypes import State


logger = logging.getLogger("agent_eval")


class AlfWorldEnv(BaseEnv):
    def __init__(
        self,
        task: AlfWorldTask,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.task: AlfWorldTask = task
        self.env = task.env
        self.state = State()
        # Track consecutive malformed agent outputs; too many in a row ends the episode early.
        self.bad_steps = 0
        self.max_bad_steps = 10
        # Track consecutive "Nothing happens" no-ops; breaks out of stuck loops.
        self.noop_steps = 0
        self.max_noop_steps = 10
        # MPO: when set, inject task.workflow into the initial prompt (Instruction mode).
        self._mpo_workflow = kwargs.get("mpo_workflow", False)

    def parse_action(self, llm_output: str) -> str:
        return parse_alfworld_action(llm_output)
    
    def conduct_action(self, action: str):
        # Step the underlying (batched) env and unwrap the single result.
        observation, reward, done, info = self.env.step([action])
        observation, reward, done = process_ob(observation[0]), info['won'][0], done[0]
        return observation, reward, done
    
    def step(self, llm_output: str) -> Tuple[str, State]:
        # Record the agent's turn.
        self.state.history.append({
            "role": "assistant",
            "content": llm_output
        })
        
        # Try to parse agent action and perform it. Handle bad actions
        try:
            action = self.parse_action(llm_output)
            observation, reward, done = self.conduct_action(action)
        except Exception as e:
            # Unparseable output: count a bad step and feed back an error obs.
            self.state.success = False
            self.state.finished = False
            self.state.reward=0
            observation = f"Observation: Error Input. Your input must contains 'Action: '"
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
            self.state.steps += 1
            self.bad_steps += 1
            # This turn was not a successful action, so it breaks any no-op streak.
            self.noop_steps = 0
            # End on budget exhaustion, or on too many consecutive parse failures.
            if self.state.steps >= self.max_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "max_steps"
                self.state.reward = 0
            elif self.bad_steps >= self.max_bad_steps:
                self.state.finished = True
                self.state.success = False
                self.state.terminate_reason = "format_error_limit"
                self.state.reward = 0
            return observation, self.state

        # Record the env response, update streaks, and apply termination rules.
        return self.record_observation(observation, reward, done)

    def record_observation(self, observation, reward, done):
        """Append an env observation to history and apply the termination rules.

        Shared by step() and by subclasses (e.g. the FoV variant) that produce
        observations without calling the underlying ALFWorld env. `observation`
        is the raw env text (without the "Observation: " prefix); `reward`/`done`
        come from the env (a synthetic look/no-op passes reward=0, done=False).
        """
        # Record the environment's response.
        observation = f"Observation: {observation}"

        self.state.history.append({
            "role": "user",
            "content": observation,
        })

        # The action parsed and executed, so reset the consecutive parse-failure streak.
        self.bad_steps = 0
        # Track consecutive no-ops ("Nothing happens"); reset on any productive step.
        if NOOP_MARKER in observation:
            self.noop_steps += 1
        else:
            self.noop_steps = 0

        # Terminate as failure on budget exhaustion, or when stuck in a no-op loop.
        self.state.steps += 1
        if self.state.steps >= self.max_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "max_steps"
            self.state.reward = reward
        elif self.noop_steps >= self.max_noop_steps:
            self.state.finished = True
            self.state.success = False
            self.state.terminate_reason = "noop_loop"
            self.state.reward = reward

        # Terminate as success if the env reports the goal is met.
        if done:
            self.state.finished = True
            self.state.success = True
            self.state.terminate_reason = "success"
            self.state.reward = reward

        return observation, self.state

    def reset(self) -> Tuple[str, State]:
        # Fresh state, then build the initial prompt: instruction + ICL + task.
        self.state = State()
        cur_task = self.task.observation
        observation, messages = prompt_with_icl(
            instruction=self.instruction,
            raw_icl=self.raw_icl[self.task.task_type],
            cur_task=cur_task,
            icl_num=1,
            workflow=self.task.workflow if self._mpo_workflow else None,
        )
        # Seed history either as one user turn or as a full conversation.
        if self.icl_format == 'first':
            self.state.history.append({
                "role": "user",
                "content": observation,
            })
        elif self.icl_format == 'conversation':
            self.state.history = messages
        return observation, self.state
