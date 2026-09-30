import json
from abc import ABC, abstractmethod
from typing import Tuple
from prompt.instruction import load_instruction
from utils.datatypes import State


class BaseEnv(ABC):
    def __init__(
        self,
        instruction_path: str,
        icl_path: str = None,
        icl_format: str = "first",
        max_steps: int = 10,
        **kwargs,
    ):
        # Expand {snippet} placeholders (BabyAI); other benchmarks pass through.
        self.instruction = load_instruction(instruction_path)
        # BabyAI is zero-shot; ALFWorld / SciWorld still pass an ICL file.
        self.raw_icl = json.load(open(icl_path)) if icl_path else None
        self.icl_format = icl_format
        self.max_steps = max_steps


    @abstractmethod
    def step(self, llm_output: str) -> Tuple[str, State]:
        pass

    @abstractmethod
    def reset(self) -> Tuple[str, State]:
        pass
