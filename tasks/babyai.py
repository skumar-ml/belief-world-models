"""BabyAI-Text task instances (BALROG MixedTrainLocal protocol).

BALROG evaluates five families on BabyAI-MixedTrainLocal-v0 (num_dists=0),
25 seeds each in the paper (10 in their default yaml). We freeze those
(family, seed) pairs under data/babyai/ so resume/debug is deterministic —
BALROG itself samples time-based seeds and is not reproducible.

No env is built here; BabyAIEnv constructs / caches one gymnasium env per
family (resample gym.make until action_kinds matches, same as BALROG).
Prompting is zero-shot (no ICL).
"""

import json
import logging
import os
from typing import Iterable, Tuple

from tasks.base import Task


logger = logging.getLogger("agent_eval")

_DEFAULT_DATA_DIR = "data/babyai"

# BALROG tasks.babyai_tasks, with the /goal suffix as our family key.
# action_kinds[0].replace(" ", "_") must equal these strings.
FAMILIES = (
    "goto",
    "pickup",
    "open",
    "putnext",
    "pick_up_seq_go_to",
)

ENV_ID = "BabyAI-MixedTrainLocal-v0"


class BabyAITask(Task):
    """A single (family, seed) episode on MixedTrainLocal."""

    task_name = "babyai"

    def __init__(
        self,
        family: str,
        seed: int,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.family = family
        self.seed = int(seed)
        # Family key for metrics / breakdowns (BabyAI is zero-shot; no ICL lookup).
        self.task_type = family

    @classmethod
    def load_tasks(
        cls,
        path: str = "",
        workflow_path: str = None,
        split: str = "test",
        part_num: int = 1,
        part_idx: int = -1,
        families=None,
    ) -> Tuple[Iterable[Task], int]:
        """Load frozen (family, seed) pairs for the given split.

        workflow_path is accepted for signature parity with the other loaders
        and is unused in v1 (no MPO metaplans for BabyAI).
        families, if set, keeps only those BALROG families (e.g. goto + pickup).
        """
        data_dir = path or _DEFAULT_DATA_DIR
        index_file = os.path.join(data_dir, f"{split}_indices.json")
        if not os.path.exists(index_file):
            raise FileNotFoundError(
                f"BabyAI index file not found: {index_file}. "
                f"Expected data/babyai/{{test,dev}}_indices.json."
            )
        pairs = json.load(open(index_file))  # list of [family, seed]

        unmapped = sorted({fam for fam, _ in pairs if fam not in FAMILIES})
        assert not unmapped, f"families missing from FAMILIES: {unmapped}"

        if families:
            keep = set(families)
            unknown = sorted(keep - set(FAMILIES))
            assert not unknown, f"unknown BabyAI families: {unknown}"
            pairs = [p for p in pairs if p[0] in keep]
            logger.info(f"BabyAI family filter {sorted(keep)} -> {len(pairs)} episodes")

        if part_num > 1:
            assert part_idx != -1
            part_len = len(pairs) // part_num + 1
            pairs = pairs[part_len * part_idx : part_len * (part_idx + 1)]
        n_tasks = len(pairs)

        def generator():
            for family, seed in pairs:
                task_id = f"{family}_{seed}"
                yield cls(task_id=task_id, family=family, seed=seed)

        return generator(), n_tasks
