"""ScienceWorld task instances.

Adapted from MPO (github.com/WeiminXiong/MPO tasks/sciworld.py). We use MPO's
precomputed split index files (data/sciworld/{test,dev,train}_indices.json) so our
test set is IDENTICAL to the one used by MPO and ReflAct: 211 episodes across 24
scientific task types. Each index entry is a [task_name, variation_idx] pair, where
task_name is the "task-N-<name>" form that ScienceWorld's env.load accepts directly.

Per-task step budgets come from data/sciworld/max_steps.json (MPO convention;
10-120 steps depending on task), carried on each task and applied in SciWorldEnv.
"""

import json
import logging
import os
from typing import Iterable, Tuple

from tasks.base import Task

logger = logging.getLogger("agent_eval")

# Directory holding the MPO index + max_steps files (data/sciworld/).
_DEFAULT_DATA_DIR = "data/sciworld"


# Exact-match mapping from each ScienceWorld test sub_task_name to an ICL "family"
# key. Families group task types that share the same game mechanic, so one ICL
# exemplar per family teaches the rules for all its variations (mirrors AlfWorld,
# which keys ICL by task family). NOTE: prefix matching on the numeric part is a
# trap ("task-1" prefixes "task-10"; "task-2" prefixes "task-2a"), so we map the
# full name explicitly.
FAMILY_MAP = {
    # thermodynamics: change a substance's state of matter (heat/cool + focus).
    "task-1-boil": "thermo",
    "task-1-freeze": "thermo",
    "task-1-melt": "thermo",
    "task-1-change-the-state-of-matter-of": "thermo",
    # build an electrical circuit that powers a component.
    "task-2-power-component": "electrical",
    "task-2-power-component-(renewable-vs-nonrenewable-energy)": "electrical",
    # wire an unknown substance into a circuit to test if it conducts.
    "task-2a-test-conductivity": "conductivity",
    "task-2a-test-conductivity-of-unknown-substances": "conductivity",
    # navigate + classify + focus on a living/non-living/plant/animal thing.
    "task-3-find-animal": "find-thing",
    "task-3-find-living-thing": "find-thing",
    "task-3-find-non-living-thing": "find-thing",
    "task-3-find-plant": "find-thing",
    # long-horizon: plant a seed, water/wait until it grows, focus.
    "task-4-grow-fruit": "grow-plant",
    "task-4-grow-plant": "grow-plant",
    # chemistry: pour + mix reagents to create a target substance, focus.
    "task-5-chemistry-mix": "chemistry-mix",
    "task-5-chemistry-mix-paint-(secondary-color)": "chemistry-mix",
    "task-5-chemistry-mix-paint-(tertiary-color)": "chemistry-mix",
    # read/compare animal lifespans, focus on the extreme.
    "task-6-lifespan-(longest-lived)": "lifespan",
    "task-6-lifespan-(shortest-lived)": "lifespan",
    "task-6-lifespan-(longest-lived-then-shortest-lived)": "lifespan",
    # observe an organism across life stages, focus on a named stage.
    "task-7-identify-life-stages-1": "life-stages",
    "task-7-identify-life-stages-2": "life-stages",
    # measurement: use a thermometer on a substance and read its temperature
    # (distinct mechanic from thermo -- measure, not change-state).
    "task-10-use-thermometer": "measure",
    "task-10-measure-melting-point-(known-substance)": "measure",
}


def family_key(sub_task_name: str) -> str:
    """Map a ScienceWorld sub_task_name to its ICL family key (see FAMILY_MAP)."""
    return FAMILY_MAP[sub_task_name]


class SciWorldTask(Task):
    """A single ScienceWorld (task_type, variation) instance."""

    task_name = "sciworld"

    def __init__(
        self,
        sub_task_name: str,
        variation_idx: int,
        max_steps: int = None,
        workflow: str = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.sub_task_name = sub_task_name    # e.g. "task-1-boil"
        self.variation_idx = variation_idx    # held-out test variation index
        # Per-task step budget from max_steps.json (MPO); SciWorldEnv uses it.
        self.max_steps = max_steps
        # MPO meta-plan for this task (None unless a metaplan jsonl is loaded).
        self.workflow = workflow
        # task_type mirrors AlfWorldTask so downstream code (metrics breakdowns)
        # can group by task family.
        self.task_type = sub_task_name
        # ICL family key used by SciWorldEnv to pick the per-family exemplar.
        self.icl_family = family_key(sub_task_name)

    @classmethod
    def load_tasks(
        cls,
        path: str = "",
        workflow_path: str = None,
        split: str = "test",
        part_num: int = 1,
        part_idx: int = -1,
    ) -> Tuple[Iterable[Task], int]:
        """Load the MPO/ReflAct (task_name, variation) index for the given split.

        The test split is the canonical 211-episode / 24-type set. No env is built
        here; the actual episodes run on the shared env inside SciWorldEnv.

        workflow_path (MPO): jsonl of {"id": <task_id>, "workflow": <str>}. When given,
        each task is tagged with its meta-plan keyed by task_id ("<sub_task_name>_<var>").
        """
        data_dir = path or _DEFAULT_DATA_DIR
        index_file = os.path.join(data_dir, f"{split}_indices.json")
        if not os.path.exists(index_file):
            raise FileNotFoundError(
                f"SciWorld index file not found: {index_file}. "
                f"Extract data/sciworld/*_indices.json from the MPO data archive."
            )
        pairs = json.load(open(index_file))  # list of [task_name, variation_idx]

        # Fail loud if any loaded task type lacks an ICL family mapping (a typo or
        # a newly added task type would otherwise silently serve the wrong exemplar).
        unmapped = sorted({t for t, _ in pairs if t not in FAMILY_MAP})
        assert not unmapped, f"sub_task_names missing from FAMILY_MAP: {unmapped}"

        # Per-task step budgets (MPO convention). Fall back to 100 if missing.
        max_steps_path = os.path.join(data_dir, "max_steps.json")
        max_steps_map = json.load(open(max_steps_path)) if os.path.exists(max_steps_path) else {}

        # MPO: load per-task meta-plans keyed by task_id. None -> no workflows.
        task_id_2_workflow = {}
        if workflow_path is not None:
            with open(workflow_path) as fr:
                for line in fr:
                    item = json.loads(line)
                    task_id_2_workflow[item["id"]] = item["workflow"]

        # Optional sharding, identical semantics to AlfWorldTask.
        if part_num > 1:
            assert part_idx != -1
            part_len = len(pairs) // part_num + 1
            pairs = pairs[part_len * part_idx: part_len * (part_idx + 1)]
        n_tasks = len(pairs)

        def generator():
            for task_name, var in pairs:
                task_id = f"{task_name}_{var}"
                yield cls(
                    task_id=task_id,
                    sub_task_name=task_name,
                    variation_idx=var,
                    max_steps=max_steps_map.get(task_name, 100),
                    workflow=task_id_2_workflow.get(task_id, None),
                )

        return generator(), n_tasks
