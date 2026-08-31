"""WALL-E 2.0 ALFWorld baseline: ported rule-based action-validity world model.

See SOURCE.md for provenance. The env (envs/walle_alfworld_env.py) wires these into the
step loop: parse the agent's action, transform state+action, run check_action, and
reject infeasible actions "in imagination" without stepping the real env.
"""

from wm.walle.action_transform import convert_action
from wm.walle.state_transform import walle_state_transform
from wm.walle.rules import load_rules, check_action
from wm.walle.sciworld_validity import (
    build_valid_set,
    is_valid_action,
    normalize_action,
)

__all__ = [
    "convert_action",
    "walle_state_transform",
    "load_rules",
    "check_action",
    "build_valid_set",
    "is_valid_action",
    "normalize_action",
]
