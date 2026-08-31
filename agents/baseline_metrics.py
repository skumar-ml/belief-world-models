"""M1 metrics for ALFWorld and ScienceWorld baseline evaluation.

Metrics:
  - SR@1            : single-trial success rate (primary, temp=0)
  - steps_to_goal   : mean/median env steps, SOLVED TASKS ONLY
  - format_error_rate: fraction of steps whose action failed to parse
  - noop_error_rate : fraction of steps that were no-ops
  - avg_reward / avg_ar_per_step : ScienceWorld primary metrics
"""

import json
import statistics
from typing import Any, Dict, List

from utils.datatypes import State

_MARKERS = {
    "AlfWorldEnv": {"format_error": "Error Input", "noop": "Nothing happens"},
    "SciWorldEnv": {"format_error": "Invalid format", "noop": "No known action matches that input"},
}
_DEFAULT_MARKERS = _MARKERS["AlfWorldEnv"]


def _base_env_family(env_class: str) -> str:
    if "SciWorld" in env_class:
        return "SciWorldEnv"
    return "AlfWorldEnv"


def _count_format_errors(state: State, marker: str) -> int:
    n = 0
    for msg in state.history:
        if isinstance(msg, dict) and msg.get("role") == "user":
            if marker in str(msg.get("content", "")):
                n += 1
    return n


def _count_noop_steps(state: State, marker: str) -> int:
    n = 0
    for msg in state.history:
        if isinstance(msg, dict) and msg.get("role") == "user":
            if marker in str(msg.get("content", "")):
                n += 1
    return n


def compute_metrics(states: List[State], env_class: str = "AlfWorldEnv") -> Dict[str, Any]:
    markers = _MARKERS.get(_base_env_family(env_class), _DEFAULT_MARKERS)
    n = len(states)
    if n == 0:
        return {"sr_at_1": None, "n_success": 0, "n_tasks": 0}

    successes = [bool(s.success) for s in states]
    n_success = sum(successes)

    solved_steps = [s.steps for s in states if s.success]
    steps_mean = statistics.mean(solved_steps) if solved_steps else None
    steps_std = statistics.stdev(solved_steps) if len(solved_steps) >= 2 else (0.0 if solved_steps else None)

    total_steps = sum(s.steps for s in states)
    total_fmt_errors = sum(_count_format_errors(s, markers["format_error"]) for s in states)
    fmt_err_rate = (total_fmt_errors / total_steps) if total_steps else 0.0

    total_noops = sum(_count_noop_steps(s, markers["noop"]) for s in states)
    noop_err_rate = (total_noops / total_steps) if total_steps else 0.0

    reasons: Dict[str, int] = {}
    for s in states:
        r = s.terminate_reason or "unknown"
        reasons[r] = reasons.get(r, 0) + 1

    total_queries = sum(getattr(s, "query_steps", 0) for s in states)
    solved_queries = [getattr(s, "query_steps", 0) for s in states if s.success]
    queries_solved_mean = statistics.mean(solved_queries) if solved_queries else None

    total_walle_rejections = sum(getattr(s, "walle_rejections", 0) for s in states)
    solved_rejections = [getattr(s, "walle_rejections", 0) for s in states if s.success]
    rejections_solved_mean = statistics.mean(solved_rejections) if solved_rejections else None

    rewards = [float(s.reward) if s.reward is not None else 0.0 for s in states]
    avg_reward = statistics.mean(rewards) if rewards else 0.0
    std_reward = statistics.stdev(rewards) if len(rewards) >= 2 else 0.0

    _EARLY_BREAKOUT = {"max_steps", "noop_loop"}
    ar_per_step_vals = []
    for s in states:
        reward = float(s.reward) if s.reward is not None else 0.0
        denom = s.steps
        if s.terminate_reason in _EARLY_BREAKOUT and getattr(s, "max_steps", None):
            denom = s.max_steps
        if denom and denom > 0:
            ar_per_step_vals.append(reward / denom)
    avg_ar_per_step = statistics.mean(ar_per_step_vals) if ar_per_step_vals else None
    std_ar_per_step = statistics.stdev(ar_per_step_vals) if len(ar_per_step_vals) >= 2 else (0.0 if ar_per_step_vals else None)

    return {
        "sr_at_1": n_success / n,
        "avg_reward": avg_reward,
        "std_reward": std_reward,
        "avg_ar_per_step": avg_ar_per_step,
        "std_ar_per_step": std_ar_per_step,
        "n_ar_per_step_tasks": len(ar_per_step_vals),
        "n_success": n_success,
        "n_tasks": n,
        "steps_to_goal_solved_mean": steps_mean,
        "steps_to_goal_solved_std": steps_std,
        "n_solved_for_steps": len(solved_steps),
        "format_error_rate": fmt_err_rate,
        "total_format_errors": total_fmt_errors,
        "noop_error_rate": noop_err_rate,
        "total_noops": total_noops,
        "total_steps": total_steps,
        "terminate_reasons": reasons,
        "total_queries": total_queries,
        "queries_solved_mean": queries_solved_mean,
        "total_walle_rejections": total_walle_rejections,
        "rejections_solved_mean": rejections_solved_mean,
    }


def write_metrics(metrics: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
