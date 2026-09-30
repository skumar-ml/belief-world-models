"""M1 metrics for ALFWorld, ScienceWorld, and BabyAI-Text baseline evaluation.

Metrics:
  - SR@1            : single-trial success rate (primary, temp=0)
  - progression_percentage : 100 * SR@1 (BALROG BabyAI average progress)
  - steps_to_goal   : mean/median env steps, SOLVED TASKS ONLY
  - format_error_rate: fraction of steps whose action failed to parse
  - noop_error_rate : fraction of steps that were no-ops
  - avg_reward / avg_ar_per_step : ScienceWorld primary metrics
  - success_within_budget : P(success and solve_time <= t); see helpers below
"""

import json
import math
import statistics
from typing import Any, Dict, List

from utils.datatypes import State

_MARKERS = {
    "AlfWorldEnv": {"format_error": "Error Input", "noop": "Nothing happens"},
    "SciWorldEnv": {"format_error": "Invalid format", "noop": "No known action matches that input"},
    "BabyAIEnv": {"format_error": "Error Input", "noop": "Nothing happens"},
}
_DEFAULT_MARKERS = _MARKERS["AlfWorldEnv"]


def _base_env_family(env_class: str) -> str:
    if "SciWorld" in env_class:
        return "SciWorldEnv"
    if "BabyAI" in env_class:
        return "BabyAIEnv"
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
        "progression_percentage": 100.0 * n_success / n,
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


def _episode_field(episode: Any, name: str, default: Any = 0) -> Any:
    """Read `name` from a State, a dict, or any duck-typed episode record."""
    if isinstance(episode, dict):
        return episode.get(name, default)
    return getattr(episode, name, default)


def solve_time(episode: Any, cost: str = "env") -> Any:
    """Env-step (or all-action) index at which a successful episode finished.

    Unsuccessful episodes return None (they never count at any budget).
    `cost="env"` is the eval horizon: WALL-E rejections and WM queries are
    not env steps. `cost="all"` adds those extra LLM turns.
    """
    if not bool(_episode_field(episode, "success", False)):
        return None
    steps = int(_episode_field(episode, "steps", 0) or 0)
    if cost == "env":
        return steps
    if cost == "all":
        queries = int(_episode_field(episode, "query_steps", 0) or 0)
        rejections = int(_episode_field(episode, "walle_rejections", 0) or 0)
        return steps + queries + rejections
    raise ValueError(f"unknown cost={cost!r}; expected 'env' or 'all'")


def success_within_budget_curve(
    episodes: List[Any],
    budgets: List[int],
    cost: str = "env",
) -> List[float]:
    """P(success and solve_time <= t) for each budget t.

    Denominator is every episode, including failures. This is the CDF of
    solve time and avoids the selection bias of solved-only steps-to-goal.
    """
    n = len(episodes)
    if n == 0:
        return [0.0 for _ in budgets]
    times = [solve_time(ep, cost=cost) for ep in episodes]
    curve = []
    for t in budgets:
        n_ok = sum(1 for tau in times if tau is not None and tau <= t)
        curve.append(n_ok / n)
    return curve


def success_within_normalized_budget_curve(
    episodes: List[Any],
    fractions: List[float],
    cost: str = "env",
    default_max_steps: int = None,
) -> List[float]:
    """P(success and solve_time <= alpha * episode_budget) for alpha in fractions.

    Use this when episodes have heterogeneous horizons (ScienceWorld 10-120).
    Episodes with no `max_steps` fall back to `default_max_steps`.
    """
    n = len(episodes)
    if n == 0:
        return [0.0 for _ in fractions]
    curve = []
    for alpha in fractions:
        n_ok = 0
        for ep in episodes:
            tau = solve_time(ep, cost=cost)
            if tau is None:
                continue
            horizon = _episode_field(ep, "max_steps", None) or default_max_steps
            if horizon is None or horizon <= 0:
                continue
            if tau <= alpha * float(horizon):
                n_ok += 1
        curve.append(n_ok / n)
    return curve


def _episode_horizon(episode: Any, default_max_steps: int = None) -> Any:
    horizon = _episode_field(episode, "max_steps", None) or default_max_steps
    if horizon is None or horizon <= 0:
        return None
    return int(horizon)


def anytime_return_at_fraction(
    episode: Any,
    alpha: float,
    default_max_steps: int = None,
    score_scale: float = 100.0,
) -> float:
    """Running-max return after floor(alpha * T) env-budget steps, in [0, 1].

    Prefers `reward_trace` (ScienceWorld running-max score / score_scale).
    Dumps without a trace fall back to binary success-within-budget.
    After the episode ends, the last recorded value is held to alpha=1.
    """
    horizon = _episode_horizon(episode, default_max_steps)
    if horizon is None:
        return 0.0
    # epsilon so 0.3 * 10 is 3, not int(2.999...) == 2
    n_allowed = int(math.floor(alpha * horizon + 1e-9))
    trace = _episode_field(episode, "reward_trace", None) or []
    if trace:
        if n_allowed <= 0:
            return 0.0
        idx = min(n_allowed, len(trace)) - 1
        return float(trace[idx]) / float(score_scale)
    tau = solve_time(episode, cost="env")
    if tau is None:
        return 0.0
    return 1.0 if tau <= n_allowed else 0.0


def reward_within_normalized_budget_curve(
    episodes: List[Any],
    fractions: List[float],
    default_max_steps: int = None,
    score_scale: float = 100.0,
) -> List[float]:
    """Mean anytime return at each budget fraction alpha.

    ScienceWorld: running-max score in [0, 1]. ALFWorld / trace-less dumps:
    binary success-within-budget (0/1). Forward-fills after termination.
    """
    n = len(episodes)
    if n == 0:
        return [0.0 for _ in fractions]
    return [
        sum(
            anytime_return_at_fraction(ep, alpha, default_max_steps, score_scale)
            for ep in episodes
        )
        / n
        for alpha in fractions
    ]


def write_metrics(metrics: Dict[str, Any], path: str) -> None:
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
