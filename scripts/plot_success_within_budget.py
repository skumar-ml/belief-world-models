"""Success-within-budget curves from ALFWorld / ScienceWorld (and BabyAI) dumps.

Metric
------
    SR(t) = P(success AND solve_time <= t)
          = (# episodes that succeeded in at most t env steps) / N

Failures never contribute. This is the CDF of solve time: it is comparable
across methods even when they solve different subsets, unlike solved-only
mean steps-to-goal.

Solve time is `State.steps` (environment steps). WALL-E rejections and WM
queries do not consume the env budget; pass `--cost all` to charge them.

ScienceWorld
------------
Success is max score >= 70. New dumps store `reward_trace`: the running-max
0-100 score after each real env step (WM queries and WALL-E rejections omitted).
The normalized / combined figures plot mean anytime return (score/100, held
after termination) against budget fraction alpha. Older dumps without a trace
fall back to binary success-within-budget.

Usage
-----
    python scripts/plot_success_within_budget.py
    python scripts/plot_success_within_budget.py --models all
    python scripts/plot_success_within_budget.py --benchmarks alfworld --cost env
    python scripts/plot_success_within_budget.py --benchmarks babyai
    python scripts/plot_success_within_budget.py --benchmarks alfworld,sciworld,babyai
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import statistics
import sys
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

# Repo root on sys.path so `agents` / `utils` imports work from any cwd.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from agents.baseline_metrics import (  # noqa: E402
    reward_within_normalized_budget_curve,
    success_within_budget_curve,
    success_within_normalized_budget_curve,
)


# Layout: outputs/<benchmark>/<agent>/<permutation>/<model>/<split>[/runN]/
_SKIP_JSON = {"metrics.json", "run_config.json"}
_RUN_RE = re.compile(r"^run(\d+)$")

# Eval-set sizes used to drop incomplete / debug dumps.
_EXPECTED_N = {
    "alfworld": 134,
    "sciworld": 211,
    # Paper BabyAI cells are 100-task family subsets (full split is 125).
    "babyai": 100,
}
_DEFAULT_T_MAX = {
    "alfworld": 30,
    "sciworld": 120,
    "babyai": 64,
}
_DEFAULT_MAX_STEPS_FALLBACK = {
    "alfworld": 30,
    "sciworld": 100,
    "babyai": 64,
}

# Paper display names. SciWorld uses walle_oracle / walle_oracle_wm.
_PERM_LABEL = {
    "original": "Original",
    "regular": "Original",  # ScienceWorld config name for the no-WM baseline
    "wm": "Belief",
    "wm_push": "Belief (push)",
    "walle": "WALL-E",
    "walle_oracle": "WALL-E",
    "wm_walle": "BB-WM",
    "wm_push_walle": "BB-WM (push)",
    "walle_oracle_wm": "BB-WM",
    "wm_det": "Belief (no mem.)",
    "wm_no_prob": "Belief-NoProb",
}
_PERM_ORDER = [
    "original",
    "regular",
    "wm",
    "wm_push",
    "walle",
    "walle_oracle",
    "wm_walle",
    "wm_push_walle",
    "walle_oracle_wm",
    "wm_det",
    "wm_no_prob",
]
_MAIN_PERMS = {
    "original",
    "regular",
    "wm",
    "walle",
    "walle_oracle",
    "wm_walle",
    "walle_oracle_wm",
}
_AGENT_LABEL = {"react": "ReAct", "reflact": "ReflAct"}
_MODEL_LABEL = {
    "Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "Qwen3-14B": "Qwen3-14B",
    "Qwen3-8B": "Qwen3-8B",
    "Qwen3-1.7B": "Qwen3-1.7B",
    "claude-sonnet-4-6": "Sonnet 4.6",
    "claude-opus-4-7": "Opus 4.7",
}
_PAPER_MODELS = ["Llama-3.1-8B-Instruct", "Qwen3-14B", "claude-sonnet-4-6"]
_COMBINED_MODELS = ["Llama-3.1-8B-Instruct", "Qwen3-14B", "claude-sonnet-4-6"]
_MODEL_ORDER = [
    "Llama-3.1-8B-Instruct",
    "Qwen3-8B",
    "Qwen3-14B",
    "claude-sonnet-4-6",
    "claude-opus-4-7",
]

# Okabe–Ito (colorblind-safe). Keys are canonical method families.
_PERM_STYLE = {
    "original": dict(color="#000000", marker="o", linestyle="-"),
    "wm": dict(color="#0072B2", marker="s", linestyle="-"),
    "wm_push": dict(color="#0072B2", marker="s", linestyle="--"),
    "walle": dict(color="#E69F00", marker="^", linestyle="-"),
    "wm_walle": dict(color="#009E73", marker="D", linestyle="-"),
    "wm_push_walle": dict(color="#009E73", marker="D", linestyle="--"),
    "wm_det": dict(color="#CC79A7", marker="v", linestyle="--"),
    "wm_no_prob": dict(color="#56B4E9", marker="X", linestyle="--"),
}
_PERM_FAMILY = {
    "original": "original",
    "regular": "original",
    "wm": "wm",
    "wm_push": "wm_push",
    "walle": "walle",
    "walle_oracle": "walle",
    "wm_walle": "wm_walle",
    "wm_push_walle": "wm_push_walle",
    "walle_oracle_wm": "wm_walle",
    "wm_det": "wm_det",
    "wm_no_prob": "wm_no_prob",
}

_BENCH_TITLE = {
    "alfworld": "ALFWorld",
    "sciworld": "ScienceWorld",
    "babyai": "BabyAI-Text",
}


def _is_episode_json(name: str) -> bool:
    return name.endswith(".json") and name not in _SKIP_JSON


def load_episode_meta(path: str) -> Dict[str, Any]:
    """Read only the trailing metadata record (skip the chat transcript)."""
    with open(path) as f:
        data = json.load(f)
    info = data[-1]
    return {
        "success": bool(info.get("success", False)),
        "steps": int(info.get("steps", 0) or 0),
        "max_steps": info.get("max_steps"),
        "query_steps": int(info.get("query_steps", 0) or 0),
        "walle_rejections": int(info.get("walle_rejections", 0) or 0),
        "reward": info.get("reward"),
        "terminate_reason": info.get("terminate_reason"),
        "task_id": os.path.splitext(os.path.basename(path))[0],
        "reward_trace": list(info.get("reward_trace") or []),
    }


def load_run_episodes(run_dir: str) -> List[Dict[str, Any]]:
    """Load every per-episode dump in `run_dir`."""
    episodes = []
    for name in sorted(os.listdir(run_dir)):
        if not _is_episode_json(name):
            continue
        episodes.append(load_episode_meta(os.path.join(run_dir, name)))
    return episodes


def _complete_enough(benchmark: str, n: int, min_frac: float) -> bool:
    expected = _EXPECTED_N.get(benchmark)
    if expected is None:
        return n > 0
    return n >= int(min_frac * expected)


def _is_failed_eval(path: str) -> bool:
    """True when every episode died on an agent API error (not a real seed)."""
    metrics_path = os.path.join(path, "metrics.json")
    if not os.path.isfile(metrics_path):
        return False
    try:
        with open(metrics_path) as f:
            metrics = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    n = int(metrics.get("n_tasks") or 0)
    reasons = metrics.get("terminate_reasons") or {}
    n_err = sum(int(c) for k, c in reasons.items() if str(k).startswith("agent_error"))
    return n > 0 and n_err == n


def discover_runs(
    outputs_root: str,
    benchmark: str,
    split: str,
    min_frac: float,
) -> List[Dict[str, Any]]:
    """Find complete eval cells under outputs/<benchmark>/.../<split>[/runN].

    When `unseen/run1` exists, the parent `unseen/` dump is skipped (duplicate
    of run1). When run2/run3 exist but run1 does not, a complete parent dump is
    treated as run1 (ScienceWorld WALL-E layout).
    """
    bench_root = os.path.join(outputs_root, benchmark)
    if not os.path.isdir(bench_root):
        return []

    found: List[Dict[str, Any]] = []
    for agent in sorted(os.listdir(bench_root)):
        agent_dir = os.path.join(bench_root, agent)
        if not os.path.isdir(agent_dir) or agent == "analysis":
            continue
        for perm in sorted(os.listdir(agent_dir)):
            perm_dir = os.path.join(agent_dir, perm)
            if not os.path.isdir(perm_dir):
                continue
            for model in sorted(os.listdir(perm_dir)):
                model_dir = os.path.join(perm_dir, model)
                split_dir = os.path.join(model_dir, split)
                if not os.path.isdir(split_dir):
                    continue

                run_children = []
                run_ids = set()
                for name in os.listdir(split_dir):
                    m = _RUN_RE.match(name)
                    if not m:
                        continue
                    child = os.path.join(split_dir, name)
                    if os.path.isdir(child):
                        run_children.append((int(m.group(1)), child))
                        run_ids.add(int(m.group(1)))

                parent_n = sum(
                    1 for n in os.listdir(split_dir) if _is_episode_json(n)
                )
                candidates: List[Tuple[int, str]] = list(run_children)
                # Parent dump is run1 only when no run1 subdirectory exists.
                if parent_n and 1 not in run_ids:
                    candidates.append((1, split_dir))
                if not candidates and parent_n:
                    candidates.append((1, split_dir))

                for run_id, path in sorted(candidates):
                    n_eps = sum(1 for n in os.listdir(path) if _is_episode_json(n))
                    if not _complete_enough(benchmark, n_eps, min_frac):
                        print(
                            f"skip incomplete {benchmark}/{agent}/{perm}/{model}/"
                            f"{split}/run{run_id} (n={n_eps})"
                        )
                        continue
                    if _is_failed_eval(path):
                        print(
                            f"skip failed eval {benchmark}/{agent}/{perm}/{model}/"
                            f"{split}/run{run_id} (all agent_error)"
                        )
                        continue
                    found.append(
                        {
                            "benchmark": benchmark,
                            "agent": agent,
                            "permutation": perm,
                            "model": model,
                            "split": split,
                            "run": run_id,
                            "path": path,
                            "n_episodes": n_eps,
                        }
                    )
    return found


def _perm_sort_key(perm: str) -> Tuple[int, str]:
    try:
        return (_PERM_ORDER.index(perm), perm)
    except ValueError:
        return (len(_PERM_ORDER), perm)


def _model_sort_key(model: str) -> Tuple[int, str]:
    try:
        return (_MODEL_ORDER.index(model), model)
    except ValueError:
        return (len(_MODEL_ORDER), model)


_PERM_FAMILY_OVERRIDE: Dict[str, str] = {}


def _style_for(perm: str) -> Dict[str, str]:
    family = _PERM_FAMILY_OVERRIDE.get(perm, _PERM_FAMILY.get(perm, perm))
    return dict(_PERM_STYLE.get(family, dict(color="#56B4E9", marker="x", linestyle="-")))


_PERM_LABEL_OVERRIDE: Dict[str, str] = {}


def _label_perm(perm: str) -> str:
    return _PERM_LABEL_OVERRIDE.get(perm, _PERM_LABEL.get(perm, perm))


def _label_model(model: str) -> str:
    return _MODEL_LABEL.get(model, model)


def _label_agent(agent: str) -> str:
    return _AGENT_LABEL.get(agent, agent)


def _auc(budgets: Sequence[float], curve: Sequence[float]) -> float:
    """Mean height of the curve (trapezoid integral / span)."""
    if len(budgets) < 2:
        return float(curve[0]) if curve else 0.0
    area = 0.0
    for i in range(1, len(budgets)):
        dt = float(budgets[i] - budgets[i - 1])
        area += 0.5 * (curve[i] + curve[i - 1]) * dt
    span = float(budgets[-1] - budgets[0])
    return area / span if span else float(curve[-1])


def aggregate_curves(
    runs: List[Dict[str, Any]],
    budgets: List[int],
    cost: str,
    fractions: Optional[List[float]] = None,
    default_max_steps: Optional[int] = None,
) -> Dict[Tuple[str, str, str], Dict[str, Any]]:
    """Group runs by (agent, permutation, model) and average SR(t) over seeds."""
    grouped: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = defaultdict(list)
    for cell in runs:
        key = (cell["agent"], cell["permutation"], cell["model"])
        grouped[key].append(cell)

    out: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for key, cells in grouped.items():
        seed_curves = []
        seed_norm = []
        seed_reward = []
        seed_reward_traced = []
        n_tasks = []
        n_success = []
        for cell in sorted(cells, key=lambda c: c["run"]):
            episodes = load_run_episodes(cell["path"])
            seed_curves.append(success_within_budget_curve(episodes, budgets, cost=cost))
            n_tasks.append(len(episodes))
            n_success.append(sum(1 for ep in episodes if ep["success"]))
            if fractions is not None:
                seed_norm.append(
                    success_within_normalized_budget_curve(
                        episodes,
                        fractions,
                        cost=cost,
                        default_max_steps=default_max_steps,
                    )
                )
                reward_curve = reward_within_normalized_budget_curve(
                    episodes,
                    fractions,
                    default_max_steps=default_max_steps,
                )
                seed_reward.append(reward_curve)
                # Prefer dumps with reward_trace so old binary seeds are not
                # mixed into the anytime-return mean.
                if any(ep.get("reward_trace") for ep in episodes):
                    seed_reward_traced.append(reward_curve)
        n_seeds = len(seed_curves)
        mean = []
        std = []
        for i in range(len(budgets)):
            vals = [c[i] for c in seed_curves]
            mean.append(statistics.mean(vals))
            std.append(statistics.stdev(vals) if n_seeds >= 2 else 0.0)
        record: Dict[str, Any] = {
            "agent": key[0],
            "permutation": key[1],
            "model": key[2],
            "n_seeds": n_seeds,
            "n_tasks": n_tasks,
            "n_success": n_success,
            "runs": [c["run"] for c in cells],
            "paths": [c["path"] for c in cells],
            "budgets": budgets,
            "mean": mean,
            "std": std,
            "auc": _auc(budgets, mean),
            "sr_horizon": mean[-1] if mean else None,
        }
        if fractions is not None and seed_norm:
            norm_mean = []
            norm_std = []
            for i in range(len(fractions)):
                vals = [c[i] for c in seed_norm]
                norm_mean.append(statistics.mean(vals))
                norm_std.append(statistics.stdev(vals) if n_seeds >= 2 else 0.0)
            record["fractions"] = fractions
            record["norm_mean"] = norm_mean
            record["norm_std"] = norm_std
            record["norm_auc"] = _auc(fractions, norm_mean)
        if fractions is not None and seed_reward:
            reward_seeds = seed_reward_traced or seed_reward
            n_reward_seeds = len(reward_seeds)
            if seed_reward_traced and n_reward_seeds < n_seeds:
                print(
                    f"anytime-return uses {n_reward_seeds}/{n_seeds} traced seeds "
                    f"for {key[0]}/{key[1]}/{key[2]}"
                )
            reward_mean = []
            reward_std = []
            for i in range(len(fractions)):
                vals = [c[i] for c in reward_seeds]
                reward_mean.append(statistics.mean(vals))
                reward_std.append(statistics.stdev(vals) if n_reward_seeds >= 2 else 0.0)
            record["reward_mean"] = reward_mean
            record["reward_std"] = reward_std
            record["reward_auc"] = _auc(fractions, reward_mean)
            record["n_reward_seeds"] = n_reward_seeds
        out[key] = record
    return out


def _apply_mpl_style() -> None:
    import matplotlib as mpl

    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.04,
        }
    )


def _marker_every(n_points: int) -> int:
    """Place a handful of markers so dense step-curves stay readable."""
    if n_points <= 12:
        return 1
    return max(1, n_points // 6)


def plot_grid(
    records: Dict[Tuple[str, str, str], Dict[str, Any]],
    agents: List[str],
    models: List[str],
    perms: List[str],
    title: str,
    xlabel: str,
    ylabel: str,
    x_key: str,
    y_key: str,
    std_key: str,
    out_base: str,
    formats: Sequence[str],
) -> None:
    """Rows = agents, columns = models; one solid curve per method.

    No figure title. One shared x-label and y-label for the whole grid;
    ReAct / ReflAct stay as left-column row labels. `title` is unused
    (kept so callers do not have to change).
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    del title  # figure-level title omitted for the paper layout
    n_rows = max(1, len(agents))
    n_cols = max(1, len(models))
    fig_w = min(3.4 * n_cols, 11.2)
    fig_h = min(2.55 * n_rows + 0.15, 6.8)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    for r, agent in enumerate(agents):
        for c, model in enumerate(models):
            ax = axes[r][c]
            ax.set_axisbelow(True)
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
            xs_last = None
            for perm in perms:
                rec = records.get((agent, perm, model))
                if rec is None:
                    continue
                xs = rec[x_key]
                ys = rec[y_key]
                ss = rec[std_key]
                style = _style_for(perm)
                markevery = _marker_every(len(xs))
                ax.plot(
                    xs,
                    ys,
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markevery=markevery,
                    markersize=4.5,
                    linewidth=1.6,
                    zorder=3,
                )
                if rec["n_seeds"] >= 2 and any(v > 1e-12 for v in ss):
                    lo = [max(0.0, y - s) for y, s in zip(ys, ss)]
                    hi = [min(1.0, y + s) for y, s in zip(ys, ss)]
                    ax.fill_between(xs, lo, hi, color=style["color"], alpha=0.18, linewidth=0, zorder=2)
                xs_last = xs
            ax.set_ylim(0.0, 1.02)
            if x_key == "fractions":
                ax.set_xlim(0.0, 1.0)
            elif xs_last is not None:
                ax.set_xlim(xs_last[0], xs_last[-1])
            if r == 0:
                ax.set_title(_label_model(model))
            # Agent name identifies the row; the metric name is shared below.
            ax.set_ylabel(_label_agent(agent) if c == 0 else "")
            ax.set_xlabel("")
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            # Only the left column and bottom row carry tick labels.
            ax.tick_params(
                axis="both",
                which="both",
                labelbottom=(r == n_rows - 1),
                labelleft=(c == 0),
            )

    legend_handles = []
    legend_labels = []
    seen_labels = set()
    for perm in perms:
        if not any((a, perm, m) in records for a in agents for m in models):
            continue
        label = _label_perm(perm)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        style = _style_for(perm)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=5,
                linewidth=1.6,
            )
        )
        legend_labels.append(label)

    if legend_handles:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="outside upper center",
            ncol=min(4, len(legend_labels)),
            frameon=False,
        )
    fig.supxlabel(xlabel)
    fig.supylabel(ylabel)
    for fmt in formats:
        path = f"{out_base}.{fmt}"
        fig.savefig(path)
        print(f"wrote {path}")
    plt.close(fig)


def plot_grid_with_reference(
    records: Dict[Tuple[str, str, str], Dict[str, Any]],
    ref_records: Dict[Tuple[str, str, str], Dict[str, Any]],
    agents: List[str],
    models: List[str],
    perms: List[str],
    xlabel: str,
    ylabel: str,
    x_key: str,
    y_key: str,
    std_key: str,
    out_base: str,
    formats: Sequence[str],
    ref_perms: Optional[Sequence[str]] = None,
    ref_label: str = "Original (uniform)",
    ref_color: str = "#7F7F7F",
    ref_linestyle: str = "-",
    ref_alpha: float = 0.4,
    layout: str = "row",
) -> None:
    """Zipf methods in the foreground, with a faint uniform Original behind.

    `layout="row"` is one panel per (agent, model), left to right. `layout="grid"`
    is the usual agents × models matrix.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    title_fs = 15
    label_fs = 15
    legend_fs = 12

    if layout == "row":
        # Model groups: Llama ReAct, Llama ReflAct, Qwen ReAct, Qwen ReflAct.
        cells = [(agent, model) for model in models for agent in agents]
        n_rows, n_cols = 1, max(1, len(cells))
        fig_w = min(2.7 * n_cols, 12.4)
        fig_h = 3.15
    else:
        cells = [(agent, model) for agent in agents for model in models]
        n_rows = max(1, len(agents))
        n_cols = max(1, len(models))
        fig_w = min(3.4 * n_cols, 11.2)
        fig_h = min(2.55 * n_rows + 0.35, 7.1)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(fig_w, fig_h),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )

    for i, (agent, model) in enumerate(cells):
        if layout == "row":
            ax = axes[0][i]
        else:
            ax = axes[agents.index(agent)][models.index(model)]
        ax.set_axisbelow(True)
        ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
        xs_last = None
        draw_ref_perms = list(ref_perms) if ref_perms is not None else list(perms)
        for perm in draw_ref_perms:
            ref = ref_records.get((agent, perm, model))
            if ref is None:
                continue
            ax.plot(
                ref[x_key],
                ref[y_key],
                color=ref_color,
                linestyle=ref_linestyle,
                marker="None",
                linewidth=1.6,
                alpha=ref_alpha,
                zorder=2,
            )
        for perm in perms:
            rec = records.get((agent, perm, model))
            if rec is None:
                continue
            style = _style_for(perm)
            xs = rec[x_key]
            ys = rec[y_key]
            ss = rec[std_key]
            ax.plot(
                xs,
                ys,
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markevery=_marker_every(len(xs)),
                markersize=4.5,
                linewidth=1.6,
                zorder=3,
            )
            if rec["n_seeds"] >= 2 and any(v > 1e-12 for v in ss):
                lo = [max(0.0, y - s) for y, s in zip(ys, ss)]
                hi = [min(1.0, y + s) for y, s in zip(ys, ss)]
                ax.fill_between(xs, lo, hi, color=style["color"], alpha=0.18, linewidth=0, zorder=2)
            xs_last = xs
        ax.set_ylim(0.0, 1.02)
        if x_key == "fractions":
            ax.set_xlim(0.0, 1.0)
        elif xs_last is not None:
            ax.set_xlim(xs_last[0], xs_last[-1])
        if layout == "row":
            ax.set_title(
                f"{_label_model(model)}  ·  {_label_agent(agent)}",
                fontsize=title_fs,
                pad=3,
            )
            show_left, show_bottom = (i == 0), True
        else:
            r, c = agents.index(agent), models.index(model)
            if r == 0:
                ax.set_title(_label_model(model), fontsize=title_fs, pad=3)
            ax.set_ylabel(_label_agent(agent) if c == 0 else "", fontsize=label_fs)
            show_left, show_bottom = (c == 0), (r == n_rows - 1)
        ax.set_xlabel("")
        ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
        ax.tick_params(
            axis="both",
            which="both",
            labelbottom=show_bottom,
            labelleft=show_left,
        )

    legend_handles = []
    legend_labels = []
    seen_labels = set()
    for perm in perms:
        if not any((a, perm, m) in records for a in agents for m in models):
            continue
        label = _label_perm(perm)
        if label in seen_labels:
            continue
        seen_labels.add(label)
        style = _style_for(perm)
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=5,
                linewidth=1.6,
            )
        )
        legend_labels.append(label)
    if any((a, perm, m) in ref_records for perm in (ref_perms or perms) for a in agents for m in models):
        legend_handles.append(
            Line2D(
                [0],
                [0],
                color=ref_color,
                linestyle=ref_linestyle,
                linewidth=1.6,
                alpha=ref_alpha,
            )
        )
        legend_labels.append(ref_label)

    fig.legend(
        legend_handles,
        legend_labels,
        loc="outside upper center",
        ncol=min(5, len(legend_labels)),
        frameon=False,
        fontsize=legend_fs,
    )
    fig.supxlabel(xlabel, fontsize=label_fs)
    fig.supylabel(ylabel, fontsize=label_fs)
    for fmt in formats:
        path = f"{out_base}.{fmt}"
        fig.savefig(path)
        print(f"wrote {path}")
    plt.close(fig)


def _draw_method_curves(
    ax,
    records: Dict[Tuple[str, str, str], Dict[str, Any]],
    agent: str,
    model: str,
    perms: List[str],
    x_key: str,
    y_key: str,
    std_key: str,
) -> None:
    """Draw one cell's method curves onto `ax`."""
    for perm in perms:
        rec = records.get((agent, perm, model))
        if rec is None:
            continue
        xs = rec[x_key]
        ys = rec[y_key]
        ss = rec[std_key]
        style = _style_for(perm)
        ax.plot(
            xs,
            ys,
            color=style["color"],
            linestyle=style["linestyle"],
            marker=style["marker"],
            markevery=_marker_every(len(xs)),
            markersize=4.0,
            linewidth=1.5,
            zorder=3,
        )
        if rec["n_seeds"] >= 2 and any(v > 1e-12 for v in ss):
            lo = [max(0.0, y - s) for y, s in zip(ys, ss)]
            hi = [min(1.0, y + s) for y, s in zip(ys, ss)]
            ax.fill_between(xs, lo, hi, color=style["color"], alpha=0.18, linewidth=0, zorder=2)


def _method_legend_handles(
    perms: List[str],
    present: Any,
) -> Tuple[list, list]:
    """Legend entries for methods that appear in `present` (callable or records dict)."""
    from matplotlib.lines import Line2D

    handles, labels, seen = [], [], set()
    for perm in perms:
        label = _label_perm(perm)
        if label in seen:
            continue
        if callable(present):
            if not present(perm):
                continue
        seen.add(label)
        style = _style_for(perm)
        handles.append(
            Line2D(
                [0],
                [0],
                color=style["color"],
                linestyle=style["linestyle"],
                marker=style["marker"],
                markersize=5,
                linewidth=1.6,
            )
        )
        labels.append(label)
    return handles, labels


def plot_combined_normalized(
    records_by_bench: Dict[str, Dict[Tuple[str, str, str], Dict[str, Any]]],
    models: List[str],
    agents: List[str],
    perms: List[str],
    out_base: str,
    formats: Sequence[str],
) -> None:
    """Paper figure: rows = benchmarks, columns = (model × ReAct/ReflAct).

    Shared x-axis is budget fraction α. Panel width is held fixed as models are
    added (Llama / Qwen / Sonnet). BabyAI, when present, is a third row. Its
    Belief and BB-WM curves are the 6x6 placement-prior world model.
    """
    import matplotlib.pyplot as plt

    benches = [b for b in ("alfworld", "sciworld", "babyai") if b in records_by_bench]
    if "alfworld" not in benches or "sciworld" not in benches or len(models) < 2 or len(agents) < 2:
        print("skip combined figure (need ALFWorld + ScienceWorld, two models, two agents)")
        return

    n_benches = len(benches)
    n_models = len(models)
    title_fs = 15
    agent_fs = 13.5
    label_fs = 15
    legend_fs = 12
    # 5.1in per model group keeps Llama+Qwen at the original 10.2in width.
    fig = plt.figure(figsize=(5.1 * n_models, 2.55 * n_benches + 0.1), layout="constrained")
    engine = fig.get_layout_engine()
    if engine is not None:
        engine.set(wspace=0.01, w_pad=0.02)
    # Half the previous 0.05 gap between model groups.
    subfigs = fig.subfigures(1, n_models, wspace=0.025)

    for si, (subfig, model) in enumerate(zip(subfigs, models)):
        subfig.suptitle(_label_model(model), fontsize=title_fs)
        axs = subfig.subplots(n_benches, 2, sharex=True, sharey=True)
        for r, bench in enumerate(benches):
            records = records_by_bench[bench]
            bench_perms = perms
            for c, agent in enumerate(agents):
                ax = axs[r, c]
                ax.set_axisbelow(True)
                ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
                _draw_method_curves(
                    ax, records, agent, model, bench_perms,
                    x_key="fractions", y_key="reward_mean", std_key="reward_std",
                )
                ax.set_xlim(0.0, 1.0)
                ax.set_ylim(0.0, 1.02)
                ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
                ax.set_xlabel("")
                if r == 0:
                    ax.set_title(_label_agent(agent), fontsize=agent_fs, pad=3)
                # Benchmark names only on the far-left column of the whole figure.
                if si == 0 and c == 0:
                    ax.set_ylabel(_BENCH_TITLE.get(bench, bench), fontsize=label_fs)
                else:
                    ax.set_ylabel("")
                ax.tick_params(
                    axis="both",
                    which="both",
                    labelbottom=(r == n_benches - 1),
                    labelleft=(si == 0 and c == 0),
                )

    def _perm_present(perm: str) -> bool:
        return any(
            perm in perms
            and (agent, perm, model) in records_by_bench[bench]
            for bench in benches
            for agent in agents
            for model in models
        )

    handles, labels = _method_legend_handles(perms, _perm_present)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="outside upper center",
            ncol=min(6, len(labels)),
            frameon=False,
            fontsize=legend_fs,
        )
    fig.supxlabel("Budget fraction", fontsize=label_fs)
    fig.supylabel("Reward", fontsize=label_fs)
    for fmt in formats:
        path = f"{out_base}.{fmt}"
        fig.savefig(path)
        print(f"wrote {path}")
    plt.close(fig)


_WM_PUSH_COMBINED_PERMS = [
    "original",
    "regular",
    "wm",
    "wm_push",
    "wm_walle",
    "walle_oracle_wm",
    "wm_push_walle",
]
_WM_PUSH_COMBINED_MODELS = ["Llama-3.1-8B-Instruct", "Qwen3-14B"]


def plot_combined_wm_push(
    records_by_bench: Dict[str, Dict[Tuple[str, str, str], Dict[str, Any]]],
    agents: List[str],
    out_base: str,
    formats: Sequence[str],
    models: Optional[List[str]] = None,
) -> None:
    """ALFWorld + ScienceWorld push figure: no WALL-E, no Sonnet, dotted push lines."""
    models = models or [
        m
        for m in _WM_PUSH_COMBINED_MODELS
        if any(k[2] == m for recs in records_by_bench.values() for k in recs)
    ]
    perms = [
        p
        for p in _WM_PUSH_COMBINED_PERMS
        if any(k[1] == p for recs in records_by_bench.values() for k in recs)
    ]
    filtered = {
        bench: recs
        for bench, recs in records_by_bench.items()
        if bench in ("alfworld", "sciworld")
    }
    saved = {key: dict(_PERM_STYLE[key]) for key in ("wm_push", "wm_push_walle")}
    _PERM_STYLE["wm_push"]["linestyle"] = ":"
    _PERM_STYLE["wm_push_walle"]["linestyle"] = ":"
    try:
        plot_combined_normalized(
            filtered,
            models=models,
            agents=agents,
            perms=perms,
            out_base=out_base,
            formats=formats,
        )
    finally:
        for key, style in saved.items():
            _PERM_STYLE[key] = style


def write_curves_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"wrote {path}")


def write_summary_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    print(f"wrote {path}")


def print_table(benchmark: str, records: Dict[Tuple[str, str, str], Dict[str, Any]]) -> None:
    """Compact AUC / SR@horizon table for the paper caption."""
    rows = sorted(
        records.values(),
        key=lambda r: (_label_agent(r["agent"]), _model_sort_key(r["model"]), _perm_sort_key(r["permutation"])),
    )
    print(f"\n=== { _BENCH_TITLE.get(benchmark, benchmark) } success-within-budget ===")
    print(
        f"{'agent':8} {'method':16} {'model':16} {'seeds':5} "
        f"{'SR@H':7} {'AUC':7} {'n':5}"
    )
    for rec in rows:
        print(
            f"{_label_agent(rec['agent']):8} {_label_perm(rec['permutation']):16} "
            f"{_label_model(rec['model']):16} {rec['n_seeds']:5d} "
            f"{rec['sr_horizon']:.3f}  {rec['auc']:.3f}  {rec['n_tasks'][0]:5d}"
        )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--outputs", default=os.path.join(_ROOT, "outputs"), help="Eval dump root.")
    p.add_argument(
        "--out-dir",
        default=os.path.join(_ROOT, "figures", "success_within_budget"),
        help="Where to write PDF/PNG/JSON/CSV.",
    )
    p.add_argument(
        "--benchmarks",
        default="alfworld,sciworld",
        help="Comma-separated: alfworld,sciworld,babyai.",
    )
    p.add_argument("--split", default="unseen", help="Eval split folder name.")
    p.add_argument(
        "--reference-split",
        default=None,
        help="If set, overlay this split as a comparison (grey dashed Original by default).",
    )
    p.add_argument(
        "--reference-methods",
        default="original,regular",
        help="Permutations from --reference-split to overlay. Default: original/regular only.",
    )
    p.add_argument(
        "--reference-label",
        default="Original (uniform)",
        help="Legend label for the reference overlay.",
    )
    p.add_argument(
        "--models",
        default="paper",
        help="'paper' = Llama-3.1-8B, Qwen3-14B, Sonnet 4.6; 'all' = every model with dumps; or a comma-separated list.",
    )
    p.add_argument(
        "--methods",
        default="main",
        help="'main' = Original/Belief/WALL-E/BB-WM; 'all' includes wm_det and wm_no_prob; or a comma-separated permutation list.",
    )
    p.add_argument(
        "--agents",
        default="react,reflact",
        help="'all' or a comma-separated list (react,reflact). Each agent is one row.",
    )
    p.add_argument("--cost", choices=["env", "all"], default="env", help="env steps, or env+queries+WALL-E rejects.")
    p.add_argument("--t-max", type=int, default=None, help="Override the x-axis horizon (absolute-step plot).")
    p.add_argument("--min-frac", type=float, default=0.9, help="Drop runs with fewer than this fraction of expected episodes.")
    p.add_argument("--formats", default="pdf,png", help="Image formats to write.")
    p.add_argument("--xlabel", default=None, help="Override the shared x-axis label.")
    p.add_argument("--ylabel", default=None, help="Override the shared y-axis label.")
    p.add_argument("--skip-normalized", action="store_true", help="Do not write the SciWorld fraction-of-budget figure.")
    p.add_argument(
        "--normalized",
        action="store_true",
        help="Primary figure uses budget fraction α of per-task T (SciWorld: anytime return).",
    )
    p.add_argument(
        "--out-stem",
        default=None,
        help="If set, write {out-stem}_sr_within_budget instead of <bench>_sr_within_budget and skip the combined figure.",
    )
    p.add_argument(
        "--perm-labels",
        default=None,
        help="Comma-separated perm=Label overrides, e.g. wm=Memory,wm_det=Belief.",
    )
    p.add_argument(
        "--perm-families",
        default=None,
        help="Comma-separated perm=style-family overrides, e.g. wm=wm_det,wm_det=wm.",
    )
    return p.parse_args()


def _parse_models(raw: str, available: Iterable[str]) -> List[str]:
    avail = list(available)
    if raw == "paper":
        chosen = [m for m in _PAPER_MODELS if m in avail]
        return chosen or sorted(avail, key=_model_sort_key)
    if raw == "all":
        return sorted(avail, key=_model_sort_key)
    requested = [m.strip() for m in raw.split(",") if m.strip()]
    missing = [m for m in requested if m not in avail]
    if missing:
        print(f"warning: requested models not found: {missing}")
    return [m for m in requested if m in avail]


def _parse_perm_labels(raw: Optional[str]) -> Dict[str, str]:
    if not raw:
        return {}
    out: Dict[str, str] = {}
    for part in raw.split(","):
        if "=" not in part:
            continue
        key, val = part.split("=", 1)
        key, val = key.strip(), val.strip()
        if key and val:
            out[key] = val
    return out


def _parse_methods(raw: str, available: Iterable[str]) -> List[str]:
    avail = set(available)
    if raw == "main":
        return [p for p in _PERM_ORDER if p in avail and p in _MAIN_PERMS]
    if raw == "all":
        extra = sorted(avail - set(_PERM_ORDER))
        return [p for p in _PERM_ORDER if p in avail] + extra
    requested = [m.strip() for m in raw.split(",") if m.strip()]
    return [p for p in requested if p in avail]


def main() -> None:
    args = parse_args()
    try:
        import matplotlib  # noqa: F401
    except ImportError:
        sys.exit(
            "matplotlib is required for paper figures. Install with:\n"
            "  pip install matplotlib\n"
            "then re-run this script."
        )

    _apply_mpl_style()
    _PERM_LABEL_OVERRIDE.update(_parse_perm_labels(args.perm_labels))
    _PERM_FAMILY_OVERRIDE.update(_parse_perm_labels(args.perm_families))
    os.makedirs(args.out_dir, exist_ok=True)
    formats = [f.strip() for f in args.formats.split(",") if f.strip()]
    benchmarks = [b.strip() for b in args.benchmarks.split(",") if b.strip()]
    records_by_bench: Dict[str, Dict[Tuple[str, str, str], Dict[str, Any]]] = {}
    combined_perms: List[str] = []

    for benchmark in benchmarks:
        runs = discover_runs(args.outputs, benchmark, args.split, args.min_frac)
        if not runs:
            print(f"no complete {benchmark} runs under {args.outputs}/{benchmark}")
            continue

        t_max = args.t_max or _DEFAULT_T_MAX.get(benchmark, 30)
        budgets = list(range(1, t_max + 1))
        fractions = [i / 20.0 for i in range(1, 21)]  # 0.05, 0.10, ..., 1.00
        default_max = _DEFAULT_MAX_STEPS_FALLBACK.get(benchmark)
        records = aggregate_curves(
            runs,
            budgets,
            cost=args.cost,
            fractions=fractions,
            default_max_steps=default_max,
        )

        models = _parse_models(args.models, {k[2] for k in records})
        perms = _parse_methods(args.methods, {k[1] for k in records})
        if args.agents == "all":
            requested_agents = None
        else:
            requested_agents = {a.strip() for a in args.agents.split(",") if a.strip()}
        agents = [a for a in ("react", "reflact") if any(k[0] == a for k in records)]
        extra_agents = sorted({k[0] for k in records} - set(agents))
        agents = agents + extra_agents
        if requested_agents is not None:
            agents = [a for a in agents if a in requested_agents]

        # Restrict the plotted grid; keep full `records` in the JSON dump.
        plot_records = {
            k: v
            for k, v in records.items()
            if k[0] in agents and k[1] in perms and k[2] in models
        }
        agents = [a for a in agents if any(k[0] == a for k in plot_records)]
        models = [m for m in models if any(k[2] == m for k in plot_records)]
        if not plot_records:
            print(f"nothing to plot for {benchmark} after --models/--methods filters")
            continue

        fig_stem = args.out_stem or f"{benchmark}_sr_within_budget"
        if args.normalized:
            xlabel = r"Budget fraction $\alpha$ of per-task $T$"
            if benchmark == "sciworld":
                ylabel, x_key, y_key, std_key = "Anytime return", "fractions", "reward_mean", "reward_std"
            else:
                ylabel, x_key, y_key, std_key = "Success rate", "fractions", "norm_mean", "norm_std"
        else:
            xlabel = "Environment-step budget $t$" if args.cost == "env" else "Action-budget $t$ (env + queries + rejects)"
            ylabel, x_key, y_key, std_key = "Success rate", "budgets", "mean", "std"
        if args.xlabel:
            xlabel = args.xlabel
        if args.ylabel:
            ylabel = args.ylabel
        ref_plot_records = None
        if args.reference_split:
            ref_runs = discover_runs(args.outputs, benchmark, args.reference_split, args.min_frac)
            if ref_runs:
                ref_all = aggregate_curves(
                    ref_runs,
                    budgets,
                    cost=args.cost,
                    fractions=fractions,
                    default_max_steps=default_max,
                )
                ref_methods = [m.strip() for m in args.reference_methods.split(",") if m.strip()]
                ref_plot_records = {
                    k: v
                    for k, v in ref_all.items()
                    if k[0] in agents and k[1] in ref_methods and k[2] in models
                }
            else:
                print(f"warning: no complete {benchmark} runs for --reference-split {args.reference_split}")

        plot_kwargs = dict(
            agents=agents,
            models=models,
            perms=perms,
            xlabel=xlabel,
            ylabel=ylabel,
            x_key=x_key,
            y_key=y_key,
            std_key=std_key,
            out_base=os.path.join(args.out_dir, fig_stem),
            formats=formats,
        )
        if ref_plot_records:
            plot_grid_with_reference(
                plot_records,
                ref_plot_records,
                ref_perms=sorted({k[1] for k in ref_plot_records}),
                ref_label=args.reference_label,
                **plot_kwargs,
            )
        else:
            plot_grid(
                plot_records,
                title=f"{_BENCH_TITLE.get(benchmark, benchmark)} success within budget",
                **plot_kwargs,
            )

        if benchmark == "sciworld" and not args.skip_normalized and not args.normalized:
            plot_grid(
                plot_records,
                agents=agents,
                models=models,
                perms=perms,
                title="ScienceWorld anytime return within fraction of per-task budget",
                xlabel=r"Budget fraction $\alpha$ of per-task $T$",
                ylabel="Anytime return",
                x_key="fractions",
                y_key="reward_mean",
                std_key="reward_std",
                out_base=os.path.join(args.out_dir, "sciworld_sr_within_normalized_budget"),
                formats=formats,
            )

        json_payload = {
            "benchmark": benchmark,
            "split": args.split,
            "cost": args.cost,
            "metric": "P(success and solve_time <= t)",
            "t_max": t_max,
            "cells": [],
        }
        csv_rows = []
        for rec in sorted(
            plot_records.values(),
            key=lambda r: (r["agent"], _model_sort_key(r["model"]), _perm_sort_key(r["permutation"])),
        ):
            json_payload["cells"].append(
                {
                    "agent": rec["agent"],
                    "permutation": rec["permutation"],
                    "method": _label_perm(rec["permutation"]),
                    "model": rec["model"],
                    "n_seeds": rec["n_seeds"],
                    "n_tasks": rec["n_tasks"],
                    "runs": rec["runs"],
                    "budgets": rec["budgets"],
                    "mean": rec["mean"],
                    "std": rec["std"],
                    "auc": rec["auc"],
                    "sr_horizon": rec["sr_horizon"],
                    "fractions": rec.get("fractions"),
                    "norm_mean": rec.get("norm_mean"),
                    "norm_std": rec.get("norm_std"),
                    "norm_auc": rec.get("norm_auc"),
                    "reward_mean": rec.get("reward_mean"),
                    "reward_std": rec.get("reward_std"),
                    "reward_auc": rec.get("reward_auc"),
                }
            )
            csv_rows.append(
                {
                    "benchmark": benchmark,
                    "agent": rec["agent"],
                    "method": _label_perm(rec["permutation"]),
                    "permutation": rec["permutation"],
                    "model": rec["model"],
                    "n_seeds": rec["n_seeds"],
                    "n_tasks": rec["n_tasks"][0],
                    "sr_horizon": f"{rec['sr_horizon']:.6f}",
                    "auc": f"{rec['auc']:.6f}",
                    "norm_auc": f"{rec.get('norm_auc', '')}",
                }
            )
        json_name = f"{args.out_stem}_curves.json" if args.out_stem else f"{benchmark}_curves.json"
        csv_name = f"{args.out_stem}_summary.csv" if args.out_stem else f"{benchmark}_summary.csv"
        write_curves_json(os.path.join(args.out_dir, json_name), json_payload)
        write_summary_csv(
            os.path.join(args.out_dir, csv_name),
            csv_rows,
            fieldnames=[
                "benchmark",
                "agent",
                "method",
                "permutation",
                "model",
                "n_seeds",
                "n_tasks",
                "sr_horizon",
                "auc",
                "norm_auc",
            ],
        )
        print_table(benchmark, plot_records)
        records_by_bench[benchmark] = plot_records
        if not combined_perms:
            combined_perms = list(perms)
        else:
            # Keep paper order; union methods seen on either benchmark.
            seen = set(combined_perms)
            combined_perms.extend(p for p in perms if p not in seen)

    if args.out_stem:
        return
    combined_models = [m for m in _COMBINED_MODELS if any(
        k[2] == m for recs in records_by_bench.values() for k in recs
    )]
    combined_agents = [a for a in ("react", "reflact") if any(
        k[0] == a for recs in records_by_bench.values() for k in recs
    )]
    plot_combined_normalized(
        records_by_bench,
        models=combined_models,
        agents=combined_agents,
        perms=combined_perms,
        out_base=os.path.join(args.out_dir, "combined_sr_within_normalized_budget"),
        formats=formats,
    )


if __name__ == "__main__":
    main()
