"""ALFWorld presentation figures.

Slides share one grid: columns are LLMs, rows are agent frameworks, axes are
the success-within-budget metric. Curves are added one method at a time.
"""

from __future__ import annotations

import os
import sys
from typing import Dict, List, Sequence, Tuple

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from scripts.plot_success_within_budget import (  # noqa: E402
    _DEFAULT_MAX_STEPS_FALLBACK,
    _DEFAULT_T_MAX,
    _apply_mpl_style,
    _draw_method_curves,
    _label_agent,
    _label_model,
    _method_legend_handles,
    aggregate_curves,
    discover_runs,
)

_MODELS = ["Llama-3.1-8B-Instruct", "Qwen3-14B", "claude-sonnet-4-6"]
_AGENTS = ["react", "reflact"]
_OUT_DIR = os.path.join(_ROOT, "figures", "presentation")
_OUTPUTS = os.path.join(_ROOT, "outputs")


def _presentation_style() -> None:
    import matplotlib.pyplot as plt

    _apply_mpl_style()
    plt.rcParams.update(
        {
            "font.size": 13,
            "axes.labelsize": 15,
            "axes.titlesize": 16,
            "legend.fontsize": 13,
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
        }
    )


def _new_grid():
    """Empty ALFWorld panels with the shared presentation layout."""
    import matplotlib.pyplot as plt

    t_max = _DEFAULT_T_MAX["alfworld"]
    n_rows, n_cols = len(_AGENTS), len(_MODELS)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(11.4, 6.2),
        squeeze=False,
        sharex=True,
        sharey=True,
        layout="constrained",
    )
    for r, agent in enumerate(_AGENTS):
        for c, model in enumerate(_MODELS):
            ax = axes[r][c]
            ax.set_axisbelow(True)
            ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
            ax.set_xlim(1, t_max)
            ax.set_ylim(0.0, 1.02)
            ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_xticks([5, 10, 15, 20, 25, 30])
            if r == 0:
                ax.set_title(_label_model(model), pad=6)
            ax.set_ylabel(_label_agent(agent) if c == 0 else "")
            ax.set_xlabel("")
            ax.tick_params(
                axis="both",
                which="both",
                labelbottom=(r == n_rows - 1),
                labelleft=(c == 0),
            )
    fig.supxlabel(r"Environment-step budget $t$")
    fig.supylabel("Success rate")
    return fig, axes


def _save(fig, out_base: str) -> None:
    import matplotlib.pyplot as plt

    os.makedirs(os.path.dirname(out_base), exist_ok=True)
    for fmt in ("pdf", "png"):
        path = f"{out_base}.{fmt}"
        fig.savefig(path, dpi=200)
        print(f"wrote {path}")
    plt.close(fig)


def _load_records(perms: Sequence[str]) -> Dict[Tuple[str, str, str], dict]:
    t_max = _DEFAULT_T_MAX["alfworld"]
    budgets = list(range(1, t_max + 1))
    runs = discover_runs(_OUTPUTS, "alfworld", "unseen", min_frac=0.9)
    runs = [
        cell
        for cell in runs
        if cell["agent"] in _AGENTS
        and cell["model"] in _MODELS
        and cell["permutation"] in perms
    ]
    records = aggregate_curves(
        runs,
        budgets,
        cost="env",
        fractions=None,
        default_max_steps=_DEFAULT_MAX_STEPS_FALLBACK["alfworld"],
    )
    missing = [
        (agent, perm, model)
        for perm in perms
        for agent in _AGENTS
        for model in _MODELS
        if (agent, perm, model) not in records
    ]
    if missing:
        print(f"warning: missing cells: {missing}")
    return records


def plot_empty_grid(out_base: str) -> None:
    """Axes only: introduce models, agents, and the success-within-budget metric."""
    _presentation_style()
    fig, _axes = _new_grid()
    _save(fig, out_base)


def plot_methods(
    out_base: str,
    perms: List[str],
    legend: bool = True,
) -> None:
    """Draw `perms` onto the shared grid. Order matches the paper legend."""
    _presentation_style()
    records = _load_records(perms)
    fig, axes = _new_grid()
    for r, agent in enumerate(_AGENTS):
        for c, model in enumerate(_MODELS):
            _draw_method_curves(
                axes[r][c],
                records,
                agent,
                model,
                perms,
                x_key="budgets",
                y_key="mean",
                std_key="std",
            )
    if legend:
        handles, labels = _method_legend_handles(
            perms,
            lambda perm: any((a, perm, m) in records for a in _AGENTS for m in _MODELS),
        )
        if handles:
            # Leave a short gap between the legend and the column titles.
            engine = fig.get_layout_engine()
            if engine is not None:
                engine.set(rect=(0, 0, 1, 0.955))
            fig.legend(
                handles,
                labels,
                loc="upper center",
                bbox_to_anchor=(0.5, 1.0),
                bbox_transform=fig.transFigure,
                ncol=min(4, len(labels)),
                frameon=False,
                borderaxespad=0,
            )
    _save(fig, out_base)
    for perm in perms:
        for agent in _AGENTS:
            for model in _MODELS:
                rec = records.get((agent, perm, model))
                if rec is None:
                    continue
                print(
                    f"{_label_agent(agent):8} {perm:12} {_label_model(model):16} "
                    f"seeds={rec['n_seeds']}  SR@30={rec['sr_horizon']:.3f}"
                )


def main() -> None:
    plot_empty_grid(os.path.join(_OUT_DIR, "alfworld_axes"))
    plot_methods(os.path.join(_OUT_DIR, "alfworld_original"), ["original"])
    plot_methods(os.path.join(_OUT_DIR, "alfworld_original_belief"), ["original", "wm"])
    plot_methods(
        os.path.join(_OUT_DIR, "alfworld_original_belief_walle"),
        ["original", "wm", "walle"],
    )
    plot_methods(
        os.path.join(_OUT_DIR, "alfworld_original_belief_walle_bbwm"),
        ["original", "wm", "walle", "wm_walle"],
    )


if __name__ == "__main__":
    main()
