"""List failed BabyAI (or any eval) episode JSONs from a run directory.

Stdlib + utils.datatypes only — safe on a login node. Skips metrics.json /
run_config.json. Writes a JSON summary of every unsuccessful episode.

Example:
  python scripts/list_failed_episodes.py \\
    --run_dir outputs/babyai/reflact/walle_oracle/Llama-3.1-8B-Instruct/unseen_goto_pickup/run1
"""

import argparse
import json
import os
import sys

# Repo root on sys.path so `utils` imports work from any cwd.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from utils.datatypes import State


def _family(task_id: str) -> str:
    """'goto_12' -> 'goto'; last '_' + digits is the seed suffix."""
    if "_" not in task_id:
        return task_id
    head, tail = task_id.rsplit("_", 1)
    return head if tail.isdigit() else task_id


def collect_failures(run_dir: str) -> dict:
    """Scan `run_dir` for episode dumps and return the failed subset."""
    run_dir = os.path.abspath(run_dir)
    if not os.path.isdir(run_dir):
        raise FileNotFoundError(f"run directory not found: {run_dir}")

    failures = []
    n_episodes = 0
    for name in sorted(os.listdir(run_dir)):
        if not name.endswith(".json") or name in ("metrics.json", "run_config.json"):
            continue
        path = os.path.join(run_dir, name)
        n_episodes += 1
        state = State.load_json(json.load(open(path)))
        if state.success:
            continue
        task_id = name.rsplit(".", 1)[0]
        failures.append({
            "task_id": task_id,
            "family": _family(task_id),
            "path": path,
            "success": bool(state.success),
            "terminate_reason": state.terminate_reason,
            "steps": state.steps,
            "max_steps": state.max_steps,
            "query_steps": state.query_steps,
            "walle_rejections": state.walle_rejections,
            "reward": state.reward,
        })

    return {
        "run_dir": run_dir,
        "n_episodes": n_episodes,
        "n_failed": len(failures),
        "failures": failures,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run_dir",
        required=True,
        help="Eval output directory containing per-episode JSON dumps.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output JSON path. Default: outputs/babyai/analysis/failed_<split>_<run>.json "
             "(kept out of the eval run dir so resume globs ignore it).",
    )
    args = parser.parse_args()

    payload = collect_failures(args.run_dir)

    if args.out:
        out_path = os.path.abspath(args.out)
    else:
        analysis = os.path.join(_ROOT, "outputs", "babyai", "analysis")
        os.makedirs(analysis, exist_ok=True)
        tag = os.path.basename(os.path.abspath(args.run_dir).rstrip("/"))
        parent = os.path.basename(os.path.dirname(os.path.abspath(args.run_dir)))
        out_path = os.path.join(analysis, f"failed_{parent}_{tag}.json")

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")

    print(f"Wrote {payload['n_failed']}/{payload['n_episodes']} failures to {out_path}")
    for item in payload["failures"]:
        print(f"  {item['task_id']:20} {item['terminate_reason']:24} {item['path']}")


if __name__ == "__main__":
    main()
