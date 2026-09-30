"""Evaluation harness for Belief-Based World Models on ALFWorld, ScienceWorld, and BabyAI-Text.

Output layout:
  ALFWorld:  outputs/alfworld/<agent>/<permutation>/<model>/<split>[/runN]/
  SciWorld:  outputs/sciworld/<agent>/<permutation>/<model>/<split>[/runN]/
  BabyAI:    outputs/babyai/<agent>/<permutation>/<model>/<split>[/runN]/
"""

import argparse
import json
import logging
import os
import pathlib
from typing import Any, Dict

from colorama import Fore
from tqdm import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

import agents as agents
import envs as envs
import tasks as tasks
from agents.baseline_metrics import compute_metrics, write_metrics
from agents.prompt_history import build_prompt_messages
from utils.datatypes import State

logger = logging.getLogger("agent_eval")
args = None


def interactive_loop(
    task: tasks.Task,
    agent: agents.BaseAgent,
    env_config: Dict[str, Any],
) -> State:
    logger.info(f"Loading environment: {env_config['env_class']}")
    env: envs.BaseEnv = getattr(envs, env_config["env_class"])(task, **env_config)
    env.args = args
    observation, state = env.reset()

    logger.info(f"\n{Fore.YELLOW}{observation}{Fore.RESET}")

    while not state.finished:
        try:
            # JSON / state.history stays full; the API may see a windowed / compacted view.
            messages = build_prompt_messages(
                state.history,
                max_obs=env_config.get("prompt_max_obs"),
                compact_walle=bool(env_config.get("prompt_compact_walle", False)),
            )
            llm_output: str = agent(messages) or ""
            logger.info(f"\n{Fore.GREEN}{llm_output}{Fore.RESET}\n")
        except Exception as e:
            logger.info(f"Agent failed with error: {type(e).__name__}: {e}")
            state.success = False
            state.finished = True
            state.terminate_reason = f"agent_error: {type(e).__name__}"
            break

        try:
            observation, state = env.step(llm_output)
        except Exception as e:
            logger.info(f"Env step failed with error: {type(e).__name__}: {e}")
            state.success = False
            state.finished = True
            state.terminate_reason = f"env_error: {type(e).__name__}"
            break

        if not state.finished:
            logger.info(f"\n{Fore.BLUE}{observation}{Fore.RESET}\n")
        if state.finished:
            break

    if state.reward is not None:
        logger.info(
            f"Task finished in {state.steps} steps. Success: {state.success}. Reward: {state.reward}"
        )
    else:
        logger.info(f"Task finished in {state.steps} steps. Success: {state.success}")

    return state


def main(args: argparse.Namespace):
    with open(os.path.join(args.exp_path, f"{args.exp_config}.json")) as f:
        exp_config: Dict[str, Any] = json.load(f)
    with open(os.path.join(args.agent_path, f"{args.agent_config}.json")) as f:
        agent_config: Dict[str, Any] = json.load(f)

    if args.model_name is not None:
        agent_config["config"]["model_name"] = args.model_name
    if args.api_base is not None:
        agent_config["config"]["api_base"] = args.api_base
    if args.api_key is not None:
        agent_config["config"]["api_key"] = args.api_key

    model_tag = (
        agent_config["config"].get("model_tag")
        or agent_config["config"]["model_name"].split("/")[-1]
    )

    split_name = {"dev": "seen", "test": "unseen", "train": "train"}.get(args.split, args.split)
    split_suffix = exp_config.get("env_config", {}).get("split_suffix")
    if split_suffix:
        split_name = f"{split_name}_{split_suffix}"
    # ALFWorld only. CLI overrides env_config.placement; default stays uniform.
    # Zipf runs land in <split>_zipf so they cannot overwrite a uniform eval.
    placement = None
    if exp_config.get("benchmark") == "alfworld":
        configured = exp_config.get("env_config", {}).get("placement")
        placement = args.placement or configured or "uniform"
        if placement not in ("uniform", "zipf"):
            raise ValueError(f"placement must be 'uniform' or 'zipf', got {placement!r}")
        if placement == "zipf":
            from tasks.alfworld_zipf import zipf_output_split
            split_name = zipf_output_split(split_name)
    # Optional BabyAI family subset. Accept comma / + / : so Slurm --export
    # (which splits on commas) can pass FAMILIES=goto+pickup. Families filter
    # the task list only; they do not change the output folder (always
    # unseen/seen/train) so goto+pickup and putnext+seq share one run dir.
    families = None
    if getattr(args, "families", None):
        raw = args.families.replace("+", ",").replace(":", ",")
        families = [f.strip() for f in raw.split(",") if f.strip()]
    if args.output_dir is not None:
        output_path = args.output_dir
        method_label = (
            f"{exp_config['agent']}_{exp_config['permutation']}"
            if "agent" in exp_config and "permutation" in exp_config
            else args.method
        )
    else:
        try:
            benchmark = exp_config["benchmark"]
            agent_name = exp_config["agent"]
            permutation = exp_config["permutation"]
        except KeyError as e:
            raise KeyError(
                f"exp_config '{args.exp_config}' is missing required output-path field {e}. "
                f"Add 'benchmark'/'agent'/'permutation' to the config, or pass --output_dir."
            ) from None
        segments = ["outputs", benchmark, agent_name, permutation, model_tag, split_name]
        if args.run is not None:
            segments.append(f"run{args.run}")
        output_path = os.path.join(*segments)
        method_label = f"{agent_name}_{permutation}"
    pathlib.Path(output_path).mkdir(parents=True, exist_ok=True)

    file_handler = logging.FileHandler(os.path.join(output_path, "log.txt"), mode="w")
    logging.basicConfig(
        format="%(message)s",
        handlers=[logging.StreamHandler(), file_handler],
    )

    env_config = exp_config["env_config"]
    if placement is not None:
        env_config["placement"] = placement
    if args.max_steps is not None:
        env_config["max_steps"] = args.max_steps
    logger.info(f"Experiment config: \n{json.dumps(exp_config, indent=2)}")
    logger.info(f"Method: {method_label} | Model: {model_tag} | Split: {args.split} ({split_name})")
    logger.info(f"max_steps: {env_config.get('max_steps')}")

    def _redact(cfg):
        c = json.loads(json.dumps(cfg))
        inner = c.get("config", {})
        if inner.get("api_key"):
            inner["api_key"] = "***REDACTED***"
        return c

    run_config_path = os.path.join(output_path, "run_config.json")
    if args.override or not os.path.exists(run_config_path):
        run_config = {
            "exp_config": exp_config,
            "agent_config": _redact(agent_config),
            "cli_args": {
                k: ("***REDACTED***" if k == "api_key" and v else v)
                for k, v in vars(args).items()
            },
            "output_path": output_path,
            "split": split_name,
        }
        with open(run_config_path, "w") as f:
            json.dump(run_config, f, indent=2)
        logger.info(f"Wrote run provenance to {run_config_path}")
    else:
        logger.info(f"run_config.json exists (resume without --override); leaving it as-is.")

    if env_config["env_class"] in ("SciWorldEnv", "WMSciWorldEnv",
                                    "DeterministicWMSciWorldEnv",
                                    "WalleOracleSciWorldEnv", "WalleWMOracleSciWorldEnv"):
        from scienceworld import ScienceWorldEnv
        from utils.sciworld_score import sciworld_monkey_patch
        sciworld_monkey_patch()
        env_config["env"] = ScienceWorldEnv("", envStepLimit=200)

    task_config: Dict[str, Any] = exp_config["task"]
    task_class: tasks.Task = getattr(tasks, task_config["task_class"])
    load_kwargs = {
        "path": task_config.get("filepath", ""),
        "split": args.split,
        "part_num": args.part_num,
        "part_idx": args.part_idx,
    }
    if families is not None:
        load_kwargs["families"] = families
    if placement is not None:
        load_kwargs["placement"] = placement
    all_tasks, n_tasks = task_class.load_tasks(**load_kwargs)

    agent: agents.LMAgent = getattr(agents, agent_config["agent_class"])(agent_config["config"])

    state_list = []
    done_task_id = []
    if os.path.exists(output_path) and not args.override:
        for file in os.listdir(output_path):
            if not file.endswith("json") or file in ("metrics.json", "run_config.json"):
                continue
            state = State.load_json(json.load(open(os.path.join(output_path, file))))
            state_list.append(state)
            done_task_id.append(file.rsplit(".", 1)[0])
        if done_task_id:
            logger.info(f"Existing output found. {len(done_task_id)} tasks already done.")

    n_todo_tasks = n_tasks - len(done_task_id)
    logging.info(f"Running interactive loop for {n_tasks} tasks ({n_todo_tasks} remaining).")

    with logging_redirect_tqdm():
        pbar = tqdm(total=n_todo_tasks)

        for i, task in enumerate(all_tasks):
            if args.debug and i >= args.debug_n:
                break
            if str(task.task_id) in done_task_id:
                continue

            state = interactive_loop(task, agent, env_config)

            state_list.append(state)
            json.dump(
                state.to_dict(),
                open(os.path.join(output_path, f"{task.task_id}.json"), "w"),
                indent=4,
            )
            pbar.update(1)
        pbar.close()

    logger.info("All tasks done.")
    logger.info(f"Output saved to {output_path}")

    metrics = compute_metrics(state_list, env_class=env_config["env_class"])
    metrics.update(
        {
            "method": method_label,
            "model": model_tag,
            "split": split_name,
            "max_steps": env_config.get("max_steps"),
            "temperature": agent_config["config"].get("temperature"),
            "n_tasks": len(state_list),
            "debug": bool(args.debug),
            "icl_path": env_config.get("icl_path"),
            "icl_format": env_config.get("icl_format"),
            "simplification": env_config.get("simplification"),
            "agent": exp_config.get("agent"),
            "permutation": exp_config.get("permutation"),
            "families": families,
        }
    )
    write_metrics(metrics, os.path.join(output_path, "metrics.json"))
    logger.info("\n=== Metrics ===\n" + json.dumps(metrics, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate an agentic policy on ALFWorld, ScienceWorld, or BabyAI-Text.")
    parser.add_argument("--exp_path", type=str, default="./configs/task")
    parser.add_argument("--exp_config", type=str, default="alfworld")
    parser.add_argument("--method", type=str, default="react",
                        help="Metrics label only. Output path is derived from the exp_config.")
    parser.add_argument("--split", type=str, default="test", choices=["test", "dev", "train"],
                        help="test=unseen (134 ALF / 211 Sci / 125 BabyAI), dev=seen.")
    parser.add_argument("--run", type=int, default=None,
                        help="Optional run index. Appends run<N> to the output path.")
    parser.add_argument("--part_num", type=int, default=1)
    parser.add_argument("--part_idx", type=int, default=-1)
    parser.add_argument("--agent_path", type=str, default="./configs/model")
    parser.add_argument("--agent_config", type=str, default="react_llama8b")
    parser.add_argument("--model_name", type=str, required=False)
    parser.add_argument("--max_steps", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_n", type=int, default=5)
    parser.add_argument("--override", action="store_true")
    parser.add_argument(
        "--families",
        type=str,
        default=None,
        help="BabyAI only: families to keep. Use + on Slurm (--export splits on commas), e.g. goto+pickup.",
    )
    parser.add_argument("--api_base", type=str, required=False)
    parser.add_argument("--api_key", type=str, required=False)
    parser.add_argument("--output_dir", type=str, required=False)
    parser.add_argument(
        "--placement",
        type=str,
        default=None,
        choices=["uniform", "zipf"],
        help="ALFWorld goal-object placement. Overrides env_config.placement. "
             "Default uniform. zipf writes outputs under <split>_zipf.",
    )

    args = parser.parse_args()
    logger.setLevel(logging.DEBUG if args.debug else logging.INFO)
    main(args)
