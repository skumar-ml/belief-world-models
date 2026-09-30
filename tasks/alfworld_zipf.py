"""Zipf placement cache for ALFWorld text games.

The released ``game.tw-pddl`` files stay where they are. A Zipf run writes a
sibling cache (``<ALFWORLD_DATA>_zipf``) with symlinks for ``traj_data.json``
and ``base_config.yaml``, and a rewritten ``game.tw-pddl`` whose goal-object
instances were resampled from the shared per-type rank table. The cache is
reused when that table is unchanged.
"""

import contextlib
import hashlib
import io
import json
import logging
import os
import random
import re
import shutil
from typing import Callable, Dict, Iterable, List, Optional, Tuple

from wm.affinity_prior import (
    ZIPF_ALPHA,
    _RANK_PATH,
    zipf_type_weights,
)


logger = logging.getLogger("agent_eval")

# Bump when the rewrite rules change so a previous fallback is not reused.
REWRITE_VERSION = 2

_SPLIT_DIR = {
    "test": "valid_unseen",
    "dev": "valid_seen",
    "train": "train",
}

_IN_RECEP = re.compile(r"\(inReceptacle\s+(\S+)\s+(\S+)\)")
_IN_RECEP_OBJ = re.compile(r"\(inReceptacleObject\s+(\S+)\s+(\S+)\)")
_OBJ_AT = re.compile(r"\(objectAtLocation\s+(\S+)\s+(\S+)\)")
_RECEP_AT = re.compile(r"\(receptacleAtLocation\s+(\S+)\s+(\S+)\)")
_OBJ_TYPE = re.compile(r"\(objectType\s+(\S+)\s+(\S+)\)")
_RECEP_TYPE = re.compile(r"\(receptacleType\s+(\S+)\s+(\S+)\)")


def zipf_output_split(split_name: str) -> str:
    """Route Zipf evals to ``<split>_zipf`` so they cannot clobber uniform runs."""
    if split_name.endswith("_zipf"):
        return split_name
    return f"{split_name}_zipf"


def cache_root_for(src_root: str) -> str:
    return os.path.abspath(src_root).rstrip("/") + "_zipf"


def rank_fingerprint() -> str:
    """Hash of the checked-in rank table. The cache is reused while this matches."""
    h = hashlib.sha256()
    with open(_RANK_PATH, "rb") as f:
        h.update(f.read())
    h.update(f"\nalpha={ZIPF_ALPHA}\nrewrite={REWRITE_VERSION}\n".encode())
    return h.hexdigest()


def pddl_type_name(type_token: str) -> str:
    """``WatchType`` / ``SinkBasinType`` -> ``watch`` / ``sinkbasin``."""
    name = type_token[:-4] if type_token.endswith("Type") else type_token
    return name.lower()


def instance_rng(game_key: str, obj_id: str) -> random.Random:
    """Stable per-instance draw. The rank table itself is not re-shuffled."""
    digest = hashlib.sha256(f"{game_key}\0{obj_id}".encode()).hexdigest()
    return random.Random(int(digest[:16], 16))


def sample_receptacle(
    rng: random.Random,
    weights: Dict[str, float],
    instances_by_type: Dict[str, List[str]],
) -> Optional[str]:
    """Sample a receptacle type from the Zipf, then a uniform instance of it."""
    if not weights:
        return None
    types = sorted(weights)
    chosen_type = rng.choices(types, weights=[weights[t] for t in types], k=1)[0]
    instances = sorted(instances_by_type[chosen_type])
    if not instances:
        return None
    return rng.choice(instances)


def _replace_binary(text: str, pred: str, arg1: str, arg2: str) -> Tuple[str, int]:
    pat = re.compile(rf"\({pred}\s+{re.escape(arg1)}\s+\S+\)")
    repl = f"({pred} {arg1} {arg2})"
    return pat.subn(repl, text, count=1)


def _insert_before_goal(text: str, fact: str) -> str:
    """Insert a fact inside `(:init ...)`, which closes just before `(:goal`."""
    needle = "(:goal"
    idx = text.find(needle)
    line = f"        ({fact})\n"
    if idx == -1:
        return text + "\n" + line
    close = text.rfind(")", 0, idx)
    if close == -1:
        return text[:idx] + line + text[idx:]
    return text[:close] + line + text[close:]


def _set_binary(text: str, pred: str, arg1: str, arg2: str) -> str:
    text, n = _replace_binary(text, pred, arg1, arg2)
    if n == 0:
        text = _insert_before_goal(text, f"{pred} {arg1} {arg2}")
    return text


def _drop_in_receptacle_object(text: str, obj_id: str) -> str:
    pat = re.compile(
        rf"^[ \t]*\(inReceptacleObject\s+{re.escape(obj_id)}\s+\S+\)[ \t]*\n",
        re.M,
    )
    return pat.sub("", text)


def rewrite_pddl_problem(
    problem: str,
    object_type: str,
    exclude_types: Optional[Iterable[str]] = None,
    game_key: str = "",
    choose: Optional[Callable[[str, Dict[str, float], Dict[str, List[str]]], Optional[str]]] = None,
) -> Optional[Tuple[str, Dict[str, str]]]:
    """Move every placed instance of ``object_type``. Other objects stay put.

    Returns ``(new_problem, {instance_id: receptacle_id})``, or None when the
    scene has no Zipf support left or no placed instance of this type.
    Each instance is drawn independently. A target that sits inside a movable
    receptacle (``inReceptacleObject``) is lifted out and placed in the sampled
    fixed receptacle. Heat/cool/clean flags are left untouched.
    """
    def _ground(pairs):
        # Goal schemata use `?o` / `?r`. Those are not placed instances.
        return [(a, b) for a, b in pairs if not a.startswith("?") and not str(b).startswith("?")]

    obj_types = dict(_ground(_OBJ_TYPE.findall(problem)))
    recep_types = {
        rid: pddl_type_name(tok) for rid, tok in _ground(_RECEP_TYPE.findall(problem))
    }
    recep_locs = dict(_ground(_RECEP_AT.findall(problem)))
    in_recep = dict(_ground(_IN_RECEP.findall(problem)))
    in_recep_obj = _ground(_IN_RECEP_OBJ.findall(problem))
    at_loc = dict(_ground(_OBJ_AT.findall(problem)))

    # Fixed receptacles are the ones with a location. Movable containers
    # (a bowl, a cup) are not placement targets.
    by_type: Dict[str, List[str]] = {}
    for rid, rtype in recep_types.items():
        if rid not in recep_locs:
            continue
        by_type.setdefault(rtype, []).append(rid)

    exclude = [t.lower() for t in (exclude_types or []) if t]
    weights = zipf_type_weights(object_type, by_type.keys(), exclude)
    if not weights:
        return None

    want = object_type.lower()
    targets = [oid for oid, tok in obj_types.items() if pddl_type_name(tok) == want]
    placed = []
    inner_of = {obj: parent for obj, parent in in_recep_obj}
    for oid in targets:
        # Some released games record the goal object with objectAtLocation
        # only. Those instances are still placed, and get an inReceptacle
        # fact when we move them.
        if oid in in_recep or oid in inner_of or oid in at_loc:
            placed.append(oid)
    if not placed:
        return None

    def _choose(obj_id: str) -> Optional[str]:
        if choose is not None:
            return choose(obj_id, weights, by_type)
        return sample_receptacle(instance_rng(game_key, obj_id), weights, by_type)

    moved: Dict[str, str] = {}
    text = problem
    for oid in placed:
        recep = _choose(oid)
        if recep is None or recep not in recep_locs:
            return None
        if oid in inner_of:
            text = _drop_in_receptacle_object(text, oid)
        text = _set_binary(text, "inReceptacle", oid, recep)
        loc = recep_locs[recep]
        text = _set_binary(text, "objectAtLocation", oid, loc)
        # Anything still inside this object moves with it.
        for child, parent in in_recep_obj:
            if parent == oid:
                text = _set_binary(text, "objectAtLocation", child, loc)
        moved[oid] = recep
    return text, moved


def _symlink(src: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    rel = os.path.relpath(src, os.path.dirname(dst))
    if os.path.islink(dst):
        if os.readlink(dst) == rel:
            return
        os.remove(dst)
    elif os.path.exists(dst):
        return
    os.symlink(rel, dst)


def _meta_fresh(dest_dir: str, fingerprint: str) -> bool:
    meta_path = os.path.join(dest_dir, ".zipf_meta.json")
    game_path = os.path.join(dest_dir, "game.tw-pddl")
    traj_path = os.path.join(dest_dir, "traj_data.json")
    if not (os.path.exists(game_path) and os.path.islink(traj_path) and os.path.exists(meta_path)):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (OSError, json.JSONDecodeError):
        return False
    return meta.get("rank_sha256") == fingerprint and meta.get("alpha") == ZIPF_ALPHA


def _write_meta(dest_dir: str, fingerprint: str, status: str, reason: str) -> None:
    with open(os.path.join(dest_dir, ".zipf_meta.json"), "w") as f:
        json.dump(
            {
                "alpha": ZIPF_ALPHA,
                "rewrite_version": REWRITE_VERSION,
                "rank_sha256": fingerprint,
                "status": status,
                "reason": reason,
            },
            f,
        )
        f.write("\n")


def _planner_env():
    from textworld.core import EnvInfos
    from textworld.envs.pddl.pddl import PddlEnv

    return PddlEnv(EnvInfos(policy_commands=True))


def assess_game(env, game: dict) -> str:
    """``ok``, ``won``, ``unsolvable``, or ``error: ...``."""
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            env.load(game)
            state = env.reset()
    except SystemExit as exc:
        # fast_downward's PDDL parser raises SystemExit on a bad problem.
        # Swallow it so one game cannot take down the cache build.
        return f"error: {exc}"
    except Exception as exc:
        return f"error: {exc}"
    if state.get("won"):
        return "won"
    if not state.get("policy_commands"):
        return "unsolvable"
    return "ok"


def _goal_from_traj(traj: dict) -> Tuple[str, str]:
    params = traj.get("pddl_params") or {}
    obj = (params.get("object_target") or "").lower()
    parent = (params.get("parent_target") or "").lower()
    task_type = traj.get("task_type") or ""
    # look_at_obj_in_light names the lamp, not a container the object must be in.
    if "look_at" in task_type:
        parent = ""
    return obj, parent


def _materialize_trial(
    src_dir: str,
    dest_dir: str,
    game_key: str,
    fingerprint: str,
    env,
    cache_root: str,
) -> str:
    os.makedirs(dest_dir, exist_ok=True)
    _symlink(
        os.path.join(src_dir, "traj_data.json"),
        os.path.join(dest_dir, "traj_data.json"),
    )
    src_game = os.path.join(src_dir, "game.tw-pddl")
    dest_game = os.path.join(dest_dir, "game.tw-pddl")
    with open(src_game) as f:
        game = json.load(f)
    with open(os.path.join(src_dir, "traj_data.json")) as f:
        traj = json.load(f)

    obj_type, parent = _goal_from_traj(traj)
    rewritten = None
    reason = ""
    if not obj_type:
        reason = "missing object_target"
    else:
        exclude = [parent] if parent else []
        rewritten = rewrite_pddl_problem(
            game["pddl_problem"], obj_type, exclude_types=exclude, game_key=game_key
        )
        if rewritten is None:
            reason = "empty candidate list"

    status = "rewritten"
    if rewritten is None:
        status = "fallback"
    else:
        new_problem, moved = rewritten
        candidate = dict(game)
        candidate["pddl_problem"] = new_problem
        candidate["solvable"] = True
        verdict = assess_game(env, candidate)
        if verdict.startswith("error"):
            # A failed translate can leave the planner unusable for the next game.
            env = _planner_env()
            verdict = assess_game(env, candidate)
        if verdict != "ok":
            status = "fallback"
            reason = verdict
            rewritten = None
        else:
            reason = f"moved {len(moved)}"
            game = candidate

    if status == "fallback":
        shutil.copy2(src_game, dest_game)
    else:
        with open(dest_game, "w") as f:
            json.dump(game, f)
    _write_meta(dest_dir, fingerprint, status, reason)
    if status == "fallback":
        logger.info("zipf fallback %s: %s", game_key, reason)
        with open(os.path.join(cache_root, "zipf_fallbacks.jsonl"), "a") as f:
            f.write(json.dumps({"game": game_key, "reason": reason}) + "\n")
    return status, env


def ensure_zipf_cache(src_root: str, split: str) -> str:
    """Build or reuse the Zipf cache for ``split`` and return its root.

    ``split`` is the eval name (``test`` / ``dev`` / ``train``), matched to
    ``valid_unseen`` / ``valid_seen`` / ``train`` on disk.
    """
    if split not in _SPLIT_DIR:
        raise ValueError(f"Unknown ALFWorld split {split!r}")
    src_root = os.path.abspath(src_root)
    cache = cache_root_for(src_root)
    fingerprint = rank_fingerprint()
    os.makedirs(cache, exist_ok=True)
    base = os.path.join(src_root, "base_config.yaml")
    if os.path.exists(base):
        _symlink(base, os.path.join(cache, "base_config.yaml"))
    logic = os.path.join(src_root, "logic")
    if os.path.isdir(logic):
        _symlink(logic, os.path.join(cache, "logic"))

    split_dir = _SPLIT_DIR[split]
    src_split = os.path.join(src_root, "json_2.1.1", split_dir)
    dst_split = os.path.join(cache, "json_2.1.1", split_dir)
    trials = []
    for root, _dirs, files in os.walk(src_split):
        # Same skips as AlfredTWEnv.collect_game_files. Sliced games can make
        # the planner run for many minutes, and movable games use a predicate
        # the domain does not declare.
        if "movable" in root or "Sliced" in root:
            continue
        if "game.tw-pddl" in files and "traj_data.json" in files:
            game_path = os.path.join(root, "game.tw-pddl")
            try:
                with open(game_path) as f:
                    solvable = json.load(f).get("solvable")
            except (OSError, json.JSONDecodeError):
                continue
            if not solvable:
                continue
            trials.append(root)
    trials.sort()

    pending = []
    for trial in trials:
        rel = os.path.relpath(trial, src_split)
        dest = os.path.join(dst_split, rel)
        if _meta_fresh(dest, fingerprint):
            continue
        pending.append((trial, dest, rel))

    if not pending:
        logger.info("zipf cache up to date (%d games) at %s", len(trials), cache)
        return cache

    logger.info(
        "building zipf cache: %d new or stale games (%d already current) under %s",
        len(pending), len(trials) - len(pending), dst_split,
    )
    env = _planner_env()
    n_rewritten = 0
    n_fallback = 0
    for i, (trial, dest, rel) in enumerate(pending, start=1):
        status, env = _materialize_trial(trial, dest, rel, fingerprint, env, cache)
        if status == "rewritten":
            n_rewritten += 1
        else:
            n_fallback += 1
        if i % 20 == 0 or i == len(pending):
            logger.info("zipf cache progress %d/%d", i, len(pending))
    logger.info(
        "zipf cache ready at %s: %d rewritten, %d kept original",
        cache, n_rewritten, n_fallback,
    )
    return cache
