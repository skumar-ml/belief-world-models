"""Object -> receptacle affinity prior for the Stage-1 belief world model.

Derived (not hand-authored) from ALFWorld's own placement constraint,
`alfworld.gen.constants.VAL_RECEPTACLE_OBJECTS`, which maps each receptacle TYPE
to the set of object TYPES the simulator is allowed to place in it. We invert it
to object_type -> {allowed receptacle types}; this is the same constraint ALFWorld
uses to populate a scene, so a uniform prior over allowed slots is grounded in the
generative process, not guessed.

Naming: the constant keys are CamelCase (`SinkBasin`, `CounterTop`); the TextWorld
observations the agent sees use the lowercased form (`sinkbasin`, `countertop`).
`str.lower()` reproduces every text token exactly (verified against the ICL traces),
so we key the prior on the lowercased type.

See wiki/syntheses/m3-stage1-spec.md for the design contract.
"""

import json
import os
import random
from collections import defaultdict
from functools import lru_cache
from typing import Dict, Iterable, List, Optional, Set

# One shuffle of allowed receptacle types per object type, shared by every game.
# Rank 1 is index 0. Weights are k^{-ZIPF_ALPHA}. Regenerating this file must
# use ZIPF_RANK_SEED so the sampler and the belief prior stay tied together.
ZIPF_ALPHA = 1.25
ZIPF_RANK_SEED = 20260923
_RANK_PATH = os.path.join(os.path.dirname(__file__), "zipf_ranks.json")


@lru_cache(maxsize=1)
def _object_to_receptacle_types() -> Dict[str, frozenset]:
    """object_type (lowercased) -> frozenset of allowed receptacle types (lowercased).

    Inverts VAL_RECEPTACLE_OBJECTS once and caches it.
    """
    from alfworld.gen.constants import VAL_RECEPTACLE_OBJECTS

    # Invert receptacle->objects into object->receptacles, lowercasing both to match
    # the text tokens the agent sees (CamelCase keys -> 'sinkbasin' etc.).
    inv: Dict[str, Set[str]] = defaultdict(set)
    for recep_type, obj_types in VAL_RECEPTACLE_OBJECTS.items():
        rt = recep_type.lower()
        for obj_type in obj_types:
            inv[obj_type.lower()].add(rt)
    return {obj: frozenset(receps) for obj, receps in inv.items()}


def allowed_receptacle_types(object_type: str) -> frozenset:
    """Receptacle types (lowercased) an object of this type may be placed in.

    Returns an empty frozenset for unknown object types (caller should treat that
    as "no prior knowledge" — uniform over all present receptacles).
    """
    return _object_to_receptacle_types().get(object_type.lower(), frozenset())


def receptacle_type(receptacle_id: str) -> str:
    """Strip the trailing instance number: 'cabinet 4' / 'cabinet4' -> 'cabinet'."""
    return receptacle_id.rsplit(" ", 1)[0].strip() if " " in receptacle_id \
        else "".join(c for c in receptacle_id if not c.isdigit()).strip()


def uniform_prior(
    object_type: str,
    present_receptacles: List[str],
) -> Dict[str, float]:
    """Stage-1 affinity prior: uniform over allowed receptacle *instances* present.

    Normalization (option B, resolved 2026-06-25): collect every present receptacle
    instance whose TYPE is allowed for this object, then spread probability uniformly
    over that instance set. A type with more instances present therefore carries more
    total prior mass — matching the simulator's uniform-at-random placement among
    allowed slots.

    Falls back to uniform over ALL present receptacles if the object type is unknown
    or none of its allowed types are present in the scene (so the belief is never empty).

    Args:
        object_type: e.g. "spraybottle" (case-insensitive).
        present_receptacles: receptacle ids in the scene, e.g.
            ["cabinet 1", "cabinet 2", "countertop 1", "toilet 1"].

    Returns:
        {receptacle_id: probability}, summing to 1.0 over `present_receptacles`.
    """
    if not present_receptacles:
        return {}

    allowed = allowed_receptacle_types(object_type)
    candidates = [r for r in present_receptacles if receptacle_type(r) in allowed]

    # Fallback: unknown object, or none of its allowed types are in this scene.
    if not candidates:
        candidates = list(present_receptacles)

    p = 1.0 / len(candidates)
    return {r: (p if r in candidates else 0.0) for r in present_receptacles}


def build_zipf_ranks(seed: int = ZIPF_RANK_SEED) -> Dict[str, List[str]]:
    """Shuffle each object type's allowed receptacle types once.

    Object types are visited in sorted order with a single RNG, so the whole
    table is determined by `seed`. Rank 1 is the first entry of each list.
    """
    mapping = _object_to_receptacle_types()
    rng = random.Random(seed)
    ranks: Dict[str, List[str]] = {}
    for obj in sorted(mapping):
        types = sorted(mapping[obj])
        rng.shuffle(types)
        ranks[obj] = types
    return ranks


def save_zipf_ranks(path: str = _RANK_PATH, seed: int = ZIPF_RANK_SEED) -> Dict[str, List[str]]:
    ranks = build_zipf_ranks(seed)
    payload = {"alpha": ZIPF_ALPHA, "seed": seed, "ranks": ranks}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    _load_zipf_ranks.cache_clear()
    return ranks


@lru_cache(maxsize=1)
def _load_zipf_ranks() -> Dict[str, List[str]]:
    with open(_RANK_PATH) as f:
        payload = json.load(f)
    if payload.get("alpha") != ZIPF_ALPHA:
        raise ValueError(
            f"{_RANK_PATH} has alpha={payload.get('alpha')}, expected {ZIPF_ALPHA}"
        )
    return {obj: list(types) for obj, types in payload["ranks"].items()}


def zipf_rank(object_type: str) -> List[str]:
    """Allowed receptacle types for this object, best rank first. Shared by every game."""
    return list(_load_zipf_ranks().get(object_type.lower(), []))


def zipf_type_weights(
    object_type: str,
    present_types: Iterable[str],
    exclude_types: Optional[Iterable[str]] = None,
    alpha: float = ZIPF_ALPHA,
) -> Dict[str, float]:
    """Zipf mass over receptacle types that are present and not excluded.

    Types missing from the scene, and `exclude_types` (the goal receptacle
    type), are dropped and the remaining weights renormalized. An empty dict
    means this scene has no Zipf support left.
    """
    present = {t.lower() for t in present_types}
    exclude = {t.lower() for t in (exclude_types or []) if t}
    weighted = []
    for rank, recep_type in enumerate(zipf_rank(object_type), start=1):
        if recep_type in exclude or recep_type not in present:
            continue
        weighted.append((recep_type, rank ** (-alpha)))
    if not weighted:
        return {}
    total = sum(w for _, w in weighted)
    return {recep_type: w / total for recep_type, w in weighted}


def zipf_prior(
    object_type: str,
    present_receptacles: List[str],
    exclude_types: Optional[Iterable[str]] = None,
) -> Dict[str, float]:
    """Instance prior: type-level Zipf, split uniformly across instances of that type.

    Receptacles whose type gets no mass (absent, excluded, or not allowed) are
    present in the dict with probability 0. Falls back to a uniform distribution
    over the non-excluded receptacles when the object type has no Zipf ranks
    or every ranked type was dropped, so the belief is never empty.
    """
    if not present_receptacles:
        return {}

    by_type: Dict[str, List[str]] = defaultdict(list)
    for recep in present_receptacles:
        by_type[receptacle_type(recep)].append(recep)

    weights = zipf_type_weights(object_type, by_type.keys(), exclude_types)
    if not weights:
        exclude = {t.lower() for t in (exclude_types or []) if t}
        candidates = [r for r in present_receptacles if receptacle_type(r) not in exclude]
        if not candidates:
            candidates = list(present_receptacles)
        p = 1.0 / len(candidates)
        return {r: (p if r in set(candidates) else 0.0) for r in present_receptacles}

    dist = {r: 0.0 for r in present_receptacles}
    for recep_type, mass in weights.items():
        instances = by_type[recep_type]
        share = mass / len(instances)
        for inst in instances:
            dist[inst] = share
    return dist
