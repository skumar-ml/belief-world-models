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

from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Set


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
