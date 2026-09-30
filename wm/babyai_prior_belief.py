"""Placement-prior belief world model for BabyAI-Text.

This is the BabyAI world model. Once wall rays pin both axes of the known 6x6
interior, each unlocated referent gets a uniform categorical over legal cells
(generation: not on the start pose, not Manhattan d<2 from start). FoV empties,
walls, the agent's cell, and other objects zero mass and renormalize. Collapse
on first matching sighting or pickup.

Until the box is localized, residual mass stays on UNSEEN.
"""

from typing import Dict, List, Optional, Set, Tuple

from wm.babyai_belief import BabyAIBeliefState
from wm.babyai_goal import BabyAIGoal
from wm.babyai_room import (
    RoomBox,
    ascii_room,
    start_forbidden,
    visible_world_cells,
)
from wm.babyai_scene import (
    INVENTORY,
    UNSEEN,
    ParsedObs,
    cell_key,
    parse_cell_key,
    referent_key,
)


class BabyAIPriorBeliefState(BabyAIBeliefState):
    """Memory map + 6x6 placement prior once the room box is pinned."""

    def __init__(self, goal: BabyAIGoal, track_belief: bool = True):
        super().__init__(goal, track_belief=track_belief)
        self.room = RoomBox()
        self._start_forbidden = start_forbidden((0, 0))
        self._prior_seeded: Set[str] = set()

    @property
    def room_localized(self) -> bool:
        return self.room.localized()

    def _after_fold(self, parsed: ParsedObs) -> None:
        # Pin axes from this FoV's wall rays, then fill the ring and vis-empties.
        for s in parsed.sightings:
            if s.is_wall_ray:
                self.room.pin_from_ray((self.x, self.y), self.rel_to_abs(s.forward, s.right))
        if not self.room.localized():
            return
        for pos in self.room.wall_cells():
            self._write_cell(pos, kind="wall", typ="wall", overwrite=False)
        # Visible interior cells with no listed object are empty (objects do not occlude).
        occupied = {
            pos
            for pos, fact in self.cells.items()
            if fact.kind in ("object", "door")
        }
        vis = visible_world_cells(self.x, self.y, self.heading, self.room.wall_cells())
        for pos in vis:
            if pos == (self.x, self.y) or pos in occupied:
                continue
            if self.room.contains_interior(pos):
                self._write_empty(pos)

    def _update_beliefs(self, parsed: ParsedObs) -> None:
        if not self.track_belief:
            return
        for key, belief in self.beliefs.items():
            if self.inventory and referent_key(*self.inventory) == key:
                belief.collapse(INVENTORY)
                self._prior_seeded.add(key)
                continue
            if belief.located_at == INVENTORY and self.inventory is None:
                hit = self._visible_match(parsed, key)
                if hit is not None:
                    belief.collapse(cell_key(*hit))
                elif key in self.objects and self.objects[key]:
                    belief.collapse(self.objects[key][-1])
                continue
            if belief.located_at is not None:
                continue
            hit = self._visible_match(parsed, key)
            if hit is not None:
                belief.collapse(cell_key(*hit))
                self._prior_seeded.add(key)
                continue
            if self.room.localized() and key not in self._prior_seeded:
                cands = [cell_key(*c) for c in self._candidate_cells(key)]
                if cands:
                    belief.seed_uniform(cands)
                    self._prior_seeded.add(key)
            if not self.room.localized():
                continue
            for atom in self._eliminated_atoms(key):
                belief.eliminate(atom)

    def _candidate_cells(self, key: str) -> List[Tuple[int, int]]:
        """Interior cells still allowed for an unlocated referent."""
        banned = self._eliminated_positions(key)
        return [c for c in self.room.interior_cells() if c not in banned]

    def _eliminated_positions(self, key: str) -> Set[Tuple[int, int]]:
        banned = set(self._start_forbidden)
        banned.add((self.x, self.y))
        for pos, fact in self.cells.items():
            if fact.kind == "empty":
                banned.add(pos)
            elif fact.kind == "wall":
                banned.add(pos)
            elif fact.kind in ("object", "door"):
                if referent_key(fact.color, fact.type or "") != key:
                    banned.add(pos)
        return banned

    def _eliminated_atoms(self, key: str) -> List[str]:
        return [cell_key(*p) for p in self._eliminated_positions(key)]

    def remaining_ranking(self, query_key: str) -> List[Tuple[str, float]]:
        """Non-zero cells, nearest first (mass is uniform so distance is the rank)."""
        ranking = self.where_is(query_key)
        scored = []
        for atom, p in ranking:
            if p <= 0 or atom in (UNSEEN, INVENTORY):
                continue
            xy = parse_cell_key(atom)
            if xy is None:
                continue
            fwd, right = self.abs_to_rel(xy)
            scored.append((atom, p, abs(fwd) + abs(right), fwd, right))
        scored.sort(key=lambda t: (t[2], t[3], t[4], t[0]))
        return [(atom, p) for atom, p, *_ in scored]

    def quadrant_mass(self, query_key: str) -> Dict[str, float]:
        """Mass of remaining cells by egocentric half-plane."""
        buckets = {k: 0.0 for k in ("front", "back", "left", "right", "here")}
        for atom, p in self.where_is(query_key):
            if p <= 0 or atom in (UNSEEN, INVENTORY):
                continue
            xy = parse_cell_key(atom)
            if xy is None:
                continue
            fwd, right = self.abs_to_rel(xy)
            if fwd == 0 and right == 0:
                buckets["here"] += p
            elif abs(fwd) >= abs(right):
                buckets["front" if fwd > 0 else "back"] += p
            else:
                buckets["right" if right > 0 else "left"] += p
        return buckets

    def candidate_atoms(self, query_key: Optional[str] = None) -> Set[str]:
        """Cells with residual mass, unioned across referents if `query_key` is None."""
        keys = [query_key.strip().lower()] if query_key else list(self.beliefs)
        atoms = set()
        for key in keys:
            if not key:
                continue
            for atom, p in self.where_is(key):
                if p > 0 and parse_cell_key(atom) is not None:
                    atoms.add(atom)
        return atoms

    def cell_mass(self, pos: Tuple[int, int]) -> List[Tuple[str, float]]:
        """Unlocated referents that still put mass on this cell."""
        atom = cell_key(*pos)
        out = []
        for key, belief in self.beliefs.items():
            if belief.located_at is not None:
                continue
            p = belief.dist.get(atom, 0.0)
            if p > 0:
                out.append((key, p))
        return out

    def ascii_map(self, query_key: Optional[str] = None) -> str:
        return ascii_room(
            self.room,
            (self.x, self.y),
            self.heading,
            self.cells,
            self.candidate_atoms(query_key),
        )
