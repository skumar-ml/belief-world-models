"""Start-frame belief world model for BabyAI-Text.

Factored abstract state parsed from (action, observation) text only — no MiniGrid
oracle, no pre-filled 6x6. Pose lives in a start-pose frame: reset is (0, 0)
facing +Y. Cells are written only when a description or odometry admits them.

Categorical belief is residual: P(referent @ UNSEEN / OTHER_ROOM / seen cell).
Collapse on first sighting; after that last-known location is deterministic.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

from wm.babyai_goal import BabyAIGoal, Referent
from wm.babyai_scene import (
    INVENTORY,
    OTHER_ROOM,
    UNSEEN,
    ParsedObs,
    Sighting,
    cell_key,
    is_uninformative,
    parse_observation,
    referent_key,
)


# Heading 0 = +Y (start facing), then clockwise: +X, -Y, -X.
_HEADING_VEC = ((0, 1), (1, 0), (0, -1), (-1, 0))
_HEADING_NAME = ("start-forward", "start-right", "start-back", "start-left")


def _right_vec(heading: int) -> Tuple[int, int]:
    dx, dy = _HEADING_VEC[heading]
    return dy, -dx


def _rel_to_abs(x: int, y: int, heading: int, forward: int, right: int) -> Tuple[int, int]:
    hx, hy = _HEADING_VEC[heading]
    rx, ry = _right_vec(heading)
    return x + forward * hx + right * rx, y + forward * hy + right * ry


@dataclass
class CellFact:
    """A cell the text or odometry has forced us to admit exists."""

    kind: str                          # empty | wall | object | door
    color: Optional[str] = None
    type: Optional[str] = None
    door_state: Optional[str] = None
    last_seen_step: int = 0
    visited: bool = False


@dataclass
class Progress:
    """Mission bits that can be read off inventory / door wording."""

    carrying_target: bool = False
    first_seq_done: bool = False
    door_open: bool = False


class ObjectBelief:
    """Categorical P(referent @ atom) over residual buckets plus seen cells."""

    def __init__(self, referent: str):
        self.referent = referent
        self.located_at: Optional[str] = None
        self.dist: Dict[str, float] = {UNSEEN: 1.0}

    def collapse(self, atom: str) -> None:
        self.located_at = atom
        keys = set(self.dist) | {atom}
        self.dist = {k: (1.0 if k == atom else 0.0) for k in keys}

    def seed_uniform(self, atoms: List[str]) -> None:
        """Replace residual UNSEEN with a uniform categorical over `atoms`."""
        if self.located_at is not None:
            return
        uniq = list(dict.fromkeys(atoms))
        if not uniq:
            return
        p = 1.0 / len(uniq)
        self.dist = {a: p for a in uniq}

    def eliminate(self, atom: str) -> None:
        if self.located_at is not None or atom not in self.dist:
            return
        self.dist[atom] = 0.0
        total = sum(self.dist.values())
        if total > 0:
            self.dist = {k: p / total for k, p in self.dist.items()}

    def split_other_room(self) -> None:
        """Once a door is seen, peel half of UNSEEN onto OTHER_ROOM."""
        if self.located_at is not None or OTHER_ROOM in self.dist:
            return
        u = self.dist.get(UNSEEN, 0.0)
        if u <= 0:
            return
        self.dist[UNSEEN] = u / 2.0
        self.dist[OTHER_ROOM] = u / 2.0

    def ranking(self, exclude: Set[str]) -> List[Tuple[str, float]]:
        cand = [(a, p) for a, p in self.dist.items() if p > 0.0 and a not in exclude]
        return sorted(cand, key=lambda x: (-x[1], x[0]))


class BabyAIBeliefState:
    """Deterministic seen-map + residual location beliefs for one episode."""

    def __init__(self, goal: BabyAIGoal, track_belief: bool = True):
        self.goal = goal
        self.track_belief = track_belief
        self.x, self.y, self.heading = 0, 0, 0
        self.inventory: Optional[Tuple[str, str]] = None
        self.cells: Dict[Tuple[int, int], CellFact] = {}
        # referent -> last-known atoms (usually one; two if duplicate color+type).
        self.objects: Dict[str, List[str]] = {}
        self.progress = Progress()
        self.step = 0
        self._last_obs: Optional[str] = None
        self.beliefs: Dict[str, ObjectBelief] = {}
        if track_belief:
            for ref in goal.referents:
                self.beliefs.setdefault(ref.key, ObjectBelief(ref.key))
        self._mark_visited(0, 0)

    # ----------------------------------------------------------------- pose
    def front_cell(self) -> Tuple[int, int]:
        hx, hy = _HEADING_VEC[self.heading]
        return self.x + hx, self.y + hy

    def heading_name(self) -> str:
        return _HEADING_NAME[self.heading]

    def rel_to_abs(self, forward: int, right: int) -> Tuple[int, int]:
        return _rel_to_abs(self.x, self.y, self.heading, forward, right)

    def abs_to_rel(self, cell: Tuple[int, int]) -> Tuple[int, int]:
        """Return (forward, right) of `cell` from the current pose."""
        dx, dy = cell[0] - self.x, cell[1] - self.y
        hx, hy = _HEADING_VEC[self.heading]
        rx, ry = _right_vec(self.heading)
        return dx * hx + dy * hy, dx * rx + dy * ry

    def _front_blocked(self) -> bool:
        cell = self.cells.get(self.front_cell())
        if cell is None:
            return False
        if cell.kind in ("wall", "object"):
            return True
        return cell.kind == "door" and cell.door_state != "open"

    def _apply_action(self, action: str, new_obs: str) -> None:
        """Turn always; forward only if the new view or map says we moved."""
        if action == "turn left":
            self.heading = (self.heading - 1) % 4
            return
        if action == "turn right":
            self.heading = (self.heading + 1) % 4
            return
        if action != "go forward":
            return
        unchanged = self._last_obs is not None and new_obs == self._last_obs
        if self._front_blocked() or unchanged:
            return
        hx, hy = _HEADING_VEC[self.heading]
        self.x += hx
        self.y += hy
        self._mark_visited(self.x, self.y)

    # ----------------------------------------------------------------- cells
    def _mark_visited(self, x: int, y: int) -> None:
        fact = self.cells.get((x, y))
        if fact is None:
            self.cells[(x, y)] = CellFact(kind="empty", last_seen_step=self.step, visited=True)
            return
        fact.visited = True
        fact.last_seen_step = self.step

    def _write_cell(
        self,
        pos: Tuple[int, int],
        kind: str,
        color: Optional[str] = None,
        typ: Optional[str] = None,
        door_state: Optional[str] = None,
        overwrite: bool = True,
    ) -> None:
        cur = self.cells.get(pos)
        if cur is not None and not overwrite and cur.kind != "empty":
            cur.last_seen_step = self.step
            return
        visited = bool(cur.visited) if cur is not None else (pos == (self.x, self.y))
        self.cells[pos] = CellFact(
            kind=kind,
            color=color,
            type=typ,
            door_state=door_state,
            last_seen_step=self.step,
            visited=visited,
        )

    def _write_empty(self, pos: Tuple[int, int]) -> None:
        cur = self.cells.get(pos)
        if cur is not None and cur.kind != "empty":
            cur.last_seen_step = self.step
            return
        self._write_cell(pos, kind="empty", overwrite=False)

    def _record_object(self, key: str, atom: str) -> None:
        # Inventory wipes prior cells; a new cell drops inventory; keep at most two.
        if atom == INVENTORY:
            self.objects[key] = [INVENTORY]
            return
        prev = [a for a in self.objects.get(key, []) if a != INVENTORY]
        if atom not in prev:
            prev.append(atom)
        self.objects[key] = prev[-2:]

    def _referent_matches(self, color: Optional[str], typ: Optional[str], ref: Referent) -> bool:
        if typ != ref.type:
            return False
        return ref.color is None or color == ref.color

    # ----------------------------------------------------------------- observe
    def observe(self, action: str, observation: str) -> None:
        """Fold one (action, observation) pair into the map + residual beliefs."""
        action = (action or "").strip().lower()
        obs = (observation or "").strip()
        if obs.lower().startswith("observation:"):
            obs = obs.split(":", 1)[1].strip()
        if is_uninformative(obs):
            return

        self.step += 1
        prev_inv = self.inventory
        self._apply_action(action, obs)

        parsed = parse_observation(obs)
        self._fold_fov(parsed)
        self._fold_inventory(action, parsed, prev_inv)
        self._after_fold(parsed)
        self._update_beliefs(parsed)
        self._update_progress()
        self._last_obs = obs

    def _after_fold(self, parsed: ParsedObs) -> None:
        """Hook for subclasses (room localization). Memory WM is a no-op."""
        return

    def _fold_fov(self, parsed: ParsedObs) -> None:
        """Write wall rays (empties + wall) and visible objects/doors."""
        for s in parsed.sightings:
            if s.is_wall_ray:
                self._fold_wall_ray(s)
                continue
            pos = self.rel_to_abs(s.forward, s.right)
            self._write_cell(
                pos,
                kind=s.kind,
                color=s.color,
                typ=s.type,
                door_state=s.door_state,
            )
            if s.kind in ("object", "door") and s.type:
                self._record_object(referent_key(s.color, s.type), cell_key(*pos))

    def _fold_wall_ray(self, s: Sighting) -> None:
        """Wall at distance N; cells 1..N-1 on that axis are empty."""
        dist = s.forward if s.forward else abs(s.right)
        if dist <= 0:
            return
        sign_f = 1 if s.forward > 0 else (-1 if s.forward < 0 else 0)
        sign_r = 1 if s.right > 0 else (-1 if s.right < 0 else 0)
        for i in range(1, dist):
            self._write_empty(self.rel_to_abs(sign_f * i, sign_r * i))
        wall_pos = self.rel_to_abs(s.forward, s.right)
        self._write_cell(wall_pos, kind="wall", typ="wall")

    def _fold_inventory(
        self, action: str, parsed: ParsedObs, prev_inv: Optional[Tuple[str, str]]
    ) -> None:
        """Sync carry from the observation; apply pick/drop to the front cell."""
        self.inventory = parsed.carrying
        front = self.front_cell()
        if action == "pick up" and self.inventory is not None:
            color, typ = self.inventory
            key = referent_key(color, typ)
            self._record_object(key, INVENTORY)
            cur = self.cells.get(front)
            if cur is not None and cur.kind == "object" and cur.type == typ:
                self._write_cell(front, kind="empty")
        elif action == "drop" and prev_inv is not None and self.inventory is None:
            color, typ = prev_inv
            key = referent_key(color, typ)
            self._write_cell(front, kind="object", color=color, typ=typ)
            self._record_object(key, cell_key(*front))
            locs = self.objects.get(key, [])
            self.objects[key] = [a for a in locs if a != INVENTORY] or [cell_key(*front)]

    def _update_beliefs(self, parsed: ParsedObs) -> None:
        if not self.track_belief:
            return
        door_seen = any(s.kind == "door" for s in parsed.sightings)
        empties = {cell_key(*pos) for pos, fact in self.cells.items() if fact.kind == "empty"}

        for key, belief in self.beliefs.items():
            if self.inventory and referent_key(*self.inventory) == key:
                belief.collapse(INVENTORY)
                continue
            if belief.located_at == INVENTORY and self.inventory is None:
                # Dropped: re-bind from this FoV or the last recorded cell.
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
                continue
            if door_seen:
                belief.split_other_room()
            for atom in empties:
                belief.eliminate(atom)
            # Occupied by a different object: not a candidate.
            for pos, fact in self.cells.items():
                if fact.kind not in ("object", "door", "wall"):
                    continue
                if fact.kind in ("object", "door") and referent_key(fact.color, fact.type or "") == key:
                    continue
                belief.eliminate(cell_key(*pos))

    def _visible_match(self, parsed: ParsedObs, key: str) -> Optional[Tuple[int, int]]:
        for s in parsed.sightings:
            if s.kind not in ("object", "door") or not s.type:
                continue
            if referent_key(s.color, s.type) == key:
                return self.rel_to_abs(s.forward, s.right)
            # Uncolored mission door matches any seen door.
            if key == "door" and s.kind == "door":
                return self.rel_to_abs(s.forward, s.right)
        return None

    def _update_progress(self) -> None:
        refs = self.goal.referents
        if not refs:
            return
        if self.inventory is not None:
            inv_key = referent_key(*self.inventory)
            self.progress.carrying_target = any(
                inv_key == r.key or (r.color is None and self.inventory[1] == r.type)
                for r in refs
            )
            if self.goal.act == "seq":
                pickup = refs[0]
                if inv_key == pickup.key or (
                    pickup.color is None and self.inventory[1] == pickup.type
                ):
                    self.progress.first_seq_done = True
        else:
            self.progress.carrying_target = False
        if self.goal.act == "open":
            target = refs[0]
            self.progress.door_open = any(
                f.kind == "door"
                and f.door_state == "open"
                and self._referent_matches(f.color, f.type, target)
                for f in self.cells.values()
            )

    # ----------------------------------------------------------------- queries
    def where_is(self, query_key: str) -> List[Tuple[str, float]]:
        """Ranked atoms for a referent key ('green key', 'door', …)."""
        key = query_key.strip().lower()
        if self.inventory and self._inv_matches(key):
            return [(INVENTORY, 1.0)]
        if not self.track_belief:
            return [(a, 1.0) for a in self.objects.get(key, [])]
        belief = self.beliefs.get(key)
        if belief is None:
            locs = self.objects.get(key, [])
            if locs:
                return [(a, 1.0) for a in locs]
            return [(UNSEEN, 1.0)]
        if belief.located_at is not None:
            return [(belief.located_at, 1.0)]
        return belief.ranking(set())

    def _inv_matches(self, key: str) -> bool:
        if self.inventory is None:
            return False
        inv = referent_key(*self.inventory)
        return inv == key or self.inventory[1] == key or key in inv

    def locate_referent(self, key: str) -> Tuple[str, Optional[str]]:
        """('inventory'|'located'|'unknown', atom)."""
        if self.inventory and self._inv_matches(key):
            return ("inventory", INVENTORY)
        locs = self.objects.get(key, [])
        if locs:
            return ("located", locs[0])
        belief = self.beliefs.get(key)
        if belief is not None and belief.located_at is not None:
            return ("located", belief.located_at)
        return ("unknown", None)

    def contents_of(self, pos: Tuple[int, int]) -> Optional[CellFact]:
        return self.cells.get(pos)

    def is_searched(self, pos: Tuple[int, int]) -> bool:
        return pos in self.cells

    def last_seen_objects(self) -> List[Tuple[str, str]]:
        """(referent, atom) pairs in first-seen order."""
        out = []
        for key, atoms in self.objects.items():
            for atom in atoms:
                out.append((key, atom))
        return out
