"""Room-level belief world model for SciWorld.

Maintains an abstract state parsed from text observations plus a categorical belief
P(target @ room) over the fixed 10-room set, seeded from room priors and updated by
presence/absence renormalization as the agent explores.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from wm import sciworld_scene as scene
from wm.sciworld_goal import SciWorldGoal
from wm.sciworld_room_prior import candidate_rooms, uniform_room_prior


@dataclass
class Room:
    name: str
    searched: bool = False
    contents: Set[str] = field(default_factory=set)


class RoomBelief:
    """Categorical P(target @ room) over the fixed room set."""

    _STATED_ROOM_WEIGHT = 0.9

    def __init__(self, target_type: str, rooms: List[str],
                 forced_room: Optional[str] = None,
                 stated_room: Optional[str] = None):
        self.target_type = target_type
        self.located_at: Optional[str] = None
        if forced_room and forced_room in rooms:
            self.dist: Dict[str, float] = {r: (1.0 if r == forced_room else 0.0)
                                           for r in rooms}
        elif stated_room and stated_room in rooms:
            base = uniform_room_prior(target_type, rooms)
            w = self._STATED_ROOM_WEIGHT
            self.dist = {}
            residual_total = sum(p for r, p in base.items() if r != stated_room) or 1.0
            for r in rooms:
                if r == stated_room:
                    self.dist[r] = w
                else:
                    self.dist[r] = (1.0 - w) * (base.get(r, 0.0) / residual_total)
        else:
            self.dist = uniform_room_prior(target_type, rooms)

    def collapse(self, room: str) -> None:
        self.located_at = room
        self.dist = {r: (1.0 if r == room else 0.0) for r in self.dist}

    def eliminate(self, room: str) -> None:
        if self.located_at is not None or room not in self.dist:
            return
        self.dist[room] = 0.0
        total = sum(self.dist.values())
        if total > 0:
            self.dist = {r: p / total for r, p in self.dist.items()}

    def ranking(self, exclude_searched: Set[str]) -> List[Tuple[str, float]]:
        cand = [(r, p) for r, p in self.dist.items()
                if p > 0.0 and r not in exclude_searched]
        return sorted(cand, key=lambda x: (-x[1], x[0]))


class SciWorldBeliefState:
    """The location world model for one SciWorld episode."""

    def __init__(self, goal: SciWorldGoal, rooms: Optional[List[str]] = None,
                 stated_locations: Optional[Dict[str, str]] = None,
                 track_belief: bool = True):
        self.goal = goal
        # Ablation switch: when False, drop room priors / stated-location mass and
        # answer where-is from observed facts only (inventory, seen rooms/containers).
        self.track_belief = track_belief
        self.rooms: Dict[str, Room] = {r: Room(name=r) for r in (rooms or scene.ROOMS)}
        self.stated_locations: Dict[str, str] = {
            k.strip().lower(): v.strip().lower()
            for k, v in (stated_locations or {}).items()
        }
        self.agent_location: Optional[str] = None
        self.inventory: Set[str] = set()
        self.focus: Optional[str] = None
        self.containers: Dict[str, Set[str]] = {}
        self.object_locations: Dict[str, Tuple[str, str]] = {}
        self.held_contents: Dict[str, Optional[Set[str]]] = {}
        self._last_dump_inline: Dict[str, Set[str]] = {}
        self._seen_containers: Set[str] = set()
        self.beliefs: Dict[str, RoomBelief] = {}
        if track_belief:
            self.beliefs[goal.target_type] = RoomBelief(
                goal.target_type, list(self.rooms), forced_room=goal.named_location
            )
            for obj, room in self.stated_locations.items():
                if obj == goal.target_type.strip().lower():
                    continue
                self.beliefs[obj] = RoomBelief(obj, list(self.rooms), stated_room=room)

    @property
    def target_belief(self) -> RoomBelief:
        return self.beliefs[self.goal.target_type]

    @staticmethod
    def _referent_matches(referent: str, obj_type: str) -> bool:
        referent = referent.strip()
        obj_type = obj_type.strip()
        if referent == obj_type:
            return True
        toks = obj_type.split()
        head = toks[-1] if toks else obj_type
        if len(head) <= 3:
            return obj_type in referent or referent in obj_type
        return head in referent.split()

    def _get_belief(self, obj_type: str) -> RoomBelief:
        key = obj_type.strip().lower()
        b = self.beliefs.get(key)
        if b is not None:
            return b
        b = RoomBelief(key, list(self.rooms))
        for rname, rm in self.rooms.items():
            if not rm.searched:
                continue
            if any(self._referent_matches(o, key) for o in rm.contents):
                b.collapse(rname)
                break
            b.eliminate(rname)
        self.beliefs[key] = b
        return b

    def observe(self, action: str, observation: str) -> None:
        obs = (observation or "").strip()
        if scene.is_noop(obs):
            return

        moved = scene.parse_agent_move(action, obs)
        if moved:
            self.agent_location = moved

        picked = scene.parse_pickup(obs)
        if picked:
            self.inventory.add(picked)
            if self._is_container(picked):
                seed = self._last_dump_inline.get(picked)
                self.held_contents[picked] = set(seed) if seed is not None else None
            self.object_locations.pop(picked, None)
            for b in self.beliefs.values():
                if self._referent_matches(picked, b.target_type):
                    b.located_at = "inventory"

        dropped = scene.parse_putdown(obs)
        if dropped:
            self.inventory.discard(dropped)
            self.held_contents.pop(dropped, None)

        pc = scene.parse_pour_contents(obs)
        if pc:
            src, dest = pc
            if src in self.inventory:
                self.held_contents[src] = set()
            if dest in self.inventory:
                self.held_contents[dest] = None

        poured = scene.parse_pour_named(obs)
        if poured:
            subst, dest = poured
            if dest in self.inventory:
                cur = self.held_contents.get(dest)
                self.held_contents[dest] = ({subst} if cur is None else cur | {subst})
            for src, contents in self.held_contents.items():
                if src != dest and contents and subst in contents:
                    contents.discard(subst)

        focused = scene.parse_focus(obs)
        if focused:
            self.focus = focused

        if scene.is_room_dump(obs):
            room = scene.parse_room_name(obs)
            if room:
                self.agent_location = room
                objs = scene.objects_in_dump(obs)
                self.rooms[room].searched = True
                self.rooms[room].contents = set(objs)
                for cname, items in scene.container_contents(obs):
                    self.containers[cname] = set(items)
                    self._seen_containers.add(cname)
                    for item in items:
                        self.object_locations[item] = (room, cname)
                self._last_dump_inline = scene.inline_contents(obs)
                self._seen_containers.update(self._last_dump_inline.keys())
                for cname, items in self._last_dump_inline.items():
                    for item in items:
                        self.object_locations[item] = (room, cname)

                for b in self.beliefs.values():
                    if any(self._referent_matches(h, b.target_type) for h in self.inventory):
                        b.located_at = "inventory"
                        continue
                    if any(self._referent_matches(o, b.target_type) for o in objs):
                        b.collapse(room)
                    else:
                        b.eliminate(room)

        li = scene.parse_look_inside(obs)
        if li and self.agent_location:
            container, items = li
            room = self.agent_location
            self.containers[container] = set(items)
            self._seen_containers.add(container)
            for item in items:
                self.object_locations[item] = (room, container)
            tb = self.beliefs.get(self.goal.target_type)
            if tb is not None and tb.located_at is None:
                if any(self._referent_matches(o, self.goal.target_type) for o in items):
                    tb.collapse(room)
                elif self.rooms[room].searched:
                    tb.eliminate(room)

    def _searched_rooms(self) -> Set[str]:
        return {r for r, rm in self.rooms.items() if rm.searched}

    def observed_locations(self, object_type: str) -> List[str]:
        """Rooms where this type has actually been seen (not inventory, not prior)."""
        key = object_type.strip().lower()
        seen: List[str] = []
        loc = self.container_of(key)
        if loc:
            seen.append(loc[0])
        for rname, rm in self.rooms.items():
            if not rm.searched:
                continue
            if any(self._referent_matches(o, key) for o in rm.contents):
                seen.append(rname)
        return list(dict.fromkeys(seen))

    def where_is(self, object_type: str) -> List[Tuple[str, float]]:
        key = object_type.strip().lower()
        if any(self._referent_matches(held, key) for held in self.inventory):
            return [("inventory", 1.0)]
        if not self.track_belief:
            return [(r, 1.0) for r in self.observed_locations(object_type)]
        b = self._get_belief(object_type)
        if b.located_at is not None:
            return [(b.located_at, 1.0)]
        return b.ranking(self._searched_rooms())

    def is_room(self, name: str) -> bool:
        return name.strip().lower() in self.rooms

    def has_information(self, object_type: str) -> bool:
        key = object_type.strip().lower()
        if not self.track_belief:
            if any(self._referent_matches(held, key) for held in self.inventory):
                return True
            return bool(self.observed_locations(key))
        if any(self._referent_matches(stated, key) for stated in self.stated_locations):
            return True
        if any(self._referent_matches(held, key) for held in self.inventory):
            return True
        if self.container_of(key) is not None:
            return True
        if any(self._referent_matches(ref, key) for ref in self.object_locations):
            return True
        for rm in self.rooms.values():
            if rm.searched and any(self._referent_matches(o, key) for o in rm.contents):
                return True
        return bool(candidate_rooms(key))

    def _is_container(self, referent: str) -> bool:
        return referent in self._seen_containers

    def contents_of(self, room_or_container: str) -> Optional[Set[str]]:
        key = room_or_container.strip().lower()
        rm = self.rooms.get(key)
        if rm is not None:
            return set(rm.contents) if rm.searched else None
        if key in self.containers:
            return set(self.containers[key])
        return None

    def container_of(self, object_type: str) -> Optional[Tuple[str, str]]:
        key = object_type.strip().lower()
        if any(self._referent_matches(held, key) for held in self.inventory):
            return None
        for ref, loc in self.object_locations.items():
            if self._referent_matches(ref, key):
                return loc
        return None

    def is_searched(self, room: str) -> bool:
        rm = self.rooms.get(room.strip().lower())
        return bool(rm and rm.searched)
