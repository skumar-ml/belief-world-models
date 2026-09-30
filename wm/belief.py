"""Stage-1 belief world model for ALFWorld (standard, full-layout setting).

Maintains a factored, hand-specified abstract state parsed purely from the text
observations the policy already sees (no PDDL/oracle), plus a calibrated categorical
belief P(object instance @ receptacle) over the known receptacle set, seeded by the
affinity prior and updated by presence/absence renormalization.

Design contract: wiki/syntheses/m3-stage1-spec.md.

The module is deliberately env-agnostic and side-effect free: it consumes (action,
observation) text pairs via `observe(...)` and answers questions via `query(...)`.
The wrapper that splices it into the eval loop lives elsewhere.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from wm.affinity_prior import receptacle_type, uniform_prior, zipf_prior


# --------------------------------------------------------------------------- #
# Observation templates (standard AlfWorldEnv / TextWorld).
# Confirmed against prompt/icl_examples/alfworld_reflact_icl.json traces.
# --------------------------------------------------------------------------- #
# "On the cabinet 1, you see a cloth 1, a soapbar 1, a soapbottle 1."
# "On the countertop 2, you see nothing."
# "The cabinet 2 is closed."
# "You open the cabinet 2. The cabinet 2 is open. In it, you see a candle 1, and a spraybottle 2."
# "You pick up the spraybottle 2 from the cabinet 2."
# "You put the spraybottle 2 in/on the toilet 1."
# "You clean the lettuce 1 using the sinkbasin 1."
# "You heat the egg 2 using the microwave 1."
# "You cool the pan 1 using the fridge 1."
# Actions: "go to <recep>", "open <recep>", "take <obj> from <recep>",
#          "put <obj> in/on <recep>", "clean/heat/cool <obj> with <recep>", "use <obj>".

_RE_ON_SEE = re.compile(r"^(?:On|In) the (.+?), you see (.+?)\.?$", re.IGNORECASE)
_RE_OPEN_REVEAL = re.compile(r"In it, you see (.+?)\.?$", re.IGNORECASE)
_RE_CLOSED = re.compile(r"^The (.+?) is closed\.?$", re.IGNORECASE)
_RE_OPENED = re.compile(r"^You open the (.+?)\. ", re.IGNORECASE)
_RE_PICK = re.compile(r"^You pick up the (.+?) from the (.+?)\.?$", re.IGNORECASE)
_RE_PUT = re.compile(r"^You put the (.+?) in/on the (.+?)\.?$", re.IGNORECASE)
_RE_CLEAN = re.compile(r"^You clean the (.+?) using the (.+?)\.?$", re.IGNORECASE)
_RE_HEAT = re.compile(r"^You heat the (.+?) using the (.+?)\.?$", re.IGNORECASE)
_RE_COOL = re.compile(r"^You cool the (.+?) using the (.+?)\.?$", re.IGNORECASE)

# An object id is "<type> <num>" (e.g. "spraybottle 2"). Receptacles share this form.
_RE_ENTITY = re.compile(r"\b([a-z]+) (\d+)\b")


def _object_type(entity_id: str) -> str:
    """'spraybottle 2' -> 'spraybottle'."""
    return entity_id.rsplit(" ", 1)[0].strip()


def _parse_entity_list(blob: str) -> List[str]:
    """Pull canonical '<type> <num>' ids out of a 'you see ...' list.

    Handles 'a cloth 1, a soapbar 1, and a soapbottle 1' and 'nothing'.
    """
    if blob.strip().lower().startswith("nothing"):
        return []
    return [f"{m.group(1)} {m.group(2)}" for m in _RE_ENTITY.finditer(blob)]


# A place objects can be: tracks whether we've looked inside and what we saw.
@dataclass
class Receptacle:
    id: str
    open: Optional[bool] = None       # None = unknown (e.g. an openable not yet touched)
    searched: bool = False            # have we observed its contents?
    contents: Set[str] = field(default_factory=set)


# A movable object instance: where it is and which transforms have been applied.
@dataclass
class Obj:
    id: str
    type: str
    location: Optional[str] = None    # receptacle id, "inventory", or None (unknown)
    clean: bool = False
    heated: bool = False
    cooled: bool = False


# The parsed task goal — tells the WM which object type(s) to track.
@dataclass
class Goal:
    act: str                          # pick_and_place|clean|heat|cool|look|picktwo
    target_type: str
    transform: Optional[str]          # clean|heat|cool|None
    dest_recep_type: Optional[str]
    count: int = 1


class ObjectBelief:
    """Categorical P(instance @ receptacle) over the known receptacle set.

    One of these per tracked object instance. Mass is renormalized over the
    still-unsearched receptacles as absence evidence arrives; once the instance is
    located, the belief collapses (probability 1 on its receptacle) and is inert.
    """

    def __init__(
        self,
        object_type: str,
        receptacles: List[str],
        dist: Optional[Dict[str, float]] = None,
    ):
        # Seed the posterior with the affinity prior; located_at/instance_id stay
        # None until the instance is actually observed somewhere. `dist` overrides
        # the uniform prior (Zipf runs pass zipf_prior).
        self.object_type = object_type
        self.located_at: Optional[str] = None   # receptacle id where the instance was found
        self.instance_id: Optional[str] = None  # the specific object id bound to this belief
        seeded = uniform_prior(object_type, receptacles) if dist is None else dist
        self.dist: Dict[str, float] = dict(seeded)

    def collapse(self, receptacle_id: str, instance_id: Optional[str] = None) -> None:
        # Instance found: pin all mass on its receptacle and bind the concrete id.
        self.located_at = receptacle_id
        self.instance_id = instance_id
        self.dist = {r: (1.0 if r == receptacle_id else 0.0) for r in self.dist}

    def eliminate(self, receptacle_id: str) -> None:
        """Object is NOT in `receptacle_id` (searched, absent). Zero it, renormalize."""
        if self.located_at is not None or receptacle_id not in self.dist:
            return
        self.dist[receptacle_id] = 0.0
        total = sum(self.dist.values())
        if total > 0:
            self.dist = {r: p / total for r, p in self.dist.items()}
        # total==0 (all receptacles searched, object never seen): leave flat-zero;
        # ranking() will return empty and the policy is on its own — a real edge case
        # worth surfacing in smoke tests.

    def ranking(self, exclude_searched: Set[str]) -> List[Tuple[str, float]]:
        """Non-zero-probability receptacles, best-first, excluding searched ones."""
        cand = [
            (r, p) for r, p in self.dist.items()
            if p > 0.0 and r not in exclude_searched
        ]
        return sorted(cand, key=lambda x: (-x[1], x[0]))


class BeliefState:
    """The full Stage-1 world model for one episode."""

    def __init__(
        self,
        goal: Goal,
        receptacles: List[str],
        track_belief: bool = True,
        report_probabilities: bool = True,
        placement: str = "uniform",
    ):
        # Start every known receptacle unsearched; objects/agent are discovered via observe().
        self.goal = goal
        # Ablation switch: when False, drop the probabilistic layer entirely (no affinity
        # prior, no ObjectBelief) and answer where-is from observed facts only.
        self.track_belief = track_belief
        # When False, where-is still uses this distribution to choose the candidate
        # set, but the answer lists those receptacles in random order with no masses.
        self.report_probabilities = report_probabilities
        # "zipf" seeds the target beliefs from the shared rank table. Uniform
        # runs keep the instance-uniform affinity prior. Only the goal object
        # type is resampled in the games, so other types stay on uniform_prior.
        self.placement = placement or "uniform"
        self.receptacles: Dict[str, Receptacle] = {
            r: Receptacle(id=r) for r in receptacles
        }
        self.objects: Dict[str, Obj] = {}
        self.agent_location: Optional[str] = None
        self.inventory: Set[str] = set()
        # One belief per required instance of the target type (count for picktwo).
        # Skipped for the deterministic-only ablation.
        if track_belief and self.placement == "zipf":
            # Drop the goal receptacle type on place-in tasks so the prior does
            # not put mass on a receptacle that would already finish the episode.
            # Look-at names the lamp, which is not a container goal.
            exclude = None
            if goal.act != "look" and goal.dest_recep_type:
                exclude = [goal.dest_recep_type]
            seed = zipf_prior(goal.target_type, receptacles, exclude_types=exclude)
            self.target_beliefs = [
                ObjectBelief(goal.target_type, receptacles, dist=seed)
                for _ in range(max(1, goal.count))
            ]
        elif track_belief:
            self.target_beliefs = [
                ObjectBelief(goal.target_type, receptacles)
                for _ in range(max(1, goal.count))
            ]
        else:
            self.target_beliefs = []

    # ----------------------------------------------------------------- updates
    def _ensure_object(self, obj_id: str) -> Obj:
        if obj_id not in self.objects:
            self.objects[obj_id] = Obj(id=obj_id, type=_object_type(obj_id))
        return self.objects[obj_id]

    def _record_contents(self, recep_id: str, items: List[str]) -> None:
        """A receptacle's contents were revealed: update receptacle + objects + belief."""
        # Mark the receptacle searched and record what's in it.
        if recep_id not in self.receptacles:
            self.receptacles[recep_id] = Receptacle(id=recep_id)
        rec = self.receptacles[recep_id]
        rec.searched = True
        rec.contents = set(items)
        # Every observed object is now known to be here.
        for obj_id in items:
            self._ensure_object(obj_id).location = recep_id

        # Belief update for each unlocated target instance.
        present_target_ids = [
            o for o in items if _object_type(o) == self.goal.target_type
        ]
        for belief in self.target_beliefs:
            if belief.located_at is not None:
                continue
            # Assign distinct seen instances to distinct unlocated beliefs.
            already = {b.instance_id for b in self.target_beliefs if b.instance_id}
            free = [o for o in present_target_ids if o not in already]
            if free:
                belief.collapse(recep_id, instance_id=free[0])
            else:
                belief.eliminate(recep_id)

    def observe(self, action: str, observation: str) -> None:
        """Fold one (action, observation) text pair into the abstract state + belief.

        `observation` is the raw env text WITHOUT the "Observation: " prefix.
        Unrecognized observations are ignored (belief simply doesn't update).
        """
        action = (action or "").strip()
        obs = (observation or "").strip()

        # Track agent location from the action it just took.
        m_goto = re.match(r"go to (.+)", action, re.IGNORECASE)
        if m_goto:
            self.agent_location = m_goto.group(1).strip()

        # "The X is closed."
        m = _RE_CLOSED.match(obs)
        if m:
            rid = m.group(1).strip()
            self.receptacles.setdefault(rid, Receptacle(id=rid)).open = False
            return

        # "You open the X. The X is open. In it, you see ..."
        if _RE_OPENED.match(obs):
            rid = _RE_OPENED.match(obs).group(1).strip()
            self.receptacles.setdefault(rid, Receptacle(id=rid)).open = True
            mr = _RE_OPEN_REVEAL.search(obs)
            if mr:
                self._record_contents(rid, _parse_entity_list(mr.group(1)))
            return

        # "On/In the X, you see ..."  (go-to a surface, or an already-open container)
        m = _RE_ON_SEE.match(obs)
        if m:
            rid, blob = m.group(1).strip(), m.group(2)
            self._record_contents(rid, _parse_entity_list(blob))
            return

        # "You pick up the Y from the X."
        m = _RE_PICK.match(obs)
        if m:
            obj_id, rid = m.group(1).strip(), m.group(2).strip()
            o = self._ensure_object(obj_id)
            o.location = "inventory"
            self.inventory.add(obj_id)
            if rid in self.receptacles:
                self.receptacles[rid].contents.discard(obj_id)
            return

        # "You put the Y in/on the X."
        m = _RE_PUT.match(obs)
        if m:
            obj_id, rid = m.group(1).strip(), m.group(2).strip()
            o = self._ensure_object(obj_id)
            o.location = rid
            self.inventory.discard(obj_id)
            self.receptacles.setdefault(rid, Receptacle(id=rid)).contents.add(obj_id)
            return

        # Status-changing actions.
        for regex, attr in ((_RE_CLEAN, "clean"), (_RE_HEAT, "heated"), (_RE_COOL, "cooled")):
            m = regex.match(obs)
            if m:
                setattr(self._ensure_object(m.group(1).strip()), attr, True)
                return

    # ------------------------------------------------------------------ queries
    def _searched_ids(self) -> Set[str]:
        return {r for r, rec in self.receptacles.items() if rec.searched}

    def observed_locations(self, object_type: str) -> List[str]:
        """Distinct receptacles where an object of this type has actually been seen.

        Pure deterministic scan over observed objects: excludes held (inventory) and
        never-located (None) objects. No prior, no belief — only observed facts.
        """
        seen = [
            o.location for o in self.objects.values()
            if o.type == object_type and o.location not in (None, "inventory")
        ]
        # Preserve first-seen order while de-duplicating.
        return list(dict.fromkeys(seen))

    def where_is(self, object_type: str) -> List[Tuple[str, float]]:
        """Ranked receptacles for a type. Uses target beliefs when it's the target."""
        # Deterministic-only ablation: report observed receptacles, else empty.
        if not self.track_belief:
            return [(r, 1.0) for r in self.observed_locations(object_type)]
        if object_type == self.goal.target_type:
            searched = self._searched_ids()
            for belief in self.target_beliefs:
                if belief.located_at is None:
                    return belief.ranking(searched)
                # If all located, fall through to report a located one.
            return [(b.located_at, 1.0) for b in self.target_beliefs if b.located_at]
        # Non-target object with no maintained belief: prior over unsearched.
        searched = self._searched_ids()
        prior = uniform_prior(object_type, list(self.receptacles))
        cand = [(r, p) for r, p in prior.items() if p > 0 and r not in searched]
        return sorted(cand, key=lambda x: (-x[1], x[0]))

    def locate_instance(self, obj_id: str) -> Tuple[str, Optional[str]]:
        """Look up a specific object instance (e.g. 'peppershaker 2').

        Returns (status, location): status is "inventory" (held), "located"
        (with the receptacle id), or "unknown" (never observed / location not set).
        """
        o = self.objects.get(obj_id)
        if o is None or o.location is None:
            return ("unknown", None)
        if o.location == "inventory":
            return ("inventory", None)
        return ("located", o.location)

    def receptacle_types(self) -> Set[str]:
        """The set of known receptacle types (e.g. {'cabinet', 'countertop', ...})."""
        return {receptacle_type(r) for r in self.receptacles}

    def contents_of(self, receptacle_id: str) -> Optional[Set[str]]:
        rec = self.receptacles.get(receptacle_id)
        if rec is None or not rec.searched:
            return None
        return set(rec.contents)

    def is_searched(self, receptacle_id: str) -> bool:
        rec = self.receptacles.get(receptacle_id)
        return bool(rec and rec.searched)
