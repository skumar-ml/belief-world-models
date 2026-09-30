"""MixedTrainLocal room prior: one 8x8 room, 6x6 walkable interior.

Wall-to-wall span is 7 (walls at 0 and 7 in MiniGrid). Pose lives in the
start-frame map; wall rays pin the box in that frame. No oracle.
"""

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set, Tuple

from wm.babyai_scene import cell_key


ROOM_SIZE = 8
INTERIOR_SIZE = 6
WALL_SPAN = 7          # wall cell to opposite wall cell
VIEW_SIZE = 7          # MiniGrid agent_view_size
# Heading 0 = +Y (start facing), then clockwise: +X, -Y, -X.
_HEADING_VEC = ((0, 1), (1, 0), (0, -1), (-1, 0))
_HEADING_ARROW = ("^", ">", "v", "<")


def _right_vec(heading: int) -> Tuple[int, int]:
    dx, dy = _HEADING_VEC[heading]
    return dy, -dx


def rel_to_abs(
    x: int, y: int, heading: int, forward: int, right: int
) -> Tuple[int, int]:
    hx, hy = _HEADING_VEC[heading]
    rx, ry = _right_vec(heading)
    return x + forward * hx + right * rx, y + forward * hy + right * ry


def abs_to_rel(
    x: int, y: int, heading: int, cell: Tuple[int, int]
) -> Tuple[int, int]:
    dx, dy = cell[0] - x, cell[1] - y
    hx, hy = _HEADING_VEC[heading]
    rx, ry = _right_vec(heading)
    return dx * hx + dy * hy, dx * rx + dy * ry


def start_forbidden(origin: Tuple[int, int] = (0, 0)) -> Set[Tuple[int, int]]:
    """Cells objects cannot occupy at generation (Manhattan d < 2 from start)."""
    x, y = origin
    return {(x, y), (x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)}


@dataclass
class RoomBox:
    """Axis-aligned wall ring in the start-pose frame, filled in from wall rays."""

    x_lo: Optional[int] = None
    x_hi: Optional[int] = None
    y_lo: Optional[int] = None
    y_hi: Optional[int] = None

    def localized(self) -> bool:
        return None not in (self.x_lo, self.x_hi, self.y_lo, self.y_hi)

    def pin_from_ray(self, agent: Tuple[int, int], wall: Tuple[int, int]) -> None:
        """One cardinal wall ray pins both walls on that axis (span is known)."""
        ax, ay = agent
        wx, wy = wall
        if wx == ax and wy != ay:
            if wy > ay:
                if self.y_hi is None:
                    self.y_hi = wy
                    self.y_lo = wy - WALL_SPAN
            elif self.y_lo is None:
                self.y_lo = wy
                self.y_hi = wy + WALL_SPAN
            return
        if wy == ay and wx != ax:
            if wx > ax:
                if self.x_hi is None:
                    self.x_hi = wx
                    self.x_lo = wx - WALL_SPAN
            elif self.x_lo is None:
                self.x_lo = wx
                self.x_hi = wx + WALL_SPAN

    def interior_cells(self) -> List[Tuple[int, int]]:
        if not self.localized():
            return []
        return [
            (x, y)
            for x in range(self.x_lo + 1, self.x_hi)
            for y in range(self.y_lo + 1, self.y_hi)
        ]

    def wall_cells(self) -> Set[Tuple[int, int]]:
        if not self.localized():
            return set()
        cells = set()
        for x in range(self.x_lo, self.x_hi + 1):
            cells.add((x, self.y_lo))
            cells.add((x, self.y_hi))
        for y in range(self.y_lo, self.y_hi + 1):
            cells.add((self.x_lo, y))
            cells.add((self.x_hi, y))
        return cells

    def contains_interior(self, pos: Tuple[int, int]) -> bool:
        if not self.localized():
            return False
        x, y = pos
        return self.x_lo < x < self.x_hi and self.y_lo < y < self.y_hi


def _process_vis(walls: Set[Tuple[int, int]], width: int, height: int, agent: Tuple[int, int]):
    """MiniGrid Grid.process_vis over a view-sized grid (walls block, objects do not)."""
    mask = [[False] * height for _ in range(width)]
    ax, ay = agent
    if not (0 <= ax < width and 0 <= ay < height):
        return mask
    mask[ax][ay] = True
    for j in range(height - 1, -1, -1):
        for i in range(0, width - 1):
            if not mask[i][j] or (i, j) in walls:
                continue
            mask[i + 1][j] = True
            if j > 0:
                mask[i + 1][j - 1] = True
                mask[i][j - 1] = True
        for i in range(width - 1, 0, -1):
            if not mask[i][j] or (i, j) in walls:
                continue
            mask[i - 1][j] = True
            if j > 0:
                mask[i - 1][j - 1] = True
                mask[i][j - 1] = True
    return mask


def visible_world_cells(
    x: int,
    y: int,
    heading: int,
    wall_cells: Iterable[Tuple[int, int]],
    view_size: int = VIEW_SIZE,
) -> Set[Tuple[int, int]]:
    """World cells inside the current 7x7 FoV that MiniGrid would mark visible."""
    walls_view = set()
    half = view_size // 2
    for wx, wy in wall_cells:
        fwd, right = abs_to_rel(x, y, heading, (wx, wy))
        i, j = half + right, (view_size - 1) - fwd
        if 0 <= i < view_size and 0 <= j < view_size:
            walls_view.add((i, j))
    mask = _process_vis(walls_view, view_size, view_size, (half, view_size - 1))
    out = set()
    for i in range(view_size):
        for j in range(view_size):
            if not mask[i][j]:
                continue
            fwd, right = (view_size - 1) - j, i - half
            out.add(rel_to_abs(x, y, heading, fwd, right))
    return out


def heading_arrow(heading: int) -> str:
    return _HEADING_ARROW[heading % 4]


def ascii_room(
    room: RoomBox,
    agent: Tuple[int, int],
    heading: int,
    cells: Dict[Tuple[int, int], object],
    candidate_atoms: Set[str],
) -> str:
    """Print the wall ring with +Y (start-forward) toward the top of the dump."""
    if not room.localized():
        return "Room box is not localized yet."
    glyphs = []
    for y in range(room.y_hi, room.y_lo - 1, -1):
        row = []
        for x in range(room.x_lo, room.x_hi + 1):
            pos = (x, y)
            if pos == agent:
                row.append(heading_arrow(heading))
                continue
            fact = cells.get(pos)
            if fact is not None and getattr(fact, "kind", None) == "wall":
                row.append("#")
            elif fact is not None and getattr(fact, "kind", None) == "object":
                typ = getattr(fact, "type", None) or "o"
                row.append(typ[0].upper())
            elif fact is not None and getattr(fact, "kind", None) == "door":
                row.append("D")
            elif cell_key(*pos) in candidate_atoms:
                row.append("?")
            elif fact is not None and getattr(fact, "kind", None) == "empty":
                row.append(".")
            elif room.contains_interior(pos):
                row.append(".")
            else:
                row.append("#")
        glyphs.append(" ".join(row))
    return "\n".join(glyphs)
