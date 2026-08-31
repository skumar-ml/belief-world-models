"""Pure (alfworld-free) helpers for reading the ALFWorld reset observation.

Kept dependency-light so `wm/` is importable and testable on a login node without
the OpenGL-bearing alfworld env stack. The FoV and WM envs import these helpers.
"""

import re
from typing import List


# The standard reset observation enumerates every receptacle in one sentence:
#   "Looking quickly around you, you see a cabinet 4, a cabinet 3, ... and a toilet 1.\n"
RECEP_BLOB_RE = re.compile(r"Looking quickly around you, you see (.*?)\.\s*\n", re.DOTALL)


def parse_receptacles(blob: str) -> List[str]:
    """Split an enumeration blob like "a cabinet 4, a cabinet 3, and a toilet 1" into
    canonical receptacle tokens like ["cabinet 4", "cabinet 3", "toilet 1"]."""
    receptacles = []
    for chunk in re.split(r",|\band\b", blob):
        name = re.sub(r"^an?\s+", "", chunk.strip())
        if name:
            receptacles.append(name)
    return receptacles
