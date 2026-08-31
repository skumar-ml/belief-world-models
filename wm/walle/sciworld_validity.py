"""Membership test against ScienceWorld's getValidActionObjectCombinations().

DEPRECATED as the Phase 0a gate mechanism. The WalleOracleSciWorldEnv gate NO LONGER uses
this: getValidActionObjectCombinations() UNDER-approximates the parser (it omits parseable
forms like `examine X`, `deactivate X`-when-already-off, `read X`), so membership-based
rejection falsely blocked valid actions and collapsed SR 2.5-4x. The env now uses the
PARSER itself as the oracle (step + the "No known action matches that input" no-op signal);
see envs/walle_oracle_sciworld_env.py. This module is retained only for analysis / possible
Phase 0b use of the valid-combination enumeration; it is not on the eval path.

`ScienceWorldEnv.getValidActionObjectCombinations()` returns the flat list of concrete
action strings for the current state (e.g. "go to kitchen", "focus on potato"). This module
wraps a membership test against that set.

Design notes:
- SOUNDNESS FIRST. The gate must never reject an action the parser would actually accept
  (a false rejection breaks a valid trajectory). We therefore only *reject* when the
  action is absent from the valid set under conservative normalization; we never approve
  via fuzzy expansion. Verified: the valid set is state-accurate at the pre-step moment
  (an executed valid action IS in the pre-step set; a parser-invalid action is NOT).
- Phase 0a is deliberately parser-validity ONLY. Parser-valid but task-terminating
  actions (e.g. `focus on <wrong object>`) ARE in the valid set, so this gate passes them
  through untouched — catching them is Phase 0b's job.
- Non-physical / meta verbs the agent might emit around the ReAct loop (`think:`,
  `query`) are not env actions; the gate treats an empty/None action as passthrough and
  the env class decides.
"""

import re


def normalize_action(action_str):
    """Conservative canonicalization for matching against the valid set.

    Lowercase, collapse internal whitespace, strip trailing punctuation. We do NOT
    rewrite verbs or drop words — that could turn a parser-invalid action into an
    apparent match (an unsound approval). ScienceWorld's own valid strings are already
    lowercase with single spaces, so this only absorbs harmless agent-side formatting
    noise (extra spaces, a trailing period, stray capitalization).
    """
    if action_str is None:
        return None
    s = action_str.strip().lower()
    s = re.sub(r"\s+", " ", s)
    s = s.rstrip(" .!")
    return s


def build_valid_set(valid_actions):
    """Normalize the env's valid-action list into a set for O(1) membership tests."""
    return {normalize_action(a) for a in (valid_actions or [])}


def is_valid_action(action_str, valid_set):
    """True iff the (normalized) action is in the (normalized) valid set.

    An empty/None action is treated as valid=True (passthrough): it is not a physical
    env action to gate — the caller's parse/bad-step path handles malformed output.
    """
    norm = normalize_action(action_str)
    if not norm:
        return True
    return norm in valid_set
