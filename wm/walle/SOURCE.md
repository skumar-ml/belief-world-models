# Provenance of the WALL-E port

The files in `wm/walle/` are ported from the official WALL-E 2.0 repository so we
can run WALL-E as an ALFWorld baseline in this harness.

- **Paper:** WALL-E 2.0: World Alignment by NeuroSymbolic Learning improves World
  Model-based LLM Agents. Zhou et al., arXiv:2504.15785.
- **Repo:** https://github.com/elated-sawyer/WALL-E
- **Commit ported from:** `3dad1ed29854cf0a49a35c2e4589e31d1a3200f0` (2025-12-03)

## What was ported

- `rules_code_alfworld.json` — **verbatim** copy of
  `alfworld/symbolic_knowledge/alfworld/rules_code.json`. The 15 shipped ALFWorld
  action-validity rules (WALL-E's released neurosymbolic knowledge for ALFWorld).
  Each rule is a Python function `f(state, action, scene_graph) -> (feedback, success,
  suggestion)` that flags an *infeasible* action. Note: none of the ALFWorld rule
  bodies actually reference `scene_graph`; we pass `scene_graph=None`.
- `state_transform.py` — port of `alfworld_runs/stateinfo_transform/state_info_transform.py`
  (the text→state-dict extractor the rules were written against), adapted to consume this
  harness's `state.history` list instead of WALL-E's raw ReAct transcript string.
- `action_transform.py` — verbatim port of
  `alfworld_runs/stateinfo_transform/action_info_transform.py` (`convert_action`).

## What was NOT ported

- The offline rule-*learning* loop (explore → mine NL rules → LLM codegen → set-cover
  prune). We use WALL-E's released rule set directly.
- Scene graphs / knowledge graphs — Mars-only; unused by the ALFWorld rules.
