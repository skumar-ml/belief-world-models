"""Load and run WALL-E's ported ALFWorld action-validity rules.

Mirrors WALL-E's `RuleVerifier.load_functions` / `run_all_functions` (commit 3dad1ed,
see SOURCE.md). Each rule is a Python function `f(state, action, scene_graph) ->
(feedback, success, suggestion)` that flags an *infeasible* action; a True never
confirms feasibility, it only means "this rule doesn't object". The first rule that
returns success=False rejects the action.
"""

import json
import os
import re

_RULES_JSON = os.path.join(os.path.dirname(__file__), "rules_code_alfworld.json")


def LLM_request(request, *args, **kwargs):
    """Safe stub for the helper some WALL-E rules may call.

    None of the 15 shipped ALFWorld rules use it, but we define it so a stray call
    can't crash the gate. Returns a benign "True" (rules that would branch on it treat
    the action as allowed).
    """
    return "True"


def load_rules(path=_RULES_JSON):
    """exec each rule-function string into a callable. Returns list[callable].

    The ALFWorld rules take a positional `scene_graph` arg they never use; we keep the
    signature intact and pass scene_graph=None at call time (see check_action).
    """
    with open(path, "r") as f:
        rule_strings = json.load(f)

    # Shared namespace so rules can see LLM_request and each other.
    ns = {"LLM_request": LLM_request, "re": re, "json": json}
    functions = []
    for func_str in rule_strings:
        # Normalize any spaces in the function name (mirrors WALL-E's loader).
        func_def_match = re.search(r'def\s+([\w\s]+)\s*\(', func_str)
        if func_def_match:
            original = func_def_match.group(1)
            func_str = func_str.replace(f'def {original}(', f'def {original.replace(" ", "_")}(')
        try:
            exec(func_str, ns)
        except Exception:
            continue  # skip a rule that won't even compile
        name_match = re.search(r'def\s+(\w+)\s*\(', func_str)
        if name_match:
            functions.append(ns[name_match.group(1)])
    return functions


def check_action(state, action, rules):
    """Run all rules against (state, action). Return the first failure, else success.

    Result dict: {"success": bool, "feedback": str, "suggestion": str}. A rule that
    raises is treated defensively as no-objection (success=True for that rule),
    matching WALL-E's try/except behavior.
    """
    for func in rules:
        try:
            feedback, success, suggestion = func(state=state, action=action, scene_graph=None)
        except TypeError:
            # Rule defined without the scene_graph param — retry with two args.
            try:
                feedback, success, suggestion = func(state=state, action=action)
            except Exception:
                continue
        except Exception:
            continue
        if not success:
            return {"feedback": feedback, "success": False, "suggestion": suggestion}
    return {"feedback": "You completed the action successfully.", "success": True, "suggestion": ""}
