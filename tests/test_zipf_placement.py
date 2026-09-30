"""Zipf placement: shared rank table, game rewrite, and the uniform default."""

import hashlib
import inspect
import json
import os
import re
import shutil
import tempfile
import unittest
from unittest.mock import patch

from tasks.alfworld import AlfWorldTask
from tasks.alfworld_zipf import (
    assess_game,
    ensure_zipf_cache,
    pddl_type_name,
    rewrite_pddl_problem,
    zipf_output_split,
    _planner_env,
)
from wm.affinity_prior import (
    ZIPF_ALPHA,
    ZIPF_RANK_SEED,
    build_zipf_ranks,
    uniform_prior,
    zipf_prior,
    zipf_rank,
)
from wm.belief import BeliefState, Goal


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNSEEN = os.path.join(ROOT, "data/alfworld/json_2.1.1/valid_unseen")

_PROBLEM = """
(define (problem t)
(:domain alfred)
(:init
(objectType CreditCard1 CreditCardType)
(objectType CreditCard2 CreditCardType)
(objectType Knife1 KnifeType)
(receptacleType Drawer1 DrawerType)
(receptacleType Cabinet1 CabinetType)
(receptacleType Dresser1 DresserType)
(receptacleType Counter1 CounterTopType)
(receptacleAtLocation Drawer1 loc_d)
(receptacleAtLocation Cabinet1 loc_c)
(receptacleAtLocation Dresser1 loc_r)
(receptacleAtLocation Counter1 loc_k)
(inReceptacle CreditCard1 Drawer1)
(inReceptacle CreditCard2 Cabinet1)
(inReceptacle Knife1 Drawer1)
(inReceptacleObject CreditCard2 Bowl1)
(inReceptacleObject Spatula1 CreditCard2)
(objectAtLocation CreditCard1 loc_d)
(objectAtLocation CreditCard2 loc_c)
(objectAtLocation Knife1 loc_d)
(objectAtLocation Spatula1 loc_c)
(isHot CreditCard1)
)
(:goal
(and
(inReceptacle ?o ?r)
(objectType ?o CreditCardType)
)
)
)
"""


def _fact_map(problem, pred):
    pairs = re.findall(rf"\({pred}\s+(\S+)\s+(\S+)\)", problem)
    return {a: b for a, b in pairs if not a.startswith("?")}


class ZipfRankTests(unittest.TestCase):
    def test_table_is_stable_and_per_type(self):
        payload = json.load(open(os.path.join(ROOT, "wm/zipf_ranks.json")))
        self.assertEqual(payload["alpha"], ZIPF_ALPHA)
        self.assertEqual(payload["seed"], ZIPF_RANK_SEED)
        self.assertEqual(payload["alpha"], 1.25)
        rebuilt = build_zipf_ranks()
        self.assertEqual(rebuilt, payload["ranks"])
        self.assertEqual(build_zipf_ranks(), build_zipf_ranks())
        self.assertNotEqual(zipf_rank("creditcard"), zipf_rank("watch"))

    def test_type_mass_splits_across_instances_and_drops_the_goal(self):
        present = ["sidetable 1", "sidetable 2", "toilet 1", "cart 1"]
        dist = zipf_prior("spraybottle", present)
        self.assertAlmostEqual(sum(dist.values()), 1.0)
        self.assertAlmostEqual(dist["sidetable 1"], dist["sidetable 2"])
        self.assertGreater(dist["sidetable 1"], dist["toilet 1"])
        self.assertGreater(dist["toilet 1"], dist["cart 1"])
        # Rank 1 is shared, so each sidetable still carries more than the next type.
        self.assertGreater(dist["sidetable 1"] * 2, dist["toilet 1"])

        excluded = zipf_prior("spraybottle", present, exclude_types=["toilet"])
        self.assertEqual(excluded["toilet 1"], 0.0)
        self.assertAlmostEqual(sum(excluded.values()), 1.0)
        self.assertGreater(excluded["sidetable 1"], dist["sidetable 1"])

    def test_uniform_prior_stays_flat(self):
        present = ["sidetable 1", "sidetable 2", "toilet 1"]
        dist = uniform_prior("spraybottle", present)
        self.assertAlmostEqual(sum(dist.values()), 1.0)
        self.assertEqual(len({round(p, 10) for p in dist.values()}), 1)


class BeliefPriorTests(unittest.TestCase):
    def test_uniform_default_includes_the_goal_receptacle(self):
        goal = Goal("pick_and_place", "watch", None, "safe", count=1)
        present = ["countertop 1", "drawer 1", "safe 1"]
        belief = BeliefState(goal, present)
        self.assertEqual(belief.placement, "uniform")
        self.assertGreater(belief.target_beliefs[0].dist["safe 1"], 0.0)
        self.assertAlmostEqual(sum(belief.target_beliefs[0].dist.values()), 1.0)

    def test_zipf_drops_goal_type_and_picktwo_gets_two_copies(self):
        goal = Goal("picktwo", "watch", None, "safe", count=2)
        present = ["countertop 1", "drawer 1", "safe 1", "cabinet 1"]
        belief = BeliefState(goal, present, placement="zipf")
        self.assertEqual(len(belief.target_beliefs), 2)
        dist = belief.target_beliefs[0].dist
        self.assertEqual(dist["safe 1"], 0.0)
        self.assertEqual(dist["cabinet 1"], 0.0)
        self.assertGreater(dist["countertop 1"], dist["drawer 1"])
        self.assertAlmostEqual(sum(dist.values()), 1.0)
        belief.target_beliefs[0].eliminate("countertop 1")
        self.assertGreater(belief.target_beliefs[1].dist["countertop 1"], 0.0)

        looked = BeliefState(
            Goal("look", "watch", None, "safe", count=1),
            present,
            placement="zipf",
        )
        self.assertGreater(looked.target_beliefs[0].dist["safe 1"], 0.0)

    def test_nontarget_query_stays_uniform_on_a_zipf_run(self):
        goal = Goal("pick_and_place", "watch", None, "safe", count=1)
        present = ["countertop 1", "drawer 1", "safe 1"]
        belief = BeliefState(goal, present, placement="zipf")
        ranked = dict(belief.where_is("spraybottle"))
        prior = {
            r: p for r, p in uniform_prior("spraybottle", present).items() if p > 0
        }
        self.assertEqual(set(ranked), set(prior))


class RewriteTests(unittest.TestCase):
    def test_picktwo_moves_both_instances_and_leaves_the_knife(self):
        seen = {}

        def choose(obj_id, weights, by_type):
            self.assertNotIn("dresser", weights)
            self.assertIn("drawer", weights)
            seen.setdefault(obj_id, len(seen))
            return "Counter1" if seen[obj_id] == 0 else "Drawer1"

        result = rewrite_pddl_problem(
            _PROBLEM, "creditcard", exclude_types=["dresser"], choose=choose
        )
        self.assertIsNotNone(result)
        problem, moved = result
        self.assertEqual(set(moved), {"CreditCard1", "CreditCard2"})
        self.assertEqual(_fact_map(problem, "inReceptacle")["CreditCard1"], "Counter1")
        self.assertEqual(_fact_map(problem, "inReceptacle")["CreditCard2"], "Drawer1")
        self.assertEqual(_fact_map(problem, "objectAtLocation")["CreditCard1"], "loc_k")
        self.assertEqual(_fact_map(problem, "objectAtLocation")["CreditCard2"], "loc_d")
        self.assertEqual(_fact_map(problem, "inReceptacle")["Knife1"], "Drawer1")
        self.assertEqual(_fact_map(problem, "objectAtLocation")["Knife1"], "loc_d")
        self.assertNotIn("(inReceptacleObject CreditCard2", problem)
        self.assertIn("(inReceptacleObject Spatula1 CreditCard2)", problem)
        self.assertEqual(_fact_map(problem, "objectAtLocation")["Spatula1"], "loc_d")
        self.assertIn("(isHot CreditCard1)", problem)
        self.assertIn("(inReceptacle ?o ?r)", problem)

    def test_independent_draws(self):
        placements = []
        for i in range(40):
            result = rewrite_pddl_problem(
                _PROBLEM, "creditcard", exclude_types=["dresser"], game_key=f"game-{i}"
            )
            problem, moved = result
            placements.append(tuple(sorted(moved.values())))
            for recep in moved.values():
                self.assertNotEqual(recep, "Dresser1")
        self.assertGreater(len(set(placements)), 1)

    def test_location_only_instance_is_placed(self):
        # Released games sometimes omit inReceptacle and keep objectAtLocation.
        problem = _PROBLEM
        problem = problem.replace("(inReceptacle CreditCard1 Drawer1)\n", "")
        problem = problem.replace("(inReceptacle CreditCard2 Cabinet1)\n", "")
        problem = problem.replace("(inReceptacleObject CreditCard2 Bowl1)\n", "")
        result = rewrite_pddl_problem(
            problem, "creditcard", exclude_types=["dresser"], game_key="loc-only"
        )
        self.assertIsNotNone(result)
        rewritten, moved = result
        self.assertEqual(set(moved), {"CreditCard1", "CreditCard2"})
        locs = {"Drawer1": "loc_d", "Counter1": "loc_k", "Dresser1": "loc_r"}
        for oid, recep in moved.items():
            self.assertIn(f"(inReceptacle {oid} {recep})", rewritten)
            self.assertNotEqual(recep, "Dresser1")
            self.assertEqual(_fact_map(rewritten, "objectAtLocation")[oid], locs[recep])
        self.assertEqual(_fact_map(rewritten, "inReceptacle")["Knife1"], "Drawer1")

    def test_empty_candidates(self):
        only_goal = _PROBLEM.replace("(receptacleAtLocation Drawer1 loc_d)\n", "")
        only_goal = only_goal.replace("(receptacleAtLocation Counter1 loc_k)\n", "")
        only_goal = only_goal.replace("(receptacleAtLocation Cabinet1 loc_c)\n", "")
        self.assertIsNone(
            rewrite_pddl_problem(only_goal, "creditcard", exclude_types=["dresser"])
        )

    def test_uniform_switch_defaults(self):
        self.assertEqual(
            inspect.signature(AlfWorldTask.load_tasks).parameters["placement"].default,
            "uniform",
        )
        self.assertEqual(zipf_output_split("unseen"), "unseen_zipf")
        self.assertEqual(zipf_output_split("unseen_zipf"), "unseen_zipf")
        self.assertEqual(zipf_output_split("unseen_foo"), "unseen_foo_zipf")


class PlannerCacheTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.env = _planner_env()
        cls.watch = os.path.join(
            UNSEEN,
            "pick_and_place_simple-Watch-None-Safe-219/trial_T20190907_074556_124850",
        )
        cls.picktwo = os.path.join(
            UNSEEN,
            "pick_two_obj_and_place-SoapBar-None-Cabinet-424/trial_T20190909_081720_491733",
        )

    def test_real_games_are_solvable_and_not_already_won(self):
        for trial, obj, parent in (
            (self.watch, "watch", "safe"),
            (self.picktwo, None, None),
        ):
            game = json.load(open(os.path.join(trial, "game.tw-pddl")))
            traj = json.load(open(os.path.join(trial, "traj_data.json")))
            obj = obj or traj["pddl_params"]["object_target"].lower()
            parent = parent if parent is not None else traj["pddl_params"]["parent_target"].lower()
            original = game["pddl_problem"]
            result = rewrite_pddl_problem(
                original, obj, exclude_types=[parent], game_key=trial
            )
            self.assertIsNotNone(result, trial)
            problem, moved = result
            self.assertGreaterEqual(len(moved), 2 if "pick_two" in trial else 1)
            types = {
                rid: pddl_type_name(tok)
                for rid, tok in re.findall(r"\(receptacleType\s+(\S+)\s+(\S+)\)", problem)
            }
            locs = dict(re.findall(r"\(receptacleAtLocation\s+(\S+)\s+(\S+)\)", problem))
            at = _fact_map(problem, "objectAtLocation")
            for oid, rid in moved.items():
                self.assertNotEqual(types[rid], parent)
                self.assertEqual(at[oid], locs[rid])
            # Distractors keep their original receptacle.
            old_in = _fact_map(original, "inReceptacle")
            new_in = _fact_map(problem, "inReceptacle")
            for oid, rid in old_in.items():
                if oid not in moved:
                    self.assertEqual(new_in[oid], rid)
            for flag in ("isHot", "isCool", "isClean"):
                self.assertEqual(original.count(f"({flag} "), problem.count(f"({flag} "))
            candidate = dict(game)
            candidate["pddl_problem"] = problem
            self.assertEqual(assess_game(self.env, candidate), "ok", trial)

        forced = dict(json.load(open(os.path.join(self.watch, "game.tw-pddl"))))
        safe = next(
            rid for rid, tok in re.findall(
                r"\(receptacleType\s+(\S+)\s+(\S+)\)", forced["pddl_problem"]
            )
            if pddl_type_name(tok) == "safe"
        )
        won_problem, _moved = rewrite_pddl_problem(
            forced["pddl_problem"], "watch", game_key="won",
            choose=lambda _oid, _w, _by: safe,
        )
        forced["pddl_problem"] = won_problem
        self.assertEqual(assess_game(self.env, forced), "won")
        self.assertTrue(assess_game(self.env, {"pddl_problem": "nope"}).startswith("error"))

    def test_cache_keeps_original_on_reject_and_does_not_touch_source(self):
        src_hash = hashlib.sha256(open(os.path.join(self.watch, "game.tw-pddl"), "rb").read()).hexdigest()
        with tempfile.TemporaryDirectory() as td:
            root = os.path.join(td, "alfworld")
            trial = os.path.join(
                root, "json_2.1.1/valid_unseen/pick/trial"
            )
            os.makedirs(trial)
            shutil.copy(os.path.join(self.watch, "game.tw-pddl"), trial)
            shutil.copy(os.path.join(self.watch, "traj_data.json"), trial)
            with open(os.path.join(root, "base_config.yaml"), "w") as f:
                f.write("k: v\n")
            os.makedirs(os.path.join(root, "logic"))
            source_bytes = open(os.path.join(trial, "game.tw-pddl"), "rb").read()

            with patch("tasks.alfworld_zipf.assess_game", return_value="unsolvable"):
                cache = ensure_zipf_cache(root, "test")
            cached = os.path.join(cache, "json_2.1.1/valid_unseen/pick/trial")
            self.assertEqual(open(os.path.join(cached, "game.tw-pddl"), "rb").read(), source_bytes)
            self.assertTrue(os.path.islink(os.path.join(cached, "traj_data.json")))
            self.assertTrue(os.path.islink(os.path.join(cache, "base_config.yaml")))
            self.assertTrue(os.path.islink(os.path.join(cache, "logic")))
            meta = json.load(open(os.path.join(cached, ".zipf_meta.json")))
            self.assertEqual(meta["status"], "fallback")
            self.assertEqual(meta["reason"], "unsolvable")
            reasons = [
                json.loads(line)["reason"]
                for line in open(os.path.join(cache, "zipf_fallbacks.jsonl"))
            ]
            self.assertIn("unsolvable", reasons)
            self.assertEqual(open(os.path.join(trial, "game.tw-pddl"), "rb").read(), source_bytes)
            # Same rank table: the fallback is reused, not rebuilt.
            again = ensure_zipf_cache(root, "test")
            self.assertEqual(again, cache)
            self.assertEqual(open(os.path.join(cached, "game.tw-pddl"), "rb").read(), source_bytes)

        with tempfile.TemporaryDirectory() as td:
            root = os.path.join(td, "alfworld")
            trial = os.path.join(root, "json_2.1.1/valid_unseen/pick/trial")
            os.makedirs(trial)
            shutil.copy(os.path.join(self.watch, "game.tw-pddl"), trial)
            shutil.copy(os.path.join(self.watch, "traj_data.json"), trial)
            source_bytes = open(os.path.join(trial, "game.tw-pddl"), "rb").read()
            cache = ensure_zipf_cache(root, "test")
            cached = os.path.join(cache, "json_2.1.1/valid_unseen/pick/trial")
            meta = json.load(open(os.path.join(cached, ".zipf_meta.json")))
            self.assertEqual(meta["status"], "rewritten")
            rewritten = open(os.path.join(cached, "game.tw-pddl"), "rb").read()
            self.assertNotEqual(rewritten, source_bytes)
            self.assertEqual(open(os.path.join(trial, "game.tw-pddl"), "rb").read(), source_bytes)
            self.assertTrue(os.path.islink(os.path.join(cached, "traj_data.json")))
            again = ensure_zipf_cache(root, "test")
            self.assertEqual(again, cache)
            self.assertEqual(open(os.path.join(cached, "game.tw-pddl"), "rb").read(), rewritten)

        self.assertEqual(
            hashlib.sha256(open(os.path.join(self.watch, "game.tw-pddl"), "rb").read()).hexdigest(),
            src_hash,
        )


if __name__ == "__main__":
    unittest.main()
