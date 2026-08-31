"""Monkey-patch ScienceWorldEnv.step to expose a 0-100 integer score.

ScienceWorld's native score is a 0-1 float; this patch (adapted from MPO,
github.com/WeiminXiong/MPO utils/replace_sciworld_score.py) rescales it to an
integer 0-100 "score", derives a per-step delta "reward", and terminates the
episode when the score goes negative (a hard-failure signal in ScienceWorld).

Apply once at startup, before constructing the env:
    from utils.sciworld_score import sciworld_monkey_patch
    sciworld_monkey_patch()
"""

from scienceworld import ScienceWorldEnv


def step(self, inputStr: str):
    observation = self.server.step(inputStr)
    raw_score = self.server.getScore()
    score = int(round(100 * raw_score))
    isCompleted = self.server.getCompleted()
    numMoves = self.getNumMoves()

    reward = score - self.lastStepScore
    self.lastStepScore = score

    if numMoves > self.envStepLimit:
        isCompleted = True

    if score < 0:
        isCompleted = True

    infos = {
        "moves": numMoves,
        "raw_score": raw_score,
        "score": score,
        "reward": reward,
        "look": self.look(),
        "inv": self.inventory(),
        "taskDesc": self.taskdescription(),
        "valid": self.getValidActionObjectCombinations(),
        "variationIdx": self.variationIdx,
        "taskName": self.taskName,
        "simplificationStr": self.simplificationStr,
    }

    return observation, reward, isCompleted, infos


def sciworld_monkey_patch():
    """Apply the 0-100 score monkey-patch (stock ScienceWorld scoring)."""
    ScienceWorldEnv.step = step
