"""Populations that learn by imitation, for the post "Is it rational to be good?".

A hundred players, each with a strategy for the repeated prisoner's dilemma. At every step one player looks at
another and copies their strategy, the more likely the better the other has done against everyone (the Fermi
rule); now and then a player tries a strategy at random instead. Payoffs are exact (games.py).

    uv run --group articles python articles/game_theory/simulate.py

writes data/evolution.json: one run of each population over time, and averages over twenty runs.
"""

import json
import random
from math import exp
from pathlib import Path

from games import STRATEGIES, payoffs

DATA = Path(__file__).parent / "data"
W, NOISE = 0.95, 0.02  # about twenty rounds per relationship; one move in fifty comes out wrong
N, STEPS, EVERY = 100, 600_000, 1_000
SELECTION, EXPLORE = 2.0, 0.002
POPULATIONS = {
    "without": ["Always defect", "Tit for tat", "Always cooperate"],
    "with": ["Always defect", "Tit for tat", "Always cooperate", "Generous tit for tat"],
}


def run(names, seed):
    rng = random.Random(seed)
    k = len(names)
    table = []
    for a in names:
        row = []
        for b in names:
            row.append(payoffs(STRATEGIES[a], STRATEGIES[b], W, NOISE)[0])
        table.append(row)
    counts = [0] * k
    counts[0] = N  # everyone starts as a defector

    def pick():
        r = rng.randrange(N)
        for i in range(k):
            if r < counts[i]:
                return i
            r -= counts[i]

    def fitness(i):  # the mean payoff against everyone else in the population
        total = 0.0
        for j in range(k):
            total += table[i][j] * counts[j]
        return (total - table[i][i]) / (N - 1)

    shares, payoff_over_time = [], []
    for step in range(STEPS):
        learner = pick()
        if rng.random() < EXPLORE:
            new = rng.randrange(k)
        else:
            model = pick()
            copy = 1 / (1 + exp(-SELECTION * (fitness(model) - fitness(learner))))
            new = model if rng.random() < copy else learner
        counts[learner] -= 1
        counts[new] += 1
        if step % EVERY == 0:
            shares.append([c / N for c in counts])
            mean = 0.0
            for i in range(k):
                mean += counts[i] / N * fitness(i)
            payoff_over_time.append(mean)
    return shares, payoff_over_time


def main():
    result = {"w": W, "noise": NOISE, "n": N, "steps": STEPS, "every": EVERY, "populations": {}}
    for key, names in POPULATIONS.items():
        shown, mean_payoffs, defector_majority = None, [], []
        for seed in range(1, 21):
            shares, payoff_over_time = run(names, seed)
            if seed == 1:
                shown = {"shares": shares, "payoff": payoff_over_time}
            mean_payoffs.append(sum(payoff_over_time) / len(payoff_over_time))
            majority = 0
            for row in shares:
                if row[0] > 0.5:
                    majority += 1
            defector_majority.append(majority / len(shares))
        result["populations"][key] = {
            "names": names,
            "run": shown,
            "mean_payoff": sum(mean_payoffs) / len(mean_payoffs),
            "defectors_in_majority": sum(defector_majority) / len(defector_majority),
            "runs": len(mean_payoffs),
        }
        print(key, round(result["populations"][key]["mean_payoff"], 2),
              round(result["populations"][key]["defectors_in_majority"], 2))
    DATA.mkdir(exist_ok=True)
    (DATA / "evolution.json").write_text(json.dumps(result), encoding="utf-8")


if __name__ == "__main__":
    main()
