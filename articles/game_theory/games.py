"""The games of the post "Is it rational to be good?".

The repeated prisoner's dilemma between memory-one strategies is a Markov chain on the four outcomes of a
round, so the payoffs are computed exactly instead of by playing: with a probability w of another round and a
probability `noise` that a move comes out as the opposite of what was meant.
"""

import numpy as np

T, R, P, S = 5, 3, 1, 0  # temptation, reward, punishment, sucker's payoff: Axelrod's numbers

# a round's outcome from one player's side: what I did, what the other did
OUTCOMES = ("CC", "CD", "DC", "DD")
MINE = np.array([R, S, T, P], dtype=float)
THEIRS = np.array([R, T, S, P], dtype=float)
SWAP = [0, 2, 1, 3]  # the same outcome seen from the other side


def strategy(first, cc, cd, dc, dd):
    # a memory-one strategy: the probability of cooperating in the first round, and after each outcome
    return np.array([first, cc, cd, dc, dd], dtype=float)


def zero_determinant(chi, baseline, first):
    # Press and Dyson: a strategy that pins my payoff to the other's, s_me - l = chi (s_them - l).
    # p - (1, 1, 0, 0) must equal phi ((MINE - l) - chi (THEIRS - l)); take the largest phi that keeps every
    # probability in [0, 1]
    direction = (MINE - baseline) - chi * (THEIRS - baseline)
    base = np.array([1, 1, 0, 0], dtype=float)
    phi = np.inf
    for b, d in zip(base, direction, strict=True):
        if d > 0:
            phi = min(phi, (1 - b) / d)
        elif d < 0:
            phi = min(phi, -b / d)
    p = base + phi * direction
    return np.concatenate([[first], p])


STRATEGIES = {
    "Always cooperate": strategy(1, 1, 1, 1, 1),
    "Always defect": strategy(0, 0, 0, 0, 0),
    "Tit for tat": strategy(1, 1, 0, 1, 0),
    "Grim trigger": strategy(1, 1, 0, 0, 0),
    "Win-stay, lose-shift": strategy(1, 1, 0, 0, 1),
    "Generous tit for tat": strategy(1, 1, 1 / 3, 1, 1 / 3),
    "Suspicious tit for tat": strategy(0, 1, 0, 1, 0),
    "Joss": strategy(1, 0.9, 0, 0.9, 0),
    "Random": strategy(0.5, 0.5, 0.5, 0.5, 0.5),
    "Alternator": strategy(0, 0, 0, 1, 1),
    # the guarantee of a zero-determinant strategy holds in the long run; an extortioner that opened with
    # cooperation could lose the first round, so it opens with a defection
    "Extortioner": zero_determinant(chi=3, baseline=P, first=0),
    "Generous ZD": zero_determinant(chi=3, baseline=R, first=1),
}


def with_noise(p, noise):
    return p * (1 - noise) + (1 - p) * noise


def outcome_weights(p, q, w, noise=0.0):
    # the expected discounted share of rounds that end in each outcome, from p's side
    p, q = with_noise(p, noise), with_noise(q, noise)
    first = np.array([p[0] * q[0], p[0] * (1 - q[0]), (1 - p[0]) * q[0], (1 - p[0]) * (1 - q[0])])
    moves = np.zeros((4, 4))
    for state in range(4):
        mine, theirs = p[1 + state], q[1 + SWAP[state]]
        moves[state] = [mine * theirs, mine * (1 - theirs), (1 - mine) * theirs, (1 - mine) * (1 - theirs)]
    # rounds 0, 1, 2, ... weighted by w^t and normalised: (1 - w) first (I - w M)^-1
    return (1 - w) * first @ np.linalg.inv(np.eye(4) - w * moves)


def payoffs(p, q, w, noise=0.0):
    share = outcome_weights(p, q, w, noise)
    return float(share @ MINE), float(share @ THEIRS)


def tournament(names, w, noise=0.0):
    # Axelrod's round robin: everyone against everyone, itself included; the score is the mean per round
    table = np.zeros((len(names), len(names)))
    for i, a in enumerate(names):
        for j, b in enumerate(names):
            table[i, j] = payoffs(STRATEGIES[a], STRATEGIES[b], w, noise)[0]
    return table
