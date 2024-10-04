# misc.py - miscellaneous helper functions
import numpy as np


def gen_rng_game(player_moves=(2, 2), reward_range=(0, 100)):
    if len(reward_range) != 2:
        raise ValueError("Range of rewards not specified properly")
    return np.random.randint(reward_range[0], reward_range[1], size=player_moves + tuple([len(player_moves)]))


def gen_starting_policies(player_moves=(2, 2)):
    policies = np.empty(len(player_moves), dtype=np.ndarray)
    for player in range(len(player_moves)):
        policies[player] = np.random.default_rng().dirichlet(np.ones(player_moves[player]))
    return policies


def to_multiplayer(m):
    fm = np.array(m).flatten()
    num_players = len(np.array(m).shape)
    ret = []
    for e in fm:
        ret += [e for _ in range(0, num_players)]
    return np.array(ret).reshape(np.array(m).shape + tuple([num_players]))


def idx_from_actions(actions: np.ndarray | list[int] | tuple[int], num_actions: np.ndarray | list[int] | tuple[int]):
    if np.min(np.array(num_actions) - 1 - np.array(actions)) < 0:
        raise ValueError("Action indices out of bounds")
    idx = 0
    for i, a in enumerate(actions):
        c = 1
        for j in range(i + 1, len(num_actions)):
            c *= num_actions[j]
        idx += a * c
    return idx


def actions_from_idx(idx: int, num_actions: np.ndarray | list[int] | tuple[int]):
    if idx < 0 or idx >= np.prod(np.array(num_actions)):
        raise ValueError("Index out of bounds")
    actions = []
    for i in range(len(num_actions)):
        c = 1
        for j in range(i + 1, len(num_actions)):
            c *= num_actions[j]
        actions.append(idx // c)
        idx = idx % c
    return actions


def normalize(m: np.ndarray):
    max_v = np.max(m)
    min_v = np.min(m)
    if max_v == min_v:
        return m - min_v
    return (m - min_v) / (max_v - min_v)


def rk4_step(x0, h, f):
    k1 = h * f(x0)
    k2 = h * f(x0 + 0.5 * k1)
    k3 = h * f(x0 + 0.5 * k2)
    k4 = h * f(x0 + k3)
    return x0 + (1.0 / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
