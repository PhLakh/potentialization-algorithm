# replicator.py - replicator equation code
import numpy as np
from misc import actions_from_idx, normalize, rk4_step
from functools import partial


class Replicator:
    def __init__(self, game_m, pot_m, starting_policies=None):
        self.game_utility_matrix = np.array(game_m)
        if (len(self.game_utility_matrix.shape) - 1) != self.game_utility_matrix.shape[-1]:
            raise ValueError("Number of utilities does not match number of players in the game matrix")
        self.num_players = self.game_utility_matrix.shape[-1]
        self.num_actions = self.game_utility_matrix.shape[:-1]
        self.starting_policies = starting_policies
        self.game_utilities = self.game_utility_matrix.flatten().reshape(-1, self.num_players)
        self.norm_game_utilities = np.apply_along_axis(normalize, 0, self.game_utilities.flatten())
        self.norm_game_utilities = self.norm_game_utilities.reshape(-1, self.num_players)

        self.pot_utility_matrix = np.array(pot_m)
        if (len(self.pot_utility_matrix.shape) - 1) != self.pot_utility_matrix.shape[-1]:
            raise ValueError("Number of utilities does not match number of players in the potential matrix")
        self.pot_utilities = self.pot_utility_matrix.flatten().reshape(-1, self.num_players)
        self.norm_pot_utilities = np.apply_along_axis(normalize, 0, self.pot_utilities.flatten())
        self.norm_pot_utilities = self.norm_pot_utilities.reshape(-1, self.num_players)

        self.current_player_policies = None
        self.pol_hist, self.dist_hist, self.mov_f_hist, self.exp_f_hist = None, None, None, None

    def init_starting_policies(self, policies=None):
        if policies is not None:
            self.starting_policies = np.array(policies)
            self.current_player_policies = np.copy(self.starting_policies)
            return
        policies = np.empty(self.num_players, dtype=np.ndarray)
        for player in range(self.num_players):
            policies[player] = np.ones(self.num_actions[player]) / self.num_actions[player]
        self.starting_policies = policies
        self.current_player_policies = np.copy(self.starting_policies)

    def execute_run(self, step_size=1e-3, max_steps=int(1e6), conv_steps=int(1e4), conv_delta=1e-6, pot_run=False):
        self.current_player_policies = np.copy(self.starting_policies)
        self.pol_hist, self.dist_hist, self.mov_f_hist, self.exp_f_hist = [], [], [], []
        conv_step = 0
        max_dist = conv_delta
        convergent = False
        for steps in range(max_steps):
            self.pol_hist.append(self.current_player_policies)
            conv_step += 1 if max_dist < conv_delta else -conv_step
            if conv_step >= conv_steps:
                convergent = True
                break
            f_dxdt = partial(self.calc_dxdt, utilities=self.norm_pot_utilities if pot_run else self.norm_game_utilities)
            self.current_player_policies = rk4_step(self.current_player_policies, step_size, f_dxdt)
            self.dist_hist.append(np.vectorize(np.linalg.norm)(self.current_player_policies - self.pol_hist[-1]))
            max_dist = np.max(self.dist_hist[-1])
        return convergent

    def calc_dxdt(self, policies, utilities):
        exp_f, mov_f = self.calc_fitness_values(policies, utilities)
        if len(self.pol_hist) != len(self.mov_f_hist):
            self.mov_f_hist.append(mov_f)
            self.exp_f_hist.append(exp_f)
        dxdt = np.empty(self.num_players, dtype=np.ndarray)
        for player in range(self.num_players):
            dxdt[player] = policies[player] * (mov_f[player] - exp_f[player])
        return dxdt

    def calc_fitness_values(self, policies, utilities):
        mov_f = np.empty(self.num_players, dtype=np.ndarray)
        for player in range(self.num_players):
            mov_f[player] = np.zeros(self.num_actions[player])
        for idx, u in enumerate(utilities):
            actions = actions_from_idx(idx, self.num_actions)
            for player in range(self.num_players):
                prob = 1
                for player_idx, action in enumerate(actions):
                    if player_idx == player:
                        continue
                    prob *= policies[player_idx][action]
                mov_f[player][actions[player]] += prob * u[player]
        exp_f = np.empty(self.num_players, dtype=float)
        for player in range(self.num_players):
            exp_f[player] = np.dot(policies[player], mov_f[player])
        return exp_f, mov_f
