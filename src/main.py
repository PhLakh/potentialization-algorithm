# main.py - simulation driver code
from misc import gen_rng_game, to_multiplayer, gen_starting_policies
from replicator import Replicator
from potential import Potential
from datetime import timedelta
import numpy as np
import _pickle
import bz2
import json
import time
import os

params = {
    "num_games": 1000,
    "player_moves": (4, 4, 4),
    "run_params": {
        "step_size": 1e-2,
        "max_steps": int(1e5),
        "conv_steps": int(1e3),
        "conv_delta": 1e-10
    }
}

data_path = os.path.join(os.path.pardir, "data")
if not os.path.exists(data_path):
    os.makedirs(data_path)
    print(f'[MKDIR]\t{os.path.abspath(data_path)}')

game_dir = "x".join([str(m) for m in params["player_moves"]])
game_type_path = os.path.join(data_path, game_dir)
if not os.path.exists(game_type_path):
    os.makedirs(game_type_path)
    print(f'[MKDIR]\t{os.path.abspath(game_type_path)}')

run_entries = os.listdir(game_type_path)
existing_runs = [int(e.split("_")[1]) for e in run_entries if os.path.isdir(os.path.join(game_type_path, e))]
run_path = os.path.join(game_type_path, f'run_{str(int(np.max(existing_runs, initial=0) + 1))}')
os.makedirs(run_path)
print(f'[MKDIR]\t{os.path.abspath(run_path)}')

param_path = os.path.join(run_path, "_params.json")
with open(param_path, "w") as file:
    file.write(json.dumps(params, indent=4))
    print(f'[SAVE]\t{os.path.abspath(param_path)}')

print("-------------------------------------------------------")
start_t = time.time()
for game_num in range(1, params["num_games"] + 1):
    print(f'[GAME NUM] {game_num}/{params["num_games"]}')
    game_data = {}
    G = gen_rng_game(params["player_moves"], reward_range=(0, np.prod(params["player_moves"])))
    game_data["game_matrix"] = G
    P = Potential(G)
    P.remove_weak_improvement_cycles()
    P.create_potential_matrix()
    game_data["pot_matrix"] = P.potential_matrix
    game_data["pot_const"] = True if len(np.unique(P.potential_matrix.flatten())) == 1 else False
    R = Replicator(G, to_multiplayer(P.potential_matrix))
    R.init_starting_policies(gen_starting_policies(params["player_moves"]))
    print(P.potential_matrix)

    _t1 = time.time()
    print("[POT RUN]")
    game_data["pot_conv"] = R.execute_run(**params["run_params"], pot_run=True)
    game_data["pot_pol_hist"] = R.pol_hist.copy()
    game_data["pot_dist_hist"] = R.dist_hist.copy()
    game_data["pot_mov_f_hist"] = R.mov_f_hist.copy()
    game_data["pot_exp_f_hist"] = R.exp_f_hist.copy()
    _t2 = time.time()
    print(f'\ttime:\t{timedelta(seconds=round(_t2 - _t1))}')
    print(f'\tsteps:\t{len(game_data["pot_pol_hist"])}')
    print(f'\tconv:\t{game_data["pot_conv"]}')

    _t1 = time.time()
    print("[GAME RUN]")
    game_data["game_conv"] = R.execute_run(**{**params["run_params"], "max_steps": len(game_data["pot_pol_hist"])}, pot_run=False)
    game_data["game_pol_hist"] = R.pol_hist.copy()
    game_data["game_dist_hist"] = R.dist_hist.copy()
    game_data["game_mov_f_hist"] = R.mov_f_hist.copy()
    game_data["game_exp_f_hist"] = R.exp_f_hist.copy()
    _t2 = time.time()
    print(f'\ttime:\t{timedelta(seconds=round(_t2 - _t1))}')
    print(f'\tsteps:\t{len(game_data["game_pol_hist"])}')
    print(f'\tconv:\t{game_data["game_conv"]}')

    game_path = os.path.join(run_path, f'game_{game_num}.pbz2')
    with bz2.BZ2File(game_path, "wb") as file:
        _pickle.dump(game_data, file)
        print(f'[SAVE]\t{os.path.abspath(game_path)}')

    end_t = time.time()
    print("[TIME]")
    print(f'\telapsed:\t{timedelta(seconds=round(end_t - start_t))}')
    print(f'\ttotal est:\t{timedelta(seconds=round((end_t - start_t) * params["num_games"] / game_num))}')
    print("-------------------------------------------------------")
