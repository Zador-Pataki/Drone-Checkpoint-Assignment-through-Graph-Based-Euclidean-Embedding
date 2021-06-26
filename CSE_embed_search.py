import numpy as np
from time import time
from tqdm import tqdm, trange

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pathlib import Path

from world import World
from simulation import Simulation
from controller import Controller

# STRATEGITES TO INVESTIGATE
# 1: recluster each iter, reassign after reclsutering
# 2: recluster only if clsuter is empty and reassign all drones to clusters and immediatley reassign drones to
#    ckeckpoints in new clusters
# 3: recluster only if clsuter is empty and reassign all drones to clusters. However, only reassign drone to new
#    checkpoint if it reached previously assigned checkpoint. (catch: what if drone A assigned to cluster 1 is still
#    assigned to only checkpoint in cluster 2? Then drone B has no free target. Best options: reassign drone A or have
#    both drone A and B target checkoint in cluster 2. Latter is easiest to program.
# 4: recluster if drone reached a checkpoint (when number of checkpoints changes) but only reassign drone that reached checkpoint
cse_strategy_list=[2, 3, 4]

#np.random.seed(21)

# EXPERIMENTAL SETUP
#   # STATIC
world_size = [1000,1000,1000]
view_radius = 600
speed=10
dt=3
assignment_group='spectral_clustering'
assignment_strategy='CSE'
n_tests = 1 # NUMBER OF TEST PER CSE_STRATEGY X N_AGENTS X N_CHECKPOINTS COMBINATION
finish_process = False # WE ONLY MEASURE TIME UNTIL N_AGENTS == N_CHECKPOINTS

#   # DYNAMIC
n_agents_list=[3,10,50,200] # NUMBER OF AGENTS WE WILL EXPERIMENT WITH
n_checkpoints_list=[500,1000,1500] # NUMBER OF CHECKPOINTS WE WILL EXPERIMENT WITH


csv_name = "experiments/experiment_labels_strats_"
for i in cse_strategy_list:
    csv_name += str(i)
    csv_name += '_'

csv_name += "n_agents_"
for i in n_agents_list:
    csv_name += str(i)
    csv_name += '_'
csv_name += '.csv'
print(csv_name)

run_experiments=False



if run_experiments:
    experimental_results = np.zeros((len(cse_strategy_list), len(n_agents_list)*len(n_checkpoints_list)))

    experiment_labels = []
    for n_agents in n_agents_list:
        for n_checkpoints in n_checkpoints_list:
            experiment_labels.append('n_agents: ' + str(n_agents) + ' n_checkpoints: ' + str(n_checkpoints))

    for i, cse_strategy in enumerate(cse_strategy_list):
        strategy_iter = 0
        for n_agents in n_agents_list:
            for n_checkpoints in n_checkpoints_list:
                time_list = []
                for _ in trange(n_tests):
                    world = World(n_agents=n_agents, n_checkpoints=n_checkpoints, world_size=world_size, view_radius=view_radius)
                    controller = Controller(assignment_group=assignment_group, assignment_strategy=assignment_strategy, cse_strategy=cse_strategy)
                    simulation = Simulation(speed=speed, dt=dt, finish_process=finish_process)

                    simulation.simulate(world, controller)
                    time_list.append(simulation.time_elapsed_list[-1])
                  

                experimental_results[i,strategy_iter] = np.mean(time_list)
                strategy_iter += 1

    experimental_results_df = pd.DataFrame(experimental_results,index=cse_strategy_list)
    experimental_results_df.to_csv(csv_name, header=experiment_labels)
i = 0
pd.set_option('display.max_columns',600)
pd.set_option('display.width',600)
for cse_strategy in cse_strategy_list:
    j = 0
    for n_agents in n_agents_list:
        path = 'experiments/experiment_labels_strats_'+str(cse_strategy)+'_n_agents_' + str(n_agents)+'_.csv'
        if j == 0: 
            df_i = pd.read_csv(path, index_col=0)

        else:
            df_i = pd.concat([df_i, pd.read_csv(path, index_col=0)], axis=1)
        j += 1
    if i == 0: 
        df = df_i
    else: df = pd.concat([df, df_i], axis=0)
    i += 1

print(df)
print(df.to_numpy())
times = df.to_numpy()
print(np.argmin(times, axis=0))
compared = times / times[1,:][np.newaxis,:]
print(compared)
print(np.mean(compared, axis=1))


