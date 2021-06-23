from world import World
from simulation import Simulation
from controller import Controller
from time import time
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np

def write_experiment_file(experiment_results_dir, experiment_number, **variables):
    """
    SAVES TEXT FILE CONTAINING EXPERIMENT PARAMETERS AND RESULTS
    :param experiment_number (int)
    :param experiment_results_dir (string)
    :param variables: all variables that we want to store
    """
    text = ''
    for var in variables:
        text = text + str(var) + ': ' + str(variables[str(var)]) + '\n'
    text_file = open(experiment_results_dir + '/' + str(experiment_number) + "/experiment.txt","w")
    text_file.write(text)
    text_file.close()

def save_checkpoints_plot(experiment_results_dir, experiment_number, times, n_checkpoints):
    """
    SAVES CHECKPOINTS VS TIME PLTO
    :param experiment_results_dir (string)
    :param experiment_number (int)
    :param times (list): containing time elapsed at each increment
    :param n_checkpoints (list): containing number of checkpoints at each increment
    """
    plt.plot(times, n_checkpoints)
    plt.xlabel("Number of checkpoints left")
    plt.ylabel("Simulation time")
    plt.savefig(experiment_results_dir+'/' + str(experiment_number)+'/checkpoints_v_time.png')
    plt.close()

def save_experiment_plot(n_agents, n_checkpoints_list, assignment_group_scores, experiment_results_dir):
    for assignment_group in assignment_group_scores:
        print(len(n_checkpoints_list), len(assignment_group_scores[assignment_group]))
        plt.plot(n_checkpoints_list, assignment_group_scores[assignment_group], label=assignment_group)
    plt.xlabel("Number of checkpoints")
    plt.ylabel("Mean simulation time")
    plt.legend()
    plt.savefig(experiment_results_dir+'/' + str(experiment_number)+'/' +str(n_agents)+'.png')
    plt.close()

if __name__ == "__main__":
    # EXPERIMENT INFO
    experiment_results_dir = 'experiments'
    experiment_number = 0
    np.random.seed(2)



    # SYSTEM PARAMETERS
    n_agents_list=[4,10,50,200]
    n_checkpoints_list=[400,1000,5000]
    world_size = [5000, 5000, 5000]
    view_radius = 3000
    # CONTROLLER PARAMETERS
    assignment_group_ḷist = ['points', 'spectral_clustering']
    assignment_strategy_list = ['random']
    # SIMULATION PARAMETERS
    speed=10
    dt=1
    assignments_each_iter=False
    # NUMBER OF TESTS PER HYPERPARAMETER SET
    n_tests = 10



world = World(n_agents=20, n_checkpoints=100, world_size=[1000, 1000, 1000], view_radius=600)
#controller = Controller(assignment_group='spectral_clustering', assignment_strategy='CSE')
controller = Controller(assignment_group='spectral_clustering', assignment_strategy='CSE')
simulation = Simulation(speed=speed, dt=dt, assignments_each_iter=assignments_each_iter)

start = time()
simulation.simulate(world, controller)
#print("process time:", time()-start)

    # start = time()
    # for n_agents in n_agents_list:
    #     assignment_group_scores = {}
    #     for assignment_group in assignment_group_ḷist:
    #         assignment_group_scores[assignment_group]=[]
    #     for n_checkpoints in tqdm(n_checkpoints_list):
    #         for assignment_group in assignment_group_ḷist:
    #             #print(assignment_group_scores)
    #             for assignment_strategy in assignment_strategy_list:
    #                 scores = []
    #                 for _ in range(n_tests):
    #                     world = World(n_agents=n_agents, n_checkpoints=n_checkpoints, world_size=world_size, view_radius=view_radius)
    #                     controller = Controller(assignment_group=assignment_group, assignment_strategy=assignment_strategy)
    #                     simulation = Simulation(speed=speed, dt=dt, assignments_each_iter=assignments_each_iter)
    #
    #                     start = time()
    #                     simulation.simulate(world, controller)
    #                     scores.append(simulation.time_elapsed_list[-1])
    #                     #print("process time:", time()-start)
    #                 assignment_group_scores[assignment_group].append(np.mean(scores))
    #     save_experiment_plot(n_agents, n_checkpoints_list, assignment_group_scores, experiment_results_dir)
    #
    #
    #
    # write_experiment_file(experiment_results_dir, experiment_number,
    #                       n_agents=n_agents,
    #                       n_checkpoints=n_checkpoints,
    #                       n_world_size=world_size,
    #                       view_radius=view_radius,
    #                       assignment_group=assignment_group,
    #                       assignment_strategy=assignment_strategy,
    #                       speed=speed,
    #                       dt=dt,
    #                       assignments_each_iter=assignments_each_iter,
    #                       time_taken=simulation.time_elapsed_list[-1])
    #
    # save_checkpoints_plot(experiment_results_dir, experiment_number, simulation.time_elapsed_list, simulation.number_of_checkpoints_list)
    #







