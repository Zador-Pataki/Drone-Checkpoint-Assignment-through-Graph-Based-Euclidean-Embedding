#from world import World
#from simulation import Simulation
#from controller import Controller
from scipy.linalg import eigh
from scipy.optimize import linprog

n = 300
m = 300
import numpy as np
from time import time

X_agents = np.random.uniform(0,20, (n, 5))
X_checkpoints = np.random.uniform(0,20, (m, 5))



def get_assignment_probabilities(X_agents, X_checkpoints, method):
    n = X_agents.shape[0]
    m = X_checkpoints.shape[0]

    l_u = (0,1)

    A_eq = np.zeros((n, n*m))
    for i, x in enumerate(A_eq):
        for y in range(i*m, (i+1)*m):
            A_eq[i, y]=1
    b_eq = np.ones((n, 1))

    A_ineq = np.zeros((m, n*m))
    for i in range(m):
        for j in range(n):

            A_ineq[i, i+j*m] = 1
    b_ineq = np.ones((m,1))

    costs = np.linalg.norm(X_agents[:,np.newaxis,:]-X_checkpoints[np.newaxis,:,:], axis=2).flatten()[np.newaxis,:]

    P = linprog(costs, A_eq=A_eq, b_eq=b_eq, A_ub=A_ineq, b_ub=b_ineq, bounds=l_u, method=method)
    return P

start = time()
P = np.reshape(get_assignment_probabilities(X_agents, X_checkpoints, 'highs').x, (n,m))
print(time()-start)

start = time()
P = np.reshape(get_assignment_probabilities(X_agents, X_checkpoints, 'highs-ds').x, (n,m))
print(time()-start)

start = time()
P = np.reshape(get_assignment_probabilities(X_agents, X_checkpoints, 'highs-ipm').x, (n,m))
print(time()-start)

