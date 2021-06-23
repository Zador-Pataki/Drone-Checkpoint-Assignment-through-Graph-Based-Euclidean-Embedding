import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import SpectralClustering
from time import sleep, time
import matplotlib.pyplot as plt
from scipy.optimize import linprog


class Controller:
    def __init__(self, assignment_group, assignment_strategy, cse_strategy):
        self.assignment_group = assignment_group
        self.assignment_strategy = assignment_strategy
        self.cse_strategy = cse_strategy
        if not self.assignment_group == 'points':
            self.cluster_assignment = []

    def get_directions(self, world_object):
        """
        RETURNS THE DIRECTIONS IN WHICH EACH DRONE SHOULD PROGRESS
        :param world_object: object containing all world information
        :return: directions
        """
        self.set_assignment(world_object)
        target_coord = np.zeros((world_object.n_agents, 3))
        for i in range(world_object.n_agents):
            target_coord[i,:] = world_object.checkpoints_coord[world_object.agents_dict[i]['assigned_idx'], :]
        directions = target_coord - world_object.agents_coord
        directions = normalize(directions, axis=1)
        return directions

    def get_checkpoint_cluster_labels(self, D, assignment_group, n_clusters):
        if assignment_group == 'spectral_clustering':
            SC = SpectralClustering(n_clusters=n_clusters, affinity="precomputed")

            try:labels_SC = SC.fit_predict(D)
            except:
                plt.imshow(D)
                plt.colorbar()
                plt.show()
                print('error')

            return labels_SC

    def get_permute_idx(self, labels):
        perm=[]
        for j in range(labels.max()+1):
            for i in range(len(labels)):
                if labels[i] == j:
                    perm.append(i)
        return perm

    def get_permuted_matrix(self, matrix, perm):
        permuted_matrix = np.zeros(matrix.shape)
        permuted_matrix[:, :] = matrix[perm, :]
        permuted_matrix[:, :] = permuted_matrix[:, perm]
        return permuted_matrix

    def get_assignment_probabilities(self, X_agents, X_checkpoints):
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
        if n>m:
            A_ineq = -A_ineq
            b_ineq = -b_ineq


        costs = np.linalg.norm(X_agents[:,np.newaxis,:]-X_checkpoints[np.newaxis,:,:], axis=2).flatten()[np.newaxis,:]

        P = linprog(costs, A_eq=A_eq, b_eq=b_eq, A_ub=A_ineq, b_ub=b_ineq, bounds=l_u, method='highs-ds').x
        P= np.reshape(P,(n,m))

        return P

    def get_assignments_euclidean(self, X_agents, X_checkpoints):
        """
        GET ASSIGNMENTS FOR EACH AGENT
        :param X_agents:
        :param X_checkpoints:
        :return:
        """
        P = self.get_assignment_probabilities(X_agents, X_checkpoints)
        assignments = np.argmax(P, axis=1)
        return assignments

    def set_assignment(self, world_object):
        """
        RETURNS THE THE INDEX OF THE CHECKPOINTS TO WHICH THE DRONES ARE ASSIGNED
        :param agents_coord: coordinates of each drone
        :param checkpoints_coord: coordinates of each checkpoint
        :return: assignments
        """
        assignments=[]
        if self.assignment_group == 'points' or world_object.n_agents>=len(world_object.remaining_checkpoints):
            if self.assignment_strategy == 'random':
                free_agents = 0
                for i in range(world_object.n_agents):
                    if world_object.agents_dict[i]['assigned_idx'] is None:
                        free_agents+=1
                if free_agents > 0:
                    for i in world_object.remaining_checkpoints:
                        assign = True
                        for j in range(world_object.n_agents):
                            if world_object.agents_dict[j]['assigned_idx']:
                                if i == world_object.agents_dict[j]['assigned_idx']:
                                    assign = False
                        if assign: assignments.append(i)
                    if len(assignments)>0:
                        if free_agents > len(assignments):
                            for i in range(int(free_agents/len(assignments))):
                                assignments_ = assignments.copy()
                                np.random.shuffle(assignments_)
                                assignments = assignments + assignments_
                        assignments = assignments[:free_agents]
                        np.random.shuffle(assignments)
                        update=0
                        for i in range(world_object.n_agents):
                            if world_object.agents_dict[i]['assigned_idx'] is None:
                                world_object.agents_dict[i]['assigned_idx'] = assignments[update]
                                update+=1
                    else:
                        for i in range(world_object.n_agents):
                            if world_object.agents_dict[i]['assigned_idx'] is None:
                                world_object.agents_dict[i]['assigned_idx'] = np.random.choice(world_object.remaining_checkpoints)


                # for i in range(int(world_object.agents_coord.shape[0]/len(world_object.remaining_checkpoints))+1):
                #     idx = np.arange(len(world_object.remaining_checkpoints))
                #     np.random.shuffle(idx)
                #     assignments = assignments + idx.tolist()
                # assignments = assignments[:world_object.agents_coord.shape[0]]
                # np.random.shuffle(assignments)
                # assignments = list(np.array(world_object.remaining_checkpoints)[assignments])

            if self.assignment_strategy == 'CSE':
                X_embed = world_object.get_CSE(world_object.get_distance_matrix(world_object.get_adjacency_matrix()), 5)
                X_embed_agents = X_embed[:world_object.n_agents, :]
                X_embed_checkpoints = X_embed[world_object.n_agents:, :]
                assignments = self.get_assignments_euclidean(X_embed_agents, X_embed_checkpoints)
                assignments = list(np.array(world_object.remaining_checkpoints)[assignments])
                for i in range(world_object.n_agents):
                    world_object.agents_dict[i]['assigned_idx'] = assignments[i]

        elif self.assignment_group == 'spectral_clustering':
            recluster = False
            for i in range(world_object.n_agents):
                if not world_object.agents_dict[i]['cluster_idx']:
                    recluster = True
            if recluster:
                D = world_object.get_distance_matrix(world_object.get_adjacency_matrix())
                D_copy = D.copy()
                D_copy = np.nan_to_num(D_copy, posinf=0)
                D = np.nan_to_num(D, posinf=D_copy.max()*6)
                D_checkpoints = D[world_object.n_agents:, world_object.n_agents:]
                labels_SC = self.get_checkpoint_cluster_labels(D_checkpoints, assignment_group=self.assignment_group, n_clusters=world_object.n_agents)

                # perm = self.get_permute_idx(labels_SC)
                # permuted_D = self.get_permuted_matrix(D_checkpoints, perm)
                # world_object.save_distance_plot(world_object.get_distance_matrix(world_object.get_adjacency_matrix())[world_object.n_agents:, world_object.n_agents:])
                # idx = np.nonzero(permuted_D > D_copy.max())
                # permuted_D[idx]=np.inf
                # world_object.save_distance_plot(permuted_D, boundaries='clusters', cluster_labels=labels_SC)# DISPLAY CLUSTERING

                if self.assignment_strategy == 'random':
                    idx = np.arange(world_object.n_agents)
                    np.random.shuffle(idx)

                    for i in np.unique(labels_SC):
                        class_idx = list(np.nonzero(labels_SC==i)[0])
                        world_object.agents_dict[idx[i]]['cluster_idx']=list(np.array(world_object.remaining_checkpoints)[class_idx])

                elif self.assignment_strategy == 'CSE':
                    X_embed = world_object.get_CSE(world_object.get_distance_matrix(world_object.get_adjacency_matrix()), 5)
                    X_embed_agents = X_embed[:world_object.n_agents, :]
                    X_embed_checkpoints = X_embed[world_object.n_agents:, :]
                    cluster_centers = np.zeros((world_object.n_agents, X_embed.shape[1]))
                    class_idxs = []
                    for i in np.unique(labels_SC):
                        class_idx = list(np.nonzero(labels_SC==i)[0])
                        class_idxs.append(class_idx)
                        X_embed_checkpoints_i = X_embed_checkpoints[class_idx, :]
                        cluster_centers[i,:] = np.sum(X_embed_checkpoints_i, axis=0)/X_embed_checkpoints_i.shape[0]

                    cluster_assignments = self.get_assignments_euclidean(X_embed_agents, cluster_centers)
                    for i in range(world_object.n_agents):
                        world_object.agents_dict[i]['cluster_idx'] = list(np.array(world_object.remaining_checkpoints)[class_idxs[cluster_assignments[i]]])

            if self.assignment_strategy == 'random':
                for i in range(world_object.n_agents):
                    if not world_object.agents_dict[i]['assigned_idx']:
                        assignments.append(np.random.choice(world_object.agents_dict[i]['cluster_idx']))
                    else: assignments.append(world_object.agents_dict[i]['assigned_idx'])
                for i in range(world_object.n_agents):
                    world_object.agents_dict[i]['assigned_idx'] = assignments[i]
            elif self.assignment_strategy == 'CSE':
                for i in range(world_object.n_agents):
                    if not world_object.agents_dict[i]['assigned_idx']:# or recluster:
                        distances=world_object.agents_coord[i][np.newaxis,:]-world_object.checkpoints_coord[world_object.agents_dict[i]['cluster_idx']]
                        distances = np.linalg.norm(distances, axis=1)
                        assignment = np.argmin(distances)
                        assignment = world_object.agents_dict[i]['cluster_idx'][assignment]
                        world_object.agents_dict[i]['assigned_idx'] = assignment


