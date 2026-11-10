"""

Probabilistic Road Map (PRM) Planner

Origin author: Atsushi Sakai (@Atsushi_twi)

"""

import math
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import KDTree

from environment import CollisionCheckingEnvironment

# parameter
N_SAMPLE = 500  # number of sample_points
N_KNN = 10  # number of edge from one sampled point

show_animation = True


class Node2D:
    """
    Node class for dijkstra search
    """

    def __init__(self, x, y, theta, cost, parent_index):
        self.x = x
        self.y = y
        self.theta = theta
        self.cost = cost
        self.parent_index = parent_index

    def __str__(self):
        return str(self.x) + "," + str(self.y) + "," +\
               str(self.cost) + "," + str(self.parent_index)


def planWithCustomPRM(env: CollisionCheckingEnvironment) -> (np.ndarray, float, float):
    samples = sample_points(env.bounds)

    if len(env.start_point) == 3:  # 2D
        start_x, start_y, start_theta = env.start_point
        goal_x, goal_y, goal_theta = env.end_point
        samples['x'] = np.concatenate((samples['x'], np.array([start_x, goal_x])))
        samples['y'] = np.concatenate((samples['y'], np.array([start_y, goal_y])))
        samples['theta'] = np.concatenate((samples['theta'], np.array([start_theta, goal_theta])))

        start_node = Node2D(start_x, start_y, start_theta, 0.0, -1)
        goal_node = Node2D(goal_x, goal_y, goal_theta, 0.0, -1)

    else:  # 3D
        assert len(env.start_point) == 4
        raise NotImplementedError("3D PRM not implemented")

    if show_animation:
        plt.plot(samples['x'], samples['y'], ".b")

    road_map = generate_road_map(samples, env._is_configuration_free)
    path = dijkstra_planning(start_node, goal_node, road_map, samples)
    return path, 0, 0


def generate_road_map(samples, coll_fn):
    road_map = []
    n_sample = len(samples['x'])
    if len(samples.keys()) == 3:  # 2D
        sample_kd_tree = KDTree(data=np.vstack((samples['x'], samples['y'], np.cos(samples['theta']), np.sin(samples['theta']))).T, copy_data=True)
        for (i, ix, iy, itheta) in zip(range(n_sample), samples['x'], samples['y'], samples['theta']):
            dists, indexes = sample_kd_tree.query([ix, iy, np.cos(itheta), np.sin(itheta)], k=n_sample)
            edge_id = []
            for ii in range(1, len(indexes)):
                if not coll_fn([ix, iy, itheta]):
                    edge_id.append(indexes[ii])
                if len(edge_id) >= N_KNN:
                    break
            road_map.append(edge_id)
    else:  # 3D
        assert len(samples.keys()) == 4
        sample_kd_tree = KDTree(data=np.vstack((samples['x'], samples['y'], samples['z'])).T, copy_data=True)
        # for (i, ix, iy) in zip(range(n_sample), sample_x, sample_y):
        #
        #     dists, indexes = sample_kd_tree.query([ix, iy], k=n_sample)
        #     edge_id = []
        #
        #     for ii in range(1, len(indexes)):
        #         if not coll_fn([ix, iy]):
        #             edge_id.append(indexes[ii])
        #
        #         if len(edge_id) >= N_KNN:
        #             break
        #
        #     road_map.append(edge_id)

    #  plot_road_map(road_map, sample_x, sample_y)

    return road_map


def dijkstra_planning(s_node, g_node, road_map, samples):
    open_set, closed_set = dict(), dict()
    open_set[len(road_map) - 2] = s_node

    path_found = True

    while True:
        if not open_set:
            print("Cannot find path")
            path_found = False
            break

        c_id = min(open_set, key=lambda o: open_set[o].cost)
        current = open_set[c_id]

        # show graph
        # if show_animation and len(closed_set.keys()) % 2 == 0:
        #     # for stopping simulation with the esc key.
        #     plt.gcf().canvas.mpl_connect(
        #         'key_release_event',
        #         lambda event: [exit(0) if event.key == 'escape' else None])
        #     plt.plot(current.x, current.y, "xg")
        #     plt.pause(0.001)

        if c_id == (len(road_map) - 1):
            print("goal is found!")
            g_node.parent_index = current.parent_index
            g_node.cost = current.cost
            break

        # Remove the item from the open set
        del open_set[c_id]
        # Add it to the closed set
        closed_set[c_id] = current

        # expand search grid based on motion model
        for i in range(len(road_map[c_id])):
            n_id = road_map[c_id][i]
            if isinstance(s_node, Node2D):
                dx = samples['x'][n_id] - current.x
                dy = samples['y'][n_id] - current.y
                d = math.hypot(dx, dy)
                node = Node2D(samples['x'][n_id], samples['y'][n_id], samples['theta'][n_id], current.cost + d, c_id)
            else:
                raise NotImplementedError("3D PRM not implemented")

            if n_id in closed_set:
                continue
            # Otherwise if it is already in the open set
            if n_id in open_set:
                if open_set[n_id].cost > node.cost:
                    open_set[n_id].cost = node.cost
                    open_set[n_id].parent_index = c_id
            else:
                open_set[n_id] = node

    if path_found is False:
        return None

    # generate final course
    if isinstance(s_node, Node2D):
        path = []
        path.append([g_node.x, g_node.y, g_node.theta])
        parent_index = g_node.parent_index
        while parent_index != -1:
            n = closed_set[parent_index]
            path.append([n.x, n.y, n.theta])
            parent_index = n.parent_index
    else:
        raise NotImplementedError("3D PRM not implemented")

    return path


def plot_road_map(road_map, sample_x, sample_y):  # pragma: no cover

    for i, _ in enumerate(road_map):
        for ii in range(len(road_map[i])):
            ind = road_map[i][ii]

            plt.plot([sample_x[i], sample_x[ind]],
                     [sample_y[i], sample_y[ind]], "-k")


def sample_points(bounds):
    max_x = bounds['x'][1]
    max_y = bounds['y'][1]
    min_x = bounds['x'][0]
    min_y = bounds['y'][0]

    samples = {}
    samples['x'] = np.random.uniform(min_x, max_x, N_SAMPLE)
    samples['y'] = np.random.uniform(min_y, max_y, N_SAMPLE)

    if 'z' in bounds:
        raise NotImplementedError("3D PRM not implemented")
        max_z = bounds['z'][1]
        min_z = bounds['z'][0]
    else:
        assert 'theta' in bounds
        max_theta = bounds['theta'][1]
        min_theta = bounds['theta'][0]
        samples['theta'] = np.random.uniform(min_theta, max_theta, N_SAMPLE)

    return samples
