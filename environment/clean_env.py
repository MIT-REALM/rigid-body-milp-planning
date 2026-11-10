from .basic_env import BaseEnvironment

import numpy as np
import networkx as nx
import shapely
import shapely.geometry as geom

import scipy.spatial as spatial

import matplotlib
import matplotlib.pyplot as plt


class CleanEnv(BaseEnvironment):
	"""
	'clean' in the sense that it does not import python-fcl, thus can be used with parallization
	"""

	def __init__(self, agent, c_length, radius, env_name=None, bounds=None):
		super().__init__(agent, c_length, radius, env_name, bounds)

	def sample_outside(self, max_sample_num, additional_obs: tuple, sample_num=5):
		"""
		sample points outside of obstacles and some configured area
		:param max_sample_num:
		:return:
		"""
		# sample points in the whole space
		if self.cartesian_dim == 2:
			xs = np.random.uniform(self.bounds['x'][0], self.bounds['x'][1], max_sample_num)
			ys = np.random.uniform(self.bounds['y'][0], self.bounds['y'][1], max_sample_num)
			points = np.vstack((xs, ys)).T
		elif self.cartesian_dim == 3:
			xs = np.random.uniform(self.bounds['x'][0], self.bounds['x'][1], max_sample_num)
			ys = np.random.uniform(self.bounds['y'][0], self.bounds['y'][1], max_sample_num)
			zs = np.random.uniform(self.bounds['z'][0], self.bounds['z'][1], max_sample_num)
			points = np.vstack((xs, ys, zs)).T
		else:
			raise NotImplementedError(f"Dimension {self.cartesian_dim} not implemented")

		# remove points inside obstacles
		for A, b in zip(self.obstacles_H['A'], self.obstacles_H['b']):
			indices = np.where(np.all(A @ points.T <= b.reshape(-1, 1), axis=0))[0]
			points = np.delete(points, indices, axis=0)

		# remove points inside additional obstacles
		addtional_A_list, addtional_b_list = additional_obs
		for A, b in zip(addtional_A_list, addtional_b_list):
			A = np.array(A)
			b = np.array(b)
			indices = np.where(np.all(A @ points.T <= b.reshape(-1, 1), axis=0))[0]
			points = np.delete(points, indices, axis=0)

		distance_matrix = spatial.distance.squareform(spatial.distance.pdist(points, metric='euclidean'))
		triu_indices = np.triu_indices(distance_matrix.shape[0], k=0)
		distance_matrix[triu_indices] = np.inf

		distance_array = np.min(distance_matrix, axis=0)
		sorted_indices = np.flip(np.argsort(distance_array))
		return points[sorted_indices[:min(sample_num, points.shape[0])], :]

	def sample_from_vis_graph(self, vis_G: nx.Graph, sample_num):
		"""
		sample points from the visible graph for iris initialization
		"""
		edge_list = list(vis_G.edges(data=True))
		# if len(edge_list) > 2048:
		# 	seed_pos_array = np.array([edge_list[edge_idx][2]['ooc_coverage_points'][0] for edge_idx in np.random.choice(len(edge_list), 2048)])
		# else:
		# 	seed_pos_array = np.array([edge[2]['ooc_coverage_points'][0] for edge in edge_list])
		seed_pos_array = np.array([edge[2]['ooc_coverage_points'][0] for edge in edge_list])

		if seed_pos_array.shape[0] == 0:
			return np.zeros((0, self.cartesian_dim, self.cartesian_dim)), np.zeros((0, self.cartesian_dim))

		# distance_matrix = spatial.distance.squareform(spatial.distance.pdist(seed_pos_array, metric='euclidean'))
		# triu_indices = np.triu_indices(distance_matrix.shape[0], k=0)
		# distance_matrix[triu_indices] = np.inf
		# distance_array = np.min(distance_matrix, axis=0)
		# sorted_indices = np.flip(np.argsort(distance_array))

		# init_d = seed_pos_array[sorted_indices[:min(sample_num, seed_pos_array.shape[0])], :]
		init_d = seed_pos_array[:min(sample_num, seed_pos_array.shape[0]), :]
		init_C = self.point_radius * np.array([np.diag([1 for _ in range(seed_pos_array.shape[-1])]) for _ in range(init_d.shape[0])])
		return init_C, init_d


