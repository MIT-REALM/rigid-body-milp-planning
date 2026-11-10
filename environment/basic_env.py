import yaml
import itertools
import time

import numpy as np

import shapely
import shapely.geometry as geom


from utils.config import NUM_AGENT_EDGE_DIV
from utils.geometry_operation import get_H_rep_from_V
from environment.predefined_2Denv import *
from environment.predefined_3Denv import *


class BaseEnvironment:
	def __init__(self, agent, c_length, radius, env_name, bounds):
		self.cartesian_dim = 3 if "3D" in env_name else 2

		# agent representations
		self.agent = agent  # shapely Polygon with only vertex
		self.agent_dense_edge_array = self._create_dense_agent()  # (num_of_edges, sample_num, cart_dim)

		self.obstacles_V = []  # V-rep of polytope obstacles
		self.obstacles_H = {'A': [], 'b': []}  # H-rep of polytope obstacles, Ax<=b
		self.obstacles_map = {}  # dict of 'Polytope' type obstacles
		self.bounds = bounds

		# fill in obstacles_V and obstacles_map
		self.env_name = env_name
		if ".yaml" in env_name:
			assert self._load_from_yaml_file(env_name)
		elif self.cartesian_dim == 2:
			self._create_predefined_2Denv(env_name, c_length)
		else:
			assert self.cartesian_dim == 3
			self._create_predefined_3Denv(env_name, c_length)

		if self.cartesian_dim == 2:
			self._parse_bounds_as_obstacles()
		# fill in obstacles_H
		self._get_obstacle_H_rep()

		self.point_radius = 0.02
		self.end_region_radius = radius
	# 	# self.n_dims = len(self.bounds.keys())

	# # @property
	# # def bounds(self):
	# #    return self.bounds
	#
	# # def add_obstacles(self, obstacles):
	# # 	self.obstacles = self.obstacles + obstacles
	# # 	self._calculate_scene_dimensions()

	############################## PREDEFINED ENV ################################
	def _create_predefined_2Denv(self, env_name, c_length):
		if env_name == 'corner2D':
			self.start_point, self.end_point, self.obstacles_V = corner_2Denv(
				bounds=self.bounds, tunnel_width=c_length)
		elif env_name == 'bugtrap2D':
			self.start_point, self.end_point, self.obstacles_V = bugtrap_2Denv(
				bounds=self.bounds, tunnel_width=c_length)
		elif env_name == 'chull-2D':
			self.start_point, self.end_point, self.obstacles_V = chull_2Denv(
				bounds=self.bounds, tunnel_width=c_length)
		elif env_name == 'infeas1-2D':
			self.start_point, self.end_point, self.obstacles_V = infeas1_2Denv(
				bounds=self.bounds)
		elif env_name == 'STL2-2D':
			self.start_point, self.end_point, self.obstacles_V = STL2_2Denv(
				bounds=self.bounds, tunnel_width=c_length)
		elif env_name == 'STL3-2D':
			self.start_point, self.end_point, self.obstacles_V = STL3_2Denv(
				bounds=self.bounds, tunnel_width=c_length)
		else:
			raise NotImplementedError(f"2D Environment {env_name} not implemented")

	def _create_predefined_3Denv(self, env_name, c_length):
		# https://github.com/mikedh/trimesh/blob/main/examples/offscreen_render.py
		if env_name == 'easy-3D':
			self.start_point, self.end_point, self.obstacles_V = easy_3Denv(
				bounds=self.bounds, tunnel_width=c_length)
		elif env_name == 'dense-3D':
			self.start_point, self.end_point, self.obstacles_V = dense_3Denv(
				bounds=self.bounds, tunnel_width=c_length)
		elif env_name == 'peg-3D':
			self.start_point, self.end_point, self.obstacles_V = peg_3Denv(
				bounds=self.bounds, tunnel_width=c_length)
		elif env_name == 'demo-3D':
			self.start_point, self.end_point, self.obstacles_V = demo_3Denv(
				bounds=self.bounds, tunnel_width=c_length)
		else:
			raise NotImplementedError(f"3D Environment {env_name} not implemented")


	############################## YAML PARSING ################################

	# def save_to_yaml(self, yaml_file):
	# 	yaml_dict = {}
	# 	obstacles = {}
	# 	for i, ob in enumerate(self.obstacles_V):
	# 		ob_dict = {}
	# 		ob_dict['shape'] = 'polygon'
	# 		ob_dict['corners'] = [list(t) for t in list(ob['geom'].boundary.coords)]
	# 		ob_name = "obstacle%.4d" % i
	# 		obstacles[ob_name] = ob_dict
	# 	yaml_dict['environment'] = {'obstacles': obstacles}
	#
	# 	f = open(yaml_file, 'w')
	# 	f.write(yaml.dump(yaml_dict, default_flow_style=None))
	# 	f.close()

	def _load_from_yaml_file(self, yaml_file):
		if "simple10" in yaml_file:
			self.start_point = (-1, 0, 0)
			self.end_point = (10.5, 5.5, 0)
		elif 'challenging600' in yaml_file:
			self.start_point = (-2, -1, 0)
			self.end_point = (12.5, 3.5, 0)
		else:
			raise NotImplementedError(f"YAML Environment {yaml_file} not implemented")

		f = open(yaml_file)
		self.data = yaml.safe_load(f)
		f.close()
		return self._parse_yaml_data(self.data)

	def _parse_yaml_data(self, data):
		if 'environment' in data:
			env = data['environment']
			self._parse_yaml_obstacles(env['obstacles'])
			# self.parse_yaml_features(env['features'])
			return True
		else:
			return False

	def _parse_yaml_obstacles(self, obstacles):
		self.obstacles_V = []
		self.obstacles_map = {}
		for name, description in obstacles.items():
			# Double underscore not allowed in region names.
			if name.find("__") != -1:
				raise Exception("Names cannot contain double underscores.")
			if description['shape'] == 'polygon':
				parsed = self._parse_polygon(description)
			else:
				raise Exception("not a polygon")
			if not parsed.is_valid:
				raise Exception("%s is not valid!" % name)
			self.obstacles_V.append({'name': name, 'geom': parsed})
			self.obstacles_map[name] = parsed

	def _parse_polygon(self, description):
		_points = description['corners']
		for points in itertools.permutations(_points):
			polygon = geom.Polygon(points)
			if polygon.is_valid:
				return polygon

	############################## PREPROCESS ################################

	def _create_dense_agent(self):
		# return: dense edge array of agent, (num_of_edges, sample_num, cart_dim)
		if self.cartesian_dim == 2:
			boundary_coords = np.array(self.agent.boundary.coords)
			edge_array = np.concatenate([boundary_coords[1:, np.newaxis], boundary_coords[:-1, np.newaxis]], axis=1)
		else:
			edge_idx = self.agent.edges_unique
			edge_array = np.array(self.agent.vertices)[edge_idx]  # (num_of_edges, 2, cart_dim)

		sampled_edge_array = np.linspace(edge_array[:, 0, :], edge_array[:, 1, :], NUM_AGENT_EDGE_DIV, endpoint=True, axis=1)
		return sampled_edge_array

	def _parse_bounds_as_obstacles(self):
		polygon_points = [
			[[self.bounds['x'][0], self.bounds['y'][0]], [self.bounds['x'][0], self.bounds['y'][1]],
			 [self.bounds['x'][0] - 1, self.bounds['y'][0]], [self.bounds['x'][0], self.bounds['y'][0]]],  # left
			[[self.bounds['x'][1], self.bounds['y'][0]], [self.bounds['x'][1], self.bounds['y'][1]],
			 [self.bounds['x'][1] + 1, self.bounds['y'][0]], [self.bounds['x'][1], self.bounds['y'][0]]],  # right
			[[self.bounds['x'][0], self.bounds['y'][0]], [self.bounds['x'][1], self.bounds['y'][0]],
			 [self.bounds['x'][0], self.bounds['y'][0] - 1], [self.bounds['x'][0], self.bounds['y'][0]]],  # bottom
			[[self.bounds['x'][0], self.bounds['y'][1]], [self.bounds['x'][1], self.bounds['y'][1]],
			 [self.bounds['x'][0], self.bounds['y'][1] + 1], [self.bounds['x'][0], self.bounds['y'][1]]]  # top
		]
		polygon_center_list = [
			[self.bounds['x'][0] - 0.5, (self.bounds['y'][0] + self.bounds['y'][1]) / 2, 0],
			[self.bounds['x'][1] + 0.5, (self.bounds['y'][0] + self.bounds['y'][1]) / 2, 0],
			[(self.bounds['x'][0] + self.bounds['x'][1]) / 2, self.bounds['y'][0] - 0.5, 0],
			[(self.bounds['x'][0] + self.bounds['x'][1]) / 2, self.bounds['y'][1] + 0.5, 0],
		]
		polygon_size_list = [
			[1, self.bounds['y'][1] - self.bounds['y'][0], 1],
			[1, self.bounds['y'][1] - self.bounds['y'][0], 1],
			[self.bounds['x'][1] - self.bounds['x'][0], 1, 1],
			[self.bounds['x'][1] - self.bounds['x'][0], 1, 1],
		]
		for i, (points, polygon_center, polygon_size) in enumerate(zip(polygon_points, polygon_center_list, polygon_size_list)):
			polygon = geom.Polygon(points)
			self.obstacles_V.append({'name': f'bounds_{i}', 'geom': polygon, 'drake_center': polygon_center, 'drake_size': polygon_size})
			self.obstacles_map['bounds'] = polygon

	def _get_obstacle_H_rep(self):
		for obs in self.obstacles_V:
			if self.cartesian_dim == 2:
				verts = np.array(obs['geom'].exterior.coords)[:-1, :]
			else:
				verts = np.array(obs['geom'].vertices)
			H_A, H_b = get_H_rep_from_V(verts)
			self.obstacles_H['A'].append(H_A)
			self.obstacles_H['b'].append(H_b)

