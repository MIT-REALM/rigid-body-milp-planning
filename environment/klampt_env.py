from environment.collision_checking_env import CollisionCheckingEnvironment

import PythonCDT as cdt
import shapely
import trimesh
import networkx as nx
import numpy as np
from scipy.spatial.transform import Rotation, Slerp
from klampt.plan.cspace import CSpace

ETA = 0.1

class KlamptSE2CSpace(CSpace):
	def __init__(self):
		CSpace.__init__(self)
		self.properties['euclidean'] = False

	def interpolate(self, a, b, u):
		"""Interpolate between two configurations."""
		x = (1 - u) * a[0] + u * b[0]
		y = (1 - u) * a[1] + u * b[1]
		theta = np.arctan2(np.sin((1 - u) * a[2] + u * b[2]), np.cos((1 - u) * a[2] + u * b[2]))
		return [x, y, theta]

class KlamptSE3CSpace(CSpace):
	def __init__(self):
		CSpace.__init__(self)
		self.properties['euclidean'] = False

	@staticmethod
	def interpolate(a, b, u):
		"""Interpolate between two configurations."""
		x = (1 - u) * a[0] + u * b[0]
		y = (1 - u) * a[1] + u * b[1]
		z = (1 - u) * a[2] + u * b[2]
		rs = Rotation.from_quat([[a[3], a[4], a[5], a[6]],
								 [b[3], b[4], b[5], b[6]]])
		slerp_cls = Slerp([0, 1], rs)
		q = slerp_cls(u).as_quat()
		return [x, y, z, q[0], q[1], q[2], q[3]]

	def sample(self):
		xyz = np.random.uniform(low=self.properties['minimum'][:3], high=self.properties['maximum'][:3])
		R = Rotation.random().as_quat()
		return [xyz[0], xyz[1], xyz[2], R[0], R[1], R[2], R[3]]

class KlamptCustomSE2CSpace(KlamptSE2CSpace):
	def __init__(self, bounds, obstacles_V):
		KlamptSE2CSpace.__init__(self)
		self.bounds = bounds
		self.obstacles_V = obstacles_V
		self.adjacency_graph = self._setup_triangulation_free_space()  # G_T
		self.rdmap = nx.Graph()

	def sample(self):
		w_sum = sum(self.samplers['weights'])
		probs = [(1 - ETA) * w / w_sum + ETA / len(self.samplers['weights']) for w in self.samplers['weights']]
		self.sampler_idx = np.random.choice(len(self.samplers['sampler_list']), p=probs)
		return self.samplers['sampler_list'][self.sampler_idx].sample()

	def update_sampler(self, nodes, new_nodes, new_edges):
		# update rdmap
		prev_connected_components = len(list(nx.connected_components(self.rdmap)))
		self.rdmap.add_nodes_from([tuple(new_node) for new_node in new_nodes])
		self.rdmap.add_edges_from(new_edges)
		reward = (len(list(nx.connected_components(self.rdmap))) == prev_connected_components)

		# update sampler
		prob = (1 - ETA) * self.samplers['weights'][self.sampler_idx] / sum(self.samplers['weights']) + ETA / len(self.samplers['weights'])
		self.samplers['weights'][self.sampler_idx] *= np.exp(reward * ETA / len(self.samplers['weights']) / prob) # to be updated
		for sampler, feature_point in zip(self.samplers['sampler_list'], self.samplers['feature_points']):
			sampler.update_graph(nodes, new_edges, feature_point)

	def set_up_sampler(self, feature_points):
		x_list = feature_points[0]
		y_list = feature_points[1]
		x_list[0], y_list[0] = 0, 0  # add origin
		self.samplers = {
			'sampler_list': [],
			'feature_points': [],
			'weights': [],
		}
		self.sampler_weights = []
		for x, y in zip(x_list, y_list):
			self.samplers['sampler_list'].append(WCO2DSampler(self.adjacency_graph.nodes()))
			self.samplers['feature_points'].append([x, y])
			self.samplers['weights'].append(1)

	def _setup_triangulation_free_space(self):
		G = nx.Graph()
		node_pos_dict = {}
		self.triangulation = cdt.Triangulation(cdt.VertexInsertionOrder.AS_PROVIDED,
											   cdt.IntersectingConstraintEdges.RESOLVE, 0.0)

		vert_idx_displacement = 0

		# handle boundary
		'''
		the first three vertices are super triangle, which will be removed later;
		other vertices are added as provided order.
		However, when adding the constraint edges, super vertices don't need to be explicitly considered.
		'''
		boundary_verts = [cdt.V2d(self.bounds['x'][0], self.bounds['y'][0]),
						  cdt.V2d(self.bounds['x'][0], self.bounds['y'][1]),
						  cdt.V2d(self.bounds['x'][1], self.bounds['y'][1]),
						  cdt.V2d(self.bounds['x'][1], self.bounds['y'][0])]
		self.triangulation.insert_vertices(boundary_verts)
		boundary_edges = [cdt.Edge(0 + vert_idx_displacement, 1 + vert_idx_displacement),
						  cdt.Edge(1 + vert_idx_displacement, 2 + vert_idx_displacement),
						  cdt.Edge(2 + vert_idx_displacement, 3 + vert_idx_displacement),
						  cdt.Edge(3 + vert_idx_displacement, 0 + vert_idx_displacement)]
		self.triangulation.insert_edges(boundary_edges)
		vert_idx_displacement = len(self.triangulation.vertices) - 3

		# handle obstacles
		for obs in self.obstacles_V:
			if (not 'name' in obs) or (not 'bounds' in obs['name']):
				verts = np.array(obs['geom'].exterior.coords)[:-1, :]
				verts_idx = []
				for vert_coord in verts:
					vert = cdt.V2d(vert_coord[0], vert_coord[1])
					if vert in self.triangulation.vertices:
						verts_idx.append(self.triangulation.vertices.index(vert) - 3)
					else:
						self.triangulation.insert_vertices([vert])
						verts_idx.append(vert_idx_displacement)
						vert_idx_displacement += 1
				self.triangulation.insert_edges([cdt.Edge(verts_idx[idx], verts_idx[(idx + 1) % len(verts_idx)])
												 for idx in range(len(verts_idx))])

		self.triangulation.erase_super_triangle()

		# summarize as a graph
		for triangle in self.triangulation.triangles:
			triangle_vert_idx = triangle.vertices  # list of 3
			triangle_verts = np.array(
				[[self.triangulation.vertices[idx].x, self.triangulation.vertices[idx].y] for idx in triangle_vert_idx])
			centroid = np.mean(triangle_verts, axis=0)
			if self.is_point_free(centroid):
				node_shapely = shapely.geometry.Polygon(triangle_verts)
				G.add_node(node_shapely)
				node_pos_dict[node_shapely] = centroid
		# add edges in graph
		node_list = list(G.nodes())
		for node_idx, node_i in enumerate(node_list):
			for node_j in node_list[node_idx + 1:]:
				if isinstance(shapely.intersection(node_i, node_j), shapely.geometry.LineString):
					G.add_edge(node_i, node_j)
		# from utils.plot2D import plot_graph_on_ax
		# import matplotlib.pyplot as plt
		# ax = plt.gca()
		# plot_graph_on_ax(G, node_pos_dict, plot_node_patch=True, ax=ax)
		# plt.show()
		return G

	def is_point_free(self, pos):
		assert len(pos) == 2
		x, y = pos
		for obs in self.obstacles_V:
			if obs['geom'].contains(shapely.geometry.Point(x, y)):
				return False
		return True


class WCO2DSampler:
	def __init__(self, nodes):
		self.all_nodes = list(nodes)
		self.non_iso_nodes = []

	def sample(self):
		# non_iso_nodes = [node for node in self.channel_graph.nodes() if len(list(self.channel_graph.neighbors(node))) > 1]
		# if len(non_iso_nodes) == 0:
		# 	non_iso_nodes = list(self.channel_graph.nodes())
		non_iso_nodes = self.non_iso_nodes if len(self.non_iso_nodes) > 0 else self.all_nodes
		verts_x, verts_y = non_iso_nodes[np.random.randint(len(non_iso_nodes))].boundary.xy

		s = np.random.random()
		t = np.random.random()
		in_triangle = s + t <= 1
		v0 = np.array([verts_x[0], verts_y[0]])
		u = np.array([verts_x[1], verts_y[1]]) - v0
		v = np.array([verts_x[2], verts_y[2]]) - v0
		p = v0 + s * u + t * v if in_triangle else v0 + (1 - s) * u + (1 - t) * v

		return (p[0], p[1], np.random.uniform(-np.pi, np.pi))

	def update_graph(self, nodes, new_edges, feature_point):
		# return
		for (node_i, node_j) in new_edges:
			interpolation = np.linspace(nodes[node_i], nodes[node_j], 10, endpoint=True)
			for node_idx in range(len(self.all_nodes) - 1, -1, -1):
				for inter_point in interpolation:
					x = inter_point[0] + feature_point[0] * np.cos(inter_point[2]) - feature_point[1] * np.sin(inter_point[2])
					y = inter_point[1] + feature_point[0] * np.sin(inter_point[2]) + feature_point[1] * np.cos(inter_point[2])
					if self.all_nodes[node_idx].contains(shapely.geometry.Point([x, y])):
						self.non_iso_nodes.append(self.all_nodes.pop(node_idx))
						# print(f"add {node_idx} to non_iso_nodes")
						break

class KlamptCustomSE3CSpace(KlamptSE3CSpace):
	def __init__(self, bounds, obstacles_V, env_name):
		KlamptSE3CSpace.__init__(self)
		self.bounds = bounds
		self.obstacles_V = obstacles_V
		self.decomposition_nodes = self._setup_triangulation_free_space(env_name)  # G_T
		self.rdmap = nx.Graph()

	def sample(self):
		w_sum = sum(self.samplers['weights'])
		probs = [(1 - ETA) * w / w_sum + ETA / len(self.samplers['weights']) for w in self.samplers['weights']]
		self.sampler_idx = np.random.choice(len(self.samplers['sampler_list']), p=probs)
		return self.samplers['sampler_list'][self.sampler_idx].sample()

	def update_sampler(self, nodes, new_nodes, new_edges):
		# update rdmap
		prev_connected_components = len(list(nx.connected_components(self.rdmap)))
		self.rdmap.add_nodes_from([tuple(new_node) for new_node in new_nodes])
		self.rdmap.add_edges_from(new_edges)
		reward = (len(list(nx.connected_components(self.rdmap))) == prev_connected_components)

		# update sampler
		prob = (1 - ETA) * self.samplers['weights'][self.sampler_idx] / sum(self.samplers['weights']) + ETA / len(self.samplers['weights'])
		self.samplers['weights'][self.sampler_idx] *= np.exp(reward * ETA / len(self.samplers['weights']) / prob) # to be updated
		for sampler, feature_point in zip(self.samplers['sampler_list'], self.samplers['feature_points']):
			sampler.update_graph(nodes, new_edges, feature_point, self.interpolate)

	def set_up_sampler(self, feature_points):
		feature_points = np.concatenate([feature_points, np.zeros((1, 3))], axis=0)
		self.samplers = {
			'sampler_list': [],
			'feature_points': [],
			'weights': [],
		}
		self.sampler_weights = []
		for feature_point in feature_points:
			self.samplers['sampler_list'].append(WCO3DSampler(self.decomposition_nodes))
			self.samplers['feature_points'].append(feature_point)
			self.samplers['weights'].append(1)

	def _setup_triangulation_free_space(self, env_name):
		if env_name == 'easy-3D':
			return [
				trimesh.primitives.Box(bounds=[[-5, -1.4, 0], [5, 5, 3]]),
				trimesh.primitives.Box(bounds=[[-5, -5, 0], [5, -1.6, 3]]),
				trimesh.primitives.Box(bounds=[[0, -1.6, 0], [3.4, -1.4, 3]]),
			]
		elif env_name == 'dense-3D':
			return [
				trimesh.primitives.Box(bounds=[[-5, 0, 0], [-1.1, 5, 3]]),  # -1.4
				trimesh.primitives.Box(bounds=[[-5, -1.4, 2], [-1.1, 0, 3]]),



				trimesh.primitives.Box(bounds=[[-0.9, -1.4, 0], [5, 5, 3]]),
				trimesh.primitives.Box(bounds=[[-5, -1.4, 0], [-0.9, 0, 2]]),
				trimesh.primitives.Box(bounds=[[-5, -5, 0], [5, -1.9, 3]]),

				trimesh.primitives.Box(bounds=[[-5, -1.9, 0], [-4.5, -1.6, 3]]),

				trimesh.primitives.Box(bounds=[[-3.5, -1.9, 0], [-0.5, -1.6, 3]]),
				trimesh.primitives.Box(bounds=[[0.5, -1.9, 0], [5, -1.6, 3]]),
				trimesh.primitives.Box(bounds=[[-4.5, -1.9, 2], [-3.5, -1.6, 3]]),
				trimesh.primitives.Box(bounds=[[-0.5, -1.9, 2], [0.5, -1.6, 3]]),
				trimesh.primitives.Box(bounds=[[2, -1.6, 0], [3.4, -1.4, 2]]),
			]
		elif env_name == 'peg-3D':
			return [
				trimesh.primitives.Box(bounds=[[-1, 3, 0.4], [1, 5, 3]]),
				trimesh.primitives.Box(bounds=[[-5, -0.5, 0], [5, 3, 3]]),
				trimesh.primitives.Box(bounds=[[2, -3.5, 0], [5, -0.5, 3]]),
				trimesh.primitives.Box(bounds=[[-5, -5, 0], [5, -3.5, 3]]),
			]
		else:
			raise NotImplementedError(f"env_name {env_name} not implemented")


class WCO3DSampler:
	def __init__(self, nodes):
		self.all_nodes = list(nodes)
		self.non_iso_nodes = []

	def sample(self):
		non_iso_nodes = self.non_iso_nodes if len(self.non_iso_nodes) > 0 else self.all_nodes
		sample_box = non_iso_nodes[np.random.randint(len(non_iso_nodes))]

		position = sample_box.sample_volume(1)[0, :]
		rotation = Rotation.random().as_quat()

		return (position[0], position[1], position[2], rotation[0], rotation[1], rotation[2], rotation[3])

	def update_graph(self, nodes, new_edges, feature_point, interpolate_fn):
		# return
		for (node_i, node_j) in new_edges:
			interpolation = [interpolate_fn(nodes[node_i], nodes[node_j], t) for t in np.linspace(0, 1, 10, endpoint=True)]
			for node_idx in range(len(self.all_nodes) - 1, -1, -1):
				for inter_point in interpolation:
					xyz = np.array(inter_point[:3]) + Rotation.from_quat(inter_point[3:]).apply(feature_point[:3])
					if (self.all_nodes[node_idx].bounds[0] < xyz).all() and (xyz < self.all_nodes[node_idx].bounds[1]).all():
						self.non_iso_nodes.append(self.all_nodes.pop(node_idx))
						# print(f"add {node_idx} to non_iso_nodes")
						break

class KlamptEnvironment(CollisionCheckingEnvironment):
	"""
	Environment that uses Klampt for motion planning.
	"""
	def __init__(self, agent, c_length, radius, env_name=None, bounds=None, method_name=None):
		super().__init__(agent, c_length, radius, env_name, bounds)
		self._setup_klampt(method_name)

	def _setup_klampt(self, method_name):
		if not 'z' in self.bounds:
			if 'prm' in method_name:
				self.klampt_space = KlamptSE2CSpace()
			else:
				assert 'wco' in method_name
				self.klampt_space = KlamptCustomSE2CSpace(self.bounds, self.obstacles_V)
				self.klampt_space.set_up_sampler(self.agent.convex_hull.boundary.xy)

			self.klampt_space.setBounds([self.bounds['x'], self.bounds['y'], self.bounds['theta']])
			self.klampt_space.eps = 0.3  # the resolution of the states to be checked along the path between two nodes
			self.random_start_point = self.klampt_space.sample()
			while not self._is_configuration_free(self.random_start_point):
				self.random_start_point = self.klampt_space.sample()
			self.klampt_start_point = self.start_point

			self.random_end_point = self.klampt_space.sample()
			while not self._is_configuration_free(self.random_end_point):
				self.random_end_point = self.klampt_space.sample()
			self.klampt_end_point = self.end_point

		else:
			if 'prm' in method_name:
				self.klampt_space = KlamptSE3CSpace()
			else:
				assert 'wco' in method_name
				self.klampt_space = KlamptCustomSE3CSpace(self.bounds, self.obstacles_V, self.env_name)
				self.klampt_space.set_up_sampler(self.agent.convex_hull.vertices)

			self.klampt_space.setBounds([self.bounds['x'], self.bounds['y'], self.bounds['z'],
										[-1, 1], [-1, 1], [-1, 1], [-1, 1]])
			self.klampt_space.eps = 0.1  # the resolution of the states to be checked along the path between two nodes
			self.random_start_point = self.klampt_space.sample()
			while not self._is_configuration_free(self.random_start_point):
				self.random_start_point = self.klampt_space.sample()

			start_xyz = self.start_point[:3, -1]
			start_rot = Rotation.from_matrix(self.start_point[:3, :3]).as_quat()
			self.klampt_start_point = [start_xyz[0], start_xyz[1], start_xyz[2],
									   start_rot[0], start_rot[1], start_rot[2], start_rot[3]]

			self.random_end_point = self.klampt_space.sample()
			while not self._is_configuration_free(self.random_end_point):
				self.random_end_point = self.klampt_space.sample()

			end_xyz = self.end_point[:3, -1]
			end_rot = Rotation.from_matrix(self.end_point[:3, :3]).as_quat()
			self.klampt_end_point = [end_xyz[0], end_xyz[1], end_xyz[2],
									  end_rot[0], end_rot[1], end_rot[2], end_rot[3]]

		self.klampt_space.addFeasibilityTest(func=self.klampt_space.inBounds)
		self.klampt_space.addFeasibilityTest(func=self._is_configuration_free)
