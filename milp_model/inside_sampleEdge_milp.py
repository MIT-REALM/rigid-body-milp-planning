import itertools

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils.config import BIG_M

from .basic_wp_milp import Basic_wp_MILP
from utils.geometry_operation import compute_2D_piecewise_reachable_set, compute_3D_piecewise_reachable_set


class Inside_SampleEdge_MILP(Basic_wp_MILP):
	"""
	This class formulates the collision avoidance constraint as follows:
	1. The free work space is represented as the union of a sequence convex polytopes: A_i x <= b_i
	2. The reachable set of the path is restricted to be **inside** the free work space.
	3. Accurate reachable set is hard to represented as polytopes, so we sample points on the edge of the reachable set
		and use them to approximate the reachable set.
	"""

	def __init__(self, name, num_waypoint, env, start_config, end_config, A_list, b_list, profiler=None):
		self.A_list = A_list
		self.b_list = b_list

		super().__init__(name, num_waypoint, env, start_config, end_config, profiler)

	def _init_collision_free_constraints_2D(self):
		"""
		iris bigM formulation;
		a bigM variable for each (iris polytope) & (vertex of piece convex hull) pair;
		force vertex and sampled edge points to be out of the obstacles. (asymptomatically sufficient)
		"""
		var_waypoint_x = self.variables['waypoint_x']
		var_waypoint_y = self.variables['waypoint_y']
		var_waypoint_theta_cos = self.variables['waypoint_theta_cos']
		var_waypoint_theta_sin = self.variables['waypoint_theta_sin']
		var_avg_theta_cos = self.variables['avg_theta_cos']
		var_avg_theta_sin = self.variables['avg_theta_sin']

		for piece_idx in range(self.num_wp + 1):
			"""
			check the line segments completely lies inside the iris union
			
			LINE SEGMENTS TO CHECK: 
			i) agent at a given waypoint
			ii) the line segment of a given vertice between waypoints
			"""
			with self.profiler.profile("compute_piecewise_reachable_set"):
				to_check_segments = compute_2D_piecewise_reachable_set(
					self.env.agent,
					start_config=[var_waypoint_x[piece_idx], var_waypoint_y[piece_idx],
								  [var_waypoint_theta_cos[piece_idx], var_waypoint_theta_sin[piece_idx]]],
					end_config=[var_waypoint_x[piece_idx + 1], var_waypoint_y[piece_idx + 1],
								[var_waypoint_theta_cos[piece_idx + 1], var_waypoint_theta_sin[piece_idx + 1]]],
					avg_theta=[var_avg_theta_cos[piece_idx], var_avg_theta_sin[piece_idx]]
				)

			(xs, ys, xc, yc, xe, ye) = to_check_segments
			# check line segments
			with self.profiler.profile(f"add_line_constr_{self.name}"):
				self._enforce_line_inside_union((xs[:-1], ys[:-1]), (xc[:-1], yc[:-1]))
				self._enforce_line_inside_union((xc[:-1], yc[:-1]), (xe[:-1], ye[:-1]))
				# self._enforce_line_inside_union((xs[:-1], ys[:-1]), (xs[1:], ys[1:]))
				self._enforce_line_inside_union((xe[:-1], ye[:-1]), (xe[1:], ye[1:]))

	def _init_collision_free_constraints_3D(self):
		var_waypoint_x = self.variables['waypoint_x']
		var_waypoint_y = self.variables['waypoint_y']
		var_waypoint_z = self.variables['waypoint_z']
		var_rot_mat = self.variables['rotation_mat']

		for piece_idx in range(self.num_wp + 1):
			"""
			check the line segments completely lies inside the iris union

			LINE SEGMENTS TO CHECK: 
			i) agent at a given waypoint
			ii) the line segment of a given vertice between waypoints
			"""
			with self.profiler.profile("compute_piecewise_reachable_set"):
				to_check_segments = compute_3D_piecewise_reachable_set(
					self.env.agent,
					start_xyz=[var_waypoint_x[piece_idx], var_waypoint_y[piece_idx], var_waypoint_z[piece_idx]],
					end_xyz=[var_waypoint_x[piece_idx + 1], var_waypoint_y[piece_idx + 1], var_waypoint_z[piece_idx + 1]],
					rot_mat=var_rot_mat
				)

			(xs1, ys1, zs1, xe1, ye1, ze1), (xs2, ys2, zs2, xe2, ye2, ze2) = to_check_segments
			# check line segments
			with self.profiler.profile(f"add_line_constr_{self.name}"):
				self._enforce_line_inside_union((xs1, ys1, zs1), (xe1, ye1, ze1))
				self._enforce_line_inside_union((xs2, ys2, zs2), (xe2, ye2, ze2))

	def _add_waypoint_collision_free_constraints(self, to_list_flag: bool):
		raise NotImplementedError("add_waypoint_collision_free_constraints is not implemented for Inside_SampleEdge_MILP")
		if to_list_flag:
			var_start_x = self.variables['waypoint_x'][-2][-1]
			var_start_y = self.variables['waypoint_y'][-2][-1]
			var_start_theta_cos = self.variables['waypoint_theta_cos'][-2][-1]
			var_start_theta_sin = self.variables['waypoint_theta_sin'][-2][-1]
		else:
			var_start_x = self.variables['waypoint_x'][-2]
			var_start_y = self.variables['waypoint_y'][-2]
			var_start_theta_cos = self.variables['waypoint_theta_cos'][-2]
			var_start_theta_sin = self.variables['waypoint_theta_sin'][-2]

		with self.profiler.profile("compute_piecewise_reachable_set"):
			c_hull_verts = compute_2D_piecewise_reachable_set(
				self.env.agent,
				start_config=[var_start_x, var_start_y, [var_start_theta_cos, var_start_theta_sin]],
				end_config=[self.variables['waypoint_x'][-1], self.variables['waypoint_y'][-1],
							[self.variables['waypoint_theta_cos'][-1], self.variables['waypoint_theta_sin'][-1]]],
				avg_theta=[self.variables['avg_theta_cos'][-1], self.variables['avg_theta_sin'][-1]]
			)

		# vertices inside one polytope
		iris_polytope_num = len(self.A_list)
		agent_vert_num = len(c_hull_verts)
		point_num_per_vert = 2
		z = self.model.addMVar(shape=(iris_polytope_num, agent_vert_num, point_num_per_vert),
							   vtype=GRB.BINARY,
							   name=f"var_z_{self.num_wp}")  # z=1 <=> vertex is inside the polytope
		with self.profiler.profile(f"add_point_constr_{self.name}"):
			for vert_idx in range(agent_vert_num):
				for ii in range(point_num_per_vert):
					for iris_polytope_idx in range(iris_polytope_num):
						for edge_idx in range(len(self.b_list[iris_polytope_idx])):
							self.model.addConstr(
								self.A_list[iris_polytope_idx][edge_idx] @ c_hull_verts[vert_idx][ii].T <=
								np.array(self.b_list[iris_polytope_idx][edge_idx]) +
								BIG_M * (1 - z[iris_polytope_idx, vert_idx, ii]),
								name=f"{self.num_wp}:Airis_{iris_polytope_idx}<=b+z*M")
				self.model.addConstr(z[:, vert_idx, ii].sum() >= 1,
									 name=f"{self.num_wp}:sum(z)<=edge_num-1")

		# check line segments
		for vert_idx in range(agent_vert_num):
			with self.profiler.profile(f"add_line_constr_{self.name}"):
				# i) at a given waypoint, the line segment connecting two adjacent vertices
				self._enforce_line_wedge_separation(
					c_hull_verts[vert_idx][0], c_hull_verts[(vert_idx + 1)//agent_vert_num][0],
					self.to_check_wedge_list)
				# ii) for a given vertex, the line segment connecting waypoints
				self._enforce_line_wedge_separation(
					c_hull_verts[vert_idx][0], c_hull_verts[vert_idx][1], self.to_check_wedge_list)
				self._enforce_line_wedge_separation(
					c_hull_verts[vert_idx][1], c_hull_verts[vert_idx][2], self.to_check_wedge_list)

	def _enforce_line_inside_union(self, point_As, point_Bs):
		"""
		:param point_As / point_Bs: (x_list, y_list, maybe z_list)
		the constraint will enforce both the points take >= sign for at least one line
		"""
		lineseg_num = point_As[0].shape[0]
		interpolation_num = 10 + 1
		interpolation_weight = np.linspace(0, 1, num=interpolation_num, endpoint=True)

		zz = self.model.addMVar(shape=(lineseg_num, interpolation_num, len(self.b_list)),
								vtype=GRB.BINARY, name=f"point_inside_poly, 1 for being inside")
		for poly_idx, (A, b) in enumerate(zip(self.A_list, self.b_list)):
			for interpolation_idx in range(interpolation_num):
				self.model.addConstr(
					gp.quicksum([A[:, ax_idx, np.newaxis] @
								 (point_As[ax_idx] + (1 - interpolation_weight[interpolation_idx]) * (point_Bs[ax_idx] - point_As[ax_idx]))[np.newaxis, :]
								  for ax_idx in range(A.shape[-1])]) <=
					BIG_M * (1 - zz[np.newaxis, :, interpolation_idx, poly_idx]) + b, name=f"point_inside_poly_{poly_idx}")  # shape: (A_edge_num, lineseg_num)

		self.model.addConstr(gp.quicksum([zz[:, :, ax_idx] for ax_idx in range(len(self.b_list))]) >= 1, name=f"all_points_inside_poly")

		# case 1: start and end are in the same poly
		z1 = self.model.addMVar(shape=(lineseg_num, len(self.b_list)), vtype=GRB.BINARY, name=f"index_case_1")
		self.model.addConstr(z1 <= zz[:, 0, :], name=f"start")
		self.model.addConstr(z1 <= zz[:, -1, :], name=f"end")

		# case 2: one interpolated point connects start and end poly
		if len(self.A_list) == 2:
			z2 = self.model.addMVar(shape=(lineseg_num, 1), vtype=GRB.BINARY, name=f"index_case_2")
			self.model.addConstr(zz[:, :, 0] @ np.ones(interpolation_num) + zz[:, :, 1] @ np.ones(interpolation_num)
								 >= (1 + interpolation_num) - BIG_M * (1 - z2[:, 0]), name=f"constr_case_2")
		else:
			assert len(self.A_list) == 3, "only support 2 or 3 iris polytopes"
			z2 = self.model.addMVar(shape=(lineseg_num, 3), vtype=GRB.BINARY, name=f"index_case_2")
			# poly 1 and 2
			# self.model.addConstr(gp.quicksum([zz[:, :, ax_idx] for ax_idx in [0, 1]]) >= 1 - BIG_M * (1 - z2[:, 0, np.newaxis]),
			# 					 name=f"all_points_inside_poly_0_1")
			# self.model.addConstr(gp.quicksum([zz[:, :, ax_idx] for ax_idx in [0, 2]]) >= 1 - BIG_M * (1 - z2[:, 1, np.newaxis]),
			# 					 name=f"all_points_inside_poly_0_2")
			# self.model.addConstr(gp.quicksum([zz[:, :, ax_idx] for ax_idx in [1, 2]]) >= 1 - BIG_M * (1 - z2[:, 2, np.newaxis]),
			# 					 name=f"all_points_inside_poly_1_2")
			self.model.addConstr(zz[:, :, 0] @ np.ones(interpolation_num) + zz[:, :, 1] @ np.ones(interpolation_num)
								 >= (1 + interpolation_num) - BIG_M * (1 - z2[:, 0]), name=f"constr_case_2_1")
			self.model.addConstr(zz[:, :, 0] @ np.ones(interpolation_num) + zz[:, :, 2] @ np.ones(interpolation_num)
								 >= (1 + interpolation_num) - BIG_M * (1 - z2[:, 1]), name=f"constr_case_2_2")
			self.model.addConstr(zz[:, :, 1] @ np.ones(interpolation_num) + zz[:, :, 2] @ np.ones(interpolation_num)
								 >= (1 + interpolation_num) - BIG_M * (1 - z2[:, 2]), name=f"constr_case_2_3")

		# either case 1 or case 2
		self.model.addConstr(z1 @ np.ones(len(self.b_list)) + gp.quicksum([z2[:, ax_idx] for ax_idx in range(z2.shape[-1])]) >= 1, name=f"either_case_1_or_2")
