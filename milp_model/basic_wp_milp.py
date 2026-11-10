from abc import ABC, abstractmethod

import numpy as np
import gurobipy as gp
from gurobipy import GRB

from utils.config import *  # consider switching to dictionary

import warnings
from lightning.pytorch.profilers import SimpleProfiler


class Basic_wp_MILP(ABC):
	"""
	This class is a basic MILP model for solving the waypoint problem. Between each pair of waypoints, the agent is
	allowed to move in a straight line, or rotate in place. The agent is assumed to be a polygon, and the environment
	is assumed to be a 2D/3D region with polytopic obstacles.
	"""

	def __init__(self, name, num_waypoint, env, start_config, end_config, profiler=None):
		# assert num_waypoint >= 1, "num_waypoint must be at least 1, if 0 resort to reachability check"
		self.profiler = profiler or SimpleProfiler()

		self.env = env
		self.start_config = start_config
		self.end_config = end_config

		self.name = '0'  # name
		self.model = gp.Model(name)
		self.num_wp = num_waypoint

		self.need_to_remove_constraints = []  # for adding waypoints

		self.model.setParam('OutputFlag', 0)
		self.model.setParam('Threads', 2)
		# self.model.setParam("LogToConsole", 0)
		# self.model.setParam("OptimalityTol", 1e-2)
		# self.model.setParam("MIPFocus", 1)
		# self.model.setParam("SolutionLimit", 1)
		# self.model.setParam("Heuristics", 0.05)

		if self.env.cartesian_dim == 2:
			with self.profiler.profile("init_variables"):
				self._init_variables_2D()
			with self.profiler.profile("init_basic_constraints"):
				self._init_constraints_2D()
			with self.profiler.profile("init_collision_free_constraints"):
				self._init_collision_free_constraints_2D()
		else:
			assert self.env.cartesian_dim == 3, "Only 2D and 3D environments are supported"
			with self.profiler.profile("init_variables"):
				self._init_variables_3D()
			with self.profiler.profile("init_basic_constraints"):
				self._init_constraints_3D()
			with self.profiler.profile("init_collision_free_constraints"):
				self._init_collision_free_constraints_3D()

		self._init_objective()

	def __len__(self):
		return self.num_wp

	def __str__(self):
		return self.name

	def _init_variables_2D(self):
		variables = {}

		## Decision variables, translation or rotation
		variables['tr_decision'] = self.model.addMVar(
			shape=(self.num_wp + 1,), vtype=GRB.BINARY, name=f"tr_decision_{self.num_wp}")

		## Translation variables
		variables['waypoint_x'] = self.model.addMVar(
			shape=(self.num_wp + 2,), lb=self.env.bounds['x'][0], ub=self.env.bounds['x'][1],
			vtype=GRB.CONTINUOUS, name="x_of_waypoints")
		variables['waypoint_y'] = self.model.addMVar(
			shape=(self.num_wp + 2,), lb=self.env.bounds['y'][0], ub=self.env.bounds['y'][1],
			vtype=GRB.CONTINUOUS, name="y of waypoints")

		## Delta-Translation variables
		variables['piece_x_diff'] = variables['waypoint_x'][1:] - variables['waypoint_x'][:-1]
		variables['piece_y_diff'] = variables['waypoint_y'][1:] - variables['waypoint_y'][:-1]

		## Rotation and Delta-Rotation variables
		variables['waypoint_theta_idx'] = self.model.addMVar(
			shape=(self.num_wp + 2, NUM_THETA_DIV), vtype=GRB.BINARY, name="theta index of waypoints")
		variables.update(self.compute_theta_related(variables['waypoint_theta_idx']))
		self.model.addConstr(variables['waypoint_theta_idx'] @ np.ones((NUM_THETA_DIV,)) == 1, name="theta_idx_sum=1")

		variables['var_avg_theta_idx'] = self.model.addMVar(
			shape=(self.num_wp + 1, 2 * NUM_THETA_DIV), vtype=GRB.BINARY, name="avg_theta of two consecutive waypoints")
		var_rot_direction_idx = self.model.addMVar(
			shape=(self.num_wp + 1,), vtype=GRB.INTEGER, lb=-1, ub=1, name="rot_direction of two consecutive waypoints")
		variables.update(self.compute_avg_theta_related(variables['var_avg_theta_idx']))
		self.model.addConstr(variables['var_avg_theta_idx'] @ np.ones((2 * NUM_THETA_DIV,)) == 1,
							 name="avg_theta_idx_sum=1")
		self.model.addConstr((variables['waypoint_theta'][1:] + variables['waypoint_theta'][:-1]) / 2 ==
							 variables['avg_theta'] + np.pi * var_rot_direction_idx,
							 name="avg_theta")

		## Restrict delta_theta
		piece_theta_diff_a = 2 * (variables['avg_theta'] - variables['waypoint_theta'][:-1])
		piece_theta_diff_b = 2 * (variables['waypoint_theta'][1:] - variables['avg_theta'])
		piece_theta_diff_a_abs = self.model.addMVar(
			shape=(self.num_wp + 1,), lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="abs_delta_theta_a")
		piece_theta_diff_b_abs = self.model.addMVar(
			shape=(self.num_wp + 1,), lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="abs_delta_theta_b")
		self.model.addConstr(piece_theta_diff_a_abs >= piece_theta_diff_a, name="abs_delta_theta_a>=delta_theta_a")
		self.model.addConstr(piece_theta_diff_a_abs >= -piece_theta_diff_a, name="abs_delta_theta_a>=-delta_theta_a")
		self.model.addConstr(piece_theta_diff_b_abs >= piece_theta_diff_b, name="abs_delta_theta_b>=delta_theta_b")
		self.model.addConstr(piece_theta_diff_b_abs >= -piece_theta_diff_b, name="abs_delta_theta_b>=-delta_theta_b")

		abs_theta_diff_idx = self.model.addMVar(shape=(self.num_wp + 1,), vtype=GRB.BINARY, name="abs_delta_theta_idx")
		variables['abs_theta_diff'] = self.model.addMVar(
			shape=(self.num_wp + 1,), lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name="abs_delta_theta")
		self.model.addConstr(variables['abs_theta_diff'] >= piece_theta_diff_a_abs - BIG_M * abs_theta_diff_idx,
							 name="delta_theta>=|a| or")
		self.model.addConstr(variables['abs_theta_diff'] >= piece_theta_diff_b_abs - BIG_M * (1 - abs_theta_diff_idx),
							 name="delta_theta>=|b| or")
		self.model.addConstr(variables['abs_theta_diff'] <= np.ones((self.num_wp + 1,)) * DELTA_THETA_UL,
							 name="delta_theta<=pi/2")

		self.variables = variables

	def _init_variables_3D(self):
		"""
		different from 2D, inside 3D environment, only translation is allowed
		"""
		variables = {}

		## Translation variables
		variables['waypoint_x'] = self.model.addMVar(
			shape=(self.num_wp + 2,), lb=self.env.bounds['x'][0], ub=self.env.bounds['x'][1],
			vtype=GRB.CONTINUOUS, name="x_of_waypoints")
		variables['waypoint_y'] = self.model.addMVar(
			shape=(self.num_wp + 2,), lb=self.env.bounds['y'][0], ub=self.env.bounds['y'][1],
			vtype=GRB.CONTINUOUS, name="y of waypoints")
		variables['waypoint_z'] = self.model.addMVar(
			shape=(self.num_wp + 2,), lb=self.env.bounds['z'][0], ub=self.env.bounds['z'][1],
			vtype=GRB.CONTINUOUS, name="z of waypoints")

		## Delta-Translation variables
		variables['piece_x_diff'] = variables['waypoint_x'][1:] - variables['waypoint_x'][:-1]
		variables['piece_y_diff'] = variables['waypoint_y'][1:] - variables['waypoint_y'][:-1]
		variables['piece_z_diff'] = variables['waypoint_z'][1:] - variables['waypoint_z'][:-1]

		## Rotation and Delta-Rotation variables
		variables['waypoint_theta_idx'] = self.model.addMVar(
			shape=(ROTATION_MATS.shape[0]), vtype=GRB.BINARY, name="rot_mat index of all waypoints")
		variables.update(self.compute_3d_rot_mat(variables['waypoint_theta_idx']))
		self.model.addConstr(variables['waypoint_theta_idx'] @ np.ones((ROTATION_MATS.shape[0],)) == 1, name="theta_idx_sum=1")

		self.variables = variables

	def _init_objective(self):
		# Auxiliary variables for objective function (L1 norm)
		abs_x_diff = self.model.addMVar(shape=(self.num_wp + 1,), lb=0, ub=GRB.INFINITY,
										vtype=GRB.CONTINUOUS, name="abs_delta_x")
		abs_y_diff = self.model.addMVar(shape=(self.num_wp + 1,), lb=0, ub=GRB.INFINITY,
										vtype=GRB.CONTINUOUS, name="abs_delta_y")
		self.model.addConstr(abs_x_diff >= self.variables['piece_x_diff'], name="abs_x_diff>=delta_x")
		self.model.addConstr(abs_x_diff >= -self.variables['piece_x_diff'], name="abs_x_diff>=-delta_x")
		self.model.addConstr(abs_y_diff >= self.variables['piece_y_diff'], name="abs_y_diff>=delta_y")
		self.model.addConstr(abs_y_diff >= -self.variables['piece_y_diff'], name="abs_y_diff>=-delta_y")

		if self.env.cartesian_dim == 3:
			abs_z_diff = self.model.addMVar(shape=(self.num_wp + 1,), lb=0, ub=GRB.INFINITY,
											vtype=GRB.CONTINUOUS, name="abs_delta_z")
			self.model.addConstr(abs_z_diff >= self.variables['piece_z_diff'], name="abs_z_diff>=delta_z")
			self.model.addConstr(abs_z_diff >= -self.variables['piece_z_diff'], name="abs_z_diff>=-delta_z")

		# Set objective
		objective = np.ones(self.num_wp + 1).T @ abs_x_diff  # cost for x_translation
		objective += np.ones(self.num_wp + 1).T @ abs_y_diff  # cost for y_translation
		if self.env.cartesian_dim == 2:
			pass
			# objective += np.ones(self.num_wp + 1).T @ self.variables['abs_theta_diff']  # cost for rotation
		else:
			objective += np.ones(self.num_wp + 1).T @ abs_z_diff
		self.model.setObjective(objective, GRB.MINIMIZE)

	def _init_constraints_2D(self):
		## 1. The first waypoint is the start point
		start_config_idx = self.model.addMVar(
			shape=(len(self.start_config),), vtype=GRB.BINARY, name="start_config_idx")
		self.model.addConstr(start_config_idx.sum() == 1, name="start_config_idx_sum=1")
		self.model.addConstr(self.variables['waypoint_x'][0] == start_config_idx @ np.array(
								 [config[0] for config in self.start_config]), name="start_x")
		self.model.addConstr(self.variables['waypoint_y'][0] == start_config_idx @ np.array(
								 [config[1] for config in self.start_config]), name="start_y")
		self.model.addConstr(self.variables['waypoint_theta'][0] == start_config_idx @ np.array(
			[config[2] for config in self.start_config]), name="start_theta")

		## 2. The last waypoint is the end point
		self.variables['end_config_idx'] = self.model.addMVar(
			shape=(len(self.end_config),), vtype=GRB.BINARY, name="end_config_idx")
		self.model.addConstr(self.variables['end_config_idx'].sum() == 1, name="end_config_idx_sum=1")
		self.need_to_remove_constraints.append(
			self.model.addConstr(self.variables['waypoint_x'][-1] == self.variables['end_config_idx'] @ np.array(
				[config[0] for config in self.end_config]), name="end_x"))
		self.need_to_remove_constraints.append(
			self.model.addConstr(self.variables['waypoint_y'][-1] == self.variables['end_config_idx'] @ np.array(
				[config[1] for config in self.end_config]), name="end_y"))
		self.need_to_remove_constraints.append(
			self.model.addConstr(self.variables['waypoint_theta'][-1] == self.variables['end_config_idx'] @ np.array(
			[config[2] for config in self.end_config]), name="end_theta"))

		## 3. Only translate or rotate
		self.model.addConstr(self.variables['piece_x_diff'] <= BIG_M * self.variables['tr_decision'],
							 name="delta_x<=M*b")
		self.model.addConstr(self.variables['piece_x_diff'] >= -BIG_M * self.variables['tr_decision'],
							 name="delta_x>=-M*b")
		self.model.addConstr(self.variables['piece_y_diff'] <= BIG_M * self.variables['tr_decision'],
							 name="delta_y<=M*b")
		self.model.addConstr(self.variables['piece_y_diff'] >= -BIG_M * self.variables['tr_decision'],
							 name="delta_y>=-M*b")
		self.model.addConstr(self.variables['abs_theta_diff'] <= BIG_M * (1 - self.variables['tr_decision']),
							 name="abs_delta_theta<=M*(1-b)")

	def _init_constraints_3D(self):
		## 1. The first waypoint is the start point
		start_config_idx = self.model.addMVar(
			shape=(len(self.start_config),), vtype=GRB.BINARY, name="start_config_idx")
		self.model.addConstr(start_config_idx.sum() == 1, name="start_config_idx_sum=1")
		self.model.addConstr(self.variables['waypoint_x'][0] == start_config_idx @ np.array(
								 [config[0][-1] for config in self.start_config]), name="start_x")
		self.model.addConstr(self.variables['waypoint_y'][0] == start_config_idx @ np.array(
								 [config[1][-1] for config in self.start_config]), name="start_y")
		self.model.addConstr(self.variables['waypoint_z'][0] == start_config_idx @ np.array(
								 [config[2][-1] for config in self.start_config]), name="start_z")
		self.model.addConstr(self.variables['rotation_mat'] == gp.quicksum([
			start_config_idx[i] * config[:3, :3] for i, config in enumerate(np.array(self.start_config))]), name="start_theta")

		## 2. The last waypoint is the end point
		self.variables['end_config_idx'] = self.model.addMVar(
			shape=(len(self.end_config),), vtype=GRB.BINARY, name="end_config_idx")
		self.model.addConstr(self.variables['end_config_idx'].sum() == 1, name="end_config_idx_sum=1")
		self.need_to_remove_constraints.append(
			self.model.addConstr(self.variables['waypoint_x'][-1] == self.variables['end_config_idx'] @ np.array(
				[config[0][-1] for config in self.end_config]), name="end_x"))
		self.need_to_remove_constraints.append(
			self.model.addConstr(self.variables['waypoint_y'][-1] == self.variables['end_config_idx'] @ np.array(
				[config[1][-1] for config in self.end_config]), name="end_y"))
		self.need_to_remove_constraints.append(
			self.model.addConstr(self.variables['waypoint_z'][-1] == self.variables['end_config_idx'] @ np.array(
				[config[2][-1] for config in self.end_config]), name="end_z"))
		self.need_to_remove_constraints.append(
			self.model.addConstr(self.variables['rotation_mat'] == gp.quicksum([
				self.variables['end_config_idx'][i] * config[:3, :3] for i, config in enumerate(np.array(self.end_config))]), name="end_theta"))

	@abstractmethod
	def _init_collision_free_constraints_2D(self):
		pass

	@abstractmethod
	def _init_collision_free_constraints_3D(self):
		pass

	def solve(self) -> bool:
		self.model.optimize()
		if self.model.status in [GRB.INFEASIBLE, GRB.Status.INF_OR_UNBD]:
			# self.model.computeIIS()
			# self.model.write(f'{self.name}.ilp')
			# warnings.warn(f'User Warning: Optimization failed due to infeasibility; wrote model to {self.name}.ilp')
			return False
		else:
			return True

	def solve_with_time_limit(self, time_limit: float) -> str:
		self.model.setParam('TimeLimit', time_limit)
		self.model.optimize()
		if self.model.status in [GRB.INFEASIBLE, GRB.Status.INF_OR_UNBD]:
			return 'infeasible'
		elif self.model.status == GRB.TIME_LIMIT:
			return 'in_progress'
		else:
			return 'solved'

	def get_path(self):
		assert self.model.status not in [GRB.LOADED, GRB.INPROGRESS, GRB.INFEASIBLE, GRB.Status.INF_OR_UNBD], f"Model status: {self.model.status}"
		if self.env.cartesian_dim == 2:
			# shape: (num_wp + 2, 3)
			waypoints_theta = self.retrieve_waypoints(self.variables['waypoint_theta_idx'])
			waypoints_avg_theta = self.retrieve_waypoints(self.variables['var_avg_theta_idx'])
			recovered_theta_related = self.compute_theta_related(waypoints_theta)
			recovered_avg_theta_related = self.compute_avg_theta_related(waypoints_avg_theta)
			return np.concatenate([
				self.retrieve_waypoints(self.variables['waypoint_x']).reshape(-1, 1),
				self.retrieve_waypoints(self.variables['waypoint_y']).reshape(-1, 1),
				recovered_theta_related['waypoint_theta'].reshape(-1, 1)], axis=1
			), {
				"avg_theta": recovered_avg_theta_related['avg_theta'],
				"avg_theta_cos": recovered_avg_theta_related['avg_theta_cos'],
				"avg_theta_sin": recovered_avg_theta_related['avg_theta_sin'],
			}
		else:
			# shape: (num_wp + 2, 4, 4)
			rot_mat = np.einsum('i,ijk->jk', self.variables['waypoint_theta_idx'].x, ROTATION_MATS)  # (3, 3)
			trans_mats = np.pad(rot_mat[np.newaxis].repeat(self.num_wp + 2, axis=0), ((0, 0), (0, 1), (0, 1)), mode='constant', constant_values=0)  # (n, 4, 4)
			trans_mats[:, -1, -1] = 1
			trans_mats[:, :-1, -1] = np.concatenate([
				self.retrieve_waypoints(self.variables['waypoint_x']).reshape(-1, 1),
				self.retrieve_waypoints(self.variables['waypoint_y']).reshape(-1, 1),
				self.retrieve_waypoints(self.variables['waypoint_z']).reshape(-1, 1)], axis=1
			)
			return trans_mats, {}

	################################### Add waypoints ###################################

	def add_waypoint(self):
		raise NotImplementedError("add_waypoint is not implemented with iris")
		# TODO: add waypoint in the middle of the path
		self.num_wp += 1
		to_list_flag = (not isinstance(self.variables['tr_decision'], list))

		with self.profiler.profile("aa-remove_constraints"):
			for constr in self.need_to_remove_constraints:
				self.model.remove(constr)

		with self.profiler.profile("aa-add_variable"):
			self._add_waypoint_variable(to_list_flag=to_list_flag)
		with self.profiler.profile("aa-add_objectives"):
			self._add_waypoint_objective()
		with self.profiler.profile("aa-add_constraints"):
			self._add_waypoint_constraints()
			self._add_waypoint_collision_free_constraints(to_list_flag=to_list_flag)

		with self.profiler.profile("aa-update"):
			self.model.update()

	def _add_waypoint_variable(self, to_list_flag: bool):
		if to_list_flag:
			for key in self.variables:
				if not key in ['end_config_idx', 'end_config_lambda']:
					self.variables[key] = [self.variables[key]]

		## Decision variables, translation or rotation
		self.variables['tr_decision'].append(self.model.addMVar(
			shape=(1,), vtype=GRB.BINARY, name=f"tr_decision_{self.num_wp}"))

		## Translation variables
		self.variables['waypoint_x'].append(self.model.addMVar(
			shape=(1,), lb=self.env.bounds['x'][0], ub=self.env.bounds['x'][1],
			vtype=GRB.CONTINUOUS, name=f"x_of_waypoint_{self.num_wp}"))
		self.variables['waypoint_y'].append(self.model.addMVar(
			shape=(1,), lb=self.env.bounds['y'][0], ub=self.env.bounds['y'][1],
			vtype=GRB.CONTINUOUS, name=f"y_of_waypoint_{self.num_wp}"))

		## Delta-Translation variables
		if to_list_flag:
			self.variables['piece_x_diff'].append(self.variables['waypoint_x'][-1] - self.variables['waypoint_x'][-2][-1])
			self.variables['piece_y_diff'].append(self.variables['waypoint_y'][-1] - self.variables['waypoint_y'][-2][-1])
		else:
			self.variables['piece_x_diff'].append(self.variables['waypoint_x'][-1] - self.variables['waypoint_x'][-2])
			self.variables['piece_y_diff'].append(self.variables['waypoint_y'][-1] - self.variables['waypoint_y'][-2])

		## Rotation and Delta-Rotation variables
		self.variables['waypoint_theta_idx'].append(self.model.addMVar(
			shape=(NUM_THETA_DIV,), vtype=GRB.BINARY, name=f"theta index of waypoint_{self.num_wp}"))
		update_theta_dict = self.compute_theta_related(self.variables['waypoint_theta_idx'][-1])
		for key in update_theta_dict:
			self.variables[key].append(update_theta_dict[key])
		self.model.addConstr(self.variables['waypoint_theta_idx'][-1] @ np.ones((NUM_THETA_DIV,)) == 1,
							 name=f"theta_idx_{self.num_wp}=1")

		self.variables['var_avg_theta_idx'].append(self.model.addMVar(
			shape=(2 * NUM_THETA_DIV,), vtype=GRB.BINARY, name=f"avg_theta of waypoint_{self.num_wp}"))
		var_rot_direction_idx = self.model.addMVar(
			shape=(1,), vtype=GRB.INTEGER, lb=-1, ub=1, name=f"rot_direction of waypoint_{self.num_wp}")
		update_avg_theta_dict = self.compute_avg_theta_related(self.variables['var_avg_theta_idx'][-1])
		for key in update_avg_theta_dict:
			self.variables[key].append(update_avg_theta_dict[key])
		self.model.addConstr(self.variables['var_avg_theta_idx'][-1] @ np.ones((2 * NUM_THETA_DIV,)) == 1,
							 name=f"avg_theta_idx_{self.num_wp}=1")
		if to_list_flag:
			self.model.addConstr((self.variables['waypoint_theta'][-1] + self.variables['waypoint_theta'][-2][-1]) / 2 ==
								 self.variables['avg_theta'][-1] + np.pi * var_rot_direction_idx,
								 name=f"avg_theta_{self.num_wp}")
		else:
			self.model.addConstr((self.variables['waypoint_theta'][-1] + self.variables['waypoint_theta'][-2]) / 2 ==
								 self.variables['avg_theta'][-1] + np.pi * var_rot_direction_idx,
								 name=f"avg_theta_{self.num_wp}")

		## Restrict delta_theta
		if to_list_flag:
			piece_theta_diff_a = 2 * (self.variables['avg_theta'][-1] - self.variables['waypoint_theta'][-2][-1])
		else:
			piece_theta_diff_a = 2 * (self.variables['avg_theta'][-1] - self.variables['waypoint_theta'][-2])
		piece_theta_diff_b = 2 * (self.variables['waypoint_theta'][-1] - self.variables['avg_theta'][-1])
		piece_theta_diff_a_abs = self.model.addMVar(
			shape=(1,), lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"abs_delta_theta_a_{self.num_wp}")
		piece_theta_diff_b_abs = self.model.addMVar(
			shape=(1,), lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"abs_delta_theta_b_{self.num_wp}")
		self.model.addConstr(piece_theta_diff_a_abs >= piece_theta_diff_a, name=f"abs_delta_theta_a_{self.num_wp}>=delta_theta_a")
		self.model.addConstr(piece_theta_diff_a_abs >= -piece_theta_diff_a, name=f"abs_delta_theta_a_{self.num_wp}>=-delta_theta_a")
		self.model.addConstr(piece_theta_diff_b_abs >= piece_theta_diff_b, name=f"abs_delta_theta_b_{self.num_wp}>=delta_theta_b")
		self.model.addConstr(piece_theta_diff_b_abs >= -piece_theta_diff_b, name=f"abs_delta_theta_b_{self.num_wp}>=-delta_theta_b")

		abs_theta_diff_idx = self.model.addMVar(shape=(1,), vtype=GRB.BINARY, name=f"abs_delta_theta_idx_{self.num_wp}")
		self.variables['abs_theta_diff'].append(self.model.addMVar(
			shape=(1,), lb=0, ub=GRB.INFINITY, vtype=GRB.CONTINUOUS, name=f"abs_delta_theta_{self.num_wp}"))
		self.model.addConstr(self.variables['abs_theta_diff'][-1] >= piece_theta_diff_a_abs - BIG_M * abs_theta_diff_idx,
							 name=f"delta_theta_{self.num_wp}>=|a| or")
		self.model.addConstr(self.variables['abs_theta_diff'][-1] >= piece_theta_diff_b_abs - BIG_M * (1 - abs_theta_diff_idx),
							 name=f"delta_theta_{self.num_wp}>=|b| or")
		self.model.addConstr(self.variables['abs_theta_diff'][-1] <= DELTA_THETA_UL, name=f"delta_theta_{self.num_wp}<=pi/2")

	def _add_waypoint_objective(self):
		prev_objecive = self.model.getObjective()

		abs_x_diff = self.model.addMVar(shape=(1,), lb=0, ub=GRB.INFINITY,
										vtype=GRB.CONTINUOUS, name=f"abs_delta_x_{self.num_wp}")
		abs_y_diff = self.model.addMVar(shape=(1,), lb=0, ub=GRB.INFINITY,
										vtype=GRB.CONTINUOUS, name=f"abs_delta_y_{self.num_wp}")
		self.model.addConstr(abs_x_diff >= self.variables['piece_x_diff'][-1], name=f"abs_x_diff_{self.num_wp}>=delta_x")
		self.model.addConstr(abs_x_diff >= -self.variables['piece_x_diff'][-1], name=f"abs_x_diff_{self.num_wp}>=-delta_x")
		self.model.addConstr(abs_y_diff >= self.variables['piece_y_diff'][-1], name=f"abs_y_diff_{self.num_wp}>=delta_y")
		self.model.addConstr(abs_y_diff >= -self.variables['piece_y_diff'][-1], name=f"abs_y_diff_{self.num_wp}>=-delta_y")

		objective = prev_objecive + abs_x_diff + abs_y_diff + self.variables['abs_theta_diff'][-1]
		self.model.setObjective(objective, GRB.MINIMIZE)

	def _add_waypoint_constraints(self):
		## 1. The first waypoint is the start point, no need to modify constraints
		## 2. The last waypoint is the end point, change to new end point
		self.need_to_remove_constraints.append(
			self.model.addConstr(self.variables['waypoint_x'][-1] == self.variables['end_config_idx'] @ np.array(
				[config[1][0] for config in self.end_config]), name=f"end_x"))
		self.need_to_remove_constraints.append(
			self.model.addConstr(self.variables['waypoint_y'][-1] == self.variables['end_config_idx'] @ np.array(
				[config[1][1] for config in self.end_config]), name=f"end_y"))
		self.need_to_remove_constraints.append(
			self.model.addConstr(self.variables['waypoint_theta'][-1] == self.variables['end_config_idx'] @ np.array(
				[config[0] for config in self.end_config]), name="end_theta"))

		## 3. Only translate or rotate
		self.model.addConstr(self.variables['piece_x_diff'][-1] <= BIG_M * self.variables['tr_decision'][-1],
							 name=f"delta_x_{self.num_wp}<=M*b")
		self.model.addConstr(self.variables['piece_x_diff'][-1] >= -BIG_M * self.variables['tr_decision'][-1],
							 name=f"delta_x_{self.num_wp}>=-M*b")
		self.model.addConstr(self.variables['piece_y_diff'][-1] <= BIG_M * self.variables['tr_decision'][-1],
							 name=f"delta_y_{self.num_wp}<=M*b")
		self.model.addConstr(self.variables['piece_y_diff'][-1] >= -BIG_M * self.variables['tr_decision'][-1],
							 name=f"delta_y_{self.num_wp}>=-M*b")
		self.model.addConstr(self.variables['abs_theta_diff'][-1] <= BIG_M * (1 - self.variables['tr_decision'][-1]),
							 name=f"abs_delta_theta_{self.num_wp}<=M*(1-b)")

	@abstractmethod
	def _add_waypoint_collision_free_constraints(self, to_list_flag: bool):
		pass

	################################### Auxiliary functions ###################################
	@staticmethod
	def compute_theta_related(theta_idx_variable):
		theta = theta_idx_variable @ np.linspace(-np.pi, np.pi, NUM_THETA_DIV, endpoint=False)
		theta_cos = theta_idx_variable @ np.cos(np.linspace(-np.pi, np.pi, NUM_THETA_DIV, endpoint=False))
		theta_sin = theta_idx_variable @ np.sin(np.linspace(-np.pi, np.pi, NUM_THETA_DIV, endpoint=False))

		return {
			'waypoint_theta': theta,
			'waypoint_theta_cos': theta_cos,
			'waypoint_theta_sin': theta_sin,
		}

	@staticmethod
	def compute_avg_theta_related(avg_theta_idx_variable):
		avg_theta = avg_theta_idx_variable @ np.linspace(-np.pi, np.pi, 2 * NUM_THETA_DIV, endpoint=False)
		avg_theta_cos = avg_theta_idx_variable @ np.cos(np.linspace(-np.pi, np.pi, 2 * NUM_THETA_DIV, endpoint=False))
		avg_theta_sin = avg_theta_idx_variable @ np.sin(np.linspace(-np.pi, np.pi, 2 * NUM_THETA_DIV, endpoint=False))

		return {
			'avg_theta': avg_theta,
			'avg_theta_cos': avg_theta_cos,
			'avg_theta_sin': avg_theta_sin,
		}

	@staticmethod
	def compute_3d_rot_mat(theta_idx_variable):
		return {
			'rotation_mat': gp.quicksum([theta_idx_variable[idx] * ROTATION_MATS[idx] for idx in range(theta_idx_variable.shape[0])])
		}

	@staticmethod
	def retrieve_waypoints(waypoints_variable):
		if isinstance(waypoints_variable, list):
			second_dim = waypoints_variable[-1].x.shape[0]
			return np.concatenate([var.x.reshape(-1, second_dim) for var in waypoints_variable], axis=0)
		else:
			return waypoints_variable.x
