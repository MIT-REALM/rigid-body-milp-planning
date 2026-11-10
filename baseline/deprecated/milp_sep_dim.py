import time
import warnings

import gurobipy as gp
from gurobipy import GRB

from environment.basic_env import BaseEnvironment
from utils.config import *
from utils.geometry_operation import compute_2D_piecewise_reachable_set

from milp_model import Inside_SampleEdge_MILP
from milp_model.basic_wp_milp import Basic_wp_MILP


def solve_waypoint_seperating_dim(
		env: BaseEnvironment,
		maximum_piece: int,
		A_list: np.ndarray,
		b_list: np.ndarray,
		milp_formulation: str = 'naive_bigM_sampleEdge',
) -> (np.ndarray, dict):
	# make sure env.start_point[2] and env.end_point[2] are N * 2pi / DELTA_THETA_DIV.

	for num_waypoint in range(8, maximum_piece):
		# Initialize the model
		if milp_formulation == 'iris_bigM_sampleEdge':
			milp_to_solve = Inside_SampleEdge_MILP(
				name="waypoint", num_waypoint=num_waypoint, env=env,
				start_config=[(env.start_point[2], env.start_point[:2])],
				end_config=[(env.end_point[2], tuple(env.end_point[:2].tolist()))],
				A_list=A_list, b_list=b_list,
			)
		else:
			warnings.warn(f"Collision avoidance version {milp_formulation} not implemented as class.")
			milp_to_solve = Basic_wp_MILP(
				name="waypoint", num_waypoint=num_waypoint, env=env,
				start_config=[(env.start_point[2], env.start_point[:2])],
				end_config=[(env.end_point[2], tuple(env.end_point[:2].tolist()))],
			)
			model = milp_to_solve.model
			assert len(env.bounds) == 3

			variables = milp_to_solve.variables

			add_collision_avoidance_constraints(
				model, env, num_waypoint,
				variables=variables,
				version=milp_formulation,
				A_list=A_list, b_list=b_list,
				waypoints=[variables['waypoint_x'], variables['waypoint_y'], variables['waypoint_theta']])

		milp_to_solve.solve()

		path, avg_theta_info = milp_to_solve.get_path()
		return path, avg_theta_info


def add_collision_avoidance_constraints(model: gp.Model, env: BaseEnvironment, num_waypoint: int, **kwargs):
	var_waypoint_x = kwargs['variables']['waypoint_x']
	var_waypoint_y = kwargs['variables']['waypoint_y']
	var_waypoint_theta_cos = kwargs['variables']['waypoint_theta_cos']
	var_waypoint_theta_sin = kwargs['variables']['waypoint_theta_sin']
	var_avg_theta_cos = kwargs['variables']['avg_theta_cos']
	var_avg_theta_sin = kwargs['variables']['avg_theta_sin']
	for piece_idx in range(num_waypoint + 1):
		if kwargs['version'] == 'naive_bigM_overRestrictive':
			"""
			naive bigM formulation; 
			a bigM variable for each (obstacle edge) & (vertex of piece convex hull) pair;
			forcing all vertex to be of same side of a hyperplane, too restrictive.
			"""
			c_hull_verts = compute_2D_piecewise_reachable_set(
				env.agent,
				[var_waypoint_x[piece_idx], var_waypoint_y[piece_idx],
				 [var_waypoint_theta_cos[piece_idx], var_waypoint_theta_sin[piece_idx]]],
				[var_waypoint_x[piece_idx + 1], var_waypoint_y[piece_idx + 1],
				 [var_waypoint_theta_cos[piece_idx + 1], var_waypoint_theta_sin[piece_idx + 1]]],
				[var_avg_theta_cos[piece_idx], var_avg_theta_sin[piece_idx]]
			)
			for obs_A, obs_b, obs_V in zip(env.obstacles_H['A'], env.obstacles_H['b'], env.obstacles_V):
				edge_num = obs_A.shape[0]
				agent_vert_num = len(c_hull_verts)
				z = model.addMVar(shape=(edge_num,), vtype=GRB.BINARY,
								  name=f"var_z_{obs_V['name']}_{piece_idx}")
				for vert_idx in range(agent_vert_num):
					for edge_idx in range(edge_num):
						for ii in range(3):
							model.addConstr(
								obs_A[edge_idx] @ c_hull_verts[vert_idx][ii].T >= obs_b[edge_idx] - BIG_M * z[edge_idx],
								name=f"{piece_idx}:Aobs_{obs_V['name']}>=b-z*M")
				model.addConstr(np.ones((1, edge_num)) @ z <= np.ones((1, agent_vert_num)) * (edge_num - 1),
								name=f"{piece_idx}:sum(z)<=edge_num-1")

		elif kwargs['version'] == 'naive_bigM_sampleEdge':
			"""
			naive bigM formulation; 
			a bigM variable for each (obstacle edge) & (vertex of piece convex hull) pair;
			force vertex and sampled edge points to be out of the obstacles. (asymptomatically sufficient)
			"""
			c_hull_verts = compute_2D_piecewise_reachable_set(
				env.agent_dense,
				[var_waypoint_x[piece_idx], var_waypoint_y[piece_idx],
				 [var_waypoint_theta_cos[piece_idx], var_waypoint_theta_sin[piece_idx]]],
				[var_waypoint_x[piece_idx + 1], var_waypoint_y[piece_idx + 1],
				 [var_waypoint_theta_cos[piece_idx + 1], var_waypoint_theta_sin[piece_idx + 1]]],
				[var_avg_theta_cos[piece_idx], var_avg_theta_sin[piece_idx]]
			)
			for obs_A, obs_b, obs_V in zip(env.obstacles_H['A'], env.obstacles_H['b'], env.obstacles_V):
				edge_num = obs_A.shape[0]
				agent_vert_num = len(c_hull_verts)
				z = model.addMVar(shape=(edge_num, agent_vert_num), vtype=GRB.BINARY,
								  name=f"var_z_{obs_V['name']}_{piece_idx}")
				for vert_idx in range(agent_vert_num):
					for edge_idx in range(edge_num):
						for ii in range(3):
							model.addConstr(
								obs_A[edge_idx] @ c_hull_verts[vert_idx][ii].T >= obs_b[edge_idx] - BIG_M * z[
									edge_idx, vert_idx],
								name=f"{piece_idx}:Aobs_{obs_V['name']}>=b-z*M")
				model.addConstr(np.ones((1, edge_num)) @ z <= np.ones((1, agent_vert_num)) * (edge_num - 1),
								name=f"{piece_idx}:sum(z)<=edge_num-1")

		elif kwargs['version'] == 'naive_bigM_sampleSeparatingLine':
			"""
			naive bigM formulation; 
			a bigM variable for each (obstacle separating line) & (triangulated agent) pair;
			force reachable set of the triangle to be of another side of the separating line.
			
			same as support function
			"""
			for obs_idx, obs_sep_line in enumerate(env.obstacles_separating_line):
				z = model.addMVar(shape=(SEPLINE_NUM_PER_OBS, len(env.agent_triangulated_array)),
								  vtype=GRB.BINARY, name=f"var_z_{obs_idx}_collision_avoidance")
				for tri_idx, tri_agent in enumerate(env.agent_triangulated_array):
					tri_agent_vert_num = 3
					c_hull_verts = compute_2D_piecewise_reachable_set(
						tri_agent,
						[var_waypoint_x[piece_idx], var_waypoint_y[piece_idx],
						 [var_waypoint_theta_cos[piece_idx], var_waypoint_theta_sin[piece_idx]]],
						[var_waypoint_x[piece_idx + 1], var_waypoint_y[piece_idx + 1],
						 [var_waypoint_theta_cos[piece_idx + 1], var_waypoint_theta_sin[piece_idx + 1]]],
						[var_avg_theta_cos[piece_idx], var_avg_theta_sin[piece_idx]]
					)  # shape: (tri_agent_vert_num, 3, 2) = (3, 3, 2)
					c_hull_verts = np.array(c_hull_verts).reshape(-1, 2)  # shape: (tri_agent_vert_num * 3, 2) = (9, 2)

					for vert_idx, vert in enumerate(c_hull_verts):
						for line_idx, line in enumerate(obs_sep_line['bs']):
							model.addConstr(c_hull_verts[vert_idx] @ obs_sep_line['As'][:, line_idx] >=
											obs_sep_line['bs'][line_idx] - BIG_M * z[line_idx, tri_idx],
											name=f"{obs_idx}_{tri_idx}:Aobs_{vert_idx}>=b-z*M")
					model.addConstr(np.ones((1, SEPLINE_NUM_PER_OBS)) @ z[:, tri_idx] <= len(obs_sep_line['bs']) - 1,
									name=f"{obs_idx}_{tri_idx}:sum(z)<=edge_num-1")


		else:
			raise NotImplementedError(f"Collision avoidance version {kwargs['version']} not implemented.")


