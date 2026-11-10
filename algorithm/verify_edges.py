from multiprocessing import Process, Queue, Value
import multiprocessing as mp

from lightning.pytorch.profilers import SimpleProfiler
from tqdm import tqdm

import numpy as np
import networkx as nx

import shapely
from shapely import affinity, union_all
import shapely.geometry as geom

from milp_model import Inside_SampleEdge_MILP
from environment.clean_env import CleanEnv as Environment
from utils.config import *
from utils.geometry_operation import *

from .pre_processing import compute_free_C_on_boundary


def prepare_batch_MILP_inputs_offline(env: Environment, coarse_graph: nx.Graph, dense_graph: nx.Graph,
							  tracker: SimpleProfiler, **kwargs) -> list:
	# TODO: all can be optimized to multi-processing.Pool
	if env.cartesian_dim == 2:
		assert "ax" in kwargs
		from utils.plot2D import plot_poly
	elif env.cartesian_dim == 3:
		from utils.plot3D import plot_poly
	else:
		raise NotImplementedError(f"Dimension {env.cartesian_dim} not implemented")


	# grow free C on polytope boundary
	with tracker.profile("compute_free_C_on_boundary"):
		prepare_endpoints_list = []
		for edge in coarse_graph.edges():
			free_patch_list, graph_list = compute_free_C_on_boundary(
				coarse_graph.edges[edge]['intersection'], env.agent_dense_edge_array,
				poly_1=(coarse_graph.nodes[edge[0]]['A'], np.array(coarse_graph.nodes[edge[0]]['b'])),
				poly_2=(coarse_graph.nodes[edge[1]]['A'], np.array(coarse_graph.nodes[edge[1]]['b']))
			)
			coarse_graph.edges[edge]['free_patch'] = free_patch_list

			# ''''''
			# plot free patches
			# for patch in free_patch_list:
			# 	for config in patch:
			# 		# plot_poly(poly=env.agent, transform=config, color='blue', alpha=0.3, **kwargs)
			# 		plot_poly(poly=env.agent, transform=config, color='blue', **kwargs)

			# patch = free_patch_list[3]
			# for config in patch:
			# 	# plot_poly(poly=env.agent, transform=config, color='blue', alpha=0.3, **kwargs)
			# 	plot_poly(poly=env.agent, transform=config, color='blue', **kwargs)
			# ''''''
			prepare_endpoints_list.append([edge, free_patch_list, graph_list])

	with tracker.profile("construct_dense_graph"):
		for patches in prepare_endpoints_list:
			for patch, graph in zip(patches[1], patches[2]):
				if len(patch) > 0:
					dense_graph.add_node(tuple(patch))
					dense_graph.nodes[tuple(patch)]['coarse_nodes'] = patches[0]
					dense_graph.nodes[tuple(patch)]['connectivity_graph'] = graph
					dense_graph.nodes[tuple(patch)]['config_position'] = [tuple(coord) for coord in graph.graph['coord']]

	with ((((tracker.profile("small_MILPs"))))):
		# init inputs for small MILPs
		prepare_MILP_inputs = []
		dense_nodes = list(dense_graph.nodes())

		for node_A_idx, node_A in enumerate(dense_nodes):
			for node_B in dense_nodes[node_A_idx + 1:]:
				poly_A_tuple = dense_graph.nodes[node_A]['coarse_nodes']
				poly_B_tuple = dense_graph.nodes[node_B]['coarse_nodes']
				poly_B1 = poly_B_tuple[0] in poly_A_tuple
				poly_B2 = poly_B_tuple[1] in poly_A_tuple

				# check if the two patches are connected in the coarse graph
				if poly_B1 and poly_B2:
					poly_tuple = poly_A_tuple
				elif poly_B1 and (not poly_B2):
					poly_tuple = (poly_A_tuple[0], poly_A_tuple[1], poly_B_tuple[1])
					# overlap_idx = poly_A_tuple.index(poly_B_tuple[0])
					# overlap_A, overlap_b = coarse_graph.nodes[poly_A_tuple[overlap_idx]]['A'], coarse_graph.nodes[poly_A_tuple[overlap_idx]]['b']
					# if not np.all(overlap_A @ np.array(dense_graph.nodes[node_A]['config_position']).T < overlap_b.reshape(-1, 1)) \
					# 	or not np.all(overlap_A @ np.array(dense_graph.nodes[node_B]['config_position']).T < overlap_b.reshape(-1, 1)):
					# 	continue

				elif (not poly_B1) and poly_B2:
					poly_tuple = (poly_A_tuple[0], poly_A_tuple[1], poly_B_tuple[0])
					# overlap_idx = poly_A_tuple.index(poly_B_tuple[1])
					# overlap_A, overlap_b = coarse_graph.nodes[poly_A_tuple[overlap_idx]]['A'], coarse_graph.nodes[poly_A_tuple[overlap_idx]]['b']
					# if not np.all(overlap_A @ np.array(dense_graph.nodes[node_A]['config_position']).T < overlap_b.reshape(-1, 1)) \
					# 	or not np.all(overlap_A @ np.array(dense_graph.nodes[node_B]['config_position']).T < overlap_b.reshape(-1, 1)):
					# 	continue
				else:
					continue

				prepare_MILP_inputs.append((
					poly_tuple, env, node_A, node_B, tracker, f"seg-0"
				))
	return prepare_MILP_inputs


def solve_batch_MILP_offline(dense_graph: nx.Graph, MILP_inputs: list, tracker: SimpleProfiler, **kwargs):
	debug = 0
	if debug:
		for MILP_input in tqdm(MILP_inputs):
			result = solve_growing_MILP(MILP_input)
			if result is not None:
				start_patch, end_patch, path_seg = result
				dense_graph.add_edge(start_patch, end_patch, path=path_seg)
	else:
		# rewrite multiprocessing with map
		with mp.Pool(min((mp.cpu_count() - 8) // 2, len(MILP_inputs))) as pool:
			new_edges = pool.starmap(solve_growing_MILP, [(MILP_input, None) for MILP_input in MILP_inputs])

		for new_edge in new_edges:
			if new_edge is not None:
				start_patch, end_patch, path_seg = new_edge
				dense_graph.add_edge(start_patch, end_patch, path=path_seg)

def solve_growing_MILP(input_dict, return_queue=None):
	current_MILP = create_small_MILP(input_dict)
	while True:
		# solved_flag = current_MILP.solve()
		# if solved_flag:
		solved_flag = current_MILP.solve_with_time_limit(15)
		if solved_flag == 'solved':
			if return_queue is not None:
				return_queue.put((tuple(input_dict['start_config']), tuple(input_dict['end_config']), current_MILP.get_path()[0]))
			return tuple(input_dict['start_config']), tuple(input_dict['end_config']), current_MILP.get_path()[0]
			break
		else:
			return None
			break
			current_MILP.add_waypoint()

		if current_MILP.num_wp > MAX_NUM_WAYPOINT:
			break
	return tuple(input_dict['start_config']), tuple(input_dict['end_config']), current_MILP.get_path()[0]


def create_small_MILP(input_dict) -> Inside_SampleEdge_MILP:
	# TODO: both Ab_list and edge_vert can be optimized by pre-computing
	A_list, b_list = [], []
	for (A, b) in input_dict['Ab_list']:
		A_list.append(A)
		b_list.append(np.array(b).reshape(-1, 1))

	milp_to_solve = Inside_SampleEdge_MILP(
		name=input_dict['name'] if 'name' in input_dict else '',
		env=input_dict['env'], profiler=input_dict['profiler'],
		num_waypoint=0,  #input_dict['init_num_waypoint'] if 'init_num_waypoint' in input_dict else 0,
		start_config=input_dict['start_config'], end_config=input_dict['end_config'],
		##### specific to this formulation #####
		A_list=A_list, b_list=b_list
	)
	return milp_to_solve


def fast_verify_single_patch(dense_graph: nx.Graph, MILP_input: dict):
	'''
	fast verification of a patch with single config with another patch, only allowing 0 intermediate waypoints
	'''
	assert len(MILP_input['start_config']) == 1, "single config only"
	if len(MILP_input['start_config'][0]) == 3:  # 2D
		start_x, start_y, theta = MILP_input['start_config'][0]
		start_p = np.array([start_x, start_y])
		end_configs = np.array(MILP_input['end_config'])
		if len(end_configs.shape) == 1:
			return
		same_theta_mask = np.isclose(end_configs[:, -1], theta)
		end_config_p = end_configs[same_theta_mask, :2]
		end_config_p_idx = np.where(same_theta_mask)[0]
		agent_verts = np.array(MILP_input['env'].agent.boundary.coords[:-1])  # (num_of_verts, 2)
		r_verts = np.stack((agent_verts[:, 0] * np.cos(theta) - agent_verts[:, 1] * np.sin(theta),
							agent_verts[:, 0] * np.sin(theta) + agent_verts[:, 1] * np.cos(theta)), axis=-1)
	else:
		assert len(MILP_input['start_config'][0]) == 4, "2D or 3D only"
		start_p = np.array(MILP_input['start_config'][0][:3, -1])
		rot_mat = MILP_input['start_config'][0][:3, :3]
		end_configs = np.array(MILP_input['end_config'])
		same_rot_mask = np.all(np.isclose(end_configs[:, :3, :3], rot_mat), axis=(-2, -1))
		end_config_p = end_configs[same_rot_mask, :3, -1]
		end_config_p_idx = np.where(same_rot_mask)[0]
		agent_verts = np.array(MILP_input['env'].agent.vertices)  # (num_of_verts, 3)
		r_verts = np.matmul(agent_verts, rot_mat.T)

	# assume the start and end config are all collision-free, so only need to check line segments connecting same vertex
	start_verts = r_verts + start_p  # (num_of_verts, cart_dim)
	start_verts = np.repeat(start_verts[np.newaxis, :, :], len(end_config_p), axis=0)  # (num_of_end_config, num_of_verts, cart_dim)
	end_verts = np.repeat(r_verts[np.newaxis, :, :], len(end_config_p), axis=0) + end_config_p[:, np.newaxis, :]  # (num_of_end_config, num_of_verts, cart_dim)
	to_check_segments = np.linspace(start_verts, end_verts, num=10, axis=-2)  # (num_of_end_config, num_of_verts, sample_num, cart_dim)
	bool_result = check_polytope_inside_convex_polys(to_check_segments, MILP_input['Ab_list'][0], MILP_input['Ab_list'][1])
	if np.any(bool_result):
		tuple_start = (tuple([tuple(col) if MILP_input['env'].cartesian_dim == 3 else col for col in MILP_input['start_config'][0]]), )
		p_diff = np.linalg.norm(end_config_p - start_p, axis=-1)
		end_config_idx = end_config_p_idx[np.argmax(bool_result * (1000 - p_diff))]
		dense_graph.add_edge(tuple_start, tuple(MILP_input['end_config']),
							 path=[MILP_input['start_config'][0], MILP_input['end_config'][end_config_idx]])

