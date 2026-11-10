import time

import networkx as nx
import numpy as np
import multiprocessing as mp
from functools import partial

from lightning.pytorch.profilers import SimpleProfiler
from matplotlib import pyplot as plt

from environment.clean_env import CleanEnv as Environment

from .pre_processing import construct_visibility_graph, grow_coarse_iris_roadmap, prune_coarse_graph
from .constrained_iris import constrained_iris
from .verify_edges import solve_batch_MILP_offline, prepare_batch_MILP_inputs_offline, fast_verify_single_patch

from utils.color_config import *

FLAG_MULTIPROCESSING = 0
FLAG_ALL_PATH = 1


####################### ONLINE INFERENCE ############################
def inference_with_rdmap(env: Environment, coarse_graph: nx.Graph, dense_graph: nx.Graph, **kwargs) -> (np.ndarray, float):
	online_time = 0
	execution_tracker = SimpleProfiler()

	# prepare MILP inputs
	prepare_start_time = time.time()
	inference_MILP_inputs = []
	start_p = env.start_point[:2] if env.cartesian_dim == 2 else env.start_point[:3, -1]
	end_p = env.end_point[:2] if env.cartesian_dim == 2 else env.end_point[:3, -1]
	for (coarse_node, node_attr) in coarse_graph.nodes(data=True):
		with execution_tracker.profile("prepare_MILP_inputs"):
			if np.all(node_attr['A'] @ start_p <= node_attr['b']):
				# plot_poly(ax, coarse_node, 'green', alpha=0.3)
				for neighbor in coarse_graph.neighbors(coarse_node):
					for patch in coarse_graph.edges[coarse_node, neighbor]['free_patch']:
						inference_MILP_inputs.append({
							'poly': tuple([coarse_node, neighbor]),
							'Ab_list': tuple([[node_attr['A'], node_attr['b']],
										   [coarse_graph.nodes[neighbor]['A'], coarse_graph.nodes[neighbor]['b']]]),
							'env': env, 'start_config': tuple([env.start_point]),
							'end_config': patch, 'profiler': execution_tracker, 'name': "-"
						})
			if np.all(node_attr['A'] @ end_p <= node_attr['b']):
				# plot_poly(ax, coarse_node, 'yellow', alpha=0.3)
				for neighbor in coarse_graph.neighbors(coarse_node):
					for patch in coarse_graph.edges[coarse_node, neighbor]['free_patch']:
						inference_MILP_inputs.append({
							'poly': tuple([coarse_node, neighbor]),
							'Ab_list': tuple([[node_attr['A'], node_attr['b']],
											  [coarse_graph.nodes[neighbor]['A'], coarse_graph.nodes[neighbor]['b']]]),
							'env': env, 'start_config': tuple([env.end_point]),
							'end_config': patch, 'profiler': execution_tracker, 'name': "-"
						})
	online_time += time.time() - prepare_start_time

	# add start and goal to dense graph for inference
	add_node_start_time = time.time()
	with execution_tracker.profile("add_start_goal_to_dense_graph"):
		dense_graph.graph['start_patch'] = (tuple([tuple(col) if env.cartesian_dim == 3 else col for col in env.start_point]), )
		dense_graph.graph['goal_patch'] = (tuple([tuple(col) if env.cartesian_dim == 3 else col for col in env.end_point]), )
		dense_graph.add_node(dense_graph.graph['start_patch'], pos=start_p)
		dense_graph.add_node(dense_graph.graph['goal_patch'], pos=end_p)
	online_time += time.time() - add_node_start_time

	verify_start_time = time.time()
	# verify all patches with multiprocessing
	# with mp.Pool(min(mp.cpu_count() - 8, len(inference_MILP_inputs))) as pool:
	# 	pool.starmap(fast_verify_single_patch, [(dense_graph, MILP_input) for MILP_input in inference_MILP_inputs])
	for MILP_input in inference_MILP_inputs:
		with execution_tracker.profile("verify_single_patch"):
			fast_verify_single_patch(dense_graph, MILP_input)
	# solve_batch_MILP_offline(dense_graph, inference_MILP_inputs, tracker=execution_tracker, debug_ax=ax)
	online_time += time.time() - verify_start_time

	# retrieve the path
	try:
		with execution_tracker.profile("retrieve_coarse_path"):
			retrieval_start_time = time.time()
			patch_path = nx.shortest_path(dense_graph, dense_graph.graph['start_patch'], dense_graph.graph['goal_patch'])
			path = []
			with execution_tracker.profile("retrieve_...._path"):
				for start_patch, end_patch in zip(patch_path[:-1], patch_path[1:]):
					path_seg = dense_graph[start_patch][end_patch]['path']
					if env.cartesian_dim == 2:
						if np.linalg.norm(np.array(start_patch)-path_seg[0], axis=-1).min() > 1e-4:
							assert np.linalg.norm(np.array(start_patch)-path_seg[-1], axis=-1).min() < 1e-4
							path_seg = path_seg[::-1]
					else:
						if np.linalg.norm(np.array(start_patch)-path_seg[0], axis=(-2, -1)).min() > 1e-4:
							assert np.linalg.norm(np.array(start_patch)-path_seg[-1], axis=(-2, -1)).min() < 1e-4
							path_seg = path_seg[::-1]
					path.append(path_seg)
			online_time += time.time() - retrieval_start_time
		with execution_tracker.profile("retrieve_intersection_path"):
			inpatch_retrieval_start_time = time.time()
			for patch_idx in range(len(patch_path) - 2, 0, -1):
				c_graph = dense_graph.nodes[patch_path[patch_idx]]['connectivity_graph']
				if env.cartesian_dim == 2:
					nodes_in_c = list(c_graph.nodes)
					s_node = tuple(nodes_in_c[np.argmin(np.linalg.norm(np.array(nodes_in_c) - path[patch_idx - 1][-1], axis=1))])
					e_node = tuple(nodes_in_c[np.argmin(np.linalg.norm(np.array(nodes_in_c) - path[patch_idx][0], axis=1))])
					path_seg = nx.shortest_path(c_graph, s_node, e_node)
					if not path_seg[0] == s_node:
						assert path_seg[0] == e_node
						path_seg = path_seg[::-1]
				else:
					nodes_in_c = list(c_graph.nodes)
					s_node = tuple(nodes_in_c[np.argmin(np.linalg.norm(np.array(nodes_in_c)[:, :3, -1] - np.array(path[patch_idx - 1][-1])[:3, -1], axis=(-1)))])
					e_node = tuple(nodes_in_c[np.argmin(np.linalg.norm(np.array(nodes_in_c)[:, :3, -1] - np.array(path[patch_idx][0])[:3, -1], axis=(-1)))])
					raw_path_seg = nx.shortest_path(c_graph, s_node, e_node)
					if not raw_path_seg[0] == s_node:
						assert raw_path_seg[0] == e_node
						raw_path_seg = raw_path_seg[::-1]
					path_seg = []
					for translation in raw_path_seg:
						new_seg = np.array(path[patch_idx - 1][-1])
						new_seg[:3, -1] = np.array(translation)[:3, -1]
						path_seg.append(new_seg)
					if not np.allclose(path_seg[-1], path[patch_idx][0]):
						path_seg.append(path[patch_idx][0])
				if len(path_seg) > 2:
					path.insert(patch_idx, path_seg[1:-1])
				else:
					pass
			path = np.concatenate(path, axis=0)
			online_time += time.time() - inpatch_retrieval_start_time

		# count waypoint number
		waypoint_num = 2
		for c_a, c_b, c_c in zip(path[:-2], path[1:-1], path[2:]):
			if env.cartesian_dim == 2:
				positional_diff = np.linalg.norm(c_a[:2] - c_b[:2]) + np.linalg.norm(c_b[:2] - c_c[:2])
			else:
				positional_diff = np.linalg.norm(c_a[:3, -1] - c_b[:3, -1]) + np.linalg.norm(c_b[:3, -1] - c_c[:3, -1])
			if positional_diff > 1e-4:
				waypoint_num += 1
		print('-' * 10, f"Waypoint number: {waypoint_num}.", '-' * 10)
	except nx.NetworkXNoPath:
		path = None
		pass
		if env.cartesian_dim == 2:
			from utils.plot2D import plot_graph_on_ax
			graph_node_pos = {tuple(patch): patch[np.random.randint(len(patch))][:2] for patch in dense_graph.nodes}
		elif env.cartesian_dim == 3:
			from utils.plot3D import plot_graph_on_ax  # plot_poly, plot_line,
			graph_node_pos = {tuple(patch): np.array(patch[np.random.randint(len(patch))])[:3, -1] for patch in dense_graph.nodes}
		else:
			raise NotImplementedError(f"Dimension {env.cartesian_dim} not implemented")
		# print(dense_graph)
		# plot_graph_on_ax(dense_graph, node_pos=graph_node_pos, plot_node_patch=False, **kwargs)
		plot_graph_on_ax(coarse_graph, node_pos={node: coarse_graph.nodes[node]['pos'] for node in coarse_graph.nodes}, plot_node_patch=True, **kwargs)
		import open3d as o3d
		o3d.io.write_image("test.png", kwargs['renderer'].render_to_image(), 9)
		# plt.show()
		exit()
	print(execution_tracker.summary())
	return path, online_time


####################### GROW ROADMAP ############################


def construct_iris_rdmap(env: Environment, **kwargs) -> (nx.Graph, nx.Graph, float):
	if env.cartesian_dim == 2:
		assert "ax" in kwargs
		from utils.plot2D import plot_poly, plot_line, plot_iris_result, plot_graph_on_ax
	elif env.cartesian_dim == 3:
		from utils.plot3D import plot_point, plot_iris_result, plot_graph_on_ax  #plot_poly, plot_line,
	else:
		raise NotImplementedError(f"Dimension {env.cartesian_dim} not implemented")

	offline_time = 0
	execution_tracker = SimpleProfiler()

	# visibility graph
	visibility_start_time = time.time()
	with execution_tracker.profile("construct_visibility_graph"):
		visibility_graph = construct_visibility_graph(node_num=1024, obstacles_H=env.obstacles_H, bounds=env.bounds,
													  bloating_radius= 2 * env.point_radius)
	offline_time += time.time() - visibility_start_time
	print(offline_time, "Visibility graph constructed.")

	# construct coarse graph based on visibility graph
	iris_start_time = time.time()
	coarse_graph = nx.Graph(A_list=[], b_list=[])
	for iteration in range(8):
		with execution_tracker.profile("sample_seeds"):
			init_C, init_d = env.sample_from_vis_graph(visibility_graph, sample_num=5)
			init_C[:, iteration%2] *=2

		with execution_tracker.profile("constrained_iris"):
			if init_d.shape[0] == 0:
				print(f"Warning: No feasible initial d, iteration {iteration}.")
				break
			else:
				A_list, b_list, C_list, d_list = constrained_iris(
					C_array=init_C, d_array=init_d, env=env, profiler=execution_tracker)
				# plot_iris_result(A_list, b_list, C_list, d_list, **kwargs)
		with execution_tracker.profile("construct_coarse_iris_roadmap"):
			grow_coarse_iris_roadmap(G=coarse_graph, A=A_list, b=b_list, C=np.array(C_list), d=d_list,
									 vis_graph=visibility_graph, bloating_radius=env.point_radius,)
	with execution_tracker.profile("prune_coarse_graph"):
		pruned_coarse_graph = prune_coarse_graph(coarse_graph)
		# pruned_coarse_graph = coarse_graph
	offline_time += time.time() - iris_start_time
	print(f"Offline time - coarse graph: {offline_time:.1f}s.")

	# # plot_graph_on_ax(graph=visibility_graph, node_pos={node: visibility_graph.nodes[node]['pos'] for node in visibility_graph.nodes},
	# # 				 plot_node_patch=False, **kwargs)
	# plot_graph_on_ax(pruned_coarse_graph, node_pos={node: pruned_coarse_graph.nodes[node]['pos'] for node in pruned_coarse_graph.nodes}, plot_node_patch=True, **kwargs)
	# plt.show()
	# img_list = []
	# render = kwargs['renderer']
	# import open3d.visualization.rendering as rendering
	# for node in pruned_coarse_graph.nodes:
	# 	plot_graph_on_ax([node], node_pos={node: pruned_coarse_graph.nodes[node]['pos'] for node in pruned_coarse_graph.nodes}, plot_node_patch=True, renderer=render)
	# 	img = render.render_to_image()
	# 	img_list.append(np.asarray(img))
	# 	print(img_list[-1].shape)
	# 	del render
	#
	# 	render = rendering.OffscreenRenderer(720 * 4, 720 * 3)
	# 	render.scene.set_background([1, 1, 1, 1])
	# 	# plot env
	# 	for obs in env.obstacles_V:
	# 		o3d_geom = obs['geom'].copy().as_open3d
	# 		o3d_geom.compute_vertex_normals()
	# 		if not 'ceiling' in obs['name']:
	# 			render.scene.add_geometry(obs['name'], o3d_geom, WALL_COLOR)
	#
	# 	render.setup_camera(vertical_field_of_view=60.0,
	# 						center=[0, 0, 0],
	# 						eye=[0, -4, 10],
	# 						up=[0, 1, 0],
	# 						near_clip=-1,
	# 						far_clip=-1)
	# import open3d as o3d
	# # o3d.io.write_image("testt.png", kwargs['renderer'].render_to_image(), 9)
	# plt.imsave("testt.png", np.mean(img_list, axis=0).astype(np.uint8))
	# plt.imshow(np.mean(img_list, axis=0).astype(np.uint8))
	# plt.show()
	# exit()

	MILP_start_time = time.time()
	dense_graph = nx.Graph()
	with execution_tracker.profile("collect_all_MILPs"):
		# add nodes to dense graph and prepare MILP inputs
		batch_MILP_inputs = prepare_batch_MILP_inputs_offline(env, pruned_coarse_graph, dense_graph, tracker=execution_tracker, **kwargs)

	batch_MILP_inputs = [{
		'poly': m[0],
		'Ab_list': tuple([[pruned_coarse_graph.nodes[poly]['A'], pruned_coarse_graph.nodes[poly]['b']] for poly in m[0]]),
		'env': m[1], 'start_config': m[2], 'end_config': m[3], 'profiler': m[4], 'name': m[5]
	} for m in batch_MILP_inputs]
	offline_time += time.time() - MILP_start_time
	print(f'Info: {len(batch_MILP_inputs)} small MILPs, coarse: {pruned_coarse_graph}, dense: {dense_graph}.')

	solve_MILP_start_time = time.time()
	solve_batch_MILP_offline(dense_graph, batch_MILP_inputs, tracker=execution_tracker, **kwargs)
	offline_time += time.time() - solve_MILP_start_time
	return pruned_coarse_graph, dense_graph, offline_time
