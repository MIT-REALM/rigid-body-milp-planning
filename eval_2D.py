import os
import time
import pickle
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

sssssstart_time = time.time()
import open3d  # noqa: F401  -- must load before torch; see the macOS note in the README
import pytorch_lightning as pl

from shapely.geometry import Polygon

from algorithm.grow_roadmap import *
from baseline.gcs.iris import iris
from baseline.deprecated.milp_sep_dim import solve_waypoint_seperating_dim

from environment import CleanEnv, CollisionCheckingEnvironment, DrakeEnvironment

from utils.plot2D import plot_environment, plot_planned_path


############################ INIT ################################
def get_env_config(env_name: str) -> (dict, float):
	if env_name == "corner2D":
		l = 4
		bounds = {'x': (0, l), 'y': (0, l), 'theta': (-np.pi, np.pi)}
		characteristic_length = 0.8
	elif env_name == "chull-2D":
		bounds = {'x': (-5, 5), 'y': (-5, 5), 'theta': (-np.pi, np.pi)}
		characteristic_length = 1.0
	elif env_name == "bugtrap2D":
		bounds = {'x': (-2, 28), 'y': (-3, 21), 'theta': (-np.pi, np.pi)}
		characteristic_length = .6
	elif env_name == "STL2-2D":
		characteristic_length = 3.0
		bounds = {'x': (0, 7 * characteristic_length), 'y': (0, 5 * characteristic_length), 'theta': (-np.pi, np.pi)}
	elif env_name == "STL3-2D":
		characteristic_length = 2.0
		bounds = {'x': (0, 10 * characteristic_length * 1.5), 'y': (0, 10 * characteristic_length),
				  'theta': (-np.pi, np.pi)}
	elif "infeas" in env_name:
		bounds = {'x': (0, 10), 'y': (0, 10), 'theta': (-np.pi, np.pi)}
		characteristic_length = None
	elif "simple10" in env_name:
		bounds = {'x': (-4, 12), 'y': (-5, 9), 'theta': (-np.pi, np.pi)}
		characteristic_length = None
	elif "challenging600" in env_name:
		bounds = {'x': (-5, 15), 'y': (-4, 5), 'theta': (-np.pi, np.pi)}
		characteristic_length = None
	else:
		raise NotImplementedError(f"Environment {env_name} not implemented")
	return bounds, characteristic_length


def init_plots(bounds: dict) -> list:
	minx, maxx = bounds['x']
	miny, maxy = bounds['y']
	if len(method_list) == 1:
		width, height = 12, 9
	else:
		max_width, max_height = 12, 5.5
		width, height = max_width, (maxy - miny) * max_width / (maxx - minx)
		if height > 5:
			width, height = (maxx - minx) * max_height / (maxy - miny), max_height
	fig, axes = plt.subplots(1, len(method_list), figsize=(width * len(method_list), height))
	if not isinstance(axes, list):
		axes = [axes]
	return axes


if __name__ == "__main__":
	##### arguments #####
	additional_plot = 1
	seed = 1234
	pl.seed_everything(seed)
	# mode = "vis_env"
	mode = "timing"
	# mode = "teaser"

	env_list = [
		["corner2D", "bugtrap2D"],
		["infeas1-2D", "chull-2D"],  # , "infeas2-2D", "infeas3-2D", "infeas4-2D"],
		["STL2-2D", "STL3-2D"],
		["src/simple10.yaml", "src/challenging600.yaml"],
	]
	env_name = env_list[0][1]

	all_method_list = [
		"single_milp", "ciris_ours", "tri_ours",
		"ompl-rrt", 'ompl-rrt_star', "ompl-prm",
		'klampt-prm', 'klampt-wco',
		'gcs',
		'custom-prm']  # klampt no longer recommended
	# method_list = ['ompl-rrt', 'ompl-prm', 'ciris_ours']  # 'gcs'
	# method_list = ['klampt-wco']
	# method_list = ['ompl-prm']
	# method_list = ['klampt-prm']
	method_list = ['ciris_ours']
	# method_list = ['gcs']

	# L = 0.6  # 0.3, 0.9
	# # stick
	# agent_name = "stick"
	# agent = Polygon([(-L, -0.05), (L, -0.05), (L, 0.05), (-L, 0.05), (-L, -0.05)])
	# drake_agent_box_list = {'size': [[2 * L, 0.1, 1]],
	# 						'translation': [[0, 0, 0]]}

	L = 0.6
	W = 0.4
	agent_name = "L"
	agent = Polygon([(-L, -0.05), (-L, W), (-L + 0.1, W), (-L + 0.1, 0.05), (L, 0.05), (L, -0.05), (-L, -0.05)])
	drake_agent_box_list = {'size': [[2 * L, 0.1, 1], [0.1, W, 1]],
							'translation': [[0, 0, 0], [-L + 0.05, W/2, 0]]}

	# agent = Polygon([(-0.05, -0.05), (L, -0.05), (L, 0.05), (0.05, 0.05),
	# 				 (0.05, L), (-0.05, L), (-0.05, -0.05)])  # L 0.4
	# agent = Polygon([(-L, -L), (L, -L), (L, L), (-L, L), (-L, -L)])   # box 0.25

	# L = 3.5
	# agent_name = "L"
	# agent = Polygon([(-0.05, -0.05), (L, -0.05), (L, 0.05), (0.05, 0.05), (0.05, L), (-0.05, L), (-0.05, -0.05)])
	# L = 5.  # for teaser
	# agent_name = "L"
	# agent = Polygon([(-0.05, -0.05), (L, -0.05), (L, 0.05), (0.05, 0.05), (1.05, L), (1., L), (-0.05, -0.05)])

	bounds, characteristic_length = get_env_config(env_name)
	axes = init_plots(bounds)

	init_time = time.time()
	print(f"Initialization time: {init_time - sssssstart_time:.3f}s.")

	for method_idx, method in enumerate(method_list):
		print('=' * 20, '\n', f"Running {method} on {env_name}...")
		# init env
		if 'ompl' in method:
			from environment import OMPLEnvironment
			Environment = OMPLEnvironment
		elif 'klampt' in method:
			from environment import KlamptEnvironment
			Environment = partial(KlamptEnvironment, method_name=method)
		elif method == "gcs":
			Environment = partial(DrakeEnvironment, seed=seed, agent_box_list=drake_agent_box_list)
		elif 'custom' in method:
			Environment = CollisionCheckingEnvironment
		else:
			Environment = CleanEnv

		environment = Environment(agent, c_length=None if ".yaml" in env_name else characteristic_length, radius=0.3,
								  env_name=env_name, bounds=bounds)

		# init plot
		if len(method_list) == 1:
			ax = axes[method_idx]
		else:
			ax = axes[0][method_idx]
		plot_environment(environment, ax)
		ax.set_xlim(bounds['x'])
		ax.set_ylim(bounds['y'])

		# solve
		if method == "single_milp":
			seed_points = environment.sample_free(5, sample_in_config=False)
			# seed_points = np.array([[-2, -4], [4, 8], [10.5, 5.5]])
			A_list, b_list, C_list, d_list, sol_found = iris(seed_points=seed_points, env=environment)
			# milp
			path, path_raw = solve_waypoint_seperating_dim(environment, maximum_piece=30, A_list=A_list, b_list=b_list,
														   milp_formulation="naive_bigM_sampleEdge")
			total_time = time.time() - start_time
			plot_planned_path(ax, path, environment.agent_dense, with_line=True, additional=path_raw)
			ax.title(f"Total time: {total_time:.3f}s.")
		elif 'ompl' in method:
			from baseline import planWithOMPL
			online_time_list = []
			trials = 1
			for _ in range(trials):
				path, offline_time, online_time = planWithOMPL(environment, solver=method)
				online_time_list.append(online_time)
			online_time = np.array(online_time_list).mean()
			print(f"Online time: mean {1000 * np.array(online_time_list).mean():.1f}ms, "
				  f"std {1000 * np.std(np.array(online_time_list)):.1f}ms.")
		elif 'klampt' in method:
			from baseline import planWithKlampt
			online_time_list = []
			trials = 5
			for _ in range(trials):
				path, offline_time, online_time = planWithKlampt(environment, solver=method)
				online_time_list.append(online_time)
			online_time = np.array(online_time_list).mean()
			print(f"Online time: mean {1000 * np.array(online_time_list).mean():.1f}ms, "
				  f"std {1000 * np.std(np.array(online_time_list)):.1f}ms.")
		elif 'custom' in method:
			from baseline import planWithCustomPRM
			print('Correct execution route')
			path, offline_time, online_time = planWithCustomPRM(environment)
		elif method == 'gcs':
			from baseline import planWithGCS

			path, offline_time, online_time = planWithGCS(environment, agent_name)
		elif 'ours' in method:
			assert method == "ciris_ours"
			"""
			'2d-corner': stick - 0.6
			'2d-bugtrap': stick - 0.6
			'2d-STL2': stick - 0.6
			'2d-STL3': stick - 0.6
			'2d-peginhole': 
			"""
			file_name = f"result/{agent_name}-env_vis/{env_name}.pkl"
			# Reusing a cached roadmap means no offline work happens in this run,
			# so offline_time must still be defined for the plot title below.
			offline_time = 0
			if mode == "vis_env":
				with open(file_name, 'rb') as f:
					coarse_graph, dense_graph = pickle.load(f)
			else:
				assert mode == "timing"
				if os.path.exists(file_name):
					with open(file_name, 'rb') as f:
						coarse_graph, dense_graph = pickle.load(f)
				else:
					coarse_graph, dense_graph, offline_time = construct_iris_rdmap(environment, ax=ax)
				with open(file_name, 'wb') as f:
					pickle.dump([coarse_graph, dense_graph], f)

			# coarse_graph, dense_graph, offline_time = construct_iris_rdmap(environment, ax=ax)
			path, online_time = inference_with_rdmap(environment, coarse_graph, dense_graph, ax=ax)

		# ax.set_title(f"Time: {end_time - start_time:.5f}s.")
		else:
			raise NotImplementedError(f"Method {method} not implemented")

		# plot
		# try:
		print(f'{len(path)} waypoints')
		plot_planned_path(ax, path, environment.agent, with_line=bool('ours' in method))
		if mode == "vis_env":
			ax.xaxis.set_visible(False)
			ax.yaxis.set_visible(False)
			plt.savefig(f"result/{agent_name}-env_vis/{env_name}.pdf", bbox_inches='tight')

		else:
			ax.set_title(f"Method: {method}, "
						 f"OfflineT: {offline_time:.1f}s, "
						 f"OnlineT: {online_time * 1000:.1f}ms."
						 )
			if method == 'gcs':
				plt.savefig(f"result/{agent_name}-gcs/{env_name}.png", bbox_inches='tight')
		# except Exception as e:
		# 	print(e)
		# 	print(f"online_time: {online_time * 1000:.1f}ms.")
		# 	print("No path found")

	plt.show()
