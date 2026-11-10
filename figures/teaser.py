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

from environment import CleanEnv, DrakeEnvironment

from utils.plot2D import plot_environment, plot_planned_path


############################ INIT ################################
def get_env_config(env_name: str) -> (dict, float):
	if env_name == "bugtrap2D":
		bounds = {'x': (-2, 14), 'y': (-3, 9), 'theta': (-np.pi, np.pi)}  # (16, 12)
		characteristic_length = 0.7
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
	seed = 1  # 1
	pl.seed_everything(seed)
	mode = "teaser"

	env_list = [
		["corner2D", "bugtrap2D"],
		["infeas1-2D"],  # , "infeas2-2D", "infeas3-2D", "infeas4-2D"],
		["STL2-2D", "STL3-2D"],
		["src/simple10.yaml", "src/challenging600.yaml"],
	]
	env_name = env_list[0][1]

	all_method_list = [
		"single_milp", "ciris_ours", "tri_ours",
		"ompl-rrt", 'ompl-rrt_star', "ompl-prm",
		'gcs']  # klampt no longer recommended
	method_list = ['ciris_ours']

	L = 1.2
	W = 0.5
	agent_name = "L"
	agent = Polygon([(-L, -0.05), (-L, W), (-L + 0.1, W), (-L + 0.1, 0.05), (L, 0.05), (L, -0.05), (-L, -0.05)])
	drake_agent_box_list = {'size': [[2 * L, 0.1, 1], [0.1, W, 1]],
							'translation': [[0, 0, 0], [-L + 0.05, W / 2, 0]]}

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
		elif method == "gcs":
			Environment = partial(DrakeEnvironment, seed=seed, agent_box_list=drake_agent_box_list)
		else:
			Environment = CleanEnv

		environment = Environment(agent, c_length=None if ".yaml" in env_name else characteristic_length, radius=0.3,
								  env_name=env_name, bounds=bounds)
		environment.start_point = (bounds['x'][0] + 0.35 * (bounds['x'][1] - bounds['x'][0]) - 0.7,
								   (bounds['y'][0] + bounds['y'][1]) / 2, 0)
		environment.end_point = (bounds['x'][1] - 1, bounds['y'][1] - 1.5, np.pi / 2)


		# init plot
		if len(method_list) == 1:
			ax = axes[method_idx]
		else:
			ax = axes[0][method_idx]
		plot_environment(environment, ax)
		ax.set_xlim(bounds['x'])
		ax.set_ylim(bounds['y'])

		assert method == "ciris_ours"
		coarse_graph, dense_graph, offline_time = construct_iris_rdmap(environment, ax=ax)
		path, online_time = inference_with_rdmap(environment, coarse_graph, dense_graph, ax=ax)

		# plot
		try:
			# path = [environment.start_point, environment.end_point]
			plot_planned_path(ax, path, environment.agent, with_line=True)
			ax.xaxis.set_visible(False)
			ax.yaxis.set_visible(False)
			plt.savefig(f"teaser.pdf", bbox_inches='tight')
		except Exception as e:
			print(e)
			pass

	plt.show()
