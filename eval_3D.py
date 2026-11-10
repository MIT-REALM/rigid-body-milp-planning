import os
import time
import pickle

sssssstart_time = time.time()

import trimesh
import trimesh.transformations as tf
import open3d as o3d
import open3d.visualization.rendering as rendering

from algorithm.grow_roadmap import *

from environment import CleanEnv, DrakeEnvironment

from utils.plot3D import plot_planned_path
from utils.color_config import *


############################ INIT ################################
def get_env_config(env_name: str) -> (dict, float):
	if env_name == "demo-3D":
		bounds = {'x': (-6, 6), 'y': (-5, 5), 'z': (0, 3)}
		characteristic_length = 1.0
	else:
		bounds = {'x': (-5, 5), 'y': (-5, 5), 'z': (0, 3)}
		characteristic_length = 1.0
	return bounds, characteristic_length


if __name__ == "__main__":
	##### arguments #####
	additional_plot = 1
	seed = 12
	np.random.seed(seed)

	# mode = "vis_env"
	mode = "timing"

	env_list = ["easy-3D", "dense-3D", "peg-3D", "demo-3D"]
	env_name = env_list[2]

	# all_method_list = [
	# 	"single_milp", "ciris_ours",
	# 	"ompl-rrt", 'ompl-rrt_star', "ompl-prm",
	# 	'klampt-rrt']  # klampt no longer recommended
	method = 'ciris_ours'
	# method = 'ompl-prm'
	# method = 'klampt-prm'
	# method = 'klampt-wco'
	# method = 'gcs'

	# agent_name = 'stick'  # pad
	# agent = trimesh.creation.box([.1, .8, 1.])  # piano bbox  #(1.5, 1.8, 1)  prev (.5, .8, 1.)
	# drake_agent_box_list = {'size': [[0.1, 0.8, 1]], 'translation': [[0, 0, 0]]}

	agent_name = 'L'  # bed
	agent_1 = trimesh.creation.box([.1, .8, 1.])
	agent_2 = trimesh.creation.box([.4, .1, 1.], transform=tf.translation_matrix([.15, -.4, 0]))
	agent = trimesh.boolean.union([agent_1, agent_2])
	drake_agent_box_list = {'size': [[0.1, 0.8, 1], [0.4, 0.1, 1]],
							'translation': [[0, 0, 0], [0.15, -0.4, 0]]}

	bounds, characteristic_length = get_env_config(env_name)

	init_time = time.time()
	print(f"Initialization time: {init_time - sssssstart_time:.3f}s.")

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
	else:
		Environment = CleanEnv
	environment = Environment(agent, c_length=None if ".yaml" in env_name else characteristic_length, radius=0.3,
							  env_name=env_name, bounds=bounds)

	# init renderer
	render = rendering.OffscreenRenderer(720 * 4, 720 * 3)
	render.scene.set_background([1, 1, 1, 1])
	# plot env
	for obs in environment.obstacles_V:
		o3d_geom = obs['geom'].copy().as_open3d
		o3d_geom.compute_vertex_normals()
		if not 'ceiling' in obs['name']:
			render.scene.add_geometry(obs['name'], o3d_geom, WALL_COLOR)

	render.setup_camera(vertical_field_of_view=60.0,
						  center=[0, 0, 0],
						  eye=[0, -4, 10],
						  up=[0, 1, 0],
						  near_clip=-1,
						  far_clip=-1)
	# render.

	# solve
	if 'ompl' in method:
		from baseline import planWithOMPL

		online_time_list = 0
		trials = 1
		for _ in range(trials):
			path, offline_time, online_time = planWithOMPL(environment, solver=method)
			online_time_list += online_time
		online_time = online_time_list / trials
	elif 'klampt' in method:
		from baseline import planWithKlampt

		online_time_list = []
		trials = 1
		for _ in range(trials):
			path, offline_time, online_time = planWithKlampt(environment, solver=method)
			online_time_list.append(online_time)
		online_time = np.array(online_time_list).mean()
		print(f"Online time: mean {1000 * np.array(online_time_list).mean():.1f}ms, "
			  f"std {1000 * np.std(np.array(online_time_list)):.1f}ms.")
	elif method == 'gcs':
		from baseline import planWithGCS

		path, offline_time, online_time = planWithGCS(environment, agent_name)
	elif 'ours' in method:
		assert method == "ciris_ours"
		# file_name = f"result/{agent_name}-env_vis/{env_name}.pkl"
		# if os.path.exists(file_name):
		# 	with open(file_name, 'rb') as f:
		# 		coarse_graph, dense_graph = pickle.load(f)
		# else:
		# 	coarse_graph, dense_graph, offline_time = construct_iris_rdmap(environment, renderer=render)
		# 	with open(file_name, 'wb') as f:
		# 		pickle.dump([coarse_graph, dense_graph], f)

		coarse_graph, dense_graph, offline_time = construct_iris_rdmap(environment, renderer=render)
		path, online_time = inference_with_rdmap(environment, coarse_graph, dense_graph, renderer=render)
	else:
		raise NotImplementedError(f"Method {method} not implemented")

	# plot
	# try:
	with open('peg3d_path.pkl', 'wb') as f:
		pickle.dump(path, f)
	from scipy.spatial.transform import Rotation, Slerp
	with open('peg3d_path.pkl', 'rb') as f:
		path_raw = pickle.load(f)
	def interpolate(a, b, u):
		"""Interpolate between two configurations."""
		x = (1 - u) * a[0] + u * b[0]
		y = (1 - u) * a[1] + u * b[1]
		z = (1 - u) * a[2] + u * b[2]
		rs = Rotation.from_quat([a[3], b[3]])
		slerp_cls = Slerp([0, 1], rs)
		q = slerp_cls(u).as_quat()
		return [x, y, z, [q[0], q[1], q[2], q[3]]]


	path = []
	for i in range(len(path_raw) - 1):
		start_point = [path_raw[i][0, -1], path_raw[i][1, -1], path_raw[i][2, -1], Rotation.from_matrix(path_raw[i][:3, :3]).as_quat()]
		end_point = [path_raw[i+1][0, -1], path_raw[i+1][1, -1], path_raw[i+1][2, -1], Rotation.from_matrix(path_raw[i+1][:3, :3]).as_quat()]
		for t in list(np.linspace(0, 1, 12, endpoint=False)):
			interpolation = interpolate(start_point, end_point, t)
			if len(interpolation) == 4:
				new_point = np.eye(4)
				new_point[:3, :3] = Rotation.from_quat(interpolation[3]).as_matrix()
				new_point[:3, 3] = interpolation[:3]
			else:
				new_point = interpolation
			path.append(new_point)

	import cv2
	from utils.plot3D import plot_poly
	width = 720 * 4
	height = 720 * 3
	video = cv2.VideoWriter('peg3d.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 20.0, (width, height))

	for wp in path:
		del render
		render = rendering.OffscreenRenderer(width, height)
		render.scene.set_background([1, 1, 1, 1])
		# plot env
		for obs in environment.obstacles_V:
			o3d_geom = obs['geom'].copy().as_open3d
			o3d_geom.compute_vertex_normals()
			if not 'ceiling' in obs['name']:
				render.scene.add_geometry(obs['name'], o3d_geom, WALL_COLOR)

		plot_poly(render, agent.copy(), transform=wp, color=INTER_COLOR, mesh_name="start")

		render.setup_camera(vertical_field_of_view=60.0,
							center=[0, 0, 0],
							eye=[0, -4, 10],
							up=[0, 1, 0],
							near_clip=-1,
							far_clip=-1)
		video.write(np.asarray(render.render_to_image()))

	cv2.destroyAllWindows()
	video.release()

	# plot_planned_path(render, path, agent, with_line=bool('ours' in method))
	# if mode == "vis_env":
	# 	figure_name = f"result/{agent_name}-env_vis/{env_name}.png"
	# else:
	# 	# print(
	# 	# 	f"Method: {method}, "
	# 	# 	f"Total OfflineT: {offline_time:.1f}s, "
	# 	# 	f"OnlineT: {online_time * 1000:.1f}ms."
	# 	# )
	# 	if method == 'gcs':
	# 		figure_name = f"result/{agent_name}-gcs/{env_name}.png"
	# 	else:
	# 		figure_name = f"test.png"
	#
	# o3d.io.write_image(figure_name, render.render_to_image(), 9)
	# print(f"Saved to {figure_name}")
