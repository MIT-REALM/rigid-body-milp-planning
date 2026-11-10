from klampt.plan.cspace import MotionPlan

import time

import numpy as np
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt

stepsize = 0.3


def planWithKlampt(env, solver: str) -> (np.ndarray, float, float):
	offline_time = 0
	online_time = 0

	# initialization
	initialization_start_time = time.time()
	if solver == "klampt-rrt":
		planner = MotionPlan(env.klampt_space, type='rrt', perturbationRadius=stepsize, bidirectional=True, connectionThreshold=2)
	elif solver == "klampt-prm":
		planner = MotionPlan(env.klampt_space, type='prm', knn=10, connectionThreshold=2)
		planner.setEndpoints(env.random_start_point, env.random_end_point)
		planner.planMore(2000)
		while len(planner.getRoadmap()[0]) < 2000:
			print(f"Growing roadmap: {len(planner.getRoadmap()[0])}")
			planner.planMore(200)
	elif solver == "klampt-wco":
		planner = MotionPlan(env.klampt_space, type='prm', knn=10, connectionThreshold=2)
		planner.setEndpoints(env.random_start_point, env.random_end_point)
		prev_rdmap_node_num = 0
		prev_rdmap_edge_num = 0
		while len(planner.getRoadmap()[0]) < 2000:
			# print(f"Growing roadmap: {len(planner.getRoadmap()[0])}")
			planner.planMore(10)
			current_rdmap = planner.getRoadmap()
			if len(planner.getRoadmap()[0]) > prev_rdmap_node_num:
				env.klampt_space.update_sampler(
					current_rdmap[0],
					new_nodes=current_rdmap[0][prev_rdmap_node_num:],
					new_edges=current_rdmap[1][prev_rdmap_edge_num:])
				prev_rdmap_node_num = len(current_rdmap[0])
				prev_rdmap_edge_num = len(current_rdmap[1])
	else:
		raise NotImplementedError(f"Klampt_solver {solver} not implemented")

	print('gogogogogo')

	# env.klampt_space.samplers['weights'] = np.ones(len(env.klampt_space.samplers['sampler_list']))
	start_idx = planner.addMilestone(env.klampt_start_point)
	end_idx = planner.addMilestone(env.klampt_end_point)
	start_time = time.time()
	for i in range(60):
		planner.planMore(100)
		path_raw = planner.getPath(milestone1=start_idx, milestone2=end_idx)
		if path_raw is not None:
			print(f'======={i}========, {len(path_raw)}')
			break
	online_time += time.time() - start_time

	if path_raw is not None:
		# interpolation, (not tracking time)
		start_time = time.time()
		path_raw = np.array(path_raw)
		path = []
		for i in range(len(path_raw) - 1):
			start_point = path_raw[i]
			end_point = path_raw[i + 1]
			for t in list(np.linspace(0, 1, 10, endpoint=False)):
				interpolation = env.klampt_space.interpolate(start_point, end_point, t)
				if len(interpolation) == 7:
					new_point = np.eye(4)
					new_point[:3, :3] = Rotation.from_quat(interpolation[3:]).as_matrix()
					new_point[:3, 3] = interpolation[:3]
				else:
					new_point = interpolation
				path.append(new_point)
		online_time += time.time() - start_time

		return np.array(path), offline_time, online_time
	else:
		print("No solution found.")
		return None, offline_time, online_time
