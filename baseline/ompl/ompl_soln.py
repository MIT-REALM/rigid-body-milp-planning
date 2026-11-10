import time

import numpy as np
from ompl import geometric as og
from ompl import base as ob
from matplotlib import pyplot as plt

import trimesh.transformations as tf

def planWithOMPL(env, solver: str) -> (np.ndarray, float, float):
	offline_time = 0
	online_time = 0

	# initialization
	initialization_start_time = time.time()
	if 'rrt' in solver:
		env.ompl_ss.setStartAndGoalStates(env.ompl_start, env.ompl_goal)
		if solver == "ompl-rrt":
			planner = og.RRT(env.ompl_ss.getSpaceInformation())
		elif solver == "ompl-rrt_star":
			planner = og.RRTstar(env.ompl_ss.getSpaceInformation())
		else:
			raise NotImplementedError(f"OMPL_solver {solver} not implemented")
		planner.setGoalBias(0.05)
	elif 'prm' in solver:
		env.ompl_ss.setStartAndGoalStates(env.ompl_random_start, env.ompl_random_goal)
		if solver == "ompl-prm":
			planner = og.PRM(env.ompl_ss.getSpaceInformation())
		else:
			raise NotImplementedError(f"OMPL_solver {solver} not implemented")
	else:
		raise NotImplementedError(f"OMPL_solver {solver} not implemented")
	if 'z' in env.bounds:
		stepsize = 0.1
		resolution = stepsize / max(env.bounds['x'][1] - env.bounds['x'][0], env.bounds['y'][1] - env.bounds['y'][0], env.bounds['z'][1] - env.bounds['z'][0])
	else:
		stepsize = 0.06  # 0.1
		resolution = stepsize / max(env.bounds['x'][1] - env.bounds['x'][0], env.bounds['y'][1] - env.bounds['y'][0])
	env.ompl_ss.getSpaceInformation().setStateValidityCheckingResolution(resolution)  # fraction of max extent, default: 0.01; Granularity of collision checking along an edge.
	env.ompl_ss.setPlanner(planner)
	env.ompl_ss.simplifySolution(False)
	initialization_time = time.time() - initialization_start_time
	if 'rrt' in solver:
		online_time += initialization_time
	elif 'prm' in solver:
		offline_time += initialization_time
	else:
		raise NotImplementedError(f"OMPL_solver {solver} not implemented")

	# presolve for prm
	if 'prm' in solver:  # TODO: prm pre-grow roadmap
		presolve_start_time = time.time()

		# # change sampler
		# # sampler = env.ompl_ss.getSpaceInformation().allocValidStateSampler()
		# si = env.ompl_ss.getSpaceInformation()
		# env.ompl_ss.getSpaceInformation().setValidStateSamplerAllocator(
		# 	ob.StateSamplerAllocator(lambda si: ob.GaussianValidStateSampler(si))
		# )

		env.ompl_ss.setup()
		planner.growRoadmap(1.0)  # 15
		planner.clearQuery()
		presolve_time = time.time() - presolve_start_time
		offline_time += presolve_time

	# solve
	solve_start_time = time.time()
	env.ompl_ss.setStartAndGoalStates(env.ompl_start, env.ompl_goal)
	solved = env.ompl_ss.solve(40.0)
	online_time += time.time() - solve_start_time

	if solved:
		# get solution path
		retrieval_start_time = time.time()
		path_raw = env.ompl_ss.getSolutionPath()
		print("-" * 10, f"{len(path_raw.getStates())} waypoints", "-" * 10)
		path_raw.interpolate(100)
		path = []
		if env.cartesian_dim == 2:
			for state in path_raw.getStates():
				path.append([state.getX(), state.getY(), state.getYaw()])
		elif env.cartesian_dim == 3:
			for state in path_raw.getStates():
				path.append(tf.concatenate_matrices(
					tf.translation_matrix([state.getX(), state.getY(), state.getZ()]),
					tf.quaternion_matrix([state.rotation().x, state.rotation().y, state.rotation().z, state.rotation().w])))
		else:
			raise ValueError("Only 2D and 3D are supported.")
		online_time += time.time() - retrieval_start_time

		return path, offline_time, online_time
	else:
		print("No solution found.")
		return None, offline_time, online_time

class WcoSasmpler(ob.ValidStateSampler):
	pass

