import numpy as np
from scipy.sparse import csc_array
from qpsolvers import solve_qp
from cvxpy import Variable, Problem, Maximize, log_det, SOC
import cvxpy as cp

import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Ellipse
from matplotlib.transforms import Affine2D

from environment.basic_env import BaseEnvironment

from lightning.pytorch.profilers import SimpleProfiler


def iris(seed_points, env: BaseEnvironment, max_iters=40):
	"""
	seed_points (np.ndarray): Shape (bs, 2), list of seed points
	C (np.ndarray): Shape (bs, 2, 2), positive semidefinite matrix (i.e., Ellipse = {Cx + d | ||x|| < 1})
	d (np.ndarray): Shape (bs, 2), center point of the ellipse
	A, b: Ax <= b
	"""
	profiler = SimpleProfiler()

	bs = seed_points.shape[0]
	A_list = []
	b_list = []

	# init ellipsoids
	C_list = np.zeros((bs, 2, 2))
	C_list[:, 0, 0] = env.point_radius
	C_list[:, 1, 1] = env.point_radius
	d_list = seed_points

	# init obstacles with vertices
	O = [np.array(obs['geom'].exterior.coords) for obs in env.obstacles_V]

	iters = 0
	sol_found = True

	for _ in range(1):

		with profiler.profile("SeparatingHyperplanes"):
			A_list, b_list, hyp_sol = SeparatingHyperplanes(C_list, d_list, O)
			if not hyp_sol:
				print('Couldn"t find hyperplanes')
				continue

		with profiler.profile("InscribedEllipsoid"):
			C_list, d_list = InscribedEllipsoid(A_list, b_list)

		# if (np.linalg.det(Cs[-1]) - np.linalg.det(Cs[-2])) / np.linalg.det(Cs[-2]) < tolerance:
		# 	break

	print(profiler.summary())
	return A_list, b_list, C_list, d_list, sol_found


# major functions
def SeparatingHyperplanes(C_list, d_list, O, max_iters=10):
	"""
	Finds a set of separating hyperplanes between a set of convex obstacles and an ellipsoid.
	Distance all defined under the ellipsoidal metric
	Returns:
		np.ndarray: Shape (2, N) (where N is the number of hyperplanes you found),
			each column represents the normal of one hyperplane
			(i.e., a in a.T @ x = b)
		np.ndarray: Shape (N, 1) (where N is the number of hyperplanes you found),
			each row represents the offset of one hyperplane
			(i.e., b in a.T @ x = b)
		bool: True if solution is found within maximum number of iterations, else False
	"""
	C_inv_list = np.linalg.inv(C_list)
	C_inv2_list = C_inv_list @ C_inv_list.transpose((0, 2, 1))

	bs = C_list.shape[0]
	remaining_seed = list(range(bs))
	O_remaining_mask = np.ones((bs, len(O)), dtype=bool)
	A_list = [[] for _ in range(bs)]
	B_list = [[] for _ in range(bs)]
	iters = 0
	solution_found = False

	seed_to_V_distance = np.array([[np.min(np.linalg.norm(C_inv @ (oi - d).T, axis=1)) for oi in O]
								   for C_inv, d in zip(C_inv_list, d_list)])  # shape: (bs, len(O))
	assert seed_to_V_distance.shape == O_remaining_mask.shape

	for _ in range(max_iters):
		iters += 1
		if len(remaining_seed) == 0:
			# print('No more obstacles to check')
			break

		# Find the obstacle with the closest vertex to the center of the ellipsoid
		seed_to_V_distance[~O_remaining_mask] = np.inf
		closest_obs_idx_list = np.argmin(seed_to_V_distance[remaining_seed], axis=1)

		# Find the closest point on this obstacle to the ellipsoid
		x_star = ClosestPointOnObstacle(C_list[remaining_seed], C_inv_list[remaining_seed], d_list[remaining_seed],
										[O[closest_obs_idx] for closest_obs_idx in closest_obs_idx_list])

		# Compute the tangent plane
		new_A, new_B = TangentPlane(C_list[remaining_seed], C_inv2_list[remaining_seed], d_list[remaining_seed], x_star)

		# Update the parameters of the tangent plane to build an intersection of half spaces
		for idx in range(len(remaining_seed)):
			A_list[remaining_seed[idx]].append(new_A[idx].reshape((1, -1)))
			B_list[remaining_seed[idx]].append(new_B[idx])

		# Check if this half plane allows us to check of any other obstacles in the same iteration
		for seed_idx in range(len(remaining_seed) - 1, -1, -1):
			O_remaining_mask[remaining_seed[seed_idx], closest_obs_idx_list[seed_idx]] = False
			for idx in np.nonzero(O_remaining_mask[remaining_seed[seed_idx]])[0]:
				o_rem = O[idx]
				if (o_rem @ new_A[seed_idx] >= new_B[seed_idx]).all():
					O_remaining_mask[remaining_seed[seed_idx], idx] = False
			if not O_remaining_mask[remaining_seed[seed_idx]].any():
				remaining_seed.pop(seed_idx)

	for idx in range(bs):
		A_list[idx] = np.concatenate(A_list[idx], axis=0)
	solution_found = True
	return A_list, B_list, solution_found


def InscribedEllipsoid(A_list, b_list):
	method = ''
	if method == 'drake':  # 21s for 60 problems
		# Imported lazily so that Drake is not needed for the default cvxpy path.
		from pydrake.all import MathematicalProgram, Solve
		n = 2
		C_list, d_list = [], []
		for A, b in zip(A_list, b_list):
			prog = MathematicalProgram()

			C = prog.NewContinuousVariables(n, n)
			d = prog.NewContinuousVariables(n)
			prog.AddPositiveSemidefiniteConstraint(C)
			prog.AddMaximizeLogDeterminantCost(C)

			b = np.array(b).reshape((-1, 1))
			for ai, bi in zip(A, b):
				z0 = bi - ai.T @ d
				z1 = ai @ C
				z = np.concatenate([z0, z1])
				prog.AddLorentzConeConstraint(z)

			result = Solve(prog)

			C_best, d_best = result.GetSolution(C), result.GetSolution(d)
			C_list.append(np.expand_dims(C_best, axis=0))
			d_list.append(np.expand_dims(d_best, axis=0))
	else:
		# cvxpy
		C_list, d_list = [], []
		for A, b in zip(A_list, b_list):
			n = A.shape[1]

			C = Variable((n, n), PSD=True)
			d = Variable(n)
			obj = Maximize(log_det(C))

			constraints = []
			b = np.array(b).reshape((-1, 1))
			for ai, bi in zip(A, b):
				z0 = bi - ai.T @ d
				z1 = ai @ C
				constraints.append(SOC(z0, z1))

			prob = Problem(obj, constraints)
			# prob.solve(solver=cp.SCS, max_iters=100)
			prob.solve(solver=cp.MOSEK)

			C_list.append(np.expand_dims(C.value, axis=0))
			d_list.append(np.expand_dims(d.value, axis=0))
	return np.concatenate(C_list), np.concatenate(d_list)


# helper functions
def ClosestPointOnObstacle(C_list, C_inv_list, d_list, corresponding_o_list):
	"""
	Computes the closest point on a convex polygonal obstacle to an ellipse described by a positive
	semidefinite matrix applied to a unit sphere and a center point.

	Returns:
		numpy.ndArray: Shape (len(d_list), 2), 2D coordinate of the closest point on the
			obstacle to the ellipse
	"""
	# available solvers: [
	# 'highs',0.24
	# 'osqp', 0.17
	# 'proxqp', 0.12
	# 'scs']  0.21s for 200 problems

	results_all = []
	for C, C_inv, d, o in zip(C_list, C_inv_list, d_list, corresponding_o_list):
		# variable: x_tilde (dim: 2) & w (dim: len(o))
		var_dim = 2 + len(o)
		v_tilde = C_inv @ (o - d).T  # shape: (2, len(o))
		A_1 = np.concatenate((np.zeros(2), np.ones(len(o)))).reshape(1, -1)  # shape: (1, 2 + len(o))
		A_2 = np.concatenate((-np.eye(2), v_tilde), axis=1)  # shape: (2, 2 + len(o))
		P = csc_array((np.ones(2), (np.array([0, 1]), np.array([0, 1]))), shape=(var_dim, var_dim))
		G = csc_array((-np.ones(len(o)), (np.arange(2, var_dim), np.arange(2, var_dim))), shape=(var_dim, var_dim))
		result = solve_qp(P=P, q=np.zeros(var_dim),
						  G=G, h=np.zeros(var_dim),  # -w <= 0
						  A=np.concatenate((A_1, A_2), axis=0), b=np.array([[1], [0], [0]]),
						  # sum(w) = 1 && - x_tilde + v_tilde @ w = 0
						  solver='osqp', #'proxqp',
						  verbose=False)
		x_tilde = result[:2]
		# w = result[2:]
		x_star = C @ x_tilde.T + d
		results_all.append(x_star.reshape(1, -1))
	return np.concatenate(results_all, axis=0)


def TangentPlane(C_list, C_inv2_list, d_list, x_star_list):
	a = (2 * C_inv2_list @ np.expand_dims(x_star_list - d_list, axis=-1)).squeeze(axis=-1)
	b = np.einsum("ij,ij->i", a, x_star_list)
	return a, b

# def update_polytope(A, b, d):
# 	global current_region
# 	ineq = np.hstack((A.T, -b))
# 	hs = HalfspaceIntersection(ineq, d, incremental=False)
# 	points = hs.intersections
# 	centered_points = points - d
# 	thetas = np.arctan2(centered_points[:, 1], centered_points[:, 0])
# 	idxs = np.argsort(thetas)
# 	current_region = points[idxs]


# plotting
# def draw(ax):
# 	global seed_point, As, bs, Cs, ds, regions
# 	ax.cla()
# 	ax.set_xlim(limits[0])
# 	ax.set_ylim(limits[1])
# 	ax.set_aspect("equal")
# 	if not (seed_point is None):
# 		ax.scatter([seed_point[0]], [seed_point[1]])
# 	if len(As) > 0:
# 		A = As[-1]
# 		b = bs[-1]
# 		for i in range(len(b)):
# 			w = A[:, i]
# 			intercept = b[i]
# 			xx = np.linspace(*limits[0])
# 			yy = (-w[0] / w[1]) * xx + (intercept / w[1])
# 			ax.additional_plot(xx, yy, color="blue")
# 		draw_intersection(A, b, ds[-1])
