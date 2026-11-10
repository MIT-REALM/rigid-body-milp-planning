import numpy as np
from scipy.sparse import csc_array
from qpsolvers import solve_qp
from cvxpy import Variable, Problem, Maximize, log_det, SOC
import cvxpy as cp

from environment.basic_env import BaseEnvironment

from lightning.pytorch.profilers import SimpleProfiler


def constrained_iris(C_array: np.ndarray, d_array: np.ndarray, env: BaseEnvironment, profiler: SimpleProfiler):
	"""
	C (np.ndarray): Shape (bs, 2, 2) or (bs, 3, 3), positive semidefinite matrix (i.e., Ellipse = {Cx + d | ||x|| < 1})
	d (np.ndarray): Shape (bs, 2) or (bs, 3), center point of the ellipse
	A, b: Ax <= b
	"""
	assert C_array.shape[1] == env.cartesian_dim
	init_seeds = d_array.copy()

	# init obstacles with vertices
	if env.cartesian_dim == 2:
		O = [np.array(obs['geom'].exterior.coords) for obs in env.obstacles_V]
	elif env.cartesian_dim == 3:
		O = [np.array(obs['geom'].vertices) for obs in env.obstacles_V]
	else:
		raise NotImplementedError(f"Dimension {env.cartesian_dim} not implemented")

	with profiler.profile("SeparatingHyperplanes"):
		A_list, b_list, hyp_sol = SeparatingHyperplanes(C_array, d_array, O)
		if not hyp_sol:
			raise RuntimeError('Couldn"t find hyperplanes, please check the input ellipsoid')

	for _ in range(5):
		prev_area = np.linalg.det(C_array)
		with profiler.profile("InscribedEllipsoid"):
			new_C_array, new_d_array = InscribedEllipsoid(A_list, b_list)
			# check whether the seed is inside the new ellipsoid
			inv_new_C_array = np.linalg.inv(new_C_array)
			x0_norm = np.linalg.norm(inv_new_C_array @ (init_seeds - new_d_array)[:, :, np.newaxis], axis=1).flatten()
			C_array = np.where(x0_norm[:, None, None] < 1, new_C_array, C_array)

		with profiler.profile("SeparatingHyperplanes"):
			A_list, b_list, hyp_sol = SeparatingHyperplanes(C_array, d_array, O)
			if not hyp_sol:
				print('Couldn"t find hyperplanes')

		# if np.all(np.linalg.det(C_array) - prev_area < prev_area * 5e-2):
		# 	break
	return A_list, b_list, C_array, d_array


# major functions
def SeparatingHyperplanes(C_list, d_list, O, max_iters=20):
	"""
	Finds a set of separating hyperplanes between a set of convex obstacles and an ellipsoid.
	Distance all defined under the ellipsoidal metric
	Returns:
		np.ndarray: Shape (2 or 3, N) (where N is the number of hyperplanes you found),
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
		A = np.concatenate(A_list[idx], axis=0)
		b = np.array(B_list[idx])
		pariwise_inner_product = np.einsum('ik,jk->ij', A, A)
		closer_and_not_greater_b = (pariwise_inner_product >= 1 - 5e-5) & (b[:, np.newaxis] <= b)
		np.fill_diagonal(closer_and_not_greater_b, False)
		to_be_removed = np.any(closer_and_not_greater_b, axis=0)

		A_list[idx] = A[~to_be_removed]
		B_list[idx] = b[~to_be_removed]
	solution_found = True
	return A_list, B_list, solution_found


def InscribedEllipsoid(A_list, b_list):
	method = 'cvxpy'
	if method == 'drake':  # 21s for 60 problems
		# Imported lazily so that Drake is not needed for the default 'cvxpy' path.
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
		numpy.ndArray: Shape (len(d_list), 2 or 3), 2D coordinate of the closest point on the
			obstacle to the ellipse
	"""
	# available solvers: [
	# 'highs',0.24
	# 'osqp', 0.17
	# 'proxqp', 0.12
	# 'scs']  0.21s for 200 problems

	x_dim = C_list.shape[1]
	results_all = []
	for C, C_inv, d, o in zip(C_list, C_inv_list, d_list, corresponding_o_list):
		# variable: x_tilde (dim: x_dim) & w (dim: len(o))
		var_dim = x_dim + len(o)
		v_tilde = C_inv @ (o - d).T  # shape: (2, len(o))
		A_1 = np.concatenate((np.zeros(x_dim), np.ones(len(o)))).reshape(1, -1)  # shape: (1, x_dim + len(o))
		A_2 = np.concatenate((-np.eye(x_dim), v_tilde), axis=1)  # shape: (x_dim, x_dim + len(o))
		P = csc_array((np.ones(x_dim), (np.arange(x_dim), np.arange(x_dim))), shape=(var_dim, var_dim))
		G = csc_array((-np.ones(len(o)), (np.arange(x_dim, var_dim), np.arange(x_dim, var_dim))), shape=(var_dim, var_dim))
		result = solve_qp(P=P, q=np.zeros(var_dim),
						  G=G, h=np.zeros(var_dim),  # -w <= 0
						  A=np.concatenate((A_1, A_2), axis=0), b=np.array([[1]] + [[0] for _ in range(x_dim)]),
						  # sum(w) = 1 && - x_tilde + v_tilde @ w = 0
						  solver='osqp', #'proxqp',
						  verbose=False)
		x_tilde = result[:x_dim]
		# w = result[2:]
		x_star = C @ x_tilde.T + d
		results_all.append(x_star.reshape(1, -1))
	return np.concatenate(results_all, axis=0)


def TangentPlane(C_list, C_inv2_list, d_list, x_star_list):
	a = (2 * C_inv2_list @ np.expand_dims(x_star_list - d_list, axis=-1)).squeeze(axis=-1)
	b = np.einsum("ij,ij->i", a, x_star_list)

	norm = np.linalg.norm(a, axis=1, keepdims=True)
	a = a / norm
	b = b / norm.flatten()
	return a, b
