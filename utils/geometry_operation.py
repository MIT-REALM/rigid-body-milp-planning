import numpy as np
# import cdd  # https://pycddlib.readthedocs.io/en/latest/quickstart.html
# import pypoman  # based on cddlib, don't bother to use pypoman
import scipy

import shapely
import trimesh

from utils.config import DELTA_THETA_UL


def get_H_rep_from_V(verts: np.ndarray) -> (np.ndarray, np.ndarray):
	# H-rep of polytope, Ax<=b
	try:
		verts_cdd = np.concatenate((np.ones((verts.shape[0], 1)), verts), axis=1)
		V_cdd = cdd.Matrix(verts_cdd, number_type='float')
		V_cdd.rep_type = cdd.RepType.GENERATOR
		poly = cdd.Polyhedron(V_cdd)
		H_cdd = np.array(poly.get_inequalities(), dtype=np.float32)
		A, b = -H_cdd[:, 1:], H_cdd[:, 0]
	except:
		equations_scipy = scipy.spatial.ConvexHull(verts).equations
		A, b = equations_scipy[:, :-1], -equations_scipy[:, -1]
	return A, b

def get_V_rep_from_H(A: np.ndarray, b: np.ndarray, d: np.ndarray) -> np.ndarray:
	ineq = np.hstack((A, -b.reshape(-1, 1)))
	hs = scipy.spatial.HalfspaceIntersection(ineq, d, incremental=False)
	points = hs.intersections
	centered_points = points - d
	thetas = np.arctan2(centered_points[:, 1], centered_points[:, 0])
	idxs = np.argsort(thetas)
	v = points[idxs]
	return v

def get_convex_poly_intersection(poly_list):
	if isinstance(poly_list[0], shapely.geometry.Polygon):
		intersection = shapely.intersection_all(poly_list)
		return intersection.area > 1e-3, intersection
	else:
		intersection = trimesh.boolean.intersection(poly_list, engine='manifold')
		return intersection.vertices.shape[0] > 0, intersection


def check_polytope_inside_convex_polys(polytope_verts, poly_1, poly_2):
	"""
	Check if a polytope is inside the union of two convex polytopes, defined by their H-rep.
	:param polytope_verts: may be of multiple dimensions, with the last 3 dim represents a single polytope;
						  (..., num_of_poly_edges, sample_num_on_an_edge, cart_dim)
	:param poly_1 / 2: (poly_A, poly_b) with A @ x <= b
	:return: binary matrix, of shape polytope_verts.shape[:-3], representing if the polytope is inside the union.
	"""
	poly_1_H_A, poly_1_H_b = poly_1
	poly_2_H_A, poly_2_H_b = poly_2

	point_in_poly_1 = np.all(
		np.matmul(polytope_verts, poly_1_H_A.T) <= np.expand_dims(poly_1_H_b, axis=[i for i in range(polytope_verts.ndim - 4)]),
		axis=-1)
	point_in_poly_2 = np.all(
		np.matmul(polytope_verts, poly_2_H_A.T) <= np.expand_dims(poly_2_H_b, axis=[i for i in range(polytope_verts.ndim - 4)]),
		axis=-1)
	edge_basic = np.all(np.logical_or(point_in_poly_1, point_in_poly_2), axis=-1)
	edge_case1 = np.logical_or(np.all(point_in_poly_1, axis=-1),
							   np.all(point_in_poly_2, axis=-1))  # all samples on an edge belong to same poly
	edge_case2 = np.any(np.logical_and(point_in_poly_1, point_in_poly_2),
						axis=-1)  # one sample on the edge belong to both poly
	free_edge = np.logical_and(edge_basic, np.logical_or(edge_case1, edge_case2))
	free_C = np.all(free_edge, axis=-1)
	return free_C


def check_intersect_line_point(line: shapely.LineString, point: shapely.Point) -> bool:
	"""
	:param point_A / point_B / inside_point: (x, y)
	:return: (ax, ay, b), s.t. ax * x + ay * y <= b, taking the side making 'inside point' satisfying the condition
	"""
	line_coords = np.array(line.coords)
	x1, y1 = line_coords[0, :]
	x2, y2 = line_coords[1, :]
	x, y = list(point.coords)[0]

	if x > x1 and x > x2:
		return False
	if x < x1 and x < x2:
		return False
	if y > y1 and y > y2:
		return False
	if y < y1 and y < y2:
		return False

	# (y - y1) = (y2 - y1) / (x2 - x1) * (x - x1) => (y - y1) * (x2 - x1) = (y2 - y1) * (x - x1)
	return abs((y - y1) * (x2 - x1) - (y2 - y1) * (x - x1)) < 1e-3


def compute_2D_piecewise_reachable_set(agent: shapely.Polygon, start_config, end_config, avg_theta):
	"""
	Compute the reachable set for each piece of the path.
	Translation: strict, convex hull of start and goal polytopes;
	Rotation: over-approximate, convex hulll of vertices and arcs(further over-approximated by polytope).

	x_i = rho_x * cos(theta_i) - rho_y * sin(theta_i) + x_i0
	y_i = rho_x * sin(theta_i) + rho_y * cos(theta_i) + y_i0

	:param start/end_config: [x, y, [cos(theta), sin(theta)]]
	:param avg_theta: [cos(avg_theta), sin(avg_theta)]
	:return [points used for line_segment_list], where line_segment_list_i = [4 line segments]:
			(vert_i_start, vert_i_mid), (vert_i_mid, vert_i_end), (vert_i_start, vert_i+1_start), (vert_i_end, vert_i+1_end)
	"""
	# line_segments = []
	verts = np.array(agent.boundary.coords)

	rho_x, rho_y = verts[:, 0], verts[:, 1]
	rho = np.linalg.norm(verts[:, :2], axis=1)
	with np.errstate(divide='ignore', invalid='ignore'):
		cos_theta_0 = np.divide(rho_x, rho, out=np.ones_like(rho_x), where=~np.isclose(rho, 0))
		sin_theta_0 = np.divide(rho_y, rho, out=np.zeros_like(rho_y), where=~np.isclose(rho, 0))

	# start
	xs = rho_x * start_config[-1][0] - rho_y * start_config[-1][1] + start_config[0]
	ys = rho_x * start_config[-1][1] + rho_y * start_config[-1][0] + start_config[1]
	# end
	xe = rho_x * end_config[-1][0] - rho_y * end_config[-1][1] + end_config[0]
	ye = rho_x * end_config[-1][1] + rho_y * end_config[-1][0] + end_config[1]
	# intermediate
	xc = start_config[0] + rho / np.cos(DELTA_THETA_UL / 2) * (
			avg_theta[0] * cos_theta_0 - avg_theta[1] * sin_theta_0)
	yc = start_config[1] + rho / np.cos(DELTA_THETA_UL / 2) * (
			avg_theta[0] * sin_theta_0 + avg_theta[1] * cos_theta_0)
	line_segments = (xs, ys, xc, yc, xe, ye)

	return line_segments

def compute_3D_piecewise_reachable_set(agent: trimesh.Trimesh, start_xyz, end_xyz, rot_mat):
	"""
	Compute the reachable set for each piece of the path.
	Translation: strict, convex hull of start and goal polytopes;
	Rotation: over-approximate, convex hulll of vertices and arcs(further over-approximated by polytope).

	x_i = rho_x * cos(theta_i) - rho_y * sin(theta_i) + x_i0
	y_i = rho_x * sin(theta_i) + rho_y * cos(theta_i) + y_i0

	:param start/end_config: [x, y, [cos(theta), sin(theta)]]
	:param avg_theta: [cos(avg_theta), sin(avg_theta)]
	:return [points used for line_segment_list], where line_segment_list_i = [4 line segments]:
			(vert_i_start, vert_i_mid), (vert_i_mid, vert_i_end), (vert_i_start, vert_i+1_start), (vert_i_end, vert_i+1_end)
	"""
	# line_segments = []
	verts = np.array(agent.vertices)
	edges_idx = np.array(agent.edges_unique)
	r_verts = sum([verts[:, axis_idx, np.newaxis] @ rot_mat[np.newaxis, :, axis_idx] for axis_idx in range(3)])

	# group 1: at end configuration, edges on agent
	# start
	xs1 = r_verts[edges_idx[:, 0]][:, 0] + end_xyz[0]
	ys1 = r_verts[edges_idx[:, 0]][:, 1] + end_xyz[1]
	zs1 = r_verts[edges_idx[:, 0]][:, 2] + end_xyz[2]
	# end
	xe1 = r_verts[edges_idx[:, 1]][:, 0] + end_xyz[0]
	ye1 = r_verts[edges_idx[:, 1]][:, 1] + end_xyz[1]
	ze1 = r_verts[edges_idx[:, 1]][:, 2] + end_xyz[2]
	line_segments_1 = (xs1, ys1, zs1, xe1, ye1, ze1)

	# group 2: edges connects same vert between start and end configuration
	# start
	xs2 = r_verts[:, 0] + start_xyz[0]
	ys2 = r_verts[:, 1] + start_xyz[1]
	zs2 = r_verts[:, 2] + start_xyz[2]
	# end
	xe2 = r_verts[:, 0] + end_xyz[0]
	ye2 = r_verts[:, 1] + end_xyz[1]
	ze2 = r_verts[:, 2] + end_xyz[2]
	line_segments_2 = (xs2, ys2, zs2, xe2, ye2, ze2)

	return line_segments_1, line_segments_2

def sample_grid_on_facet(facet_verts: np.ndarray, facet_normal: np.ndarray, sample_per_dim: int, poly_H_rep: tuple):
	to_be_filled_idx = np.argmax(np.abs(facet_normal))
	mesh_grid_idx = [i for i in range(3) if i != to_be_filled_idx]
	d = -np.sum(facet_normal * facet_verts[0])  # ax + by + cz + d = 0

	ax1_min, ax1_max = np.min(facet_verts[:, mesh_grid_idx[0]]), np.max(facet_verts[:, mesh_grid_idx[0]])
	ax2_min, ax2_max = np.min(facet_verts[:, mesh_grid_idx[1]]), np.max(facet_verts[:, mesh_grid_idx[1]])
	ax1_grid, ax2_grid = np.meshgrid(np.linspace(ax1_min, ax1_max, sample_per_dim),
									np.linspace(ax2_min, ax2_max, sample_per_dim))
	ax3_grid = -(facet_normal[mesh_grid_idx[0]] * ax1_grid + facet_normal[mesh_grid_idx[1]] * ax2_grid + d) / facet_normal[to_be_filled_idx]

	points = np.zeros((sample_per_dim, sample_per_dim, 3))
	points[:, :, to_be_filled_idx] = ax3_grid
	points[:, :, mesh_grid_idx[0]] = ax1_grid
	points[:, :, mesh_grid_idx[1]] = ax2_grid

	# Create a mask to filter out points outside the facet polygon => outside intersection polygon
	mask = np.ones(ax1_grid.shape, dtype=bool)
	for poly_H in poly_H_rep:
		inside_poly_mask = np.all(np.matmul(points, poly_H[0].T) <= poly_H[1] + 1e-5, axis=-1)
		mask = np.logical_and(mask, inside_poly_mask)

	return points, mask
