import numpy as np
import networkx as nx

import shapely.geometry as geom
import trimesh
import trimesh.transformations as tf

import scipy
from scipy.ndimage import label as group_nonzero
import scipy.spatial as spatial
from typing import List, Tuple

from utils.config import *
from utils.geometry_operation import *

# DELTA_THETA_UL = np.pi / 12
NUM_THETA_DIV = 40
NUM_BOUNDARY_LINE_DIV = 180

################################ unique functions ################################
def grow_coarse_iris_roadmap(G, A, b, C, d, vis_graph=None, bloating_radius=0) -> None:
	# sort all polygons by area
	C_eigenvalue, C_eigenvector = np.linalg.eig(C)  # eigenvectors are normalized
	area_order = np.argsort(np.prod(C_eigenvalue, axis=1))[::-1]  # descending order

	for idx in area_order:
		# if the area is too small, skip
		if C_eigenvalue[idx].min() <= bloating_radius:
			continue

		new_vertices = get_V_rep_from_H(np.array(A[idx]), np.array(b[idx]), d=np.array(d[idx]))
		if d.shape[1] == 2:  # 2D
			iris_poly = geom.Polygon(new_vertices).convex_hull
		else:
			assert d.shape[1] == 3  # 3D
			iris_poly = trimesh.points.PointCloud(new_vertices).convex_hull

		# exclude iris polygons that already exist in the graph
		can_be_added_flag = True
		existing_nodes = list(G.nodes)
		for existing_node in existing_nodes:
			if np.all(G.nodes[existing_node]['A'] @ d[idx] <= G.nodes[existing_node]['b']) and \
					abs(np.dot(C_eigenvector[idx, :, np.argmax(C_eigenvalue[idx])], G.nodes[existing_node]['major_axis'])) > 0.6:
				can_be_added_flag = False
				break
		if not can_be_added_flag:
			continue

		# add nodes
		G.add_node(iris_poly)
		G.graph['A_list'].append(A[idx])
		G.graph['b_list'].append(b[idx])
		G.nodes[iris_poly]['A'] = A[idx]
		G.nodes[iris_poly]['b'] = b[idx]
		G.nodes[iris_poly]['C'] = C[idx]
		G.nodes[iris_poly]['d'] = d[idx]
		G.nodes[iris_poly]['major_axis'] = C_eigenvector[idx, :, np.argmax(C_eigenvalue[idx])]
		G.nodes[iris_poly]['verts'] = new_vertices
		G.nodes[iris_poly]['pos'] = iris_poly.centroid.coords[0] if d.shape[1] == 2 else iris_poly.centroid

		# add edges
		for candidate_neighbor in existing_nodes:
			intersection = get_convex_poly_intersection([iris_poly, candidate_neighbor])
			if intersection[0]:
				G.add_edge(iris_poly, candidate_neighbor, intersection=intersection[1])

		# remove edge from vis_graph
		if vis_graph is not None and len(vis_graph.edges) > 0:
			g_edges = list(vis_graph.edges())
			edge_points = np.array([vis_graph.edges[edge]['interpolated_points'] for edge in g_edges])  # shape: (edge_num, interpolation_num, dim)
			edge_flags = np.array([vis_graph.edges[edge]['near_coverage_flag'] for edge in g_edges])  # shape: (edge_num, interpolation_num)

			new_iris_flag = np.all(edge_points @ G.nodes[iris_poly]['A'].T <= G.nodes[iris_poly]['b'] + bloating_radius * np.linalg.norm(G.nodes[iris_poly]['A'], axis=1), axis=-1)
			updated_flags = np.logical_or(edge_flags, new_iris_flag)
			vis_graph.add_edges_from([(g_edges[i][0], g_edges[i][1], {
				'interpolated_points': edge_points[i],
				'near_coverage_flag': updated_flags[i],
				'ooc_coverage_points': edge_points[i][~updated_flags[i]],
			}) for i in range(len(g_edges))])  # update edge attributes

			## remove edges that has been covered
			vis_graph.remove_edges_from([g_edges[i] for i in np.argwhere(np.all(updated_flags, axis=-1)).flatten()])

			## remove isolated nodes
			vis_graph.remove_nodes_from(list(nx.isolates(vis_graph)))


def prune_coarse_graph(redundant_graph: nx.Graph) -> nx.Graph:
	redundant_nodes = list(redundant_graph.nodes(data=True))

	G = nx.Graph()

	# sort all polygons by area
	C_eigenvalue, C_eigenvector = np.linalg.eig(np.array([node[1]['C'] for node in redundant_nodes]))  # eigenvectors are normalized
	area_order = np.argsort(np.prod(C_eigenvalue, axis=1))[::-1]  # descending order

	for idx in area_order:
		# exclude iris polygons that already exist in the graph
		can_be_added_flag = True
		existing_nodes = list(G.nodes)
		for existing_node in existing_nodes:
			if np.allclose(redundant_nodes[idx][1]['pos'], G.nodes[existing_node]['pos'], atol=C_eigenvalue[idx].min())\
					and np.allclose(redundant_nodes[idx][1]['major_axis'], G.nodes[existing_node]['major_axis'], atol=0.2):
				can_be_added_flag = False
				break

			if np.all(redundant_nodes[idx][1]['verts'] @ G.nodes[existing_node]['A'].T <= G.nodes[existing_node]['b'] + 0.2 * C_eigenvalue.min() * np.linalg.norm(G.nodes[existing_node]['A'], axis=-1)):
				can_be_added_flag = False
				break

		if not can_be_added_flag:
			continue

		# add nodes
		G.add_node(redundant_nodes[idx][0], **redundant_nodes[idx][1])

		# add edges
		for candidate_neighbor in existing_nodes:
			if redundant_graph.has_edge(redundant_nodes[idx][0], candidate_neighbor):
				G.add_edge(redundant_nodes[idx][0], candidate_neighbor, **redundant_graph.edges[redundant_nodes[idx][0], candidate_neighbor])

	return G


################################ shared functions ################################

def compute_free_C_on_boundary(intersection, agent_edge, poly_1: tuple, poly_2: tuple) -> (list, list):
	"""
	:param intersection: shapely.geom.Polygon or trimesh.Trimesh
	:param agent_edge: (num_of_edges, sample_num, cart_dim)
	:param poly_1 / poly_2: (poly_A, poly_b) with A @ x <= b
	"""
	if agent_edge.shape[-1] == 2:  # 2D
		boundary = np.array(intersection.boundary.coords)
		free_list = []
		graph_list = []

		# free_C: binary matrix, (theta, lambda)
		theta_list = np.linspace(-np.pi, np.pi, NUM_THETA_DIV, endpoint=False)[:, np.newaxis, np.newaxis]  # (NUM_THETA_DIV, 1, 1)

		for start_point, end_point in zip(boundary[:-1], boundary[1:]):
			discretized_boundary = np.linspace(start_point, end_point, NUM_BOUNDARY_LINE_DIV + 1, endpoint=False)[1:]

			bloated_agent_edge = agent_edge / np.cos(2 * np.pi / NUM_THETA_DIV)  # (num_of_edges, sample_num, 2)
			r_verts = np.expand_dims(bloated_agent_edge, axis=0).repeat(NUM_THETA_DIV, axis=0)  # (NUM_THETA_DIV, num_of_edges, sample_num, 2)
			r_verts = np.stack((r_verts[:, :, :, 0] * np.cos(theta_list) - r_verts[:, :, :, 1] * np.sin(theta_list),
								r_verts[:, :, :, 0] * np.sin(theta_list) + r_verts[:, :, :, 1] * np.cos(theta_list)), axis=-1)  # (NUM_THETA_DIV, num_of_edges, sample_num, 2)
			xyr_verts = np.expand_dims(r_verts, axis=1).repeat(NUM_BOUNDARY_LINE_DIV, axis=1).astype(np.float32)  # (NUM_THETA_DIV, NUM_BOUNDARY_LINE_DIV, num_of_edges, sample_num, 2)
			xyr_verts += discretized_boundary[np.newaxis, :, np.newaxis, np.newaxis, :]
			free_C = check_polytope_inside_convex_polys(xyr_verts, poly_1, poly_2)  # (NUM_THETA_DIV, NUM_BOUNDARY_LINE_DIV)

			# handling continuity at \pm pi
			free_C = np.concatenate((free_C, free_C[np.newaxis, 0, :]), axis=0)
			grouped_array, num_groups = group_nonzero(free_C)
			group_index_list = [i for i in range(1, num_groups + 1)]
			for lambda_idx in range(NUM_BOUNDARY_LINE_DIV):
				if grouped_array[0, lambda_idx] and (not grouped_array[0, lambda_idx] == grouped_array[-1, lambda_idx]):
					group_index_list.remove(grouped_array[-1, lambda_idx])
					grouped_array = np.where(grouped_array == grouped_array[-1, lambda_idx], grouped_array[0, lambda_idx], grouped_array)

			# get lambda range for each theta
			for group_idx in group_index_list:
				group_coord = np.argwhere(grouped_array == group_idx)
				graph_list.append(construct_connectivity_graph_in_patch(theta_list.flatten(), discretized_boundary, group_coord))
				graph_list[-1].graph['coord'] = discretized_boundary[np.any(free_C, axis=0)].tolist()
				free_list.append(list(graph_list[-1].nodes))

		# free_list [[points (x, y, theta) in patch 1], [points in patch 2], ...]
		# graph_list [connectivity graph for patch 1, connectivity graph for patch 2, ...]

	else:  # 3D
		# free_C: binary matrix, (rot_mat, face_x, face_y)
		theta_list = ROTATION_MATS  # (rot_n, 3, 3)
		padded_theta_list = np.pad(theta_list, ((0, 0), (0, 1), (0, 1)), mode='constant', constant_values=0)  # (rot_n, 4, 4)
		padded_theta_list[:, 3, 3] = 1
		rot_n = theta_list.shape[0]

		free_list = []
		graph_list = []

		sample_per_dim = 10
		for facet, facet_normal in zip(intersection.facets_boundary, intersection.facets_normal):
			facet_V = intersection.vertices[np.unique(facet.reshape(-1))]
			discretized_boundary, boundary_mask = sample_grid_on_facet(facet_V, facet_normal, sample_per_dim,
														poly_H_rep=(poly_1, poly_2))  # (sample_per_dim, sample_per_dim, 3)

			r_verts = np.expand_dims(agent_edge, axis=0).repeat(rot_n, axis=0)  # (rot_n, num_of_agent_edges, agent_edge_sample_num, 3)
			r_verts = np.einsum('ijk,ibck->ibcj', theta_list,
								r_verts)  # (rot_n, num_of_agent_edges, agent_edge_sample_num, 3)
			xyr_verts = r_verts[:, np.newaxis, np.newaxis].repeat(sample_per_dim, axis=1).repeat(sample_per_dim, axis=2).astype(np.float32)
			xyr_verts += discretized_boundary[np.newaxis, :, :, np.newaxis, np.newaxis, :]  # (rot_mat, inter_dim1, inter_dim2, num_of_agent_edges, agent_edge_sample_num, 3)
			free_C = check_polytope_inside_convex_polys(xyr_verts, poly_1, poly_2)  # free_C: binary matrix, (rot_mat, face_x, face_y)free_C = check_polytope_inside_convex_polys(xyr_verts, poly_1, poly_2)  # free_C: binary matrix, (rot_mat, sample_per_facet)
			free_C = np.logical_and(free_C, boundary_mask)

			# TODO: correct connectivity graph
			all_rot_free_C = np.all(free_C, axis=(0))

			grouped_array, num_groups = group_nonzero(all_rot_free_C)
			group_index_list = [i for i in range(1, num_groups + 1)]

			# get lambda range for each theta
			for group_idx in group_index_list:
				group_coord = np.argwhere(grouped_array == group_idx)
				graph_list.append(
					construct_connectivity_graph_in_patch(padded_theta_list, discretized_boundary, group_coord))
				if len(graph_list[-1].nodes) == 0:
					graph_list.pop()
					continue
				graph_list[-1].graph['coord'] = discretized_boundary[all_rot_free_C].tolist()

				free_p_list = list(graph_list[-1].nodes)
				to_be_add_free_list = [tf.concatenate_matrices(free_p, rot_mat) for free_p in free_p_list for rot_mat in padded_theta_list]
				free_list.append([tuple(tuple(row) for row in mat) for mat in to_be_add_free_list])


		# free_list [[points (4 * 4 transform_mat) in patch 1], [points in patch 2], ...]
		# graph_list [connectivity graph for patch 1, connectivity graph for patch 2, ...]
	return free_list, graph_list


def construct_visibility_graph(node_num: int, obstacles_H, bounds, bloating_radius) -> nx.Graph:
	visible_radius = 0.2 * np.max([bounds['x'][1] - bounds['x'][0], bounds['y'][1] - bounds['y'][0]])
	G = nx.Graph()

	### nodes
	# sample points in the whole space
	sample_num = 5000 #2048
	xs = np.random.uniform(bounds['x'][0], bounds['x'][1], sample_num)
	ys = np.random.uniform(bounds['y'][0], bounds['y'][1], sample_num)
	if 'z' in bounds:
		zs = np.random.uniform(bounds['z'][0], bounds['z'][1], sample_num)
		points = np.vstack((xs, ys, zs)).T
	else:
		points = np.vstack((xs, ys)).T

	# remove points inside obstacles
	for A, b in zip(obstacles_H['A'], obstacles_H['b']):
		indices = np.where(np.all(points @ A.T <= b + bloating_radius * np.linalg.norm(A, axis=1), axis=1))[0]
		points = np.delete(points, indices, axis=0)

	# add nodes
	points = points[np.random.randint(low=0, high=len(points), size=min(node_num, points.shape[0])), :]
	G.add_nodes_from([(tuple(point), {'pos': point}) for point in points])

	### edges
	# find neighbors for each node
	distance_matrix = spatial.distance.squareform(spatial.distance.pdist(points, metric='euclidean'))
	triu_indices = np.triu_indices(distance_matrix.shape[0], k=0)
	distance_matrix[triu_indices] = np.inf

	# add edges
	candidate_edge_pairs_idx = np.argwhere(distance_matrix < visible_radius)
	candidate_edge_start = points[candidate_edge_pairs_idx[:, 0]]
	candidate_edge_end = points[candidate_edge_pairs_idx[:, 1]]

	# check if the line segment intersects with any obstacles
	interpolated_points = np.linspace(candidate_edge_start, candidate_edge_end, num=40, endpoint=False)  # shape: (20, n, point_dim)
	collision_free_flag = np.ones(interpolated_points.shape[1], dtype=bool)
	for A, b in zip(obstacles_H['A'], obstacles_H['b']):
		new_flag = np.any(np.all(interpolated_points @ A.T <= b + bloating_radius * np.linalg.norm(A, axis=-1), axis=-1), axis=0)
		collision_free_flag = np.logical_and(collision_free_flag, ~new_flag)
	G.add_edges_from([(tuple(candidate_edge_start[i]), tuple(candidate_edge_end[i]), {
		'interpolated_points': interpolated_points[:, i],
		'near_coverage_flag': np.zeros(interpolated_points.shape[0], dtype=bool),
		'ooc_coverage_points': interpolated_points[:, i],
	}) for i in np.argwhere(collision_free_flag).reshape(-1)])

	return G


def construct_connectivity_graph_in_patch(theta_array, xy_array, coord_array) -> nx.Graph:
	# TODO: just for the sake of simplicity, no correctness guarantee
	G = nx.Graph()
	if xy_array.shape[-1] == 2:  # 2D
		coord_distance_mat = spatial.distance.squareform(spatial.distance.pdist(coord_array, metric='euclidean'))
		triu_indices = np.triu_indices(coord_distance_mat.shape[0], k=0)
		coord_distance_mat[triu_indices] = np.inf

		# add edges
		pairs = np.argwhere(coord_distance_mat == 1)
		for pair in pairs:
			coord_1, coord_2 = coord_array[pair[0]], coord_array[pair[1]]
			node1 = (xy_array[coord_1[1]][0], xy_array[coord_1[1]][1], theta_array[coord_1[0] % len(theta_array)])
			node2 = (xy_array[coord_2[1]][0], xy_array[coord_2[1]][1], theta_array[coord_2[0] % len(theta_array)])
			G.add_edge(node1, node2)

		node_list = list(G.nodes)
		node_array = np.array(node_list)

		if len(node_array) > 0:
			# add edges for nodes with the same position or orientation
			# position_distance_mat = spatial.distance.squareform(spatial.distance.pdist(node_array[:, :2], metric='euclidean'))
			rotation_distance_mat = spatial.distance.squareform(spatial.distance.pdist(node_array[:, 2:], metric='euclidean'))
			triu_indices = np.triu_indices(rotation_distance_mat.shape[0], k=0)
			# position_distance_mat[triu_indices] = np.inf
			rotation_distance_mat[triu_indices] = np.inf

			pairs = np.argwhere(rotation_distance_mat == 0)
			for pair in pairs:
				G.add_edge(node_list[pair[0]], node_list[pair[1]])

	else:  # 3D
		coord_distance_mat = spatial.distance.squareform(spatial.distance.pdist(coord_array[:, 0, np.newaxis], metric='euclidean'))
		triu_indices = np.triu_indices(coord_distance_mat.shape[0], k=0)
		coord_distance_mat[triu_indices] = np.inf

		# # add edges
		# pairs = np.argwhere(coord_distance_mat == 1)
		# for pair in pairs:
		# 	coord_1, coord_2 = coord_array[pair[0]], coord_array[pair[1]]
		# 	for theta in theta_array:
		# 		node1 = tf.concatenate_matrices(tf.translation_matrix(xy_array[coord_1[0], coord_1[1]]), theta).tolist()
		# 		node2 = tf.concatenate_matrices(tf.translation_matrix(xy_array[coord_2[0], coord_2[1]]), theta).tolist()
		# 		G.add_edge(tuple(tuple(row) for row in node1), tuple(tuple(row) for row in node2))
		#
		# for coord in coord_array:
		# 	for theta_1_idx in range(len(theta_array)):
		# 		for theta_2_idx in range(theta_1_idx + 1, len(theta_array)):
		# 			theta_1, theta_2 = theta_array[theta_1_idx], theta_array[theta_2_idx]
		# 			node1 = tf.concatenate_matrices(tf.translation_matrix(xy_array[coord[0], coord[1]]), theta_1).tolist()
		# 			node2 = tf.concatenate_matrices(tf.translation_matrix(xy_array[coord[0], coord[1]]), theta_2).tolist()
		# 			G.add_edge(tuple(tuple(row) for row in node1), tuple(tuple(row) for row in node2))

		# add edges
		pairs = np.argwhere(coord_distance_mat == 1)
		for pair in pairs:
			coord_1, coord_2 = coord_array[pair[0]], coord_array[pair[1]]
			node1 = tf.translation_matrix(xy_array[coord_1[0], coord_1[1]])
			node2 = tf.translation_matrix(xy_array[coord_2[0], coord_2[1]])
			G.add_edge(tuple(tuple(row) for row in node1), tuple(tuple(row) for row in node2))

	return G

