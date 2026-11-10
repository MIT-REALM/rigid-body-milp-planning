import warnings
import io
from PIL import Image
import time

import open3d as o3d
import numpy as np
import networkx as nx

import trimesh
import trimesh.transformations as tf
import open3d.visualization.rendering as rendering

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

from scipy.spatial import HalfspaceIntersection

from utils.geometry_operation import compute_2D_piecewise_reachable_set
from utils.color_config import *

def plot_point(renderer: rendering.OffscreenRenderer, points: np.ndarray):
	if len(points.shape) == 1:
		points = points.reshape(1, -1)
	assert points.shape[1] == 3

	for point in points:
		sphere = o3d.geometry.TriangleMesh.create_sphere(.1)
		sphere.compute_vertex_normals()
		sphere.translate(point)
		renderer.scene.add_geometry(f"sphere-{time.time()}", sphere, GRAPH_POINT_COLOR)

def plot_line(renderer: rendering.OffscreenRenderer, endpoints, color: list):
	path = trimesh.load_path(np.array(endpoints), face_color=color, line_width=0.1)
	# path.visual = path.visual.to_texture()
	scene.add_geometry(path, geom_name="line")

def plot_poly(scene: rendering.OffscreenRenderer, poly: trimesh.Trimesh, transform, color, mesh_name):
	"""
	:param transformation: (4, 4) matrix
	:param color: (r, g, b, alpha)
	"""
	poly_o3d = poly.as_open3d
	poly_o3d.compute_vertex_normals()
	poly_o3d.transform(transform)

	scene.scene.scene.add_geometry(geometry=poly_o3d, name=mesh_name, material=color)


def plot_ellipse(scene, C, d):
	# Ellipse = {Cx + d | ||x|| < 1}
	ball = trimesh.creation.uv_sphere(radius=1, face_colors=(245, 144, 39, 150))
	ball.visual = ball.visual.to_texture()
	scale_mat = tf.identity_matrix()
	scale_mat[:3, :3] = C
	scene.add_geometry(ball, transform=tf.concatenate_matrices(
		tf.translation_matrix(d), scale_mat))


def plot_intersection(A, b, d, scene):
	ineq = np.hstack((A, -b.T))
	hs = HalfspaceIntersection(ineq, d, incremental=False)
	points = hs.intersections

	polytope = trimesh.points.PointCloud(points).convex_hull
	polytope.visual.face_colors = (0, 0, 200, 100)
	polytope.visual = polytope.visual.to_texture()
	scene.add_geometry(polytope)


# visualize algorithm results
def plot_planned_path(scene: rendering.OffscreenRenderer, path, agent: trimesh.geometry, with_line=False):
	assert len(path[0]) == 4

	plot_poly(scene, agent.copy(), transform=path[0], color=START_COLOR, mesh_name="start")
	plot_poly(scene, agent.copy(), transform=path[-1], color=GOAL_COLOR, mesh_name="end")

	for idx, transform in enumerate(path[1: -1]):
		plot_poly(scene, agent.copy(), color=INTER_COLOR, transform=transform, mesh_name=f"intermediate-{idx}")

	if with_line:
		agent_verts = np.array(agent.vertices)
		for i in range(len(path) - 1):
			# start and end
			if np.allclose(path[i][:3, :3], path[i + 1][:3, :3]):
				rot_mat = path[i][:3, :3]
				r_verts = np.matmul(agent_verts, rot_mat.T)
				start_verts = r_verts + path[i][:3, -1]
				end_verts = r_verts + path[i + 1][:3, -1]
				points = np.concatenate((start_verts, end_verts), axis=0)
				plot_poly(scene, trimesh.points.PointCloud(points).convex_hull, color=INTER_COLOR,
						  transform=tf.identity_matrix(), mesh_name=f"line-{i}")
			else:
				assert np.allclose(path[i][:3, -1], path[i + 1][:3, -1])
				plot_sphere(scene, radius=np.max(np.linalg.norm(agent_verts, axis=1)), translate=path[i][:3, -1], color=INTER_COLOR)
				# print("Different rotation matrices in consecutive transforms.")

def plot_sphere(render: rendering.OffscreenRenderer, radius, translate, color):
	sphere = o3d.geometry.TriangleMesh.create_sphere(radius)
	sphere.compute_vertex_normals()
	sphere.translate(translate)
	render.scene.add_geometry(f"sphere-{time.time()}", sphere, color)

def plot_iris_result(A_list, b_list, C_list, d_list, scene):
	for C, d in zip(C_list, d_list):
		plot_ellipse(C=C, d=d, scene=scene)
	for A, b, d in zip(A_list, b_list, d_list):
		plot_intersection(A=A, b=np.array(b).reshape(1, -1), d=d.reshape(-1), scene=scene)


def plot_graph_on_ax(graph_nodes, node_pos, plot_node_patch: bool, renderer):
	# Plotting nodes
	for node, pos in node_pos.items():
		plot_point(renderer, pos) # Node as a dot

	# # Plotting edges
	# for edge in graph.edges():
	# 	plot_line(renderer, [node_pos[edge[0]], node_pos[edge[1]]], color=[0, 0, 255, 255])

	if plot_node_patch:
		for node_idx, node in enumerate(graph_nodes):
			plot_poly(renderer, node, tf.identity_matrix(), color=GRAPH_POLY_COLOR, mesh_name=f"graph_node_{node_idx}")
