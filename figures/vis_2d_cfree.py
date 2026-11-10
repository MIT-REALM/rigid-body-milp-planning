import os
import pickle

import numpy as np
from shapely.geometry import Polygon
import trimesh
import open3d as o3d
import open3d.visualization.rendering as rendering

from scipy import spatial

from utils.geometry_operation import get_V_rep_from_H
from environment.collision_checking_env import CollisionCheckingEnvironment


def ciris():
	# render = rendering.OffscreenRenderer(840, 640)

	red = rendering.MaterialRecord()
	red.base_color = [1, 0., 0.0, 1.0]
	red.shader = "defaultLit"

	green = rendering.MaterialRecord()
	green.base_color = [0.0, 1, 0.0, 1.0]
	green.shader = "defaultLit"

	blue = rendering.MaterialRecord()
	# blue.base_color = [32 / 255, 150 / 255, 160 / 255, 1]
	blue.base_color = [219 / 255, 159 / 255, 229 / 255, 1]
	# blue.shader = "defaultLitTransparency"
	blue.shader = "defaultLit"

	yellow = rendering.MaterialRecord()
	yellow.base_color = [1.0, 0.75, 0.0, 1.0]
	yellow.shader = "defaultLit"

	white = rendering.MaterialRecord()
	white.base_color = [1.0, 1.0, 1.0, 1.0]
	white.shader = "defaultLitTransparency"  # "defaultLit"

	with open(f"../result/stick-gcs/bugtrap2D.pkl", "rb") as f:
		regions = pickle.load(f)
	t_geom_list = []
	for rid, region in enumerate(regions):
		e = region.MaximumVolumeInscribedEllipsoid()
		v = get_V_rep_from_H(A=region.A(), b=region.b(), d=e.center())
		t_geom_list.append(trimesh.points.PointCloud(v).convex_hull)

	o3d_geom_list = []
	for rid, t_geom in enumerate(t_geom_list):
		o3d_geom = t_geom.as_open3d
		# o3d_geom.compute_triangle_normals()
		o3d_geom.compute_vertex_normals()
		# o3d_geom.paint_uniform_color(np.random.rand(3) * 0.8 + 0.2)
		material = rendering.MaterialRecord()
		material.base_color = [np.random.rand() for _ in range(3)] + [0.6]
		material.shader = "defaultLitTransparency"  # "defaultLit"
		o3d_geom_list.append({'name': f"region_{rid}", 'geometry': o3d_geom, 'material': material})

	# render.scene.show_axes(True)
	frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=4, origin=[-2.2, -3.3, -1])
	o3d.visualization.draw(
		[frame] + o3d_geom_list,
		# [frame, point_cloud],
		# window_name="Sample",
		width=840 * 2 + 200,
		height=640 * 2,
		lookat=[5.5, 3, 0],
		up=[0, np.cos(0.1), np.sin(0.1)],
		eye=[5.5, -7.5, 14],
		bg_color=[1, 1, 1, 1],
		show_ui=True,
		show_skybox=False,
		# front=[0., -0.6, 0.8],
		# zoom=0.6,
	)


def sample():
	L = 0.6
	agent = Polygon([(-L, -0.05), (L, -0.05), (L, 0.05), (-L, 0.05), (-L, -0.05)])
	environment = CollisionCheckingEnvironment(agent, c_length=1, radius=0.3, env_name='bugtrap2D',
											   bounds={'x': (-2, 12 + 2), 'y': (-3, 8 + 1), 'theta': (-np.pi, np.pi)})

	xx, yy, tt = np.meshgrid(np.linspace(-2, 12, 70),
							 np.linspace(-3, 8, 70),
							 np.linspace(-np.pi, np.pi, 40))  # (120, 120, 30)

	if os.path.exists(f"sample_mask_{xx.shape}.pkl"):
		mask = np.load(f"sample_mask_{xx.shape}.pkl", allow_pickle=True)
		assert mask.shape == xx.shape
	else:
		mask = np.zeros_like(xx, dtype=bool)
		for i in range(xx.shape[0]):
			for j in range(xx.shape[1]):
				for k in range(xx.shape[2]):
					if environment._is_configuration_free([xx[i, j, k], yy[i, j, k], tt[i, j, k]]):
						mask[i, j, k] = True
		np.save(f"sample_mask_{xx.shape}.pkl", mask)

	# render = rendering.OffscreenRenderer(840, 640)

	green = rendering.MaterialRecord()
	green.base_color = [0.0, 1, 0.0, 1.0]
	green.shader = "defaultLit"

	blue = rendering.MaterialRecord()
	# blue.base_color = [32 / 255, 150 / 255, 160 / 255, 1]
	blue.base_color = [219 / 255, 159 / 255, 229 / 255, 1]
	# blue.shader = "defaultLitTransparency"
	blue.shader = "defaultLit"

	white = rendering.MaterialRecord()
	white.base_color = [1.0, 1.0, 1.0, 1.0]
	white.shader = "defaultLitTransparency"  # "defaultLit"

	boundary_mask = np.zeros(xx.shape, dtype=bool)
	boundary_mask[0, :, :] = 1
	boundary_mask[-1, :, :] = 1
	boundary_mask[:, 0, :] = 1
	boundary_mask[:, -1, :] = 1
	boundary_mask[:, :, 0] = 1
	boundary_mask[:, :, -1] = 1
	boundary_mask[1:, :, :] = np.logical_or(boundary_mask[1:, :, :], np.logical_not(mask[:-1, :, :]))
	boundary_mask[:-1, :, :] = np.logical_or(boundary_mask[:-1, :, :], np.logical_not(mask[1:, :, :]))
	boundary_mask[:, 1:, :] = np.logical_or(boundary_mask[:, 1:, :], np.logical_not(mask[:, :-1, :]))
	boundary_mask[:, :-1, :] = np.logical_or(boundary_mask[:, :-1, :], np.logical_not(mask[:, 1:, :]))
	boundary_mask[:, :, 1:] = np.logical_or(boundary_mask[:, :, 1:], np.logical_not(mask[:, :, :-1]))
	boundary_mask[:, :, :-1] = np.logical_or(boundary_mask[:, :, :-1], np.logical_not(mask[:, :, 1:]))
	boundary_mask = np.logical_and(boundary_mask, mask)

	points = np.concatenate((xx[boundary_mask, np.newaxis], yy[boundary_mask, np.newaxis], tt[boundary_mask, np.newaxis]), axis=1)
	pcd = o3d.geometry.PointCloud()
	pcd.points = o3d.utility.Vector3dVector(np.array(points))
	point_cloud = pcd

	coord_distance_mat = spatial.distance.squareform(
		spatial.distance.pdist(np.argwhere(boundary_mask), metric='euclidean'))
	lines = np.argwhere(coord_distance_mat < 1.5)
	line_set = o3d.geometry.LineSet(
		points=pcd.points,
		lines=o3d.utility.Vector2iVector(lines),
	)
	color = [67 / 255, 78 / 255, 79 / 255]
	pcd.colors = o3d.utility.Vector3dVector(np.array([color for _ in range(len(points))]))
	line_set.colors = o3d.utility.Vector3dVector(np.array([color for _ in range(len(lines))]))

	# o3d_geom.compute_triangle_normals()
	# o3d_geom.compute_vertex_normals()
	# o3d_geom = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=9)

	# render.scene.add_geometry(f"region_sample", o3d_geom, blue)
	#
	# render.scene.scene.set_sun_light(direction=[0., 0.0, -1.], color=[1, 1, 1.0], intensity=3000)
	# render.scene.scene.enable_sun_light(True)

	# render.scene.show_axes(True)
	frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=4, origin=[-2.2, -3.3, -1])

	# o3d.visualization.draw_geometries(
	# 	[frame, line_set],
	# 	# [frame, point_cloud],
	# 	# window_name="Sample",
	# 	width=840 * 2 + 200,
	# 	height=640 * 2,
	# 	lookat=[5.5, 3, 0],
	# 	up=[0, np.cos(0.1), np.sin(0.1)],
	# 	front=[0., -0.6, 0.8],
	# 	# point_show_normal=False,
	# 	# mesh_show_wireframe=False,
	# 	# mesh_show_back_face=True
	# 	zoom=0.6,
	# )
	o3d.visualization.draw(
		[frame, line_set],
		# [frame, point_cloud],
		# window_name="Sample",
		width=840 * 2 + 200,
		height=640 * 2,
		lookat=[5.5, 3, 0],
		up=[0, np.cos(0.1), np.sin(0.1)],
		eye=[5.5, -7.5, 14],
		bg_color=[1, 1, 1, 1],
		show_ui=True,
		show_skybox=False,
		# front=[0., -0.6, 0.8],
		# zoom=0.6,
	)


if __name__ == "__main__":
	# ciris()
	sample()
