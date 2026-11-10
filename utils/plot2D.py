import warnings

import numpy as np
import networkx as nx

from shapely import affinity, union_all
import shapely.geometry as geom

from matplotlib import pyplot as plt
from matplotlib.patches import Polygon

from utils.geometry_operation import compute_2D_piecewise_reachable_set, get_V_rep_from_H


def plot_environment(env, ax):
	# Continue plotting on 'fig'...
	for i, obs in enumerate(env.obstacles_V):
		plot_poly(ax, obs['geom'], color='pink', alpha=0.5, zorder=20, transform=[0, 0, 0])
	plot_poly(ax, geom.Polygon((
			(5, 2), (5, 5), (0, 5), (5, 2)
		)), color='pink', alpha=0.5, zorder=20, transform=[0, 0, 0])
	ax.set_aspect('equal', adjustable='box')


def plot_line(ax, line, color='gray'):
	x, y = line.xy
	ax.plot(x, y, color=color, linewidth=3, solid_capstyle='round', zorder=1)


def plot_poly(ax, poly: geom.Polygon, transform, color, alpha=1.0, zorder=1):
	"""
	:param transform: (x, y, theta)
	"""
	transformed_poly = affinity.rotate(poly, transform[2], origin=(0, 0), use_radians=True)
	transformed_poly = affinity.translate(transformed_poly, transform[0], transform[1])
	patch = Polygon(np.asarray(transformed_poly.exterior.coords), fc=color, ec="black", alpha=alpha, zorder=zorder)
	ax.add_patch(patch)


def plot_ellipse(C, d, ax: plt.Axes):
	# Ellipse = {Cx + d | ||x|| < 1}
	ts = np.linspace(0, 2 * np.pi)
	points = np.array([np.cos(ts), np.sin(ts)])
	points = C @ points + d.reshape(-1, 1)
	ax.plot(*(points), color='orange', solid_capstyle='round')


def plot_intersection(A, b, d, ax: plt.Axes):
	current_region = get_V_rep_from_H(A, b, d)
	ax.add_patch(Polygon(current_region, color="blue", alpha=0.25))


# visualize algorithm results
def plot_planned_path(ax, path, agent: geom.Polygon, with_line=False, additional=None):
	assert len(path[0]) == 3

	for transform in path:
		plot_poly(ax, agent, transform=transform, color='green', alpha=0.5)

	if with_line:
		for i in range(len(path) - 1):
			# start and end
			patches = []
			ps_poly = affinity.rotate(agent, path[i][2], origin=(0, 0), use_radians=True)
			ps_poly = affinity.translate(ps_poly, path[i][0], path[i][1])
			patches.append(ps_poly)
			pe_poly = affinity.rotate(agent, path[i + 1][2], origin=(0, 0), use_radians=True)
			pe_poly = affinity.translate(pe_poly, path[i + 1][0], path[i + 1][1])
			patches.append(pe_poly)

			# print(path[i][2] / np.pi * 180)
			# reachable set
			if additional is not None:
				assert "avg_theta" in additional
				avg_theta = additional["avg_theta"][i]
				avg_theta_cos = additional["avg_theta_cos"][i]
				avg_theta_sin = additional["avg_theta_sin"][i]
			else:
				if abs(path[i][2] - path[i + 1][2]) < np.pi:
					avg_theta = (path[i][2] + path[i + 1][2]) / 2
				else:
					avg_theta = (path[i][2] + path[i + 1][2]) / 2 - np.pi
				avg_theta_cos = np.cos(avg_theta)
				avg_theta_sin = np.sin(avg_theta)
				# warnings.warn("No additional info provided, using naive average theta")
			vert_segment_lists = compute_2D_piecewise_reachable_set(
				agent,
				start_config=[path[i][0], path[i][1], [np.cos(path[i][2]), np.sin(path[i][2])]],
				end_config=[path[i + 1][0], path[i + 1][1], [np.cos(path[i + 1][2]), np.sin(path[i + 1][2])]],
				avg_theta=[avg_theta_cos, avg_theta_sin]
			)
			if i == 0 or i == len(path) - 2:
				for j in range(len(vert_segment_lists[0]) - 1):
					new_patch = geom.Polygon([
						[vert_segment_lists[0][j], vert_segment_lists[1][j]],
						[vert_segment_lists[4][j], vert_segment_lists[5][j]],
						[vert_segment_lists[4][j + 1], vert_segment_lists[5][j + 1]],
						[vert_segment_lists[0][j + 1], vert_segment_lists[1][j + 1]],
						[vert_segment_lists[0][j], vert_segment_lists[1][j]],
					]).convex_hull
					if new_patch.is_valid and new_patch.area > 1e-5:
						patches.append(new_patch)
			else:
				for j in range(len(vert_segment_lists[0])):
					new_patch = geom.Polygon([
						[path[i][0], path[i][1]],
						[vert_segment_lists[0][j], vert_segment_lists[1][j]],
						[vert_segment_lists[2][j], vert_segment_lists[3][j]],
						[vert_segment_lists[4][j], vert_segment_lists[5][j]],
						[path[i + 1][0], path[i + 1][1]],
						[path[i][0], path[i][1]],
					]).convex_hull
					if new_patch.is_valid and new_patch.area > 1e-5:
						patches.append(new_patch)
			merged_patch = union_all(patches)
			if isinstance(merged_patch, geom.MultiPolygon):
				merged_patch = list(merged_patch.geoms)
				assert len(merged_patch) == 1
				merged_patch = merged_patch[0]
			else:
				assert isinstance(merged_patch, geom.Polygon)
			plot_poly(ax, merged_patch, color='green', alpha=0.3, transform=[0, 0, 0])
			# (xs, ys, xc, yc, xe, ye) = vert_segment_lists
			# for x1, y1, x2, y2 in zip(xs[:-1], ys[:-1], xc[:-1], yc[:-1]):
			# 	ax.plot([x1, x2], [y1, y2], color='green', linewidth=1, solid_capstyle='round')
			# for x1, y1, x2, y2 in zip(xc[:-1], yc[:-1], xe[:-1], ye[:-1]):
			# 	ax.plot([x1, x2], [y1, y2], color='green', linewidth=1, solid_capstyle='round')
			# for x1, y1, x2, y2 in zip(xe[:-1], ye[:-1], xe[1:], ye[1:]):
			# 	ax.plot([x1, x2], [y1, y2], color='green', linewidth=1, solid_capstyle='round')

	plot_poly(ax, agent, transform=path[0], color='red')
	plot_poly(ax, agent, transform=path[-1], color='red')

def plot_iris_result(A_list, b_list, C_list, d_list, ax: plt.Axes):
	for C, d in zip(C_list, d_list):
		plot_ellipse(C, d, ax)
	for A, b, d in zip(A_list, b_list, d_list):
		plot_intersection(A, np.array(b).reshape(1, -1), d.reshape(-1), ax)


def plot_graph_on_ax(graph: nx.Graph, node_pos, plot_node_patch: bool, ax):
	if plot_node_patch:
		for node in list(graph.nodes()):
			plot_poly(ax, node, color='blue', alpha=0.2, transform=[0, 0, 0])

	# Plotting nodes
	for node, pos in node_pos.items():
		plt.scatter(pos[0], pos[1], s=20, c='r')  # Node as a dot

	# Plotting edges
	for edge in graph.edges():
		x0, y0 = node_pos[edge[0]]
		x1, y1 = node_pos[edge[1]]
		plt.plot([x0, x1], [y0, y1], 'b-', lw=2)  # Line for edge


# visualize some metric
def plot_time_division(ax, time_dict, bounds=None, figsize=None):
	category_names = ['Strongly disagree', 'Disagree',
					  'Neither agree nor disagree', 'Agree', 'Strongly agree']
	results = {
		'Question 1': [10, 15, 17, 32, 26],
		'Question 2': [26, 22, 29, 10, 13],
		'Question 3': [35, 37, 7, 2, 19],
		'Question 4': [32, 11, 9, 15, 33],
		'Question 5': [21, 29, 5, 5, 40],
		'Question 6': [8, 19, 5, 30, 38]
	}
	labels = list(results.keys())
	data = np.array(list(results.values()))
	data_cum = data.cumsum(axis=1)
	category_colors = plt.colormaps['RdYlGn'](
		np.linspace(0.15, 0.85, data.shape[1]))

	fig, ax = plt.subplots(figsize=(9.2, 5))
	ax.invert_yaxis()
	ax.xaxis.set_visible(False)
	ax.set_xlim(0, np.sum(data, axis=1).max())

	for i, (colname, color) in enumerate(zip(category_names, category_colors)):
		widths = data[:, i]
		starts = data_cum[:, i] - widths
		rects = ax.barh(labels, widths, left=starts, height=0.5,
						label=colname, color=color)

		r, g, b, _ = color
		text_color = 'white' if r * g * b < 0.5 else 'darkgrey'
		ax.bar_label(rects, label_type='center', color=text_color)
	ax.legend(ncols=len(category_names), bbox_to_anchor=(0, 1),
			  loc='lower left', fontsize='small')

	return fig, ax
