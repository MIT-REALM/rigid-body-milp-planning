import shapely
import numpy as np


# CAUTION: for the sake of simplicity of handling constrained CDT, we don't consider the case where
#          the obstacles have overlaps.

###################### structured env ######################
def corner_2Denv(bounds: dict, tunnel_width: float) -> tuple:
	start_point = (bounds['x'][0] + 1, bounds['y'][0] + 0.35, 0)
	end_point = (bounds['x'][1] - 0.35, bounds['y'][1] - 1, np.pi / 2)

	box_list = [
		# [minx, maxx, miny, maxy]
		[bounds['x'][0], bounds['x'][1] - tunnel_width, bounds['y'][0] + tunnel_width, bounds['y'][1]]
	]
	obstacles = [{
		'geom': shapely.geometry.box(minx=box[0], maxx=box[1], miny=box[2], maxy=box[3]),
		'drake_size': [box[1] - box[0], box[3] - box[2], 1],
		'drake_center': [(box[0] + box[1]) / 2, (box[2] + box[3]) / 2, 0]
	} for box in box_list]
	return start_point, end_point, obstacles


def bugtrap_2Denv(bounds: dict, tunnel_width: float) -> tuple:
	start_point = (bounds['x'][0] + 0.35 * (bounds['x'][1] - bounds['x'][0]) - 0.4,
				   (bounds['y'][0] + bounds['y'][1]) / 2, 0)
	end_point = (bounds['x'][1] - 1, bounds['y'][1] - 1.2, np.pi / 2)
	box_list = [
		# [minx, maxx, miny, maxy]
		[bounds['x'][0] + 0.66 * (bounds['x'][1] - bounds['x'][0]),
		 bounds['x'][0] + 0.8 * (bounds['x'][1] - bounds['x'][0]),
		 (bounds['y'][0] + bounds['y'][1] + tunnel_width) / 2, 
		 bounds['y'][1]],
		[bounds['x'][0] + 0.66 * (bounds['x'][1] - bounds['x'][0]),
		 bounds['x'][0] + 0.8 * (bounds['x'][1] - bounds['x'][0]),
		 bounds['y'][0],
		 (bounds['y'][0] + bounds['y'][1] - tunnel_width) / 2],
		[bounds['x'][0] + 0.4 * (bounds['x'][1] - bounds['x'][0]),
		 bounds['x'][0] + 0.5 * (bounds['x'][1] - bounds['x'][0]),
		 (bounds['y'][0] + bounds['y'][1] - 2 * tunnel_width) / 2,
		 (bounds['y'][0] + bounds['y'][1] + 2 * tunnel_width) / 2],
		[bounds['x'][0] + 0.15 * (bounds['x'][1] - bounds['x'][0]),
		 bounds['x'][0] + 0.5 * (bounds['x'][1] - bounds['x'][0]),
		 (bounds['y'][0] + bounds['y'][1] + 2 * tunnel_width) / 2,
		 bounds['y'][1] - 3 * tunnel_width],
		[bounds['x'][0] + 0.15 * (bounds['x'][1] - bounds['x'][0]),
		 bounds['x'][0] + 0.5 * (bounds['x'][1] - bounds['x'][0]),
		 bounds['y'][0] + 3 * tunnel_width,
		 (bounds['y'][0] + bounds['y'][1] - 2 * tunnel_width) / 2]
	]
	obstacles = [{
		'geom': shapely.geometry.box(minx=box[0], maxx=box[1], miny=box[2], maxy=box[3]),
		'drake_size': [box[1] - box[0], box[3] - box[2], 1],
		'drake_center': [(box[0] + box[1]) / 2, (box[2] + box[3]) / 2, 0]
	} for box in box_list]
	return start_point, end_point, obstacles

def chull_2Denv(bounds: dict, tunnel_width: float) -> tuple:
	start_point = (-3, 3, -np.pi/2)
	end_point = (3, -3, np.pi/2)
	radius = 1.2
	obstacles = [{
		'geom': shapely.geometry.Polygon((
			(radius * np.cos(np.pi/10 * i), radius * np.sin(np.pi/10 * i)) for i in range(20 + 1)
		))},{
		'geom': shapely.geometry.Polygon((
			(-5, 2), (-5, 5), (0, 5), (-5, 2)
		))},
		# {
		# 'geom': shapely.geometry.Polygon((
		# 	(5, 1), (5, 5), (2, 5), (5, 1)
		# ))}
	]
	return start_point, end_point, obstacles


def infeas1_2Denv(bounds: dict) -> tuple:  # peg-in-hole
	start_point = (bounds['x'][0] + 0.6, (bounds['y'][0] + bounds['y'][1]) / 2, np.pi / 2)
	end_point = (bounds['x'][1] - 2, (bounds['y'][0] + bounds['y'][1]) / 2, 0)
	obstacles = [
		{'geom': shapely.geometry.box(minx=1.6, maxx=3.8, miny=6, maxy=8)},
		{'geom': shapely.geometry.box(minx=3.8, maxx=6, miny=2, maxy=6)},
		{'geom': shapely.geometry.box(minx=7.6, maxx=10, miny=0, maxy=4.8)},
		{'geom': shapely.geometry.box(minx=7.6, maxx=10, miny=5.2, maxy=10)}
	]
	box_list = [
		# [minx, maxx, miny, maxy]
		[1.6, 3.8, 6, 8],
		[3.8, 6, 2, 6],
		[7.6, 10, 0, 4.8],
		[7.6, 10, 5.2, 10]
	]
	obstacles = [{
		'geom': shapely.geometry.box(minx=box[0], maxx=box[1], miny=box[2], maxy=box[3]),
		'drake_size': [box[1] - box[0], box[3] - box[2], 1],
		'drake_center': [(box[0] + box[1]) / 2, (box[2] + box[3]) / 2, 0]
	} for box in box_list]
	return start_point, end_point, obstacles


def STL2_2Denv(bounds: dict, tunnel_width: float) -> tuple:
	wall_width = 1.
	start_point = (bounds['x'][0] + 0.5 * tunnel_width, bounds['y'][0] + 3.5 * tunnel_width, np.pi / 2)
	end_point = (bounds['x'][0] + 6.5 * tunnel_width, bounds['y'][0] + 4.5 * tunnel_width, np.pi / 2)
	box_list = [
		# [minx, maxx, miny, maxy]
		[bounds['x'][0], bounds['x'][0] + wall_width,
		 bounds['y'][0] + wall_width, bounds['y'][0] + 3 * tunnel_width],
		[bounds['x'][0], bounds['x'][0] + wall_width,
		 bounds['y'][0] + 4 * tunnel_width, bounds['y'][1] - wall_width],
		[bounds['x'][0] + wall_width, bounds['x'][0] + 6 * tunnel_width - wall_width,
		 bounds['y'][1] - wall_width, bounds['y'][1]],
		[bounds['x'][0] + wall_width, bounds['x'][0] + 5 * tunnel_width - wall_width,
		 bounds['y'][0], bounds['y'][0] + wall_width],
		[bounds['x'][1] - wall_width, bounds['x'][1],
		 bounds['y'][0] + wall_width, bounds['y'][1]],
		[bounds['x'][0] + tunnel_width, bounds['x'][0] + 3 * tunnel_width,
		 bounds['y'][0] + tunnel_width - wall_width, bounds['y'][0] + tunnel_width],
		[bounds['x'][0], bounds['x'][0] + wall_width,
		 bounds['y'][0], bounds['y'][0] + wall_width],
		[bounds['x'][1] - wall_width, bounds['x'][1],
		 bounds['y'][0], bounds['y'][0] + wall_width],
		[bounds['x'][0], bounds['x'][0] + wall_width,
		 bounds['y'][1] - wall_width, bounds['y'][1]],
		[bounds['x'][0] + 6 * tunnel_width - wall_width, bounds['x'][0] + 6 * tunnel_width,
		 bounds['y'][0] + 2 * tunnel_width, bounds['y'][1] - wall_width],
		############################################## 10 ###########################################
		[bounds['x'][0] + 6 * tunnel_width - wall_width, bounds['x'][0] + 6 * tunnel_width,
		 bounds['y'][1] - wall_width, bounds['y'][1]],
		[bounds['x'][0] + 6 * tunnel_width - wall_width, bounds['x'][0] + 6 * tunnel_width,
		 bounds['y'][1] - wall_width - 3 * tunnel_width, bounds['y'][1] - 3 * tunnel_width],
		[bounds['x'][0] + 5 * tunnel_width, bounds['x'][0] + 6 * tunnel_width - wall_width,
		 bounds['y'][1] - wall_width - 3 * tunnel_width, bounds['y'][1] - 3 * tunnel_width],
		[bounds['x'][0] + 5 * tunnel_width - wall_width, bounds['x'][0] + 5 * tunnel_width,
		 bounds['y'][0] + 2 * tunnel_width, bounds['y'][1] - tunnel_width - wall_width],
		[bounds['x'][0] + 5 * tunnel_width - wall_width, bounds['x'][0] + 5 * tunnel_width,
		 bounds['y'][1] - tunnel_width - wall_width, bounds['y'][1] - tunnel_width],
		[bounds['x'][0] + 3 * tunnel_width, bounds['x'][0] + 5 * tunnel_width - wall_width,
		 bounds['y'][1] - tunnel_width - wall_width, bounds['y'][1] - tunnel_width],
		[bounds['x'][0] + 3 * tunnel_width - wall_width, bounds['x'][0] + 3 * tunnel_width,
		 bounds['y'][1] - tunnel_width - wall_width, bounds['y'][1] - tunnel_width],
		[bounds['x'][0] + tunnel_width, bounds['x'][0] + 3 * tunnel_width - wall_width,
		 bounds['y'][1] - tunnel_width - wall_width, bounds['y'][1] - tunnel_width],
		[bounds['x'][0] + 3 * tunnel_width - wall_width, bounds['x'][0] + 3 * tunnel_width,
		 bounds['y'][1] - 2 * tunnel_width - wall_width, bounds['y'][1] - tunnel_width - wall_width],
		[bounds['x'][0] + 5 * tunnel_width - wall_width, bounds['x'][0] + 5 * tunnel_width,
		 bounds['y'][0], bounds['y'][0] + wall_width],
		############################################## 20 ###########################################
		[bounds['x'][0] + 5 * tunnel_width, bounds['x'][1] - wall_width,
		 bounds['y'][0], bounds['y'][0] + wall_width],
		[bounds['x'][0] + 5 * tunnel_width - wall_width, bounds['x'][0] + 5 * tunnel_width,
		 bounds['y'][0] + wall_width, bounds['y'][0] + tunnel_width - wall_width],
		[bounds['x'][0] + 5 * tunnel_width - wall_width, bounds['x'][0] + 5 * tunnel_width,
		 bounds['y'][0] + tunnel_width - wall_width, bounds['y'][0] + tunnel_width],
		[bounds['x'][0] + 5 * tunnel_width, bounds['x'][0] + 6 * tunnel_width,
		 bounds['y'][0] + tunnel_width - wall_width, bounds['y'][0] + tunnel_width],
		[bounds['x'][0] + 4 * tunnel_width, bounds['x'][0] + 5 * tunnel_width - wall_width,
		 bounds['y'][0] + tunnel_width - wall_width, bounds['y'][0] + tunnel_width],
		[bounds['x'][0] + 4 * tunnel_width - wall_width, bounds['x'][0] + 4 * tunnel_width,
		 bounds['y'][0] + tunnel_width, bounds['y'][0] + 2 * tunnel_width - wall_width],
		[bounds['x'][0] + 4 * tunnel_width - wall_width, bounds['x'][0] + 4 * tunnel_width,
		 bounds['y'][0] + 2 * tunnel_width - wall_width, bounds['y'][0] + 2 * tunnel_width],
		[bounds['x'][0] + 4 * tunnel_width - wall_width, bounds['x'][0] + 4 * tunnel_width,
		 bounds['y'][0] + 2 * tunnel_width, bounds['y'][0] + 3 * tunnel_width],
		[bounds['x'][0] + 2 * tunnel_width, bounds['x'][0] + 4 * tunnel_width - wall_width,
		 bounds['y'][0] + 2 * tunnel_width - wall_width, bounds['y'][0] + 2 * tunnel_width],
		[bounds['x'][0] + 2 * tunnel_width - wall_width, bounds['x'][0] + 2 * tunnel_width,
		 bounds['y'][0] + 2 * tunnel_width, bounds['y'][0] + 3 * tunnel_width - wall_width],
		############################################## 30 ###########################################
		[bounds['x'][0] + 2 * tunnel_width - wall_width, bounds['x'][0] + 2 * tunnel_width,
		 bounds['y'][0] + 3 * tunnel_width - wall_width, bounds['y'][0] + 3 * tunnel_width],
		[bounds['x'][0] + 1 * tunnel_width, bounds['x'][0] + 2 * tunnel_width - wall_width,
		 bounds['y'][0] + 3 * tunnel_width - wall_width, bounds['y'][0] + 3 * tunnel_width],
		[bounds['x'][0] + 1 * tunnel_width - wall_width, bounds['x'][0] + 1 * tunnel_width,
		 bounds['y'][0] + 3 * tunnel_width - wall_width, bounds['y'][0] + 3 * tunnel_width],
		[bounds['x'][0] + 1 * tunnel_width - wall_width, bounds['x'][0] + 1 * tunnel_width,
		 bounds['y'][0] + 2 * tunnel_width, bounds['y'][0] + 3 * tunnel_width - wall_width],
		## optional
		[bounds['x'][0] + 2 * tunnel_width - wall_width, bounds['x'][0] + 2 * tunnel_width,
		 bounds['y'][0] + 2 * tunnel_width - wall_width, bounds['y'][0] + 2 * tunnel_width],
		[bounds['x'][0] + 5 * tunnel_width - wall_width, bounds['x'][0] + 5 * tunnel_width,
		 bounds['y'][0] + 2 * tunnel_width - wall_width, bounds['y'][0] + 2 * tunnel_width],
		[bounds['x'][0] + 4 * tunnel_width - wall_width, bounds['x'][0] + 4 * tunnel_width,
		 bounds['y'][0] + 1 * tunnel_width - wall_width, bounds['y'][0] + 1 * tunnel_width],
	]
	obstacles = [{
		'geom': shapely.geometry.box(minx=box[0], maxx=box[1], miny=box[2], maxy=box[3]),
		'drake_size': [box[1] - box[0], box[3] - box[2], 1],
		'drake_center': [(box[0] + box[1]) / 2, (box[2] + box[3]) / 2, 0]
	} for box in box_list]
	return start_point, end_point, obstacles


def STL3_2Denv(bounds: dict, tunnel_width: float) -> tuple:
	xy_ratio = 1.5
	start_point = (bounds['x'][0] + 0.4 * tunnel_width * xy_ratio, bounds['y'][0] + 3.5 * tunnel_width, 0)
	end_point = (bounds['x'][0] + 9 * tunnel_width * xy_ratio, bounds['y'][0] + 0.4 * tunnel_width, 0)
	box_list = [
		# [minx, maxx, miny, maxy]
		[bounds['x'][0] + 1 * tunnel_width * xy_ratio, bounds['x'][0] + 1.4 * tunnel_width * xy_ratio,
		 bounds['y'][0] + 0 * tunnel_width, bounds['y'][0] + 9 * tunnel_width],
		[bounds['x'][0] + 2.2 * tunnel_width * xy_ratio, bounds['x'][0] + 2.6 * tunnel_width * xy_ratio,
		 bounds['y'][0] + 0 * tunnel_width, bounds['y'][0] + 5 * tunnel_width],
		[bounds['x'][0] + 2.2 * tunnel_width * xy_ratio, bounds['x'][0] + 2.6 * tunnel_width * xy_ratio,
		 bounds['y'][0] + 6 * tunnel_width, bounds['y'][1]],
		[bounds['x'][0] + 3.4 * tunnel_width * xy_ratio, bounds['x'][0] + 3.8 * tunnel_width * xy_ratio,
		 bounds['y'][0] + 0 * tunnel_width, bounds['y'][0] + 9 * tunnel_width],
		[bounds['x'][0] + 4.6 * tunnel_width * xy_ratio, bounds['x'][0] + 5 * tunnel_width * xy_ratio,
		 bounds['y'][0] + 1 * tunnel_width, bounds['y'][1]],
		[bounds['x'][0] + 5.8 * tunnel_width * xy_ratio, bounds['x'][0] + 6.2 * tunnel_width * xy_ratio,
		 bounds['y'][0] + 7 * tunnel_width, bounds['y'][1]],
		[bounds['x'][0] + 5.8 * tunnel_width * xy_ratio, bounds['x'][0] + 6.2 * tunnel_width * xy_ratio,
		 bounds['y'][0] + 0 * tunnel_width, bounds['y'][0] + 6 * tunnel_width],
		[bounds['x'][0] + 7 * tunnel_width * xy_ratio, bounds['x'][0] + 7.4 * tunnel_width * xy_ratio,
		 bounds['y'][0] + 1 * tunnel_width, bounds['y'][1]],
		[bounds['x'][0] + 8.2 * tunnel_width * xy_ratio, bounds['x'][0] + 8.6 * tunnel_width * xy_ratio,
		 bounds['y'][0] + 0 * tunnel_width, bounds['y'][0] + 3.8 * tunnel_width],
		[bounds['x'][0] + 8.2 * tunnel_width * xy_ratio, bounds['x'][0] + 8.6 * tunnel_width * xy_ratio,
		 bounds['y'][0] + 3.8 * tunnel_width, bounds['y'][0] + 4.4 * tunnel_width],
		############################################## 10 ###########################################
		[bounds['x'][0] + 8.2 * tunnel_width * xy_ratio, bounds['x'][0] + 8.6 * tunnel_width * xy_ratio,
		 bounds['y'][0] + 4.4 * tunnel_width, bounds['y'][0] + 5.8 * tunnel_width],
		[bounds['x'][0] + 8.2 * tunnel_width * xy_ratio, bounds['x'][0] + 8.6 * tunnel_width * xy_ratio,
		 bounds['y'][0] + 5.8 * tunnel_width, bounds['y'][0] + 6.4 * tunnel_width],
		[bounds['x'][0] + 8.6 * tunnel_width * xy_ratio, bounds['x'][0] + 9.4 * tunnel_width * xy_ratio,
		 bounds['y'][0] + 3.8 * tunnel_width, bounds['y'][0] + 4.4 * tunnel_width],
		[bounds['x'][0] + 8.6 * tunnel_width * xy_ratio, bounds['x'][0] + 9.4 * tunnel_width * xy_ratio,
		 bounds['y'][0] + 5.8 * tunnel_width, bounds['y'][0] + 6.4 * tunnel_width],
		[bounds['x'][0] + 8.2 * tunnel_width * xy_ratio, bounds['x'][0] + 8.6 * tunnel_width * xy_ratio,
		 bounds['y'][0] + 6.4 * tunnel_width, bounds['y'][0] + 8 * tunnel_width],
		[bounds['x'][0] + 8.2 * tunnel_width * xy_ratio, bounds['x'][0] + 8.6 * tunnel_width * xy_ratio,
		 bounds['y'][0] + 8 * tunnel_width, bounds['y'][0] + 8.6 * tunnel_width],
		[bounds['x'][0] + 8.6 * tunnel_width * xy_ratio, bounds['x'][0] + 9.4 * tunnel_width * xy_ratio,
		 bounds['y'][0] + 8 * tunnel_width, bounds['y'][0] + 8.6 * tunnel_width],
		[bounds['x'][0] + 9 * tunnel_width * xy_ratio, bounds['x'][1],
		 bounds['y'][0] + 6.9 * tunnel_width, bounds['y'][0] + 7.5 * tunnel_width],
		[bounds['x'][0] + 9 * tunnel_width * xy_ratio, bounds['x'][1],
		 bounds['y'][0] + 4.8 * tunnel_width, bounds['y'][0] + 5.4 * tunnel_width],
		[bounds['x'][0] + 9 * tunnel_width * xy_ratio, bounds['x'][1],
		 bounds['y'][0] + 2.7 * tunnel_width, bounds['y'][0] + 3.3 * tunnel_width],
	]
	obstacles = [{
		'geom': shapely.geometry.box(minx=box[0], maxx=box[1], miny=box[2], maxy=box[3]),
		'drake_size': [box[1] - box[0], box[3] - box[2], 1],
		'drake_center': [(box[0] + box[1]) / 2, (box[2] + box[3]) / 2, 0]
	} for box in box_list]
	return start_point, end_point, obstacles


###################### unstructured env ######################
def simple_env():
	pass
