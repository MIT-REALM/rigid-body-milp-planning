import trimesh
import trimesh.transformations as tf

import numpy as np

from utils.config import ROTATION_MATS

###################### structured env ######################
def easy_3Denv(bounds: dict, tunnel_width: float) -> tuple:
	# https: // www.youtube.com / watch?v = HdfAzUXvmOQ
	# https://github.com/roboticslibrary/rl-examples/blob/21cef495c3878e4a3a8983ef7b8bd26465fd8caa/rlsg/piano/room.wrl
	start_config = (-3, -3, 1.5, 0)
	end_config = (-3, 2, 1.5, 0)

	start_mat = tf.translation_matrix(start_config[:3])
	start_mat[:3, :3] = ROTATION_MATS[start_config[-1]]
	end_mat = tf.translation_matrix(end_config[:3])
	end_mat[:3, :3] = ROTATION_MATS[end_config[-1]]

	obstacles = []
	for obstacle in _create_easy_wall(name='wall'):
		obstacles.append(_create_trimesh_geom(obstacle))

	return start_mat, end_mat, obstacles


def dense_3Denv(bounds: dict, tunnel_width: float) -> tuple:
	# https: // www.youtube.com / watch?v = HdfAzUXvmOQ
	# https://github.com/roboticslibrary/rl-examples/blob/21cef495c3878e4a3a8983ef7b8bd26465fd8caa/rlsg/piano/room.wrl
	start_config = (-3, -3, 1.5, 0)
	end_config = (-3, 2, 1.5, 0)

	start_mat = tf.translation_matrix(start_config[:3])
	start_mat[:3, :3] = ROTATION_MATS[start_config[-1]]
	end_mat = tf.translation_matrix(end_config[:3])
	end_mat[:3, :3] = ROTATION_MATS[end_config[-1]]

	obstacles = []
	raw_obstacles = (
			_create_hard_wall(name='wall') +
			_create_cupboard(translation=np.array([-4, -1.76, 0]), name='cupboard-1') +
			_create_cupboard(translation=np.array([-0, -1.76, 0]), name='cupboard-2')
	)
	for obstacle in raw_obstacles:
		obstacles.append(_create_trimesh_geom(obstacle))

	return start_mat, end_mat, obstacles

def demo_3Denv(bounds: dict, tunnel_width: float) -> tuple:
	start_config = (-5, -4, 1.5, 0)
	end_config = (5, 4, 1.5, 0)

	start_mat = tf.translation_matrix(start_config[:3])
	start_mat[:3, :3] = ROTATION_MATS[start_config[-1]]
	end_mat = tf.translation_matrix(end_config[:3])
	end_mat[:3, :3] = ROTATION_MATS[end_config[-1]]

	obstacles = []
	raw_obstacles = (
			_create_demo_wall(name='wall')
	)
	for obstacle in raw_obstacles:
		obstacles.append(_create_trimesh_geom(obstacle))

	return start_mat, end_mat, obstacles

def peg_3Denv(bounds: dict, tunnel_width: float) -> tuple:
	start_config = (-4, -4, 2, 0)
	end_config = (0, 3.5, 1.5, 10)

	start_mat = tf.translation_matrix(start_config[:3])
	start_mat[:3, :3] = ROTATION_MATS[start_config[-1]]
	end_mat = tf.translation_matrix(end_config[:3])
	end_mat[:3, :3] = ROTATION_MATS[end_config[-1]]

	name = 'wall'
	obstacles = []
	raw_obstacles = [
		{'name': f"{name}-floor", 'drake_type': 'box', 'drake_size': [10, 10, 0.01],
		 'drake_center': np.array([0, 0, -0.005])},
		{'name': f"{name}-ceiling", 'drake_type': 'box', 'drake_size': [10, 10, 0.01],
		 'drake_center': np.array([0, 0, 3.005])},
		{'name': f"{name}-north", 'drake_type': 'box', 'drake_size': [0.01, 10, 3],
		 'drake_center': np.array([5.005, 0, 1.5])},
		{'name': f"{name}-south", 'drake_type': 'box', 'drake_size': [0.01, 10, 3],
		 'drake_center': np.array([-5.005, 0, 1.5])},
		{'name': f"{name}-west", 'drake_type': 'box', 'drake_size': [10, 0.01, 3],
		 'drake_center': np.array([0, 5.005, 1.5])},
		{'name': f"{name}-east", 'drake_type': 'box', 'drake_size': [10, 0.01, 3],
		 'drake_center': np.array([0, -5.005, 1.5])},
		{'name': f"{name}-div1", 'drake_type': 'box', 'drake_size': [4, 2, 3],
		 'drake_center': np.array([-3, 4, 1.5])},
		{'name': f"{name}-div2", 'drake_type': 'box', 'drake_size': [4, 2, 3],
		 'drake_center': np.array([3, 4, 1.5])},
		{'name': f"{name}-div3", 'drake_type': 'box', 'drake_size': [7, 3, 3],
		 'drake_center': np.array([-1.5, -2, 1.5])},
		{'name': f"{name}-div4", 'drake_type': 'box', 'drake_size': [2, 2, 0.4],
		 'drake_center': np.array([0, 4, 0.2])},
	]
	for obstacle in raw_obstacles:
		obstacles.append(_create_trimesh_geom(obstacle))

	return start_mat, end_mat, obstacles


###################### env components ######################
def _create_trimesh_geom(obstacle):
	if obstacle['drake_type'] == 'box':
		obstacle['geom'] = trimesh.creation.box(obstacle['drake_size'],
												transform=tf.translation_matrix(obstacle['drake_center']))
	else:
		raise NotImplementedError(f"Unknown type: {obstacle['drake_type']}")
	return obstacle


def _create_easy_wall(translation=np.array([0, 0, 0]), name='wall'):
	wall = [
		{'name': f"{name}-floor", 'drake_type': 'box', 'drake_size': [10, 10, 0.01],
		 'drake_center': translation + np.array([0, 0, -0.005])},
		{'name': f"{name}-ceiling", 'drake_type': 'box', 'drake_size': [10, 10, 0.01],
		 'drake_center': translation + np.array([0, 0, 3.005])},
		{'name': f"{name}-north", 'drake_type': 'box', 'drake_size': [0.01, 10, 3],
		 'drake_center': translation + np.array([5.005, 0, 1.5])},
		{'name': f"{name}-south", 'drake_type': 'box', 'drake_size': [0.01, 10, 3],
		 'drake_center': translation + np.array([-5.005, 0, 1.5])},
		{'name': f"{name}-west", 'drake_type': 'box', 'drake_size': [10, 0.01, 3],
		 'drake_center': translation + np.array([0, 5.005, 1.5])},
		{'name': f"{name}-east", 'drake_type': 'box', 'drake_size': [10, 0.01, 3],
		 'drake_center': translation + np.array([0, -5.005, 1.5])},
		{'name': f"{name}-div1", 'drake_type': 'box', 'drake_size': [5, 0.2, 3],
		 'drake_center': translation + np.array([-2.5, -1.5, 1.5])},
		{'name': f"{name}-div2", 'drake_type': 'box', 'drake_size': [1.6, 0.2, 3],
		 'drake_center': translation + np.array([4.2, -1.5, 1.5])},
	]
	return wall


def _create_hard_wall(translation=np.array([0, 0, 0]), name='wall'):
	wall = [
		{'name': f"{name}-floor", 'drake_type': 'box', 'drake_size': [10, 10, 0.01],
		 'drake_center': translation + np.array([0, 0, -0.005])},
		{'name': f"{name}-ceiling", 'drake_type': 'box', 'drake_size': [10, 10, 0.01],
		 'drake_center': translation + np.array([0, 0, 3.005])},
		{'name': f"{name}-north", 'drake_type': 'box', 'drake_size': [0.01, 10, 3],
		 'drake_center': translation + np.array([5.005, 0, 1.5])},
		{'name': f"{name}-south", 'drake_type': 'box', 'drake_size': [0.01, 10, 3],
		 'drake_center': translation + np.array([-5.005, 0, 1.5])},
		{'name': f"{name}-west", 'drake_type': 'box', 'drake_size': [10, 0.01, 3],
		 'drake_center': translation + np.array([0, 5.005, 1.5])},
		{'name': f"{name}-east", 'drake_type': 'box', 'drake_size': [10, 0.01, 3],
		 'drake_center': translation + np.array([0, -5.005, 1.5])},
		{'name': f"{name}-div1", 'drake_type': 'box', 'drake_size': [7, 0.2, 3],
		 'drake_center': translation + np.array([-1.5, -1.5, 1.5])},
		{'name': f"{name}-div2", 'drake_type': 'box', 'drake_size': [1.6, 0.2, 3],
		 'drake_center': translation + np.array([4.2, -1.5, 1.5])},
		{'name': f"{name}-div3", 'drake_type': 'box', 'drake_size': [1.4, 0.2, 1],
		 'drake_center': translation + np.array([2.7, -1.5, 2.5])},
		{'name': f"{name}-div4", 'drake_type': 'box', 'drake_size': [0.2, 5, 3],
		 'drake_center': translation + np.array([-1, 2.5, 1.5])},
		{'name': f"{name}-div5", 'drake_type': 'box', 'drake_size': [0.2, 1.4, 1],
		 'drake_center': translation + np.array([-1, -0.7, 2.5])},
	]
	return wall


def _create_demo_wall(translation=np.array([0, 0, 0]), name='wall'):
	wall = [
		{'name': f"{name}-floor", 'drake_type': 'box', 'drake_size': [12, 10, 0.01],
		 'drake_center': translation + np.array([0, 0, -0.005])},
		{'name': f"{name}-ceiling", 'drake_type': 'box', 'drake_size': [12, 10, 0.01],
		 'drake_center': translation + np.array([0, 0, 3.005])},
		{'name': f"{name}-north", 'drake_type': 'box', 'drake_size': [0.01, 10, 3],
		 'drake_center': translation + np.array([6.005, 0, 1.5])},
		{'name': f"{name}-south", 'drake_type': 'box', 'drake_size': [0.01, 10, 3],
		 'drake_center': translation + np.array([-6.005, 0, 1.5])},
		{'name': f"{name}-west", 'drake_type': 'box', 'drake_size': [12, 0.01, 3],
		 'drake_center': translation + np.array([0, 5.005, 1.5])},
		{'name': f"{name}-east", 'drake_type': 'box', 'drake_size': [12, 0.01, 3],
		 'drake_center': translation + np.array([0, -5.005, 1.5])},

		{'name': f"{name}-easy-1", 'drake_type': 'box', 'drake_size': [1, 4, 3],
		 'drake_center': translation + np.array([-2, -3, 1.5])},
		{'name': f"{name}-medium-1", 'drake_type': 'box', 'drake_size': [5, 3, 3],
		 'drake_center': translation + np.array([0, 3.5, 1.5])},
		{'name': f"{name}-hard-1", 'drake_type': 'box', 'drake_size': [0.4, 3, 3],
		 'drake_center': translation + np.array([2.3, 0.5, 1.5])},
		{'name': f"{name}-hard-2", 'drake_type': 'box', 'drake_size': [0.4, 3, 3],
		 'drake_center': translation + np.array([2.3, -3.5, 1.5])},


		# {'name': f"{name}-div1", 'drake_type': 'box', 'drake_size': [7, 0.4, 3],
		#  'drake_center': translation + np.array([-1.5, -1.6, 1.5])},
		# {'name': f"{name}-div2", 'drake_type': 'box', 'drake_size': [1.6, 0.4, 3],
		#  'drake_center': translation + np.array([4.2, -1.6, 1.5])},
		# {'name': f"{name}-div3", 'drake_type': 'box', 'drake_size': [1.4, 0.4, 1],
		#  'drake_center': translation + np.array([2.7, -1.6, 2.5])},
		# {'name': f"{name}-div4", 'drake_type': 'box', 'drake_size': [0.4, 5, 3],
		#  'drake_center': translation + np.array([-1, 1.1, 1.5])},
		# {'name': f"{name}-div5", 'drake_type': 'box', 'drake_size': [0.4, 1.4, 1],
		#  'drake_center': translation + np.array([-1, 4.3, 0.5])},
		#
		# {'name': f"{name}-div6", 'drake_type': 'box', 'drake_size': [2.6, 2, 2],
		#  'drake_center': translation + np.array([3.7, -4, 1])},
		# {'name': f"{name}-div7", 'drake_type': 'box', 'drake_size': [1.6, 2.4, 3],
		#  'drake_center': translation + np.array([0, 0, 1.5])},

		# {'name': f"{name}-center", 'drake_type': 'box', 'drake_size': [4, 4, 3],
		#  'drake_center': translation + np.array([0, 0, 1.5])},
		# {'name': f"{name}-south-full1", 'drake_type': 'box', 'drake_size': [0.6, 1.2, 3],
		#  'drake_center': translation + np.array([1.7, -2.6, 1.5])},
		# {'name': f"{name}-south-full2", 'drake_type': 'box', 'drake_size': [0.6, 1.2, 3],
		#  'drake_center': translation + np.array([1.7, -4.4, 1.5])},
		#
		# {'name': f"{name}-east-full1", 'drake_type': 'box', 'drake_size': [1.2, 0.6, 3],
		#  'drake_center': translation + np.array([2.6, 1.7, 1.5])},
		# {'name': f"{name}-east-full2", 'drake_type': 'box', 'drake_size': [1.2, 0.6, 3],
		#  'drake_center': translation + np.array([4.4, 1.7, 1.5])},
		#
		# {'name': f"{name}-west-full", 'drake_type': 'box', 'drake_size': [2.4, 0.6, 3],
		#  'drake_center': translation + np.array([-3.2, -1.7, 1.5])},
		# {'name': f"{name}-west-half", 'drake_type': 'box', 'drake_size': [0.6, 0.6, 1],
		#  'drake_center': translation + np.array([-4.7, -1.7, 0.5])},
	]
	return wall


def _create_cupboard(translation=np.array([0, 0, 0]), name='cupboard'):
	cupboard = [
		{'name': f"{name}-bottom", 'drake_type': 'box', 'drake_size': [1, 0.3, 0.02],
		 'drake_center': translation + np.array([0, 0, 0.01])},
		{'name': f"{name}-top", 'drake_type': 'box', 'drake_size': [1, 0.3, 0.02],
		 'drake_center': translation + np.array([0, 0, 1.99])},
		{'name': f"{name}-left", 'drake_type': 'box', 'drake_size': [0.02, 0.3, 2],
		 'drake_center': translation + np.array([-0.5, 0, 1])},
		{'name': f"{name}-right", 'drake_type': 'box', 'drake_size': [0.02, 0.3, 2],
		 'drake_center': translation + np.array([0.5, 0, 1])},
		{'name': f"{name}-back", 'drake_type': 'box', 'drake_size': [1, 0.02, 2],
		 'drake_center': translation + np.array([0, 0.14, 1])},
		{'name': f"{name}-shelf1", 'drake_type': 'box', 'drake_size': [1, 0.3, 0.02],
		 'drake_center': translation + np.array([0, 0, 0.5])},
		{'name': f"{name}-shelf2", 'drake_type': 'box', 'drake_size': [1, 0.3, 0.02],
		 'drake_center': translation + np.array([0, 0, 1])},
		{'name': f"{name}-shelf3", 'drake_type': 'box', 'drake_size': [1, 0.3, 0.02],
		 'drake_center': translation + np.array([0, 0, 1.5])},
		{'name': f"{name}-door1", 'drake_type': 'box', 'drake_size': [0.4975, 0.02, 1.98],
		 'drake_center': translation + np.array([-0.25, -0.16, 1])},
		{'name': f"{name}-door2", 'drake_type': 'box', 'drake_size': [0.4975, 0.02, 1.98],
		 'drake_center': translation + np.array([0.25, -0.16, 1])},
	]
	return cupboard
