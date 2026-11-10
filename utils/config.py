import numpy as np

# PARAMS
DELTA_THETA_UL = np.pi / 12  # 3
BIG_M = 1000
MAX_NUM_WAYPOINT = 1

# DISCRETIZATION
NUM_THETA_DIV = 60  # discretized rotation, number of pieces
assert NUM_THETA_DIV % 2 == 0

DELTA_BOUNDARY_STEP_XY = 0.1
NUM_AGENT_EDGE_DIV = 10 + 1  # discretized agent edge, number of pieces
NUM_BOUNDARY_LINE_DIV = 160 + 1  #20 + 1  # discretized boundary line, number of pieces

SAMPLE_STEPSIZE_UL = 0.3

def create_rotation_mat_array(mode) -> np.ndarray:  # (n, 3, 3)
	if mode == 'Cuboid':  # total 24 rotation matrices
		axis_vec = np.concatenate([np.identity(3), -np.identity(3)], axis=0)  # (6, 3)
		z_col = np.cross(axis_vec[:, np.newaxis], axis_vec[np.newaxis])
		x_idx, y_idx = np.nonzero(np.any(z_col, axis=(-1)))
		x_col = axis_vec[x_idx]
		y_col = axis_vec[y_idx]
		z_col = z_col[x_idx, y_idx]
		return np.stack([x_col, y_col, z_col], axis=-1)
	else:
		raise NotImplementedError('Rotation mode not implemented')


ROTATION_3D_MODE = 'Cuboid'
ROTATION_MATS = create_rotation_mat_array(ROTATION_3D_MODE)
