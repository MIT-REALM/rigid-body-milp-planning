import time
import pickle
import multiprocessing as mp

import numpy as np

from pydrake.solvers import MosekSolver
from pydrake.trajectories import PiecewisePolynomial
from scipy.spatial.transform import Rotation as R

from environment import DrakeEnvironment
from baseline.gcs.linear_gcs import LinearGCS

def planWithGCS(env: DrakeEnvironment, agent_name) -> (np.ndarray, float, float):
    offline_time = 0
    online_time = 0
    
    ciris_start_time = time.time()
    convex_relaxation = True
    try:
        from pydrake.planning import IrisInConfigurationSpaceFromCliqueCover
        regions = IrisInConfigurationSpaceFromCliqueCover(checker=env.collision_checker, options=env.iris_options, generator=env.random_generator, sets=[])
        with open(f"result/{agent_name}-gcs/{env.env_name}.pkl", "wb") as f:
            pickle.dump(regions, f)
        print("gcs saved")
        print("-" * 10, f"C-iris finished, {time.time() - ciris_start_time}", "-" * 10)
        exit()
    except Exception as e:
        print(e)
        with open(f"result/{agent_name}-gcs/{env.env_name}.pkl", "rb") as f:
            regions = pickle.load(f)
    offline_time += time.time() - ciris_start_time
    
    init_GCS_start_time = time.time()
    l_gcs = LinearGCS(regions)
    l_gcs.setSolver(MosekSolver())
    offline_time += time.time() - init_GCS_start_time
    print("-" * 10, f"LinearGCS initialized, {offline_time}", "-" * 10)

    inference_start_time = time.time()
    if env.cartesian_dim == 2:
        x_start, x_goal = env.start_point, env.end_point
    else:
        x_start = np.concatenate([env.start_point[:3, -1], R.from_matrix(env.start_point[:3, :3]).as_euler('xyz')])
        x_goal = np.concatenate([env.end_point[:3, -1], R.from_matrix(env.end_point[:3, :3]).as_euler('xyz')])
    try:
        l_gcs.addSourceTarget(x_start, x_goal)
        waypoints, _ = l_gcs.SolvePath(convex_relaxation, verbose=True)  # waypoints, results_dict; time in results_dict only contain solver time
    except Exception as e:
        print(e)
        waypoints = None
    online_time += time.time() - inference_start_time

    # interpolation, (not tracking time)
    if waypoints is None:
        return None, offline_time, online_time

    print("-" * 10, f"{waypoints.shape[1]} waypoints", "-" * 10)
    t_knots = np.linspace(0, 1, waypoints.shape[1])
    lin_traj = PiecewisePolynomial.FirstOrderHold(t_knots, waypoints)
    samples = np.linspace(lin_traj.start_time(), lin_traj.end_time())
    if env.cartesian_dim == 2:
        path = np.squeeze([lin_traj.value(s) for s in samples])
    else:
        path = []
        for s in samples:
            x = lin_traj.value(s)
            x_mat = np.eye(4)
            x_mat[:3, :3] = R.from_euler('xyz', x[3:].squeeze()).as_matrix()
            x_mat[:3, 3] = x[:3].squeeze()
            path.append(x_mat)

    return path, offline_time, online_time
