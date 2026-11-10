from environment.collision_checking_env import CollisionCheckingEnvironment

from ompl import base as ob
from ompl import geometric as og

import numpy as np
import fcl
import trimesh
import trimesh.transformations as tf

class OMPLEnvironment(CollisionCheckingEnvironment):
    """
    Environment that uses OMPL for motion planning.
    """
    def __init__(self, agent, c_length, radius, env_name=None, bounds=None):
        super().__init__(agent, c_length, radius, env_name, bounds)
        self._setup_ompl()

    def _setup_ompl(self):
        if self.cartesian_dim == 2:
            self.ompl_space = ob.SE2StateSpace()

            bounds = ob.RealVectorBounds(2)
            bounds.setLow(0, self.bounds['x'][0])
            bounds.setLow(1, self.bounds['y'][0])
            bounds.setHigh(0, self.bounds['x'][1])
            bounds.setHigh(1, self.bounds['y'][1])
            self.ompl_space.setBounds(bounds)

            self.ompl_start = ob.State(self.ompl_space)
            self.ompl_start().setX(self.start_point[0])
            self.ompl_start().setY(self.start_point[1])
            self.ompl_start().setYaw(self.start_point[2])

            self.ompl_random_start = ob.State(self.ompl_space)
            self.ompl_random_start().setX(np.random.uniform(self.bounds['x'][0], self.bounds['x'][1]))
            self.ompl_random_start().setY(np.random.uniform(self.bounds['y'][0], self.bounds['y'][1]))
            self.ompl_random_start().setYaw(np.random.uniform(-np.pi, np.pi))

            self.ompl_goal = ob.State(self.ompl_space)
            self.ompl_goal().setX(self.end_point[0])
            self.ompl_goal().setY(self.end_point[1])
            self.ompl_goal().setYaw(self.end_point[2])

            self.ompl_random_goal = ob.State(self.ompl_space)
            self.ompl_random_goal().setX(np.random.uniform(self.bounds['x'][0], self.bounds['x'][1]))
            self.ompl_random_goal().setY(np.random.uniform(self.bounds['y'][0], self.bounds['y'][1]))
            self.ompl_random_goal().setYaw(np.random.uniform(-np.pi, np.pi))
        elif self.cartesian_dim == 3:
            self.ompl_space = ob.SE3StateSpace()

            bounds = ob.RealVectorBounds(3)
            bounds.setLow(0, self.bounds['x'][0])
            bounds.setLow(1, self.bounds['y'][0])
            bounds.setLow(2, self.bounds['z'][0])
            bounds.setHigh(0, self.bounds['x'][1])
            bounds.setHigh(1, self.bounds['y'][1])
            bounds.setHigh(2, self.bounds['z'][1])
            self.ompl_space.setBounds(bounds)

            self.ompl_start = ob.State(self.ompl_space)
            self.ompl_start().setXYZ(self.start_point[0][-1], self.start_point[1][-1], self.start_point[2][-1])
            q_start = tf.quaternion_from_matrix(self.start_point)
            self.ompl_start().rotation().x = q_start[0]
            self.ompl_start().rotation().y = q_start[1]
            self.ompl_start().rotation().z = q_start[2]
            self.ompl_start().rotation().w = q_start[3]

            self.ompl_random_start = ob.State(self.ompl_space)
            self.ompl_random_start().setXYZ(np.random.uniform(self.bounds['x'][0], self.bounds['x'][1]),
                                            np.random.uniform(self.bounds['y'][0], self.bounds['y'][1]),
                                            np.random.uniform(self.bounds['z'][0], self.bounds['z'][1]))
            q_random_start = tf.random_quaternion()
            self.ompl_random_start().rotation().x = q_random_start[0]
            self.ompl_random_start().rotation().y = q_random_start[1]
            self.ompl_random_start().rotation().z = q_random_start[2]
            self.ompl_random_start().rotation().w = q_random_start[3]


            self.ompl_goal = ob.State(self.ompl_space)
            self.ompl_goal().setXYZ(self.end_point[0][-1], self.end_point[1][-1], self.end_point[2][-1])
            q_goal = tf.quaternion_from_matrix(self.end_point)
            self.ompl_goal().rotation().x = q_goal[0]
            self.ompl_goal().rotation().y = q_goal[1]
            self.ompl_goal().rotation().z = q_goal[2]
            self.ompl_goal().rotation().w = q_goal[3]

            self.ompl_random_goal = ob.State(self.ompl_space)
            self.ompl_random_goal().setXYZ(np.random.uniform(self.bounds['x'][0], self.bounds['x'][1]),
                                           np.random.uniform(self.bounds['y'][0], self.bounds['y'][1]),
                                           np.random.uniform(self.bounds['z'][0], self.bounds['z'][1]))
            q_random_goal = tf.random_quaternion()
            self.ompl_random_goal().rotation().x = q_random_goal[0]
            self.ompl_random_goal().rotation().y = q_random_goal[1]
            self.ompl_random_goal().rotation().z = q_random_goal[2]
            self.ompl_random_goal().rotation().w = q_random_goal[3]
        else:
            raise ValueError("Only 2D and 3D are supported.")

        self.ompl_ss = og.SimpleSetup(self.ompl_space)
        self.ompl_ss.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))


    def is_state_valid(self, state):
        if self.cartesian_dim == 2:
            x = state.getX()
            y = state.getY()
            theta = state.getYaw()
            return self._is_configuration_free((x, y, theta))
        elif self.cartesian_dim == 3:
            x = state.getX()
            y = state.getY()
            z = state.getZ()
            q = (state.rotation().x, state.rotation().y, state.rotation().z, state.rotation().w)
            return self._is_configuration_free((x, y, z, q))
        else:
            raise ValueError("Only 2D and 3D are supported.")

