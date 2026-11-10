from .basic_env import BaseEnvironment

import os

import numpy as np
import trimesh

from pydrake.planning import RobotDiagramBuilder
from pydrake.common import RandomGenerator

from src.drake_urdf.obstacle_urdf import obstacle_primitive_urdf
from src.drake_urdf.agent_urdf import agent_box_3d_urdf, agent_box_2d_urdf


class DrakeEnvironment(BaseEnvironment):
    """
    Environment that uses Drake for C-IRIS and GCS motion planning.
    """
    def __init__(self, agent, agent_box_list, c_length, radius, seed, env_name=None, bounds=None):
        super().__init__(agent, c_length, radius, env_name, bounds)
        self._setup_drake(seed, agent_box_list)
        print("Drake environment initialized.")

    def _setup_drake(self, seed, agent_box_list):
        os.environ["MOSEKLM_LICENSE_FILE"] = os.path.dirname(__file__) + "/../../mosek/mosek.lic"

        self.builder = RobotDiagramBuilder()
        self.drake_parser = self.builder.parser()
        self.plant = self.builder.plant()
        self._add_drake_agent(agent_box_list)
        self._add_drake_obstacles()
        self.plant.Finalize()

        self.diagram = self.builder.Build()
        
        try:
            from pydrake.planning import IrisFromCliqueCoverOptions, SceneGraphCollisionChecker
            self.collision_checker = SceneGraphCollisionChecker(model=self.diagram, edge_step_size=0.125, robot_model_instances=[self.plant.GetModelInstanceByName("agent")])
            
            self.iris_options = IrisFromCliqueCoverOptions()
            self.iris_options.num_points_per_coverage_check = 5000
            self.iris_options.num_points_per_visibility_round = 1000
            if self.cartesian_dim == 2:
                self.iris_options.coverage_termination_threshold = 0.9 #75
            else:
                self.iris_options.coverage_termination_threshold = 0.7
            self.iris_options.minimum_clique_size = 10
            self.iris_options.iteration_limit = 20
            
            self.random_generator = RandomGenerator(seed)
        except:
            pass

    def _add_drake_agent(self, agent_box_list):
        if self.cartesian_dim == 2:
            self.drake_parser.AddModelsFromString(agent_box_2d_urdf(agent_box_list['size'], agent_box_list['translation'], self.bounds), "urdf")
        else:
            assert self.cartesian_dim == 3
            self.drake_parser.AddModelsFromString(agent_box_3d_urdf(agent_box_list['size'], agent_box_list['translation'], self.bounds), "urdf")

    def _add_drake_obstacles(self):
        if self.cartesian_dim == 2:
            box_size_list = [obstacle['drake_size'] for obstacle in self.obstacles_V]
            box_center_list = [obstacle['drake_center'] for obstacle in self.obstacles_V]
            box_type_list = ['box'] * len(box_size_list)
            self.drake_parser.AddModelsFromString(obstacle_primitive_urdf(box_size_list, box_center_list, box_type_list), "urdf")
        else:
            assert self.cartesian_dim == 3
            size_list = [obstacle['drake_size'] for obstacle in self.obstacles_V]
            center_list = [obstacle['drake_center'] for obstacle in self.obstacles_V]
            type_list = [obstacle['drake_type'] for obstacle in self.obstacles_V]
            self.drake_parser.AddModelsFromString(obstacle_primitive_urdf(size_list, center_list, type_list), "urdf")

