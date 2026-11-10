from .basic_env import BaseEnvironment

import numpy as np

import fcl
import trimesh

class CollisionCheckingEnvironment(BaseEnvironment):
	"""
	Environment that uses python-fcl for collision checking.
	Specialized for method requiring explicit collision checking, e.g., sampling-based methods.
	"""
	def __init__(self, agent, c_length, radius, env_name=None, bounds=None):
		super().__init__(agent, c_length, radius, env_name, bounds)
		self._setup_fcl()

	############################## mask ###############################

	def _setup_fcl(self):
		fcl_meshes = []
		for i, obs in enumerate(self.obstacles_V):
			if self.cartesian_dim == 2:
				mesh_dict = trimesh.creation.extrude_polygon(obs['geom'], height=1)
			elif self.cartesian_dim == 3:
				mesh_dict = obs['geom'].copy()
			else:
				raise ValueError("Only 2D and 3D are supported.")
			mesh_dict = trimesh.exchange.export.export_dict(mesh_dict)
			mesh = fcl.BVHModel()
			mesh.beginModel(len(mesh_dict['vertices']), len(mesh_dict['faces']))
			mesh.addSubModel(np.array(mesh_dict['vertices']), np.array(mesh_dict['faces']))
			mesh.endModel()
			fcl_meshes.append(fcl.CollisionObject(mesh))

		self.obstacle_manager = fcl.DynamicAABBTreeCollisionManager()
		self.obstacle_manager.registerObjects(fcl_meshes)
		self.obstacle_manager.setup()

		if self.cartesian_dim == 2:
			agent_mesh_dict = trimesh.creation.extrude_polygon(self.agent, height=1)
		elif self.cartesian_dim == 3:
			agent_mesh_dict = self.agent.copy()
		else:
			raise ValueError("Only 2D and 3D are supported.")
		agent_mesh_dict = trimesh.exchange.export.export_dict(agent_mesh_dict)
		agent_mesh = fcl.BVHModel()
		agent_mesh.beginModel(len(agent_mesh_dict['vertices']), len(agent_mesh_dict['faces']))
		agent_mesh.addSubModel(np.array(agent_mesh_dict['vertices']), np.array(agent_mesh_dict['faces']))
		agent_mesh.endModel()
		self.agent_collision_object = fcl.CollisionObject(agent_mesh)

		self.point_collision_object = fcl.CollisionObject(fcl.Cylinder(self.point_radius, 1))

	def _is_configuration_free(self, pos):
		req = fcl.CollisionRequest(num_max_contacts=100, enable_contact=True)
		rdata = fcl.CollisionData(request=req)

		if self.cartesian_dim == 2:
			assert len(pos) == 3
			x, y, theta = pos
			z = 0
			rot_mat = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
		elif self.cartesian_dim == 3:
			if len(pos) == 7:
				x, y, z, q_x, q_y, q_z, q_w = pos
			else:
				assert len(pos) == 4
				x, y, z, (q_x, q_y, q_z, q_w) = pos
			rot_mat = trimesh.transformations.quaternion_matrix([q_w, q_x, q_y, q_z])[:3, :3]
		else:
			raise ValueError("Only 2D and 3D are supported.")

		self.agent_collision_object.setTranslation(np.array([x, y, z]))
		self.agent_collision_object.setRotation(rot_mat)
		self.obstacle_manager.collide(self.agent_collision_object, rdata, fcl.defaultCollisionCallback)
		return not rdata.result.is_collision
