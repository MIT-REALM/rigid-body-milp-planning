def agent_box_3d_urdf(box_size_list, box_center_list, bounds):
	box_link_list = [f"""
		<joint name="joint_box{i+1}" type="fixed">
				<parent link="virtual_ee"/>
				<child link="link_box{i+1}"/>
		</joint>
		<link name="link_box{i+1}">
			<collision>
				<origin rpy="0 0 0" xyz="{pos[0]} {pos[1]} {pos[2]}"/>
				<geometry><box size="{box[0]} {box[1]} {box[2]}"/></geometry>
			</collision>
		</link>
	""" for i, (box, pos) in enumerate(zip(box_size_list, box_center_list))]
	box_link = "".join(box_link_list)

	all_urdf = f"""
		<robot name="agent">
			<joint name="world_joint" type="fixed">
				<origin xyz="0 0 0"/>
				<parent link="world"/>
				<child link="base"/>
			</joint>
			<link name="base">
				<inertial>
					<origin rpy="0 0 0" xyz="0 0 0"/>
					<mass value="0.0"/>
					<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
				</inertial>
			</link>
			<joint name="x" type="prismatic">
				<origin xyz="0 0 0"/>
				<parent link="base"/>
				<child link="virtual_xy"/>
				<axis xyz="1 0 0"/>
				<limit lower="{bounds['x'][0]}" upper="{bounds['x'][1]}"/>
			</joint>
			<link name="virtual_xy">
				<inertial>
					<origin rpy="0 0 0" xyz="0 0 0"/>
					<mass value="0.0"/>
					<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
				</inertial>
			</link>
			<joint name="y" type="prismatic">
				<origin xyz="0 0 0"/>
				<parent link="virtual_xy"/>
				<child link="virtual_yz"/>
				<axis xyz="0 1 0"/>
				<limit lower="{bounds['y'][0]}" upper="{bounds['y'][1]}"/>
			</joint>
			<link name="virtual_yz">
				<inertial>
					<origin rpy="0 0 0" xyz="0 0 0"/>
					<mass value="0.0"/>
					<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
				</inertial>
			</link>
			<joint name="z" type="prismatic">
				<origin xyz="0 0 0"/>
				<parent link="virtual_yz"/>
				<child link="virtual_z_euler1"/>
				<axis xyz="0 0 1"/>
				<limit lower="{bounds['z'][0]}" upper="{bounds['z'][1]}"/>
			</joint>
			<link name="virtual_z_euler1">
				<inertial>
					<origin rpy="0 0 0" xyz="0 0 0"/>
					<mass value="0.0"/>
					<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
				</inertial>
			</link>
			<joint name="euler1" type="revolute">
				<origin xyz="0 0 0"/>
				<parent link="virtual_z_euler1"/>
				<child link="virtual_euler12"/>
				<axis xyz="1 0 0"/>
				<limit lower="-3.1416" upper="3.1416"/>
			</joint>
			<link name="virtual_euler12">
				<inertial>
					<origin rpy="0 0 0" xyz="0 0 0"/>
					<mass value="0.0"/>
					<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
				</inertial>
			</link>
			<joint name="euler2" type="revolute">
				<origin xyz="0 0 0"/>
				<parent link="virtual_euler12"/>
				<child link="virtual_euler23"/>
				<axis xyz="0 1 0"/>
				<limit lower="-3.1416" upper="3.1416"/>
			</joint>
			<link name="virtual_euler23">
				<inertial>
					<origin rpy="0 0 0" xyz="0 0 0"/>
					<mass value="0.0"/>
					<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
				</inertial>
			</link>
			<joint name="euler3" type="revolute">
				<origin xyz="0 0 0"/>
				<parent link="virtual_euler23"/>
				<child link="virtual_ee"/>
				<axis xyz="0 0 1"/>
				<limit lower="-3.1416" upper="3.1416"/>
			</joint>
			<link name="virtual_ee">
				<inertial>
					<origin rpy="0 0 0" xyz="0 0 0"/>
					<mass value="0.0"/>
					<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
				</inertial>
			</link>
			{box_link}
		</robot>
	"""
	return all_urdf


def agent_box_2d_urdf(box_size_list, box_center_list, bounds):
	box_link_list = [f"""
		<joint name="joint_box{i+1}" type="fixed">
				<parent link="virtual_ee"/>
				<child link="link_box{i+1}"/>
		</joint>
		<link name="link_box{i+1}">
			<collision>
				<origin rpy="0 0 0" xyz="{pos[0]} {pos[1]} {pos[2]}"/>
				<geometry><box size="{box[0]} {box[1]} {box[2]}"/></geometry>
			</collision>
		</link>
	""" for i, (box, pos) in enumerate(zip(box_size_list, box_center_list))]
	box_link = "".join(box_link_list)

	all_urdf = f"""
		<robot name="agent">
			<joint name="world_joint" type="fixed">
				<origin xyz="0 0 0"/>
				<parent link="world"/>
				<child link="base"/>
			</joint>
			<link name="base">
				<inertial>
					<origin rpy="0 0 0" xyz="0 0 0"/>
					<mass value="0.0"/>
					<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
				</inertial>
			</link>
			<joint name="x" type="prismatic">
				<origin xyz="0 0 0"/>
				<parent link="base"/>
				<child link="virtual_xy"/>
				<axis xyz="1 0 0"/>
				<limit lower="{bounds['x'][0]}" upper="{bounds['x'][1]}"/>
			</joint>
			<link name="virtual_xy">
				<inertial>
					<origin rpy="0 0 0" xyz="0 0 0"/>
					<mass value="0.0"/>
					<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
				</inertial>
			</link>
			<joint name="y" type="prismatic">
				<origin xyz="0 0 0"/>
				<parent link="virtual_xy"/>
				<child link="virtual_y_theta"/>
				<axis xyz="0 1 0"/>
				<limit lower="{bounds['y'][0]}" upper="{bounds['y'][1]}"/>
			</joint>
			<link name="virtual_y_theta">
				<inertial>
					<origin rpy="0 0 0" xyz="0 0 0"/>
					<mass value="0.0"/>
					<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
				</inertial>
			</link>
			<joint name="theta" type="revolute">
				<origin xyz="0 0 0"/>
				<parent link="virtual_y_theta"/>
				<child link="virtual_ee"/>
				<axis xyz="0 0 1"/>
				<limit lower="{bounds['theta'][0]}" upper="{bounds['theta'][1]}"/>
			</joint>
			<link name="virtual_ee">
				<inertial>
					<origin rpy="0 0 0" xyz="0 0 0"/>
					<mass value="0.0"/>
					<inertia ixx="0" ixy="0" ixz="0" iyy="0" iyz="0" izz="0"/>
				</inertial>
			</link>
			{box_link}
		</robot>
	"""
	return all_urdf
