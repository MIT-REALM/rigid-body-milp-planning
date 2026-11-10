def obstacle_primitive_urdf(size_list, center_list, type_list):
	link_list = []
	for i, (link_size, link_pos, link_type) in enumerate(zip(size_list, center_list, type_list)):
		if link_type == 'box':
			link_list.append(f"""
			<joint name="world_joint_box{i + 1}" type="fixed">
				<parent link="world"/>
				<child link="link_box{i + 1}"/>
			</joint>
			<link name="link_box{i + 1}">
				<collision>
					<origin rpy="0 0 0" xyz="{link_pos[0]} {link_pos[1]} {link_pos[2]}"/>
					<geometry><box size="{link_size[0]} {link_size[1]} {link_size[2]}"/></geometry>
				</collision>
			</link>
		""")
		else:
			raise NotImplementedError(f"Unknown type: {link_type}")


	all_link = "".join(link_list)

	all_urdf = f"""
		<robot name="obstacles">
			{all_link}
		</robot>
	"""
	return all_urdf
