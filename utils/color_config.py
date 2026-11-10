import open3d.visualization.rendering as rendering

WALL_COLOR = rendering.MaterialRecord()
WALL_COLOR.base_color = [0.7, 0.7, 0.7, 0.5]
WALL_COLOR.shader = "defaultLit"

# TOP_COLOR = rendering.MaterialRecord()
# TOP_COLOR.base_color = [0.7, 0.7, 0.7, 0.1]
# TOP_COLOR.shader = "defaultLitTransparency"

START_COLOR = rendering.MaterialRecord()
START_COLOR.base_color = [0.7, 0., 0., 1]
START_COLOR.shader = "defaultLit"

GOAL_COLOR = rendering.MaterialRecord()
GOAL_COLOR.base_color = [0.7, 0., 0., 1]  #[0, 0.7, 0., 1]
GOAL_COLOR.shader = "defaultLit"

INTER_COLOR = rendering.MaterialRecord()
INTER_COLOR.base_color = [0, 0.7, 0., 0.9]  # [0, 0, 0.7, 0.6]
INTER_COLOR.shader = "defaultLitTransparency"

GRAPH_POINT_COLOR = rendering.MaterialRecord()
GRAPH_POINT_COLOR.base_color = [0.7, 0, 0., 1]
GRAPH_POINT_COLOR.shader = "defaultLit"

GRAPH_POLY_COLOR = rendering.MaterialRecord()
GRAPH_POLY_COLOR.base_color = [0., 0.9, 0, 0.96]
GRAPH_POLY_COLOR.shader = "defaultLitTransparency"
