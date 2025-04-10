import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, ArticulationCfg
from isaaclab.markers import VisualizationMarkersCfg

# CUBE_ARTICULATION_CFG = ArticulationCfg(
#         prim_path="/World/envs/env_0/Cube",

CUBE_CFG = RigidObjectCfg(
        prim_path="/World/envs/env_0/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.2, 0.2, 0.2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.5,
                dynamic_friction=0.5,
                restitution=0.1
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.0, 0.0)),
        ),
    )

GOAL_CFG =VisualizationMarkersCfg(
        prim_path="/World/Goal/Cones",
        markers={
            "cone_0": sim_utils.SphereCfg(
                radius=0.1,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.0)),
            ),
        }
    )