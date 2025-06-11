import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Failure to visualize the orientation of rigid body views.")
# Add this near the top of RigidCatheter_Sim2Real.py
parser.add_argument("--joint_friction", type=float, default=1437.3613279764427, help="Joint friction coefficient")
parser.add_argument("--joint_armature", type=float, default=1.1076207565248166e-9, help="Joint armature value")
parser.add_argument("--static_friction", type=float, default=0.3416394187752424, help="Static friction for ground plane")
parser.add_argument("--dynamic_friction", type=float, default=0.38257183723258514, help="Dynamic friction for ground plane")
parser.add_argument("--restitution", type=float, default=0.0, help="Restitution for ground plane")
parser.add_argument("--linear_damping", type=float, default=4.300245716026765, help="Linear dampening for the rigid body")
parser.add_argument("--angular_damping", type=float, default=4.791183107948886, help="Angular dampening for the rigid body")
parser.add_argument("--output_file", type=str, default='None', help="Path to save optimization results")
parser.add_argument("--joint_stiffness", type=float, default=5.0e-3, help="Joint stiffness for the rigid body")
parser.add_argument("--joint_damping", type=float, default=5.0e-3, help="Joint damping for the rigid body")
parser.add_argument("--sim_dt", type=float, default=0.007602586620180273, help="Simulation time step")
parser.add_argument("--position_iteration_count", type=int, default=15)
parser.add_argument("--velocity_iteration_count", type=int, default=50)
parser.add_argument("--compliant_contact_damping", type=float, default=0.0, help="Compliant contact damping for the rigid body")
parser.add_argument("--compliant_contact_stiffness", type=float, default=0.0, help="Compliant contact stiffness for the rigid body")


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# args_cli.headless = True

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaacsim.core.utils.prims as prim_utils
import omni
import isaaclab.sim as sim_utils
from isaaclab.sim import SimulationContext
from isaaclab.assets import Articulation, ArticulationCfg
import omni.physics.tensors as tensors # Used for RigidBodyView
from isaaclab_assets.robots.rigidcatheter import RIGID_CATHETER_CFG, RIGID_CATHETER_SIM2REAL_CFG  # isort:skip
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.actuators import ImplicitActuatorCfg, IdealPDActuatorCfg
import torch.nn.functional as F         # new
from isaaclab.magnetic import MagneticEntity
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
import isaaclab.utils.math as math_utils

import os
import tqdm as tqdm
import math

def set_articulation_material_friction(articulation, static_friction, dynamic_friction, restitution):
    """Set the physics material properties (friction) for all shapes in an articulation.
    
    Args:
        articulation: The articulation to apply friction values to
        static_friction: Static friction coefficient value
        dynamic_friction: Dynamic friction coefficient value
    """
    # Get physics view for the articulation
    root_physx_view = articulation.root_physx_view
    
    # Get the current material properties
    materials = root_physx_view.get_material_properties()
    
    # Set static and dynamic friction for all shapes
    # Materials tensor has shape: (num_envs, num_shapes, 3)
    # Where the 3 values are: static friction, dynamic friction, restitution
    materials[:, :, 0] = static_friction  # Set static friction
    materials[:, :, 1] = dynamic_friction  # Set dynamic friction
    materials[:, :, 2] = restitution  # Set restitution to 0.0 (or any other value you want)
    
    # Apply the updated materials to the simulation
    env_ids = torch.arange(0, device=articulation.device)
    root_physx_view.set_material_properties(materials, env_ids)    

def design_scene():
    # Ground-plane
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)
    # Lights
    # Load the RigidCatheter
    usd_file_path = os.path.join(os.path.dirname(__file__), "RigidCatheterSim2Real.usd")
    print(f"[INFO]: Loading RigidCatheter USD file from {usd_file_path}")
    prim_utils.create_prim("/World/Origin1", "Xform", translation=(0.0, -0.01859, 0.0), orientation=(0.70711, 0.0, 0.0, 0.70711))
    rigid_catheter_cfg = ArticulationCfg(
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_file_path,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                rigid_body_enabled=True,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=100.0,
                sleep_threshold=0.0,
                stabilization_threshold=1e-6,
                solver_position_iteration_count=args_cli.position_iteration_count,
                solver_velocity_iteration_count=args_cli.velocity_iteration_count,
                # enable_gyroscopic_forces=True,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                articulation_enabled=True,
                enabled_self_collisions=False,
                solver_position_iteration_count=args_cli.position_iteration_count,
                solver_velocity_iteration_count=args_cli.velocity_iteration_count,
                sleep_threshold=0.0,
                stabilization_threshold=1e-6,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(),
        actuators={
            "spherical_joint": ImplicitActuatorCfg(
                joint_names_expr=[r"SphericalJoint[1-9]:\d", r"SphericalJoint1[0-1]:\d"],
                stiffness=args_cli.joint_stiffness,
                damping=args_cli.joint_damping,
            )
        }
    )
    # rigid_catheter_cfg = RIGID_CATHETER_CFG
    rigid_catheter_cfg.prim_path = "/World/Origin1/rigid_catheter"
    rigid_catheter = Articulation(cfg = rigid_catheter_cfg)


    cfg = sim_utils.GroundPlaneCfg(
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=args_cli.static_friction,
            dynamic_friction=args_cli.dynamic_friction,
            restitution=args_cli.restitution,
            compliant_contact_damping=args_cli.compliant_contact_damping,
            compliant_contact_stiffness=args_cli.compliant_contact_stiffness,
        )
    )
    cfg.func("/World/defaultGroundPlane", cfg)

    ARROW_CFG = VisualizationMarkersCfg(
        prim_path="/World/Goal/Arrows",
        markers={
            "torque_arrow": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/UIElements/arrow_x.usd",
                scale=(1.0, 0.2, 0.2),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )
    arrow_markers = VisualizationMarkers(ARROW_CFG)
    scene_entities = {"rigid_catheter": rigid_catheter, "arrows": arrow_markers}
    return scene_entities

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, Articulation]):
    rigid_catheter = entities["rigid_catheter"]
    torque_arrows = entities["arrows"]
    rigid_catheter.write_joint_friction_coefficient_to_sim(args_cli.joint_friction)
    rigid_catheter.write_joint_armature_to_sim(args_cli.joint_armature)
    set_articulation_material_friction(rigid_catheter, args_cli.static_friction, args_cli.dynamic_friction, args_cli.restitution)

    sim_time = 0.0
    # sim.set_simulation_dt(physics_dt=1.0/(40.0*3), rendering_dt=2)
    sim_dt = sim.get_physics_dt()
    count = 0
    sim_view = tensors.create_simulation_view("torch")
    magnet_view = sim_view.create_rigid_body_view("/World/Origin1/rigid_catheter/Magnet_*")
    while simulation_app.is_running():
        torques = torch.ones_like(magnet_view.get_transforms()[:,:3])*1e-6
        torques[:, :2] = 0.0
        forces = torch.zeros_like(torques)
        all_indices = torch.arange(magnet_view.count, device="cuda")
        magnet_view.apply_forces_and_torques_at_position(force_data=forces, torque_data=torques, position_data=None, indices=all_indices, is_global=True)
        positions = magnet_view.get_transforms()[:, :3].clone().detach()
        positions[:, 2] = 0.004  # Set z-coordinate to 0.0
        quad = magnet_view.get_transforms()[:, 3:7].clone().detach()
        scales = torch.tensor([0.02, 0.003, 0.003]).unsqueeze(0).repeat(positions.shape[0], 1)
        torque_arrows.visualize(translations=positions,orientations=quad,scales=scales)
        sim.step()
        sim_time += sim_dt
        count += 1
        rigid_catheter.update(sim_dt)
        sim_view.update_articulations_kinematic()
   
def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device,
                                      dt=args_cli.sim_dt,
                                      render_interval=1)
    sim = SimulationContext(sim_cfg)
    # # Set main camera
    # sim.set_camera_view(eye=[-8, 0, 8], target=[0.0, 0.0, 2])
    sim.set_camera_view(eye=[0.0, 0.0, 0.1], target=[0.0, 0.0, 0.0])
    # Design scene
    scene_entities = design_scene()
    # Play the simulator
    sim.reset()
    # Now we are ready!
    print("[INFO]: Setup complete...")
    # Run the simulator
    run_simulator(sim, scene_entities)
    


if __name__ == "__main__":
    # run the main function
    main()
    simulation_app.close()
