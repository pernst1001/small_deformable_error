import argparse

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Tutorial on spawning and interacting with a rigid object.")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import omni.physx
import torch

import isaacsim.core.utils.prims as prim_utils
import omni
import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import RigidObject, RigidObjectCfg, DeformableObject, DeformableObjectCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR, ISAACLAB_NUCLEUS_DIR
from isaaclab.sim import SimulationContext, PhysxCfg
from pxr import PhysxSchema  # import PhysX USD schema classes



def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)

    cylinder = RigidObjectCfg(
        prim_path="/World/Rigid",
        spawn=sim_utils.CylinderCfg(
            radius=0.00075,
            height=0.015,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.48e-3),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.0002,
                rest_offset=0.0001,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 1.0), metallic=0.2),
            physics_material=sim_utils.materials.RigidBodyMaterialCfg(static_friction=0.38, dynamic_friction=0.3, restitution=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.002), rot=(0.70711, 0.0, 0.70711, 0.0), lin_vel=(0.0, 0.0, 0.0), ang_vel=(0.0, 0.0, 0.0)),
    )
    rigid = RigidObject(cfg=cylinder)
    
    # Define and spawn the deformable (soft) cuboid
    soft_cylinder = DeformableObjectCfg(
        prim_path="/World/Deformable",
        spawn=sim_utils.MeshCylinderCfg(
            radius=0.0005,
            height=0.025,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0003, contact_offset=0.0002, simulation_hexahedral_resolution=8, solver_position_iteration_count=50),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.45, youngs_modulus=40e6),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.02, 0.0, 0.002), rot=(0.70711, 0.0, 0.70711, 0.0)),
    )
    deformable = DeformableObject(soft_cylinder)  # create the deformable cuboid in the scene

    scene_entities = {"deformable": deformable, "rigid": rigid}
    return scene_entities

def create_attachement(prim1: str, prim2: str):
    '''Create a magnetic attachement'''
    # Create a PhysX attachment to rigidly bind the soft cuboid to the cube
    stage = omni.usd.get_context().get_stage()
    soft_prim = stage.GetPrimAtPath(prim1)
    rigid_prim = stage.GetPrimAtPath(prim2)
    attachment_path = soft_prim.GetPath().AppendElementString("attachment")  # child prim path for attachment
    attachment = PhysxSchema.PhysxPhysicsAttachment.Define(stage, attachment_path)
    attachment.GetActor0Rel().SetTargets([soft_prim.GetPath()])   # actor0: soft body prim
    attachment.GetActor1Rel().SetTargets([rigid_prim.GetPath()])  # actor1: rigid body prim
    # Apply auto attachment API first
    auto_attachment_api = PhysxSchema.PhysxAutoAttachmentAPI.Apply(attachment.GetPrim())
    # Then create the attribute on the API instance
    # auto_attachment_api.CreateDeformableVertexOverlapOffsetAttr(0.002, False)
    return auto_attachment_api


def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, RigidObject]):
    """Runs the simulation loop."""
    # Extract scene entities
    # note: we only do this here for readability. In general, it is better to access the entities directly from
    #   the dictionary. This dictionary is replaced by the InteractiveScene class in the next tutorial.
    attachement_api = create_attachement("/World/Rigid/geometry/mesh", "/World/Deformable/geometry/mesh")
    rigid = entities["rigid"]
    deformable = entities["deformable"]
    sim_dt = sim.get_physics_dt()
    rendering_dt = sim.get_rendering_dt()
    print(f"[INFO]: Simulation time-step: {sim_dt}, Rendering time-step: {rendering_dt}")
    sim_time = 0.0
    count = 0
    # Simulate physics
    print("[INFO]: Starting simulation...")
    while True:
        # perform step     
        if count == 1:
            attachement_api.CreateDeformableVertexOverlapOffsetAttr(0.005, False)
        #     pass
        sim.step()
        # update sim-time
        sim_time += sim_dt
        count += 1
        rigid.update(sim_dt)
        deformable.update(sim_dt)



def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, 
                                      dt=1 / 1000, 
                                      render_interval=2,
                                      physx=PhysxCfg(bounce_threshold_velocity=0.0,
                                                     enable_stabilization=False)
                                      )
    sim = SimulationContext(sim_cfg)
    # # Set main camera
    # sim.set_camera_view(eye=[-8, 0, 8], target=[0.0, 0.0, 2])
    sim.set_camera_view(eye=[0.0, -0.05, 0.2], target=[0.0, 0.0, 0.01])
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
    # close sim app
    simulation_app.close()
