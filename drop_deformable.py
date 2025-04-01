import argparse
from isaaclab.app import AppLauncher
# add argparse arguments
parser = argparse.ArgumentParser(description="Testcase for similarity between Isaac Sim & Lab for deformable object (BUG?).")
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
"""Rest everything follows."""
import isaaclab.sim as sim_utils
from isaaclab.assets import DeformableObject, DeformableObjectCfg, RigidObject, RigidObjectCfg
from isaaclab.sim import SimulationContext

def design_scene():
    """Designs the scene."""
    # Ground-plane
    cfg = sim_utils.GroundPlaneCfg()
    cfg.func("/World/defaultGroundPlane", cfg)
    # Lights
    cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    cfg.func("/World/Light", cfg)
    # Define and spawn the deformable (soft) cylinder
    soft_cylinder = DeformableObjectCfg(
        prim_path="/World/Deformable/Connection1",
        spawn=sim_utils.MeshCylinderCfg(
            radius=0.0005,
            height=0.025,
            deformable_props=sim_utils.DeformableBodyPropertiesCfg(rest_offset=0.0003, contact_offset=0.0002, simulation_hexahedral_resolution=8, solver_position_iteration_count=50),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.5, 0.1, 0.0)),
            physics_material=sim_utils.DeformableBodyMaterialCfg(poissons_ratio=0.45, youngs_modulus=40e6),
        ),
        init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.003), rot=(0.70711, 0.0, 0.70711, 0.0)),
    )
    deformable_object = DeformableObject(soft_cylinder)  # create the deformable cylinder in the scene

    scene_entities = {"deformable_object": deformable_object}
    return scene_entities

def run_simulator(sim: sim_utils.SimulationContext, entities: dict[str, DeformableObject]):
    """Runs the simulation loop."""
    deformable_object = entities["deformable_object"]
    sim_dt = sim.get_physics_dt()
    sim_time = 0.0
    count = 0
    # Simulate physics
    while simulation_app.is_running():
        deformable_object.update(sim_dt)
        sim.step()
        sim_time += sim_dt
        count += 1
        sim.render()

def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device,
                                      dt=1/1000, # otherwise no stable simulation
                                      render_interval=2)
    sim = SimulationContext(sim_cfg)
    # Set main camera
    sim.set_camera_view(eye=[0.3, 0.0, 0.2], target=[0.0, 0.0, 0.0])
    # Design scene
    scene_entities = design_scene()
    # Play the simulator
    sim.reset()
    run_simulator(sim, scene_entities)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
