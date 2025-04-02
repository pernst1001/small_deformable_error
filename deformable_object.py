from omni.isaac.kit import SimulationApp

# Configuration for standalone simulation
config = {"renderer": "RayTracedLighting", "headless": False}

# Launch Isaac Sim
simulation_app = SimulationApp(config)

# Import necessary modules
from pxr import Usd, UsdGeom
import omni.usd
import omni.isaac.core.utils.stage as stage_utils
from omni.isaac.core import World

# Get the USD stage
stage = omni.usd.get_context().get_stage()

# Define your USD file path
usd_path = "/home/pascal/Documents/small_deformable_error/isaac_lab_created_deformable.usd"

# Load the USD file
if stage_utils.is_stage_loading():
    stage_utils.update_stage_async()
    
omni.usd.get_context().open_stage(usd_path)

# Wait for the stage to load
while stage_utils.is_stage_loading():
    simulation_app.update()

# Create a World with physics
world = World(stage_units_in_meters=1.0)
world.reset()

# Simulation loop
for i in range(20000):
    # Step the simulation
    world.step(render=True)
    
    # Exit if window is closed
    if simulation_app.is_exiting():
        break

# Cleanup
simulation_app.close()