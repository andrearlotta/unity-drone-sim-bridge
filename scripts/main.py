
from unity_drone_sim_bridge.examples.yolo_system_1tree.main import run_simulation

if __name__ == '__main__':
    # Run the simulation with the default parameters
    run_simulation(g_function='mlp', simulation_steps=100)