
from unity_drone_sim_bridge.examples.reduced_order_system.main import run_simulation

if __name__ == '__main__':
    # Run the simulation with the default parameters
    run_simulation(g_function='gp', simulation_steps=100)