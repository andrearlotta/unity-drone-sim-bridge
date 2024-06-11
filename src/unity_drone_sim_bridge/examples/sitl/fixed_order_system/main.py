from unity_drone_sim_bridge.g_func_lib.g_func_tools import load_g
from unity_drone_sim_bridge.examples.sitl.fixed_order_system.template_model import template_model
from unity_drone_sim_bridge.examples.sitl.fixed_order_system.template_mpc import template_mpc
from unity_drone_sim_bridge.examples.sitl.fixed_order_system.template_simulator import Simulator
import numpy as np
from do_mpc.data import save_results

# Constants
GRID_SIZE = (100, 100)  # Grid size (number of trees along x and y)
SPACING = 7.5  # Spacing between trees in meters
MIN_DISTANCE = 1.5  # Minimum distance from drone to any tree in meters

def run_simulation(g_function = 'gp', simulation_steps= 100):
    dim_lambda = 20
    model = template_model(g=load_g(g_function),dim_lambda=dim_lambda)
    simulator = Simulator(model,
                          dim_lambda=dim_lambda,
                          grid_size=GRID_SIZE, 
                          spacing=SPACING, 
                          min_distance=MIN_DISTANCE)
    mpc = template_mpc(model=model, 
                       get_obs=simulator.mpc_get_obs,
                       get_lambdas=simulator.mpc_get_lambdas, 
                       get_residual_H=simulator.mpc_get_residual_H)
    n_true = 0
    mpc = template_mpc( model=model, get_obs=simulator.mpc_get_obs,
                        get_lambdas=simulator.mpc_get_lambdas, 
                        get_residual_H=simulator.mpc_get_residual_H,
                        get_cond_y= lambda: np.array(n_true*[1] + (dim_lambda-n_true)*[0]))
    """
    Time Metrics
    """
    csv_filename = f'setup_time_{g_function}_cond_fixed.csv'
    t_wall_total_times = []
    
    """
    Run the simulation loop.
    """
    for i in range(simulation_steps):
        n_true = (0 + i) % 20
        print('Step:', i)
        """Main loop for commanding the drone and updating state."""
        print('Command')
        simulator.bridge.pubData(simulator.u_k)

        print('Observe and update state')
        simulator.update(simulator.bridge.getData())

        print(simulator.x_k)
        if i == 0:
            mpc.x0 = np.concatenate(list(simulator.x_k.values()), axis=None)
            mpc.set_initial_guess()

        print('MPC step')
        simulator.u_k['cmd_pose'] = mpc.make_step(np.concatenate(list(simulator.x_k.values()), axis=None))

    save_results([mpc])
