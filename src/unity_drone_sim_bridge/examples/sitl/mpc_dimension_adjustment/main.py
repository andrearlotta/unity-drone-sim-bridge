from unity_drone_sim_bridge.g_func_lib.g_func_tools import load_g
from unity_drone_sim_bridge.examples.sitl.mpc_dimension_adjustment.template_model import template_model, adjust_dimension
from unity_drone_sim_bridge.examples.sitl.mpc_dimension_adjustment.template_mpc import template_mpc
from unity_drone_sim_bridge.examples.sitl.mpc_dimension_adjustment.template_simulator import Simulator
import numpy as np
from do_mpc.data import save_results

INITIAL_LAMBDA_DIM = 1
GRID_SIZE = (100, 100)  # Grid size (number of trees along x and y)
SPACING = 7.5  # Spacing between trees in meters
MIN_DISTANCE = 1.5  # Minimum distance from drone to any tree in meters

def run_simulation(g_function = 'gp', simulation_steps= 100):
    g = load_g(g_function)
    model = template_model(g=g,dim_lambda = INITIAL_LAMBDA_DIM)
    simulator = Simulator(model,
                          dim_lambda=INITIAL_LAMBDA_DIM,
                          grid_size=GRID_SIZE, 
                          spacing=SPACING, 
                          min_distance=MIN_DISTANCE)
    mpc = template_mpc(model=model, get_obs=simulator.mpc_get_obs,
                       get_lambdas=simulator.mpc_get_lambdas, 
                       get_residual_H=simulator.mpc_get_residual_H)
    """
    Run the simulation loop.
    """
    for i in range(simulation_steps):
        print('Step:', i)
        """
        Main loop for commanding the drone and updating state.
        """
        print('Command')
        simulator.bridge.pubData(simulator.u_k)
        simulator.adjust_lambda(i % 20)
        adjust_dimension(model, g, INITIAL_LAMBDA_DIM)
        mpc = template_mpc(model=model, get_obs=simulator.mpc_get_obs,
                           get_lambdas=simulator.mpc_get_lambdas,
                           get_residual_H=simulator.mpc_get_residual_H)
        
        print('Observe and update state')
        simulator.update(simulator.bridge.getData())
        print(simulator.x_k)
        if i == 0:
            mpc.x0 = np.concatenate(list(simulator.x_k.values()), axis=None)
            mpc.set_initial_guess()

        print('MPC step')
        simulator.u_k['cmd_pose'] = mpc.make_step(np.concatenate(list(simulator.x_k.values()), axis=None))

    save_results([mpc])

