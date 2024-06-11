from unity_drone_sim_bridge.g_func_lib.g_func_tools import load_g
from unity_drone_sim_bridge.examples.reduced_order_system.template_model import template_model
from unity_drone_sim_bridge.examples.reduced_order_system.template_mpc import template_mpc
from unity_drone_sim_bridge.examples.reduced_order_system.template_simulator import Simulator
import numpy as np
from do_mpc.data import save_results

def run_simulation(g_function = 'gp', simulation_steps= 100):
    model = template_model(g=load_g(g_function))
    simulator = Simulator(model)
    mpc = template_mpc(model=model, get_obs=simulator.mpc_get_obs,
                       get_lambdas=simulator.mpc_get_lambdas, 
                       get_residual_H=simulator.mpc_get_residual_H)
    """
    Run the simulation loop.
    """
    for i in range(simulation_steps):
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

