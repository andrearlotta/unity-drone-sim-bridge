from unity_drone_sim_bridge.surrogate_lib.surrogate_func_tools import load_g
from unity_drone_sim_bridge.examples.reduced_order_system.template_model import template_model
from unity_drone_sim_bridge.examples.reduced_order_system.template_mpc import template_mpc
from unity_drone_sim_bridge.examples.reduced_order_system.template_simulator import Simulator
from unity_drone_sim_bridge.ros_com_lib.bridge_class import BridgeClass
from unity_drone_sim_bridge.ros_com_lib.sensors import SENSORS
import numpy as np
from do_mpc.data import save_results
import time
def run_simulation(g_function = 'gp',  simulation_steps= 10, rt=False, gpu=False, viz=True, n_inputs=3):
    bridge = BridgeClass(SENSORS)
    trees_pos = np.array(bridge.callServer({"trees_poses": None})["trees_poses"])
    
    g=load_g(g_function, 
             rt=rt, 
             gpu=gpu, 
             synthetic=False,
             hidden_layer=2,
             hidden_size=16, 
             n_inputs=n_inputs)
    model = template_model(g=g,dim_lambda=4, dim_obs=4)
    simulator = Simulator(model,trees_pos, dim_lambda=4, dim_obs=4)
    mpc = template_mpc(model=model, get_obs=simulator.mpc_get_obs,
                       get_lambdas=simulator.mpc_get_lambdas, 
                       get_residual_H=simulator.mpc_get_residual_H)
    """
    Run the simulation loop.
    """
    frequency = 2  # Hz
    period = 1 / frequency  # period in seconds
    for i in range(simulation_steps):
        start_time = time.time()  # Record the start time of the loop
        print('Step:', i)

        print('Observe and update state')
        simulator.update(bridge.getData())
        print(simulator.x_k)
        if i == 0:
            mpc.x0 = np.concatenate(list(simulator.x_k.values()), axis=None)
            mpc.set_initial_guess()            

        print('MPC step')
        simulator.u_k['cmd_pose'] = mpc.make_step(np.concatenate(list(simulator.x_k.values()), axis=None)) * period
        
        """Main loop for commanding the drone and updating state."""
        print('Command')
        bridge.pubData(simulator.u_k)
        bridge.pubData({"predicted_path": mpc.data.prediction(('_x', 'x_robot'), -1), "tree_markers": simulator})
        
        # Calculate the elapsed time for the loop iteration
        elapsed_time = time.time() - start_time

        # Sleep for the remaining time to maintain the 10 Hz frequency
        time_to_sleep = period - elapsed_time
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
    save_results([mpc])

