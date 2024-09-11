from unity_drone_sim_bridge.surrogate_lib.surrogate_func_tools import load_g
from unity_drone_sim_bridge.examples.yolo_order_system.template_model import template_model
from unity_drone_sim_bridge.examples.yolo_order_system.template_mpc import template_mpc
from unity_drone_sim_bridge.examples.yolo_order_system.template_simulator import Simulator
import numpy as np
from do_mpc.data import save_results
import time


def run_simulation(g_function = 'mlp', simulation_steps= 10, rt=False, gpu=False, viz=True, use_yolo=True):
    g=load_g(g_function, 
             rt=rt, 
             gpu=gpu, 
             synthetic=False,
             hidden_layer=2,
             hidden_size=64, 
             use_yolo=use_yolo)
    
    model = template_model(g=g, 
                           dim_lambda=1)
    
    simulator = Simulator(model, 
                          dim_lambda=1)
    
    mpc = template_mpc(model=model, g=g,
                       get_nn_input= simulator.mpc_nn_inputs)
    
    """
    Run the simulation loop.
    """
    frequency = 5  # Hz
    period = 1 / frequency  # period in seconds

    for i in range(simulation_steps):
        start_time = time.time()  # Record the start time of the loop

        print('Step:', i)

        """
        command transmission
        """
        print('Command')
        print(simulator.u_k)
        simulator.bridge.pubData(simulator.u_k)
        
        """
        State Observation
        """
        print('Observe and update state')
        simulator.update(simulator.bridge.getData())
        print(simulator.x_k)
        if i == 0:
            mpc.x0 = simulator.get_mpc_x0()
            mpc.set_initial_guess()
            simulator.bridge.pubData({"predicted_path": mpc.data.prediction(('_x', 'x_robot'), -1), "tree_markers": simulator})
        
        """
        solver execution
        """
        print('MPC step')
        simulator.u_k['cmd_pose'] = mpc.make_step(simulator.get_mpc_x0())


        # Calculate the elapsed time for the loop iteration
        elapsed_time = time.time() - start_time

        # Sleep for the remaining time to maintain the 10 Hz frequency
        time_to_sleep = period - elapsed_time
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)

    save_results([mpc])

