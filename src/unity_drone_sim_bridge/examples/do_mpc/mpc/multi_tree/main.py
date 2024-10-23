from unity_drone_sim_bridge.examples.do_mpc.mpc.multi_tree.template_model import template_model
from unity_drone_sim_bridge.examples.do_mpc.mpc.multi_tree.template_mpc import template_mpc
from unity_drone_sim_bridge.examples.do_mpc.mpc.multi_tree.template_simulator import Simulator
from unity_drone_sim_bridge.ros_com_lib.bridge_class import BridgeClass
from unity_drone_sim_bridge.ros_com_lib.sensors import SENSORS
import numpy as np
from do_mpc.data import save_results
import time
from unity_drone_sim_bridge.surrogate_lib.surrogate_func_tools import entropy_np

def run_simulation(simulation_steps=10, T=0.25, N=30):
    bridge = BridgeClass(SENSORS)

    x_trees = np.array(bridge.callServer({"trees_poses": None})["trees_poses"])

    model = template_model(x_trees)
    
    simulator = Simulator(  model,
                            x_trees)
    
    mpc = template_mpc(model, T, N)

    """
    Run the simulation loop.
    """
    i = 0
    while i < simulation_steps and entropy_np(simulator.x_k["lambda"])>1e-2:
        print(entropy_np(simulator.x_k["lambda"]))
        start_time = time.time()  # Record the start time of the loop

        print('Step:', i)

        print('Observe and update state')
        simulator.update(bridge.getData())

        if i == 0:
            mpc.x0 = simulator.get_mpc_x0()
            mpc.set_initial_guess()
        
        """
        solver execution
        """
        print('MPC step')
        simulator.u_k['cmd_pose'] = mpc.make_step(simulator.get_mpc_x0()) * T

        print('Command')

        bridge.pubData({"predicted_path": mpc.data.prediction(('_x', 'x_robot'), -1), "tree_markers": simulator})
        bridge.pubData(simulator.u_k)

        # Calculate the elapsed time for the loop iteration
        elapsed_time = time.time() - start_time

        # Sleep for the remaining time to maintain the 10 Hz frequency
        time_to_sleep = T*5
        if time_to_sleep > 0:
            time.sleep(time_to_sleep)
        i += 1
    save_results([mpc])

