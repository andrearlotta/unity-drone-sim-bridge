from unity_drone_sim_bridge.surrogate_lib.surrogate_func_tools import load_g
from unity_drone_sim_bridge.examples.sitl.fixed_order_system.template_model import template_model
from unity_drone_sim_bridge.examples.sitl.fixed_order_system.template_mpc import template_mpc
from unity_drone_sim_bridge.examples.sitl.fixed_order_system.template_simulator import Simulator
from do_mpc.data import save_results
import numpy as np
import csv
# Constants
GRID_SIZE = (100, 100)  # Grid size (number of trees along x and y)
SPACING = 7.5  # Spacing between trees in meters
MIN_DISTANCE = 1.5  # Minimum distance from drone to any tree in meters

def run_simulation(g_function = 'gp', simulation_steps= 100, rt=True, gpu=True):
    dim_lambda = 4 #2**simulation_steps
    g=load_g(g_function, rt=rt, gpu=gpu, synthetic=False, hidden_layer=2,hidden_size=64, n_inputs=3)
    model = template_model(g=g,dim_lambda=dim_lambda)
    simulator = Simulator(model,
                          dim_lambda=dim_lambda,
                          grid_size=GRID_SIZE, 
                          spacing=SPACING, 
                          min_distance=MIN_DISTANCE)
    n_true = 2**0
    mpc = template_mpc( model=model, get_obs=simulator.mpc_get_obs,
                        get_lambdas=simulator.mpc_get_lambdas, 
                        get_residual_H=simulator.mpc_get_residual_H,
                        get_cond_y=lambda: np.vstack( [np.ones((n_true,1)), np.zeros((dim_lambda-n_true,1))]))
    """
    Time Metrics
    """
    r = '_rt' if rt else '_normal'
    g_ = '_gpu' if gpu else '_cpu'
    csv_filename = f'setup_time_{g_function}{r}{g_}_fixed_dim_input_refactor_weight_obstacle.csv'
    t_wall_total_times = []
    x_mpc_dims = []
    """
    Run the simulation loop.
    """
    for i in range(simulation_steps):
        n_true = 2**(1+i)
        print('Step:', i)
        """Main loop for commanding the drone and updating state."""
        print('Command')
        simulator.bridge.pubData(simulator.u_k)

        print('Observe and update state')
        simulator.update(simulator.bridge.getData())

        #print(simulator.x_k)
        if i == 0:
            mpc.x0 = np.concatenate(list(simulator.x_k.values()), axis=None)
            mpc.set_initial_guess()

        print('MPC step')
        simulator.u_k['cmd_pose'] = mpc.make_step(np.concatenate(list(simulator.x_k.values()), axis=None))
        # Append t_wall_total data
        t_wall_total_times.append(mpc.data['t_wall_total'][-1,0])
        x_mpc_dims.append(n_true)

    print(mpc.data['t_wall_total'])

    save_results([mpc])
    # Save setup times and t_wall_total times to a CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 't_wall_total (s)'])
        for i, (x_mpc_dim, t_wall_total) in enumerate(zip(x_mpc_dims, t_wall_total_times)):
            writer.writerow([x_mpc_dim, t_wall_total])
    print(csv_filename)

if __name__ == '__main__':
    # Run the simulation with the default parameters
    for g in ['mlp']:
        for rt in [True]:
            for gpu in [True]:
                run_simulation(g_function=g, rt=rt, gpu=gpu, simulation_steps=4)