from unity_drone_sim_bridge.g_func_lib.g_func_tools import load_g
from unity_drone_sim_bridge.examples.sitl.mpc_dimension_adjustment.template_model import template_model, adjust_dimension
from unity_drone_sim_bridge.examples.sitl.mpc_dimension_adjustment.template_mpc import template_mpc
from unity_drone_sim_bridge.examples.sitl.mpc_dimension_adjustment.template_simulator import Simulator
from do_mpc.data import save_results
import numpy as np
import csv
import time

INITIAL_LAMBDA_DIM = 2
GRID_SIZE = (100, 100)  # Grid size (number of trees along x and y)
SPACING = 7.5  # Spacing between trees in meters
MIN_DISTANCE = 1.5  # Minimum distance from drone to any tree in meters

def run_simulation(g_function = 'gp', cond=True, simulation_steps= 100, rt= True, gpu=True):
    g = load_g(g_function, rt=rt, gpu=gpu)
    model = template_model(g=g,dim_lambda = INITIAL_LAMBDA_DIM, dim_obs = 5, cond=cond)
    simulator = Simulator(model,
                          dim_lambda=INITIAL_LAMBDA_DIM,
                          grid_size=GRID_SIZE,
                          spacing=SPACING, 
                          min_distance=MIN_DISTANCE)
    mpc = template_mpc(model=model, get_obs=simulator.mpc_get_obs,
                       get_lambdas=simulator.mpc_get_lambdas, 
                       get_residual_H=simulator.mpc_get_residual_H)
    """
    Time Metrics
    """
    c =  'cond' if cond else 'no_cond'
    r = '_rt' if rt else ''
    g_ = '_gpu' if gpu else '_cpu'
    csv_filename = f'setup_time_{g_function}_variable_dim_{c}{r}{g_}_refactor_mod.csv'
    t_wall_total_times = []
    setup_times = []
    x_mpc_dims = []
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
        simulator.adjust_lambda(i + INITIAL_LAMBDA_DIM % 20)
        #adjust_dimension(model, g, INITIAL_LAMBDA_DIM)
        model = template_model(g=g,dim_lambda = i + INITIAL_LAMBDA_DIM % 20, dim_obs = 5, cond=cond)
        start = time.time()
        mpc = template_mpc(model=model, get_obs=simulator.mpc_get_obs,
                           get_lambdas=simulator.mpc_get_lambdas,
                           get_residual_H=simulator.mpc_get_residual_H,
                           )
        setup_times.append(time.time() - start)
        print('Observe and update state')
        simulator.update(simulator.bridge.getData())
        mpc.x0 = np.concatenate(list(simulator.x_k.values()), axis=None)
        mpc.set_initial_guess()

        print('MPC step')
        simulator.u_k['cmd_pose'] = mpc.make_step(np.concatenate(list(simulator.x_k.values()), axis=None))
        
        # Append t_wall_total data
        t_wall_total_times.append(mpc.data['t_wall_total'])
        x_mpc_dims.append(i + INITIAL_LAMBDA_DIM % 20)

    save_results([mpc])
    # Save setup times and t_wall_total times to a CSV file
    with open(csv_filename, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Iteration', 'Setup Time (s)', 't_wall_total (s)'])
        for i, (x_mpc_dim, setup_time, t_wall_total) in enumerate(zip(x_mpc_dims, setup_times, t_wall_total_times)):
            writer.writerow([x_mpc_dim, setup_time, *t_wall_total[0]])
    print(csv_filename)
if __name__ == '__main__':
    # Run the simulation with the default parameters
    for g in ['mlp']:
        for cond in [True, False]:
            for rt in [True, False]:
                for gpu in [True, False]:
                    run_simulation(g_function=g, cond=cond, rt=rt, gpu=gpu, simulation_steps=20)