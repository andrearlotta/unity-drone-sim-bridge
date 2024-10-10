#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This script implements a Model Predictive Control (MPC) for a drone using CasADi and a pre-trained neural network model.
The goal is to optimize the drone's trajectory by minimizing the change in entropy of a Bayesian update, which is influenced by the drone's state and observations from a neural network model.

Key Components:
- Neural network model loading and integration with CasADi.
- Definition of the Bayesian update and entropy calculations.
- Setup of the MPC problem with CasADi, including objective functions and constraints.
- Simulation loop that applies the MPC to control the drone.

Parameters and objective functions are highlighted throughout the code.
"""

import os
import casadi as ca
import numpy as np
import time
import torch
import csv
import l4casadi as l4c  # Custom module for integrating PyTorch models with CasADi
from unity_drone_sim_bridge.surrogate_lib.nn_models import SurrogateNetworkFixedOutput  # Custom neural network model
from unity_drone_sim_bridge.plot_lib.draw import Draw_MPC_point_stabilization_v1
from unity_drone_sim_bridge.examples.sitl.setup_fake_field import generate_tree_positions
from unity_drone_sim_bridge.surrogate_lib.surrogate_func_tools import g_map_casadi, bayes, fake_bayes, entropy
from unity_drone_sim_bridge.surrogate_lib.nn_tools import LoadNN
import matplotlib.pyplot as plt
from scipy.ndimage import maximum_filter, label, find_objects

# =============================================================================
# Helper Functions
# =============================================================================


def shift_movement(T, t0, x0, u, x_f, f):
    """
    Shifts the control horizon for the MPC.

    Args:
        T: Sampling time.
        t0: Current time.
        x0: Current state.
        u: Current control inputs.
        x_f: Future states.
        f: System dynamics function.

    Returns:
        Updated time, state, control inputs, and future states.
    """
    f_value = f(x0, u[:, 0])
    st = x0 + T * f_value.full()
    t = t0 + T
    u_end = np.concatenate((u[:, 1:], u[:, -1:]), axis=1)
    x_f = np.concatenate((x_f[:, 1:], x_f[:, -1:]), axis=1)

    return t, st, u_end, x_f

# Function to log test data
def log_test_data(test_id, num_iter, total_time, initial_pose, final_pos, distance_error, angle_error, csv_writer):
    csv_writer.writerow([
        test_id,
        num_iter,
        total_time,
        initial_pose[0],
        initial_pose[1],
        initial_pose[2],
        final_pos[0],
        final_pos[1],
        final_pos[2],
        distance_error,
        angle_error
    ])

# =============================================================================
# Main Execution Block
# =============================================================================

if __name__ == "__main__":
    # Parameters
    T = .25             # Sampling time [s]
    N = 150             # Prediction horizon
    rob_diam = 0.3     # Robot diameter [m]
    v_max = 0.5        # Maximum linear velocity [m/s]
    omega_max = np.pi / 18.0  # Maximum angular velocity [rad/s]
    trees = generate_tree_positions([2,1], 8.0) + 0.1
    sim_time = 50     # Total simulation time

    initial_poses = []
    angles = np.linspace(0, 2 * np.pi, num=4, endpoint=False)
    distances = np.arange(3., 8.0, 2.5)
    for distance in distances:
        for angle in angles:
            x = distance * np.cos(angle)
            y = distance * np.sin(angle)
            for theta in angles:
                initial_poses.append([x, y, theta])
    """        [ 4.24264069e+00,  4.24264069e+00,  3.92699082e+00],
        [ 4.24264069e+00,  4.24264069e+00,  4.71238898e+00],
        [ 4.94974747e+00,  4.94974747e+00,  3.92699082e+00],"""
    initial_poses = [

        [-1.10218212e-15, -6.00000000e+00,  4.71238898e+00],
        [-2.12132034e+00, -2.12132034e+00,  3.92699082e+00],
        [-2.12132034e+00, -2.12132034e+00,  4.71238898e+00],
        [-5.51091060e-16, -3.00000000e+00,  4.71238898e+00],
        [-7.34788079e-16, -4.00000000e+00,  4.71238898e+00],
        [ 2.82842712e+00, -2.82842712e+00,  5.49778714e+00],
        [-9.18485099e-16, -5.00000000e+00,  4.71238898e+00],
        [ 3.53553391e+00, -3.53553391e+00,  3.92699082e+00],
        [ 3.53553391e+00, -3.53553391e+00,  5.49778714e+00],
        [-1.10218212e-15, -6.00000000e+00,  3.92699082e+00],
        [ 4.24264069e+00, -4.24264069e+00,  2.35619449e+00],
        [ 4.24264069e+00, -4.24264069e+00,  4.71238898e+00],
        [ 4.94974747e+00, -4.94974747e+00,  3.14159265e+00],
        [ 4.94974747e+00, -4.94974747e+00,  3.92699082e+00],
        [ 4.94974747e+00, -4.94974747e+00,  4.71238898e+00],
        [ 4.94974747e+00, -4.94974747e+00,  5.49778714e+00]]

    # State variables
    x = ca.MX.sym("x")
    y = ca.MX.sym("y")
    theta = ca.MX.sym("theta")
    states = ca.vertcat(x, y, theta)
    n_states = states.size()[0]

    # Control variables
    vx = ca.MX.sym("vx")
    vy = ca.MX.sym("vy")
    w = ca.MX.sym("w")
    controls = ca.vertcat(vx, vy, w)
    n_controls = controls.size()[0]

    # System dynamics (rhs)
    rhs = controls

    # Define the system dynamics function
    f = ca.Function("f", [states, controls], [rhs], ["input_state", "control_input"], ["rhs"])

    # Load the neural network model (same as in your previous code)
    hidden_size = 16
    hidden_layer = 2
    n_input = 3
    gpu = False  # Use GPU if available
    model_ = LoadNN(hidden_size, hidden_layer, n_input)
    l4c_model = l4c.L4CasADi(model_, batched=True, device="cuda" if gpu else "cpu")
    model = g_map_casadi(l4c_model, trees)

    for i in range(20):
        x_sym = ca.MX.sym('x', 2, 3)
        y_sym = l4c_model(x_sym)
        f_ = ca.Function('y_', [x_sym], [y_sym])
        df_ = ca.Function('dy', [x_sym], [ca.jacobian(y_sym, x_sym)])

        x = ca.DM([[0., 2., np.pi/18], [0., 2., np.pi/18]])
        l4c_model(x)
        f_(x)
        df_(x)
        
    # Define the MPC optimization variables
    U = ca.MX.sym("U", n_controls, N)
    X = ca.MX.sym("X", n_states, N + 1)
    P = ca.MX.sym("P", n_states + len(trees))  # Parameter vector (initial state and previous bayes)

    # Cost function weights
    Q = np.array([[1e-1/(len(trees)*N)]])  # Weight for bayes change
    R = np.diag([1e-6, 1e-6, 1e-6])  # Weights for control inputs

    # Objective function initialization
    obj = 0  # Objective function
    g = []   # Constraints list
    g.append(X[:, 0] - P[:n_states])  # Initial state constraint

    # Target parameters
    bayes_list = []
    bayes_list.append(P[n_states:])  # Initial bayes passed as a parameter
    entropy_list =[]
    entropy_list.append(entropy(bayes_list[-1]))

    # Define the MPC problem over the prediction horizon
    for i in range(N):
        # State transition using system dynamics
        x_next_ = f(X[:, i], U[:, i]) * T + X[:, i]
        model_output_i_plus_1 = model(X[:, i], trees)
        
        # Perform Bayesian update
        bayes_list.append(bayes(bayes_list[-1], model_output_i_plus_1))
        entropy_list.append(entropy(bayes_list[-1]))

        # Objective function: Minimize the change in bayes and control effort
        obj += Q * entropy_list[i] 

        # Add the state transition constraint
        g.append(X[:, i + 1] - x_next_)
  
    #### obstacle definition
    obs_diam = 2.5
    for i in range(N+1):
        for j in range(len(trees)):
            g.append(ca.sqrt((X[0, i]-trees[j,0])**2+(X[1, i]-trees[j,1])**2)-(obs_diam))

    # Flatten optimization variables for the solver
    opt_variables = ca.vertcat(ca.reshape(U, -1, 1), ca.reshape(X, -1, 1))

    # Define the Nonlinear Programming (NLP) problem for the solver
    nlp_prob = {"f": obj, "x": opt_variables, "p": P, "g": ca.vertcat(*g)}
    opts_setting = {
        "ipopt.linear_solver": "ma27",
        "ipopt.max_iter": 500,
        "ipopt.timing_statistics": 'no',

        "ipopt.sb":"yes",

        "ipopt.nlp_scaling_method":"equilibration-based",
        "ipopt.obj_scaling_factor": +1,
        "ipopt.nlp_scaling_min_value" : 1e-10,
        "ipopt.tol": 1e-2,
        "print_time": False,

        "ipopt.print_info_string": "yes",
        "ipopt.output_file": "nmpc_mul_shooting_mx_l4casadi_output",
        "ipopt.print_level":5,
        "ipopt.hessian_approximation" : 'limited-memory',
        "ipopt.warm_start_init_point" : 'yes',
        "ipopt.mu_strategy" : 'monotone',
        #"ipopt.mu_init" : 1e-4,
        "ipopt.warm_start_bound_push" : 1e-6,
        "ipopt.warm_start_mult_bound_push" : 1e-2,
        "ipopt.line_search_method" : "filter",
        "ipopt.alpha_for_y":"primal-and-full",
        #"ipopt.accept_every_trial_step": "yes"
    }

    # Create the solver instance
    solver = ca.nlpsol("solver", "ipopt", nlp_prob, opts_setting)

    # Set constraints bounds
    lbg = []  # Lower bound for constraints
    ubg = []  # Upper bound for constraints
    lbx = []   # Lower bounds on variables
    ubx = []   # Upper bounds on variables

    for _ in range(N+1):
        lbg.append(0.0)
        ubg.append(0.0)
        lbg.append(0.0)
        ubg.append(0.0)
        lbg.append(0.0)
        ubg.append(0.0)
    for _ in range((N+1)*len(trees)):
        lbg.append(0.0)
        ubg.append(np.inf)

    # Set bounds for control inputs
    for _ in range(N):
        lbx.extend([-v_max, -v_max, -omega_max])
        ubx.extend([v_max, v_max, omega_max])

    # Set bounds for state variables
    for _ in range(N + 1):
        lbx.extend([-30.0, -30.0, -np.inf])
        ubx.extend([30.0, 30.0, np.inf])

    # Create directory for plots
    plots_dir = 'nmpc_entropy_test'
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)

    with open('nmpc_entropy_test.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        ## Update the header to include distance and angle errors
        writer.writerow(['Test ID', 'Iterations', 'Total Time (s)', 'Initial X', 'Initial Y', 'Initial Theta','Final X', 'Final Y', 'Final Theta', 'Distance Error', 'Angle Error'])

        for test_id, initial_pose in enumerate(initial_poses):

            print(f"Running test {test_id + 1} with initial pose: {initial_pose}")

            # Initial conditions
            t0 = 0.0
            x0 = np.array(initial_pose).reshape(-1, 1)  # Initial state
            x0_ = x0.copy()
            y0 = np.array([0.5]*len(trees))  # Initial observation
            b0 = bayes(y0, y0)
            e0 = entropy(b0)
            x_m = np.zeros((n_states, N + 1))
            next_states = x_m.copy()
            u0 = np.zeros((n_controls, N))  # Initial control inputs
            x_c = []  # State history
            u_c = []  # Control input history
            t_c = [t0]  # Time history
            xx = []     # State trajectory
            lam_g0 = []
            lam_x0 = []
            index_t = []
            meas_history = np.array([y0]).flatten()
            bayes_history = np.array([b0]).flatten()
            entropy_history = [e0]
            # Start MPC loop
            mpciter = 0
            start_time = time.time()
            
            while e0 > 0.1:
                # Set the parameters for the solver
                c_p = np.concatenate((x0.reshape(-1, 1), np.array([b0]).reshape(-1, 1)))
                init_control = np.concatenate((u0.T.reshape(-1, 1), next_states.T.reshape(-1, 1)))

                # Solve the optimization problem
                if mpciter == 0:
                    for _ in range(5):
                        solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
                t_ = time.time()
                res = solver(
                    x0=init_control,
                    p=c_p,
                    lbg=lbg,
                    lbx=lbx,
                    ubg=ubg,
                    ubx=ubx,
                )
                index_t.append(time.time() - t_)

                # Extract the optimal control inputs and predicted states
                estimated_opt = res["x"].full()
                lam_g0 = res["lam_g"]
                lam_x0 = res["lam_x"]

                u0 = estimated_opt[: N * n_controls].reshape(N, n_controls).T
                x_m = estimated_opt[N * n_controls:].reshape(N + 1, n_states).T

                # Store the state and control trajectories
                x_c.append(x_m.T)
                u_c.append(u0[:, 0])
                t_c.append(t0)

                # Apply the first control input and update the state
                t0, x0, u0, next_states = shift_movement(T, t0, x0, u0, x_m, f)

                # Perform the Bayesian update and calculate entropy
                y1 = model(x0, trees)
                y0 = y1
                

                # Update Bayesian values for trees within 6 meters
                b1 = bayes(b0, y1)
                print(y1)
                print(b0)
                b0 = b1
                e0 = entropy(b0)  # Calculate entropy after Bayesian update
                print(e0)
                
                # Update history
                bayes_history = np.vstack((bayes_history, np.array([b0]).flatten()))
                meas_history = np.vstack((meas_history, np.array([y1]).flatten()))
                entropy_history.append(e0)  # Append entropy value

                # Store the state trajectory
                xx.append(x0)

                mpciter += 1

            # Total simulation time
            total_time = time.time() - start_time
            final_position = x0.flatten()

            # Call the plotting class and get the errors
            draw_result = Draw_MPC_point_stabilization_v1(
                rob_diam=rob_diam,
                init_state=initial_pose,
                target_state=None,
                robot_states=xx,
                z= ((meas_history- 0.5) / (0.5)) * (0.78 - 0.5) + 0.5,
                function_to_plot=model_,
                trees=trees,
                test_id =test_id + 1,
                plots_dir=plots_dir,
                obstacle= trees,
                obstacle_r=2.9
            )

            # Get the errors from the draw_result object
            distance_error = draw_result.min_distance
            angle_error = np.rad2deg(draw_result.min_angle_error)

            # Log results to CSV
            log_test_data(
                test_id + 1,
                mpciter,
                total_time,
                initial_pose,
                final_position,
                distance_error,
                angle_error,
                writer
            )

            print("Total Execution Time [s]:", total_time)

            # Visualization
            plt.figure()
            plt.plot(t_c, np.array(entropy_history).flatten(), label='Entropy')
            for i in range(bayes_history.shape[-1]):
                #plt.plot(t_c, meas_history[:, i], marker='x', label=f'NN output {i}')
                plt.plot(t_c, bayes_history[:, i], marker='o', label=f'bayes {i}')
            plt.legend()
            plt.title(f'Bayes Update Test {test_id + 1}')
            plt.xlabel('Time [s]')
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, f'BayesNNUpdate_Test_{test_id + 1}.png'))


            plt.figure()
            u_c = np.array(u_c)
            plt.plot(t_c[1:], u_c[:, 0], marker='x', label=f'dx')
            plt.plot(t_c[1:], u_c[:, 1], marker='o', label=f'dy')
            plt.plot(t_c[1:], u_c[:, 2], marker='o', label=f'dyaw')
            plt.legend()
            plt.title(f'Output Control Test {test_id + 1}')
            plt.xlabel('Time [s]')
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, f'OutputControl_Test_{test_id + 1}.png'))

            plt.figure()
            plt.plot(np.arange(len(index_t)), index_t, marker='o')
            plt.title(f'Solver Time Test {test_id + 1}')
            plt.xlabel('Iterations [#]')
            plt.grid(True)
            plt.savefig(os.path.join(plots_dir, f'SolverTime_Test_{test_id + 1}.png'))

    print("All tests completed.")
