from unity_drone_sim_bridge.surrogate_lib.nn_tools import LoadNN
import l4casadi as l4c

from unity_drone_sim_bridge.surrogate_lib.surrogate_func_tools import g_map_casadi, bayes, entropy
from unity_drone_sim_bridge.examples.casadi.nmpc.template_simulator import Simulator
import numpy as np
import time

from unity_drone_sim_bridge.ros_com_lib.bridge_class import BridgeClass
from unity_drone_sim_bridge.ros_com_lib.sensors import SENSORS
from unity_drone_sim_bridge.generic_tools import *

from unity_drone_sim_bridge.plot_lib.draw import Draw_MPC_point_stabilization_v1  # Custom visualization module

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


def run_simulation(g_function = 'mlp', simulation_steps= 10, rt=False, gpu=False, viz=True, n_inputs=3):
    bridge = BridgeClass(SENSORS)
    trees_pos = np.array(bridge.callServer({"trees_poses": None})["trees_poses"])
    
    # Parameters
    T = 0.25            # Sampling time [s]
    N = 30              # Prediction horizon
    rob_diam = 0.3      # Robot diameter [m]
    v_max = .5        # Maximum linear velocity [m/s]
    omega_max =  np.pi / 18.0 # Maximum angular velocity [rad/s]

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

    # Load the neural network model
    hidden_size = 16
    hidden_layer = 2
    n_input = 3
    gpu = False  # Set to True if GPU is available
    model_ = LoadNN(hidden_size, hidden_layer, n_input)
    l4c_model = l4c.L4CasADi(model_, batched=True, device="cuda" if gpu else "cpu")
    model = g_map_casadi(l4c_model, trees_pos)

# =============================================================================
# MPC DEFINITION
# =============================================================================
    # Define the MPC optimization variables
    U = ca.MX.sym("U", n_controls, N)
    X = ca.MX.sym("X", n_states, N + 1)
    P = ca.MX.sym("P", n_states + len(trees_pos))  # Parameter vector (initial state and previous bayes)

    # Cost function weights
    Q = np.array([[1e-1/(len(trees_pos)*N)]])  # Weight for bayes change
    R = np.diag([1e-6, 1e-6, 1e-6])  # Weights for control input

    # Objective function initialization
    obj = 0  # Objective function
    g = []   # Constraints list
    g.append(X[:, 0] - P[:n_states])  # Initial state constrain

    # Target parameters
    bayes_list = []
    bayes_list.append(P[n_states:])  # Initial bayes passed as a parameter
    entropy_list =[]
    entropy_list.append(entropy(bayes_list[-1]))

    # Define the MPC problem over the prediction horizon
    for i in range(N):
        # State transition using system dynamics
        x_next_ = f(X[:, i], U[:, i]) * T + X[:, i]
        model_output_i_plus_1 = model(X[:, i], trees_pos)
        
        # Perform Bayesian update
        bayes_list.append(bayes(bayes_list[-1], model_output_i_plus_1))
        entropy_list.append(entropy(bayes_list[-1]))
        
        # Objective function: Minimize the change in bayes and control effort
        obj += Q * entropy_list[i] 

        # Add the state transition constraint
        g.append(X[:, i + 1] - x_next_)

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


    # Set bounds for control inputs
    for _ in range(N):
        lbx.extend([-v_max, -v_max, -omega_max])
        ubx.extend([v_max, v_max, omega_max])

    # Set bounds for state variables
    for _ in range(N + 1):
        lbx.extend([-30.0, -30.0, -np.inf])
        ubx.extend([30.0, 30.0, np.inf])

# =============================================================================
# SIMULATION DEFINITION
# =============================================================================

    simulator = Simulator(states, trees=trees_pos,
                          dim_lambda=1)
    
    t0 = 0.0

    x_m = np.zeros((n_states, N + 1))
    next_states = x_m.copy()
    
    u0 = np.zeros((n_controls, N))  # Initial control inputs
    x_c = []  # State history
    u_c = []  # Control input history
    t_c = [t0]  # Time history
    xx = []     # State trajectory

    meas_history =  simulator.x_k['y'].flatten()
    bayes_history =  simulator.x_k['lambda'].flatten()  # To store bayes values over time
    entropy_history = [entropy(bayes_history)]

    sim_time = 150  # Total simulation time

    # Start MPC loop
    mpciter = 0
    start_time = time.time()
    index_t = []

    """
    Run the simulation loop.
    """
    # Main MPC loop
    while mpciter * T < sim_time :
        print('Step:', mpciter)

        
        """
        State Observation
        """
        print('Observe and update state')
        simulator.update(bridge.getData())
        x0 = simulator.x_k["x_robot"].reshape(-1, 1)
        xx.append(simulator.x_k["x_robot"].reshape(-1, 1))
        print("y0: ", np.array([simulator.x_k["y"]]).flatten())
        meas_history = np.vstack((meas_history, np.array([model(x0, trees_pos)]).flatten()))
        bayes_history = np.vstack((bayes_history, np.array([simulator.x_k["lambda"]]).flatten()))
        entropy_history.append(np.array(entropy(meas_history[-1])).flatten())

        c_p =  np.concatenate((x0, model(x0, trees_pos))).reshape(-1, 1)
        init_control = np.concatenate((u0.T.reshape(-1, 1), next_states.T.reshape(-1, 1)))
        
        
        # Solve the optimization problem
        if mpciter == 0 :
            for i in range(5):  solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        
        t_ = time.time()
        res = solver(x0=init_control, p=c_p, lbg=lbg, lbx=lbx, ubg=ubg, ubx=ubx)
        index_t.append(time.time() - t_)

        # Extract the optimal control inputs and predicted states
        estimated_opt = res["x"].full()
        u0 = estimated_opt[: N * n_controls].reshape(N, n_controls).T
        x_m = estimated_opt[N * n_controls :].reshape(N + 1, n_states).T

        # Store the state and control trajectories
        x_c.append(x_m.T)
        u_c.append(u0[:, 0])
        t_c.append(t0)


        """
        command transmission
        """
        simulator.u_k['cmd_pose'] = T * f(x0, u0[:, 0])

        # Apply the first control input and update the state
        t0, x0, u0, next_states = shift_movement(T, t0, x0, u0, x_m, f)
        bridge.pubData({"predicted_path": next_states, "tree_markers": simulator})

        bridge.pubData(simulator.u_k)

        # Calculate the elapsed time for the loop iteration
        elapsed_time = time.time() - start_time

        # Sleep for the remaining time to maintain the 10 Hz frequency

        time.sleep(T)
        
        mpciter += 1



    # Compute average computation time per iteration
    t_v = np.array(index_t)
    print("Average solver time per iteration and standard deviation:", t_v.mean(), t_v.std())
    print("Average total time per iteration:", (time.time() - start_time) / mpciter)

    draw_result = Draw_MPC_point_stabilization_v1(
        rob_diam=rob_diam,
        init_state=xx[0],
        target_state=None,
        robot_states=xx,
        z=((meas_history- 0.5) / (0.5)) * (0.78 - 0.5) + 0.5,
        function_to_plot=model_,
        trees=trees_pos,
        obstacle=trees_pos,
        obstacle_r=0
    )

    # Visualization of the results

    import matplotlib.pyplot as plt
    # Plot the bayes history
    plt.figure()
    plt.plot(t_c, np.array(entropy_history).flatten(), label='Entropy')
    for i in range(bayes_history.shape[-1]):
        #plt.plot(t_c, meas_history[:, i], marker='x', label=f'NN output {i}')
        plt.plot(t_c, bayes_history[:, i], marker='o', label=f'bayes {i}')
    plt.title('Bayes Update and NN Surrogate Output over Time')
    plt.xlabel('Time [s]')
    plt.legend()
    plt.grid(True)
    plt.savefig('BayesNNOverTime.eps', format='eps')

    plt.figure()
    plt.plot(np.arange(len(index_t)), index_t, marker='o', label='bayes')
    plt.title('Solver Time over Iteration')
    plt.xlabel('Iterations [#]')
    plt.ylabel('Time [s]')
    plt.grid(True)
    plt.savefig('SolverTimeOverIteration.eps', format='eps')

    plt.figure()
    plt.plot(np.arange(len(u_c)), np.asarray(u_c)[:,0], marker='o', label='ux')
    plt.plot(np.arange(len(u_c)), np.asarray(u_c)[:,1], marker='o', label='uy')
    plt.plot(np.arange(len(u_c)), np.rad2deg(np.asarray(u_c)[:,2]), marker='o', label='utheta')
    plt.title('U over Iteration')
    plt.xlabel('Iterations [#]')
    plt.ylabel('dX [m][deg]')
    plt.legend()
    plt.grid(True)

    plt.show()


