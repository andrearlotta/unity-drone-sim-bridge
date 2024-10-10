import numpy as np
from unity_drone_sim_bridge.surrogate_lib.gp_tools import load_ca_gp
from unity_drone_sim_bridge.surrogate_lib.nn_tools import LoadNN
from unity_drone_sim_bridge.generic_tools import *
import casadi as ca
import l4casadi as l4c
'''
setup g functions tools
'''

def load_g(mode='gp', hidden_size=16, hidden_layer=2, synthetic=True, rt=False, gpu=True, n_inputs=3):
    if  mode == 'mlp':
        mlp = LoadNN(hidden_size,hidden_layer, synthetic=synthetic, rt=rt, gpu=gpu, n_inputs=n_inputs)
        return l4c.L4CasADi(mlp, batched=True, device="cuda" if gpu else "cpu")
    elif mode == 'gp':
        gp =  load_ca_gp(synthetic=synthetic)
        return lambda drone_pos,drone_yaw, tree_pos_single: g_gp(drone_pos,drone_yaw, tree_pos_single, gp)

def g_gp(drone_pos,drone_yaw, tree_pos_single, gp):
    return  ca.MX(gp.predict(drone_yaw, [], np.zeros((1,1)))[0])


'''
casadi map tools
'''

def g_map_casadi(l4c_model, x_trees, norm=True):
    """
    Maps a CasADi function over a set of tree positions.

    Args:
        l4c_model: CasADi-compatible neural network model.
        x_trees: Array of tree positions.

    Returns:
        A CasADi function that computes the model output for all tree positions.
    """
    drone_state1 = ca.MX.sym('drone_pos', 3,1)
    wrapped_angle = ca.fmod(drone_state1[-1] + ca.pi, 2 * ca.pi) - ca.pi 

    trees_lambda = ca.MX.sym('trees_lambda', x_trees.shape)
    difference = ca.repmat(drone_state1[:2].T, trees_lambda.shape[0], 1) - trees_lambda

    casadi_quad_approx_sym_out =l4c_model(
        ca.horzcat(difference, ca.repmat(wrapped_angle, trees_lambda.shape[0], 1))
    )
    # Normalize the output to the range [0.5, 1.0]
    if norm : casadi_quad_approx_sym_out = ((casadi_quad_approx_sym_out - 0.5) / (0.78 - 0.5)) * (1.0 - 0.5) + 0.5
    f = ca.Function('F_single', [drone_state1, trees_lambda], [casadi_quad_approx_sym_out])

    return f

'''
casadi map tools for fixed model dimension
'''

def setup_g_inline_casadi_fixed_cond(f):
    drone_pos = ca.MX.sym('drone_pos', 2)
    drone_yaw = ca.MX.sym('drone_yaw')
    
    tree_pos_single = ca.MX.sym('tree_pos_single',2)
    var_cond_single = ca.MX.sym('tree_cond')

    # Calculate condition for single tree
    condition_single = fov_weight_fun_casadi(drone_pos, drone_yaw, ca.reshape(tree_pos_single, (1, 2)))
    # Apply the condition to single inputs
    y_single = f(drone_pos,drone_yaw, tree_pos_single) # condition_single * (f(drone_pos,drone_yaw, tree_pos_single)  - 0.5) + 0.5 

    cond_y_single = ca.if_else(var_cond_single, y_single, 0.5)

    # Create CasADi function for single evaluation
    return ca.Function('F_single', [drone_pos, drone_yaw, tree_pos_single, var_cond_single], [cond_y_single])

def g_map_casadi_fixed_cond(F_single, x_trees_dim, cond_=None):
    F_mapped = F_single.map(x_trees_dim[0]) 
    drone_pos_sym = ca.MX.sym('drone_pos', 2)
    drone_yaw_sym = ca.MX.sym('drone_yaw')
    tree_lambda_sym = ca.MX.sym('tree_lambda', x_trees_dim)
    cond_sym = ca.MX.sym('tree_cond', x_trees_dim[0])
    # Use mapped function
    y_all = F_mapped(drone_pos_sym, drone_yaw_sym, tree_lambda_sym.T, cond_sym).T

    return ca.Function('F_final', [drone_pos_sym, drone_yaw_sym, tree_lambda_sym, cond_sym], [y_all])

'''
cost function
'''

def entropy(lambda_):
    """
    Calculates the entropy of a probability distribution.

    Args:
        lambda_: Probability distribution (CasADi MX or DM object).

    Returns:
        Entropy value (CasADi MX or DM object).
    """
    # Adding a small epsilon to avoid log(0)
    epsilon = 1e-12
    lambda_ = ca.fmax(lambda_, epsilon)  # To avoid log(0) issues
    one_minus_lambda = ca.fmax(1 - lambda_, epsilon)

    return -ca.sum1(lambda_ * ca.log10(lambda_) / ca.log10(2) +
                    one_minus_lambda * ca.log10(one_minus_lambda) / ca.log10(2))

def entropy_np(lambda_):
    """
    Calculates the entropy of a probability distribution.

    Args:
        lambda_: Probability distribution (numpy array).

    Returns:
        Entropy value (float).
    """
    # Adding a small epsilon to avoid log(0)
    epsilon = 1e-12
    lambda_ = np.maximum(lambda_, epsilon)  # To avoid log(0) issues
    one_minus_lambda = np.maximum(1 - lambda_, epsilon)

    return -np.sum(lambda_ * np.log2(lambda_) +
                   one_minus_lambda * np.log2(one_minus_lambda))

def bayes(lambda_k, y_z):
    """
    Performs a Bayesian update.

    Args:
        lambda_k: Prior probabilities.
        y_z: Likelihoods.

    Returns:
        Posterior probabilities after Bayesian update.
    """
    numerator = ca.times(lambda_k, y_z)
    denominator = numerator + ca.times((1 - lambda_k), (1 - y_z))
    return numerator / denominator

def fake_bayes(lambda_k, y_z):
    """
    Performs a Bayesian update and returns the maximum value between the inputs.

    Args:
        lambda_k: Prior probabilities.
        y_z: Likelihoods.

    Returns:
        The element-wise maximum value between lambda_k and y_z.
    """
    return ca.fmin(ca.fmax(lambda_k, y_z), 1)

def bayes_np(lambda_k, y_z):
    # Calculate the numerator
    numerator = np.multiply(lambda_k, y_z)
    # Calculate the denominator
    denominator = numerator + np.multiply(1 - lambda_k, 1 - y_z)
    # Perform the division
    return np.divide(numerator, denominator)
