import numpy as np
from unity_drone_sim_bridge.g_func_lib.gp_tools import LoadCaGP
from unity_drone_sim_bridge.g_func_lib.nn_tools import LoadNN
from unity_drone_sim_bridge.g_func_lib.generic_tools import *
import casadi as ca

'''
casadi map tools
'''

def setup_g_inline_casadi(f, cond= False):
    drone_pos = ca.MX.sym('drone_pos', 2)
    drone_yaw = ca.MX.sym('drone_yaw')
    tree_pos_single = ca.MX.sym('tree_pos_single',2)

    # Calculate condition for single tree
    condition_single = fov_weight_fun_casadi(drone_pos, drone_yaw, ca.reshape(tree_pos_single, (1, 2)))
    # Apply the condition to single inputs
    y_single = condition_single * (f(drone_pos,drone_yaw, tree_pos_single)  - 0.5) + 0.5 
    cond_y_single = ca.if_else(condition_single, y_single, 0.5)

    # Create CasADi function for single evaluation
    return ca.Function('F_single', [drone_pos, drone_yaw, tree_pos_single], [cond_y_single if cond else y_single])

def g_map_casadi(F_single, Xtrees_dim):
    F_mapped = F_single.map(Xtrees_dim[0]) 
    drone_pos_sym = ca.MX.sym('drone_pos', 2)
    drone_yaw_sym = ca.MX.sym('drone_yaw')
    tree_lambda_sym = ca.MX.sym('tree_lambda', Xtrees_dim)
    # Use mapped function
    y_all = F_mapped(drone_pos_sym, drone_yaw_sym, tree_lambda_sym.T).T

    return ca.Function('F_final', [drone_pos_sym, drone_yaw_sym, tree_lambda_sym], [y_all])

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
    y_single = condition_single * (f(drone_pos,drone_yaw, tree_pos_single)  - 0.5) + 0.5 

    cond_y_single = ca.if_else(var_cond_single, y_single, 0.5)

    # Create CasADi function for single evaluation
    return ca.Function('F_single', [drone_pos, drone_yaw, tree_pos_single, var_cond_single], [y_single])

def g_map_casadi_fixed_cond(F_single, Xtrees_dim, cond_=None):
    F_mapped = F_single.map(Xtrees_dim[0]) 
    drone_pos_sym = ca.MX.sym('drone_pos', 2)
    drone_yaw_sym = ca.MX.sym('drone_yaw')
    tree_lambda_sym = ca.MX.sym('tree_lambda', Xtrees_dim)
    cond_sym = ca.MX.sym('tree_cond', Xtrees_dim[0])
    # Use mapped function
    y_all = F_mapped(drone_pos_sym, drone_yaw_sym, tree_lambda_sym.T, cond_sym).T

    return ca.Function('F_final', [drone_pos_sym, drone_yaw_sym, tree_lambda_sym, cond_sym], [y_all])

'''
setup g functions tools
'''

def load_g(mode='gp', hidden_size=64, hidden_layer=5):
    if  mode == 'mlp':
        mlp = LoadNN(hidden_size,hidden_layer, synthetic=False)
        return lambda drone_pos,drone_yaw, tree_pos_single,: g_nn(drone_pos,drone_yaw, tree_pos_single, mlp)
    elif mode == 'gp':
        gp =  LoadCaGP(synthetic=True)
        return lambda drone_pos,drone_yaw, tree_pos_single: g_gp(drone_pos,drone_yaw, tree_pos_single, gp)

def g_gp(drone_pos,drone_yaw, tree_pos_single, gp):
    return  ca.MX(gp.predict(drone_yaw, [], np.zeros((1,1)))[0]) #((1 + ca.cos(drone_yaw_sym))/ 15 + 0.5)  # np.ones(drone_yaw_sym.shape) * ca.cos(drone_yaw_sym) + 1

def g_nn(drone_pos,drone_yaw, tree_pos_single, nn):
    return nn(drone_yaw)

'''
bayes function
'''

def bayes(lambda_k,y_z):
    return ca.times(lambda_k, y_z) / (ca.times(lambda_k, y_z) + (1-lambda_k) * (1-y_z))
