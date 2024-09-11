import numpy as np
from unity_drone_sim_bridge.surrogate_lib.gp_tools import load_ca_gp
from unity_drone_sim_bridge.surrogate_lib.nn_tools import LoadNN
from unity_drone_sim_bridge.generic_tools import *
import casadi as ca

'''
casadi map tools
'''
def g_map_cond_casadi(  f,
                        trees_pos,
                        use_yolo =True):
    drone_pos1 = ca.MX.sym('drone_pos', 2)
    drone_yaw1 = ca.MX.sym('drone_yaw')
    tree_pos1  = ca.MX.sym('tree_pos_single',2)

    fov_f = fov_weight_fun_casadi()
    y_single = 0.5 * ca.fabs(ca.cos(drone_yaw1)) + 0.5
    F_single = ca.Function('F_single', [drone_pos1, drone_yaw1, tree_pos1], [y_single])

    #   map the function
    drone_pos_sym = ca.MX.sym('drone_pos', 2)
    drone_yaw_sym = ca.MX.sym('drone_yaw')
    tree_lambda_sym = ca.MX.sym('tree_lambda', (trees_pos.shape))
    F_mapped = F_single.map(trees_pos.shape[0])

    condition = ca.MX.sym('tree_cond', trees_pos.shape[0] )
    # (drone_objects_distances_casadi(drone_pos,drone_yaw,trees_pos) < 8.0)
    # Apply the if-else condition element-wise
    result = ca.if_else(condition, F_mapped(drone_pos_sym, drone_yaw_sym, tree_lambda_sym.T).T, 0.5)

    return F_mapped

def g_map_casadi_rt(l4c_model_order2, params_x_robot_tree_tree_lambda):
        drone_pos1 = ca.MX.sym('drone_pos', 2)
        tree_pos1 = ca.MX.sym('tree_pos_single', 2)
        drone_yaw1 = ca.MX.sym('tree_pos_single', 1)
        params = ca.MX.sym('params', 3)
        phi = ca.atan2(drone_pos1[1]-tree_pos1[1], drone_pos1[0]-tree_pos1[0])  + np.pi/2
        
        # Second-order Taylor (Quadratic) approximation of the model as Casadi Function
        casadi_quad_approx_sym_out = l4c_model_order2(ca.vertcat(ca.power(ca.sum1(ca.power(drone_pos1-tree_pos1, 2)),(1./2)),
                                                                ca.fmod(phi + 2 * np.pi, 2 * np.pi),
                                                                np.mod(phi - drone_yaw1 + np.pi/2 + np.pi, 2 * np.pi)))
        print(l4c_model_order2.get_sym_params().shape)
        f = ca.Function('F_single',
                                            [drone_pos1, drone_yaw1, tree_pos1,
                                            l4c_model_order2.get_sym_params()],
                                            [casadi_quad_approx_sym_out])

        F_mapped = f.map(params_x_robot_tree_tree_lambda.shape[0])
        tree_lambda_sym = ca.MX.sym('tree_lambda', params_x_robot_tree_tree_lambda.shape[0], 2)
        tree_param_lambda_sym = ca.MX.sym('param_nn', params_x_robot_tree_tree_lambda.shape)
        # Use mapped function
        y_all = F_mapped(drone_pos1, drone_yaw1, tree_lambda_sym.T, tree_param_lambda_sym.T).T

        return ca.Function('F_final', [drone_pos1, drone_yaw1, tree_lambda_sym, tree_param_lambda_sym], [y_all])

def g_map_casadi(l4c_model, x_trees):
        drone_state1 = ca.MX.sym('drone_pos', 3)
        tree_pos1 = ca.MX.sym('tree_pos_single', 2)
 
        # Second-order Taylor (Quadratic) approximation of the model as Casadi Function
        casadi_quad_approx_sym_out = l4c_model(ca.vertcat(  drone_state1[:2]-tree_pos1,
                                                            drone_state1[-1]))
        f = ca.Function('F_single',
                        [drone_state1, tree_pos1],
                        [casadi_quad_approx_sym_out])

        F_mapped = f.map(x_trees.shape[0])
        tree_lambda_sym = ca.MX.sym('trees_lambda', x_trees.shape[0], 2)

        # Use mapped function
        y_all = F_mapped(drone_state1, tree_lambda_sym.T).T

        return ca.Function('F_final', [drone_state1, tree_lambda_sym], [y_all])

def setup_g_inline_casadi(f, cond= False):
    drone_pos1 = ca.MX.sym('drone_pos', 2)
    drone_yaw1 = ca.MX.sym('drone_yaw')
    tree_pos1 = ca.MX.sym('tree_pos_single',2)

    # Calculate condition for single tree
    y_single = f(drone_pos1,drone_yaw1, tree_pos1)
    F_single = ca.Function('F_single', [drone_pos1, drone_yaw1, tree_pos1], [y_single])
    
    return F_single

def __g_map_casadi(f, x_trees_dim):

    drone_pos1 = ca.MX.sym('drone_pos', 2)
    drone_yaw1 = ca.MX.sym('drone_yaw')
    tree_pos1  = ca.MX.sym('tree_pos_single',2)

    
    F_mapped = f.map(x_trees_dim[0])
    tree_lambda_sym = ca.MX.sym('tree_lambda', x_trees_dim)
    # Use mapped function
    y_all = F_mapped(drone_pos1, drone_yaw1, tree_lambda_sym.T).T

    return ca.Function('F_final', [drone_pos1, drone_yaw1, tree_lambda_sym], [y_all])

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
setup g functions tools
'''

def load_g(mode='gp', hidden_size=64, hidden_layer=2, synthetic=True, rt=False, gpu=True, use_yolo=True, naive=False):
    if  mode == 'mlp':
        mlp = LoadNN(hidden_size,hidden_layer, synthetic=synthetic, rt=rt, gpu=gpu, use_yolo=use_yolo, naive=naive)
        return mlp
    elif mode == 'gp':
        gp =  load_ca_gp(synthetic=synthetic)
        return lambda drone_pos,drone_yaw, tree_pos_single: g_gp(drone_pos,drone_yaw, tree_pos_single, gp)

def g_gp(drone_pos,drone_yaw, tree_pos_single, gp):
    return  ca.MX(gp.predict(drone_yaw, [], np.zeros((1,1)))[0])

'''
bayes function
'''

def bayes(lambda_k,y_z):
    return ca.times(lambda_k, y_z) / (ca.times(lambda_k, y_z) + (1-lambda_k) * (1-y_z))

def bayes_np(lambda_k, y_z):
    # Calculate the numerator
    numerator = np.multiply(lambda_k, y_z)
    # Calculate the denominator
    denominator = numerator + np.multiply(1 - lambda_k, 1 - y_z)
    # Perform the division
    return np.divide(numerator, denominator)


""""def g_map_casadi(f, x_trees_dim):
        drone_pos1 = ca.MX.sym('drone_pos', 2)
        tree_pos1 = ca.MX.sym('tree_pos_single', 2)
        drone_yaw1 = ca.MX.sym('tree_pos_single', 1)
        phi = ca.atan2(drone_pos1[1]-tree_pos1[1], drone_pos1[0]-tree_pos1[0])  + np.pi
        
        # Calculate condition for single tree
        y_single = 0.5 + 0.25 * 0.25 *( 1 + ca.cos(phi)) * ( 1 + ca.sin(np.mod(phi - drone_yaw1 + np.pi + np.pi/2, 2 * np.pi)- np.pi)) * ( 1 / (1 + ca.power(ca.sum1(ca.power(drone_pos1-tree_pos1, 2)),(1./2))))  # f(ca.vertcat(ca.power(ca.sum1(ca.power(drone_pos1-tree_pos1, 2)),(1./2)),
                                #phi,
                                #np.mod(phi - drone_yaw1 + np.pi + np.pi/2, 2 * np.pi)- np.pi))
        
        
        f = ca.Function('F_single', [drone_pos1, drone_yaw1, tree_pos1], [y_single])

        F_mapped = f.map(x_trees_dim[0])
        tree_lambda_sym = ca.MX.sym('tree_lambda', x_trees_dim)
        # Use mapped function
        y_all = F_mapped(drone_pos1, drone_yaw1, tree_lambda_sym.T).T

        return ca.Function('F_final', [drone_pos1, drone_yaw1, tree_lambda_sym], [y_all])"""