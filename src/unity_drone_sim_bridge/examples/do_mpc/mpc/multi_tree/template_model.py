from casadi import *
from casadi.tools import *
import do_mpc
from unity_drone_sim_bridge.surrogate_lib.surrogate_func_tools import *
from unity_drone_sim_bridge.qi_lib.qi_tools import *
from unity_drone_sim_bridge.surrogate_lib.gp_tools import *

def template_model(trees, symvar_type='MX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete'  # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    ray_obs = 1.0

    # States struct (optimization variables):
    x_robot = model.set_variable(var_type='_x', var_name='x_robot', shape=3)
    lambda_ = model.set_variable(var_type='_x', var_name='lambda', shape=trees.shape[0])
    lambda_prev = model.set_variable(var_type='_x', var_name='lambda_prev', shape=trees.shape[0])
    model.set_variable(var_type='_x', var_name='y', shape=trees.shape[0])

    # Input struct (optimization variables):
    u_x_robot = model.set_variable(var_type='_u', var_name='x_robot_set', shape=3)

    g = fov_weight_fun_casadi(trees)
    
    # Define the expressions using the created functions
    H = entropy(lambda_)
    obstacle_expression =  drone_objects_distances_casadi(x_robot[:2] + u_x_robot[:2],
                                                          trees,
                                                          ray = ray_obs)

    # Set expressions
    model.set_expression('obstacle_distance',obstacle_expression)
    model.set_expression('H', H(lambda_))
    model.set_expression('H_prev', H(lambda_prev))
    model.set_expression('y', g(x_robot,trees))
    model.set_expression('cost_function',  H(lambda_))

    # Set RHS
    model.set_rhs('x_robot', x_robot + u_x_robot)
    model.set_rhs('lambda', bayes(lambda_, model.aux['y']))
    model.set_rhs('lambda_prev', lambda_)
    model.set_rhs('y', model.aux['y'])

    model.setup()

    return model    