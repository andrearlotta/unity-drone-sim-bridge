import numpy as np
from casadi import *
from casadi.tools import *
import do_mpc
from unity_drone_sim_bridge.surrogate_lib.surrogate_func_tools import *
from unity_drone_sim_bridge.qi_lib.qi_tools import *
from unity_drone_sim_bridge.surrogate_lib.gp_tools import *

def template_model(dim_lambda=1, dim_obs=1, symvar_type='MX', g=None, rt=False):
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
    lambda_ = model.set_variable(var_type='_x', var_name='lambda', shape=dim_lambda)
    lambda_prev = model.set_variable(var_type='_x', var_name='lambda_prev', shape=dim_lambda)
    model.set_variable(var_type='_x', var_name='y', shape=dim_lambda)
    
    # Input struct (optimization variables):
    u_x_robot = model.set_variable(var_type='_u', var_name='x_robot_set', shape=3)

    # Time-varying parameters
    x_robot_tree_lambda = np.array([[0.0,0.0]])
    x_robot_tree_obs = np.array([[0.0,0.0]])
    mapped_g = g_map_casadi(g,x_robot_tree_lambda)
    
    # Define the expressions using the created functions
    H = entropy(lambda_)
    H_prev = entropy(lambda_prev)
    y_expr = mapped_g(x_robot,
                      x_robot_tree_lambda)
    
    obstacle_expression =  drone_objects_distances_casadi(x_robot[:2] + u_x_robot[:2],
                                                          x_robot_tree_obs,
                                                          ray = ray_obs)

    # Set expressions
    model.set_expression('obstacle_distance',obstacle_expression)
    model.set_expression('H', H)
    model.set_expression('H_prev', H_prev)
    model.set_expression('y', y_expr)
    model.set_expression('cost_function',  -(H - H_prev))

    # Set RHS
    model.set_rhs('x_robot', x_robot + u_x_robot)
    model.set_rhs('lambda', bayes(lambda_, model.aux['y']))
    model.set_rhs('lambda_prev', lambda_)
    model.set_rhs('y', model.aux['y'])

    model.setup()

    return model    