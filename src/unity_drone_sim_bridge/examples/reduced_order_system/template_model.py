import numpy as np
from casadi import *
from casadi.tools import *
import do_mpc
from unity_drone_sim_bridge.surrogate_lib.surrogate_func_tools import *
from unity_drone_sim_bridge.qi_lib.qi_tools import *
from unity_drone_sim_bridge.surrogate_lib.gp_tools import *

def template_model(dim_lambda=5, dim_obs=5, symvar_type='MX', g=None):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete'  # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    ray_obs = 1.0
    
    # States struct (optimization variables):
    x_robot = model.set_variable(var_type='_x', var_name='x_robot', shape=(3,1))
    lambda_ = model.set_variable(var_type='_x', var_name='lambda', shape=(dim_lambda,1))
    lambda_prev = model.set_variable(var_type='_x', var_name='lambda_prev', shape=(dim_lambda,1))
    y = model.set_variable(var_type='_x', var_name='y', shape=(dim_lambda,1))
    
    # Input struct (optimization variables):
    u_x_robot = model.set_variable(var_type='_u', var_name='x_robot_set', shape=(3,1))

    # Time-varying parameters
    reduced_order_x_robot_tree_lambda = model.set_variable('_tvp', 'reduced_order_x_robot_tree_lambda', (dim_lambda, 2))
    reduced_order_x_robot_tree_obs = model.set_variable('_tvp', 'reduced_order_x_robot_tree_obs', (dim_obs, 2))
    residual_h = model.set_variable('_tvp', 'reduced_order_h', shape=(1,1))
    residual_h_prev = model.set_variable('_tvp', 'reduced_order_h_prev', shape=(1,1))

    mapped_g = g_map_casadi(g, reduced_order_x_robot_tree_lambda)

    # Define the expressions using the created functions
    H = entropy(lambda_) + residual_h
    H_prev = entropy(lambda_prev) + residual_h_prev
    y_expr =  mapped_g(x_robot + u_x_robot, reduced_order_x_robot_tree_lambda)
    cost_function = -(H - H_prev)
    obstacle_expression =  drone_objects_distances_casadi(  x_robot[:2]+u_x_robot[:2],
                                                            reduced_order_x_robot_tree_obs,
                                                            ray = ray_obs
                                                        )

    # Set expressions
    model.set_expression('obstacle_distance',obstacle_expression)
    model.set_expression('H', H)
    model.set_expression('H_prev', H_prev)
    model.set_expression('y', y_expr)
    model.set_expression('cost_function', cost_function)

    model.set_rhs('x_robot', x_robot + u_x_robot)
    model.set_rhs('lambda', bayes(lambda_, model.aux['y']))
    model.set_rhs('lambda_prev', lambda_)
    model.set_rhs('y', model.aux['y'])

    model.setup()

    return model