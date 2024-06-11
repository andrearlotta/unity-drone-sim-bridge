import numpy as np
from casadi import *
from casadi.tools import *
import do_mpc
from unity_drone_sim_bridge.g_func_lib.g_func_tools import *
from unity_drone_sim_bridge.qi_lib.qi_tools import *
from unity_drone_sim_bridge.g_func_lib.gp_tools import *

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
    u_x_robot = model.set_variable(var_type='_u', var_name='u_x_robot', shape=(3,1))

    # Time-varying parameters
    nearby_trees_lambda = model.set_variable('_tvp', 'nearby_trees_lambda', (dim_lambda, 2))
    nearby_trees_obs = model.set_variable('_tvp', 'nearby_trees_obs', (dim_obs, 2))
    residual_h = model.set_variable('_tvp', 'residual_h', shape=(1,1))
    residual_h_prev = model.set_variable('_tvp', 'residual_h_prev', shape=(1,1))
    conditional_y = model.set_variable('_tvp', 'conditional_y', shape=(1,1))

    mapped_g = setup_g_inline_casadi(g)

    # Expressions
    def compute_H(lambda_, residual_h):
        return sum1(lambda_ * log(lambda_)) + residual_h

    def compute_H_prev(lambda_prev, residual_h_prev):
        return sum1(lambda_prev * log(lambda_prev)) + residual_h_prev

    def compute_y(x_robot,nearby_trees_lambda):
        return g_map_casadi_fixed_cond(mapped_g, nearby_trees_lambda.shape)(
            x_robot[:2],
            x_robot[-1],
            nearby_trees_lambda,
            conditional_y
        )

    def compute_cost_function(H, H_prev):
        return -(H - H_prev)

    # Define the expressions using the created functions
    H = compute_H(lambda_, residual_h)
    H_prev = compute_H_prev(lambda_prev, residual_h_prev)
    y_expr = compute_y(x_robot + u_x_robot, nearby_trees_lambda)
    cost_function = compute_cost_function(H, H_prev)
    obstacle_expression =  drone_objects_distances_casadi(  x_robot[:2]+u_x_robot[:2],
                                                            nearby_trees_obs,
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