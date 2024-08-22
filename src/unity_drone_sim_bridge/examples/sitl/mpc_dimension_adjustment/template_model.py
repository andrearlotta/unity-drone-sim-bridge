import numpy as np
from casadi import *
from casadi.tools import *
import do_mpc
from unity_drone_sim_bridge.g_func_lib.g_func_tools import *
from unity_drone_sim_bridge.qi_lib.qi_tools import *
from unity_drone_sim_bridge.g_func_lib.gp_tools import *

def template_model(dim_lambda=5, dim_obs=5, symvar_type='MX', g=None, cond=False):
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
    y = model.set_variable(var_type='_x', var_name='y', shape=dim_lambda)
    
    # Input struct (optimization variables):
    u_x_robot = model.set_variable(var_type='_u', var_name='u_x_robot', shape=3)

    # Time-varying parameters
    residual_h = model.set_variable('_tvp', 'residual_h', shape=1)
    residual_h_prev = model.set_variable('_tvp', 'residual_h_prev', shape=1)
    nearby_trees_lambda_x = model.set_variable('_tvp', 'nearby_trees_lambda_x',  shape=dim_lambda)
    nearby_trees_lambda_y = model.set_variable('_tvp', 'nearby_trees_lambda_y',  shape=dim_lambda)
    nearby_trees_obs_x = model.set_variable('_tvp', 'nearby_trees_obs_x',  shape=dim_obs)
    nearby_trees_obs_y = model.set_variable('_tvp', 'nearby_trees_obs_y',  shape=dim_obs)
    

    # Expressions
    def compute_H(lambda_, residual_h):
        return sum1(lambda_ * log(lambda_)) + residual_h

    def compute_H_prev(lambda_prev, residual_h_prev):
        return sum1(lambda_prev * log(lambda_prev)) + residual_h_prev

    map_g = g_map_casadi(g, hcat([nearby_trees_lambda_x,nearby_trees_lambda_y]).shape)


    def compute_cost_function(H, H_prev):
        return -(H - H_prev)

    # Define the expressions using the created functions
    H = compute_H(lambda_, residual_h)
    H_prev = compute_H_prev(lambda_prev, residual_h_prev)
    y_expr = map_g(x_robot[:2]+u_x_robot[:2],
                   x_robot[-1]+u_x_robot[-1],
                   hcat([nearby_trees_lambda_x, nearby_trees_lambda_y]))
    cost_function = compute_cost_function(H, H_prev)
    obstacle_expression =  drone_objects_distances_casadi(  x_robot[:2]+u_x_robot[:2],
                                                            hcat([nearby_trees_obs_x, nearby_trees_obs_y]),
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


"""  
    File "/home/pantheon/mpc-drone/ros/venv/lib/python3.9/site-packages/do_mpc/model/_model.py", line 580, in set_variable
    assert self.flags['setup'] == False, 'Cannot call .set_variable after setup.'
    AssertionError: Cannot call .set_variable after setup.
"""
def adjust_dimension(model, g, dim_lambda):
    lambda_ = model.set_variable(var_type='_x', var_name='lambda', shape=(dim_lambda,1))
    lambda_prev = model.set_variable(var_type='_x', var_name='lambda_prev', shape=(dim_lambda,1))
    y = model.set_variable(var_type='_x', var_name='y', shape=(dim_lambda,1))
    
    mapped_g = setup_g_inline_casadi(g)
    
    def compute_y(x_robot,nearby_trees_lambda):
        return g_map_casadi(mapped_g, nearby_trees_lambda.shape)(
            x_robot[:2],
            x_robot[-1],
            nearby_trees_lambda
        )
    
    y_expr = compute_y(model.x['x_robot'] + model.x['u_x_robot'], model.tvp['nearby_trees_lambda'])
    model.set_expression('y', y_expr)
    model.set_rhs('lambda', bayes(lambda_, model.aux['y']))
    model.set_rhs('lambda_prev', lambda_)
    model.set_rhs('y', model.aux['y'])