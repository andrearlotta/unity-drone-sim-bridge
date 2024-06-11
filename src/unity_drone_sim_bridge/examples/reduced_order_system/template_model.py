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
    Xrobot = model.set_variable(var_type='_x', var_name='Xrobot', shape=(3,1))
    lambda_ = model.set_variable(var_type='_x', var_name='lambda', shape=(dim_lambda,1))
    lambda_prev = model.set_variable(var_type='_x', var_name='lambda_prev', shape=(dim_lambda,1))
    y = model.set_variable(var_type='_x', var_name='y', shape=(dim_lambda,1))
    
    # Input struct (optimization variables):
    U_Xrobot = model.set_variable(var_type='_u', var_name='Xrobot_set', shape=(3,1))

    # Time-varying parameters
    ReducedOrderXrobot_tree_lambda = model.set_variable('_tvp', 'ReducedOrderXrobot_tree_lambda', (dim_lambda, 2))
    ReducedOrderXrobot_tree_obs = model.set_variable('_tvp', 'ReducedOrderXrobot_tree_obs', (dim_obs, 2))
    ResidualH = model.set_variable('_tvp', 'ReducedOrderH', shape=(1,1))
    ResidualH_prev = model.set_variable('_tvp', 'ReducedOrderH_prev', shape=(1,1))

    mapped_g = setup_g_inline_casadi(g)

    # Expressions
    def compute_H(lambda_, ResidualH):
        return sum1(lambda_ * log(lambda_)) + ResidualH

    def compute_H_prev(lambda_prev, ResidualH_prev):
        return sum1(lambda_prev * log(lambda_prev)) + ResidualH_prev

    def compute_y(Xrobot,ReducedOrderXrobot_tree_lambda):
        return g_map_casadi(mapped_g, ReducedOrderXrobot_tree_lambda.shape)(
            Xrobot[:2],
            Xrobot[-1],
            ReducedOrderXrobot_tree_lambda
        )

    def compute_cost_function(H, H_prev):
        return -(H - H_prev)

    # Define the expressions using the created functions
    H = compute_H(lambda_, ResidualH)
    H_prev = compute_H_prev(lambda_prev, ResidualH_prev)
    y_expr = compute_y(Xrobot + U_Xrobot, ReducedOrderXrobot_tree_lambda)
    cost_function = compute_cost_function(H, H_prev)
    obstacle_expression =  drone_objects_distances_casadi(  Xrobot[:2]+U_Xrobot[:2],
                                                            ReducedOrderXrobot_tree_obs,
                                                            ray = ray_obs
                                                        )

    # Set expressions
    model.set_expression('obstacle_distance',obstacle_expression)
    model.set_expression('H', H)
    model.set_expression('H_prev', H_prev)
    model.set_expression('y', y_expr)
    model.set_expression('cost_function', cost_function)

    model.set_rhs('Xrobot', Xrobot + U_Xrobot)
    model.set_rhs('lambda', bayes(lambda_, model.aux['y']))
    model.set_rhs('lambda_prev', lambda_)
    model.set_rhs('y', model.aux['y'])

    model.setup()

    return model