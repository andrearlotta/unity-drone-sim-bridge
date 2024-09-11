from acados_template import AcadosModel
from casadi import MX, vertcat, sin, cos, log, sum1
import numpy as np
from unity_drone_sim_bridge.surrogate_lib.surrogate_func_tools import *

def export_model(g) -> AcadosModel:

    model_name = 'drone_discrete_sys'

    # set up states & controls
    x      = MX.sym('x',3)
    prob      = MX.sym('prob')
    prev_prob  = MX.sym('prev_prob')
    L = MX.sym('L')

    s = vertcat(x, prob, prev_prob, L)

    dx = MX.sym('dx',3)

    # Time-varying parameters
    x_robot_tree_lambda = np.array([[0.0,0.0]])
    mapped_g = g_map_casadi(g,x_robot_tree_lambda)
    fov = fov_weight_fun_casadi()
    f_expl = vertcat(    x + dx,
                        mapped_g(  x[:2]+dx[:2],
                                    x[-1]+dx[-1],
                                    x_robot_tree_lambda),
                        prob,
                        -(sum1(prob * log(prob) + 1e-9) - sum1(prev_prob * log(prev_prob)) + 1e-9)
                        )


    model = AcadosModel()

    model.disc_dyn_expr = f_expl
    model.x = s
    model.u = dx
    model.name = model_name

    # store meta information
    model.x_labels = ['$x$ [m]','$y$ [m]', r'$\theta$ [rad]']#, r'$\lambda_{k}$', r'$\lambda_{k-1}$']
    model.u_labels = [r'$v_x$', r'$v_y$', r'$\dot{\theta}$ [rad/s]']
    model.t_label = '$t$ [s]'

    return model

from acados_template import AcadosModel
import torch

import l4casadi as l4c

from casadi import SX, DM, vertcat, sin, cos, tan, exp, if_else, pi , atan , logic_and , sqrt , fabs , atan2 , MX , DM

    
def robot_model(g):
    model_name = "robot_model"
    # yamle file paramters
    # State
    x = MX.sym('x') 
    y = MX.sym('y')   
    v = MX.sym('v')  
    theta = MX.sym('theta') 

    sym_x = vertcat(x, y, v ,theta)

    # Input
    a = MX.sym('a')
    w = MX.sym('w')
    sym_u = vertcat(a, w)

    # Derivative of the States
    x_dot = MX.sym('x_dot')
    y_dot = MX.sym('y_dot')
    theta_dot = MX.sym('theta_dot')
    v_dot = MX.sym('v_dot')
    x_dot = vertcat(x_dot, y_dot, v_dot, theta_dot)

    ## Model of Robot
    f_expl = vertcat(   sym_x[2] * cos(sym_x[3]),
                        sym_x[2] * sin(sym_x[3]),
                        sym_u[0],
                        sym_u[1])
    f_impl = x_dot - f_expl

    model = AcadosModel()

    print(model)

    x_robot_tree_lambda = np.array([[0.0,0.0]])
    mapped_g = g_map_casadi(g,x_robot_tree_lambda)

    torch.cuda.empty_cache()

    print("l4model " , g)

    model.cost_y_expr = vertcat(sym_x, sym_u , mapped_g( sym_x[:2],
                                    theta,
                                    x_robot_tree_lambda))
    model.cost_y_expr_e = vertcat(sym_x, mapped_g( sym_x[:2],
                                   theta,
                                    x_robot_tree_lambda))
    model.f_impl_expr = f_impl
    model.f_expl_expr = f_expl
    model.x = sym_x
    model.xdot = x_dot
    model.u = sym_u
    model.name = "robot_model"

    return model