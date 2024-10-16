#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
import os
import do_mpc


def template_mpc(model, get_obs, get_lambdas, get_residual_H, silence_solver = False):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 25,
        'n_robust': 0,
        'open_loop': 0,
        't_step': .5,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 2,
        'collocation_ni': 1,
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        'nlpsol_opts': {        "ipopt.linear_solver": "ma27",
        "ipopt.max_iter": 500,
        "ipopt.timing_statistics": 'no',

        "ipopt.sb":"yes",

        "ipopt.nlp_scaling_method":"equilibration-based",
        "ipopt.obj_scaling_factor": +1,
        "ipopt.nlp_scaling_min_value" : 1e-10,
        "ipopt.tol": 1e-2,
        "print_time": False,

        "ipopt.print_info_string": "yes",
        "ipopt.output_file": "nmpc_mul_shooting_mx_l4casadi_output",
        "ipopt.print_level":5,
        "ipopt.hessian_approximation" : 'limited-memory',
        "ipopt.warm_start_init_point" : 'yes',
        "ipopt.mu_strategy" : 'monotone',
        #"ipopt.mu_init" : 1e-4,
        "ipopt.warm_start_bound_push" : 1e-6,
        "ipopt.warm_start_mult_bound_push" : 1e-2,
        "ipopt.line_search_method" : "filter",
        "ipopt.alpha_for_y":"primal-and-full",
        #"ipopt.accept_every_trial_step": "yes"
    }

    }

    mpc.set_param(**setup_mpc)

    if silence_solver:
        mpc.settings.supress_ipopt_output()


    mterm = model.aux['cost_function']
    lterm = model.aux['cost_function'] # stage cost


    mpc.set_objective(mterm=mterm, lterm=lterm)


    mpc.bounds['lower','_u','x_robot_set'] = 2 * [-.5] + [-np.pi/18]
    mpc.bounds['upper','_u','x_robot_set'] = 2 * [+.5] + [+np.pi/18]
    
    # Avoid the obstacles:
    #mpc.set_nl_cons('obstacles', -model.aux['obstacle_distance'], 0)

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(time_x):
        for k in range(mpc.settings.n_horizon):
            tvp_template['_tvp', k, 'reduced_order_x_robot_tree_obs'] = get_obs()
            tvp_template['_tvp', k, 'reduced_order_x_robot_tree_lambda'] = get_lambdas()
            tvp_template['_tvp', k, 'reduced_order_h_prev'] = get_residual_H()
            tvp_template['_tvp', k, 'reduced_order_h'] = get_residual_H()
        return tvp_template
    mpc.set_tvp_fun(tvp_fun)

    mpc.setup()

    return mpc