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


def template_mpc(model, get_obs, get_lambdas, get_residual_H, get_cond_y, silence_solver = False):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 10,
        'n_robust': 0,
        'open_loop': 0,
        't_step': 1.0,
        'state_discretization': 'collocation',
        'collocation_type': 'radau',
        'collocation_deg': 2,
        'collocation_ni': 1,
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        'nlpsol_opts': {'ipopt.linear_solver': 'MA27', 'ipopt.warm_start_init_point' : 'yes'}
    }

    mpc.set_param(**setup_mpc)

    if silence_solver:
        mpc.settings.supress_ipopt_output()


    mterm = model.aux['cost_function']
    lterm = model.aux['cost_function'] # stage cost


    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u_x_robot=np.array(3*[1e-2]))


    mpc.bounds['lower','_u','u_x_robot'] = 2 * [-.5] + [-np.pi/40]
    mpc.bounds['upper','_u','u_x_robot'] = 2 * [+.5] + [+np.pi/40]
    
    # Avoid the obstacles:
    mpc.set_nl_cons('obstacles', -model.aux['obstacle_distance'], 0)

    tvp_template = mpc.get_tvp_template()

    def tvp_fun(time_x):
        gen_cond = get_cond_y()
        gen_obs = get_obs()
        gen_lambdas = get_lambdas()
        gen_residual_H = get_residual_H()
        gen_residual_H = get_residual_H()
        print(np.count_nonzero(gen_cond))
        for k in range(mpc.settings.n_horizon):
            tvp_template['_tvp', k,'conditional_y'] = gen_cond
            tvp_template['_tvp', k, 'nearby_trees_obs'] = gen_obs
            tvp_template['_tvp', k, 'nearby_trees_lambda'] = gen_lambdas
            tvp_template['_tvp', k, 'residual_h_prev'] = gen_residual_H
            tvp_template['_tvp', k, 'residual_h'] = gen_residual_H
        return tvp_template
    mpc.set_tvp_fun(tvp_fun)

    mpc.setup()

    return mpc