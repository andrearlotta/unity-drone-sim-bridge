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


def template_mpc(model,g, get_nn_input, silence_solver = False, rt=False):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_horizon': 20,
        'open_loop': 1,
        't_step': 0.2,
        #'collocation_deg': 10,
        #'collocation_ni': 10,
        'store_full_solution': True,
        # Use MA27 linear solver in ipopt for faster calculations:
        'nlpsol_opts': {
                "ipopt.linear_solver": "ma27",
                #"ipopt.sb": "yes",
                "ipopt.max_iter": 1000,
                "ipopt.acceptable_tol": 1e-3,
                "ipopt.tol": 1e-4,
                "ipopt.print_level": 5,
                "print_time": False,

            }
    }

    mpc.set_param(**setup_mpc)

    if silence_solver:
        mpc.settings.supress_ipopt_output()


    mterm = model.aux['cost_function']
    lterm = model.aux['cost_function'] # stage cost


    mpc.set_objective(mterm=mterm, lterm=lterm)
    #mpc.set_rterm(x_robot_set=np.array(2*[1e-3]+[1e-5]))


    mpc.bounds['lower','_u','x_robot_set'] = 2 * [-.6] + [-np.pi/4]
    mpc.bounds['upper','_u','x_robot_set'] = 2 * [+.6] + [+np.pi/4]
    
    # Avoid the obstacles:
    #mpc.set_nl_cons('obstacles', -model.aux['obstacle_distance'], 0)

    mpc.setup()

    return mpc