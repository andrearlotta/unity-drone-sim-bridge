# -*- coding: future_fstrings -*-
#
# Copyright (c) The acados authors.
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosSimSolver

import numpy as np
import casadi as cs
from model import export_model
from unity_drone_sim_bridge.surrogate_lib.surrogate_func_tools import *

def setup(x0, Fmax, N_horizon, Tf):
    # create ocp object to formulate the OCP
    ocp = AcadosOcp()

    # Load the surrogate model
    g = load_g('mlp', 
               rt=False, 
               gpu=False, 
               synthetic=True,
               hidden_layer=2,
               hidden_size=64, 
               use_yolo=True)

    model = export_model(g)
    nx = model.x.rows()
    nu = model.u.rows()
    
    # set model
    ocp.model = model

    # Define the entropy function
    def entropy(p):
        return -p * cs.log(p)

    cost_expr = 10*cs.sum1(cs.norm_1(model.x))

    # cost matrices
    Q_mat = np.diag([.0, .0, 0.0, 0.0, 0.0, 1.0])
    R_mat =np.diag([ 1e-4,  1e-4, 1e-4])

    # path cost
    ocp.cost.cost_type = 'NONLINEAR_LS'
    ocp.cost.yref = np.zeros((nx+nu,))
    ocp.cost.yref[3] = 1.0
    ocp.model.cost_y_expr = ca.vertcat(model.x, model.u)
    ocp.cost.W = ca.diagcat(Q_mat, R_mat).full()

    # terminal cost
    ocp.cost.cost_type_e = 'NONLINEAR_LS'
    ocp.cost.yref_e = np.zeros((nx,))
    ocp.cost.yref_e[3] = 1.0
    ocp.model.cost_y_expr_e = model.x
    ocp.cost.W_e = Q_mat


    # Set dimensions
    ocp.dims.N = N_horizon  # Use the provided horizon

    # Set initial constraints (initial state)
    ocp.constraints.x0 = x0

    # Set control input constraints
    ocp.constraints.lbu = np.array(2 * [-Fmax] + [-np.pi])
    ocp.constraints.ubu = np.array(2 * [Fmax] + [np.pi])
    ocp.constraints.idxbu = np.array([0, 1, 2])

    # Set prediction horizon
    ocp.solver_options.tf = Tf
    ocp.solver_options.integrator_type = 'DISCRETE'
    ocp.solver_options.qp_solver_cond_N = N_horizon
    ocp.solver_options.qp_solver =  'PARTIAL_CONDENSING_HPIPM' 
    ocp.solver_options.qp_solver_cond_N = 10
    ocp.solver_options.nlp_solver_type = 'SQP'
    ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
    ocp.solver_options.levenberg_marquardt = 3.0
    ocp.solver_options.nlp_solver_max_iter = 15
    ocp.solver_options.qp_solver_iter_max = 100
    ocp.solver_options.nlp_solver_tol_stat = 1e2
    ocp.solver_options.nlp_solver_tol_eq = 1e-1
    # Link to external shared library for the model
    ocp.solver_options.model_external_shared_lib_dir = g.shared_lib_dir
    ocp.solver_options.model_external_shared_lib_name = g.name

    # Generate the solver
    solver_json = f'acados_ocp_{model.name}.json'
    acados_ocp_solver = AcadosOcpSolver(ocp, json_file=solver_json)

    return acados_ocp_solver


def main():

    x0 = np.array([0.5, -8.00, -np.pi/4, 0.5, 0.5, 0.0])  # Initial state
    Fmax = 0.5
    Tf = 2.0
    N_horizon = 10

    ocp_solver = setup(x0, Fmax, N_horizon, Tf)

    nx = ocp_solver.acados_ocp.dims.nx
    nu = ocp_solver.acados_ocp.dims.nu

    Nsim = 100
    simX = np.zeros((Nsim + 1, nx))
    simU = np.zeros((Nsim, nu))
    simX[0, :] = x0
    
    t = np.zeros(Nsim)

    # Perform initial iterations to get a good initial guess
    num_iter_initial = 5
    for _ in range(num_iter_initial):
        ocp_solver.solve_for_x0(x0_bar=x0)

    # Closed-loop simulation
    for i in range(Nsim):
        # Solve the OCP and get the next control input
        simU[i, :] = ocp_solver.solve_for_x0(x0_bar=simX[i, :])

        # Record computation time
        t[i] = ocp_solver.get_stats('time_tot')
        print(simX[i, :])
        print(simU[i, :])
        # Simulate the system (discrete update)
        simX[i + 1, :3] = simX[i, :3] + simU[i, :] #* Tf/N_horizon
        simX[i + 1, 4] = simX[i, 3]
        simX[i + 1, 3] = 0.5 + fov_weight_fun_polar_numpy( simX[i+1, 2], simX[i+1, :2], thresh_distance=6)*0.5
        epsilon = 1e-9  # A small value to avoid log(0) or log of a negative number
        simX[i + 1, -1] = -(np.sum(simX[i + 1, 3] * np.log(simX[i + 1, 3] + epsilon)) -
                            np.sum(simX[i + 1, 4] * np.log(simX[i + 1, 4] + epsilon)))
        print('----')
        
    # Scale computation time to milliseconds
    t *= 1000
    print(f'Computation time in ms: min {np.min(t):.3f} median {np.median(t):.3f} max {np.max(t):.3f}')

    # Clean up solver
    ocp_solver = None


if __name__ == '__main__':
    main()
