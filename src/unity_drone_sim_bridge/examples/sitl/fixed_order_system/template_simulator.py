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
from unity_drone_sim_bridge.generic_tools import *
from unity_drone_sim_bridge.g_func_lib.g_func_tools import bayes, bayes_np
from unity_drone_sim_bridge.examples.sitl.setup_fake_field import *

class MockBridgeClass:
    def __init__(self, grid_size, spacing, min_distance):
        self.grid_size = grid_size
        self.spacing = spacing
        self.service = {
            "trees_poses": generate_tree_positions(self.grid_size, self.spacing)
            }
        self.data = {
            "gps": set_drone_position(self.service["trees_poses"], min_distance)
        }

    def callServer(self, request):
        if "trees_poses" in request:
            return self.service

    def pubData(self, u_k):
        print( u_k["cmd_pose"])
        print(self.data["gps"])
        self.data["gps"] += u_k["cmd_pose"]
        #print(f"Published data: {u_k}")

    def getData(self):
        return self.data


class Simulator:
    """
    --------------------------------------------------------------------------
    template_simulator: state log & Unity communication
    --------------------------------------------------------------------------
    """

    def __init__(self, model, dim_lambda=5, dim_obs=5, grid_size=(100,100), spacing=10, min_distance=2, fov_w=True):
        self.dim_lambda = dim_lambda
        self.dim_obs = dim_obs
        
        self.bridge = MockBridgeClass(grid_size, spacing, min_distance)

        self.trees_pos = self.bridge.callServer({"trees_poses": None})["trees_poses"]

        # Populate the self.x_k dictionary
        self.x_k = {name: 0.5 * np.ones(model.x[name].shape) if 'lambda' in name else np.zeros(model.x[name].shape) for name in model.x.keys()}

        self.x_k_full = {'lambda': 0.5 * np.ones((len(self.trees_pos))),
                        'lambda_prev': 0.5 * np.ones((len(self.trees_pos)))}
        
        self.u_k = {"cmd_pose": np.zeros((3,1)), "viz_pred_pose": None}

        # Nearest trees
        self.update_nearest_trees()

    def adjust_lambda(self, dim_lambda):
        self.dim_lambda = dim_lambda
        self.update_nearest_trees()

    def update_nearest_trees(self):
        """Update nearest tree indices."""
        self.reduced_order_lambda_idxs = n_nearest_objects(self.x_k['x_robot'][:2], self.trees_pos, num=self.dim_lambda)
        self.residual_lambda_idxs = np.setdiff1d(np.arange(self.trees_pos.shape[0]), self.reduced_order_lambda_idxs)
        self.reduced_order_obs_idxs = n_nearest_objects(self.x_k['x_robot'][:2], self.trees_pos, num=self.dim_obs)

    def nearest_n_elements(self, num):
        return n_nearest_objects(self.x_k['x_robot'][:2], self.trees_pos, num=num)

    def update(self, y_z):
        """Update the drone state."""
        self.update_nearest_trees()
        
        self.x_k['x_robot'] = np.array([y_z['gps'][0], y_z['gps'][1], y_z['gps'][-1]])

        if 'detection' in y_z and y_z['detection']:
            self.x_k['y'][self.data_association(self, y_z['gps'], ['gps']['detection'])] = [i['score'] for i in y_z['gps']['detection']]
        else:
            qi_z = ((1 + np.cos(y_z['gps'][-1])) / 15 + 0.5) # np.array([fake_nn(img_2_qi(y_z['image']))]) # 
            self.x_k['y'] =  0.5 + (qi_z - 0.5) * fov_weight_fun_numpy(
                drone_pos=y_z['gps'][:2], 
                drone_yaw=y_z['gps'][-1], 
                objects_pos=self.trees_pos[self.reduced_order_lambda_idxs][:])
            
        self.x_k_full['lambda_prev'] = self.x_k_full['lambda']
        self.x_k_full['lambda'][self.reduced_order_lambda_idxs] = bayes_np(self.x_k_full['lambda_prev'][self.reduced_order_lambda_idxs], self.x_k['y'])
        self.x_k['lambda_prev'] = self.x_k['lambda']
        self.x_k['lambda'] = self.x_k_full['lambda'][self.reduced_order_lambda_idxs]
    
    def mpc_get_obs(self):
        return self.trees_pos[self.reduced_order_obs_idxs]

    def mpc_get_lambdas(self):
        return self.trees_pos[self.reduced_order_lambda_idxs]

    def mpc_get_residual_H(self):
        return np.sum(self.x_k_full['lambda'][self.residual_lambda_idxs] * np.log(self.x_k_full['lambda'][self.residual_lambda_idxs]))