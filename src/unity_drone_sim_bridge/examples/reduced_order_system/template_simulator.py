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
from unity_drone_sim_bridge.ros_com_lib.bridge_class import BridgeClass
from unity_drone_sim_bridge.sensors import SENSORS
from unity_drone_sim_bridge.g_func_lib.generic_tools import *
from unity_drone_sim_bridge.g_func_lib.g_func_tools import bayes

class Simulator:
    """
    --------------------------------------------------------------------------
    template_simulator: state log & Unity communication
    --------------------------------------------------------------------------
    """

    def __init__(self, model, dim_lambda=5, dim_obs=5):
        self.dim_lambda = dim_lambda
        self.dim_obs = dim_obs
        
        self.bridge = BridgeClass(SENSORS)

        self.trees_pos =  np.array(self.bridge.callServer({"trees_poses": None})["trees_poses"])

        # Populate the self.x_k dictionary
        self.x_k = {name: 0.5 * np.ones(model.x[name].shape) if 'lambda' in name else np.zeros(model.x[name].shape) for name in model.x.keys()}

        self.x_k_full = {'lambda': 0.5 * np.ones((len(self.trees_pos), 1)),
                        'lambda_prev': 0.5 * np.ones((len(self.trees_pos), 1))}
        
        self.u_k = {"cmd_pose": np.zeros((3, 1)), "viz_pred_pose": None}

        # Nearest trees
        self.update_nearest_trees()

    def update_nearest_trees(self):
        """Update nearest tree indices."""
        self.ReducedOrderIdxsLambda = n_nearest_objects(self.x_k['x_robot'][:2].reshape(1, -1), self.trees_pos, num=self.dim_lambda)
        self.ResidualIdxsLambda = np.setdiff1d(np.arange(self.trees_pos.shape[0]), self.ReducedOrderIdxsLambda)
        self.ReducedOrderIdxsTree = n_nearest_objects(self.x_k['x_robot'][:2].reshape(1, -1), self.trees_pos, num=self.dim_obs)

    def update(self, y_z):
        """Update the drone state."""
        self.update_nearest_trees()
        
        self.x_k['x_robot'] = np.array([y_z['gps'][0], y_z['gps'][1], y_z['gps'][-1]])

        if 'detection' in y_z and y_z['detection']:
            self.x_k['y'][self.data_association(self, y_z['gps'], y_z['detection'])] = [i['score'] for i in y_z['gps']['detection']]
        else:
            qi_z = ((1 + np.cos(y_z['gps'][-1])) / 15 + 0.5) # np.array([fake_nn(img_2_qi(y_z['image']))]) # 
            self.x_k['y'] = 0.5 + (qi_z - 0.5) * fov_weight_fun_numpy(
                drone_pos=y_z['gps'][:2], 
                drone_yaw=y_z['gps'][-1], 
                objects_pos=self.trees_pos[self.ReducedOrderIdxsLambda][:])
        
        self.x_k_full['lambda_prev'] = self.x_k_full['lambda']
        self.x_k_full['lambda'][self.ReducedOrderIdxsLambda] = bayes(self.x_k_full['lambda_prev'][self.ReducedOrderIdxsLambda], self.x_k['y'])
        self.x_k['lambda_prev'] = self.x_k['lambda']
        self.x_k['lambda'] = self.x_k_full['lambda'][self.ReducedOrderIdxsLambda]
    
    def mpc_get_obs(self):
        return self.trees_pos[self.ReducedOrderIdxsTree]

    def mpc_get_lambdas(self):
        return self.trees_pos[self.ReducedOrderIdxsLambda]

    def mpc_get_residual_H(self):
        return np.sum(self.x_k_full['lambda'][self.ResidualIdxsLambda] * np.log(self.x_k_full['lambda'][self.ResidualIdxsLambda]))