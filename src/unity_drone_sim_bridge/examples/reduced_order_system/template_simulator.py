import numpy as np
from unity_drone_sim_bridge.ros_com_lib.bridge_class import BridgeClass
from unity_drone_sim_bridge.ros_com_lib.sensors import SENSORS
from unity_drone_sim_bridge.generic_tools import *
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
        self.reduced_order_lambda_idxs = n_nearest_objects(self.x_k['x_robot'][:2], self.trees_pos, num=self.dim_lambda)
        self.residual_lambda_idxs = np.setdiff1d(np.arange(self.trees_pos.shape[0]), self.reduced_order_lambda_idxs)
        self.reduced_order_obs_idxs = n_nearest_objects(self.x_k['x_robot'][:2].reshape(1, -1), self.trees_pos, num=self.dim_obs)

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
                objects_pos=self.trees_pos[self.reduced_order_lambda_idxs][:])
        
        self.x_k_full['lambda_prev'] = self.x_k_full['lambda']
        self.x_k_full['lambda'][self.reduced_order_lambda_idxs] = bayes(self.x_k_full['lambda_prev'][self.reduced_order_lambda_idxs], self.x_k['y'])
        self.x_k['lambda_prev'] = self.x_k['lambda']
        self.x_k['lambda'] = self.x_k_full['lambda'][self.reduced_order_lambda_idxs]
    
    def mpc_get_obs(self):
        return self.trees_pos[self.reduced_order_obs_idxs]

    def mpc_get_lambdas(self):
        return self.trees_pos[self.reduced_order_lambda_idxs]

    def mpc_get_residual_H(self):
        return np.sum(self.x_k_full['lambda'][self.residual_lambda_idxs] * np.log(self.x_k_full['lambda'][self.residual_lambda_idxs]))
    
   