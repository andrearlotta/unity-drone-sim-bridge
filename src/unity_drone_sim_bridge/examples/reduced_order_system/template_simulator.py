import numpy as np
from unity_drone_sim_bridge.generic_tools import *
from unity_drone_sim_bridge.surrogate_lib.surrogate_func_tools import bayes_np, entropy_np
from unity_drone_sim_bridge.ros_com_lib.sensors import update_robot_state
import tf2_ros

class Simulator:

    def __init__(self, model, trees, dim_lambda=5, dim_obs=5):
        self.dim_lambda = dim_lambda
        self.dim_obs = dim_obs
        
        self.trees_pos = trees
        self.x_k = {name: 0.5 * np.ones(model.x[name].shape) if 'lambda' in name else np.zeros(model.x[name].shape) for name in model.x.keys()}

        self.x_k_full = {   'lambda': 0.5 * np.ones((len(self.trees_pos), 1)),
                            'lambda_prev': 0.5 * np.ones((len(self.trees_pos), 1)),
                            'y': 0.5 * np.ones((len(self.trees_pos), 1)),}
        
        self.u_k = {"cmd_pose": np.zeros((3, 1)), "viz_pred_pose": None}
        
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.x_k['x_robot']  = update_robot_state(self.tf_buffer)

        # Nearest trees
        self.update_nearest_trees()

    def update_nearest_trees(self):
        """Update nearest tree indices.""" 
        self.reduced_order_lambda_idxs = n_nearest_objects_np(self.x_k['x_robot'][:2], self.trees_pos, num=self.dim_lambda)
        self.residual_lambda_idxs = np.setdiff1d(np.arange(self.trees_pos.shape[0]), self.reduced_order_lambda_idxs)
        self.reduced_order_obs_idxs = n_nearest_objects_np(self.x_k['x_robot'][:2].reshape(1, -1), self.trees_pos, num=self.dim_obs)


    def update(self, y_z):
        """Update the drone state."""
        self.x_k['x_robot']  = update_robot_state(self.tf_buffer)
        self.update_nearest_trees()
       
        self.x_k_full['lambda_prev'] = self.x_k['lambda'].copy()

        scores = np.array(y_z['tree_scores'])
        self.x_k['y'] = scores
        self.x_k_full['lambda'][np.nonzero(scores!=0.5)[0]] = bayes_np(
            self.x_k['lambda_prev'][np.nonzero(scores!=0.5)[0]], 
            self.x_k['y'][np.nonzero(scores!=0.5)[0]]
        )

        self.x_k['lambda_prev'] = self.x_k_full['lambda_prev'][self.reduced_order_lambda_idxs]
        self.x_k['lambda'] = self.x_k_full['lambda'][self.reduced_order_lambda_idxs]
    
    def mpc_get_obs(self):
        return self.trees_pos[self.reduced_order_obs_idxs]

    def mpc_get_lambdas(self):
        return self.trees_pos[self.reduced_order_lambda_idxs]

    def mpc_get_residual_H(self):
        return entropy_np(self.x_k_full['lambda'][self.residual_lambda_idxs])
   