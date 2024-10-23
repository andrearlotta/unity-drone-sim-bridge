import numpy as np
from unity_drone_sim_bridge.surrogate_lib.surrogate_func_tools import bayes_np
from unity_drone_sim_bridge.ros_com_lib.sensors import update_robot_state
import tf2_ros


class Simulator:
    def __init__(self, model, trees):
        self.dim_lambda = trees.shape[0]
        self.trees_pos = trees
        self.x_k = {name: np.zeros(model.x[name].shape) for name in model.x.keys()}
        self.x_k['lambda']= 0.5 * np.ones((len(self.trees_pos)))
        self.x_k['lambda_prev']= 0.5 * np.ones((len(self.trees_pos)))
        self.x_k['y']=  0.5 * np.ones(len(self.trees_pos))
        self.u_k = {"cmd_pose": np.zeros((3, 1)), "viz_pred_pose": None}
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        self.x_k['x_robot']  = update_robot_state(self.tf_buffer)

    def update(self, y_z):
        self.x_k['x_robot']  = update_robot_state(self.tf_buffer)
        
        scores = np.array(y_z['tree_scores'])
        if scores is None: return 
        self.x_k['y'][np.nonzero(scores!=0.5)[0]] = scores[np.nonzero(scores!=0.5)]
        
        self.x_k['lambda_prev'] = self.x_k['lambda'].copy()
        
        # Update lambda values for trees in the reduced order set
        self.x_k['lambda'][np.nonzero(scores!=0.5)[0]] = bayes_np(
            self.x_k['lambda_prev'][np.nonzero(scores!=0.5)[0]], 
            self.x_k['y'][np.nonzero(scores!=0.5)[0]]
        )
    
    def get_mpc_x0(self):
        x_0 = self.x_k.copy()
        return np.concatenate(list(x_0.values()), axis=None)
