import numpy as np
from unity_drone_sim_bridge.ros_com_lib.bridge_class import BridgeClass
from unity_drone_sim_bridge.ros_com_lib.sensors import SENSORS
from unity_drone_sim_bridge.generic_tools import *
from unity_drone_sim_bridge.surrogate_lib.surrogate_func_tools import bayes_np

import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped
from tf.transformations import euler_from_quaternion

class Simulator:
    def __init__(self, states, trees, dim_lambda=1, dim_obs=1):
        self.dim_lambda = dim_lambda
        self.dim_obs = dim_obs
        
        self.trees_pos = trees
        self.x_k= {'x_robot': np.zeros(states.shape)}
        self.x_k['lambda']= 0.5 * np.ones((len(self.trees_pos)))
        self.x_k['lambda_prev']= 0.5 * np.ones((len(self.trees_pos)))
        self.x_k['y']=  0.5 * np.ones(len(self.trees_pos))
        self.u_k = {"cmd_pose": np.zeros((3, 1)), "viz_pred_pose": None}
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        self.update_robot_state()

    def update_robot_state(self):
        robot_pose = []
        while not len(robot_pose):
            try:
                # Get the transform from 'map' to 'drone_base_link'
                trans = self.tf_buffer.lookup_transform('map', 'drone_base_link', rospy.Time())

                # Extract rotation
                (_, _, yaw) = euler_from_quaternion([ trans.transform.rotation.x,  trans.transform.rotation.y,  trans.transform.rotation.z,  trans.transform.rotation.w])
                
                # Update x_robot with the transform (x, y, yaw)
                robot_pose = np.array([trans.transform.translation.x, trans.transform.translation.y, yaw])
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logwarn(f"Failed to get transform: {e}")
            
        self.x_k['x_robot']  = robot_pose

    def update(self, y_z):
        self.update_robot_state()
        scores = np.array(y_z['tree_scores'])

        print(self.x_k['y'][np.nonzero(scores!=0.5)[0]])
        self.x_k['y'][np.nonzero(scores!=0.5)[0]] = scores[np.nonzero(scores!=0.5)]
        
        self.x_k['lambda_prev'] = self.x_k['lambda'].copy()
        
        # Update lambda values for trees in the reduced order set
        self.x_k['lambda'][np.nonzero(scores!=0.5)[0]] = bayes_np(
            self.x_k['lambda_prev'][np.nonzero(scores!=0.5)[0]], 
            self.x_k['y'][np.nonzero(scores!=0.5)[0]]
        )
    
    def get_mpc_x0(self):
        x_0 = self.x_k.copy()
        x_0['lambda_prev']  =   self.x_k['lambda_prev']
        x_0['lambda']       =   self.x_k['lambda']
        x_0['y']            =   self.x_k['y']
        return np.concatenate(list(x_0.values()), axis=None)
    
    def mpc_nn_inputs(self):
        tree_pos = self.x_k['x_robot'][:2].reshape((2,))
        drone_pos = self.trees_pos
        drone_yaw = self.x_k['x_robot'][-1]
        
        # Vector difference between drone and each tree
        diff = drone_pos - tree_pos

        # Calculate angle phi for each tree, and adjust by pi/2
        phi = np.arctan2(diff[:, 1], diff[:, 0]) + np.pi/2

        # Calculate Euclidean distance from drone to each tree
        distances = np.linalg.norm(diff, axis=1)

        # Normalize phi to be in the range [0, 2*pi]
        phi_normalized = np.mod(phi + 2 * np.pi, 2 * np.pi)

        # Compute the relative angle, adjust by pi/2 and normalize
        relative_angles = np.mod(phi - drone_yaw + np.pi + np.pi/2, 2 * np.pi)

        # Combine results into a single output array (if needed for further processing)
        result = np.vstack((distances, phi_normalized, relative_angles)).T

        return result
