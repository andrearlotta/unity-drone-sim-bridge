from unity_drone_sim_bridge.nn_lib.NeuralClass import NeuralClass
from unity_drone_sim_bridge.BridgeClass import BridgeClass
from unity_drone_sim_bridge.sensors import SENSORS
from unity_drone_sim_bridge.MpcClass import MpcClass
from unity_drone_sim_bridge.StateClass import StateClass
from unity_drone_sim_bridge.nn_lib.nn_tools import *
from unity_drone_sim_bridge.qi_lib.qi_tools import *
from dataclasses import dataclass, field
from typing import List, Dict, Union, Any
import do_mpc
import casadi as ca
import rospy
from std_msgs.msg import Empty
import numpy as np
@dataclass
class DroneState(StateClass):
    trees_pos:   np.ndarray = field(default_factory=lambda: np.zeros(10))
    gp: Any = field(init=False)
    def __post_init__(self):
        super().__post_init__()
        '''
        overwritten elemets
        '''
        self.gp = loadGp()

        self.param = {
            'tree_x': self.trees_pos[:,0],
            'tree_y': self.trees_pos[:,1],
        }
        self.state_dict= {
        '_x':   {'pos_x':len(self.trees_pos),'pos_y':len(self.trees_pos), 'yaw':1, 'lambda':len(self.trees_pos)},
        '_u':   {'pos_x':1, 'pos_y':1, 'yaw':1},
        '_tvp': {'hat_lambda_z':len(self.trees_pos)}, #g(X)
        }

        self.rsh_dict = {
            'pos_x'     : (lambda mdl: mdl.x[f'pos_x'] + np.ones((len(self.trees_pos),1))@mdl.u[f'pos_x'] - self.param['tree_x']),
            'pos_y'     : (lambda mdl: mdl.x[f'pos_y'] + np.ones((len(self.trees_pos),1))@mdl.u[f'pos_y'] - self.param['tree_y']),
            'yaw'       : (lambda mdl: mdl.x[f'yaw'] + mdl.u[f'yaw']),
            'lambda'    : (lambda mdl:  0.5 * ca.logic_not(i_see_tree_casadi(ca.horzcat(mdl.x['pos_x'],mdl.x['pos_y']),mdl.x[f'yaw'])) * np.ones((len(self.trees_pos),1)) + i_see_tree_casadi(ca.horzcat(mdl.x['pos_x'],mdl.x['pos_y']),mdl.x[f'yaw']) * \
                            (np.sin(mdl.x['yaw'])/3 + 0.5)) , 
                            #ca.dot(mdl.x[f'lambda'], mdl.tvp[f'hat_lambda_z'])/(mdl.x[f'lambda'].T@mdl.tvp[f'hat_lambda_z']))
        }

        self.x_k    = np.array([0.0]*len(self.trees_pos)+  [0.0]*len(self.trees_pos)+ [0.0] + [0.5]*len(self.trees_pos))


    def update(self, y_z):

        lambda_z = np.array([(np.sin(y_z['gps'][-1])/3 + 0.5)])
        
        self.x_k[:2*len(self.trees_pos)+1] = np.concatenate((y_z['gps'][0]- self.param['tree_x'],  y_z['gps'][1]- self.param['tree_y'],[y_z['gps'][-1]]))
        
        valid_lambas =trees_satisfy_conditions(y_z['gps'][:2], 
                                 y_z['gps'][-1], 
                                 np.stack((self.param['tree_x'],self.param['tree_y']),axis=1))
        
        for el in valid_lambas:
            i = 2*len(self.trees_pos) + el +1
            self.x_k[i] =  np.dot(np.asarray([self.x_k[i]]),lambda_z)/(np.asarray([self.x_k[i]]).T@lambda_z)


@dataclass
class DroneMpc(MpcClass):
    def __post_init__(self):
        #entropy H = -lambda*ln(lambda) and then we want to max the negative entropy (max{-H})
        self.lterm= lambda mdl: ca.sum1((-mdl.x['lambda']*np.log(mdl.x['lambda'])))
        self.mterm= lambda mdl: ca.sum1((-mdl.x['lambda']*np.log(mdl.x['lambda'])))
        self.bounds_dict= {
            '_u': {'pos_x': {'upper': [1.0], 'lower': [-1.0]}, 'pos_y': {'upper': [1.0], 'lower': [-1.0]}, 'yaw': {'upper': [np.pi/10], 'lower': [-np.pi/10]}},
        }
        super().__post_init__()



class MainClass:
    def __init__(self):
        self.bridge = BridgeClass(SENSORS)
        resp = self.bridge.callServer({"trees_poses": None})
        self.state  = DroneState(trees_pos=resp["trees_poses"], model_type="continuous")
        self.state.populateModel()
        self.mpc  = DroneMpc(self.state.model)
        self.u0= {"cmd_pose" : np.zeros((3,1), dtype=float, order='C')}
        tvp_template = self.mpc.get_tvp_template()
        
        def tvp_fun(time_x):
            return tvp_template
        
        self.mpc.set_tvp_fun(tvp_fun)
        self.mpc.setup()
        print(self.mpc.bounds)
        for i in range(1000):
            print('ok')
            self.loop(i)
        rospy.spin()
        

    def loop(self, i):
        print('pubdata')
        self.bridge.pubData(self.u0)
        print(self.u0)
        print('----')
        # Observe
        # Call the bridge for pose and image
        # Process image to get lambda_zk
        print('subscribed')
        self.state.update(self.bridge.getData())
        print(self.state.x_k.reshape(-1,1))

        if i ==0:
            self.mpc.x0 = self.state.x_k.reshape(-1,1)
            print('----')
            self.mpc.set_initial_guess()
        # state.update(robot_pose, lambda_zk)
        # In state.update:
        # def update(self, robot_pose, lambda_zk):
        #   x = robot_pose
        #   lambda = bayes(lambda, lambda_zk)

        # uk = self.mpc.make_step(state.mpc_variables)
        print('mpc')
        
        self.u0['cmd_pose'] = self.mpc.make_step(self.state.x_k.reshape(-1,1))
        print(self.u0)