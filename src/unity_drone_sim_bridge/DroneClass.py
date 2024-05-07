from matplotlib import pyplot as plt
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

        self.state_dict = {
            '_x'    :   {'yaw':1, 'y':1, 'lambda':1, 'lambda_prev':1},
            '_u'    :   {'yaw':1},
        }

        self.exp_dict = {
            'H'         :   (lambda mdl: ca.sum2(mdl.x['lambda']*ca.log(mdl.x['lambda']))),
            'H_prev'    :   (lambda mdl: ca.sum2(mdl.x['lambda_prev']*ca.log(mdl.x['lambda_prev'])))
        }

        self.rsh_dict = {
            'yaw'               :   (lambda mdl:    mdl.u['yaw']),
            'y'                 :   (lambda mdl:    g(mdl.x['yaw'], self.gp)),
            'lambda'            :   (lambda mdl:    bayes(mdl.x['lambda'],mdl.x['y'])),
            'lambda_prev'       :   (lambda mdl:    mdl.x['lambda']),
        }

        self.x_k    = np.array([0.0]*len(self.trees_pos) + [0.5]*len(self.trees_pos) + [0.5]*len(self.trees_pos)*2 )


    def update(self, y_z):

        lambda_z = np.array([fake_nn(img_2_qi(y_z['image']))])
        print(y_z['gps'])
        self.x_k[0] = np.array([y_z['gps'][-1]])
        self.x_k[1] = (1 + np.cos(np.array([y_z['gps'][-1]]))/ 4 + 0.4)
        self.x_k[3] = self.x_k[2]
        self.x_k[2] = bayes(self.x_k[3],self.x_k[1])


@dataclass
class DroneMpc(MpcClass):
    def __post_init__(self):
        #entropy H = -lambda*ln(lambda) and then we want to max the negative entropy (max{-H})
        self.lterm= lambda mdl: -(mdl.aux['H'] - mdl.aux['H_prev'])
                                #mdl.u[f'yaw']**2#ca.sum1((-mdl.x['lambda']*np.log(mdl.x['lambda'])))
        self.mterm= lambda mdl: -(mdl.aux['H'] - mdl.aux['H_prev']) 
                                #mdl.x['lambda']**2 #ca.sum1((-mdl.x['lambda']*np.log(mdl.x['lambda'])))
        self.bounds_dict= {
            '_u': {'yaw': {'upper': [np.pi/180], 'lower': [-np.pi/180]}},
        }
        super().__post_init__()


class MainClass:
    def __init__(self):
        self.bridge = BridgeClass(SENSORS)
        resp = self.bridge.callServer({"trees_poses": None})
        self.state  = DroneState(trees_pos=resp["trees_poses"], model_type="continuous")
        self.state.populateModel()
        self.mpc  = DroneMpc(self.state.model)
        self.u0= {"cmd_pose" : np.deg2rad(1.0)*np.ones((1,1), dtype=float, order='C')}
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
            print(self.mpc.x0)
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