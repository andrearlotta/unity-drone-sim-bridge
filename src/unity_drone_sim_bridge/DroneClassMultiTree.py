from matplotlib import pyplot as plt
from unity_drone_sim_bridge.nn_lib.NeuralClass import NeuralClass
from unity_drone_sim_bridge.BridgeClass import BridgeClass
from unity_drone_sim_bridge.sensors import SENSORS
from unity_drone_sim_bridge.MpcClass import MpcClass
from unity_drone_sim_bridge.StateClass import StateClass
from unity_drone_sim_bridge.nn_lib.nn_tools import *
from unity_drone_sim_bridge.qi_lib.qi_tools import *
from unity_drone_sim_bridge.nn_lib.gp_nn_tools import *
from dataclasses import dataclass, field
from typing import List, Dict, Union, Any, Callable
import do_mpc
import casadi as ca
import rospy
from std_msgs.msg import Empty
import numpy as np
from do_mpc.data import save_results, load_results
from unity_drone_sim_bridge.MpcPlotter import  MPCPlotter

@dataclass
class DroneState(StateClass):
    g           :       Callable[[Any], Any] = field(default= (lambda:None))
    trees_pos   :       np.ndarray = field(default=lambda: np.zeros(1))

    def __post_init__(self):
        super().__post_init__()

        '''
        overwritten elemets
        '''
        # Create a function for evaluation
        #F = ca.Function('F', [a, x], [ca.if_else(a == 0, f(x), g(x))])

        self.param = {
            'tree_x': np.array(self.trees_pos[:,0]),
            'tree_y': np.array(self.trees_pos[:,1]),
            'Xtree':  np.array(self.trees_pos)
        }

        self.state_dict= {
            '_x':   {   'Xrobot'            :   3,
                        'Xrobot_tree'       :   len(self.trees_pos),
                        'y'                 :   len(self.trees_pos),
                        'lambda'            :   len(self.trees_pos),
                        'lambda_prev'       :   len(self.trees_pos)},
            '_u':   {   'Xrobot'            :   3},
        }

        self.exp_dict = {
            'H'                 :   (lambda mdl:    ca.sum1(mdl.x['lambda']*ca.log(mdl.x['lambda']))),
            'H_prev'            :   (lambda mdl:    ca.sum1(mdl.x['lambda_prev']*ca.log(mdl.x['lambda_prev']))),
            'interoccl_check'   :   (lambda mdl:    trees_satisfy_conditions_casadi(mdl.x['Xrobot',:2],\
                                                                                    mdl.x['Xrobot',-1],\
                                                                                    ca.MX(self.trees_pos))),
            'y'                 :   (lambda mdl:    0.5 + trees_satisfy_conditions_casadi(mdl.x['Xrobot',:2]+ mdl.u['Xrobot',:2],\
                                                                                          mdl.x['Xrobot',-1]+ mdl.u['Xrobot',-1], \
                                                                                          ca.MX(self.trees_pos)) * (self.g(mdl.x['Xrobot',-1] + mdl.u['Xrobot',-1]) - 0.5) ),
            'drone_trees_dist'  :   (lambda mdl:    drone_trees_distances_casadi(mdl.x['Xrobot',:2]+ mdl.u['Xrobot',:2], ca.MX(self.trees_pos))),
            'cost_function'     :   (lambda mdl:    - (mdl.aux['H'] - mdl.aux['H_prev'])),
        }

        self.rsh_dict = {
            'Xrobot'            :   (lambda mdl:    mdl.x['Xrobot'] + mdl.u['Xrobot']),
            'Xrobot_tree'       :   (lambda mdl:    mdl.aux['drone_trees_dist']),
            'y'                 :   (lambda mdl:    mdl.aux['y']),
            'lambda'            :   (lambda mdl:    bayes(mdl.x['lambda'],mdl.aux['y'])),
            'lambda_prev'       :   (lambda mdl:    mdl.x['lambda']),
        }

        self.x_k = {key: 0.5 * np.ones(shape) if key in 'lambda_prev' else np.zeros(shape) for key, shape in self.state_dict['_x'].items()}

    def update(self, y_z):
        qi_z = ((1 + np.cos(y_z['gps'][-1]))/ 15 + 0.5) # fake_nn(img_2_qi(y_z['image'])) #
        
        self.x_k['Xrobot'] = np.concatenate(([y_z['gps'][0]],  
                                              [y_z['gps'][1]],
                                              [y_z['gps'][-1]]))
        self.x_k['Xrobot_tree'] = drone_trees_distances(self.x_k['Xrobot'][:2] , self.trees_pos)

        self.x_k['y'] = 0.5 + (qi_z - 0.5) * trees_satisfy_conditions_np(
                                drone_pos=y_z['gps'][:2], 
                                drone_yaw=y_z['gps'][-1], 
                                tree_pos=np.stack((self.param['tree_x'],self.param['tree_y']),axis=1))

        self.x_k['lambda_prev']     =   self.x_k['lambda']
        self.x_k['lambda']          =   bayes(self.x_k['lambda_prev'],self.x_k['y'])

@dataclass
class DroneMpc(MpcClass):
    def __post_init__(self):
        """
        entropy H = -lambda*ln(lambda) and then we want to max the negative entropy (max{-H})
        """
        self.mterm = lambda mdl:    -mdl.aux['H'] #- (1/100) * ca.sum1(mdl.aux['interoccl_check'])
        self.lterm = lambda mdl:    mdl.aux['cost_function'] 
                                        #- (1/100) * ca.sum1(mdl.aux['interoccl_check'])
                                        #(1/100)*ca.sum1(gaussian(mdl.aux['drone_trees_dist'], mu=0, sig=1.0)) 
        self.rterm  = lambda mdl:   ((1/100)*ca.sum1(mdl.u['Xrobot']**2))

        self.bounds_dict    = {
            '_u':   {'Xrobot': {'upper': 2 * [.5] + [np.pi/40], 'lower': 2 * [-.5] + [-np.pi/40]}},
            '_x':   {'Xrobot': {'lower': 2 * [-10] + [-np.pi], 'upper': 2 * [30] + [np.pi] },
                     'Xrobot_tree': {'lower': self.model.x['Xrobot_tree'].shape[0]*[1]}
                     }
        }
        #self.scaling['_x', 'Xrobot'] = [1.0,1.0,.1]
        super().__post_init__()

class MainClass:
    def __init__(self, viz=True):
        self.viz = viz
        self.bridge = BridgeClass(SENSORS)
        resp = self.bridge.callServer({"trees_poses": None})["trees_poses"]
        self.state  = DroneState(model_ca='MX',model_type="discrete", g=load_g('gp'), \
                                 trees_pos=resp)

        self.state.populateModel()
        self.mpc  = DroneMpc(self.state.model)
        self.u0= {"cmd_pose" : np.zeros((3,1), dtype=float, order='C'),
                  "viz_pred_pose" : None}
        
        self.x_data = []

        tvp_template = self.mpc.get_tvp_template()   
        def tvp_fun(time_x):
            return tvp_template
        self.mpc.set_tvp_fun(tvp_fun)
        self.mpc.settings.set_linear_solver(solver_name = "MA27")
        self.mpc.settings.supress_ipopt_output()
        self.mpc.setup()

        self.plotter = MPCPlotter(self.mpc)

        self.setUpDataSaver()

        for i in range(100):
            print('ok')
            self.loop(i)
            self.saveStepDataCsv(i)
            if self.viz: pass
            #self.plotter.plot()

        #self.generateGif()
        save_results([self.mpc])
        rospy.spin()

    def setUpDataSaver(self):
        import csv
        file_xk = open('data_xk.csv', mode='w', newline='')
        file_hk = open('data_Hk.csv', mode='w', newline='')
        file_yaw = open('data_yaw.csv', mode='w', newline='')
        file_y = open('data_y.csv', mode='w', newline='')
        file_lambda = open('data_lambda.csv', mode='w', newline='')

        self.writer_xk = csv.writer(file_xk)
        self.writer_hk = csv.writer(file_hk)
        self.writer_yaw = csv.writer(file_yaw)
        self.writer_y = csv.writer(file_y)
        self.writer_lambda = csv.writer(file_lambda)

    def saveStepDataCsv(self, i):
        self.x_data.append(self.state.x_k) 
        self.writer_xk.writerow([i,     self.state.x_k])  # Adjust the data to write as needed
        self.writer_hk.writerow([i ,    self.mpc.data['_aux', 'H']] )
        self.writer_yaw.writerow([i,    self.mpc.data.prediction(('_x', 'Xrobot')).reshape(1,-1)[0]])
        self.writer_y.writerow([i,      self.mpc.data.prediction(('_x', 'y')).reshape(1,-1)[0]])
        self.writer_lambda.writerow([i, self.mpc.data.prediction(('_x', 'lambda')).reshape(1,-1)[0]])

    def loop(self, i):
        '''
        Command
        Call the bridge for publishing pose commands
        '''        
        print('pubdata\n')
        self.bridge.pubData(self.u0)
        print(self.u0)
        print('----')

        '''
        Observe
        Call the bridge for pose and image
        Process image to get y
        '''
        print('update state\n')
        self.state.update(self.bridge.getData())
        print(self.state.x_k)
        if i ==0:
            print(self.mpc.x0)
            self.mpc.x0 = np.concatenate(list(self.state.x_k.values()), axis=None)
            self.mpc.set_initial_guess()
        print('----')

        '''
        state.update(robot_pose, lambda_zk)
        In state.update:
        def update(self, robot_pose, lambda_zk):
            x = robot_pose
            lambda = bayes(lambda, lambda_zk)
            uk = self.mpc.make_step(state.mpc_variables)
        '''
        print('mpc prediction step\n')   
        self.u0['cmd_pose'] = self.mpc.make_step(np.concatenate(list(self.state.x_k.values()), axis=None))
        #if self.viz: self.u0['viz_pred_pose'] = self.mpc.data.prediction(('_x', 'Xrobot'))[:,:,:]
        print(self.u0)
        print('----')