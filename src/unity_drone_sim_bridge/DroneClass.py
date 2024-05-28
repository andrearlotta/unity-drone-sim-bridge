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
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
from unity_drone_sim_bridge.MpcPlotter import  MPCPlotter

@dataclass
class DroneState(StateClass):
    g: Callable[[Any], Any] = field(default= (lambda:None))
    trees_pos:   np.ndarray = field(default=lambda: np.zeros(1))
    def __post_init__(self):
        super().__post_init__()
        '''
        overwritten elemets
        '''
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
            'y'                 :   (lambda mdl:    self.g(mdl.x['yaw'])),
            'lambda'            :   (lambda mdl:    bayes(mdl.x['lambda'],mdl.x['y'])),
            'lambda_prev'       :   (lambda mdl:    mdl.x['lambda']),
        }

        self.x_k    = np.array([0.0]*len(self.trees_pos) + [0.5]*len(self.trees_pos) + [0.5]*len(self.trees_pos)*2 )


    def update(self, y_z):
        qi_z = np.array([fake_nn(img_2_qi(y_z['image']))])
        self.x_k[0] = np.array([y_z['gps'][-1]])
        self.x_k[1] = qi_z                                       #(1 + np.cos(np.array([y_z['gps'][-1]]))/ 4 + 0.4)
        self.x_k[3] = self.x_k[2]
        self.x_k[2] = bayes(self.x_k[3],self.x_k[1])


@dataclass
class DroneMpc(MpcClass):
    def __post_init__(self):
        #entropy H = -lambda*ln(lambda) and then we want to max the negative entropy (max{-H})
        self.lterm= lambda mdl:  -(mdl.aux['H'] - mdl.aux['H_prev'])
        self.mterm= lambda mdl: -(mdl.aux['H'] - mdl.aux['H_prev'])  
        self.rterm= lambda mdl: ((5/100000)*mdl.u['yaw'])**2
        self.bounds_dict= {
            '_u': {'yaw': {'upper': [np.pi/90], 'lower': [-np.pi/90]}},
        }
        super().__post_init__()


class MainClass:
    def __init__(self):
        self.bridge = BridgeClass(SENSORS)
        resp = self.bridge.callServer({"trees_poses": None})
        self.state  = DroneState(model_type="continuous", g=load_g('mlp'), trees_pos=resp["trees_poses"],)
        self.state.populateModel()
        self.mpc  = DroneMpc(self.state.model)
        self.u0= {"cmd_pose" : np.deg2rad(1.0)*np.ones((1,1), dtype=float, order='C')}
        self.x_data = []
        tvp_template = self.mpc.get_tvp_template()   
        def tvp_fun(time_x):
            return tvp_template
        self.mpc.set_tvp_fun(tvp_fun)
        self.mpc.setup()
        self.setUpGraphics()
        self.setUpDataSaver()

        for i in range(100):
            print('ok')
            self.loop(i)
            self.saveStepDataCsv()
        
        self.generateGif()
        save_results([self.mpc])

        rospy.spin()

    def setUpGraphics(self):
        import matplotlib.pyplot as plt
        import matplotlib as mpl
        # Customizing Matplotlib:
        mpl.rcParams['font.size'] = 18
        mpl.rcParams['lines.linewidth'] = 3
        mpl.rcParams['axes.grid'] = True
        self.mpc_graphics = do_mpc.graphics.Graphics(self.mpc.data)
        # We just want to create the plot and not show it right now. This "inline magic" supresses the output.
        self.fig, self.ax = plt.subplots(4, sharex=True, figsize=(16,9))
        self.fig.align_ylabels()
        for g in [self.mpc_graphics]:
            # Plot the angle positions (phi_1, phi_2, phi_2) on the first axis:
            g.add_line(var_type='_x', var_name='yaw', axis=self.ax[0])
            g.add_line(var_type='_x', var_name='y', axis=self.ax[1])
            g.add_line(var_type='_x', var_name='lambda', axis=self.ax[2])

            # Plot the set motor positions (phi_m_1_set, phi_m_2_set) on the second axis:
            g.add_line(var_type='_u', var_name='yaw', axis=self.ax[3])

        self.ax[0].set_ylabel('angle position [rad]')
        self.ax[1].set_ylabel('qi [#]')
        self.ax[2].set_ylabel('lambda [#]')
        self.ax[3].set_ylabel('d_yaw [rad]')
    
    def setUpDataSaver(self):
        import csv
        file_xk = open('data_xk.csv', mode='w', newline='')
        file_yaw = open('data_yaw.csv', mode='w', newline='')
        file_y = open('data_y.csv', mode='w', newline='')
        file_lambda = open('data_lambda.csv', mode='w', newline='')

        self.writer_xk = csv.writer(file_xk)
        self.writer_yaw = csv.writer(file_yaw)
        self.writer_y = csv.writer(file_y)
        self.writer_lambda = csv.writer(file_lambda)

    def generateGif(self):
        anim = FuncAnimation(self.fig, self.updateGraphics, frames=50, repeat=False)
        gif_writer = ImageMagickWriter(fps=3)
        anim.save('anim.gif', writer=gif_writer)

    def saveStepDataCsv(self):
        self.x_data.append(self.state.x_k)     
        self.writer_xk.writerow([i, self.state.x_k])  # Adjust the data to write as needed
        self.writer_yaw.writerow([i, self.mpc.data.prediction(('_x', 'yaw'   )).reshape(1,-1)[0]])
        self.writer_y.writerow([i, self.mpc.data.prediction(('_x', 'y'     )).reshape(1,-1)[0]])
        self.writer_lambda.writerow([i, self.mpc.data.prediction(('_x', 'lambda')).reshape(1,-1)[0]])

    def updateGraphics(self, t_ind):
        #for i in range(3):
        #    self.ax[i].plot(np.linspace(0,t_ind,t_ind+1),np.array(self.x_data)[0:t_ind+1,i] )
        self.mpc_graphics.plot_predictions(t_ind)
        self.mpc_graphics.reset_axes()

    def loop(self, i):
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
        print(self.state.x_k.reshape(-1,1))

        if i ==0:
            print(self.mpc.x0)
            self.mpc.x0 = self.state.x_k.reshape(-1,1)
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
        self.u0['cmd_pose'] = self.mpc.make_step(self.state.x_k.reshape(-1,1))
        print(self.u0)
        print('----')