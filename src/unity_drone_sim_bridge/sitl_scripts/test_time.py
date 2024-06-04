from matplotlib import pyplot as plt
from unity_drone_sim_bridge.core_lib.MpcClass import MpcClass
from unity_drone_sim_bridge.core_lib.StateClass import StateClass
from unity_drone_sim_bridge.g_func_lib.g_func_tools import *
from unity_drone_sim_bridge.qi_lib.qi_tools import *
from unity_drone_sim_bridge.g_func_lib.gp_tools import *
from unity_drone_sim_bridge.g_func_lib.nn_tools import *
from dataclasses import dataclass, field
import rospy
from std_msgs.msg import Empty
import numpy as np
from do_mpc.data import save_results, load_results
from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
import csv
import time
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Union, Any, Callable

def Test_nn(train=False):
    hidden_layers = [1,5,10]
    layer_dims = list(2** i for i in range(6,15,3))  # List of i values
    all_diff_times = {n_layer: [] for n_layer in hidden_layers}  # Dictionary to store diff times for each method
    all_diff_times_std = {n_layer: [] for n_layer in hidden_layers}  # Dictionary to store diff times for each method
    
    for dataset_dim in list(i*0.1 for i in range(2,8,3)):
        for n_layer in hidden_layers:
            for layer_dim in layer_dims:
                if train:
                    model = TrainNN(test_size=dataset_dim, input_size = 1, hidden_layer=n_layer, hidden_size = layer_dim, output_size = 1, learning_rate = 0.0001, num_epochs = 500, batch_size = 1)
                    nn_ca = l4c.L4CasADi(model, model_expects_batch_dim=True, device='cpu')
                    g = lambda drone_pos,drone_yaw, tree_pos_single: g_nn(drone_pos,drone_yaw, tree_pos_single, nn_ca)
                else: 
                    model = LoadNN(layer_dim,n_layer,test_size=dataset_dim)
                mpc = MainClass(model_type="continuous", model_ca='MX', g = model) # SX not working. 
                loop_ = []
                for i in range(1):
                    loop_.append(mpc.loop())
                all_diff_times[n_layer].append(np.mean(loop_))
                all_diff_times_std[n_layer].append(np.std(loop_))
        
        # Save data to CSV
        with open(f'benchmark_l4ca_nn_data{int(10*dataset_dim)}.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['i']+ hidden_layers)
            for i, diffs in zip(layer_dims, zip(*[all_diff_times[method] for method in hidden_layers])):
                writer.writerow([i] + list(diffs))
        with open(f'benchmark_l4ca_nn_data{int(10*dataset_dim)}_std.csv', mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(['i']+ hidden_layers)
            for i, diffs in zip(layer_dims, zip(*[all_diff_times_std[method] for method in hidden_layers])):
                writer.writerow([i] + list(diffs))

    ## Plotting
    #for n_layer in hidden_layers:
    #    plt.plot(layer_dims, all_diff_times[n_layer], label=n_layer)
    #
    #plt.xlabel('i')
    #plt.ylabel('Time Difference (ms)')
    #plt.title('Time Difference vs i')
    #plt.legend()
    #plt.show()


def Test_gp(g_type='gp'):
    methods = ['ME', 'TA', 'EM', 'old_ME', 'old_TA']  # List of methods
    i_values = list(2** i for i in range(6,9))  # List of i values
    all_diff_times = {method: [] for method in methods}  # Dictionary to store diff times for each method
    
    for method in methods:
        for i in i_values:
            if g_type == 'gp':
                gp = LoadCaGP(synthetic=False, N=i, method=method)  # Assuming LoadCaGP is defined elsewhere
                now = time.time_ns()
                gp.predict(ca.MX([10]), [], np.zeros((1, 1)))[0]  # Assuming the predict function is defined elsewhere
                diff = (time.time_ns() - now) / 1_000_000  # Convert diff from ns to ms
                g = lambda x_k: np.ones(x_k.shape) * gp.predict(x_k, [], np.zeros((1,1)))[0]
            mpc = MainClass(model_type="discrete", model_ca='MX', g = g)
            loop_ = []
            for i in range(10):
                loop_.append(mpc.loop())
            all_diff_times[method].append(np.mean(loop_))
    
    # Save data to CSV
    with open('time_diff_data.csv', mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['i', 'ME_time', 'TA_time', 'EM_time', 'old_ME_time', 'old_TA_time'])
        for i, diffs in zip(i_values, zip(*[all_diff_times[method] for method in methods])):
            writer.writerow([i] + list(diffs))
        
    # Plotting
    for method in methods:
        plt.plot(i_values, all_diff_times[method], label=method)
    
    plt.xlabel('i')
    plt.ylabel('Time Difference (ms)')
    plt.title('Time Difference vs i')
    plt.legend()
    plt.show()

@dataclass
class DroneState(StateClass):
    trees_pos:   np.ndarray = field(default=lambda: np.zeros(1))
    g: Callable[[Any], Any] = field(default_factory= (lambda: load_g()))
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

        self.x_k    = np.array([np.pi/3]*len(self.trees_pos) + [0.5]*len(self.trees_pos) + [0.5]*len(self.trees_pos)*2 )


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
        #self.rterm= lambda mdl: 0.01 * (mdl.u['yaw']**2) 
        self.bounds_dict= {
            '_u': {'yaw': {'upper': [np.pi/90], 'lower': [-np.pi/90]}},
        }
        super().__post_init__()

class MainClass:
    def __init__(self, model_type="continuous", model_ca='SX', g=lambda:None):
        self.state  = DroneState(model_type=model_type, model_ca=model_ca, g=g)
        self.state.populateModel()
        self.mpc  = DroneMpc(self.state.model)
        self.u0= {"cmd_pose" : np.deg2rad(1.0)*np.ones((1,1), dtype=float, order='C')}
        self.x_data = []
        self.mpc.setup()

    def loop(self):
        '''
        Observe
        Call the bridge for pose and image
        Process image to get y
        '''
        print(self.state.x_k.reshape(-1,1))
        print(self.mpc.x0)
        self.mpc.x0 = self.state.x_k.reshape(-1,1)
        print('----')
        self.mpc.set_initial_guess()

        '''
        state.update(robot_pose, lambda_zk)
        In state.update:
        def update(self, robot_pose, lambda_zk):
            x = robot_pose
            lambda = bayes(lambda, lambda_zk)
            uk = self.mpc.make_step(state.mpc_variables)
        '''

        print('mpc')
        now = time.time_ns()
        self.u0['cmd_pose'] = self.mpc.make_step(self.state.x_k.reshape(-1,1))
        diff = (time.time_ns() - now) / 1_000_000  # Convert diff from ns to ms
        print(self.u0)
        return diff

if __name__ == '__main__':
    Test_nn()