from matplotlib import pyplot as plt
from unity_drone_sim_bridge.core_lib.BridgeClass import BridgeClass
from unity_drone_sim_bridge.sensors import SENSORS
from unity_drone_sim_bridge.core_lib.MpcClass import MpcClass
from unity_drone_sim_bridge.core_lib.StateClass import StateClass
from unity_drone_sim_bridge.g_func_lib.g_func_tools import *
from unity_drone_sim_bridge.qi_lib.qi_tools import *
from unity_drone_sim_bridge.g_func_lib.gp_tools import *
from dataclasses import dataclass, field
from typing import Any, Callable
import casadi as ca
import rospy
import numpy as np
from do_mpc.data import save_results

# Constants
REDUCED_DIM_LAMBDA = 4
REDUCED_DIM_OBS = 3
'''
    Model definition
'''
@dataclass
class DroneStateReducedOrder(StateClass):
    g: Callable[[Any], Any] = field(default=(lambda: None))
    trees_pos: np.ndarray = field(default_factory=lambda: np.zeros(1))
    dim_lambda : int = field(default=REDUCED_DIM_LAMBDA)
    dim_obs : int = field(default=REDUCED_DIM_OBS)

    def __post_init__(self):
        super().__post_init__()

        # Parameters
        self.param = {
            'tree_x': np.array(self.trees_pos[:, 0]),
            'tree_y': np.array(self.trees_pos[:, 1]),
            'Xtree': np.array(self.trees_pos),
            'lambda': np.full(len(self.trees_pos), 0.5)
        }

        # State dictionary
        self.state_dict = {
            '_x': {
                'Xrobot': 3,
                'Xrobot_tree': self.dim_obs,
                'y': self.dim_lambda,
                'lambda': self.dim_lambda,
                'lambda_prev': self.dim_lambda
            },
            '_u': {'Xrobot': 3},
            '_tvp': {
                'ReducedOrderXrobot_tree_lambda': (self.dim_lambda, 2),
                'ReducedOrderXrobot_tree_obs': (self.dim_obs, 2),
                'ReducedOrderH': 1,
                'ReducedOrderH_prev': 1,
            }
        }

        mapped_g = g_map_casadi_fixed_cond(setup_g_inline_casadi_fixed_cond(self.g), [self.dim_lambda, 2])

        # Expressions dictionary
        self.exp_dict = {
            'H': lambda mdl: ca.sum1(mdl.x['lambda'] * ca.log(mdl.x['lambda'])) + mdl.tvp['ReducedOrderH'],
            'H_prev': lambda mdl: ca.sum1(mdl.x['lambda_prev'] * ca.log(mdl.x['lambda_prev'])) + mdl.tvp['ReducedOrderH_prev'],
            'y': (lambda mdl: mapped_g(  mdl.x['Xrobot',:2]+mdl.u['Xrobot',:2],
                                            mdl.x['Xrobot',-1]+mdl.u['Xrobot',-1],
                                            mdl.tvp['ReducedOrderXrobot_tree_lambda'])),
            'drone_trees_dist': lambda mdl: drone_objects_distances_casadi(
                mdl.x['Xrobot', :2] + mdl.u['Xrobot', :2],
                mdl.tvp['ReducedOrderXrobot_tree_obs']),
            'cost_function': lambda mdl: -(mdl.aux['H'] - mdl.aux['H_prev']),
        }

        # Reshaping dictionary
        self.rsh_dict = {
            'Xrobot': lambda mdl: mdl.x['Xrobot'] + mdl.u['Xrobot'],
            'Xrobot_tree': lambda mdl: mdl.aux['drone_trees_dist'],
            'y': lambda mdl: mdl.aux['y'],
            'lambda': lambda mdl: bayes(mdl.x['lambda'], mdl.aux['y']),
            'lambda_prev': lambda mdl: mdl.x['lambda'],
        }

        # Initial state values
        self.x_k = {key: 0.5 * np.ones(shape) if key == 'lambda_prev' else np.zeros(shape)
                    for key, shape in self.state_dict['_x'].items()}
        self.x_k_full = {'lambda': 0.5 * np.ones((len(self.trees_pos), 1)),
                         'lambda_prev': 0.5 * np.ones((len(self.trees_pos), 1))}
        
        # Nearest trees
        self.update_nearest_trees()

    def update_nearest_trees(self):
        """Update nearest tree indices."""
        self.ReducedOrderIdxsLambda = n_nearest_objects(self.x_k['Xrobot'][:2].reshape(1, -1), self.param['Xtree'], num=REDUCED_DIM_LAMBDA)
        self.ResidualIdxsLambda = np.setdiff1d(np.arange(self.param['lambda'].shape[0]), self.ReducedOrderIdxsLambda)
        self.ReducedOrderIdxsTree = n_nearest_objects(self.x_k['Xrobot'][:2].reshape(1, -1), self.param['Xtree'], num=REDUCED_DIM_OBS)

    def data_association(self, robot_pose, bboxes):
        pass

    def update(self, y_z):
        """Update the drone state."""
        self.update_nearest_trees()
        
        self.x_k['Xrobot'] = np.array([y_z['gps'][0], y_z['gps'][1], y_z['gps'][-1]])
        self.x_k['Xrobot_tree'] = drone_objects_distances_np(self.x_k['Xrobot'][:2], self.param['Xtree'][self.ReducedOrderIdxsTree])
        
        if 'detection' in y_z and y_z['detection']:
            self.x_k['y'][self.data_association(self, y_z['gps'], ['gps']['detection'])] = [i['score'] for i in y_z['gps']['detection']]
        else:
            qi_z = ((1 + np.cos(y_z['gps'][-1])) / 15 + 0.5)
            self.x_k['y'] = 0.5 + (qi_z - 0.5) * fov_weight_fun_numpy(
                drone_pos=y_z['gps'][:2], 
                drone_yaw=y_z['gps'][-1], 
                tree_pos=np.stack((self.param['tree_x'][self.ReducedOrderIdxsLambda], self.param['tree_y'][self.ReducedOrderIdxsLambda]), axis=1))
        
        self.x_k_full['lambda_prev'] = self.x_k_full['lambda']
        self.x_k_full['lambda'][self.ReducedOrderIdxsLambda] = bayes(self.x_k_full['lambda_prev'][self.ReducedOrderIdxsLambda], self.x_k['y'])
        self.x_k['lambda_prev'] = self.x_k['lambda']
        self.x_k['lambda'] = self.x_k_full['lambda'][self.ReducedOrderIdxsLambda]

'''
    MPC setup definition
'''
@dataclass
class DroneMpc(MpcClass):
    def __post_init__(self):
        """Initialize MPC with specific terms and constraints."""
        self.mterm = lambda mdl: mdl.aux['cost_function']
        self.lterm = lambda mdl: mdl.aux['cost_function']
        self.rterm = lambda mdl: (1/100) * ca.sum1(mdl.u['Xrobot'] ** 2)

        self.bounds_dict = {
            '_u': {'Xrobot': {'upper': 2 * [.5] + [np.pi/40], 'lower': 2 * [-.5] + [-np.pi/40]}},
            '_x': {'Xrobot_tree': {'lower': self.model.x['Xrobot_tree'].shape[0] * [1]}}
        }

        super().__post_init__()

'''
    Model instantiation
    MPC setup procedure
    Unity Bridge instatiation
'''
class MainNode:
    def __init__(self, viz=True):
        self.viz = viz
        self.bridge = BridgeClass(SENSORS)
        resp = self.bridge.callServer({"trees_poses": None})["trees_poses"]
        
        self.state = DroneStateReducedOrder(model_ca='MX', model_type="discrete", g=load_g('gp'), trees_pos=resp)
        self.state.populateModel()
        self.mpc = DroneMpc(self.state)
        
        self.u0 = {"cmd_pose": np.zeros((3, 1)), "viz_pred_pose": None}
        self.x_data = []

        self.setUpMpc()
        self.setUpDataSaver()
        self.runSimulation()
        rospy.spin()

    def setUpMpc(self):
        """Setup the MPC with a time-varying parameter function and solver settings."""
        tvp_template = self.mpc.get_tvp_template()

        def tvp_fun(time_x):
            for k in range(len(tvp_template['_tvp', :, 'ReducedOrderXrobot_tree_lambda'])):
                tvp_template['_tvp', k, 'ReducedOrderXrobot_tree_obs'] = self.state.trees_pos[self.state.ReducedOrderIdxsTree]
                tvp_template['_tvp', k, 'ReducedOrderXrobot_tree_lambda'] = self.state.trees_pos[self.state.ReducedOrderIdxsLambda]
                tvp_template['_tvp', k, 'ReducedOrderH'] = np.sum(
                    self.state.param['lambda'][self.state.ResidualIdxsLambda] * np.log(self.state.param['lambda'][self.state.ResidualIdxsLambda]))
                tvp_template['_tvp', k, 'ReducedOrderH_prev'] = (
                    tvp_template['_tvp', -1, 'ReducedOrderH'] if k == 0 and time_x != 0 else tvp_template['_tvp', k, 'ReducedOrderH']
                )
            return tvp_template

        self.mpc.set_tvp_fun(tvp_fun)
        self.mpc.settings.set_linear_solver(solver_name="MA27")
        self.mpc.settings.supress_ipopt_output()
        self.mpc.setup()

    def setUpDataSaver(self):
        """Setup CSV writers for saving data."""
        import csv
        self.writer_xk = csv.writer(open('data_xk.csv', mode='w', newline=''))
        self.writer_hk = csv.writer(open('data_Hk.csv', mode='w', newline=''))
        self.writer_yaw = csv.writer(open('data_yaw.csv', mode='w', newline=''))
        self.writer_y = csv.writer(open('data_y.csv', mode='w', newline=''))
        self.writer_lambda = csv.writer(open('data_lambda.csv', mode='w', newline=''))

    def saveStepDataCsv(self, i):
        """Save step data to CSV files."""
        self.x_data.append(self.state.x_k)
        self.writer_xk.writerow([i, self.state.x_k])
        self.writer_hk.writerow([i, self.mpc.data['_aux', 'H']])
        self.writer_yaw.writerow([i, self.mpc.data.prediction(('_x', 'Xrobot')).reshape(1, -1)[0]])
        self.writer_y.writerow([i, self.mpc.data.prediction(('_x', 'y')).reshape(1, -1)[0]])
        self.writer_lambda.writerow([i, self.mpc.data.prediction(('_x', 'lambda')).reshape(1, -1)[0]])

    def runSimulation(self):
        """Run the simulation loop."""
        for i in range(50):
            print('Step:', i)
            self.loop(i)
            self.saveStepDataCsv(i)
            if self.viz:
                pass  # Add visualization if needed

        save_results([self.mpc])

    def loop(self, i):
        """Main loop for commanding the drone and updating state."""
        # Command
        self.bridge.pubData(self.u0)

        # Observe and update state
        self.state.update(self.bridge.getData())
        
        if i == 0:
            self.mpc.x0 = np.concatenate(list(self.state.x_k.values()), axis=None)
            self.mpc.set_initial_guess()

        # MPC step
        self.u0['cmd_pose'] = self.mpc.make_step(np.concatenate(list(self.state.x_k.values()), axis=None))
