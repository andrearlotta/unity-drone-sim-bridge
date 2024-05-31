from matplotlib import pyplot as plt
from unity_drone_sim_bridge.MpcClass import MpcClass
from unity_drone_sim_bridge.StateClass import StateClass
from unity_drone_sim_bridge.nn_lib.nn_tools import *
from unity_drone_sim_bridge.qi_lib.qi_tools import *
from unity_drone_sim_bridge.nn_lib.gp_nn_tools import *
from dataclasses import dataclass, field
from typing import List, Dict, Union, Any, Callable
import do_mpc
import casadi as ca
from std_msgs.msg import Empty
import numpy as np
from do_mpc.data import save_results, load_results
from unity_drone_sim_bridge.MpcPlotter import MPCPlotter


def generate_tree_positions(grid_size, spacing):
    """Generate tree positions in a grid."""
    x_positions = np.arange(0, grid_size[0], spacing)
    y_positions = np.arange(0, grid_size[1], spacing)
    xv, yv = np.meshgrid(x_positions, y_positions)
    tree_positions = np.vstack([xv.ravel(), yv.ravel()]).T
    return tree_positions

def set_drone_position(tree_positions, min_distance):
    """Set the drone position ensuring it is at least `min_distance` away from all trees."""
    while True:
        drone_pos = np.random.rand(2) * (np.max(tree_positions, axis=0) - np.min(tree_positions, axis=0)) + np.min(tree_positions, axis=0)
        distances = np.linalg.norm(tree_positions - drone_pos, axis=1)
        if np.all(distances >= min_distance):
            break
    drone_yaw = np.pi
    return np.array([*drone_pos, drone_yaw])

# Constants
MOR_DIM_LAMBDA = 4
MOR_DIM_OBS = 3
TREE_SPACING = 3  # Spacing between trees in the grid
GRID_SIZE = (100, 100)  # Grid size (number of trees along x and y)
SPACING = 8  # Spacing between trees in meters
MIN_DISTANCE = 1.5  # Minimum distance from drone to any tree in meters

# Generate tree positions
tree_positions = generate_tree_positions(GRID_SIZE, SPACING)

# Set drone position
drone_position = set_drone_position(tree_positions, MIN_DISTANCE)

# Visualization
plt.figure(figsize=(8, 8))
plt.scatter(tree_positions[:, 0], tree_positions[:, 1], c='green', label='Trees')
plt.scatter(drone_position[0], drone_position[1], c='red', label='Drone')
plt.quiver(drone_position[0], drone_position[1], np.cos(drone_position[2]), np.sin(drone_position[2]), scale=5, color='red')
plt.xlim([0, GRID_SIZE[0]])
plt.ylim([0, GRID_SIZE[1]])
plt.legend()
plt.grid(True)
plt.xlabel('X position (m)')
plt.ylabel('Y position (m)')
plt.title('Tree Positions and Drone Initial Position')
plt.show()

# Printing positions
print("Tree Positions:")
print(tree_positions)
print("Drone Initial Position:")
print(drone_position)

@dataclass
class DroneStateMor(StateClass):
    g: Callable[[Any], Any] = field(default=(lambda: None))
    trees_pos: np.ndarray = field(default_factory=lambda: np.zeros(1))

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
                'Xrobot_tree': MOR_DIM_OBS,
                'y': MOR_DIM_LAMBDA,
                'lambda': MOR_DIM_LAMBDA,
                'lambda_prev': MOR_DIM_LAMBDA
            },
            '_u': {'Xrobot': 3},
            '_tvp': {
                'MorXrobot_tree_lambda': (MOR_DIM_LAMBDA, 2),
                'MorXrobot_tree_obs': (MOR_DIM_OBS, 2),
                'MorH': 1,
                'MorH_prev': 1
            }
        }

        # Expressions dictionary
        self.exp_dict = {
            'H': lambda mdl: ca.sum1(mdl.x['lambda'] * ca.log(mdl.x['lambda'])) + mdl.tvp['MorH'],
            'H_prev': lambda mdl: ca.sum1(mdl.x['lambda_prev'] * ca.log(mdl.x['lambda_prev'])) + mdl.tvp['MorH_prev'],
            'y': (lambda mdl: g_casadi(map_g_single_casadi(self.g),mdl.tvp['MorXrobot_tree_lambda'].shape)(mdl.x['Xrobot',:2]+mdl.u['Xrobot',:2],
                                                                                                mdl.x['Xrobot',-1]+mdl.u['Xrobot',-1],
                                                                                                mdl.tvp['MorXrobot_tree_lambda'])) if True else \
                    (lambda mdl: 0.5 + trees_satisfy_conditions_casadi(
                        mdl.x['Xrobot', :2] + mdl.u['Xrobot', :2],
                        mdl.x['Xrobot', -1] + mdl.u['Xrobot', -1],
                        mdl.tvp['MorXrobot_tree_lambda']) * (self.g(
                        mdl.x['Xrobot', :2] + mdl.u['Xrobot', :2],
                        mdl.x['Xrobot', -1] + mdl.u['Xrobot', -1],
                        mdl.tvp['MorXrobot_tree_lambda']) - 0.5)),
            'drone_trees_dist': lambda mdl: drone_trees_distances_casadi(
                mdl.x['Xrobot', :2] + mdl.u['Xrobot', :2],
                mdl.tvp['MorXrobot_tree_obs']),
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
        self.x_k = {key: 0.5 * np.ones(shape) if key == 'lambda_prev' or  key == 'lambda' else np.zeros(shape)
                    for key, shape in self.state_dict['_x'].items()}
        self.x_k_full = {'lambda': 0.5 * np.ones((len(self.trees_pos), 1)),
                         'lambda_prev': 0.5 * np.ones((len(self.trees_pos), 1))}
        
        # Nearest trees
        self.update_nearest_trees()

    def update_nearest_trees(self):
        """Update nearest tree indices."""
        self.MorIdxsLambda = n_nearest_trees(self.x_k['Xrobot'][:2].reshape(1, -1), self.param['Xtree'], num=MOR_DIM_LAMBDA)
        self.NoMorIdxsLambda = np.setdiff1d(np.arange(self.param['lambda'].shape[0]), self.MorIdxsLambda)
        self.MorIdxsTree = n_nearest_trees(self.x_k['Xrobot'][:2].reshape(1, -1), self.param['Xtree'], num=MOR_DIM_OBS)

    def update(self, y_z):
        """Update the drone state."""
        self.update_nearest_trees()

        qi_z = ((1 + np.cos(y_z['gps'][-1])) / 15 + 0.5)
        
        self.x_k['Xrobot'] = np.array([y_z['gps'][0], y_z['gps'][1], y_z['gps'][-1]])
        self.x_k['Xrobot_tree'] = drone_trees_distances(self.x_k['Xrobot'][:2], self.param['Xtree'][self.MorIdxsTree])
        self.x_k['y'] = 0.5 + (qi_z - 0.5) * trees_satisfy_conditions_np(
            drone_pos=y_z['gps'][:2], 
            drone_yaw=y_z['gps'][-1], 
            tree_pos=np.stack((self.param['tree_x'][self.MorIdxsLambda], self.param['tree_y'][self.MorIdxsLambda]), axis=1))
        
        self.x_k_full['lambda_prev'] = self.x_k_full['lambda']
        self.x_k_full['lambda'][self.MorIdxsLambda] = bayes(self.x_k_full['lambda_prev'][self.MorIdxsLambda], self.x_k['y'])
        self.x_k['lambda_prev'] = self.x_k['lambda']
        self.x_k['lambda'] = self.x_k_full['lambda'][self.MorIdxsLambda]

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


class MockBridgeClass:
    def __init__(self):
        self.data = {
            "gps": np.zeros(3)
        }

    def callServer(self, request):
        if "trees_poses" in request:
            return {"trees_poses": np.random.rand(10, 2)}

    def pubData(self, u0):
        print(f"Published data: {u0}")

    def getData(self):
        self.data["gps"] += np.random.randn(3) * 0.1
        return self.data

# Updating MainClass to use generated tree positions and drone initial position
class MainClass:
    def __init__(self, viz=True):
        self.viz = viz
        self.bridge = MockBridgeClass()
        
        # Use generated tree positions
        self.state = DroneStateMor(model_ca='MX', model_type="discrete", g=load_g('gp'), trees_pos=tree_positions)
        self.state.populateModel()
        self.mpc = DroneMpc(self.state.model)
        
        self.u0 = {"cmd_pose": np.zeros((3, 1)), "viz_pred_pose": None}
        self.x_data = []

        # Set initial drone position
        self.bridge.data["gps"] = drone_position

        self.setUpMpc()
        self.runSimulation()

    def setUpMpc(self):
        """Setup the MPC with a time-varying parameter function and solver settings."""
        tvp_template = self.mpc.get_tvp_template()

        def tvp_fun(time_x):
            for k in range(len(tvp_template['_tvp', :, 'MorXrobot_tree_lambda'])):
                tvp_template['_tvp', k, 'MorXrobot_tree_obs'] = self.state.trees_pos[self.state.MorIdxsTree]
                tvp_template['_tvp', k, 'MorXrobot_tree_lambda'] = self.state.trees_pos[self.state.MorIdxsLambda]
                tvp_template['_tvp', k, 'MorH'] = np.sum(
                    self.state.param['lambda'][self.state.NoMorIdxsLambda] * np.log(self.state.param['lambda'][self.state.NoMorIdxsLambda]))
                tvp_template['_tvp', k, 'MorH_prev'] = (
                    tvp_template['_tvp', -1, 'MorH'] if k == 0 and time_x != 0 else tvp_template['_tvp', k, 'MorH'])
            return tvp_template

        self.mpc.set_tvp_fun(tvp_fun)
        self.mpc.settings.set_linear_solver(solver_name="MA27")
        self.mpc.settings.supress_ipopt_output()
        self.mpc.setup()
        self.plotter = MPCPlotter(self.mpc)

    def runSimulation(self):
        """Run the simulation loop."""
        for i in range(50):
            print('Step:', i)
            self.loop(i)
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

# Run the simulation
if __name__ == "__main__":
    main = MainClass()
