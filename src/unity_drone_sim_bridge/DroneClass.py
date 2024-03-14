from unity_drone_sim_bridge.nn_lib.NeuralClass import NeuralClass
from unity_drone_sim_bridge.BridgeClass import BridgeClass
from unity_drone_sim_bridge.sensors import SENSORS
from unity_drone_sim_bridge.StateClass import ModelState
from unity_drone_sim_bridge.nn_lib.tools import *

class DroneState(ModelState):
    def __init__(self):
        '''
        overwritten elemets
        '''

        self.state_dict= {
        '_x':   ['x1'],
        '_u':   ['u1'],
        'tvp':  ['target_pos'],
        'aux':  [],
        'parameters': [],
        }

        self.rsh_dict= {
            'x1': lambda model: model.x['x1'] - model.tvp['target_pos'] + model.u['u1'],
        }

        self.meas_dict ={
            'y1': lambda model: model.tvp['target_pos'],
        }
        
        '''
        added elemets
        '''
        self.neural_net = NeuralClass()
        self.list_qi_angle = []

    def measPredict(self, bridge):
       
        if all(y != bridge['pose'] for y in self.y_k['pose']):
            self.y_k['qi'] += img_2_qi(bridge['image'])
            self.y_k['pose'] += bridge['pose']

        [A, F, P, O] = self.neural_net.runPredict(np.asarray([[self.y_k['qi'],self.y_k['pose']]]).reshape(-1,-1,2))[0]
        max_pos = (np.pi/ 2 - P) / F
        max_arg = wrapped_difference(self.y_k['pose'][-1], max_pos)
        