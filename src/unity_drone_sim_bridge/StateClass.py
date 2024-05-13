from dataclasses import dataclass, field
from typing import List, Dict, Union
import do_mpc
from casadi import *

@dataclass
class StateClass:
    model_type: str = field(default='continuous')
    model_ca: str = field(default='MX')
    state_dict: Dict[str, List[str]] = field(default_factory=lambda: {
        #'_x': ['x1'],
        #'_u': ['u1'],
        #'tvp': [],
        #'aux': [],
        #'parameters': [],
    })

    rsh_dict: Dict[str, callable] = field(default_factory=lambda: {
        #'x1': lambda model: model.x['x1'] + model.u['u1'],
    })
    
    exp_dict: Dict[str, callable] = field(default_factory=lambda: {
        #'x1': lambda model: model.x['x1'] + model.u['u1'],
    })
    
    meas_dict: Dict[str, callable] = field(default_factory=lambda: {
        #'y1': lambda model: model.x['x1'] + model.u['u1'],
    })

    model: do_mpc.model.Model = field(init=False)
    
    def __post_init__(self):
        self.model = do_mpc.model.Model(self.model_type, self.model_ca)

    def populateModel(self):
        # Assuming 'do_mpc' model creation
        self.model.model_type = self.model_type
        # Iteratively populate the model using the elements from state_dict
        for state_type, state_vars in self.state_dict.items():
            for var_name, dim_ in state_vars.items():
                print(state_type, var_name, dim_)
                self.model.set_variable(state_type, var_name, dim_)
        for meas_name, meas_val in self.meas_dict.items():
            self.model.set_meas(var_type=meas_name, var_name=meas_val(self.model) if callable(meas_val) else meas_val)
        for state_var, rsh in self.rsh_dict.items():
            self.model.set_rhs(state_var, rsh(self.model))
        for exp_name, exp in self.exp_dict.items():
            self.model.set_expression(exp_name, exp(self.model))
        self.model.setup()

    def updateState(y_k):
        return y_k