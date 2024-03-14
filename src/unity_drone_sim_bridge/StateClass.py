from dataclasses import dataclass, field
from typing import List, Dict, Union
from casadi import *

@dataclass
class ModelState:
    model_type: str = field(default_factory='continuous')

    model: do_mpc.model.Model = field(default_factory=do_mpc.model.Model())

    x_k: Dict[str, List[float]] = field(default_factory=lambda: {
        'x1': [],
    })

    u_k: Dict[str, List[float]] = field(default_factory=lambda: {
        'u1': [],
    }) 
    
    y_k: Dict[str, List[float]] = field(default_factory=lambda: {
        'u1': [],
    }) 

    state_dict: Dict[str, List[str]] = field(default_factory=lambda: {
        '_x': ['x1'],
        '_u': ['u1'],
        'tvp': [],
        'aux': [],
        'parameters': [],
    })

    rsh_dict: Dict[str, callable] = field(default_factory=lambda: {
        'x1': lambda model: model.x['x1'] + model.u['u1'],
    })
    
    meas_dict: Dict[str, callable] = field(default_factory=lambda: {
        'y1': lambda model: model.x['x1'] + model.u['u1'],
    })

    def populateModel(self, model):
        # Assuming 'do_mpc' model creation
        self.model.model_type = self.model_type

        # Iteratively populate the model using the elements from state_dict
        for state_type, state_vars in self.state_dict.items():
            for var_name in state_vars:
                model.set_variable(var_type=state_type, var_name=var_name)
        for meas_name, meas_val in self.meas_dict.items():
            model.set_meas(var_type=meas_name, var_name=meas_val(model) if callable(meas_val) else meas_val)
        for state_var, rsh in self.rsh_dict.items():
            self.model.set_rsh(state_var, rsh)
        
        self.model.setup()


@dataclass
class ModelMpc:

    mpc: do_mpc.mpc.MPC = field(init=False)

    lterm: callable = field(default_factory=lambda model: None)
    
    rterm: callable = field(default_factory=lambda model: None)

    bounds: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=lambda: {
        '_x': {'x1': {'upper': None, 'lower': None}},
        '_u': {'u1': {'upper': None, 'lower': None}},
    })

    scaling: Dict[str, Dict[str, Dict[str, float]]]= field(default_factory=lambda: {
        '_x': {'x1': None},
    })

    def set_tvp_fun(self):
        tvp_template = self.mpc.get_tvp_template()

        def tvp_fun():
            return tvp_template
        
        return tvp_fun
    
    def __post_init__(self,model):
        self.mpc = do_mpc.mpc.MPC(model)

        setup_mpc = {
            'n_horizon': 20,
            't_step': 0.1,
            'n_robust': 1,
            'store_full_solution': True,
        }
        self.mpc.set_param(**setup_mpc)
        self.mpc.set_objective(mterm=self.mterm, lterm=self.lterm)

        for state_type, state_vars in self.bounds.items():
            for state_var, bounds in state_vars.items():
                for lu, value in bounds.items():
                    if value is not None: self.mpc.bounds[lu,state_type, state_var] = value

        self.mpc.set_tvp_fun(self.set_tvp_fun)

        self.mpc.setup()
