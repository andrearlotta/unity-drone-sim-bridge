from dataclasses import dataclass, field
from typing import List, Dict, Union, Callable,Any
from casadi import *
import do_mpc

@dataclass
class MpcClass(do_mpc.controller.MPC):
    model: do_mpc.model.Model

    lterm: Callable[[Any], Any] = field(default_factory= (lambda: None))
    
    mterm: Callable[[Any], Any] = field(default_factory= (lambda: None))

    rterm: Callable[[Any], Any] = field(default_factory= (lambda: None))
    
    bounds_dict: Dict[str, Dict[str, Dict[str, float]]] = field(default_factory=lambda: {
        '_x': {'x1': {'upper': None, 'lower': None}},
        '_u': {'u1': {'upper': None, 'lower': None}},
    })

    scaling: Dict[str, Dict[str,float]]= field(default_factory=lambda: {
        '_x': {'x1': None},
    })

    setup_mpc: dict = field(default_factory=lambda:{
            'n_horizon': 20,
            'n_robust': 0,
            'open_loop': 0,
            't_step': 1.0,
            'state_discretization': 'collocation',
            'collocation_type': 'radau',
            'collocation_deg': 2,
            'collocation_ni': 1,
            'store_full_solution': True,
            'nlpsol_opts': {'ipopt.linear_solver': 'MA27'}
    }
    )

    def __post_init__(self):
        super().__init__(self.model)
        self.set_param(**self.setup_mpc)
        self.set_objective(mterm=self.mterm(self.model), lterm=self.lterm(self.model))
        if callable(self.rterm): self.set_rterm(Xrobot=np.array(3*[1e-5])) #np.array(3*[1e-2]))#
        for state_type, state_vars in self.bounds_dict.items():
            for state_var, bounds in state_vars.items():
                for lu, value in bounds.items():
                    if value is not None: self.bounds[lu,state_type,state_var] = value(self.model) if callable(value) else  value

