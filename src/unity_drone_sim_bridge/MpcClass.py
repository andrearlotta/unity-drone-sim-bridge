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
            'n_horizon': 10,
            't_step': .01,
            'n_robust': 1,
            'store_full_solution': True,
    }
    )

    def __post_init__(self):
        super().__init__(self.model)
        self.set_param(**self.setup_mpc)
        print(self.mterm(self.model))
        self.set_objective(mterm=self.mterm(self.model), lterm=self.lterm(self.model))
        if callable(self.rterm): self.set_rterm(self.rterm(self.model))
        for state_type, state_vars in self.bounds_dict.items():
            for state_var, bounds in state_vars.items():
                for lu, value in bounds.items():
                    print(lu,state_type,state_var)
                    if value is not None: self.bounds[lu,state_type,state_var] = value

