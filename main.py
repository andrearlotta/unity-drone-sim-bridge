from State import State
import do_mpc

class Main:
    def __init__(self):
        self.state = State()
        
    def __init_model(self):
        model_type = 'discrete' # either 'discrete' or 'continuous'
        self.model = do_mpc.model.Model(model_type)


def __init_mpc(model, silence_solver = False):
    """
    --------------------------------------------------------------------------
    mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    mpc.settings.n_horizon =  100
    mpc.settings.n_robust =  0
    mpc.settings.open_loop =  0
    mpc.settings.t_step =  0.04
    mpc.settings.state_discretization =  'collocation'
    mpc.settings.collocation_type =  'radau'
    mpc.settings.collocation_deg =  3
    mpc.settings.collocation_ni =  1
    mpc.settings.store_full_solution =  True

    if silence_solver:
        mpc.settings.supress_ipopt_output()


    mterm = None
    lterm = None # stage cost


    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(force=0.1)

    mpc.bounds['lower','_u','u'] = None
    mpc.bounds['upper','_u','u'] = None

    # Avoid the obstacles:
    mpc.set_nl_cons('obstacles', 
                    None, #-model.aux['obstacle_distance'], 
                    0)

    # Values for the masses (for robust MPC)
    #mpc.set_uncertainty_values()


    tvp_template = mpc.get_tvp_template()

    # When to switch setpoint:
    #t_switch = 4    # seconds
    #ind_switch = t_switch // mpc.settings.t_step

    def tvp_fun(t_ind): pass
    #    ind = t_ind // mpc.settings.t_step
    #    if ind <= ind_switch:
    #        tvp_template['_tvp',:, 'pos_set'] = -0.8
    #    else:
    #        tvp_template['_tvp',:, 'pos_set'] = 0.8
    #    return tvp_template

    mpc.set_tvp_fun(tvp_fun)

    mpc.setup()

    return mpc