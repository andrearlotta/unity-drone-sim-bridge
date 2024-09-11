from casadi import MX, vertcat, Function
import casadi as cs
from acados_template import AcadosSimSolver, AcadosOcpSolver, AcadosSim, AcadosOcp, AcadosModel
import numpy as np
from unity_drone_sim_bridge.surrogate_lib.surrogate_func_tools import *

# Define the drone model using CasADi
# Assume the state vector x = [x_pos, y_pos, z_pos, x_vel, y_vel, z_vel]
# Control vector u = [u1, u2, u3] where controls are accelerations in x, y, z axes
x_pos = MX.sym('x_pos')
y_pos = MX.sym('y_pos')
z_pos = MX.sym('z_pos')
x_vel = MX.sym('x_vel')
y_vel = MX.sym('y_vel')
z_vel = MX.sym('z_vel')

states = vertcat(x_pos, y_pos, z_pos)

u1 = MX.sym('u1')
u2 = MX.sym('u2')
u3 = MX.sym('u3')
controls = vertcat(u1, u2, u3)

# Define the system dynamics (simple double integrator model for the drone)
xdot = vertcat(x_vel, y_vel, z_vel)


# Create a function to describe the dynamics
f_expl = states + controls

# Define implicit dynamics (not used here, but placeholder)
f_impl = f_expl- xdot

# Define a neural network that outputs the measurements
# Let's assume you have a CasADi version of the neural network (you can import your own PyTorch model and use ONNX)
# Here we simulate the neural network as a CasADi function
measurements = MX.sym('measurements', 1)  # One measurement value to maximize


# Simulate the neural network as a CasADi function (e.g., using pre-trained weights)
# Replace this with your actual NN model import and integration
# Load the surrogate model
l4casadiNN = load_g('mlp', 
            rt=False, 
            gpu=False, 
            synthetic=True,
            hidden_layer=2,
            hidden_size=64, 
            use_yolo=True)


# Create the model object for ACADOS
model = cs.types.SimpleNamespace()
model.x = states
model.xdot = xdot
model.u = controls
model.z = l4casadiNN(states)  # Measurements from NN
model.f_expl = f_expl
model.f_impl = f_impl

# Define cost expressions
model.cost_y_expr = cs.vertcat(l4casadiNN(states), controls)
model.cost_y_expr_e = cs.vertcat(l4casadiNN(states))

# Set initial and final states (example placeholders)
x_start = np.array([10, 10, 10])
model.x_start = x_start
# Model name
model.name = "drone_model"


def acados_model(model):
    model_ac = AcadosModel()
    model_ac.f_impl_expr = model.xdot - model.f_expl
    model_ac.f_expl_expr = model.f_expl
    model_ac.x = model.x
    model_ac.xdot = model.xdot
    model_ac.u = model.u
    model_ac.name = model.name
    return model_ac
    
# Initialize OCP
ocp = AcadosOcp()

ocp.cost.cost_type = 'NONLINEAR_LS' # 'LINEAR_LS'

model_ac = acados_model(model=model)
# Set model dynamics
ocp.model = model_ac

# Set dimensions
nx = states.size1()  # Number of state variables
nu = controls.size1()  # Number of control variables
ny = nx + nu + 1  # Measurement plus all states and controls

ocp.dims.nx = nx
ocp.dims.nu = nu
ocp.dims.ny = ny
ocp.dims.ny_e = nx  # Terminal cost based only on the state

# Define cost function: maximize the measurements from the neural network
# The cost function should maximize the NN output
# Maximizing a function can be transformed into minimizing the negative of the function
ocp.model.cost_y_expr = vertcat(states, controls, -l4casadiNN(states))  # Minimize the negative of NN output
ocp.model.cost_y_expr_e = vertcat(-l4casadiNN(states))  # Terminal cost based on state
ocp.cost.yref_0 = np.zeros((ny, ))
# Set the weight for the cost function, give more weight to the NN output
W = np.diag([0]*nx + [0]*nu + [-1])  # No penalty on x, u; maximize NN output
W_e = np.array([[-1]])  # Terminal cost weights on state

ocp.cost.W = W
ocp.cost.W_e = W_e

# Define constraints (e.g., control limits)
u_min = np.array([-1, -1, -1])
u_max = np.array([1, 1, 1])

ocp.constraints.lbu = u_min
ocp.constraints.ubu = u_max
ocp.constraints.idxbu = np.array([0, 1, 2])


# Set initial state constraint
ocp.constraints.x0 = x_start  # Initial position and velocity

# Set prediction horizon
ocp.solver_options.tf = 1.0  # Time horizon of 1 second

# Solver options
ocp.solver_options.qp_solver = 'FULL_CONDENSING_HPIPM'  # QP solver
ocp.solver_options.nlp_solver_type = 'SQP_RTI'  # Real-time iteration scheme for NMPC
ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'  # Real-time iteration scheme for NMPC
ocp.solver_options.integrator_type = 'ERK'  # Implicit Runge-Kutta for integration

# Create OCP solver
ocp_solver = AcadosOcpSolver(ocp, json_file="acados_ocp.json")

# Solve the problem
status = ocp_solver.solve()

# Check status
if status != 0:
    print(f"OCP Solver returned status {status}. Check feasibility and constraints.")

# Get the optimized solution
u_opt = ocp_solver.get(0, "u")
x_opt = ocp_solver.get(0, "x")

print("Optimized control:", u_opt)
print("Optimized state:", x_opt)
