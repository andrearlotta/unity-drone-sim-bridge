import numpy as np
import casadi as cs
import torch
import torch.nn as nn
import torch.nn.functional as F
import l4casadi as l4c
import os

class SimpleNetwork(nn.Module):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.fc = nn.Linear(3, 1)
    
    def forward(self, x):
        return nn.MulConstant(0.5,True)(nn.Sigmoid()(self.fc(x)) + 0.5)

class SurrogateNetwork(nn.Module):
    def __init__(self, use_yolo, hidden_size, num_hidden_layers):
        super(SurrogateNetwork, self).__init__()
        self.use_yolo = use_yolo
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        layers = []
        input_size = 3 if use_yolo else 1

        # Input layer
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())

        # Output layer
        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())  # Ensure output is in the range [0, 1]
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

        with torch.no_grad():
            self.network[-1].weight = torch.nn.Parameter(torch.tensor([[0.5]]))
            self.network[-1].bias = torch.nn.Parameter(torch.tensor([0.5]))
        # the tensor shape you assign should match the model parameter itself

        self.network[-1].requires_grad_(False)
        
    def forward(self, x):
        output = self.network(x)
        # Scale the sigmoid output to the range [0.5, 1.0]
        return output

def LoadNN(hidden_size, hidden_layer, test_size=0.2, n_inputs=3):
    EXPERIMENT_NAME = f"surrogate_model_hiddensize{hidden_size}_hiddenlayers{hidden_layer}"
    model = SurrogateNetwork(True, hidden_size, hidden_layer,)
    
    checkpoint_dir = "./checkpoints"
    experiment_path = os.path.join(checkpoint_dir, EXPERIMENT_NAME)

    best_model_files = [os.path.join(experiment_path, f) for f in os.listdir(experiment_path) if f.startswith("best_model_") and f.endswith('.ckpt')]
    if not best_model_files:
        raise FileNotFoundError("No best model file found.")
    
    print(f"Available model files: {best_model_files}")
    CHECKPOINT_PATH = max(best_model_files, key=os.path.getctime)
    print(f"Loading model from: {CHECKPOINT_PATH}")
    
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location='cuda'))
    model.eval()
    
    return model

pyTorch_model =LoadNN(64, 2, test_size=0.2, use_yolo=False)
l4c_model = l4c.L4CasADi(pyTorch_model, model_expects_batch_dim=True, device='cuda')  # device='cuda' for GPU

x_sym = cs.MX.sym('x', 3, 1)
print(f"x_sym: {x_sym.shape}")
y_sym = l4c_model(x_sym)
print(f"y_sym: {y_sym.shape}")

x = cs.DM([[5.], [2.], [2.]])
print(f"x: {x.shape}")

f = cs.Function('y', [x_sym], [y_sym])
df = cs.Function('dy', [x_sym], [cs.jacobian(y_sym, x_sym)])
ddf = cs.Function('ddy', [x_sym], [cs.hessian(y_sym, x_sym)[0]])


try:
    print(l4c_model(x))
    print(f(x))
    print(df(x))
    print('....')
    print(ddf(x))
except RuntimeError as e:
    print(f"Runtime error: {e}")
