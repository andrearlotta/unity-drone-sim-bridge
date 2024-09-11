import torch.nn as nn
from abc import ABC, abstractmethod
import torch

# Base Neural Network Class
class BaseNN(nn.Module, ABC):
    def __init__(self):
        super(BaseNN, self).__init__()
    
    @abstractmethod
    def forward(self, x):
        pass

# Neural Network Models
class SimpleNetwork(BaseNN):
    def __init__(self):
        super(SimpleNetwork, self).__init__()
        self.fc = nn.Linear(3, 1)
    
    def forward(self, x):
        return 0.5 * (torch.sigmoid(self.fc(x)) + 0.5)

class AlternativeSurrogateNetwork(nn.Module):
    def __init__(self, use_yolo, hidden_size, num_hidden_layers):
        super(AlternativeSurrogateNetwork, self).__init__()
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
        self.network = nn.Sequential(*layers)


    def forward(self, x):
        output = self.network(x)
        # Scale the sigmoid output to the range [0.5, 1.0]
        return output


class SurrogateNetworkFixedOutput(nn.Module):
    def __init__(self, use_yolo, hidden_size, num_hidden_layers):
        super(SurrogateNetworkFixedOutput, self).__init__()
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
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        output = self.network(x)
        # Scale the sigmoid output to the range [0.5, 1.0]
        return output
