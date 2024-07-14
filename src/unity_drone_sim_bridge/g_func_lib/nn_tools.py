import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import train_test_split
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from abc import ABC, abstractmethod
from typing import List, Dict, Any
import plotly.graph_objects as go
import plotly.io as pio
import random
import csv
from unity_drone_sim_bridge.g_func_lib.load_database import *

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

class SurrogateNetwork(BaseNN):
    def __init__(self, use_yolo, hidden_size, num_hidden_layers):
        super(SurrogateNetwork, self).__init__()
        self.use_yolo = use_yolo
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers

        layers = []
        input_size = 3 if use_yolo else 1

        layers.append(nn.Linear(input_size, hidden_size))
 
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.GELU())

        layers.append(nn.Linear(hidden_size, 1))
        layers.append(nn.Sigmoid())
        layers.append(nn.Linear(hidden_size, 1))
        self.network = nn.Sequential(*layers)

        with torch.no_grad():
            self.network[-1].weight = nn.Parameter(torch.tensor([[0.5]]))
            self.network[-1].bias = nn.Parameter(torch.tensor([0.5]))

        self.network[-1].requires_grad_(False)
        
    def forward(self, x):
        return self.network(x)


# Create dataloaders
class SunDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        

# Utility Function
def load_database(DATA_PATH = "/home/pantheon/dataset_apples_2/FilteredCameraPositions.csv"):
    X = []
    Y = []

    with open(DATA_PATH, 'r') as infile:
        data = csv.reader(infile)
        next(data)  # Skip the header
        
        for i, row in enumerate(data):
            phi, rho, yaw, value = map(float, row)
            X.append([rho, phi, yaw])
            Y.append(value)
    
    return np.array(X), np.array(Y)

# Generate synthetic data
def generate_synthetic_data(X, num_samples):
    rho_min, rho_max = X[:, 0].min(), X[:, 0].max()
    phi_min, phi_max = X[:, 1].min(), X[:, 1].max()
    yaw_values = [30, -30, 60, -60, 90, -90, 180, -180]
    
    synthetic_X = []
    synthetic_Y = []

    for _ in range(num_samples):
        rho = random.uniform(rho_min, rho_max)
        phi = random.uniform(phi_min, phi_max)
        yaw = np.radians(random.choice(yaw_values) + random.uniform(-5, +5))
        value = 0.5

        synthetic_X.append([rho, phi, yaw])
        synthetic_Y.append(value)
    return np.array(synthetic_X), np.array(synthetic_Y)


# Training Function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, writer, checkpoint_path):
    best_val_loss = float('inf')
    best_model_path = ""

    for epoch in range(num_epochs):
        model.train()
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            writer.add_scalar('training loss', loss.item(), epoch*len(train_loader) + i)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(train_loader)}, Loss: {loss.item()}")

        model.eval()
        with torch.no_grad():
            val_loss = sum(criterion(model(x.to(device)), y.to(device)).item() for x, y in val_loader)
            val_loss /= len(val_loader)
            writer.add_scalar('validation loss', val_loss, epoch)
            print(f"Validation loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = f"{checkpoint_path}/best_model_epoch_{epoch+1}.ckpt"
                torch.save(model.state_dict(), best_model_path)
                print(f"Model saved at epoch {epoch+1}")

        torch.save(model.state_dict(), f"{checkpoint_path}/last_model.ckpt")

    return best_model_path, best_val_loss


# Main Training Loop
def run_training_experiments(experiments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []

    for exp in experiments:
        print(f"Running experiment: {exp['name']}")
        
        # Data preparation
        X, Y = load_database()
        if exp['synthetic']:
            synthetic_X, synthetic_Y = generate_synthetic_data(X, int(0.3 * len(X)))
            X = np.vstack((X, synthetic_X))
            Y = np.hstack((Y, synthetic_Y))

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=exp['test_size'], shuffle=True)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

        train_dataset = SunDataset(x_train, y_train)
        val_dataset = SunDataset(x_val, y_val)
        test_dataset = SunDataset(x_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=exp['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=exp['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=exp['batch_size'], shuffle=False)

        # Model setup
        if exp['model'] == 'SimpleNetwork':
            model = SimpleNetwork()
        elif exp['model'] == 'SurrogateNetwork':
            model = SurrogateNetwork(exp['use_yolo'], exp['hidden_size'], exp['hidden_layers'])
        else:
            raise ValueError(f"Unknown model type: {exp['model']}")

        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=exp['learning_rate'])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Training
        writer = SummaryWriter(f"runs/{exp['name']}")
        checkpoint_path = f"./checkpoints/{exp['name']}"
        os.makedirs(checkpoint_path, exist_ok=True)

        best_model_path, best_val_loss = train_model(
            model, train_loader, val_loader, criterion, optimizer, 
            exp['num_epochs'], device, writer, checkpoint_path
        )

        # Evaluation
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        test_loss = sum(criterion(model(x.to(device)), y.to(device)).item() for x, y in test_loader) / len(test_loader)

        # Generate predictions for the entire dataset
        with torch.no_grad():
            y_pred = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy().flatten()

        # Create and save 3D plot
        os.makedirs('plots', exist_ok=True)
        create_3d_plot(X, Y, y_pred, exp['name'])

        results.append({
            'name': exp['name'],
            'model': exp['model'],
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'best_model_path': best_model_path
        })

        writer.close()

    return results


# Function to create and save 3D plot
def create_3d_plot(X_combined, Y_combined, y_pred, exp_name):
    fig = go.Figure(data=[
        go.Scatter3d(
            x=X_combined[:, 0] * np.cos(X_combined[:, 1]),
            y=X_combined[:, 0] * np.sin(X_combined[:, 1]),
            z=Y_combined,
            mode='markers',
            marker=dict(
                size=5,
                opacity=0.8
            ),
            name="True"
        ),
        go.Scatter3d(
            x=X_combined[:, 0] * np.cos(X_combined[:, 1]),
            y=X_combined[:, 0] * np.sin(X_combined[:, 1]),
            z=y_pred,
            mode='markers',
            marker=dict(
                size=5,
                opacity=0.8
            ),
            name="Predicted"
        )
    ])
    
    fig.update_layout(
        title=f'3D Scatter Plot of Values - {exp_name}',
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Conf Value'
        ),
        legend_title="Dataset"
    )
    
    # Save the plot as an HTML file
    pio.write_html(fig, file=f'plots/{exp_name}_3d_plot.html')


# Function to create results table as a figure
def create_results_table(results):
    fig, ax = plt.subplots(figsize=(12, len(results) * 0.5))
    ax.axis('tight')
    ax.axis('off')
    
    table_data = [[result['name'], result['model'], f"{result['best_val_loss']:.6f}", f"{result['test_loss']:.6f}"] for result in results]
    table = ax.table(cellText=table_data, 
                     colLabels=['Name', 'Model', 'Best Val Loss', 'Test Loss'],
                     cellLoc='center', loc='center')
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.5)
    
    plt.title('Experiment Results')
    plt.savefig('results_table.png', bbox_inches='tight', dpi=300)
    plt.close()


# Main execution
if __name__ == '__main__':
    # Define the parameter ranges
    models = ['SurrogateNetwork']
    synthetic_options = [False]
    use_yolo_options = [True]
    test_sizes = [0.2]
    learning_rates = [0.001, 0.0001, 0.00001]
    num_epochs_options = [100, 200, 300, 500]
    batch_sizes = [1,5,10]
    hidden_sizes = [64, 128, 256]
    hidden_layers = [2, 3, 4]

    experiments = []
    exp_counter = 1

    for model in models:
        for synthetic in synthetic_options:
            for use_yolo in use_yolo_options:
                for test_size in test_sizes:
                    for lr in learning_rates:
                        for epochs in num_epochs_options:
                            for batch_size in batch_sizes:
                                exp = {
                                    'name':  f'{model}_lr{lr}_e{epochs}_bs{batch_size}_ts{test_size}_syn{synthetic}.html',
                                    'model': model,
                                    'synthetic': synthetic,
                                    'use_yolo': use_yolo,
                                    'test_size': test_size,
                                    'learning_rate': lr,
                                    'num_epochs': epochs,
                                    'batch_size': batch_size
                                }
                                
                                if model == 'SurrogateNetwork':
                                    for hidden_size in hidden_sizes:
                                        for hidden_layer in hidden_layers:
                                            exp_surrogate = exp.copy()
                                            exp_surrogate['hidden_size'] = hidden_size
                                            exp_surrogate['hidden_layers'] = hidden_layer
                                            exp_surrogate['name'] = f'{model}_hs{hidden_size}_hl{hidden_layer}_lr{lr}_e{epochs}_bs{batch_size}_ts{test_size}_syn{synthetic}'.replace(".", "_") + '.html'
                                            experiments.append(exp_surrogate)
                                            exp_counter += 1
                                else:
                                    experiments.append(exp)
                                    exp_counter += 1

    # Print the first few experiments to verify
    for i, exp in enumerate(experiments[:5]):
        print(f"Experiment {i+1}:")
        for key, value in exp.items():
            print(f"  {key}: {value}")
        print()

    print(f"Total number of experiments: {len(experiments)}")

    results = run_training_experiments(experiments)

    # Create and save results table as a figure
    create_results_table(results)
    print("Results table saved as 'results_table.png'")
    print("3D plots saved in the 'plots' directory as HTML files")