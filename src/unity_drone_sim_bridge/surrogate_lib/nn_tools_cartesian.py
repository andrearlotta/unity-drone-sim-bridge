import os
import re
import csv
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import tensorflow as tf
import casadi as ca
import l4casadi as l4c
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.io as pio
import wandb

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any

from unity_drone_sim_bridge.surrogate_lib.load_database import load_surrogate_database, generate_surrogate_synthetic_data, generate_fake_dataset
from unity_drone_sim_bridge.surrogate_lib.nn_models import *

# TensorFlow configuration
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters())

class SunDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

def LoadNN(hidden_size, hidden_layer, test_size=0.2, synthetic=False, rt=False, gpu=False, naive=False, use_yolo=True):
    EXPERIMENT_NAME = "SurrogateNetworkFixedOutput_hs64_hl2_lr0_001_e100_bs1_ts0_2_synFalse"
    #f"simple_mlp_hiddensize{hidden_size}_hiddenlayers{hidden_layer}_data{int(test_size*10)}" if not use_yolo else f"surrogate_model_hiddensize{hidden_size}_hiddenlayers{hidden_layer}"
    CHECKPOINT_PATH = find_best_model_with_highest_epoch(f"/home/pantheon/mpc-drone/checkpoints/{EXPERIMENT_NAME}")
    
    model = SurrogateNetworkFixedOutput(True, hidden_size, hidden_layer)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    print(f"Loading model from {CHECKPOINT_PATH}")

    print(count_parameters(model))
    return l4c.realtime.realtime_l4casadi.RealTimeL4CasADi(model, approximation_order=2, device="cuda" if gpu else "cpu") \
        if rt else \
            l4c.L4CasADi(model, model_expects_batch_dim=True, device="cuda" if gpu else "cpu")

def train_model(X, Y, model, train_loader, val_loader, criterion, optimizer, num_epochs, device, checkpoint_path):
    best_val_loss = float('inf')
    best_model_path = ""

    # Ensure the checkpoint path exists
    os.makedirs(f"./checkpoints/{checkpoint_path}", exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            wandb.log({'Training loss (batch)': loss.item()})
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            print(f"Epoch {epoch+1}/{num_epochs}, Step {i+1}/{len(train_loader)}, Loss: {loss.item()}")

        model.eval()
        with torch.no_grad():
            val_loss = 0
            for x_val_batch, y_val_batch in val_loader:
                x_val_batch, y_val_batch = x_val_batch.to(device), y_val_batch.to(device)
                val_outputs = model(x_val_batch)
                val_loss += criterion(val_outputs, y_val_batch)
            
            val_loss /= len(val_loader)
            wandb.log({'Validation loss': val_loss.item(), 'epoch': epoch})
            print(f"Validation loss: {val_loss}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = f"./checkpoints/{checkpoint_path}/best_model_epoch_{epoch+1}.ckpt"
                torch.save(model.state_dict(), best_model_path)

                print(f"Model saved at epoch {epoch+1}")
                model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy().flatten()

                pio.write_html(create_3d_plot(X, Y, model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy().flatten(), checkpoint_path), f'plots/{checkpoint_path}_3d_plot.html')
                wandb.log({'Predictions': wandb.Html(f'plots/{checkpoint_path}_3d_plot.html'), 'epoch': epoch})
                os.remove(f'plots/{checkpoint_path}_3d_plot.html')
        
        torch.save(model.state_dict(), f"./checkpoints/{checkpoint_path}/last_model.ckpt")

    return best_model_path, best_val_loss

def run_training_experiments(experiments: List[Dict[str, Any]]) -> None:
    results_file = 'experiment_results.csv'
    fieldnames = [
        'name', 'model', 'best_val_loss', 'test_loss', 'best_model_path',
        'hidden_size', 'hidden_layers', 'learning_rate', 'num_epochs',
        'batch_size', 'test_size', 'synthetic'
    ]

    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for exp in experiments:
        print(f"Running experiment: {exp['name']}")
        
        # Data preparation
        X, Y = load_surrogate_database()
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=exp['test_size'], shuffle=True)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, shuffle=True)

        train_dataset = SunDataset(x_train, y_train)
        val_dataset = SunDataset(x_val, y_val)
        test_dataset = SunDataset(x_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size=exp['batch_size'], shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=exp['batch_size'], shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=exp['batch_size'], shuffle=False)

        # Model setup
        if exp['model'] == 'SurrogateNetwork':
            model = SurrogateNetwork
        elif exp['model'] == 'AlternativeSurrogateNetwork':
            model = AlternativeSurrogateNetwork
        elif exp['model'] == 'SurrogateNetworkFixedOutput':
            model = SurrogateNetworkFixedOutput
        else:
            raise ValueError(f"Unknown model type: {exp['model']}")

        model = SimpleNetwork() if exp['model'] == 'SimpleNetwork' else model(exp['use_yolo'], exp['hidden_size'], exp['hidden_layers'])
        
        criterion = nn.L1Loss()
        optimizer = optim.Adam(model.parameters(), lr=exp['learning_rate'])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Initialize wandb run
        run = wandb.init(project='surrogate_model_training', name=exp['name'], config={
            'model': exp['model'],
            'synthetic': exp['synthetic'],
            'use_yolo': exp['use_yolo'],
            'test_size': exp['test_size'],
            'learning_rate': exp['learning_rate'],
            'num_epochs': exp['num_epochs'],
            'batch_size': exp['batch_size'],
            'hidden_size': exp['hidden_size'],
            'hidden_layers': exp['hidden_layers'],
        })

        best_model_path, best_val_loss = train_model(X, Y, model, train_loader, val_loader, criterion, optimizer, exp['num_epochs'], device, exp['name'])

        # Evaluation
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        test_loss = sum(criterion(model(x.to(device)), y.to(device)).item() for x, y in test_loader) / len(test_loader)

        # Generate predictions for the entire dataset
        with torch.no_grad():
            y_pred = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy().flatten()

        # Create and save 3D plot
        os.makedirs('plots', exist_ok=True)
        pio.write_html(create_3d_plot(X, Y, y_pred, exp['name']), file=f"plots/{exp['name']}_3d_plot.html")

        # Generate 500 random inputs
        random_inputs = np.array([
            [np.random.uniform(0, 10), np.random.uniform(-np.pi, np.pi), 0]
            for _ in range(500)
        ])
        random_inputs_tensor = torch.tensor(random_inputs, dtype=torch.float32).to(device)

        # Predict using the model
        prediction = model(random_inputs_tensor).detach().cpu().numpy().flatten()

        pio.write_html(create_3d_plot(random_inputs, [], prediction, exp['name']), f"plots/{exp['name']}_random_3d_plot.html")
        wandb.log({'Random Test': wandb.Html(f"plots/{exp['name']}_random_3d_plot.html")})
        os.remove(f"plots/{exp['name']}_random_3d_plot.html")

        # Write results to CSV
        result = {
            'name': exp['name'],
            'model': exp['model'],
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'best_model_path': best_model_path,
            'hidden_size': exp.get('hidden_size', 'N/A'),
            'hidden_layers': exp.get('hidden_layers', 'N/A'),
            'learning_rate': exp.get('learning_rate', 'N/A'),
            'num_epochs': exp.get('num_epochs', 'N/A'),
            'batch_size': exp.get('batch_size', 'N/A'),
            'test_size': exp.get('test_size', 'N/A'),
            'synthetic': exp.get('synthetic', 'N/A')
        }

        with open(results_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(result)
        run.finish()

    print(f"Results have been saved to {results_file}")

def random_input_test_rt(casadi_quad_approx_func, l4c_model_order2, gpu,):

    # Generate 500 random inputs
    random_inputs = []
    prediction = []
    torch_prediction = []
    for _ in range(1000):
        i_ = np.array([[np.random.uniform(0, 10), np.random.uniform(0, 2*np.pi), 0]])
        p_ = casadi_quad_approx_func(i_.reshape((3, 1)), l4c_model_order2.get_params(i_))
        print(p_)
        p_ = p_[0, 0]
        t_out = model(torch.tensor(i_, dtype=torch.float32).to("cuda" if gpu else "cpu")).detach().cpu().numpy().flatten()
        random_inputs.append(i_[0])
        torch_prediction.append(t_out)
        prediction.append(p_)

    pio.write_html(create_3d_plot(np.array(random_inputs), [], np.array(prediction).flatten(), 'test_l4casasdi'), "test_l4casasdi_random_3d_plot.html")
    pio.write_html(create_3d_plot(np.array(random_inputs), [], np.array(torch_prediction).flatten(), 'test_l4casasdi'), "test_l4casasdi_torch_random_3d_plot.html")

def random_input_test_std(l4c_model, model, gpu):
    # Generate 500 random inputs
    random_inputs = []
    prediction = []
    torch_prediction = []
    x_sym = ca.MX.sym('x', 3, 1)
    y_sym = l4c_model(x_sym)
    f = ca.Function('y', [x_sym], [y_sym])
    for _ in range(1000):
        i_ = np.array([[np.random.uniform(0, 10), np.random.uniform(0, 2*np.pi), 0]])
        p_ = f(i_.reshape((3, 1)))
        p_ = p_[0, 0]
        t_out = model(torch.tensor(i_, dtype=torch.float32).to("cuda" if gpu else "cpu")).detach().cpu().numpy().flatten()
        random_inputs.append(i_[0])
        torch_prediction.append(t_out)
        prediction.append(p_)

    pio.write_html(create_3d_plot(np.array(random_inputs), [], np.array(prediction).flatten(), 'test_l4casasdi'),       "test_l4casasdi_random_3d_plot.html")
    pio.write_html(create_3d_plot(np.array(random_inputs), [], np.array(torch_prediction).flatten(), 'test_l4casasdi'), "test_l4casasdi_torch_random_3d_plot.html")

def create_3d_plot(X_combined, Y_combined, y_pred, exp_name):
    fig = go.Figure()
    if len(Y_combined):
        fig.add_scatter3d(
            x=X_combined[:, 0],
            y=X_combined[:, 1],
            z=Y_combined,
            mode='markers',
            marker=dict(size=5, opacity=0.8),
            name="True"
        )
    fig.add_scatter3d(
        x=X_combined[:, 0],
        y=X_combined[:, 1],
        z=y_pred,
        mode='markers',
        marker=dict(size=5, opacity=0.8) if len(Y_combined) else dict(size=5, opacity=0.8, color=y_pred, colorscale='Viridis'),
        name="Predicted"
    )
    fig.update_layout(
        title=f'3D Scatter Plot of Values - {exp_name}',
        scene=dict(
            xaxis_title='X Coordinate',
            yaxis_title='Y Coordinate',
            zaxis_title='Conf Value'
        ),
        legend_title="Dataset"
    )
    return fig

def find_best_model_with_highest_epoch(folder_path):
    pattern = re.compile(r'best_model_epoch_(\d+)\.ckpt')
    return max(
        (os.path.join(folder_path, f) for f in os.listdir(folder_path) if pattern.match(f)),
        key=lambda f: int(pattern.search(f).group(1)),
        default=None
    )

# Main execution
if __name__ == '__main__':
    # Initialize wandb
    wandb.login()
    models = ['SurrogateNetworkFixedOutput']
    synthetic_options = [False]
    use_yolo_options = [True]
    test_sizes = [0.2]
    learning_rates = [1e-3]
    num_epochs_options = [50]
    batch_sizes = [1]
    hidden_sizes = [64]
    hidden_layers = [2]
    dataset_types = ['cartesian']

    experiments = []
    exp_counter = 1
    
    for dataset_type in dataset_types:
        for model in models:
            for synthetic in synthetic_options:
                for use_yolo in use_yolo_options:
                    for test_size in test_sizes:
                        for lr in learning_rates:
                            for epochs in num_epochs_options:
                                for batch_size in batch_sizes:
                                    for hidden_size in hidden_sizes:
                                        for hidden_layer in hidden_layers:
                                            exp = {
                                                'name': f'{dataset_type}_{model}_hs{hidden_size}_hl{hidden_layer}_lr{lr}_e{epochs}_bs{batch_size}_ts{test_size}_syn{synthetic}'.replace(".", "_"),
                                                'model': model,
                                                'synthetic': synthetic,
                                                'use_yolo': use_yolo,
                                                'test_size': test_size,
                                                'learning_rate': lr,
                                                'num_epochs': epochs,
                                                'batch_size': batch_size,
                                                'hidden_size': hidden_size,
                                                'hidden_layers': hidden_layer,
                                            }
                                            experiments.append(exp)
                                            exp_counter += 1

    for i, exp in enumerate(experiments[:5]):
        print(f"Experiment {i+1}:")
        for key, value in exp.items():
            print(f"  {key}: {value}")
        print()

    print(f"Total number of experiments: {len(experiments)}")

    run_training_experiments(experiments)

    print("Results have been saved to 'experiment_results.csv'")
    print("3D plots saved in the 'plots' directory as HTML files")
    print("TensorBoard logs saved in the 'runs' directory")
