import os
import csv
import numpy as np
import torch as torch
import torch.nn as nn
import torch.optim as optim
import tensorflow as tf
import l4casadi as l4c
import plotly.io as pio
import wandb

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from typing import List, Dict, Any

from unity_drone_sim_bridge.surrogate_lib.load_database import *
from unity_drone_sim_bridge.surrogate_lib.nn_models import *
from unity_drone_sim_bridge.surrogate_lib.plot_tools import *

# TensorFlow configuration
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def find_best_model_with_highest_epoch(folder_path):
    pattern = re.compile(r'best_model_epoch_(\d+)\.ckpt')
    return max(
        (os.path.join(folder_path, f) for f in os.listdir(folder_path) if pattern.match(f)),
        key=lambda f: int(pattern.search(f).group(1)),
        default=None
    )

def LoadNN(hidden_size, hidden_layer, test_size=0.2, synthetic=False, rt=False, gpu=False, n_inputs=3, cartesian=True):
    """
    Carica una rete neurale già allenata (SurrogateNetworkFixedOutput)
    """
    model_name = 'SurrogateNetworkFixedOutput'
    lr = 0.0001
    epochs = 1000
    batch_size = 2
    test_size = 0.3
    augmentation = False
    EXPERIMENT_NAME = f"{'cartesian' if cartesian else 'polar'}_{model_name}_in_{n_inputs}_hs{hidden_size}_hl{hidden_layer}_lr{lr}_e{epochs}_bs{batch_size}_ts{test_size}_syn{synthetic}_augmentation{augmentation}".replace(".", "_")
    EXPERIMENT_NAME = "cartesian_SurrogateNetworkFixedOutput_in_3_hs16_hl2_lr0_0001_e250_bs1_ts0_3_synTrue_augmentationFalse"
    CHECKPOINT_PATH = find_best_model_with_highest_epoch(f"/home/pantheon/mpc-drone/checkpoints/{EXPERIMENT_NAME}")

    model = SurrogateNetworkFixedOutput(n_inputs, hidden_size, hidden_layer)
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()

    print(f"Loading model from {CHECKPOINT_PATH}")
    print(f"Number of parameters: {count_parameters(model)}")

    return model

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

                pio.write_html(create_3d_plot(X, Y, model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy().flatten(), checkpoint_path, is_polar=(exp['dataset_type'] == 'polar')), f'plots/{checkpoint_path}_3d_plot.html')
                wandb.log({'Predictions': wandb.Html(f'plots/{checkpoint_path}_3d_plot.html'), 'epoch': epoch})
                os.remove(f'plots/{checkpoint_path}_3d_plot.html')
        
        torch.save(model.state_dict(), f"./checkpoints/{checkpoint_path}/last_model.ckpt")

    return best_model_path, best_val_loss

def run_training_experiments(experiments: List[Dict[str, Any]]) -> None:
    results_file = 'experiment_results.csv'
    fieldnames = [
        'name', 'model', 'best_val_loss', 'test_loss', 'best_model_path',
        'hidden_size', 'hidden_layers', 'learning_rate', 'num_epochs',
        'batch_size', 'test_size', 'synthetic', 'dataset_type', 'n_input', 'augmentation'
    ]

    with open(results_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for exp in experiments:
        print(f"Running experiment: {exp['name']}")
        
        if exp['synthetic']:
            X, Y = generate_fake_dataset(10000, is_polar= exp['dataset_type'] == 'polar', n_input=exp["n_input"])
        else:
            X, Y = load_surrogate_database(exp['dataset_type'] == 'polar', exp['n_input'], test_size=1.0)
        if exp['augmentation']:
            augmentation_X, augmentation_Y = generate_surrogate_augmented_data(X, int(0.3 * len(X)), exp['dataset_type'] == 'polar', n_input=exp["n_input"])

        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=exp['test_size'], shuffle=True)
        x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=exp['test_size'], shuffle=True)
        if exp['augmentation']:
            augmentation_x_train, augmentation_x_test, augmentation_y_train, augmentation_y_test = train_test_split(augmentation_X, augmentation_Y, test_size=exp['test_size'], shuffle=True)
            augmentation_x_train, augmentation_x_val, augmentation_y_train, augmentation_y_val = train_test_split(x_train, y_train, test_size=exp['test_size'], shuffle=True)
            x_train = np.vstack((x_train, augmentation_x_train))
            x_val = np.vstack((x_val, augmentation_x_val))
            y_train = np.hstack((y_train, augmentation_y_train))
            y_val = np.hstack((y_val, augmentation_y_val))
            x_test = np.vstack((x_test, augmentation_x_test))
            y_test = np.hstack((y_test, augmentation_y_test))

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

        model = SimpleNetwork() if exp['model'] == 'SimpleNetwork' else model(exp['n_input'], exp['hidden_size'], exp['hidden_layers'])
        
        criterion = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=exp['learning_rate'])

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        # Initialize wandb run
        run = wandb.init(project='NMPC_surrogate', name=exp['name'], config={
            'dataset_type': exp['dataset_type'],
            'model': exp['model'],
            'synthetic': exp['synthetic'],
            'augmentation': exp['augmentation'],
            'n_input': exp['n_input'],
            'test_size': exp['test_size'],
            'learning_rate': exp['learning_rate'],
            'num_epochs': exp['num_epochs'],
            'batch_size': exp['batch_size'],
            'hidden_size': exp['hidden_size'],
            'hidden_layers': exp['hidden_layers'],
        })

        """
            Train
        """
        best_model_path, best_val_loss = train_model(X, Y, model, train_loader, val_loader, criterion, optimizer, exp['num_epochs'], device, exp['name'])
        
        """
            Eval
        """
        model.load_state_dict(torch.load(best_model_path))
        model.eval()
        test_loss = sum(criterion(model(x.to(device)), y.to(device)).item() for x, y in test_loader) / len(test_loader)

        # Generate predictions for the entire dataset
        with torch.no_grad():
            y_pred = model(torch.tensor(X, dtype=torch.float32).to(device)).cpu().numpy().flatten()

        """
            Create and save 3D plot
        """
        os.makedirs('plots', exist_ok=True)
        pio.write_html(create_3d_plot(X, Y, y_pred, exp['name'], is_polar=(exp['dataset_type'] == 'polar')), file=f"plots/{exp['name']}_3d_plot.html")

        # Generate 500 random inputs
        random_inputs = np.array([
            [np.random.uniform(0, 10), np.random.uniform(-np.pi, np.pi), 0] if exp['dataset_type'] == 'polar' else [np.random.uniform(-10, 10), np.random.uniform(-10, 10), 0]
            for _ in range(500)
        ])
        if exp['n_input'] == 2 :  random_inputs = random_inputs[:,:2]
        random_inputs_tensor = torch.tensor(random_inputs, dtype=torch.float32).to(device)

        # Predict using the model
        prediction = model(random_inputs_tensor).detach().cpu().numpy().flatten()

        pio.write_html(create_3d_plot(random_inputs, [], prediction, exp['name'], is_polar=(exp['dataset_type'] == 'polar')), f"plots/{exp['name']}_random_3d_plot.html")
        wandb.log({'Random Test': wandb.Html(f"plots/{exp['name']}_random_3d_plot.html")})
        os.remove(f"plots/{exp['name']}_random_3d_plot.html")

        # Write results to CSV
        result = {
            'name': exp['name'],
            'dataset_type': exp.get('dataset_type','N/A'),
            'model': exp.get('model','N/A'),
            'best_val_loss': best_val_loss,
            'test_loss': test_loss,
            'best_model_path': best_model_path,
            'hidden_size': exp.get('hidden_size', 'N/A'),
            'hidden_layers': exp.get('hidden_layers', 'N/A'),
            'learning_rate': exp.get('learning_rate', 'N/A'),
            'num_epochs': exp.get('num_epochs', 'N/A'),
            'batch_size': exp.get('batch_size', 'N/A'),
            'test_size': exp.get('test_size', 'N/A'),
            'synthetic': exp.get('synthetic', 'N/A'),
            'augmentation': exp.get('augmentation', 'N/A')
        }

        with open(results_file, 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(result)
        run.finish()

    print(f"Results have been saved to {results_file}")

# Main execution
if __name__ == '__main__':
    # Initialize wandb
    wandb.login()
    models = ['SurrogateNetworkFixedOutput']
    synthetic_options = [False]
    augmentation_options = [True]
    n_input_options = [3]
    test_sizes = [0.2]
    learning_rates = [1e-5]  # Reduce learning rate
    batch_sizes = [2]  # Larger batch size
    num_epochs_options = [250]  # Increase the number of epochs, but use early stopping
    hidden_sizes = [8]
    hidden_layers = [3]
    dataset_types = ['cartesian']

    experiments = []
    exp_counter = 1
    
    for dataset_type in dataset_types:
        for model in models:
            for synthetic in synthetic_options:
                for augmentation in augmentation_options:
                    for n_input in n_input_options:
                        for test_size in test_sizes:
                            for lr in learning_rates:
                                for epochs in num_epochs_options:
                                    for batch_size in batch_sizes:
                                        for hidden_size in hidden_sizes:
                                            for hidden_layer in hidden_layers:
                                                exp = {
                                                    'name': f'{dataset_type}_{model}_in_{n_input}_hs{hidden_size}_hl{hidden_layer}_lr{lr}_e{epochs}_bs{batch_size}_ts{test_size}_syn{synthetic}_augmentation{augmentation}'.replace(".", "_"),
                                                    'model': model,
                                                    'dataset_type': dataset_type,
                                                    'synthetic': synthetic,
                                                    'augmentation': augmentation,
                                                    'n_input': n_input,
                                                    'test_size': test_size,
                                                    'learning_rate': lr,
                                                    'num_epochs': epochs,
                                                    'batch_size': batch_size,
                                                    'hidden_size': hidden_size,
                                                    'hidden_layers': hidden_layer
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
