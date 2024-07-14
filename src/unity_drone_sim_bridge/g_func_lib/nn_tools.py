import torch
import torch.nn as nn
import torch.nn.functional as F
from unity_drone_sim_bridge.g_func_lib.gp_tools import loadDatabase, loadSyntheticDatabase
import l4casadi as l4c
import os 
import numpy as np
import tensorflow as tf
import casadi as ca
import random
import csv
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


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
        layers.append(nn.GELU())

        # Hidden layers
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.GELU())

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

    
# Create dataloaders
class SunDataset(torch.utils.data.Dataset):
    def __init__(self, x, y):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]
        
def LoadNN(hidden_size,hidden_layer,test_size=0.2, synthetic=False, rt=True, gpu=True, naive=False, use_yolo=False):

        EXPERIMENT_NAME = f"simple_mlp_hiddensize{hidden_size}_hiddenlayers{hidden_layer}_data{int(test_size*10)}" if not use_yolo else f"surrogate_model_hiddensize{hidden_size}_hiddenlayers{hidden_layer}"
        if synthetic: EXPERIMENT_NAME += '_synthetic'
        model = l4c.naive.MultiLayerPerceptron(
        in_features = 3 if use_yolo else 1,
        hidden_features = hidden_size,
        out_features = 1,
        hidden_layers = hidden_layer)if naive \
        else SurrogateNetwork(use_yolo, hidden_size, hidden_layer)
        
        """torch.nn.Sequential(
            torch.nn.Linear(3 if use_yolo else 1, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, 1),
        )"""
        #activation = 'GELU'
        CHECKPOINT_PATH = f"/home/pantheon/mpc-drone/checkpoints/{EXPERIMENT_NAME}/last_model.ckpt"
        model.load_state_dict(torch.load(CHECKPOINT_PATH))
        model.eval()
        print(f"Loading model from {CHECKPOINT_PATH}")
        if rt:
            l4c_model = l4c.realtime.realtime_l4casadi.RealTimeL4CasADi(model, approximation_order=1, device="cuda" if gpu else "cpu")
            x_sym = ca.MX.sym('x',3,1)
            y_sym = l4c_model(x_sym)
            casadi_func = ca.Function('model_rt_approx',
                                    [x_sym, l4c_model.get_sym_params()],
                                    [y_sym])

            x = np.ones([ 3 if use_yolo else 1])  # torch needs batch dimension
            casadi_param = l4c_model.get_params(x)
            casadi_out = casadi_func(x, casadi_param)  # transpose for vector rep. expected by casadi
            f =lambda x: casadi_func(x, l4c_model.get_params(np.ones([ 3 if use_yolo else 1])))
        else:
            l4c_model = l4c.L4CasADi(model, model_expects_batch_dim=True, device="cuda" if gpu else "cpu")
            x_sym = ca.MX.sym('x',3,1)
            y_sym = l4c_model(x_sym) 
            f = ca.Function('y', [x_sym], [y_sym])
            x = ca.DM([[0.]] * 3 if use_yolo else 1) 
            print(l4c_model(x)) 
        return f

def TrainQiNN(test_size=0.2, input_size = 3, hidden_layer=2, hidden_size = 256, output_size = 1, learning_rate = 0.0001, num_epochs = 500, batch_size = 1, synthetic=False, naive=True):
    if synthetic:
        x, y, _, _ = loadSyntheticDatabase(N=1081)
    else:
        x, y = loadDatabase()

    # Split data
    from sklearn.model_selection import train_test_split
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.98, shuffle=True)

    import torch
    # Create batches to do inference in the whole dataset
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)


    train_dataset = SunDataset(x_train, y_train)
    val_dataset = SunDataset(x_val, y_val)
    test_dataset = SunDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()

    # Model
    print(
        input_size, hidden_size, 1, hidden_layer,
    )

    model = l4c.naive.MultiLayerPerceptron(
        in_features = 1,
        hidden_features = hidden_size,
        out_features = 1,
        hidden_layers = hidden_layer,
        activation = 'GELU') if  naive \
    else torch.nn.Sequential(
            torch.nn.Linear(1, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size, 1),
        )


    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    EXPERIMENT_NAME = f"simple_mlp_hiddensize{hidden_size}_hiddenlayers{hidden_layer}_data{int(test_size*10)}"
    if synthetic: EXPERIMENT_NAME += '_synthetic'
    CHECKPOINT_PATH = f"./checkpoints/{EXPERIMENT_NAME}"
    # Create checkpoint path if it doesn't exist
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    BEST_MODEL_PATH = ""
    BEST_MODEL_LOSS = np.inf

    # Move everything to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion.to(device)
    x_tensor = x_tensor.to(device)

    # Create tensorboard writer
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    # Best validation loss
    best_val_loss = np.inf

    for epoch in range(num_epochs):
        model.train()
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            writer.add_scalar('training loss', loss.item(), epoch*len(train_loader) + i)

            # Backward pass
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
            writer.add_scalar('validation loss', val_loss/len(val_loader), epoch)
            print(f"Validation loss: {val_loss/len(val_loader)}")

            if val_loss/len(val_loader) < best_val_loss:
                best_val_loss = val_loss
                # Remove previous best model if exists
                if BEST_MODEL_PATH != "":
                    os.remove(BEST_MODEL_PATH)
                BEST_MODEL_PATH = f"{CHECKPOINT_PATH}/best_model_epoch_{epoch+1}.ckpt"
                BEST_MODEL_LOSS = val_loss/len(val_loader)
                torch.save(model.state_dict(), BEST_MODEL_PATH)

                print(f"Model saved at epoch {epoch+1}")

        # Save last model
        torch.save(model.state_dict(), f"{CHECKPOINT_PATH}/last_model.ckpt")
        
        # Scatter in three plots the data
        import matplotlib.pyplot as plt
        # Log an image with a prediction in the whole dataset
        y_pred = model(x_tensor).cpu().detach().squeeze().numpy()
        fig = plt.figure()
        plt.scatter(x, y, label="True")
        plt.scatter(x, y_pred, label="Predicted")
        plt.legend()
        writer.add_figure('predictions', fig, epoch)

    writer.close()

    # Print results
    print("Finished training")
    print(f"Best model saved at {BEST_MODEL_PATH}")
    print(f"Best model loss: {BEST_MODEL_LOSS}")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()

    # Load best model
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    print(f"Loading best model from {BEST_MODEL_PATH}")
    
    # Test on the whole dataset and plot:
    # - Training data
    # - Training data prediction
    # - Whole dataset prediction

    import matplotlib.pyplot as plt

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        y_pred = model(x_tensor).cpu().detach().squeeze().numpy()
        y_train_pred = model(x_train_tensor).cpu().detach().squeeze().numpy()

        plt.figure()
        plt.plot(x, y, label="True data")
        plt.plot(x, y_pred, label="Prediction")
        plt.scatter(x_train, y_train, label="Training data")
        plt.scatter(x_train, y_train_pred, label="Training data prediction")
        plt.legend()
        plt.savefig(f'{CHECKPOINT_PATH}/{EXPERIMENT_NAME}_best_pred.png')
    return model

def TrainSurrogate(input_size = 3, hidden_layer=2, hidden_size = 64, output_size = 1, learning_rate = 0.0001, num_epochs = 500, batch_size = 1, naive=False):
    DATA_PATH = "/home/pantheon/dataset_apples_2/FilteredCameraPositions.csv"
# Load the dataset from CSV
    def load_database():
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

    X,Y = load_database()
    # Generate synthetic data (30% of the original dataset size)
    synthetic_X, synthetic_Y = generate_synthetic_data(X, int(0.3 * len(X)))

    # Combine the original and synthetic data
    X_combined = np.vstack((X, synthetic_X))
    Y_combined = np.hstack((Y, synthetic_Y))

    # Split data
    x_train_val, x_test, y_train_val, y_test = train_test_split(X_combined, Y_combined, test_size=0.2, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.98, shuffle=True)

    # Convert data to PyTorch tensors
    x_tensor = torch.tensor(X_combined, dtype=torch.float32)

    train_dataset = SunDataset(x_train, y_train)
    val_dataset = SunDataset(x_val, y_val)
    test_dataset = SunDataset(x_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    criterion = nn.MSELoss()

    # Model
    print(
        input_size, hidden_size, 1, hidden_layer,
    )

    model = l4c.naive.MultiLayerPerceptron(
        in_features = input_size,
        hidden_features = hidden_size,
        out_features = 1,
        hidden_layers = hidden_layer,
        ) if  naive \
    else SurrogateNetwork(True, hidden_size, hidden_layer)
    
    # activation = 'GELU'

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    EXPERIMENT_NAME = f"surrogate_model_hiddensize{hidden_size}_hiddenlayers{hidden_layer}"
    CHECKPOINT_PATH = f"./checkpoints/{EXPERIMENT_NAME}/"
    # Create checkpoint path if it doesn't exist
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    BEST_MODEL_PATH = ""
    BEST_MODEL_LOSS = np.inf

    # Move everything to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    criterion.to(device)
    x_tensor = x_tensor.to(device)

    # Create tensorboard writer
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter()

    # Best validation loss
    best_val_loss = np.inf

    for epoch in range(num_epochs):
        model.train()
        for i, (x_batch, y_batch) in enumerate(train_loader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            # Forward pass
            outputs = model(x_batch)
            loss = criterion(outputs, y_batch)
            writer.add_scalar('training loss', loss.item(), epoch*len(train_loader) + i)

            # Backward pass
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
            writer.add_scalar('validation loss', val_loss/len(val_loader), epoch)
            print(f"Validation loss: {val_loss/len(val_loader)}")

            if val_loss/len(val_loader) < best_val_loss:
                best_val_loss = val_loss
                # Remove previous best model if exists
                if BEST_MODEL_PATH != "":
                    os.remove(BEST_MODEL_PATH)
                BEST_MODEL_PATH = f"{CHECKPOINT_PATH}best_model_epoch_{epoch+1}.ckpt"
                BEST_MODEL_LOSS = val_loss/len(val_loader)
                torch.save(model.state_dict(), BEST_MODEL_PATH)

                print(f"Model saved at epoch {epoch+1}")

        # Save last model
        torch.save(model.state_dict(), f"{CHECKPOINT_PATH}last_model.ckpt")
        
        # Scatter in three plots the data
        import matplotlib.pyplot as plt
        # Log an image with a prediction in the whole dataset
        y_pred = model(x_tensor).cpu().detach().squeeze().numpy()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot true values
        sc_true = ax.scatter(X_combined[:, 0] * np.cos(X_combined[:, 1]), 
                            X_combined[:, 0] * np.sin(X_combined[:, 1]), 
                            Y_combined, label="True", alpha=0.6)

        # Plot predicted values
        sc_pred = ax.scatter(X_combined[:, 0] * np.cos(X_combined[:, 1]), 
                            X_combined[:, 0] * np.sin(X_combined[:, 1]), 
                            y_pred, label="Predicted", alpha=0.6)

        ax.set_xlabel('X (Rho * cos(Phi))')
        ax.set_ylabel('Y (Rho * sin(Phi))')
        ax.set_zlabel('Value')
        ax.set_title('3D Scatter Plot of True vs Predicted Values')

        ax.legend()
        writer.add_figure('predictions', fig, epoch)

    writer.close()

    # Print results
    print("Finished training")
    print(f"Best model saved at {BEST_MODEL_PATH}")
    print(f"Best model loss: {BEST_MODEL_LOSS}")
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()

    # Load best model
    model.load_state_dict(torch.load(BEST_MODEL_PATH))
    model.eval()
    print(f"Loading best model from {BEST_MODEL_PATH}")
    
    # Test on the whole dataset and plot:
    # - Training data
    # - Training data prediction
    # - Whole dataset prediction

    import matplotlib.pyplot as plt

    x_train_tensor = torch.tensor(x_train, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        y_pred = model(x_tensor).cpu().detach().squeeze().numpy()
        y_train_pred = model(x_train_tensor).cpu().detach().squeeze().numpy()
        # 3D Scatter plot the data
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc_train = ax.scatter(X_combined[:, 0]*np.cos(X_combined[:, 1]), X_combined[:, 1]*np.sin(X_combined[:, 1]), y_pred, label="Prediction", alpha=0.6)
        sc_train = ax.scatter(x_train[:, 0]*np.cos(x_train[:, 1]), x_train[:, 1]*np.sin(x_train[:, 1]), y_train, label="Training data", alpha=0.6)
        sc_train = ax.scatter(x_train[:, 0]*np.cos(x_train[:, 1]), x_train[:, 1]*np.sin(x_train[:, 1]), y_train_pred, label="Training data prediction", alpha=0.6)
        
        sc_val = ax.scatter(x_val[:, 0]*np.cos(x_val[:, 1]), x_val[:, 0]*np.sin(x_val[:, 1]), y_val, label="Validation", alpha=0.6)
        sc_test = ax.scatter(x_test[:, 0]*np.cos(x_test[:, 1]), x_test[:, 0]*np.sin(x_test[:, 1]), y_test, label="Test", alpha=0.6)

        ax.set_xlabel('Rho')
        ax.set_ylabel('Phi')
        ax.set_zlabel('Value')
        ax.set_title('3D Scatter Plot of Data')

        ax.legend()
        plt.savefig(f'{CHECKPOINT_PATH}{EXPERIMENT_NAME}_best_pred.png')
    return model


def plot_result(hidden_size,hidden_layer,test_size=0.2, synthetic=False, rt=True, gpu=True, naive=False, use_yolo=True):

    EXPERIMENT_NAME = f"simple_mlp_hiddensize{hidden_size}_hiddenlayers{hidden_layer}_data{int(test_size*10)}" if not use_yolo else f"surrogate_model_hiddensize{hidden_size}_hiddenlayers{hidden_layer}"
    if synthetic: EXPERIMENT_NAME += '_synthetic'
    model = l4c.naive.MultiLayerPerceptron(
    in_features = 3 if use_yolo else 1,
    hidden_features = hidden_size,
    out_features = 1,
    hidden_layers = hidden_layer)if naive \
    else SurrogateNetwork(use_yolo, hidden_size, hidden_layer)
    #activation = 'GELU'
    # Directory where checkpoints are saved
    checkpoint_dir = "./checkpoints"
    experiment_path = os.path.join(checkpoint_dir, EXPERIMENT_NAME)

    # Find the best model file
    best_model_files = [ os.path.join(experiment_path,f) for f in os.listdir(experiment_path) if f.startswith("best_model_") and f.endswith('.ckpt')]
    if not best_model_files:
        raise FileNotFoundError("No best model file found.")
    print(best_model_files)
    CHECKPOINT_PATH = max(best_model_files, key=os.path.getctime)  # Choose the latest file if there are multiple

    # Load the best model
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    print(f"Loading best model from {CHECKPOINT_PATH}")

    # Test on the whole dataset and plot:
    # - Training data
    # - Training data prediction
    # - Whole dataset prediction

    import matplotlib.pyplot as plt
    synthetic_X = []
    for _ in range(10000):
        rho = random.uniform(3, 10)
        phi = random.uniform(-2*np.pi, 2*np.pi)
        yaw = 0.0

        synthetic_X.append([rho, phi, yaw])
    synthetic_X = np.array(synthetic_X)

    x_train_tensor = torch.tensor(synthetic_X, dtype=torch.float32).unsqueeze(1).to(device)

    with torch.no_grad():
        y_pred = model(x_train_tensor).cpu().detach().squeeze().numpy()
        # 3D Scatter plot the data
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        sc_train = ax.scatter(synthetic_X[:, 0]*np.cos(synthetic_X[:, 1]), synthetic_X[:, 1]*np.sin(synthetic_X[:, 1]), y_pred, label="Prediction", alpha=0.6)

        ax.set_xlabel('Rho')
        ax.set_ylabel('Phi')
        ax.set_zlabel('Value')
        ax.set_title('3D Scatter Plot of Data')

        ax.legend()
        plt.show()

import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run the neural network training simulation with specified parameters.')

    parser.add_argument('--test_size', type=float, default=0.2, help='Proportion of the dataset to include in the test split')
    parser.add_argument('--input_size', type=int, default=1, help='Size of the input layer')
    parser.add_argument('--hidden_layer', type=int, default=2, help='Number of hidden layers')
    parser.add_argument('--hidden_size', type=int, default=64, help='Size of each hidden layer')
    parser.add_argument('--output_size', type=int, default=1, help='Size of the output layer')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='Learning rate for training')
    parser.add_argument('--num_epochs', type=int, default=100, help='Number of epochs for training')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for training')
    parser.add_argument('--synthetic', type=bool, default=True, help='Whether to use synthetic data')
    parser.add_argument('--yolo', type=bool, default=True, help='Whether to use synthetic data')

    args = parser.parse_args()
    
    if args.yolo:
        TrainSurrogate(
            hidden_layer=args.hidden_layer,
            hidden_size=args.hidden_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,

        )
    else:
        TrainQiNN(
            test_size=args.test_size,
            input_size=args.input_size,
            hidden_layer=args.hidden_layer,
            hidden_size=args.hidden_size,
            output_size=args.output_size,
            learning_rate=args.learning_rate,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            synthetic=args.synthetic
        )
    """
    plot_result(
            hidden_layer=args.hidden_layer,
            hidden_size=args.hidden_size,
    )
    """