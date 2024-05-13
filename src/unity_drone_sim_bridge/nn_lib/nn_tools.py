import numpy as np
import tensorflow as tf
import keras
import casadi
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
import csv
import os
import tf2onnx
import do_mpc
import math
import torch
import gpytorch
from unity_drone_sim_bridge.nn_lib.gp_nn_tools import loadDatabase, LoadCaGP, Cos
import l4casadi as l4c
import casadi as ca

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

'''
cond functions
'''
def ___trees_satisfy_conditions(drone_pos, drone_yaw, tree_pos, thresh_distance=7):
    # Convert inputs to CasADi symbols
    drone_pos_sym = ca.MX(drone_pos)
    drone_yaw_sym = ca.MX(drone_yaw)
    tree_pos_sym = ca.MX(tree_pos)
    
    # Calculate distance between the drone and each tree
    distances = ca.sqrt(ca.sum1((tree_pos_sym - drone_pos_sym)**2, axis=1))
    
    # Calculate direction from drone to each tree
    drone_dir = ca.vertcat(ca.cos(drone_yaw_sym), ca.sin(drone_yaw_sym))
    tree_directions = tree_pos_sym - drone_pos_sym
    tree_directions_norm = tree_directions / ca.sqrt(ca.sum1(tree_directions**2, axis=1))
    
    # Check conditions 
    indices = ca.find(ca.and1((distances < thresh_distance), 
                              (ca.sum1(drone_dir * tree_directions_norm, axis=1) > 0.9)))
    
    # Convert indices to numpy array
    indices_np = np.array(indices).flatten()
    
    return indices_np


def trees_satisfy_conditions_casadi(drone_pos_sym, drone_yaw_sym, tree_pos_sym, thresh_distance=7):
    n_trees = tree_pos_sym.shape[0]
    # Calculate distance between the drone and each tree
    distances = ca.sqrt(ca.sum2((tree_pos_sym - np.ones((n_trees,1))@drone_pos_sym.T)**2))
    # Calculate direction from drone to each tree
    drone_dir = ca.vertcat(ca.cos(drone_yaw_sym.T), ca.sin(drone_yaw_sym.T))
    tree_directions = tree_pos_sym - np.ones((n_trees,1))@drone_pos_sym
    tree_directions_norm = tree_directions / (ca.sqrt(ca.sum2(tree_directions**2))@np.ones((1,2)))
    # Check conditions
    indices =  ca.evalf(distances < thresh_distance)
    love = (ca.sum2((np.ones((n_trees,1))@drone_dir.T) * tree_directions_norm) > 0.9)
    return ca.logic_and(love,indices)


def trees_satisfy_conditions(drone_pos, drone_yaw, tree_pos, thresh_distance=7):
    # Calculate distance between the drone and each tree
    distances = np.linalg.norm(tree_pos - drone_pos, axis=1)
    # Calculate direction from drone to each tree
    drone_dir = np.array([np.cos(drone_yaw), np.sin(drone_yaw)])
    tree_directions = tree_pos - drone_pos
    tree_directions_norm = tree_directions / np.linalg.norm(tree_directions, axis=1)[:, np.newaxis]
    # Check conditions 
    indices = np.where((distances < thresh_distance) & (np.sum(drone_dir * tree_directions_norm, axis=1) > 0.9))[0]
    return indices


def obstacle_cost(drone_pos, drone_yaw, tree_pos, thresh_distance=7):
    # Calculate distance between the drone and each tree
    distances = np.linalg.norm(tree_pos - drone_pos, axis=1)


'''
g functions
'''

def load_g(mode='gp', hidden_size=64, hidden_layer=5):
    if  mode == 'mlp':
        mlp = LoadNN(hidden_size,hidden_layer)
        print(mlp)
        return lambda x: g_nn(x, mlp)
    elif mode == 'gp':
        gp =  LoadCaGP(synthetic=False)
        return lambda x: g_gp(x, gp)

def g_gp(drone_yaw_sym, gp, thresh_distance=7):
    return  np.ones(drone_yaw_sym.shape) * gp.predict(drone_yaw_sym, [], np.zeros((1,1)))[0] #np.ones(drone_yaw_sym.shape) * ca.cos(drone_yaw_sym) + 1


def g_nn(drone_yaw_sym, nn):
    return  np.ones(drone_yaw_sym.shape) * nn(drone_yaw_sym)    #cs.Function('y', [x_sym], [y_sym])

'''
nayes function
'''

def bayes(lambda_k,y_z):
    return ca.times(lambda_k, y_z) / (ca.times(lambda_k, y_z) + (1-lambda_k) * (1-y_z))

import torch
import torch.nn as nn
import torch.nn.functional as F

def LoadNN(hidden_size,hidden_layer,test_size=0.2):
        EXPERIMENT_NAME = f"simple_mlp_hiddensize{hidden_size}_hiddenlayers{hidden_layer}_data{int(test_size*10)}"
        model = l4c.naive.MultiLayerPerceptron(
        in_features = 1,
        hidden_features = hidden_size,
        out_features = 1,
        hidden_layers = hidden_layer,
        activation = 'ReLU')
        CHECKPOINT_PATH = f"./checkpoints/{EXPERIMENT_NAME}/last_model.ckpt"
        model.load_state_dict(torch.load(CHECKPOINT_PATH))
        model.eval()
        print(f"Loading model from {CHECKPOINT_PATH}")
        return l4c.L4CasADi(model, model_expects_batch_dim=True, device="cuda")

def TrainNN(test_size=0.2, input_size = 1, hidden_layer=0, hidden_size = 256, output_size = 1, learning_rate = 0.0001, num_epochs = 500, batch_size = 1):
    x, y = loadDatabase()
    # Split data
    from sklearn.model_selection import train_test_split
    x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=test_size, shuffle=True)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.98, shuffle=True)

    import torch
    # Create batches to do inference in the whole dataset
    x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(1)

    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    from torch.utils.data import DataLoader

    # Create dataloaders
    class SunDataset(torch.utils.data.Dataset):
        def __init__(self, x, y):
            self.x = torch.tensor(x, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32)

        def __len__(self):
            return len(self.x)

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

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
    activation = 'ReLU')

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Training
    EXPERIMENT_NAME = f"simple_mlp_hiddensize{hidden_size}_hiddenlayers{hidden_layer}_data{int(test_size*10)}"
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


'''    if mode == 'scikit' or mode == 'onnx':
        kernel = ExpSineSquared(
            length_scale=1.0,
            periodicity=1.0,
        )
        gaussian_process = GaussianProcessRegressor(kernel=kernel)
        gaussian_process.fit(X_train, y_train)

        if mode == 'onnx':
            import skl2onnx
            import onnxruntime
            from onnxconverter_common import FloatTensorType
            initial_type = [("X", FloatTensorType([1, 1]))]
            initial_type = [('X', FloatTensorType([None, X_train.shape[1]]))]
            onx = skl2onnx.convert_sklearn(gaussian_process, initial_types=initial_type, target_opset=9)
            gaussian_process = do_mpc.sysid.ONNXConversion(onx)

            onx64 = skl2onnx.to_onnx(gaussian_process, X[:1])
            sess64 = onnxruntime.InferenceSession(
                onx64.SerializeToString(), providers=["CPUExecutionProvider"]
            )

            from onnxsim import simplify
            model_simp, check = simplify(onx)
            assert check, "Simplified ONNX model could not be validated"
        return gaussian_process

    elif mode == 'l4casadi':
        import l4casadi as l4c
        nn = Cos() #GP_NN()
        return l4c.L4CasADi(nn, model_expects_batch_dim=True, device='cpu')  # device='cuda' for GPU
    
    elif mode == 'gpflow':
        import gpflow
        X_train, y_train = loadDatabase()
        model = gpflow.models.GPR((X_train.reshape(-1, 1), y_train.reshape(-1, 1)),
                                  gpflow.kernels.Constant(1) + gpflow.kernels.Linear(1) +
                                  gpflow.kernels.White(1) + gpflow.kernels.RBF(1), mean_function=None,
                                  noise_variance=1.0)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
        class GPR(casadi.Callback):
            def __init__(self, name, opts: dict = None):
                if opts is None:
                    opts = dict()
                casadi.Callback.__init__(self)
                self.construct(name, opts)
            def eval(self, arg):
                [mean, _] = model.predict_f(np.array(arg[0]))
                return [mean.numpy()]
        return GPR('GPR', opts={"enable_fd": True})
'''