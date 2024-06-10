
import torch
import torch.nn as nn
import torch.nn.functional as F
from unity_drone_sim_bridge.g_func_lib.gp_tools import loadDatabase, loadSyntheticDatabase
import l4casadi as l4c
import os 
import numpy as np
import tensorflow as tf

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

def LoadNN(hidden_size,hidden_layer,test_size=0.2, synthetic=False):

        EXPERIMENT_NAME = f"simple_mlp_hiddensize{hidden_size}_hiddenlayers{hidden_layer}_data{int(test_size*10)}"
        if synthetic: EXPERIMENT_NAME += '_synthetic'
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

def TrainNN(test_size=0.2, input_size = 1, hidden_layer=0, hidden_size = 256, output_size = 1, learning_rate = 0.0001, num_epochs = 500, batch_size = 1, synthetic=False):
    x, y = loadSyntheticDatabase() if synthetic else loadDatabase()
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