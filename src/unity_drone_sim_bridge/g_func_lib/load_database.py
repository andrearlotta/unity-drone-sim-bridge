import numpy as np
from unity_drone_sim_bridge.generic_tools import fake_nn
import csv
import random

def load_database(N=None):
    dataset_path = "/home/pantheon/lstm_sine_fitting/qi_csv_datasets/drone_round_sun_012_SaturationAndLuminance.csv"
    
    with open(dataset_path) as f:
        dataset_array = fake_nn(np.roll(np.array(list(csv.reader(f)), dtype=float), -int(1081/4)))
    
    y = np.stack([dataset_array, dataset_array]).reshape(-1, 1)
    X = (np.linspace(-1, 1, len(y)) * (2 * np.pi)).reshape(-1, 1)
    
    # Set a seed for reproducibility (optional)
    np.random.seed(42)
    
    if N is None:
        return X, y
    
    # Generate random indices for the subset
    num_samples = len(X)
    subset_indices = np.random.choice(num_samples, size=N, replace=False)
    
    # Create subsets of X and y based on the random indices
    X_train = X[subset_indices]
    y_train = y[subset_indices]
    
    return X_train, y_train

def load_synthetic_database(N=40, N_test=100):
    X = (np.linspace(-1, 1, N) * (2 * np.pi)).reshape(-1, 1)
    Y = ((1 + np.cos(X) + np.random.random(X.shape) * np.sqrt(0.04)) / 15 + 0.5).reshape(-1, 1)
    
    X_test = (np.linspace(-1, 1, N_test) * (2 * np.pi)).reshape(-1, 1)
    Y_test = ((1 + np.cos(X_test) + np.random.random(X_test.shape) * np.sqrt(0.04)) / 15 + 0.5).reshape(-1, 1)
    
    return X, Y, X_test, Y_test


# Utility Function
def load_surrogate_database(DATA_PATH = "/home/pantheon/dataset_apples_2/FilteredCameraPositions.csv"):
    X = []
    Y = []

    with open(DATA_PATH, 'r') as infile:
        data = csv.reader(infile)
        next(data)  # Skip the header
        
        for row in data:
            phi, rho, yaw, value = map(float, row)
            X.append([rho, phi, yaw])
            Y.append(value)
    
    return np.array(X), np.array(Y)

# Generate synthetic data
def generate_surrogate_synthetic_data(X, num_samples):
    rho_min, rho_max = X[:, 0].min(), X[:, 0].max()
    phi_min, phi_max = X[:, 1].min(), X[:, 1].max()
    yaw_values = [30, -30, 60, -60, 90, -90, 180, -180]
    
    synthetic_X = []
    synthetic_Y = []
    
    for _ in range(num_samples):
        choice = random.choice([1, 2, 3])
        if choice == 1:
            # Generate synthetic data within original range
            rho = random.uniform(rho_min, rho_max)
            phi = random.uniform(phi_min, phi_max)
        elif choice == 2:
            # Generate synthetic data in the range (0, rho_min) and (0, phi_min)
            rho = random.uniform(0, rho_min)
            phi = random.uniform(-2*np.pi, 2*np.pi)
        else:
            # Generate synthetic data in the range (rho_max, rho_max+some_value) and (15, phi_max)
            rho = random.uniform(rho_max, rho_max + 15)
            phi = random.uniform(-2*np.pi, 2*np.pi)
            
        yaw = np.radians(random.choice(yaw_values) + random.uniform(-5, +5))
        value = 0.5

        synthetic_X.append([rho, phi, yaw])
        synthetic_Y.append(value)

    return np.array(synthetic_X), np.array(synthetic_Y)
