import numpy as np
from unity_drone_sim_bridge.generic_tools import fake_nn, fov_weight_fun_polar_numpy, fov_weight_fun_numpy
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

def load_cartesian_surrogate_database(DATA_PATH='/home/pantheon/dataset_apples_FLU_Cartesian/CameraPositions_FLU_Cartesian_Dataset_Filtered.csv'):
    X = []
    Y = []

    with open(DATA_PATH, 'r') as infile:
        data = list(csv.reader(infile))
        header = data[0]  # Skip the header
        data = data[1:]  # The actual data

        # Randomly shuffle the data and then take 10% of it
        np.random.shuffle(data)
        sample_size = int(len(data) * 0.1)
        sampled_data = data[:sample_size]

        for row in sampled_data:
            x, y, yaw, value = map(float, row)
            X.append([x, y, yaw])
            Y.append(value)
    
    return np.array(X), np.array(Y)

# Generate synthetic data
def generate_cartesian_surrogate_synthetic_data(X, num_samples):
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    yaw_values = [30, -30, 60, -60, 90, -90, 180, -180]
    
    synthetic_X = []
    synthetic_Y = []
    
    for _ in range(num_samples):
        choice = random.choice([1, 2])
        if choice == 1:
            # Generate synthetic data in the range (0, rho_min) and (0, phi_min)
            x = random.uniform(0, x_min) * random.uniform(-2*np.pi, 2*np.pi)
            y = random.uniform(0, x_min) * random.uniform(-2*np.pi, 2*np.pi)
            #random.uniform(-2*np.pi, 2*np.pi)
        else:
            # Generate synthetic data in the range (rho_max, rho_max+some_value) and (5, phi_max)
            x = random.uniform(x_max, x_max + 5) * random.uniform(-2*np.pi, 2*np.pi)
            y = random.uniform(x_max, x_max + 5) * random.uniform(-2*np.pi, 2*np.pi)
            
        yaw = np.radians(random.choice(yaw_values) + random.uniform(-2, +2))
        value = 0.5

        synthetic_X.append([x, y, yaw])
        synthetic_Y.append(value)

    return np.array(synthetic_X), np.array(synthetic_Y)



# Utility Function
def load_polar_surrogate_database(DATA_PATH = '/home/pantheon/dataset_apples_3/FilteredCameraPositionsFull.csv'):
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
def generate_polar_surrogate_synthetic_data(X, num_samples):
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

# Utility Function
def load_surrogate_database(DATA_PATH = '/home/pantheon/dataset_apples_FLU_Cartesian/CameraPositions_FLU_Cartesian_Dataset_Filtered.csv'):
    X = []
    Y = []

    with open(DATA_PATH, 'r') as infile:
        data = csv.reader(infile)
        next(data)  # Skip the header
        
        for row in data:
            x, y, yaw, value = map(float, row)
            X.append([x, y, yaw])
            Y.append(value)
    
    return np.array(X), np.array(Y)

# Generate synthetic data
def generate_surrogate_synthetic_data(X, num_samples):
    x_min, x_max = X[:, 0].min(), X[:, 0].max()
    yaw_values = [30, -30, 60, -60, 90, -90, 180, -180]
    
    synthetic_X = []
    synthetic_Y = []
    
    for _ in range(num_samples):
        choice = random.choice([1, 2])
        if choice == 1:
            # Generate synthetic data in the range (0, rho_min) and (0, phi_min)
            x = random.uniform(0, x_min) * random.uniform(-2*np.pi, 2*np.pi)
            y = random.uniform(0, x_min) * random.uniform(-2*np.pi, 2*np.pi)
            #random.uniform(-2*np.pi, 2*np.pi)
        else:
            # Generate synthetic data in the range (rho_max, rho_max+some_value) and (5, phi_max)
            x = random.uniform(x_max, x_max + 5) * random.uniform(-2*np.pi, 2*np.pi)
            y = random.uniform(x_max, x_max + 5) * random.uniform(-2*np.pi, 2*np.pi)
            
        yaw = np.radians(random.choice(yaw_values) + random.uniform(-2, +2))
        value = 0.5

        synthetic_X.append([x, y, yaw])
        synthetic_Y.append(value)

    return np.array(synthetic_X), np.array(synthetic_Y)

def generate_fake_polar_dataset(num_samples):
    
    synthetic_X = []
    synthetic_Y = []
    for _ in range(num_samples):
        rho = random.uniform(0, 10)
        phi = random.uniform(-np.pi, np.pi)
        yaw = random.uniform(-np.pi, np.pi)

        value = fov_weight_fun_polar_numpy(yaw, [rho,phi], thresh_distance=5) * 0.3 + 0.5
        synthetic_X.append([rho, phi, yaw])
        synthetic_Y.append(value)

    return np.array(synthetic_X), np.array(synthetic_Y)

def generate_fake_cartesian_dataset(num_samples):
    
    synthetic_X = []
    synthetic_Y = []
    for _ in range(num_samples):
        x = random.uniform(-10, 10)
        y = random.uniform(-10, 10)
        yaw = random.uniform(-np.pi, np.pi)

        value = fov_weight_fun_numpy( np.array([x,y]),yaw, np.array([[0.0, 0.0]]), thresh_distance=5) * 0.3 + 0.5
        synthetic_X.append([x, y, yaw])
        synthetic_Y.append(value)
        if value> 0.6:
            print(value)
            print([x, y, yaw])
    print('porcaputtana')
    return np.array(synthetic_X), np.array(synthetic_Y)