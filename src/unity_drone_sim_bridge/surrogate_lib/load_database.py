import numpy as np
from unity_drone_sim_bridge.generic_tools import fake_nn, fov_weight_fun_polar_numpy, trees_satisfy_conditions_np
import csv
import random
from sklearn.model_selection import train_test_split
import os
import glob

def find_latest_folder(base_dir):
    """Finds the latest dated folder in the base directory."""
    # List all subdirectories in the base_dir
    subdirs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    
    # Sort the directories by their modified time in reverse (latest first)
    latest_folder = max(subdirs, key=lambda d: os.path.getmtime(os.path.join(base_dir, d)))
    return os.path.join(base_dir, latest_folder)


def find_csv_in_folder(folder_path):
    """Finds the first CSV file in the folder."""
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    
    if csv_files:
        return csv_files[0]  # Return the first CSV file found
    else:
        raise FileNotFoundError("No CSV files found in the folder.")


def create_detections_folder(folder_path):
    """Creates the 'detections' folder inside the given folder path."""
    detections_folder = os.path.join(folder_path, "detections")
    
    if not os.path.exists(detections_folder):
        os.makedirs(detections_folder)
        print(f"Created folder: {detections_folder}")
    else:
        print(f"Folder already exists: {detections_folder}")
    
    return detections_folder


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
def load_surrogate_database(polar, n_input, test_size=0.0):
    polar = polar

    fixedView = False
    # Paths
    base_dir = '/home/pantheon/ripe_apples_dataset/'
    base_dir = os.path.join(base_dir, "polar" if polar else "cartesian")
    base_dir = os.path.join(base_dir, "fixed_view" if fixedView else "variable_view")
    output_csv_filename = 'SurrogateDatasetCNN_Filtered.csv'

    latest_dataset_folder = find_latest_folder(base_dir=base_dir)
    # Step 4: Create output CSV file inside the latest folder
    DATA_PATH = os.path.join(latest_dataset_folder, output_csv_filename)
    print(DATA_PATH)
    print(DATA_PATH)
    X = []  
    Y = []
    with open(DATA_PATH, 'r') as infile:
        data = csv.reader(infile)
        next(data)  # Skip the header
        for row in data:
            a, b, c, value = map(float, row)
            X.append([a,b, np.deg2rad(c)] if n_input == 3 else [a,b])
            Y.append(value)
    #X, _, Y, _ = train_test_split(X, Y, test_size=test_size, shuffle=True)
    
    return np.array(X), np.array(Y)


# Generate synthetic data
def generate_surrogate_augmented_data(X, num_samples, polar, n_input):
    x_min = np.abs(X[:, 0]).min() if polar else (np.sqrt(X[:, 0]**2 + X[:, 1]**2)).min()
    x_max = np.abs(X[:, 0]).max() if polar else None
    yaw_values = [30, -30, 60, -60, 90, -90, 180, -180]
    
    synthetic_X = []
    synthetic_Y = []
    
    for _ in range(num_samples):
        #choice = random.choice([1, 2])
        #if choice == 1:
        # Generate synthetic data in the range (0, rho_min) and (0, phi_min)
        theta = random.uniform(-np.pi, np.pi)
        r = random.uniform(0, x_min) 
        a = r       if  polar else r * np.cos(theta) 
        b = theta   if  polar else r * np.sin(theta)
        #random.uniform(-2*np.pi, 2*np.pi)
        #else:
        #    # Generate synthetic data in the range (rho_max, rho_max+some_value) and (5, phi_max)
        #    a = random.uniform(x_max, x_max + 5)    if polar else random.uniform(x_max, x_max + 5) * np.cos(random.uniform(-np.pi, np.pi))
        #    b = random.uniform(-np.pi, np.pi)       if polar else random.uniform(x_max, x_max + 5) * np.sin(random.uniform(-np.pi, np.pi))
        #    
        c = np.radians(random.choice(yaw_values) + random.uniform(-2, +2))
        value = 0.5
        synthetic_X.append([a, b, c] if n_input ==3 else [a, b])
        synthetic_Y.append(value)
        
    return np.array(synthetic_X), np.array(synthetic_Y)


def generate_fake_dataset(num_samples, is_polar, n_input):
    
    synthetic_X = []
    synthetic_Y = []
    tree_pos = np.zeros((1,2))
    # Define ranges for drone position (X, Y) and yaw
    x = np.linspace(-8, 8, 16)
    y = np.linspace(-8, 8, 16)
    yaw_angles = np.linspace(-np.pi, np.pi, 60)  # More fine-grained yaw angle variation

    X, Y = np.meshgrid(x, y)
    # Iterate over the grid to vary the drone's position and yaw
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            drone_pos = np.array([X[i, j], Y[i, j]])
            
            # Iterate over all yaw angles for the current drone position
            for yaw in yaw_angles:
                value = ( trees_satisfy_conditions_np(drone_pos, yaw, tree_pos, 5)/0.5 )* 0.3 +0.5 
                synthetic_X.append([X[i, j], Y[i, j], yaw] if n_input ==3 else [X[i, j], Y[i, j]])
                synthetic_Y.append(value)

    return np.array(synthetic_X), np.array(synthetic_Y)


def plot_surface_vary_drone(tree_pos, thresh_distance):
    # Define ranges for drone position (X, Y) and yaw
    x = np.linspace(-8, 8, 32)
    y = np.linspace(-8, 8, 32)
    yaw_angles = np.linspace(-np.pi, np.pi, 60)  # More fine-grained yaw angle variation

    X, Y = np.meshgrid(x, y)
    # Iterate over the grid to vary the drone's position and yaw
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            drone_pos = np.array([X[i, j], Y[i, j]])
            
            # Iterate over all yaw angles for the current drone position
            for yaw in yaw_angles:
                value = trees_satisfy_conditions_np(drone_pos, yaw, tree_pos, thresh_distance)     
                synthetic_X.append([X[i, j], Y[i, j], yaw] if n_input ==3 else [X[i, j], Y[i, j]])
                synthetic_Y.append(value)

