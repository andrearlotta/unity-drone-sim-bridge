import numpy as np
from unity_drone_sim_bridge.g_func_lib.generic_tools import fake_nn

def loadDatabase(N=None):
    import csv
    dataset_path = "/home/pantheon/lstm_sine_fitting/qi_csv_datasets/drone_round_sun_012_SaturationAndLuminance.csv"
    dataset_array = fake_nn(np.roll(np.array(list(csv.reader(open(dataset_path))), dtype=float), -int(1081/4)))

    y = np.stack([dataset_array,dataset_array]).reshape(-1,1)
    X = (np.linspace(-1, 1, len(y)) * (2*np.pi)).reshape(-1,1)
    # Set a seed for reproducibility (optional)
    np.random.seed(42)
    if N is None: return X, y
    # Generate random indices for the subset
    num_samples = len(X)
    subset_size = N
    subset_indices = np.random.choice(num_samples, size=subset_size, replace=False)
    # Create subsets of x and y based on the random indices
    X_train = X[subset_indices]
    y_train = y[subset_indices]
    return X_train, y_train

def loadSyntheticDatabase(N=40, N_test=100):
    X = (np.linspace(-1, 1, N) * (2 * np.pi)).reshape(-1,1)   
    Y = ((1 + np.cos(X) + np.random.random(X.shape) * np.sqrt(0.04))/ 15 + 0.5).reshape(-1,1) 
    #(2 - (1 + np.cos(X) + np.random.random(X.shape) * np.sqrt(0.04))).reshape(-1,1)
    X_test = (np.linspace(-1, 1, N_test) * (2 * np.pi)).reshape(-1,1)
    Y_test = ((1 + np.cos(X_test) + np.random.random(X_test.shape) * np.sqrt(0.04))/ 15 + 0.5).reshape(-1,1)
    return X, Y, X_test, Y_test