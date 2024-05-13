import torch
import gpytorch
import math
import numpy as np
import csv
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from unity_drone_sim_bridge.gp_mpc.gp_class import GP

def sigmoid(x):
    return 1 / (1 + np.exp(-7.5 * x))

def fake_nn(x):
    x_min = 1.100 
    x_max = 1.290 
    y_min = 0.50
    y_max = 0.69
    
    x_range = x_max - x_min
    half_x_range = x_range / 2
    
    normalized_x = (x - x_min - half_x_range) / x_range
    normalized_y = sigmoid(normalized_x)
    mapped_y = y_min + normalized_y * (y_max - y_min)
    
    return mapped_y

def norm_nn(x):
    x_min = 1.100 
    x_max = 1.290 
    y_min = 0.5
    y_max = 0.69
    
    x_range = x_max - x_min
    half_x_range = x_range / 2
    
    normalized_x = (x - min(x)) / (max(x)-min(x))
    normalized_y = normalized_x
    mapped_y = y_min + normalized_y * (y_max - y_min)
    
    return mapped_y

class Cos(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.cos(input)

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

def LoadCaGP(synthetic=True, viz=True, N=40, method='ME'):
    # Limits in the training data
    ulb = []    # No inputs are used
    uub = []    # No inputs are used

    #N : Number of training data
    N_test = 100    # Number of test data

    if synthetic:
        X = (np.linspace(-1, 1, N) * (2 * np.pi)).reshape(-1,1)   
        Y = ((1 + np.cos(X) + np.random.random(X.shape) * np.sqrt(0.04))/ 10 + 0.4).reshape(-1,1) #(2 - (1 + np.cos(X) + np.random.random(X.shape) * np.sqrt(0.04))).reshape(-1,1)

        X_test = (np.linspace(-1, 1, N_test) * (2 * np.pi)).reshape(-1,1)
        Y_test = ((1 + np.cos(X_test) + np.random.random(X_test.shape) * np.sqrt(0.04))/ 10 + 0.4).reshape(-1,1) #(2 - (1 + np.cos(X_test) + np.random.random(X_test.shape) * np.sqrt(0.04))).reshape(-1,1)
        xlb = [0.0]
        xub = [2.0]
    else:
        X,Y = loadDatabase(N)
        X_test, Y_test= loadDatabase(N*2)
        xlb = [0.0]
        xub = [.5]

    """ Create GP model and optimize hyper-parameters"""
    gp = GP(X, Y, mean_func='zero', normalize=True, xlb=xlb, xub=xub, ulb=ulb,
            uub=uub, optimizer_opts=None, gp_method=method)
    gp.validate(X_test, Y_test)
    print(viz)
    if viz: plot_data(gp, X,Y,X_test, Y_test)
    return gp

def plot_data(gp, X,Y,X_test, Y_test):
    """ Plot comparison of GP prediction with exact simulation
        on a 2000 step prediction horizon
    """
    Nt = 2000
    x0 = np.array([0.0])

    cov = np.zeros((1,1))
    x = np.zeros((Nt,1))
    x_sim = np.zeros((Nt,1))

    x[0] = x0
    x_sim[0] = x0

    gp.set_method('ME')         # Use Mean Equivalence as GP method
    for i, x_ in zip(range(Nt),np.linspace(-1,1,Nt)):
        x_t, cov = gp.predict([x_*2*np.pi], [], cov)
        x[i] = np.array(x_t).flatten()

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(np.linspace(-1,1,Nt)*2*np.pi, x[:,0], 'b-', linewidth=1.0, label='GP')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.scatter(X_test, Y_test, label='dataset')
    ax.scatter(X,Y, label='train')
    
    plt.legend(loc='best')
    plt.show()

