import torch
import gpytorch
import math
import numpy as np
import csv

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([1]))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(batch_shape=torch.Size([1])), batch_shape=torch.Size([1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

def train_gp_model(model, likelihood, train_x, train_y, training_iter=500, lr=0.25):
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x)
        loss = -mll(output, train_y)
        loss.backward()
        optimizer.step()
        print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))

    model.eval()
    likelihood.eval()

    return model, likelihood

def train():
    # Training data: 20 points in [0,1] regularly spaced
    train_x_mean = torch.linspace(0, 1, 20) * (2 * math.pi)

    # True function is sin(2*pi*x) with Gaussian noise
    train_y = 2 - (1 + torch.cos(train_x_mean) + torch.randn(train_x_mean.size()) * 0.1)

    # Initialize likelihood and model
    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([1]))
    model = ExactGPModel(train_x_mean, train_y, likelihood)

    # Training settings
    training_iter = 500
    trained_model, trained_likelihood = train_gp_model(model, likelihood, train_x_mean, train_y, training_iter)

    print("Training completed!")
    return trained_model, trained_likelihood

# Define the Cosine neural network
class GP_NN(torch.nn.Module):
    def __init__(self):
        super(GP_NN, self).__init__()
        self.trained_model, self.trained_likelihood  = train()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.trained_likelihood(self.trained_model(input)).mean

class Cos(torch.nn.Module):
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.cos(input)

def loadDatabase(numpyzer=False):
    dataset_path = "/home/pantheon/lstm_sine_fitting/qi_csv_datasets/drone_round_sun_012_SaturationAndLuminance.csv"
    dataset_array = np.array(list(csv.reader(open(dataset_path))), dtype=float)[:1080]

    y = dataset_array #random.choice(dataset_array).reshape(-1, 1)
    X = np.linspace(0,360, len(y)).reshape(-1, 1)
    # Set a seed for reproducibility (optional)
    np.random.seed(42)
    # Define the percentage of data to include in the subset
    subset_percentage = 0.1 # Adjust as needed
    # Generate random indices for the subset
    num_samples = len(X)
    subset_size = int(subset_percentage * num_samples)
    subset_indices = np.random.choice(num_samples, size=subset_size, replace=False)
    # Create subsets of x and y based on the random indices
    X_train = X[subset_indices]
    y_train = y[subset_indices]
    return X_train, y_train

def loadSyntheticData(numpyzer=True):
    train_x_mean = torch.linspace(0, 1, 20) * (2 * math.pi)
    train_y = 2 - (1 + torch.cos(train_x_mean) + torch.randn(train_x_mean.size()) * math.sqrt(0.04))

    test_x = torch.linspace(0, 1, 51) * (2 * math.pi)
    test_y = 2 - (1 + torch.cos(train_x_mean) + torch.randn(train_x_mean.size()) * math.sqrt(0.04))

    if numpyzer:
        return np.asarray(train_x_mean), np.asarray(train_y),  np.asarray(test_x), np.asarray(test_y)
    else:
        return train_x_mean, train_y
    
'''
# Usage example
cos_model = GP_NN()
output = cos_model(torch.tensor([0.5]))
print("Output tensor values:", output)
'''


import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from unity_drone_sim_bridge.gp_mpc.gp_class import GP


def loadDatabase(subset_percentage):
    import csv
    dataset_path = "/home/pantheon/lstm_sine_fitting/qi_csv_datasets/drone_round_sun_012_SaturationAndLuminance.csv"
    dataset_array = np.array(list(csv.reader(open(dataset_path))), dtype=float)[:1080]

    y = dataset_array #random.choice(dataset_array).reshape(-1, 1)
    X = np.linspace(0,np.pi*2, len(y)).reshape(-1, 1)
    # Set a seed for reproducibility (optional)
    np.random.seed(42)

    # Generate random indices for the subset
    num_samples = len(X)
    subset_size = int(subset_percentage * num_samples)
    subset_indices = np.random.choice(num_samples, size=subset_size, replace=False)
    # Create subsets of x and y based on the random indices
    X_train = X[subset_indices]
    y_train = y[subset_indices]
    return X_train, y_train

def LoadCaGP(synthetic=True):
    """ System Parameters """
    dt = .01                    # Sampling time
    Nx = 1                      # Number of states
    Nu = 0                      # Number of inputs

    # Limits in the training data
    ulb = []    # No inputs are used
    uub = []    # No inputs are used

    N = 40          # Number of training data
    N_test = 100    # Number of test data

    a = True

    if synthetic:
        X = (np.linspace(-1, 1, N) * (2 * np.pi)).reshape(-1,1)
        Y = (2 - (1 + np.cos(X) + np.random.random(X.shape) * np.sqrt(0.04))).reshape(-1,1)

        X_test = (np.linspace(-1, 1, N_test) * (2 * np.pi)).reshape(-1,1)
        Y_test = (2 - (1 + np.cos(X_test) + np.random.random(X_test.shape) * np.sqrt(0.04))).reshape(-1,1)
        xlb = [0.0]
        xub = [2.0]
    else:
        X,Y = loadDatabase(0.05)
        X_test, Y_test= loadDatabase(0.5)
        xlb = [0.0]
        xub = [.5]

    plt.figure()
    ax = plt.subplot(111)
    ax.scatter(X_test, Y_test, label='GP')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    plt.legend(loc='best')
    plt.show()

    """ Create GP model and optimize hyper-parameters"""
    gp = GP(X, Y, mean_func='zero', normalize=True, xlb=xlb, xub=xub, ulb=ulb,
            uub=uub, optimizer_opts=None)
    gp.validate(X_test, Y_test)

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
    for i in range(Nt):
        x_t, cov = gp.predict([i*2*np.pi/Nt], [], cov)
        x[i] = np.array(x_t).flatten()

    plt.figure()
    ax = plt.subplot(111)
    ax.plot(np.linspace(0,1,Nt)*2*np.pi, x[:,0], 'b-', linewidth=1.0, label='GP')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.scatter(X_test, Y_test, label='dataset')
    ax.scatter(X,Y, label='train')
    
    plt.legend(loc='best')
    plt.show()
