
import torch
import gpytorch
import math
import numpy as np

# Define a Gaussian Process model using GPyTorch
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        # Initialize mean and covariance modules
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([1]))
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel(batch_shape=torch.Size([1])), batch_shape=torch.Size([1]))

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

# Train the Gaussian Process model
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

# Train the Gaussian Process model and return trained model and likelihood
def train_gp():
    train_x_mean = (np.linspace(-1, 1, 20) * (2 * np.pi)).reshape(-1,1)
    train_y = ((1 + np.cos(train_x_mean) + np.random.random(train_x_mean.shape) * np.sqrt(0.04))/ 4 + 0.4).reshape(-1,1)

    likelihood = gpytorch.likelihoods.GaussianLikelihood(batch_shape=torch.Size([1]))
    model = ExactGPModel(train_x_mean, train_y, likelihood)

    training_iter = 500
    trained_model, trained_likelihood = train_gp_model(model, likelihood, train_x_mean, train_y, training_iter)

    print("Training completed!")
    return trained_model, trained_likelihood

# Define a neural network wrapper for the trained Gaussian Process model
class GP_NN(torch.nn.Module):
    def __init__(self):
        super(GP_NN, self).__init__()
        self.trained_model, self.trained_likelihood = train_gp()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return self.trained_likelihood(self.trained_model(input)).mean