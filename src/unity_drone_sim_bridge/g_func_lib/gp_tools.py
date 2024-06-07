import torch
import gpytorch
import math
import numpy as np
import csv
import numpy as np
import casadi as ca
import matplotlib.pyplot as plt
from unity_drone_sim_bridge.gp_mpc_lib.gp_class import GP
from unity_drone_sim_bridge.g_func_lib.load_database import *

def LoadCaGP(synthetic=True, viz=True, N=40, method='ME'):
    # Limits in the training data
    ulb = []    # No inputs are used
    uub = []    # No inputs are used

    #N : Number of training data
    N_test = 100    # Number of test data

    if synthetic:
        X, Y, X_test, Y_test = loadSyntheticDatabase(N, N_test)
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
