import numpy as np
import tensorflow as tf
import keras
import gpflow
import casadi
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
import csv
import os
import tf2onnx
import do_mpc

def std_norm(x):
    return (x - np.mean(x)) / np.std(x)

def rad2deg(rad):
    pi_on_180 = 0.017453292519943295
    return rad / pi_on_180

def map_angle(angle):
    return tf.math.floormod(angle + np.pi, 2 * np.pi) - np.pi

def wrapped_difference(alpha, beta):
    # Calculate the raw difference between angles alpha and beta
    return tf.atan2(tf.sin( beta - alpha), tf.cos( beta - alpha))

@tf.function
def filter_and_pred(xyp):
    mse = keras.losses.MeanSquaredError() #keras.losses.MeanAbsoluteError()
    xy, p = xyp
    mask = xy != -100
    # Apply the mask to the original tensor
    filtered_tensor = tf.where(mask, xy, tf.constant(tf.float32.min))
    # Remove rows where all elements are -100
    filtered_tensor = tf.boolean_mask(filtered_tensor, tf.reduce_any(mask, axis=-1))        
    amp_pred, freq_pred, phase_pred, off_pred = tf.unstack(p)
    fit_sine =amp_pred * \
        tf.sin(freq_pred * filtered_tensor[:,1][:-1]  + phase_pred) \
            + off_pred
    return mse( filtered_tensor[:,0][:-1],fit_sine)
    
@tf.function
def fit_mse(y_true, params_pred):
    return  tf.math.reduce_mean(tf.map_fn(fn=filter_and_pred, elems=(y_true,params_pred),fn_output_signature=tf.float32))

import casadi as ca

def sigmoid(x):
    return 1 / (1 + ca.exp(-10.0 * x))

def fake_nn(x):
    x_min = 1.125 
    x_max = 1.260 
    y_min = 0.5
    y_max = 0.8
    
    x_range = x_max - x_min
    half_x_range = x_range / 2
    
    normalized_x = (x - x_min - half_x_range) / x_range
    normalized_y = sigmoid(normalized_x)
    mapped_y = y_min + normalized_y * (y_max - y_min)
    
    return mapped_y

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
    # Convert inputs to CasADi symbols
    #drone_pos_sym = ca.MX(drone_pos.reshape(1,2))
    #drone_yaw_sym = ca.MX(drone_yaw)
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


def loadGp(mode='scikit'):
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

    if mode == 'scikit':
        kernel = 1.0 * ExpSineSquared(
            length_scale=1.0,
            periodicity=1.0,
        )
        gaussian_process = GaussianProcessRegressor(kernel=kernel)
        gaussian_process.fit(X_train, y_train)
        gaussian_process
    elif mode == 'gpflow':
        model = gpflow.models.GPR(X_train.reshape(-1,1), y_train.reshape(-1,1), kernel=gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(variance=.01, lengthscales=1.0), period=1.0), mean_function=None)
        class GPR(casadi.Callback):
            def __init__(self, name,  opts={}):
                casadi.Callback.__init__(self)
                self.construct(name, opts)

            def eval(self, arg):
                [mean,_] = model.predict_y(np.array(arg[0]))
                return [mean]

            # After instantiating the Callback, we end up with a plain-old CasADi function object which we can evaluate numerically or symbolically. One caveat, the user is responsible of keeping at least one reference to this Function.

            # x = casadi.MX.sym("x")
            # solver = casadi.nlpsol("solver","ipopt",{"x":x,"f":gpr(x)})
            # res = solver(x0=5)

        gaussian_process = GPR('GPR', {"enable_fd":True})
    elif mode == 'onnx':
        model = gpflow.models.GPR(X_train, y_train, gpflow.kernels.Periodic(gpflow.kernels.SquaredExponential(variance=.01, lengthscales=1.0), period=1.0))
        model_input_signature = [
            tf.TensorSpec(np.array((1, 1)), name='input'),
            ]
        output_path = os.path.join('models', 'model.onnx')

        onnx_model, _ = tf2onnx.convert.from_keras(model,
                output_path=output_path,
                input_signature=model_input_signature
        )
        return do_mpc.sysid.ONNXConversion(onnx_model) #casadi_converter

        # Inputs can be numpy arrays
        #casadi_converter.convert(input=np.ones((1,3)))

        # or CasADi expressions
        #x = casadi.SX.sym('x',1,3)
        #casadi_converter.convert(input=x)

        #Query the instance with the respective layer or node name to obtain the CasADi expression of the respective layer or node:

        #print(casadi_converter['output'])

    return gaussian_process
