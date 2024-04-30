import numpy as np
import tensorflow as tf
import keras
import casadi
from sklearn.gaussian_process.kernels import ExpSineSquared
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel
import csv
import os
import tf2onnx
import do_mpc
import math
import torch
import gpytorch
from unity_drone_sim_bridge.nn_lib.gp_nn_tools import GP_NN, loadDatabase, loadSyntheticData, LoadCaGP


config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)


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
    y_max = 0.7
    
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

def g(drone_yaw_sym, gp, thresh_distance=7):
    return  np.ones(drone_yaw_sym.shape) * gp.predict(ca.fmod(drone_yaw_sym,np.pi), [], np.zeros((1,1)))[0] #np.ones(drone_yaw_sym.shape) * ca.cos(drone_yaw_sym) + 1

def bayes(lambda_k,y_z):
    return ca.times(lambda_k, y_z) / (ca.times(lambda_k, y_z) + (1-lambda_k) * (1-y_z))

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

def loadGp(mode='GP_MPC'):
    if mode == 'scikit' or mode =='onnx':
        kernel = 1.0 * ExpSineSquared(
            length_scale=1.0,
            periodicity=1.0,
        )
        gaussian_process = GaussianProcessRegressor(kernel=kernel)
        gaussian_process.fit(X_train, y_train)

        if mode =='onnx':
            import skl2onnx
            import onnxruntime
            import onnxconverter_common
            initial_type = [("X", onnxconverter_common.FloatTensorType([1, 1]))]
            initial_type = [('X', onnxconverter_common.FloatTensorType([None, X_train.shape[1]]))]
            onx =  skl2onnx.convert_sklearn(gaussian_process, initial_types=initial_type,target_opset=9)
            gaussian_process = do_mpc.sysid.ONNXConversion(onx)

            #onx64 = skl2onnx.convert_sklearn(gaussian_process, initial_types=initial_type, target_opset=19)
            onx64 = skl2onnx.to_onnx(gaussian_process, X[:1])
            
            sess64 = onnxruntime.InferenceSession(
                onx64.SerializeToString(), providers=["CPUExecutionProvider"]
            )

            from onnxsim import simplify
            # convert model
            model_simp, check = simplify(onx)
            assert check, "Simplified ONNX model could not be validated"

    elif mode == 'gpytorch':
        import l4casadi as l4c
        nn = GP_NN()
        l4c_model = l4c.L4CasADi(nn, model_expects_batch_dim=True, device='cpu', mutable=True)  # device='cuda' for GPU
        gaussian_process = l4c_model

    elif mode == 'gpflow':
        import gpflow
        X_train, y_train = loadDatabase()
        model = gpflow.models.GPR((X_train.reshape(-1,1), y_train.reshape(-1,1)), gpflow.kernels.Constant(1) + gpflow.kernels.Linear(1) + gpflow.kernels.White(1) + gpflow.kernels.RBF(1), mean_function=None, noise_variance=1.0)
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables, options=dict(maxiter=100))
        # Package the resulting regression model in a CasADi callback
        class GPR(casadi.Callback):
            def __init__(self, name, opts: dict = None):
                if opts is None:
                    opts = dict()

                casadi.Callback.__init__(self)
                self.construct(name, opts)

            def eval(self, arg):
                [mean, _] = model.predict_f(np.array(arg[0]))
                return [mean.numpy()]
        # Instantiate the Callback (make sure to keep a reference to it!)
        gpr = GPR('GPR', opts={"enable_fd": True})
        print(gpr)
        return gpr

    elif mode == 'GP_MPC':
        return LoadCaGP()
        
    return gaussian_process
