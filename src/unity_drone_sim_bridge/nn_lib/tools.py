import numpy as np
import tensorflow as tf
import keras
import cv2

def saturation(I):
    # Saturation is computed as the standard deviation of the color channels
    R = I[:, :, 0]
    G = I[:, :, 1]
    B = I[:, :, 2]

    mu = (R + G + B) / 3

    return np.sqrt(((R - mu) ** 2 + (G - mu) ** 2 + (B - mu) ** 2) / 3)

def relativeLuminance(I):
    return 0.2126 * I[:, :, 0] + 0.7152 * I[:, :, 1] + 0.0722 * I[:, :, 2]

def imhist(img):
    hist = cv2.calcHist([(img*255.0).astype(np.uint8)], [0], None, [256], [0, 256])
    return  hist / hist.sum()

def img_2_qi(img):
    img = img.astype(float) /255.0
    s_hist = np.sum(np.array(imhist(saturation(img.astype(float))))[:50, 0])
    l_hist  = np.sum(np.array(imhist(relativeLuminance(img.astype(float))))[50:200, 0])
    return (s_hist + l_hist)

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
    