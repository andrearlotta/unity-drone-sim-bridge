import casadi as ca
import numpy as np

'''
generic tools
'''

def drone_objects_distances_casadi(drone_pos_sym, objects_pos_sym):
    # Calculate distance between the drone and each object
    return ca.sqrt(ca.sum2((objects_pos_sym - np.ones((objects_pos_sym.shape[0],1))@drone_pos_sym.T)**2))

def drone_objects_distances_np(drone_pos, objects_pos):
    # Calculate distance between the drone and each object
    return np.linalg.norm(objects_pos - drone_pos, axis=1)

def n_nearest_objects(drone_pos, objects_pos, num=4):
    # Get indices of the n nearest objects
    return np.argsort(drone_objects_distances_np(drone_pos, objects_pos))[:num]

def gaussian(x, mu, sigma=1/ca.sqrt(2*ca.pi), norm=True):
    a = 1 if not norm else (sigma*ca.sqrt(2*ca.pi)) 
    return a * (
        1.0 / (ca.sqrt(2.0 * ca.pi) * sigma) * np.exp(-ca.power((x - mu) / sigma, 2.0) / 2)
    )

def sigmoid(x, alpha=10.0):
    return 1 / (1 + np.exp(-alpha*x))

def norm_sigmoid(x, thresh = 6, delta = 0.5, alpha = 10.0):
    x_min = thresh - delta
    x_max = thresh + delta
    y_min = 0.0
    y_max = 1.0
    
    normalized_x = ((x - x_min) - (x_max - x_min)/2) / (x_max - x_min) 
    normalized_y = sigmoid(normalized_x, alpha)
    mapped_y = y_min + (normalized_y * (y_max - y_min))
    
    return mapped_y

def fake_nn(x):
    x_min = 1.100 
    x_max = 1.290 
    y_min = 0.50
    y_max = 0.69
    
    x_range = x_max - x_min
    half_x_range = x_range / 2
    
    normalized_x = (x - x_min - half_x_range) / x_range
    normalized_y = sigmoid(normalized_x, alpha=7.5)
    mapped_y = y_min + normalized_y * (y_max - y_min)
    
    return mapped_y

'''
fov tools
'''

def fov_weight_fun_numpy(drone_pos, drone_yaw, objects_pos, thresh_distance=5):
    n_objects = objects_pos.shape[0]
    # Calculate distance between the drone and each object
    distances = drone_objects_distances_np(drone_pos, objects_pos)
    # Calculate direction from drone to each object    
    drone_objects_dir = objects_pos - np.tile(drone_pos, (n_objects, 1))
    drone_yaw_dir = np.vstack((np.cos(drone_yaw), np.sin(drone_yaw)))
    vect_alignment = np.sum(drone_objects_dir / np.linalg.norm(drone_objects_dir, axis=1, keepdims=True) * drone_yaw_dir.T, axis=1)
    #return norm_sigmoid(vect_alignment, thresh=0.8, delta=0.5, alpha=1) * gaussian(distances, mu=thresh_distance, sigma=5.0)
    return  norm_sigmoid(vect_alignment,  thresh=0.94, delta=0.03, alpha=2) * gaussian(distances, mu=thresh_distance, sigma=1)

def fov_weight_fun_casadi(drone_pos, drone_yaw, objects_pos, thresh_distance=5):
    n_objects = objects_pos.shape[0]
    # Define distance function
    distance_expr = ca.sqrt((drone_pos[0] - objects_pos[:, 0])**2 + (drone_pos[1] - objects_pos[:, 1])**2)
    drone_yaw_dir = ca.vertcat(ca.cos(drone_yaw), ca.sin(drone_yaw))
    drone_objects_dir = objects_pos - ca.repmat(drone_pos.T, n_objects, 1)
    
    normalized_directions = drone_objects_dir / ca.power(ca.sum2(ca.power(drone_objects_dir,2)),(1./2))
    vect_alignment = ca.mtimes(normalized_directions, drone_yaw_dir)

    #return  norm_sigmoid(vect_alignment, thresh=0.8, delta=0.5, alpha=1) * gaussian(distance_expr, mu=thresh_distance, sigma=5.0) 
    return   norm_sigmoid(vect_alignment,  thresh=0.94, delta=0.03, alpha=2)* gaussian(distance_expr, mu=thresh_distance, sigma=1)
