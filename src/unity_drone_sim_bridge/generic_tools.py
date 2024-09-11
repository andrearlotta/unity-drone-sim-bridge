import casadi as ca
import numpy as np

'''
generic tools
'''

def drone_objects_distances_casadi(drone_pos_sym, objects_pos_sym, ray=0.0):
    # Calculate distance between the drone and each object
    return ca.sqrt(ca.sum2((objects_pos_sym - np.ones((objects_pos_sym.shape[0],1))@drone_pos_sym.T)**2)) - ray * np.ones((objects_pos_sym.shape[0],1))

def drone_objects_distances_np(drone_pos, objects_pos, ray=0.0):
    # Calculate distance between the drone and each object
    return np.linalg.norm(objects_pos - drone_pos.T, axis=1) - ray

def n_nearest_objects(drone_pos, objects_pos, num=4):
    # Get indices of the n nearest objects
    return np.argsort(drone_objects_distances_np(drone_pos, objects_pos))[:num]

def gaussian_ca(x, mu, sigma=1/ca.sqrt(2*ca.pi), norm=True):
    a = 1 if not norm else (sigma*ca.sqrt(2*ca.pi)) 
    return a * (
        1.0 / (ca.sqrt(2.0 * ca.pi) * sigma) * ca.exp(-ca.power((x - mu) / sigma, 2.0) / 2)
    )

def gaussian_np(x, mu, sigma=1/ca.sqrt(2*ca.pi), norm=True):
    a = 1 if not norm else (sigma*ca.sqrt(2*ca.pi)) 
    return a * (
        1.0 / (np.sqrt(2.0 * ca.pi) * sigma) * np.exp(-np.power((x - mu) / sigma, 2.0) / 2)
    )

def sigmoid_np(x, alpha=10.0):
    return 1 / (1 + np.exp(-alpha*x))

def norm_sigmoid_np(x, thresh = 6, delta = 0.5, alpha = 10.0):
    x_min = thresh - delta
    x_max = thresh + delta
    y_min = 0.0
    y_max = 1.0
    
    normalized_x = ((x - x_min) - (x_max - x_min)/2) / (x_max - x_min) 
    normalized_y = sigmoid_np(normalized_x, alpha)
    mapped_y = y_min + (normalized_y * (y_max - y_min))
    
    return mapped_y

def sigmoid_ca(x, alpha=10.0):
    return 1 / (1 + ca.exp(-alpha*x)) # ca.Functiontion('sigmoid' , [x] , [1 / (1 + ca.exp(-alpha*x))])

def norm_sigmoid_ca(x, thresh = 6, delta = 0.5, alpha = 10.0):
    x_min = thresh - delta
    x_max = thresh + delta
    y_min = 0.0
    y_max = 1.0
    
    normalized_x = ((x - x_min) - (x_max - x_min)/2) / (x_max - x_min) 
    normalized_y = sigmoid_ca(normalized_x, alpha)
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
    normalized_y = sigmoid_np(normalized_x, alpha=7.5)
    mapped_y = y_min + normalized_y * (y_max - y_min)
    
    return mapped_y

'''
fov tools
'''
def fov_weight_fun_polar_numpy(drone_yaw, objects_pos, thresh_distance=5):
    # Unpack polar coordinates from objects_pos
    distances = objects_pos[ 0]
    angles = objects_pos[1]
    
    # Calculate the relative angle between the drone's orientation and the direction to each object
    relative_angles = angles - drone_yaw
    # Normalize these angles to be within -pi to pi for correct directional alignment
    relative_angles = (relative_angles + np.pi) % (2 * np.pi) - np.pi
    
    # Compute vector alignment using cosine, as it effectively measures the directional agreement
    vect_alignment = np.cos(relative_angles)
    
    # Compute weights using a sigmoid function and a Gaussian distribution
    # Sigmoid to check alignment (vect_alignment close to 1 means good alignment)
    # Gaussian to give higher weight to closer objects
    return norm_sigmoid_np(vect_alignment, thresh=0.94, delta=0.03, alpha=2) * gaussian_np(distances, mu=thresh_distance, sigma=1)


def fov_weight_fun_numpy(drone_pos, drone_yaw, objects_pos, thresh_distance=5):
    n_objects = objects_pos.shape[0]
    # Calculate distance between the drone and each object
    distances = drone_objects_distances_np(drone_pos, objects_pos)
    # Calculate direction from drone to each object
    drone_objects_dir = objects_pos - np.tile(drone_pos.T, (n_objects,1))
    drone_yaw_dir = np.vstack((np.cos(drone_yaw), np.sin(drone_yaw)))
    vect_alignment = np.sum(drone_objects_dir / np.linalg.norm(drone_objects_dir, axis=1, keepdims=True) * drone_yaw_dir.T, axis=1)
    #return norm_sigmoid(vect_alignment, thresh=0.8, delta=0.5, alpha=1) * gaussian(distances, mu=thresh_distance, sigma=5.0)
    return  norm_sigmoid_np(vect_alignment,  thresh=0.94, delta=0.03, alpha=2) * gaussian_np(distances, mu=thresh_distance, sigma=1)

def _fov_weight_fun_casadi(drone_pos, drone_yaw, objects_pos, thresh_distance=5):
    n_objects = objects_pos.shape[0]
    # Define distance function
    distance_expr = ca.sqrt((drone_pos[0] - objects_pos[:, 0])**2 + (drone_pos[1] - objects_pos[:, 1])**2)
    
    #return  norm_sigmoid(vect_alignment, thresh=0.8, delta=0.5, alpha=1) * gaussian(distance_expr, mu=thresh_distance, sigma=5.0) 
    return   norm_sigmoid_ca(vect_alignment(drone_pos,drone_yaw,objects_pos),  thresh=0.94, delta=0.03, alpha=2)* gaussian_ca(distance_expr, mu=thresh_distance, sigma=1)

def vect_alignment(drone_pos,drone_yaw,objects_pos):
    n_objects = objects_pos.shape[0]
    drone_yaw_dir = ca.vertcat(ca.cos(drone_yaw), ca.sin(drone_yaw))
    drone_objects_dir = objects_pos - ca.repmat(drone_pos.T, n_objects, 1)
    
    normalized_directions = drone_objects_dir / ca.power(ca.sum2(ca.power(drone_objects_dir,2)),(1./2))
    return  ca.mtimes(normalized_directions, drone_yaw_dir)

def drone_obj_angle(drone_pos, object_pos):
    drone_object_dir = drone_pos-object_pos
    angle = ca.atan2(drone_object_dir[-1], drone_object_dir[0])
    return angle


def fov_weight_fun_casadi(thresh_distance=5):
    drone_pos_sym = ca.MX.sym('drone_pos', 2)       # 2-dimensional drone position
    drone_yaw_sym = ca.MX.sym('drone_yaw')          # drone yaw angle
    objects_pos_sym = ca.MX.sym('objects_pos', 2)   # Objects positions (variable number)
    distance_expr = ca.norm_2(objects_pos_sym - drone_pos_sym)
    # Calculate vector alignment
    normalized_directions = (objects_pos_sym - drone_pos_sym) / distance_expr
    drone_yaw_dir = ca.vertcat(ca.cos(drone_yaw_sym), ca.sin(drone_yaw_sym))

    vect_alignment = normalized_directions.T @ drone_yaw_dir

    # Define parameters for norm_sigmoid_ca and gaussian_ca
    thresh = 0.94
    delta = 0.03
    alpha = 2
    sigma = 1
    
    # Apply norm_sigmoid_ca and gaussian_ca
    sigmoid_output = norm_sigmoid_ca(vect_alignment, thresh, delta, alpha)
    gaussian_output = gaussian_ca(distance_expr, mu=thresh_distance, sigma=sigma)
    
    # Combine the outputs (element-wise multiplication)
    fov_result = sigmoid_output * gaussian_output
    # Create CasADi function object
    fov_function = ca.Function('fov_function', [drone_pos_sym, drone_yaw_sym, objects_pos_sym], [fov_result])
    

    return fov_function

def get_depth_value(depth_image, x, y, inv_matrix_param, window_size=50):
    window = depth_image[y-window_size:y+window_size, x-window_size:x+window_size]
    non_zero = window[ np.where(window != 0)]
    return inv_matrix_param.dot(np.array([x,y,1]).T).T * np.mean(non_zero) * 0.001 if len(non_zero) > 0 else np.zeros((3))

def get_centroid_pixel(depth_image):
    y, x = np.where((depth_image != 0) & (depth_image < 5 / 0.001))
    if len(y) == 0 or len(x) == 0: return (None, None)
    centroid_x = np.average(x, weights=depth_image[y, x])
    centroid_y = np.average(y, weights=depth_image[y, x])
    return (int(centroid_x),int(centroid_y))
