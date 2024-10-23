import casadi as ca
import numpy as np

'''
generic tools
'''
def drone_trees_distances_np(drone_pos, tree_pos):
    # Calculate distance between the drone and each tree
    return np.sqrt(np.sum((tree_pos - np.ones((tree_pos.shape[0], 1)) @ drone_pos.reshape(1, -1)) ** 2, axis=1))

def gaussian_ca(x, mu, sig=1/ca.sqrt(2*ca.pi), norm=True):
    a = 1 if not norm else (sig*ca.sqrt(2*ca.pi)) 
    return a * (1.0 / (ca.sqrt(2.0 * ca.pi) * sig) * ca.exp(-ca.power((x - mu) / sig, 2.0) / 2))

def gaussian_np(x, mu, sig=1/np.sqrt(2*np.pi), norm=True):
    a = 1 if not norm else (sig * np.sqrt(2 * np.pi))
    return a * (1.0 / (np.sqrt(2.0 * np.pi) * sig) * np.exp(-np.power((x - mu) / sig, 2.0) / 2))

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

def sigmoid_np(x, alpha=10.0):
    return 1 / (1 + np.exp(-alpha * x))

def norm_sigmoid_np(x, thresh=6, delta=0.5, alpha=10.0):
    x_min = thresh - delta
    x_max = thresh + delta
    y_min = 0.0
    y_max = 1.0
    
    normalized_x = ((x - x_min) - (x_max - x_min) / 2) / (x_max - x_min)
    normalized_y = sigmoid_np(normalized_x, alpha)
    mapped_y = y_min + (normalized_y * (y_max - y_min))
    
    return mapped_y

def trees_satisfy_conditions_np(drone_pos, trees = np.zeros((1,2)), thresh_distance=3):
    thresh = 0.9
    delta = 0.7
    alpha = 2.0
    sig = 1.5
    
    drone_pos = drone_pos.reshape((-1,1))
    # Calculate distance between the drone and each tree
    distances =  np.sqrt(np.sum((np.ones((trees.shape[0],1)) @ drone_pos[:2].T - trees)**2, axis=1))
    
    # Calculate direction from the drone (assuming angle at drone_pos[-1])
    drone_dir = np.array([np.cos(drone_pos[-1]), np.sin(drone_pos[-1])])

    # Calculate direction from the drone to each tree
    tree_directions = np.ones((trees.shape[0],1)) @ drone_pos[:2].T -  trees # Assuming trees are at the origin

    norm_tree_directions = -tree_directions / np.linalg.norm(tree_directions, axis=1, keepdims=True)

    # Compute vector alignment between drone direction and tree directions
    vect_alignment = np.sum(np.multiply(norm_tree_directions, np.ones((trees.shape[0],1)) @ drone_dir.T), axis=1)

    # Apply sigmoid and Gaussian functions
    alignment_score = norm_sigmoid_np(vect_alignment, thresh=thresh, delta=delta, alpha=alpha)
    angular_score = norm_sigmoid_np(np.abs(np.arctan2(tree_directions[:,1], tree_directions[:,0])), thresh=thresh, delta=delta, alpha=alpha)
    distance_score = gaussian_np(distances, mu=thresh_distance, sig=sig)
    # Return final score
    return 0.5 + alignment_score * distance_score * angular_score


def casadi_arctan2(y, x):
    """
    Compute the arctangent of y/x handling the correct quadrant using CasADi.

    Parameters:
    y (casadi.SX or casadi.MX): The y-coordinate (can be symbolic).
    x (casadi.SX or casadi.MX): The x-coordinate (can be symbolic).

    Returns:
    casadi.SX or casadi.MX: The angle between the positive x-axis and the point (x, y).
    """
    pi = ca.pi  # Pi constant in CasADi
    atan = ca.atan

    # Define conditions for different quadrants
    cond1 = x > 0
    cond2 = x < 0
    cond3 = y >= 0
    cond4 = y < 0
    cond5 = y > 0
    cond6 = y < 0
    cond7 = x == 0

    # Compute the angle using conditional expressions
    angle = ca.if_else(cond1, atan(y / x),
             ca.if_else(cond2,
                        ca.if_else(cond3, atan(y / x) + pi, atan(y / x) - pi),
                        ca.if_else(cond5, pi / 2,
                                   ca.if_else(cond6, -pi / 2, 0))))

    return angle

def fov_weight_fun_casadi(trees_pos, thresh_distance=3):
    thresh = 0.9
    delta = 0.7
    alpha = 2.0
    sig = 1.5
    
    # State variables
    x = ca.MX.sym("x")
    y = ca.MX.sym("y")
    theta = ca.MX.sym("theta")
    drone_pos = ca.vertcat(x, y, theta)
    trees_pos_sym = ca.MX.sym("trees_pos", trees_pos.shape)

    # Calculate distance between the drone and each tree
    distances = drone_objects_distances_casadi(drone_pos[:2], trees_pos_sym)

    # Calculate direction from drone to each tree
    drone_dir = ca.vertcat(ca.cos(theta), ca.sin(theta))

    tree_directions = ca.MX.ones(trees_pos_sym.shape[0], 1) @ drone_pos[:2].T - trees_pos_sym  # Correct broadcasting

    # Normalize the tree direction vector
    norm_factor = ca.sqrt(ca.sum2(tree_directions**2))  # Calculate the norm for each direction
    norm_tree_directions = -tree_directions / norm_factor  # Normalize the tree directions

    # Compute vector alignment between drone direction and tree directions
    vect_alignment = ca.sum2(ca.times(norm_tree_directions,  ca.MX.ones(trees_pos_sym.shape[0], 1) @ drone_dir.T))

    # Apply sigmoid and Gaussian functions
    alignment_score = norm_sigmoid_ca(vect_alignment, thresh=thresh, delta=delta, alpha=alpha)
    distance_score = gaussian_ca(distances, mu=thresh_distance, sig=sig)
    angular_score = norm_sigmoid_ca(ca.fabs(casadi_arctan2(tree_directions[:, 1], tree_directions[:, 0])), thresh=thresh,delta=delta, alpha=alpha)
    result = 0.5 + alignment_score * distance_score * angular_score

    return ca.Function('fov_function', [drone_pos, trees_pos_sym], [result])

def drone_objects_distances_casadi(drone_pos, objects_pos, ray=0.0):
    # Calculate distance between the drone and each tree
    return ca.sqrt(ca.sum2((ca.MX.ones(objects_pos.shape[0], 1) @ drone_pos.T - objects_pos) ** 2))

def drone_objects_distances_np(drone_pos, objects_pos, ray=0.0):
    # Calculate distance between the drone and each object
    return np.linalg.norm(objects_pos - np.ones((objects_pos.shape[0],1))@ drone_pos.reshape((1,2)), axis=1) - ray

def n_nearest_objects_np(drone_pos, objects_pos, num=4):
    # Get indices of the n nearest objects
    return np.argsort(drone_objects_distances_np(drone_pos, objects_pos))[:num]

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

def vect_alignment(drone_pos,drone_yaw,objects_pos):
    n_objects = objects_pos.shape[0]
    drone_yaw_dir = ca.vertcat(ca.cos(drone_yaw), ca.sin(drone_yaw))
    drone_objects_dir = objects_pos - ca.repmat(drone_pos.T, n_objects, 1)
    
    normalized_directions = drone_objects_dir / ca.power(ca.sum2(ca.power(drone_objects_dir,2)),(1./2))
    return  ca.mtimes(normalized_directions, drone_yaw_dir)

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
