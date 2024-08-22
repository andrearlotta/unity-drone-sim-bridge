import numpy as np

def process_positions(tree_pos, drone_pos, drone_yaw):
    tree_pos = np.array(tree_pos)
    drone_pos = np.array(drone_pos)

    # Vector difference between drone and each tree
    diff = drone_pos - tree_pos

    # Calculate angle phi for each tree, and adjust by pi/2
    phi = np.arctan2(diff[:, 1], diff[:, 0]) + np.pi/2

    # Calculate Euclidean distance from drone to each tree
    distances = np.linalg.norm(diff, axis=1)

    # Normalize phi to be in the range [0, 2*pi]
    phi_normalized = np.mod(phi + 2 * np.pi, 2 * np.pi)

    # Compute the relative angle, adjust by pi/2 and normalize
    relative_angles = np.mod(phi - drone_yaw + np.pi + np.pi/2, 2 * np.pi)

    # Combine results into a single output array (if needed for further processing)
    result = np.vstack((distances, phi_normalized, relative_angles)).T

    return result

# Example usage:
tree_pos = [[10, 10], [20, 20], [30, 30]]  # example tree positions
drone_pos = [15, 15]                       # example drone position
drone_yaw = 0.5                            # example drone yaw

print(process_positions(tree_pos, drone_pos, drone_yaw))