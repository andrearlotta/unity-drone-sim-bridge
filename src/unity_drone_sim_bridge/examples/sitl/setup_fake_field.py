from matplotlib import pyplot as plt
import numpy as np

def generate_tree_positions(grid_size, spacing):
    """Generate tree positions in a grid."""
    x_positions = np.arange(0, grid_size[0]*spacing, spacing)
    y_positions = np.arange(0, grid_size[1]*spacing, spacing)
    xv, yv = np.meshgrid(x_positions, y_positions)
    tree_positions = np.vstack([xv.ravel(), yv.ravel()]).T
    return tree_positions

def set_drone_position(tree_positions, min_distance):
    """Set the drone position ensuring it is at least `min_distance` away from all trees."""
    while True:
        print(tree_positions.shape)
        drone_pos = np.random.rand(2) * (np.max(tree_positions, axis=0) - np.min(tree_positions, axis=0)) + np.min(tree_positions, axis=0)
        distances = np.linalg.norm(tree_positions - drone_pos, axis=1)
        if np.all(distances >= min_distance):
            break
    drone_yaw = np.pi
    return np.array([4., 0.0, drone_yaw]).reshape(-1,1)

def viz_field(drone_position, tree_positions, border, grid_size):
    # Visualization
    plt.figure(figsize=(8, 8))
    plt.scatter(tree_positions[:, 0], tree_positions[:, 1], c='green', label='Trees')
    plt.scatter(drone_position[0], drone_position[1], c='red', label='Drone')
    plt.quiver(drone_position[0], drone_position[1], np.cos(drone_position[2]), np.sin(drone_position[2]), scale=5, color='red')
    plt.xlim([-border, grid_size[0] + border])
    plt.ylim([-border, grid_size[1] + border])
    plt.legend()
    plt.grid(True)
    plt.xlabel('X position (m)')
    plt.ylabel('Y position (m)')
    plt.title('Tree Positions and Drone Initial Position')
    plt.show()

    # Printing positions
    print("Tree Positions:")
    print(tree_positions)
    print("Drone Initial Position:")
    print(drone_position)