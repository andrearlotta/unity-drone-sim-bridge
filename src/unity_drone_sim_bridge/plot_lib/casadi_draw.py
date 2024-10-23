import numpy as np
import plotly.graph_objects as go
import torch
from scipy.ndimage import maximum_filter, label, find_objects

class Draw_MPC_point_stabilization_v1(object):
    def __init__(self, robot_states: list, init_state: np.array, target_state: np.array = None, z=None,
                 rob_diam=0.3, export_fig=True, function_to_plot=None, trees=np.zeros((1, 2)),
                 obstacle=None, obstacle_r=None):
        self.robot_states = robot_states
        self.init_state = init_state.flatten()
        self.rob_radius = rob_diam / 2.0
        self.function_to_plot = function_to_plot
        self.z = np.array(z).flatten()
        self.obstacle = obstacle  # Obstacle center coordinates (tuple or list)
        self.obstacle_r = obstacle_r  # Obstacle radius

        # Create a single figure to hold both the trajectory and surface plot
        self.fig = go.Figure()

        # Plot the potential function if provided
        if function_to_plot:
            self.plot_function_result(function_to_plot, trees)

        # Plot the robot trajectory
        self.plot_trajectory()

        # Plot the obstacle as a circle
        if self.obstacle is not None and self.obstacle_r is not None:
            self.plot_obstacle()

        # Compute the distance between the final point and the maximum point(s)
        final_state = np.array(self.robot_states).reshape(-1, 3)[-1]
        final_position = final_state[:2]
        final_theta = final_state[2]

        distances = [np.linalg.norm(np.array(max_pt) - final_position) for max_pt in self.max_points]
        angle_errors = [self.compute_angle_error(final_theta, theta) for theta in self.max_thetas]

        print("Distances between final point and maximum point(s):", distances)
        print("Angle errors between final theta and maxima thetas (in radians):", angle_errors)
        print("Z-values at maximum points:", self.max_z_values)

        # Show the combined figure
        self.fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'),
                               title="Robot Trajectory and Function Surface")
        self.fig.show()

    def plot_trajectory(self):
        """Plot the trajectory of the robot with orientation in 3D."""
        # Extract positions and orientation from robot states
        states = np.array(self.robot_states).reshape(-1, 3)  # Assuming state is [x, y, theta]
        x = states[:, 0]
        y = states[:, 1]
        theta = states[:, 2]
        print(states)
        # Create traces for the trajectory
        trajectory_trace = go.Scatter3d(
            x=x, y=y, z=self.z,
            mode='lines+markers',
            line=dict(color='red', width=4),
            marker=dict(size=5, color='red'),
            name='Robot Trajectory'
        )

        # Plot orientation at each point using cones
        arrow_length = 1.0
        u = arrow_length * np.cos(theta)
        v = arrow_length * np.sin(theta)
        w = np.zeros_like(u)  # No change in z for orientation arrows

        # Normalize the arrows
        r = np.sqrt(u**2 + v**2 + w**2)
        u /= r
        v /= r
        w /= r

        orientation_trace = go.Cone(
            x=x, y=y, z=self.z,
            u=u, v=v, w=w,
            sizemode="absolute",
            sizeref=.2,
            showscale=False,
            colorscale=[[0, 'blue'], [1, 'blue']],
            name='Orientation'
        )

        # Add trajectory and orientation traces to the figure
        self.fig.add_trace(trajectory_trace)
        self.fig.add_trace(orientation_trace)

        # Add the initial and final positions
        self.fig.add_trace(go.Scatter3d(
            x=[self.init_state[0]], y=[self.init_state[1]], z=[self.z[0]],
            mode='markers', marker=dict(size=8, color='green'),
            name='Initial Position'
        ))

        self.fig.add_trace(go.Scatter3d(
            x=[x[-1]], y=[y[-1]], z=[self.z[-1]],
            mode='markers', marker=dict(size=8, color='red'),
            name='Final Position'
        ))

    def plot_function_result(self, func, trees):
        """Plot the maximum value of the function over theta for each (x, y) coordinate as a 3D surface."""
        x_vals = np.linspace(-10, 10, 100)
        y_vals = np.linspace(-10, 10, 100)
        X, Y = np.meshgrid(x_vals, y_vals)

        theta_vals = np.linspace(-np.pi, np.pi, 360)  # Discretize theta between -π and π

        # Initialize arrays to store the maximum values and corresponding theta
        Z_max = np.full_like(X, -np.inf)
        Theta_max = np.zeros_like(X)

        # Flatten the grid arrays for vectorized computation
        grid_xy = np.vstack([X.ravel(), Y.ravel()]).T  # Shape (N, 2)

        # Loop over all theta values to find the maximum function value at each (x, y)
        for theta in theta_vals:
            # Create input tensor for current theta
            theta_array = np.full((grid_xy.shape[0], 1), theta)
            grid_points = np.hstack([grid_xy, theta_array])  # Shape (N, 3)
            input_tensor = torch.tensor(grid_points, dtype=torch.float32)

            # Compute the function over the grid at current theta
            with torch.no_grad():
                Z_flat = func(input_tensor).cpu().numpy().flatten()

            # Reshape back to grid shape
            Z_theta = Z_flat.reshape(X.shape)

            # Update the maximum values and corresponding theta
            mask = Z_theta > Z_max
            Z_max[mask] = Z_theta[mask]
            Theta_max[mask] = theta

        Z = Z_max  # Use the maximum values for plotting

        for tree in trees:
            # Plot the function as a surface plot
            surface_trace = go.Surface(
                x=X + tree[0], y=Y + tree[1], z=Z,
                colorscale='Viridis', opacity=0.6,
                showscale=False
            )
            self.fig.add_trace(surface_trace)

        # Identify all maxima in the grid
        neighborhood_size = 5
        threshold = np.max(Z) * 0.99  # Define a threshold to find significant maxima

        data_max = maximum_filter(Z, neighborhood_size)
        maxima = (Z == data_max)
        maxima[Z < threshold] = False

        labeled, num_objects = label(maxima)
        slices = find_objects(labeled)
        x_peaks = []
        y_peaks = []
        theta_peaks = []
        z_peaks = []
        for dy, dx in slices:
            x_center = int((dx.start + dx.stop - 1) / 2)
            y_center = int((dy.start + dy.stop - 1) / 2)
            x_peaks.append(X[y_center, x_center])
            y_peaks.append(Y[y_center, x_center])
            theta_peaks.append(Theta_max[y_center, x_center])
            z_peaks.append(Z[y_center, x_center])

        self.max_points = list(zip(x_peaks, y_peaks))
        self.max_thetas = theta_peaks
        self.max_z_values = z_peaks

        # Plot all maxima
        for i, tree in enumerate(trees):
            arrow_length = 1.0
            u = arrow_length * np.cos(theta_peaks)
            v = arrow_length * np.sin(theta_peaks)
            w = np.zeros_like(u)
            r = np.sqrt(u**2 + v**2 + w**2)
            u /= r
            v /= r
            w /= r

            maxima_trace = go.Scatter3d(
                x=np.array(x_peaks) + tree[0], y=np.array(y_peaks) + tree[1], z=z_peaks,
                mode='markers+text',
                text=[f"{z_value:.2f}" for z_value in z_peaks],
                marker=dict(size=5, color='white'),
                name=f'Maxima {i+1} Points'
            )
            orientation_trace = go.Cone(
                x=np.array(x_peaks) + tree[0], y=np.array(y_peaks) + tree[1], z=z_peaks,
                u=u, v=v, w=w,
                sizemode="absolute",
                sizeref=0.2,
                showscale=False,
                colorscale=[[0, 'white'], [1, 'white']],
                name=f'Max {i+1} Orientation'
            )

            self.fig.add_trace(maxima_trace)
            self.fig.add_trace(orientation_trace)

    def plot_obstacle(self, num_points=100):
        for i in range(len(self.obstacle)):
            """Plot a circle on the surface of the function."""
            theta = np.linspace(0, 2 * np.pi, num_points)
            x_circle = self.obstacle[i,0] + self.obstacle_r * np.cos(theta)
            y_circle = self.obstacle[i,1] + self.obstacle_r * np.sin(theta)

            # Evaluate the function at the circle points
            if self.function_to_plot:
                states = np.vstack([x_circle -self.obstacle[i,0] , y_circle -self.obstacle[i,1], np.zeros_like(x_circle)]).T
                input_tensor = torch.tensor(states, dtype=torch.float32)
                with torch.no_grad():
                    z_circle = self.function_to_plot(input_tensor).cpu().numpy().flatten()
            else:
                z_circle = np.zeros_like(x_circle)

            # Plot the circle
            circle_trace = go.Scatter3d(
                x=x_circle, y=y_circle, z=z_circle,
                mode='lines',
                line=dict(color='cyan', width=4),
                name='Circle on Surface'
            )
            self.fig.add_trace(circle_trace)

    @staticmethod
    def compute_angle_error(theta1, theta2):
        """Compute the smallest difference between two angles."""
        delta_theta = theta1 - theta2
        delta_theta = (delta_theta + np.pi) % (2 * np.pi) - np.pi
        return abs(delta_theta)
