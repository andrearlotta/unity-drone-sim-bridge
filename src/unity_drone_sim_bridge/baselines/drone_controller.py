#!/usr/bin/env python

import rospy
import math
import numpy as np
import tf2_ros
import tf.transformations
from geometry_msgs.msg import Pose, Twist
import tf
import csv
import time
from unity_drone_sim_bridge.srv import GetTreesPoses
import plotly.graph_objects as go

class PIDController:
    def __init__(self, kp, kd):
        self.kp = kp
        self.kd = kd
        self.prev_error = 0

    def update(self, error):
        output = self.kp * error + self.kd * (error - self.prev_error)
        self.prev_error = error
        return output

class Logger:
    def __init__(self, filename='baselines/performance_metrics.csv'):
        self.filename = filename
        self.csv_file = open(self.filename, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        self.csv_writer.writerow(['Tree', 'Execution Time (s)', 'Distance (m)', 'Tree-to-Tree Time (s)'])
        self.performance_data = []
        self.start_time = None
        self.total_distance = 0

    def record_performance(self, tree_index, execution_time, distance, tree_to_tree_time):
        self.performance_data.append({
            'Tree': tree_index,
            'Execution Time': execution_time,
            'Distance': distance,
            'Tree-to-Tree Time': tree_to_tree_time
        })
        self.csv_writer.writerow([tree_index, execution_time, distance, tree_to_tree_time])
        self.csv_file.flush()

    def start_logging(self):
        self.start_time = time.time()

    def add_distance(self, distance):
        self.total_distance += distance

    def finalize_performance_metrics(self):
        total_time = time.time() - self.start_time
        avg_tree_to_tree_time = np.mean([data['Tree-to-Tree Time'] for data in self.performance_data])

        self.csv_writer.writerow([])
        self.csv_writer.writerow(['Total Execution Time (s)', total_time])
        self.csv_writer.writerow(['Total Distance (m)', self.total_distance])
        self.csv_writer.writerow(['Average Tree-to-Tree Time (s)', avg_tree_to_tree_time])
        self.csv_file.close()

        print(f"Total Execution Time: {total_time:.2f} s")
        print(f"Total Distance: {self.total_distance:.2f} m")
        print(f"Average Tree-to-Tree Time: {avg_tree_to_tree_time:.2f} s")

class TrajectoryGenerator:
    def __init__(self):
        rospy.init_node('trajectory_generator', anonymous=True)
        self.velocity_publisher = rospy.Publisher('/agent_0/cmd/pose', Pose, queue_size=1)
        self.rate = rospy.Rate(10)
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)
        
        self.pid_controller_x   = PIDController(kp=0.1, kd=0.01)
        self.pid_controller_y   = PIDController(kp=0.1, kd=0.01)
        self.pid_controller_yaw = PIDController(kp=0.1, kd=0.01)

        self.x = 0.0
        self.y = 0.0
        self.theta = 0.0
        self.vel_x = 0.0
        self.vel_y = 0.0

        self.logger = Logger()
        self.cruise_velocity = 0.5  # m/s
        self.circle_radius = 3.5  # Default circle radius

        rospy.wait_for_service('/obj_pose_srv')
        self.get_trees_poses = rospy.ServiceProxy('/obj_pose_srv', GetTreesPoses)
        self.tree_positions = self.get_tree_positions()

        self.safe_distance = 2.0  # Minimum safe distance from trees (in meters)
        self.repulsive_gain = 5.0  # Gain for the repulsive force
        self.attractive_gain = 1.0  # Gain for the attractive force
        self.max_repulsive_distance = 5.0  # Maximum distance for repulsive force to take effect
        self.max_velocity = 1.0  # Maximum velocity of the drone

    def get_tree_positions(self):
        try:
            pose_array_msg = self.get_trees_poses()
            return np.array([[pose.position.x, pose.position.y] for pose in pose_array_msg.trees_poses.poses])
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")
            return []

    def get_robot_position(self):
        not_ok = True
        while not_ok:
            try:
                trans = self.tf_buffer.lookup_transform("map", "drone_base_link", rospy.Time(0), rospy.Duration(0.1))
                _, _, yaw = tf.transformations.euler_from_quaternion([
                    trans.transform.rotation.x,
                    trans.transform.rotation.y,
                    trans.transform.rotation.z,
                    trans.transform.rotation.w
                ])
                return np.array([
                    trans.transform.translation.x,
                    trans.transform.translation.y,
                    yaw
                ])
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
                rospy.logerr(f"TF lookup failed: {e}")

    def polynomial_time_scaling_3rd_order(self, x0, xT, v0, vT, T):
        A = np.array([
            [0, 0, 0, 1],
            [T**3, T**2, T, 1],
            [0, 0, 1, 0],
            [3*T**2, 2*T, 1, 0]
        ])
        b = np.array([x0, xT, v0, vT])
        coeffs = np.linalg.solve(A, b)
        return coeffs[::-1]

    def move_to_point(self, current_x, current_y, target_x, target_y, next_x, next_y):
        start_time = time.time()
        distance = math.sqrt((target_x - current_x)**2 + (target_y - current_y)**2)
        self.logger.add_distance(distance)
        T = distance / self.cruise_velocity
        
        initial_vx = self.vel_x
        initial_vy = self.vel_y
        
        if next_x is not None and next_y is not None:
            final_vx = 0.1 * (next_x - target_x) / math.sqrt((next_x - target_x)**2 + (next_y - target_y)**2)
            final_vy = 0.1 * (next_y - target_y) / math.sqrt((next_x - target_x)**2 + (next_y - target_y)**2)
        else:
            final_vx = 0
            final_vy = 0
        
        trajectory_x = self.polynomial_time_scaling_3rd_order(current_x, target_x, initial_vx, final_vx, T)
        trajectory_y = self.polynomial_time_scaling_3rd_order(current_y, target_y, initial_vy, final_vy, T)
        
        start_time = rospy.get_time()
        
        while not rospy.is_shutdown():
            current_time = rospy.get_time() - start_time
            
            if current_time > T:
                break
            
            desired_x = trajectory_x[0] + trajectory_x[1]*current_time + trajectory_x[2]*current_time**2 + trajectory_x[3]*current_time**3
            desired_y = trajectory_y[0] + trajectory_y[1]*current_time + trajectory_y[2]*current_time**2 + trajectory_y[3]*current_time**3
            desired_vx = trajectory_x[1] + 2*trajectory_x[2]*current_time + 3*trajectory_x[3]*current_time**2
            desired_vy = trajectory_y[1] + 2*trajectory_y[2]*current_time + 3*trajectory_y[3]*current_time**2
            
            desired_theta = math.atan2(desired_vy, desired_vx)
            
            self.x, self.y, self.theta = self.get_robot_position()
            
            # Use PID controllers
            x_error = desired_x - self.x
            y_error = desired_y - self.y
            yaw_error = self.normalize_angle(desired_theta - self.theta)
            
            x_output = self.pid_controller_x.update(x_error)
            y_output = self.pid_controller_y.update(y_error)
            yaw_output = self.pid_controller_yaw.update(yaw_error)
            
            pose_msg = Pose()
            pose_msg.position.x = x_output
            pose_msg.position.y = y_output
            pose_msg.position.z = 0  # Assuming 2D movement
            
            # Convert the desired orientation to a quaternion
            quat = tf.transformations.quaternion_from_euler(0, 0, yaw_output)
            pose_msg.orientation.x = quat[0]
            pose_msg.orientation.y = quat[1]
            pose_msg.orientation.z = quat[2]
            pose_msg.orientation.w = quat[3]
            
            self.velocity_publisher.publish(pose_msg)
            
            self.rate.sleep()
        
        return time.time() - start_time

    def normalize_angle(self, angle):
        while angle > math.pi:
            angle -= 2 * math.pi
        while angle < -math.pi:
            angle += 2 * math.pi
        return angle

    def shortest_path(self, start_position, tree_positions):
        start_pos = np.array(start_position[:2]).reshape(1, 2)
        tree_positions = np.array(tree_positions)
        all_positions = np.vstack((start_pos, tree_positions))

        n = len(all_positions)
        unvisited = set(range(1, n))  # All points except the starting point
        path = [0]
        total_distance = 0

        current_node = 0
        while unvisited:
            next_node = min(unvisited, key=lambda node: np.linalg.norm(all_positions[current_node] - all_positions[node]))
            total_distance += np.linalg.norm(all_positions[current_node] - all_positions[next_node])
            current_node = next_node
            path.append(current_node)
            unvisited.remove(next_node)

        # Convert path indices to coordinates
        ordered_path = [all_positions[node].tolist() for node in path]

        return ordered_path, total_distance

    def move_around_point(self, center_x, center_y, radius, cruise_velocity):
        start_time = time.time()
        angular_velocity = cruise_velocity / radius
        
        # Calculate the total time needed for a full circle
        total_time = 2 * math.pi / angular_velocity
        
        current_time = 0
        self.x, self.y, self.theta = self.get_robot_position()
        start_angle = math.atan2(self.y - center_y, self.x - center_x)
        
        while current_time < total_time and not rospy.is_shutdown():
            angle = start_angle + angular_velocity * current_time
            
            desired_x = center_x + radius * math.cos(angle)
            desired_y = center_y + radius * math.sin(angle)
            desired_theta = angle + np.pi # Perpendicular to the circle, facing the center
            
            self.x, self.y, self.theta = self.get_robot_position()
            
            # Use PID controllers
            x_error = desired_x - self.x
            y_error = desired_y - self.y
            yaw_error = self.normalize_angle(desired_theta - self.theta)
            
            x_output = self.pid_controller_x.update(x_error)
            y_output = self.pid_controller_y.update(y_error)
            yaw_output = self.pid_controller_yaw.update(yaw_error)
            
            pose_msg = Pose()
            pose_msg.position.x = x_output
            pose_msg.position.y = y_output
            pose_msg.position.z = 0  # Assuming 2D movement
            
            # Convert the desired orientation to a quaternion
            quat = tf.transformations.quaternion_from_euler(0, 0, yaw_output)
            pose_msg.orientation.x = quat[0]
            pose_msg.orientation.y = quat[1]
            pose_msg.orientation.z = quat[2]
            pose_msg.orientation.w = quat[3]
            
            self.velocity_publisher.publish(pose_msg)
            
            self.rate.sleep()
            current_time = time.time() - start_time

        circle_distance = 2 * math.pi * radius
        self.logger.add_distance(circle_distance)
        return time.time() - start_time, circle_distance
    
    def calculate_repulsive_force(self, current_position, tree_positions):
        """Calculate the repulsive force from all trees."""
        repulsive_force = np.zeros(2)
        for tree in tree_positions:
            tree = np.array(tree)
            distance = np.linalg.norm(current_position - tree)
            if distance < self.max_repulsive_distance:
                force_magnitude = self.repulsive_gain * (1/distance - 1/self.max_repulsive_distance) * (1/distance**2)
                force_direction = (current_position - tree) / distance
                repulsive_force += force_magnitude * force_direction
        return repulsive_force

    def calculate_attractive_force(self, current_position, goal_position):
        """Calculate the attractive force towards the goal."""
        distance = np.linalg.norm(goal_position - current_position)
        return self.attractive_gain * (goal_position - current_position) / distance

    def move_to_point_and_circle(self, current_x, current_y, target_x, target_y, next_x, next_y):
        start_time = time.time()
        
        current_position = np.array([current_x, current_y])
        goal_position = np.array([target_x, target_y])
        
        total_time = 0
        total_distance = 0
        
        while np.linalg.norm(current_position - goal_position) > self.circle_radius:
            self.x, self.y, self.theta = self.get_robot_position()
            current_position = np.array([self.x, self.y])
            
            repulsive_force = self.calculate_repulsive_force(current_position, self.tree_positions)
            attractive_force = self.calculate_attractive_force(current_position, goal_position)
            
            total_force = repulsive_force + attractive_force
            
            # Normalize and scale the force to get the desired velocity
            velocity = (total_force / np.linalg.norm(total_force)) * self.max_velocity
            
            desired_x = current_position[0] + velocity[0]
            desired_y = current_position[1] + velocity[1]
            desired_theta = math.atan2(velocity[1], velocity[0])
            
            # Use PID controllers
            x_error = desired_x - self.x
            y_error = desired_y - self.y
            yaw_error = self.normalize_angle(desired_theta - self.theta)
            
            x_output = self.pid_controller_x.update(x_error)
            y_output = self.pid_controller_y.update(y_error)
            yaw_output = self.pid_controller_yaw.update(yaw_error)
            
            pose_msg = Pose()
            pose_msg.position.x = x_output
            pose_msg.position.y = y_output
            pose_msg.position.z = 0  # Assuming 2D movement
            
            # Convert the desired orientation to a quaternion
            quat = tf.transformations.quaternion_from_euler(0, 0, yaw_output)
            pose_msg.orientation.x = quat[0]
            pose_msg.orientation.y = quat[1]
            pose_msg.orientation.z = quat[2]
            pose_msg.orientation.w = quat[3]
            
            self.velocity_publisher.publish(pose_msg)
            
            self.rate.sleep()
            
            # Update total distance and time
            step_distance = np.linalg.norm(velocity)
            total_distance += step_distance
            total_time += 1.0 / 10  # Assuming 10 Hz control rate
        
        # Now circle around the point
        circle_time, circle_distance = self.move_around_point(target_x, target_y, self.circle_radius, self.cruise_velocity)
        
        total_time += circle_time
        total_distance += circle_distance
        
        return total_time, total_distance
    
    def run(self):
        self.logger.start_logging()
        
        start_position = self.get_robot_position()
        path, total_distance = self.shortest_path(start_position, self.tree_positions)

        self.plot_path(path,self.tree_positions)
        
        for i in range(1, len(path)):
            current_x, current_y = self.get_robot_position()[:2]
            target_x, target_y = path[i]
            next_x, next_y = path[i+1] if i+1 < len(path) else (None, None)
            
            # Move to the tree and circle around it
            total_time, total_distance = self.move_to_point_and_circle(current_x, current_y, target_x, target_y, next_x, next_y)
            
            self.logger.record_performance(i-1, total_time, total_distance, total_time - 2*math.pi*self.circle_radius/self.cruise_velocity)
            
            rospy.sleep(1)  # Wait for 1 second between trees

        self.logger.finalize_performance_metrics()
        print("Trajectory generation and inspection complete")


    def plot_path(self, path, tree_positions):
        tree_x = [tree[0] for tree in tree_positions]
        tree_y = [tree[1] for tree in tree_positions]
        path_x = [point[0] for point in path]
        path_y = [point[1] for point in path]

        fig = go.Figure()

        # Plot trees
        fig.add_trace(go.Scatter(
            x=tree_x, y=tree_y,
            mode='markers',
            marker=dict(size=10, color='green'),
            name='Trees'
        ))

        # Plot path
        fig.add_trace(go.Scatter(
            x=path_x, y=path_y,
            mode='lines+markers',
            line=dict(color='blue', width=2),
            name='Path'
        ))

        # Plot start and end points
        fig.add_trace(go.Scatter(
            x=[path_x[0]], y=[path_y[0]],
            mode='markers',
            marker=dict(size=15, color='red', symbol='star'),
            name='Start'
        ))
        fig.add_trace(go.Scatter(
            x=[path_x[-1]], y=[path_y[-1]],
            mode='markers',
            marker=dict(size=15, color='yellow', symbol='star'),
            name='End'
        ))

        # Add tree labels
        for i, tree in enumerate(tree_positions):
            fig.add_annotation(
                x=tree[0], y=tree[1],
                text=f'T{i}',
                showarrow=False,
                yshift=10
            )

        fig.update_layout(
            title='Drone Path for Tree Inspection',
            xaxis_title='X coordinate',
            yaxis_title='Y coordinate',
            showlegend=True
        )

        fig.show()
if __name__ == '__main__':
    try:
        trajectory_generator = TrajectoryGenerator()
        trajectory_generator.run()
    except rospy.ROSInterruptException:
        pass