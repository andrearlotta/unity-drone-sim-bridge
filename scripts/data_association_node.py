#!/usr/bin/env python

import rospy
import numpy as np
import cv2
from sensor_msgs.msg import CompressedImage, CameraInfo
from vision_msgs.msg import Detection2DArray
from cv_bridge import CvBridge, CvBridgeError
from geometry_msgs.msg import PointStamped, Point
from visualization_msgs.msg import Marker, MarkerArray
from image_geometry import PinholeCameraModel
from scipy.spatial import cKDTree
from unity_drone_sim_bridge.srv import GetTreesPoses
from std_msgs.msg import Float32MultiArray
import tf
import tf.transformations as tf_trans

def weight_value(n_elements, mean_score, midpoint=15, steepness=3):
    # Sigmoidal weighting based on number of elements
    weight = 1 / (1 + np.exp(-steepness * (n_elements - midpoint)))
    
    return weight * mean_score + (1 - weight) * 0.5
class DataAssociationNode:
    def __init__(self):
        rospy.init_node('bounding_box_3d_pose', anonymous=True)

        self.bridge = CvBridge()
        self.camera_info = None
        self.camera_matrix = None
        self.depth_image = None
        self.tree_poses = None

        # Parameter to control visualization
        self.publish_visualization = rospy.get_param('~publish_visualization', True)

        self.detection_sub = rospy.Subscriber("/yolov7/detect", Detection2DArray, self.detection_callback)
        self.depth_image_sub = rospy.Subscriber("/camera/depth/image/compressed", CompressedImage, self.depth_image_callback)
        self.camera_info_sub = rospy.Subscriber("/camera/depth/camera_info", CameraInfo, self.camera_info_callback)
        
        # Publisher for tree scores
        self.scores_pub = rospy.Publisher("/tree_scores", Float32MultiArray, queue_size=10)
        
        # Conditional initialization of marker publisher
        if self.publish_visualization:
            self.marker_scores_pub = rospy.Publisher("/scores_markers", MarkerArray, queue_size=10)
            self.marker_fruits_pub = rospy.Publisher("/fruits_markers", MarkerArray, queue_size=10)
        
        self.cam_model = PinholeCameraModel()

        # TF listener
        self.tf_listener = tf.TransformListener()

        # Wait for the tree pose service to become available
        rospy.wait_for_service('/obj_pose_srv')
        self.get_trees_poses = rospy.ServiceProxy('/obj_pose_srv', GetTreesPoses)

        self.update_tree_poses()
        
        rospy.spin()


    def update_tree_poses(self):
        try:
            response = self.get_trees_poses()
            self.tree_poses = np.array([[pose.position.x, pose.position.y] for pose in response.trees_poses.poses])
        except rospy.ServiceException as e:
            rospy.logerr(f"Service call failed: {e}")

    def camera_info_callback(self, msg):
        self.camera_info = msg
        self.camera_matrix = np.array(self.camera_info.K).reshape(3, 3)
        self.cam_model.fromCameraInfo(msg)

    def depth_image_callback(self, msg):
        try:
            self.depth_image = self.bridge.compressed_imgmsg_to_cv2(msg, desired_encoding="passthrough")[:,:,0]
        except CvBridgeError as e:
            rospy.logerr(e)

    def uint8_to_distance(self, value, min_dist, max_dist):
        value = max(0, min(value, 255))
        fraction = value / 255.0
        distance = max_dist - fraction * (max_dist - min_dist)
        return distance

    def associate_fruits_to_trees(self, fruit_positions):
        if self.tree_poses is None or len(fruit_positions) == 0:
            return {}

        tree_kdtree = cKDTree(self.tree_poses)
        distances, tree_indices = tree_kdtree.query(fruit_positions[:, :2])

        tree_fruit_dict = {i: [] for i in range(len(self.tree_poses))}
        for fruit_index, (tree_index, distance) in enumerate(zip(tree_indices, distances)):
            if distance <= 2.5:  # Only associate fruits if distance is <= 2.5 meters
                tree_fruit_dict[tree_index].append(fruit_positions[fruit_index])

        return tree_fruit_dict

    def transform_fruit_positions(self, fruit_positions, header):
        transformed_positions = []
        for fruit_pos in fruit_positions:
            point_camera = PointStamped()
            point_camera.point= Point(*fruit_pos)
            point_camera.header.frame_id = 'depth_camera_frame'
            try:
                point_map = self.tf_listener.transformPoint('map', point_camera)
                transformed_positions.append([point_map.point.x, point_map.point.y, point_map.point.z])
            except (tf.LookupException, tf.ConnectivityException, tf.ExtrapolationException) as e:
                rospy.logerr(e)
                continue
        return np.array(transformed_positions)

    def detection_callback(self, msg):
        if self.depth_image is None or self.camera_matrix is None or self.tree_poses is None:
            return

        fruit_positions = []
        fruit_scores = []

        for detection in msg.detections:
            bbox = detection.bbox
            xmin = int(bbox.center.x - bbox.size_x / 2)
            xmax = int(bbox.center.x + bbox.size_x / 2)
            ymin = int(bbox.center.y - bbox.size_y / 2)
            ymax = int(bbox.center.y + bbox.size_y / 2)

            xmin, xmax = max(0, xmin), min(self.depth_image.shape[1], xmax)
            ymin, ymax = max(0, ymin), min(self.depth_image.shape[0], ymax)

            depth_roi = self.depth_image[int(bbox.center.y), int(bbox.center.x)]
            non_zero_depths = depth_roi[depth_roi > 11]
            if len(non_zero_depths) == 0:
                continue

            median_depth = np.median(non_zero_depths)
            center_x, center_y = int(bbox.center.x), int(bbox.center.y)

            XYZ = np.array(self.cam_model.projectPixelTo3dRay((center_x, center_y)))
            XYZ *= self.uint8_to_distance(median_depth, 0.05, 20)

            fruit_positions.append(XYZ)
            fruit_scores.append(detection.results[0].score)
            
        fruit_positions = np.array(fruit_positions)

        # Transform fruit positions from camera frame to map frame
        transformed_fruit_positions = self.transform_fruit_positions(fruit_positions, msg.header)

        associated_fruits = self.associate_fruits_to_trees(transformed_fruit_positions)

        tree_scores = np.ones(len(self.tree_poses)) * 0.5
        for i, fruits in associated_fruits.items():
            if fruits:
                fruit_indices = np.where((transformed_fruit_positions[:, None] == fruits).all(-1).any(-1))[0]
                tree_scores[i] = weight_value(len(fruit_indices), np.mean([fruit_scores[j] for j in fruit_indices]))

        # Publish tree scores
        scores_msg = Float32MultiArray()
        scores_msg.data = tree_scores.tolist()
        self.scores_pub.publish(scores_msg)

        # Conditional visualization
        if self.publish_visualization:
            markers = MarkerArray()

            """# Tree markers
            for i, (tree_pos, score) in enumerate(zip(self.tree_poses, tree_scores)):
                if score == 0:
                    continue

                # Sphere marker for the tree
                sphere_marker = Marker()
                sphere_marker.header = msg.header
                sphere_marker.header.frame_id = 'map'
                sphere_marker.ns = "tree_markers"
                sphere_marker.id = i * 2
                sphere_marker.type = Marker.CYLINDER
                sphere_marker.action = Marker.ADD
                sphere_marker.pose.position.x = tree_pos[0]
                sphere_marker.pose.position.y = tree_pos[1]
                sphere_marker.pose.position.z = 0.0
                sphere_marker.pose.orientation.w = 1.0
                sphere_marker.scale.x = sphere_marker.scale.y = sphere_marker.scale.z = 0.5
                sphere_marker.color.a = 1.0
                sphere_marker.color.r = 1.0 - score
                sphere_marker.color.g = score
                sphere_marker.color.b = 0.0

                markers.markers.append(sphere_marker)

                # Text marker for the score
                text_marker = Marker()
                text_marker.header = msg.header
                text_marker.header.frame_id = 'map'
                text_marker.ns = "tree_scores"
                text_marker.id = i * 2 + 1
                text_marker.type = Marker.TEXT_VIEW_FACING
                text_marker.action = Marker.ADD
                text_marker.pose.position.x = tree_pos[0]
                text_marker.pose.position.y = tree_pos[1]
                text_marker.pose.position.z = 2.5
                text_marker.text = f"Score: {score:.2f}"
                text_marker.scale.z = 0.2
                text_marker.color.a = 1.0
                text_marker.color.r = text_marker.color.g = text_marker.color.b = 1.0

                markers.markers.append(text_marker)
            self.marker_scores_pub.publish(markers)"""

            # Fruit markers
            for i, (fruit_pos, score) in enumerate(zip(transformed_fruit_positions, fruit_scores)):
                # Sphere marker for the fruit
                fruit_marker = Marker()
                fruit_marker.header = msg.header
                fruit_marker.header.frame_id = 'map'
                fruit_marker.ns = "fruit_markers"
                fruit_marker.id = len(self.tree_poses) * 2 + i * 2
                fruit_marker.type = Marker.SPHERE
                fruit_marker.action = Marker.ADD
                fruit_marker.pose.position.x = fruit_pos[0]
                fruit_marker.pose.position.y = fruit_pos[1]
                fruit_marker.pose.position.z = fruit_pos[2]
                fruit_marker.pose.orientation.w = 1.0
                fruit_marker.scale.x = fruit_marker.scale.y = fruit_marker.scale.z = 0.1
                fruit_marker.color.a = 1.0
                fruit_marker.color.r = 0.0
                fruit_marker.color.g = 1.0
                fruit_marker.color.b = 0.0

                markers.markers.append(fruit_marker)

            self.marker_fruits_pub.publish(markers)

if __name__ == '__main__':
    try:
        DataAssociationNode()
    except rospy.ROSInterruptException:
        pass
