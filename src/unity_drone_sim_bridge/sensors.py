from sensor_msgs.msg import *
from std_msgs.msg import *
from geometry_msgs.msg import *
from cv_bridge import CvBridge
import numpy as np
from unity_drone_sim_bridge.srv import GetTreesPoses
import tf 
import rospy
from vision_msgs.msg import Detection2D
bridge = CvBridge()
# Define the lambda function to create PoseArray from list of [x, y, yaw]

create_pose_array = lambda data: PoseArray(
    poses=[
        Pose(
            position=Point(x=data[0, i, 0], y=data[1, i, 0], z=0),
            orientation=Quaternion(*tf.transformations.quaternion_from_euler(0, 0, data[2, i, 0]))
        ) for i in range(data.shape[1])
    ]
)

# Update the SENSORS dictionary with the new entry for the Detection2D message
SENSORS = [
  { 
    "name":  "trees_poses",
    "type":  GetTreesPoses,
    "topic": "/obj_pose_srv",
    "mode":  "srv",
    "serializer": lambda pose_array_msg: np.array([[pose.position.x, pose.position.y] for pose in pose_array_msg.trees_poses.poses]),
  },    
  { 
    "name":  "image",
    "type":  Image,
    "topic": "agent_0/rgb",
    "mode":  "sub",
    "serializer": lambda img_msg: bridge.imgmsg_to_cv2(img_msg, desired_encoding="passthrough"),
  },
  {
    "name":  "gps",
    "type":  Pose,
    "topic": "agent_0/pose",
    "mode":  "sub",
    "serializer": lambda pose_msg: np.array([pose_msg.position.x, pose_msg.position.y, pose_msg.position.z,
                                          np.arctan2(2.0 * (pose_msg.orientation.w * pose_msg.orientation.z + pose_msg.orientation.x * pose_msg.orientation.y),
                                          1.0 - 2.0 * (pose_msg.orientation.y * pose_msg.orientation.y + pose_msg.orientation.z * pose_msg.orientation.z))]),
  },
  { 
    "name":  "cmd_pose",
    "type":  Pose,
    "topic": "agent_0/cmd/pose",
    "mode":  "pub",
    "serializer":  lambda arr: Pose(position=Point(x=arr[0,0], y=arr[1,0], z=0.0), orientation=Quaternion(*tf.transformations.quaternion_from_euler(0, 0, arr[2,0]))),
  },
  { 
    "name":  "viz_pred_pose",
    "type":  PoseArray,
    "topic": "viz/pred/traj",
    "mode":  "pub",
    "serializer":  create_pose_array,
  },
  { 
    "name":  "yolo_detector",
    "type":  Detection2D,
    "topic": "/yolov7/detect",
    "mode":  "sub",
    "serializer": lambda msg: [{ 
                          'id': result.id,
                          'score': result.score,
                          'center_x': result.bbox.center.x if result.bbox else None,
                          'center_y': result.bbox.center.y if result.bbox else None,
                          'size_x': result.bbox.size_x if result.bbox else None,
                          'size_y': result.bbox.size_y if result.bbox else None
                        } for result in msg.results],  # Add the callback function here
  }
]
