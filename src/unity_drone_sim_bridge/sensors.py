from sensor_msgs.msg import *
from std_msgs.msg import *
from geometry_msgs.msg import *
from cv_bridge import CvBridge
import numpy as np
from unity_drone_sim_bridge.srv import GetTreesPoses
import tf 
bridge = CvBridge()

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
    "serializer":  lambda arr: Pose(position=Point(x=0.0, y=0.0, z=0.0), orientation=Quaternion(*tf.transformations.quaternion_from_euler(0, 0, arr[0,0]))),
  }
]