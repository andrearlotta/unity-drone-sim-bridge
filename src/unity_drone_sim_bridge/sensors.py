from sensor_msgs.msg import *
from std_msgs.msg import *
from geometry_msgs.msg import *

SENSORS = [
  { 
  "name":  "camera",
  "type":  Image,
  "topic": "agent_0/rgb/data",
  "mode":  "sub",
  "serializer": None,
},
{ 
  "name":  "gps",
  "type":  Pose,
  "topic": "agent_0/pose",
  "mode":  "sub",
  "serializer": None,
},
{ 
  "name":  "gps",
  "type":  Pose,
  "topic": "agent_0/cmd/pose",
  "mode":  "pub",
  "serializer": None,
}
]