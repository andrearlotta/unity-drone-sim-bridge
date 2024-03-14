import rospy
import cv_bridge
#from unity_drone_sim_bridge.srv import *
from sensor_msgs.msg import *
from std_msgs.msg import *
from geometry_msgs.msg import *
from cv_bridge import CvBridge

class SubscriberNode:

    def __init__(self, topic_dict=None):
        self.__set_params(topic_dict)
        self.__init_sub()
        self.__init_ros()

    def __set_params(self, dict):
        self.namespace      = rospy.get_namespace()
        self.sensor_name    = dict.get("name")
        self.topic_string   = dict.get("topic")
        self.global_name    = dict.get("mode") + "_node_" + dict.get("name")
        self.topic_type     = dict.get("type")
        self.data           = None
        self.msg2Data       = dict.get("serializer")

    def __init_sub(self):
        print(self.topic_type)
        rospy.Subscriber(self.topic_string, self.topic_type, self.callBack)
    
    def __init_ros(self):
        pass
        # Initialize the ROS node
        #rospy.init_node(self.global_name, anonymous=True, log_level=rospy.DEBUG)
        #rospy.spin()
        
    def setData(self,data):
        self.data = data

    def getData(self):
        return self.data

    def msg2Data(self):
        pass
    
    def callBack(self,msg):
        self.setData(self.msg2Data(msg))