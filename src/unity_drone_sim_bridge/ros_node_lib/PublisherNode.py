import rospy
#from unity_drone_sim_bridge.srv import *
from sensor_msgs.msg import *
from std_msgs.msg import *
from geometry_msgs.msg import *

class PublisherNode:

    def __init__(self, topic_dict=None):
        self.__set_params(topic_dict)
        self.__init_pub()
        self.__init_ros()
    
    def __init_ros(self):
        pass
        ## Initialize the ROS node
        #rospy.init_node(self.global_name, anonymous=True, log_level=rospy.DEBUG)
        #if self.rate == 0:     
        #    rospy.spin()
        #else:
        #    while self.data is None:        pass
        #    while not rospy.is_shutdown():  self.pubData()

    def __init_pub(self):
        self.pub = rospy.Publisher(self.topic_name, self.topic_type, queue_size=10)

    def __set_params(self, dict):
        self.namespace      =   rospy.get_namespace()
        self.sensor_name    = dict.get("sensor")
        self.topic_name     = dict.get("name")
        self.global_name    = dict.get("mode") + "_node_" + dict.get("name")
        self.topic_type     = dict.get("type")
        
        if dict.get("rate") is None:
            self.rate = 0
            self.pubData = self.__setAndPub
        else: 
            self.rate           = dict.get("rate")
            self.pubData = self.__setData

        self.data           = None
        self.data2Msg       = dict.get("serializer")

    def __setAndPub(self,data):
        self.setData(data)
        self.pubData(data)

    def __setData(self,data):
        self.data = data

    def getData(self):
        return self.data

    def __data2Msg(self):
        pass
    
    def __pubData(self):
        self.pub.publish(self.data2Msg())
