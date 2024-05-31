import rospy
#from unity_drone_sim_bridge.srv import *
from cv_bridge import CvBridge

class QiNode:
    def __init__(self):
        self.global_name    =   "qi_fitting_node"
        self.namespace      =   rospy.get_namespace()
        self.cv_bridge      = CvBridge()
        self.__get_params()
        self.__init_ros()
    
    def __init_ros(self):
        # Initialize the ROS node
        rospy.init_node(self.global_name, anonymous=True, log_level=rospy.DEBUG)
        # Initialize Publishers and Subscribers
        rospy.spin()
        #while not rospy.is_shutdown():
        #   callClient


    def __get_params(self):
        self.srv_name  = rospy.get_param(f"{self.namespace}srv/img_pos")

    def getSrvName(self):
        return self.srv_name

    def pos2Msg(self, pos):
        pass
    
    def msg2Pos(self, msg):
        pass
    
    def msg2Img(self, msg):
        pass
    
    def resp2Dict(self, resp):
        pass

    def callServer(self, cmd):
        rospy.wait_for_service(self.getSrvName())
        try:
            srv_call = rospy.ServiceProxy(self.getSrvName(), ImgPosCmd)
            return self.resp2Dict(srv_call(cmd))
        except rospy.ServiceException as e:
            print(f"Service call failed: {e}")
