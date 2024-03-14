from unity_drone_sim_bridge.ros_node_lib.PublisherNode import PublisherNode
from unity_drone_sim_bridge.ros_node_lib.SubscriberNode import SubscriberNode
from unity_drone_sim_bridge.ros_node_lib.ClientNode import ClientNode
import rospy
class BridgeClass:
    def __init__(self, components_list = []):
        self.__publishers_dict   = self.__subscribers_dict = self.__clients_dict = {}
        self.SetupRosCom(components_list)
        rospy.init_node("sim_bridge", anonymous=True, log_level=rospy.DEBUG)
        rospy.spin()

    def addComponent(self, sensor):
        if      sensor.get("mode") == "pub": self.__publishers_dict[sensor.get("name")]   =  PublisherNode(sensor)
        elif    sensor.get("mode") == "sub": self.__subscribers_dict[sensor.get("name")]  =  SubscriberNode(sensor)
        elif    sensor.get("mode") == "srv": self.__clients_dict[sensor.get("name")]      =  ClientNode(sensor)

    def SetupRosCom(self, components_list):
        for component_info in components_list:
            self.addComponent(component_info)
    
    def getData(self):
        return {sensor: sub.getData() for sensor, sub in self.__subscribers_dict.values()}

    def pubData(self, data_dict):
        for sensor, data in data_dict.items():
            self.pubData(sensor, data)

    def callServer(self, req_dict):
        return {server: self.callServer(server,req) for server, req in req_dict.values()}
    
    def getData(self, sensor):
        return self.__subscribers_dict[sensor].getData()

    def pubData(self, pub, data):
        self.__publishers_dict[pub].pubData(data)

    def callServer(self, server, req):
        return self.__clients_dict[server].callServer(req)
    