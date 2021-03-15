#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError


class Visualizer:

    def __init__(self):
        # Front camera
        self.front_camera_topic_name = "/prius/front_camera/image_raw"
        self.front_camera_subscriber = rospy.Subscriber(self.front_camera_topic_name, data_class=Image,
                                                        callback=self.image_callback, queue_size=1, buff_size=2 ** 24)

        # Visualization topic and publisher
        self.visualization_topic = "/prius/visualization"
        self.visualization_pub = rospy.Publisher(self.visualization_topic, Image, queue_size=1)

        # Cv Bridge
        self.cv_bridge = CvBridge()

    def image_callback(self, data):
        if data is not None:
            try:
                cv_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)


if __name__ == "__main__":
    rospy.init_node("visualizer_node")
    visualizer = Visualizer()
    rospy.spin()
