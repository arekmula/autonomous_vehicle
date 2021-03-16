#!/usr/bin/env python

# ROS imports
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# Custom ROS imports
from av_msgs.msg import Mode, States


class Collector:

    def __init__(self):
        # Front camera
        self.front_camera_topic_name = "/prius/front_camera/image_raw"
        self.front_camera_subscriber = rospy.Subscriber(self.front_camera_topic_name, data_class=Image,
                                                        callback=self.front_camera_image_callback, queue_size=1,
                                                        buff_size=2 ** 24)

        # Prius mode
        self.prius_mode_topic = "/prius/mode"
        self.prius_mode_sub = rospy.Subscriber(self.prius_mode_topic, Mode, self.prius_mode_callback)
        self.collect = False

        # Prius state
        self.prius_states_topic = "/prius/states"
        self.prius_state_sub = rospy.Subscriber(self.prius_states_topic, States, self.prius_state_callback)
        self.velocity = 0.0  # km/h
        self.steer_angle = 0.0  # TODO: radians? or degrees?

        # Cv Bridge
        self.cv_bridge = CvBridge()

        # Received image count
        self.received_image_count = 0

    def front_camera_image_callback(self, data):
        """
        Callback to "/prius/front_camera/image_raw" topic
        :param data: Input image
        :return:
        """

        # Collect the images and corresponding labels only if collect == True
        if self.collect:
            if data is not None:
                try:
                    cv_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
                except CvBridgeError as e:
                    print(e)

    def prius_mode_callback(self, data):
        """
        Callback to "/prius/mode" topic
        :param data: Input Mode message
        :return:
        """
        self.collect = data.collect

    def prius_state_callback(self, data):
        """
        Callback to "/prius/state" topic
        :param data: Input State message
        :return:
        """
        self.velocity = data.velocity
        self.steer_angle = data.steer_angle


if __name__ == "__main__":
    rospy.init_node("collector_node")
    collector = Collector()
    rospy.spin()