#!/usr/bin/env python

# ROS imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from prius_msgs.msg import Control

# Python imports
import cv2


class Visualizer:

    def __init__(self):
        # Front camera
        self.front_camera_topic_name = "/prius/front_camera/image_raw"
        self.front_camera_subscriber = rospy.Subscriber(self.front_camera_topic_name, data_class=Image,
                                                        callback=self.front_camera_image_callback, queue_size=1,
                                                        buff_size=2 ** 24)

        # Visualization topic and publisher
        self.visualization_topic = "/prius/visualization"
        self.visualization_pub = rospy.Publisher(self.visualization_topic, Image, queue_size=1)

        # Prius control
        self.prius_control_pub = rospy.Publisher("/prius", Control, self.prius_control_callback)
        self.throttle = 0.0  # Range 0 to 1, 1 is max throttle
        self.brake = 0.0  # Range 0 to 1, 1 is max brake
        self.steer = 0.0  # Range -1 to +1, +1 is maximum left turn
        self.shift_gears = 0
        self.prius_control_messages = {"throttle": self.throttle,
                                       "brake": self.brake,
                                       "steer": self.steer,
                                       "shift_gears": self.shift_gears}

        # Cv Bridge
        self.cv_bridge = CvBridge()

    # Front camera image callback
    def front_camera_image_callback(self, data):
        """
        Callback to "/prius/front_camera/image_raw" topic

        :param data: Input image
        :return:
        """
        if data is not None:
            try:
                cv_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)

            # Draw prius control messages
            cv_image = self.draw_prius_control(cv_image)

            visualization_msg = self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.visualization_pub.publish(visualization_msg)

    # Prius control message callback
    def prius_control_callback(self, data):
        """
        Callback to "/prius" topic

        :param data: Input Control message
        :return:
        """
        self.throttle = data.throttle
        self.brake = data.brake
        self.steer = data.steer
        self.shift_gears = data.shift_gears

        self.prius_control_messages = {"throttle": self.throttle,
                                       "brake": self.brake,
                                       "steer": self.steer,
                                       "shift_gears": self.shift_gears}

    def draw_prius_control(self, img):
        """
        Draws information from "/prius" topic on the visualization image

        :param img:
        :return:
        """
        text_poses = [(50, 50), (50, 64), (50, 78), (50, 92)]
        # TODO: Check in the future if messages in the same order every time
        for (key, value), text_pose in zip(self.prius_control_messages.items(), text_poses):
            img = cv2.putText(img, text="{} {}".format(key, value), org=text_pose,
                              fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 0, 0), thickness=1)

        return img


if __name__ == "__main__":
    rospy.init_node("visualizer_node")
    visualizer = Visualizer()
    rospy.spin()
