#!/usr/bin/env python

# ROS imports
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError

# Custom ROS imports
from av_msgs.msg import Mode, States
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
        self.prius_control_topic = "/prius"
        self.prius_control_sub = rospy.Subscriber(self.prius_control_topic, Control, self.prius_control_callback)
        self.throttle = 0.0  # Range 0 to 1, 1 is max throttle
        self.brake = 0.0  # Range 0 to 1, 1 is max brake
        self.steer = 0.0  # Range -1 to +1, +1 is maximum left turn
        self.shift_gears = 0
        self.prius_control_messages = {"throttle": self.throttle,
                                       "brake": self.brake,
                                       "steer": self.steer,
                                       "shift_gears": self.shift_gears}

        # Prius mode
        self.prius_mode_topic = "/prius/mode"
        self.prius_mode_sub = rospy.Subscriber(self.prius_mode_topic, Mode, self.prius_mode_callback)
        self.selfdriving = False
        self.collect = False
        self.prius_mode_messages = {"selfdriving": self.selfdriving,
                                    "collect": self.collect}

        # Prius state
        self.prius_states_topic = "/prius/states"
        self.prius_state_sub = rospy.Subscriber(self.prius_states_topic, States, self.prius_state_callback)
        self.velocity = 0  # km/h
        self.steer_angle = 0.0  # TODO: radians? or degrees?
        self.prius_state_messages = {"velocity": self.velocity, "steer_angle": self.steer_angle}

        # Cv Bridge
        self.cv_bridge = CvBridge()

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
            cv_image = self.draw_prius_status(cv_image)

            visualization_msg = self.cv_bridge.cv2_to_imgmsg(cv_image, "bgr8")
            self.visualization_pub.publish(visualization_msg)

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

    def prius_mode_callback(self, data):
        """
        Callback to "/prius/mode" topic

        :param data: Input Mode message
        :return:
        """
        self.prius_mode_messages = {"selfdriving": data.selfdriving,
                                    "collect": data.collect}

    def prius_state_callback(self, data):
        """
        Callback to "/prius/state" topic

        :param data: Input State message
        :return:
        """
        self.prius_state_messages = {"velocity": int(data.velocity), "steer_angle": data.steer}

    def draw_prius_status(self, img):
        """
        Draws information from "/prius", "/prius/mode" and "/prius/states" topic on the visualization image

        :param img: image to draw on
        :return:
        """

        img_height, img_width, _ = img.shape

        text_pose_base_x = 50

        # Prius topic
        prius_text_poses = [(text_pose_base_x, img_height - 50), (text_pose_base_x, img_height - 64),
                            (text_pose_base_x, img_width - 78), (text_pose_base_x, img_width - 92)]
        # TODO: Check in the future if messages in the same order every time
        for (key, value), text_pose in zip(self.prius_control_messages.items(), prius_text_poses):
            img = cv2.putText(img, text="{} {}".format(key, value), org=text_pose,
                              fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 255, 255), thickness=1)
        # Prius mode topic
        prius_mode_text_poses = [(text_pose_base_x, img_height - 106), (text_pose_base_x, img_height - 120)]
        for (key, value), text_pose in zip(self.prius_mode_messages.items(), prius_mode_text_poses):
            img = cv2.putText(img, text="{} {}".format(key, value), org=text_pose,
                              fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 255, 255), thickness=1)

        # Prius state topic
        prius_state_text_poses = [(text_pose_base_x, img_height - 134), (text_pose_base_x, img_height - 148)]
        for (key, value), text_pose in zip(self.prius_state_messages.items(), prius_state_text_poses):
            img = cv2.putText(img, text="{} {}".format(key, value), org=text_pose,
                              fontFace=cv2.FONT_HERSHEY_PLAIN, fontScale=1, color=(255, 255, 255), thickness=1)

        return img


if __name__ == "__main__":
    rospy.init_node("visualizer_node")
    visualizer = Visualizer()
    rospy.spin()
