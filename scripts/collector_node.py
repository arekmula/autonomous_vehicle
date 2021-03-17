#!/usr/bin/env python

# ROS imports
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image

# Custom ROS imports
from av_msgs.msg import Mode, States

# Python imports
import cv2
import pandas as pd
import os
import time


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

        # Pandas dataframe with labels
        self.df_labels = pd.DataFrame(data={"img_name": [], "velocity": [], "steer_angle": []})

        # Received image count
        self.received_image_count = 0

        # Start time - used to create directory for current run with all labels and images
        self.start_time = str(int(time.time()))
        # The directory will be saved in /root/.ros/ directory
        rospy.loginfo("Creating directory {} for images and labels".format(self.start_time))
        os.mkdir(self.start_time)

    def front_camera_image_callback(self, data):
        """
        Callback to "/prius/front_camera/image_raw" topic. The callback saves current front camera image and it's labels
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

                # Get current image name and format it to 000000x.png
                image_name = "{:06}".format(self.received_image_count) + ".png"
                # Get current labels as dataframe
                current_label = pd.DataFrame(data={"img_name": [image_name],
                                                   "velocity": [self.velocity],
                                                   "steer_angle": [self.steer_angle]})

                # Append current labels dataframe to dataframe with previous labels
                self.df_labels = self.df_labels.append(current_label, ignore_index=True)

                # Save current image
                cv2.imwrite("{}/{}".format(self.start_time, image_name), cv_image)

                # Increment received image count
                self.received_image_count += 1

    def prius_mode_callback(self, data):
        """
        Callback to "/prius/mode" topic
        :param data: Input Mode message
        :return:
        """
        if self.collect != data.collect:
            rospy.loginfo("Changed collecting data status to: ", data.collect)
        self.collect = data.collect

    def prius_state_callback(self, data):
        """
        Callback to "/prius/state" topic
        :param data: Input State message
        :return:
        """
        self.velocity = data.velocity
        self.steer_angle = data.steer_angle

    def save_labels(self):
        """
        Save labels if node shutdown requested
        :return:
        """
        rospy.loginfo("Saving labels on node exit to {}/labels.csv file".format(self.start_time))
        # Set img_name column as index name
        self.df_labels = self.df_labels.set_index("img_name")
        self.df_labels.to_csv("{}/labels.csv".format(self.start_time))


if __name__ == "__main__":
    rospy.init_node("collector_node")
    collector = Collector()
    # Save labels on node shutdown
    rospy.on_shutdown(collector.save_labels)
    rospy.spin()
