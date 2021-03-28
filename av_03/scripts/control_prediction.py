#!/usr/bin/env python

# ROS imports
import rospy
import rospkg

from prius_msgs.msg import Control
from av_msgs.msg import Mode, States
from sensor_msgs.msg import Image

# Python imports
import tensorflow as tf
import cv2
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
from simple_pid import PID


class Predictor:
    VEHICLE_MAX_SPEED = 130  # Maximum speed of a car in km

    rospack = rospkg.RosPack()
    PATH_TO_ROS_PACKAGE = rospack.get_path("av_03")  # Get path of current ros package


    def __init__(self, model_path=None):

        # Subscribers
        self.mode_sub = rospy.Subscriber("/pruis/mode", Mode, self.mode_callback)
        self.front_img_sub = rospy.Subscriber("/prius/front_camera/image_raw", Image, self.front_camera_callback, queue_size=1)
        self.states_sub = rospy.Subscriber("/prius/states", States, self.states_callback, queue_size=1)

        # Publishers
        self.control_prediction_pub = rospy.Publisher("/prius", Control, queue_size=1)

        # PID
        self.pid = PID(1, 0.0, 0.0, setpoint=1)
        self.pid.output_limits = (-1, 1)

        self.cv_bridge = CvBridge()

        self.model_output = 0.0
        self.angle_prediction = 0.0
        self.velocity_prediction = 0.0
        self.velocity_measure = 0.0

        # selfdriving mode
        self.selfdrive = False

        self.model = None
        self.session = tf.compat.v1.keras.backend.get_session()
        self.model_path = self.PATH_TO_ROS_PACKAGE + "/cnn_models/model.h5"
        if model_path is not None:
            self.model_path = model_path

        # Load model
        try:
            self.model = tf.keras.models.load_model(self.model_path)
            # Print model summary
            print(self.model.summary())
        except IOError as e:
            # Handle the exception if model wasn't read correctly
            print(e)

    def states_callback(self, data):
        # save measured velocity
        self.velocity_measure = data.velocity

    def mode_callback(self, data):
        # set selfdriving mode
        if data.selfdriving == True:
            self.selfdrive = True
        if data.selfdriving == False:
            self.selfdrive = False

    def front_camera_callback(self, data):

        if data is not None:
            try:
                cv_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)


        img = cv2.resize(cv_image, (200, 200)).astype(np.float32)/255

        with self.session.graph.as_default():
            tf.compat.v1.keras.backend.set_session(self.session)

            self.model_output = self.model.predict(img[np.newaxis, :])

            self.angle_prediction = self.model_output[0][0]
            self.velocity_prediction = self.model_output[0][1]

            print("angle: ", self.angle_prediction, "velocity: ", self.velocity_prediction)

        stamp = rospy.Time.now()

        # calculate target velocity
        target_velocity = self.velocity_prediction * self.VEHICLE_MAX_SPEED
        # pid
        self.pid.setpoint = target_velocity
        pid_output = self.pid(self.velocity_measure)
        print("velocity measure:", self.velocity_measure, "target velocity:", target_velocity, "pid output:", pid_output)


        control_msg = Control()
        control_msg.header.stamp = stamp
        # angle
        control_msg.steer = self.angle_prediction
        # velocity
        if pid_output >= 0:
            control_msg.throttle = pid_output
            control_msg.brake = 0.0
        if pid_output < 0:
            control_msg.brake = pid_output * (-1)
            control_msg.throttle = 0.0

        control_msg.shift_gears = Control.NO_COMMAND

        if self.selfdrive == True:
            self.control_prediction_pub.publish(control_msg)


if __name__ == '__main__':
    rospy.init_node("control_prediction")
    predictor = Predictor()
    rospy.spin()

# rostopic pub --once /pruis/mode av_msgs/Mode "{header:{seq: 0, stamp:{secs: 0, nsecs: 0}, frame_id: ''}, selfdriving: true, collect: false}"
# rostopic pub --r 10 /pruis/mode av_msgs/Mode "{header:{seq: 0, stamp:{secs: 0, nsecs: 0}, frame_id: ''}, selfdriving: true, collect: false}"

