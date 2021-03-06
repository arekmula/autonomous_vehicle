#!/usr/bin/env python

# ROS imports
import rospy
import rospkg
from cv_bridge import CvBridge, CvBridgeError

from prius_msgs.msg import Control
from av_msgs.msg import Mode, States
from sensor_msgs.msg import Image

# Python imports
import tensorflow as tf
import numpy as np
from simple_pid import PID


class Predictor:
    VEHICLE_MAX_SPEED = 130  # Maximum speed of a car in km
    Y_CROP_OFFSET = 288
    Y_CROP_HEIGHT = 264

    rospack = rospkg.RosPack()
    PATH_TO_ROS_PACKAGE = rospack.get_path("av_03")  # Get path of current ros package

    def __init__(self, model_path=None):
        # Subscribers
        self.mode_sub = rospy.Subscriber("/prius/mode", Mode, self.mode_callback)
        self.front_img_sub = rospy.Subscriber("/prius/front_camera/image_raw", Image, self.front_camera_callback,
                                              queue_size=1)
        self.states_sub = rospy.Subscriber("/prius/states", States, self.states_callback, queue_size=1)

        # Publishers
        self.control_prediction_pub = rospy.Publisher("/prius", Control, queue_size=1)

        # PID velocity
        self.pid_velocity = PID(0.5, 0.0, 0.0, setpoint=1)
        self.pid_velocity.output_limits = (-1, 1)

        # PID steering
        self.pid_steering = PID(5, 0.0, 0.0, setpoint=0)
        self.pid_steering.output_limits = (-1, 1)

        self.cv_bridge = CvBridge()

        self.angle_prediction = 0.0
        self.angle_measure = 0.0

        self.velocity_prediction = 0.0
        self.velocity_measure = 0.0

        self.selfdrive_mode = False

        self.model = None
        self.session = tf.compat.v1.keras.backend.get_session()
        self.model_path = self.PATH_TO_ROS_PACKAGE + "/cnn_models/model.hdf5"
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
        self.angle_measure = data.steer

    def mode_callback(self, data):
        # set selfdriving mode
        self.selfdrive_mode = data.selfdriving

    def front_camera_callback(self, data):
        if data is not None:
            try:
                cv_image = self.cv_bridge.imgmsg_to_cv2(data, "bgr8")
            except CvBridgeError as e:
                print(e)

        self.predict_control(cv_image)
        self.calcucate_control()

    def predict_control(self, cv_image):
        img = cv_image[self.Y_CROP_OFFSET:self.Y_CROP_OFFSET + self.Y_CROP_HEIGHT, :]
        img = img.astype(np.float32) / 255

        with self.session.graph.as_default():
            tf.compat.v1.keras.backend.set_session(self.session)

            model_output = self.model.predict(img[np.newaxis, :])

            self.angle_prediction = model_output[0][0]
            self.velocity_prediction = model_output[0][1]

            print("angle prediction {:.2f}, velocity_prediction {:.2f}".format(self.angle_prediction,
                                                                               self.velocity_prediction))

    def calcucate_control(self):
        stamp = rospy.Time.now()

        # calculate target velocity
        target_velocity = self.velocity_prediction * self.VEHICLE_MAX_SPEED
        # pid
        self.pid_velocity.setpoint = target_velocity
        pid_velocity_output = self.pid_velocity(self.velocity_measure)
        print("velocity measure: {:.2f}, target velocity: {:.2f}, velocity pid output: {:.2f}".format(
            self.velocity_measure,
            target_velocity,
            pid_velocity_output))

        self.pid_steering.setpoint = self.angle_prediction
        pid_steering_output = self.pid_steering(self.angle_measure)
        print("angle measure: {:.2f}, target angle: {:.2f}, angle pid output: {:.2f}".format(
            self.angle_measure,
            self.pid_steering.setpoint,
            pid_steering_output))

        control_msg = Control()
        control_msg.header.stamp = stamp
        # angle
        control_msg.steer = pid_steering_output
        # velocity
        if pid_velocity_output >= 0:
            control_msg.throttle = pid_velocity_output
            control_msg.brake = 0.0
        if pid_velocity_output < 0:
            control_msg.brake = pid_velocity_output * (-1)
            control_msg.throttle = 0.0

        control_msg.shift_gears = Control.NO_COMMAND

        if self.selfdrive_mode:
            self.control_prediction_pub.publish(control_msg)


if __name__ == '__main__':
    rospy.init_node("control_prediction")
    predictor = Predictor()
    rospy.spin()
