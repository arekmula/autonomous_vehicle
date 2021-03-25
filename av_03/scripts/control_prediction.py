#!/usr/bin/env python

# ROS imports
import rospy
import rospkg

# Python imports
import tensorflow as tf


class Predictor:
    VEHICLE_MAX_SPEED = 130  # Maximum speed of a car in km/h

    rospack = rospkg.RosPack()
    PATH_TO_ROS_PACKAGE = rospack.get_path("av_03")  # Get path of current ros package

    def __init__(self, model_path=None):

        self.model_path = self.PATH_TO_ROS_PACKAGE + "/cnn_models/model.h5"
        if model_path is not None:
            self.model_path = model_path

        # Load model
        model = tf.keras.models.load_model(self.model_path)

        # Print model summary
        print(model.summary())


if __name__ == '__main__':
    rospy.init_node("control_prediction")
    predictor = Predictor()
    rospy.spin()
