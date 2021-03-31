#!/usr/bin/env python
import rospy
import numpy as np

from nav_msgs.msg import Odometry

from av_msgs.msg import States


class State:
    def __init__(self):

        self.listener = rospy.Subscriber("/base_pose_ground_truth", Odometry, self.callback)
        self.states_pub = rospy.Publisher('/prius/states', States, queue_size=1)

        self.last_published_time = rospy.get_rostime()
        self.last_published = None
        self.timer = rospy.Timer(rospy.Duration(1. / 20.), self.timer_callback)

    def timer_callback(self, event):
        if self.last_published and self.last_published_time < rospy.get_rostime() + rospy.Duration(1.0 / 20.):
            self.callback(self.last_published)

    def callback(self, data):

        if data is not None:
            states_msg = States()

            # calculate velocity
            v = (np.sqrt(data.twist.twist.linear.x ** 2 + data.twist.twist.linear.y ** 2
                         + data.twist.twist.linear.z ** 2))

            # convert to km/h
            v = v * 3.6

            states_msg.velocity = v

            self.states_pub.publish(states_msg)


if __name__ == '__main__':
    rospy.init_node('states_node')
    state = State()
    rospy.spin()

