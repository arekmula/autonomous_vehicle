#!/usr/bin/env python
import rospy
import numpy as np

from nav_msgs.msg import Odometry

from av_msgs.msg import States


class State:
    def __init__(self):

        self.listener = rospy.Subscriber("/base_pose_ground_truth", Odometry, self.callback)
        self.states_pub = rospy.Publisher('/prius/states', States, queue_size=1)

        self.x_prev = 0
        self.y_prev = 0
        self.time_prev = 0
        self.delay_time = 0.1

    def callback(self, data):

        if data is not None:
            states_msg = States()

            # get time in seconds
            time = rospy.get_time()

            # calculate dt
            delta_time = (time - self.time_prev)

            if delta_time > self.delay_time:
                # calculate velocity
                v = (np.sqrt(data.twist.twist.linear.x ** 2 + data.twist.twist.linear.y ** 2
                             + data.twist.twist.linear.z ** 2))

                # convert to km/h
                v = v * 3.6

                states_msg.velocity = v

                self.states_pub.publish(states_msg)

                self.x_prev = data.pose.pose.position.x
                self.y_prev = data.pose.pose.position.y
                self.time_prev = time

if __name__ == '__main__':
    rospy.init_node('states_node')
    state = State()
    rospy.spin()

