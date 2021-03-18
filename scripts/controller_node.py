#!/usr/bin/env python

import rospy
import keyboard
import time
from prius_msgs.msg import Control
from av_msgs.msg import Mode


class Keyboard:

    def __init__(self):

        time.sleep(5)
        self.control_pub = rospy.Publisher("/prius", Control, queue_size=1)
        self.mode_pub = rospy.Publisher("/prius/mode", Mode, queue_size=1)

        self.collect_data = False
        self.self_driving_state = False

        self.throttle = 0.
        self.brake = 0.
        self.steer = 0.
        self.gear = Control.NO_COMMAND

        self.control_msg = Control()
        self.mode_msg = Mode()

        keyboard.on_press_key("c", self.key_collect_data, suppress=False)
        keyboard.on_press_key("s", self.key_selfdriving, suppress=False)
        keyboard.on_press_key("0", self.shift_gear, suppress=False)
        keyboard.on_press_key("1", self.shift_gear, suppress=False)
        keyboard.on_press_key("2", self.shift_gear, suppress=False)
        keyboard.on_press_key("3", self.shift_gear, suppress=False)
        keyboard.on_press_key("up", self.speed_up, suppress=False)
        keyboard.on_press_key("down", self.brake, suppress=False)
        keyboard.on_press_key("left", self.turn_left, suppress=False)
        keyboard.on_press_key("right", self.turn_right, suppress=False)

    def key_collect_data(self, e):
        self.collect_data ^= True
        self.publish()

    def key_selfdriving(self, e):
        self.self_driving_state ^= True
        self.publish()

    def shift_gear(self, e):
        if e == "0":
            self.gear = Control.NO_COMMAND
        elif e == "1":
            self.gear = Control.NEUTRAL
        elif e == "2":
            self.gear = Control.FORWARD
        elif e == "3":
            self.gear = Control.REVERSE
        self.publish()

    def speed_up(self, e):
        self.brake = 0.
        if self.throttle >= 1.:
            self.throttle = 1.
        else:
            self.throttle += 0.1
        self.publish()

    def brake(self, e):
        self.throttle = 0.
        if self.brake >= 1.:
            self.brake = 1.
        else:
            self.brake += 0.1
        self.publish()

    def turn_left(self, e):
        if self.steer <= -1.:
            self.steer = -1.
        else:
            self.steer -= 0.1
        self.publish()

    def turn_right(self, e):
        if self.steer >= 1.:
            self.steer = 1.
        else:
            self.steer += 0.1
        self.publish()

    def publish(self, event=None):
        stamp = rospy.Time.now()

        # FILL CONTROL
        self.control_msg.header.stamp = stamp
        self.control_msg.brake = self.brake
        self.control_msg.throttle = self.throttle
        self.control_msg.steer = self.steer

        # FILL MODE
        self.mode_msg.header.stamp = stamp
        self.mode_msg.selfdriving = self.self_driving_state
        self.mode_msg.collect = self.collect_data

        # PUBLISH
        self.control_pub.publish(self.control_msg)
        self.mode_pub.publish(self.mode_msg)


if __name__ == '__main__':
    rospy.init_node('controler_node')
    keyboard = Keyboard()
    rospy.spin()
