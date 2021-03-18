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

        self.throttle = 0
        self.brake = 0
        self.steer = 0
        self.gear = Control.NO_COMMAND

        self.control_msg = Control()
        self.mode_msg = Mode()

        keyboard.on_press_key("c", self.key_collect_data, suppress=False)
        keyboard.on_press_key("s", self.key_selfdriving, suppress=False)
        keyboard.on_press_key("0", self.shift_gear_no_command, suppress=False)
        keyboard.on_press_key("1", self.shift_gear_neutral, suppress=False)
        keyboard.on_press_key("2", self.shift_gear_forward, suppress=False)
        keyboard.on_press_key("3", self.shift_gear_reverse, suppress=False)
        keyboard.on_press_key("up", self.speed_up, suppress=False)
        keyboard.on_press_key("down", self.brake, suppress=False)
        keyboard.on_press_key("left", self.turn_left, suppress=False)
        keyboard.on_press_key("right", self.turn_right, suppress=False)

    def key_collect_data(self, e):
        rospy.loginfo(e)
        self.collect_data ^= True
        self.publish()

    def key_selfdriving(self, e):
        rospy.loginfo(e)
        self.self_driving_state ^= True
        self.publish()

    def shift_gear_no_command(self, e):
        rospy.loginfo("Gear: No command")
        self.gear = Control.NO_COMMAND
        self.publish()

    def shift_gear_neutral(self, e):
        rospy.loginfo("Gear: Neutral")
        self.gear = Control.NEUTRAL
        self.publish()

    def shift_gear_forward(self, e):
        rospy.loginfo("Gear: Forwad")
        self.gear = Control.FORWARD
        self.publish()

    def shift_gear_reverse(self, e):
        rospy.loginfo("Gear: Reverse")
        self.gear = Control.REVERSE
        self.publish()

    def speed_up(self, e):
        rospy.loginfo(e)
        self.brake = 0
        self.throttle = 1
        self.publish()

    def brake(self, e):
        rospy.loginfo(e)
        self.throttle = 0
        self.brake = -1
        self.publish()

    def turn_left(self, e):
        rospy.loginfo(e)
        self.steer = 1
        self.publish()

    def turn_right(self, e):
        rospy.loginfo(e)
        self.steer = -1
        self.publish()

    def publish(self, event=None):
        stamp = rospy.Time.now()

        # FILL CONTROL
        self.control_msg.header.stamp = stamp
        self.control_msg.brake = self.brake
        self.control_msg.throttle = self.throttle
        self.control_msg.steer = self.steer
        self.control_msg.shift_gears = self.gear

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
