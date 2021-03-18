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

        self.throttle = 0.0
        self.var_brake = 0.0
        self.steer = 0.0
        self.gear = Control.NO_COMMAND
        
        self.throttle_change_increment = 0.15
        self.brake_change_increment = 0.05
        self.steer_change_increment = 0.05

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

    def key_selfdriving(self, e):
        rospy.loginfo(e)
        self.self_driving_state ^= True

    def shift_gear_no_command(self, e):
        rospy.loginfo("Gear: No command")
        self.gear = Control.NO_COMMAND

    def shift_gear_neutral(self, e):
        rospy.loginfo("Gear: Neutral")
        self.gear = Control.NEUTRAL

    def shift_gear_forward(self, e):
        rospy.loginfo("Gear: Forwad")
        self.gear = Control.FORWARD

    def shift_gear_reverse(self, e):
        rospy.loginfo("Gear: Reverse")
        self.gear = Control.REVERSE

    def speed_up(self, e):
        rospy.loginfo(e)
        self.var_brake = 0.0
        if self.throttle >= 1.:
            self.throttle = 1.0
        else:
            self.throttle += self.throttle_change_increment

    def brake(self, e):
        rospy.loginfo(e)
        self.throttle = 0.0
        if self.var_brake >= 1:
            self.var_brake = 1.0
        else:
            self.var_brake += self.brake_change_increment

    def turn_left(self, e):
        rospy.loginfo(e)
        if self.steer >= 1.:
            self.steer = 1.0
        else:
            self.steer += self.steer_change_increment

    def turn_right(self, e):
        rospy.loginfo(e)
        if self.steer <= -1.:
            self.steer = -1.0
        else:
            self.steer -= self.steer_change_increment

    def publish_control(self, event=None):
        stamp = rospy.Time.now()

        # FILL CONTROL
        control_msg = Control()
        control_msg.header.stamp = stamp
        control_msg.brake = self.var_brake
        control_msg.throttle = self.throttle
        control_msg.steer = self.steer
        control_msg.shift_gears = self.gear

        # FILL MODE
        mode_msg = Mode()
        mode_msg.header.stamp = stamp
        mode_msg.selfdriving = self.self_driving_state
        mode_msg.collect = self.collect_data

        # PUBLISH
        self.control_pub.publish(control_msg)
        self.mode_pub.publish(mode_msg)
        

if __name__ == '__main__':
    rospy.init_node("controller_node")
    k = Keyboard()
    rospy.Timer(rospy.Duration(1.0 / 30.0), k.publish_control)
    rospy.spin()
