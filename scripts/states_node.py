import rospy
import numpy as np

from nav_msgs.msg import Odometry
# from tf.transformations import euler_from_quaternion, quaternion_from_euler

from av_msgs.msg import States


class State:
    def __init__(self):

        self.lisiner = rospy.Subscriber("/base_pose_ground_truth", Odometry, self.callback)
        self.states_pub = rospy.Publisher('/prius/states', States, queue_size=1)

        self.x_prev = 0
        self.y_prev = 0
        self.time_prev = 0

    def callback(self, data):
        # print(data.header.stamp)
        # print('X: ',data.pose.pose.position.x, 'Y: ',data.pose.pose.position.y, 'Z: ',data.pose.pose.position.z)
        # print('Z angle: ',data.twist.twist.angular.z)


        if data is not None:
            states_msg = States()

            # get time in seconds
            time = rospy.get_time()

            # calculate velocity
            delta_time = (time - self.time_prev)

            if delta_time > 2:
                pass
                v = (np.sqrt((data.pose.pose.position.x - self.x_prev) ** 2 + (
                            data.pose.pose.position.y - self.y_prev) ** 2)) / delta_time

                # convert to km/h (?)
                v = v * 3.6
                print v
                states_msg.velocity = v
                states_msg.steer = data.twist.twist.angular.z
                self.states_pub.publish(states_msg)


                self.x_prev = data.pose.pose.position.x
                self.y_prev = data.pose.pose.position.y
                self.time_prev = time

if __name__ == '__main__':
    rospy.init_node('states_node')
    state = State()
    rospy.spin()

# States.msg
# Header header
#
# # Vehicle velocity [km/h]
# float64 velocity
#
# # Vehicle steering angle
# float64 steer


