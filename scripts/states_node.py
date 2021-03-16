import rospy

from nav_msgs.msg import Odometry
# from tf.transformations import euler_from_quaternion, quaternion_from_euler

from av_msgs.msg import States


class State:
    def __init__(self):

        self.lisiner = rospy.Subscriber("/base_pose_ground_truth", Odometry, self.callback)
        self.states_pub = rospy.Publisher('/prius/states', States, queue_size=1)

    def callback(self, data):
        print('X: ',data.pose.pose.position.x, 'Y: ',data.pose.pose.position.y, 'Z: ',data.pose.pose.position.z)
        print('Z angle: ',data.twist.twist.angular.z)

        if data is not None:
            # calculate velocity
            #.................
            states_msg = States()
            states_msg.velocity = 0
            states_msg.steer = data.twist.twist.angular.z
            self.states_pub.publish(states_msg)


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


