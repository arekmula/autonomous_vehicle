import rospy

from nav_msgs.msg import Odometry

from av_msgs.msg import States



def callback(data):
    print('X: ',data.pose.pose.position.x, 'Y: ',data.pose.pose.position.y, 'Z: ',data.pose.pose.position.z)
    print('X: ',data.pose.pose.orientation.x, 'Y: ',data.pose.pose.orientation.y, 'Z: ',data.pose.pose.orientation.z)
    # rospy.loginfo(rospy.get_caller_id() + "Test ", data.data)


def lisiner():

    rospy.Subscriber("/base_pose_ground_truth", Odometry, callback)
    rospy.spin()

if __name__ == '__main__':
    rospy.init_node('states_node')
    lisiner()

# States.msg
# Header header
#
# # Vehicle velocity [km/h]
# float64 velocity
#
# # Vehicle steering angle
# float64 steer


