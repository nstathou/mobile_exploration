#!/usr/bin/env python3

import rospy
import ros_numpy
import numpy as np
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from keras import models

height = 120  # height to resize image (quarter of original)
width = 160  # width to resize image (quarter of original)
depth_image = np.zeros((1, height, width, 1))
model = models.load_model('perception_cnn')


# function to preprocess data before storing into .npy file
def preprocess(data):
    np.nan_to_num(data, nan=3, copy=False)  # replace nan values with max "depth"
    data = np.resize(data, (height, width))  # resize to save resources
    data = np.reshape(data, (1, height, width, 1))
    return data


def depth_image_callback(msg):
    global depth_image
    depth_image = preprocess(ros_numpy.numpify(msg))


class Move:

    def __init__(self):
        self.vel = [0.4, 0, -0.4]
        self.pub = rospy.Publisher('/cmd_vel', Twist, queue_size=3)
        self.rate = rospy.Rate(10)  # 10Hz

    def travel(self):
        # global depth_image
        twist = Twist()
        coeff = np.round(model.predict(depth_image), 3)
        twist.linear.x = round((0.2 * coeff[0][1] + 0.1 * coeff[0][0] + 0.1 * coeff[0][2]), 2)
        twist.angular.z = np.sum(np.multiply(coeff, self.vel))
        print(twist.linear.x, twist.angular.z)
        self.pub.publish(twist)
        print(coeff)

    def stop(self):
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.pub.publish(twist)
        print('stopping')


def main():
    rospy.init_node('autopilot')
    rospy.Subscriber('/camera/depth/image_raw', Image, depth_image_callback)  # subscribe to /depth/image_raw/topic
    rospy.Publisher('/cmd_vel', Twist, queue_size=25)  # publish velocity
    move = Move()
    r = rospy.Rate(10)  # rate in Hz
    try:
        while not rospy.is_shutdown():
            r.sleep()  # loop
            move.travel()
    except rospy.ROSInterruptException:
        print("exception")
    finally:
        move.stop()
        print('exiting')


if __name__ == '__main__':
    main()
