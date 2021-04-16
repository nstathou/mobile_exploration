#!/usr/bin/env python3

import rospy
import numpy as np
import ros_numpy
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
# from sensor_msgs.msg import PointCloud2
# from mpl_toolkits import mplot3d
# import matplotlib.pyplot as plt

depth_data = []  # to hold the depth images
commands = []  # to hold the command vector00
vel = [0, 0, 0, 0, 0, 0]  # to hold the velocity of our robot
height = 120  # height to resize image (quarter of original)
width = 160  # width to resize image (quarter of original)
i = 0  # velocity index for labeling


# function to preprocess data before storing into .npy file
def preprocess(data):
    np.nan_to_num(data, nan=3, copy=False)  # replace nan values with max "depth"
    data = np.resize(data, (height, width))  # resize to save resources
    return data


# callback function to get velocity from /cmd_vel topic
def vel_callback(msg):
    global vel
    vel = [msg.linear.x, msg.linear.y, msg.linear.z,
           msg.angular.x, msg.angular.y, msg.angular.z]


# callback function to get depth image from /camera/depth topic
def depth_image_callback(msg):
    global vel
    if vel[5] < 0 < vel[0]:  # turning right
        commands.append([0, 0, 1])
        depth_data.append(preprocess(ros_numpy.numpify(msg)))
    elif vel[5] > 0 and vel[0] > 0:  # turning left
        commands.append([1, 0, 0])
        depth_data.append(preprocess(ros_numpy.numpify(msg)))
    elif round(vel[5], 1) == 0 and vel[0] > 0:  # moving forward
        commands.append([0, 1, 0])
        depth_data.append(preprocess(ros_numpy.numpify(msg)))
    # print(preprocess(ros_numpy.numpify(msg)))
    print(len(depth_data), len(commands))
    print(commands[-1])


# callback function to get point cloud from /camera/depth topic
# def point_callback(msg):
# pc = ros_numpy.numpify(msg)  # turn data into np array
# height = pc.shape[0]
# width = pc.shape[1]
# points = np.zeros((height * width, 3), dtype=np.float32)  # initialize
# points[:, 0] = np.resize(pc['x'], height*width)  # x points
# points[:, 1] = np.resize(pc['y'], height*width)  # y points
# points[:, 2] = np.resize(pc['z'], height*width)  # z points
# if len(commands) > len(depth_data):
# depth_data.append(pc)
# rospy.loginfo('\n x: {}, y: {}, z: {}'.format(points[:,0],points[:,1],points[:,2]))
# plt.figure()
# ax = plt.axes(projection="3d")
# ax.plot3D(points[:,0],points[:,2],-points[:,1])
# plt.show()


def main():
    rospy.init_node('create_dataset')  # initialize node
    rospy.Subscriber('/cmd_vel', Twist, vel_callback)  # subscribe to /cmd_vel topic
    rospy.Subscriber('/camera/depth/image_raw', Image, depth_image_callback)  # subscribe to /depth/image_raw/topic
    # rospy.Subscriber('/camera/depth/points', PointCloud2, point_callback)  # subscribe to /depth/points topic
    r = rospy.Rate(10)  # rate in Hz
    try:
        while not rospy.is_shutdown():
            r.sleep()  # loop
    except rospy.ROSInterruptException:
        print("exception")
    finally:
        print(len(depth_data) / 2, len(commands) / 2)
        print('saving')
        np.save('depth_data', depth_data[0:-1:2])  # sample the data and save them
        print(commands)
        print('saved data')
        np.save('commands', commands[0:-1:2])  # sample the data and save them
        print('saved commands \n exiting')


if __name__ == '__main__':
    main()
