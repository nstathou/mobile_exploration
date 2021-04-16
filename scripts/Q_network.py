#!/usr/bin/env python3

import rospy
import ros_numpy
import numpy as np
from std_srvs.srv import Empty
import keras
import tensorflow as tf
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from keras import models
from keras import layers

height = 120  # height to resize image (quarter of original)
width = 160  # width to resize image (quarter of original)
# depth_image = np.zeros((1, height, width, 1))
# pretrained model for feature extraction
model = models.load_model('feature_maps_model')


def travel(pub, vel):
    twist = Twist()
    twist.linear.x = vel[0]
    twist.angular.z = vel[1]
    print(twist.linear.x, twist.angular.z)
    pub.publish(twist)


def stop(pub):
    twist = Twist()
    twist.linear.x = 0.0
    twist.angular.z = 0.0
    pub.publish(twist)
    print('stopping')


# function to preprocess data before storing into .npy file
def preprocess(data):
    # np.nan_to_num(data, nan=3, copy=False)  # replace nan values with max "depth"
    data = np.resize(data, (height, width))  # resize to save resources
    data = np.reshape(data, (1, height, width, 1))
    return data


def depth_image_callback(msg):
    # global depth_image
    depth_image = preprocess(ros_numpy.numpify(msg))
    return depth_image


def reset():
    rospy.wait_for_service('/gazebo/reset_world')
    reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    reset_world()
    return


def q_network(pub):
    msg = Image()
    # 1. initialize the values
    d_threshold = 0.6  # threshold distance from object to stop
    season = 100  # number of episodes to train on
    max_steps_per_episode = 10000
    q_values = np.random.randint(-50, 1, (season, 3))  # initialize Q-table
    memory = []  # memory to store the replays
    actions = {0: [0.22, 0.4], 1: [0.22, 0.0], 2: [0.22, -0.4]}  # 0: left, 1: forward, 2: right
    epsilon_min = 0.1  # Minimum epsilon greedy parameter
    epsilon_max = 1  # Maximum epsilon greedy parameter
    epsilon_interval = (epsilon_max - epsilon_min)  # Rate at which to reduce chance of random action being taken
    epsilon = 1  # Epsilon greedy parameter
    learning_rate = 0.01  # 0.000001
    gamma = 0.85  # discount_factor
    batch_size = 32
    t = 0  # keep current time
    for episode in range(season):  # 2. Start the episodes
        reset()  # 3. reset to starting position
        depth_image = depth_image_callback(msg)
        dmin = min(depth_image)  # get the minimum distance
        while dmin > d_threshold:  # 4. while depth not less that threshold
            x = model.predict(depth_image)  # 5. capture feature map of depth image
            if np.random.random() < epsilon:  # 6. select action
                action = np.argmax(q_values[t])
            else:
                action = np.random.choice([0, 1, 2])
            # 7. move according to action selected
            travel(pub, actions[action])
            depth_image = depth_image_callback(msg)
            dmin = min(depth_image)
            if dmin < d_threshold:  # 8. if depth smaller than threshold
                reward = -50  # 9. low reward = 'crash'
                x_next = None
            else:  # 10. if depth still okay
                reward = 1  # 11. +1 and keep moving
                x_next = depth_image
            memory.append([x, action, reward, x_next])  # 12. store everything in memory

        else:
            reset()


def main():
    rospy.init_node('Q_Network')
    rospy.init_node('reset_world')
    control_model = models.Sequential()
    inp = layers.Input(shape=(15, 20, 64))
    x = layers.Flatten()(inp)
    x = layers.Dense(units=7)(x)
    x = layers.Dense(units=5)(x)
    outp = layers.Dense(units=3)(x)
    control_model = tf.keras.Model(inputs=inp, outputs=outp)
    rospy.Subscriber('/camera/depth/image_raw', Image, depth_image_callback)
    pub = rospy.Publisher('/cmd_vel', Twist, queue_size=3)  # publish velocity
    r = rospy.Rate(10)  # rate in Hz


if __name__ == '__main__':
    main()
