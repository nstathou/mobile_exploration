#!/usr/bin/env python3
import random
from collections import deque
import os
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
import matplotlib.pyplot as plt

height = 120  # height to resize image (quarter of original)
width = 160  # width to resize image (quarter of original)


def reset():
    rospy.wait_for_service('/gazebo/reset_world')
    reset_world = rospy.ServiceProxy('/gazebo/reset_world', Empty)
    reset_world()
    return


def travel(pub, action):
    vel = {0: [0.22, 0.3], 1: [0.22, 0.0], 2: [0.22, -0.3]}
    twist = Twist()
    twist.linear.x = vel[action][0]
    twist.angular.z = vel[action][1]
    pub.publish(twist)


def stop(pub):
    twist = Twist()
    twist.linear.x = 0.0
    twist.angular.z = 0.0
    pub.publish(twist)


# function to preprocess data before storing into .npy file
def preprocess(data):
    np.nan_to_num(data, nan=3, copy=False)  # replace nan values with max "depth"
    data = np.resize(data, (height, width))  # resize to save resources
    data = np.reshape(data, (1, height, width, 1))
    return data


def depth_image_callback(msg):
    global depth_image
    depth_image = preprocess(ros_numpy.numpify(msg))
    return depth_image


class Robot:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=5000)
        self.learning_rate = 0.0001
        self.gamma = 0.85
        if os.path.isfile("exploration_rate.npy"):
            self.exploration_rate = np.load("exploration_rate.npy")
            print("exploration rate: {}".format(self.exploration_rate))
        else:
            self.exploration_rate = 1
        self.exploration_min = 0.01
        self.exploration_decay = 0.995
        self.control_model = self._build_model()

    def _build_model(self):
        # fully connected model
        if os.path.isfile("control_model/saved_model.pb"):
            control_model = models.load_model('control_model')
            print("loading saved model")
        else:
            control_model = models.Sequential()
            inp = layers.Input(shape=(15, 20, 64))
            x = layers.Flatten()(inp)
            x = layers.Dense(units=240, activation='relu')(x)
            x = layers.Dense(units=16, activation='relu')(x)
            outp = layers.Dense(units=3, activation='linear')(x)
            control_model = tf.keras.Model(inputs=inp, outputs=outp)
            control_model.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=self.learning_rate))
        return control_model

    def save_model(self):
        np.save("exploration_rate", self.exploration_rate)
        self.control_model.save("control_model")

    def act(self, state):
        if np.random.rand() <= self.exploration_rate:
            return random.randrange(self.action_size)
        else:
            act_values = self.control_model.predict(state)
            return np.argmax(act_values)

    def remember(self, state, action, reward, next_state):
        self.memory.append((state, action, reward, next_state))

    def replay(self, sample_batch_size):
        if len(self.memory) < sample_batch_size:
            return
        sample_batch = random.sample(self.memory, sample_batch_size)
        loss = []
        for state, action, reward, next_state in sample_batch:
            target = reward + self.gamma * np.argmax(self.control_model(next_state))
            target_f = self.control_model.predict(state)
            target_f[0][action] = target
            print("target_f: ", target_f[0])
            hist = self.control_model.fit(state, target_f, epochs=1, verbose=0)
            loss.append((hist.history['loss'][0]))
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay
        print("exploration rate: ", self.exploration_rate)
        return np.max(loss)


class MobileExploration:
    def __init__(self):
        self.sample_batch_size = 32
        self.episode = 10000
        if os.path.isfile("episode.npy"):
            self.current_episode = np.load("episode.npy")
            print("continuing from episode #{}".format(self.current_episode))
        else:
            self.current_episode = 0
        self.state_size = (15, 20, 64)
        self.action_size = 3
        self.d_threshold = 0.55
        self.perception_model = models.load_model("feature_maps_model")
        if os.path.isfile("loss_history.npy"):
            self.loss = np.ndarray.tolist(np.load("loss_history.npy"))
        else:
            self.loss = []
        self.robot = Robot(self.state_size, self.action_size)
        rospy.init_node("Q_Network")
        rospy.Subscriber("/camera/depth/image_raw", Image, depth_image_callback)
        self.msg = Image()
        self.pub = rospy.Publisher("/cmd_vel", Twist, queue_size=3)  # publish velocity
        self.r = rospy.Rate(10)  # rate in Hz

    def run(self):
        try:
            for index_episode in range(self.current_episode, self.episode):
                self.r.sleep()  # loop
                reset()
                state = self.perception_model.predict(depth_image)
                dmin = np.amin(depth_image)  # get the minimum distance
                index = 0
                reward = 0
                while dmin > self.d_threshold:
                    action = self.robot.act(state)
                    travel(self.pub, action)
                    next_state = self.perception_model.predict(depth_image)
                    dmin = np.amin(depth_image)
                    if dmin <= self.d_threshold:
                        reward = -100
                    else:
                        reward += 0.1
                    self.robot.remember(state, action, reward, next_state)
                    state = next_state
                    index += 1
                    if reward > 55:
                        dmin = self.d_threshold
                print("Episode {}# Score: {}".format(index_episode, index + 1))
                stop(self.pub)
                self.loss.append(self.robot.replay(self.sample_batch_size))
                print("loss: {}".format(self.loss[-1]))
        except KeyboardInterrupt:
            print("exception")
        finally:
            stop(self.pub)
            print("saving")
            self.robot.save_model()
            np.save("episode", self.current_episode)
            np.save("loss_history", self.loss)
            print("exiting")
        return


if __name__ == "__main__":
    explore = MobileExploration()
    explore.run()
