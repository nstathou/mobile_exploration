#!/usr/bin/env python3

import numpy as np
import keras
from keras import layers
from keras import models
import tensorflow as tf

height = 120  # height (quarter of original)
width = 160  # width (quarter of original)
num_classes = 3  # 33 commands
batch_size = 32
epochs = 12

dataset = np.load('depth_data_turn.npy')  # load saved data
commands = np.load('commands_turn.npy')

(train_data, train_labels) = (dataset, commands)  # train
(test_data, test_labels) = (dataset[0:-1:4],
                            commands[0:-1:4])  # 25% test datasets
del dataset  # delete to free up memory
del commands  # delete to free up memory

# Convert the data and the labels into tensors and split them into batches
train_data = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(batch_size)
test_data = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(batch_size)


# Build perception model
model = models.Sequential()
inp = layers.Input(shape=(height, width, 1))  # input layer
x = layers.Conv2D(filters=32, padding='same', kernel_size=(5, 5))(inp)  # first convolutional layer 32 nodes
x = layers.Activation('relu')(x)  # relu activation
x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)  # 2x2 pooling
x = layers.Conv2D(filters=32, padding='same', kernel_size=(5, 5))(x)  # second convolutional layer 32 nodes
x = layers.Activation('relu')(x)  # relu activation
x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)  # 2x2 pooling
x = layers.Conv2D(filters=64, padding='same', kernel_size=(5, 5))(x)  # third convolutional layer 64 nodes
x = layers.Activation('relu')(x)  # relu activation
x = layers.MaxPooling2D(pool_size=(2, 2), strides=2)(x)  # 2x2 pooling
x = layers.Flatten()(x)  # flatten to connect with dense layer
outp = layers.Dense(units=num_classes, activation='softmax')(x)  # fully connected with 5 outputs
model = tf.keras.Model(inputs=inp, outputs=outp)
model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(0.0001),
              metrics=tf.metrics.categorical_accuracy)
model.summary()
model.fit(train_data, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=test_data)
model.save('perception_cnn')  # save model
score = model.evaluate(test_data, verbose=2)
# results = model.predict(test_data)
