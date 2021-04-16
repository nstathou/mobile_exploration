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


def main():

    dataset = np.load('depth_data.npy')  # load saved data
    commands = np.load('commands.npy')
    test_dataset = np.load('depth_data_test.npy')
    test_commands = np.load('commands_test.npy')
    valid_dataset = np.load('depth_data_valid.npy')
    valid_commands = np.load('commands_valid.npy')
    (train_data, train_labels) = (dataset, commands)  # train, ~1300 images
    (test_data, test_labels) = (test_dataset, test_commands)  # test ~400 images
    (valid_data, valid_labels) = (valid_dataset, valid_commands)  # validation ~400 images
    del dataset, test_dataset, valid_dataset  # delete to free up memory
    del commands, test_commands, valid_commands  # delete to free up memory
    # Convert the data and the labels into tensors and split them into batches
    train_data = tf.data.Dataset.from_tensor_slices((train_data, train_labels)).batch(batch_size)
    test_data = tf.data.Dataset.from_tensor_slices((test_data, test_labels)).batch(batch_size)
    valid_data = tf.data.Dataset.from_tensor_slices((valid_data, valid_labels)).batch(batch_size)
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
    x = layers.Flatten()(x)  # flatten to connect with dense layer
    outp = layers.Dense(units=num_classes, activation='softmax')(x)  # fully connected with 3 outputs
    model = tf.keras.Model(inputs=inp, outputs=outp)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adam(0.0001),
                  metrics=tf.metrics.categorical_accuracy)
    model.summary()
    model.fit(train_data, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=valid_data)
    score = model.evaluate(test_data, verbose=2)
    results = model.predict(test_data)
    print(score)
    # make the model to extract the feature maps
    # control_model = models.Sequential()
    # inp = layers.Input(shape=(15, 20, 64))
    # x = layers.Flatten()(inp)
    # x = layers.Dense(units=7)(x)
    # x = layers.Dense(units=5)(x)
    # outp = layers.Dense(units=3)(x)
    # control_model = tf.keras.Model(inputs=inp, outputs=outp)
    # control_model.summary()
    feature_maps_model = keras.Model(inputs=model.input,
                                     outputs=model.get_layer('max_pooling2d_2').output)
    model.save('perception_cnn')  # save model
    feature_maps_model.save('feature_maps_model')  # save model


if __name__ == '__main__':
    main()
