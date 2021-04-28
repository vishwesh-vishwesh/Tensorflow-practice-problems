# -*- coding: utf-8 -*-
"""
Created on Sun Nov 22 00:09:16 2020

@author: VISHWESH
"""
import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import numpy as np
import tensorflow.keras as keras


from tensorflow.keras.datasets import mnist

(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

learning_rate = 0.5
epochs = 10
batch_size = 100

# input x - for 28 x 28 pixels = 784
x = tf.placeholder(tf.float32, [None, 784])
# now declare the output data placeholder - 10 digits
y = tf.placeholder(tf.float32, [None, 10])

# now declare the weights connecting the input to the hidden layer
W1 = tf.Variable(tf.random_normal([784, 300], stddev=0.03), name='W1')
b1 = tf.Variable(tf.random_normal([300]), name='b1')

# and the weights connecting the hidden layer to the output layer
W2 = tf.Variable(tf.random_normal([300, 10], stddev=0.03), name='W2')
b2 = tf.Variable(tf.random_normal([10]), name='b2')

#hidden layer outputs
hidden_out = tf.add(tf.matmul(x, W1), b1)
hidden_out = tf.nn.relu(hidden_out)

#final output from output layer
y_ = tf.nn.softmax(tf.add(tf.matmul(hidden_out, W2), b2))

#performing math behind the matrix operations after the y_ (crossentropy cost function)
y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)                 #avoiding log(0)
cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped) + (1 - y) * tf.log(1 - y_clipped), axis=1))

#add an optimizer
optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

init_op = tf.global_variables_initializer()  # variable initialization

#define an accuracy assessment operation
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

def fetch_batch(epoch, batch_index, batch_size):
# load the data from disk
    batch_x = X_train.reshape[:,-1]
    batch_y = Y_train
    return batch_x,batch_y


# start the session
with tf.Session() as sess:
   # initialise the variables
   sess.run(init_op)
   total_batch = int(len(X_train) / batch_size)
   for epoch in range(epochs):
        avg_cost = 0
        for i in range(total_batch):
            batch_x, batch_y = fetch_batch(epoch,i,batch_size)
            _, c = sess.run([optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y})
            avg_cost += c / total_batch
        print("Epoch:", (epoch + 1), "cost =", "{:.3f}".format(avg_cost))
   print(sess.run(accuracy, feed_dict={x: X_test, y: Y_test}))
















