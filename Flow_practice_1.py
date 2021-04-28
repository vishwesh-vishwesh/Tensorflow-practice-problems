# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 14:27:58 2020

@author: VISHWESH
Tensorflow practice
"""

import tensorflow as tf
import numpy as np
tf.compat.v1.reset_default_graph()
const = tf.constant(2.0, name="const")
rank1_tensor = tf.Variable(["tidy","mess"],tf.string)
rank2_tensor = tf.Variable([["time","space"],["vector","scalar"],["bleh","bluh"]],tf.string)
three = rank2_tensor[:,0]
print(three)
tensor1 = tf.ones([1,2,3])
tensor2 = tf.reshape(tensor1,[3,2,1])
w=print(tensor2)
tensor3 = tf.reshape(tensor2,[3,-1])

t = tf.ones([5,5,5,5])
t = tf.reshape(t,[125,-1])
#print(t)


b = tf.placeholder(tf.float32,[None,1], name="b")
c =tf.Variable(1.0,name="c")

#tensorflow operations 
d = tf.add(b,c,name="d")
e= tf.add(c,const,name="e")
a =tf.multiply(d,e,name="a")
#only run when initialized 
#set up initialization

init_op = tf.global_variables_initializer()

#start the session
with tf.Session() as sess:
    sess.run(init_op)
    a_out = sess.run(a, feed_dict={b: np.arange(0,10)[:,np.newaxis]}) 
    print("Variable a is {}".format(a_out))