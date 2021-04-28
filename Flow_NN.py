# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 17:39:02 2020

@author: VISHWESH
"""
import tensorflow as tf
from tensorflow import keras
import datetime

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt
tf.keras.backend.clear_session()

import shutil

shutil.rmtree('log_dir', ignore_errors=True)

fashion_mnist = keras.datasets.fashion_mnist  # load dataset

(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  
# split into tetsing and training
'''
train_images.shape
train_images[0,23,23]  # let's have a look at one pixel
train_labels[:10]  # let's have a look at the first 10 training labels
'''

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

plt.figure()
plt.imshow(train_images[1])
plt.colorbar()
plt.grid(False)
plt.show()

train_images = train_images / 255.0

test_images = test_images / 255.0

log2 = "logs/fit/" + datetime.datetime.now().strftime("%m%d-%H%M")
tb_callback = keras.callbacks.TensorBoard(log_dir=log2,histogram_freq=1)

img = np.reshape(train_images[0], (-1, 28, 28, 1))
file_writer = tf.summary.create_file_writer(log2)
# Using the file writer, log the reshaped image.
with file_writer.as_default():
  tf.summary.image("Training data", img, step=0)


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1)
    keras.layers.Dense(256,activation='relu'),
    keras.layers.Dense(128, activation='relu'),#hidden layer (2)
    keras.layers.Dense(64, activation='relu'),  # hidden layer(3) 
    keras.layers.Dense(10, activation='softmax') # output layer (4)
])

opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=20,callbacks=[tb_callback],validation_split=0.2)  # we pass the data, labels and epochs and watch the magic!
#%%%
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 
model.summary()
print('Test accuracy:', test_acc)

predictions = model.predict(test_images)
predictions[0]

np.argmax(predictions[0])
test_labels[0]

#%%
COLOR = 'white'
plt.rcParams['text.color'] = COLOR
plt.rcParams['axes.labelcolor'] = COLOR

def predict(model, image, correct_label):
  class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
  prediction = model.predict(np.array([image]))
  predicted_class = class_names[np.argmax(prediction)]

  show_image(image, class_names[correct_label], predicted_class)


def show_image(img, label, guess):
  plt.figure()
  plt.imshow(img, cmap=plt.cm.binary)
  plt.title("Excpected: " + label)
  plt.xlabel("Guess: " + guess)
  plt.colorbar()
  plt.grid(False)
  plt.show()


def get_number():
  while True:
    num = input("Pick a number: ")
    if num.isdigit():
      num = int(num)
      if 0 <= num <= 1000:
        return int(num)
    else:
      print("Try again...")

num = get_number()
image = test_images[num]
label = test_labels[num]
predict(model, image, label)

#%%















