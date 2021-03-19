#!/usr/bin/env python
# coding: utf-8

# # Fashion MNIST Rest Setup

# ## Initialize System

# In[ ]:

import sys
assert sys.version_info.major == 3, 'Not using Python 3. Use Runtime > Change runtime to change to Python 3.'


# In[ ]:


#Install anything not on ubuntu:latest including TensorFlow and matplotlib

print("Installing dependencies")
get_ipython().system('pip install tensorflow')
get_ipython().system('pip install matplotlib')

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

print('TensorFlow version: {}'.format(tf.__version__))


# ## Create Model

# ### Import Fashion MNIST dataset
# 
# The model will use the [Fashion MNIST](https://github.com/zalandoresearch/fashion-mnist) dataset which contains 70,000 gray scale images that are 28x28 pixels and coordinating labels. The project provides test data and additional custom created test samples were created for verification as well.
# 
# <img src="https://tensorflow.org/images/fashion-mnist-sprite.png" alt="Fashion MNIST dataset" width="550">

# In[ ]:


fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

# scale the values to 0.0 to 1.0
train_images = train_images / 255.0
test_images = test_images / 255.0

# reshape for feeding into the model
train_images = train_images.reshape(train_images.shape[0], 28, 28, 1)
test_images = test_images.reshape(test_images.shape[0], 28, 28, 1)

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

print('\ntrain_images.shape: {}, of {}'.format(train_images.shape, train_images.dtype))
print('test_images.shape: {}, of {}'.format(test_images.shape, test_images.dtype))


# ### Train and Evaluate Model

# In[ ]:


model = keras.Sequential([
  keras.layers.Conv2D(input_shape=(28,28,1), filters=8, kernel_size=3, 
                      strides=2, activation='relu', name='Conv1'),
  keras.layers.Flatten(),
  keras.layers.Dense(10, name='Dense')
])
model.summary()

testing = False
epochs = 10

model.compile(optimizer='adam', 
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=[keras.metrics.SparseCategoricalAccuracy()])
model.fit(train_images, train_labels, epochs=epochs)

test_loss, test_acc = model.evaluate(test_images, test_labels)
print('\nTest accuracy: {}'.format(test_acc))


# ## Save Model

# In[ ]:


# Fetch the Keras session and save the model
# The signature definition is defined by the input and output tensors,
# and stored with the default serving key
import tempfile

MODEL_DIR = tempfile.gettempdir()
version = 1
export_path = os.path.join(MODEL_DIR, str(version))
print('export_path = {}\n'.format(export_path))

tf.keras.models.save_model(
    model,
    export_path,
    overwrite=True,
    include_optimizer=True,
    save_format=None,
    signatures=None,
    options=None
)

print('\nSaved model:')
get_ipython().system('ls -l {export_path}')


# ## Examine Model

# In[ ]:


get_ipython().system('saved_model_cli show --dir {export_path} --all')


# ## Serve Model with TensorFlow Serving

# ### Add TensorFlow Serving as package source

# In[ ]:


#curl might not be available
get_ipython().system('apt install -y curl')

#grab tensorflow serving
get_ipython().system('echo "deb http://storage.googleapis.com/tensorflow-serving-apt stable tensorflow-model-server tensorflow-model-server-universal" | tee /etc/apt/sources.list.d/tensorflow-serving.list && curl https://storage.googleapis.com/tensorflow-serving-apt/tensorflow-serving.release.pub.gpg | apt-key add -')
get_ipython().system(' apt update')


# ### Install TensorFlow Serving

# In[ ]:


get_ipython().system('apt-get install tensorflow-model-server')


# ### Start TensorFlow Serving

# In[ ]:


os.environ["MODEL_DIR"] = MODEL_DIR


# In[ ]:


get_ipython().run_cell_magic('bash', '--bg ', 'nohup tensorflow_model_server \\\n  --rest_api_port=8999 \\\n  --model_name=fashion_model \\\n  --model_base_path="${MODEL_DIR}" >server.log 2>&1\n  \n#!tail server.log')