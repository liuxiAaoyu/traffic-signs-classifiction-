import pickle
import math
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
from six.moves import xrange
import random

import cifar10

# Reload the data
pickle_file = 'data.pickle'
with open(pickle_file, 'rb') as f:
  pickle_data = pickle.load(f)
  train_features = pickle_data['train_dataset']
  train_labels = pickle_data['train_labels']
  valid_features = pickle_data['valid_dataset']
  valid_labels = pickle_data['valid_labels']
  test_features = pickle_data['test_dataset']
  test_labels = pickle_data['test_labels']
  del pickle_data  # Free up memory
print('Data and modules loaded.')

def fill_feed_dict(images_p1,labels_p1):
  start=random.randint(0,111745)
  if start+batch_size>111745:
      images_seed=np.zeros((batch_size,32,32,3),dtype=np.float32)
      labels_seed=np.zeros((batch_size,43),dtype=np.float32)
      n=0
      for i in xrange (batch_size):
          if start+i<111745:
              images_seed[i]=train_features[start+i]
              labels_seed[i]=train_labels[start+i]
          else:
              images_seed[i]=train_features[n]
              labels_seed[i]=train_labels[n]
              n=n+1
  else:
      images_seed=train_features[start:start+batch_size]
      labels_seed=train_labels[start:start+batch_size]
  feed_dict={images_p1:images_seed,
             labels_p1:labels_seed,
             }

  return feed_dict


k_output1=64
image_width=32
image_height=32
color_channels=3
filter_size_width=5
filter_size_height=5
batch_size = 128

features=tf.placeholder(tf.float32,shape=[batch_size,image_width,image_height,color_channels])
labels=tf.placeholder(tf.float32,shape=[batch_size,43])


#input=tf.placeholder(tf.float32,shape=[128,image_width,image_height,color_channels])
weight1=tf.Variable(tf.truncated_normal([5,5,3,64]))
bias1=tf.Variable(tf.zeros(64))
keep_prob=tf.placeholder(tf.float32)

conv_layer1=tf.nn.conv2d(features,weight1,strides=[1,2,2,1],padding='SAME')
conv_layer1=tf.nn.bias_add(conv_layer1,bias1)
conv_layer1=tf.nn.relu(conv_layer1)
conv_layer1=tf.nn.dropout(conv_layer1,keep_prob)
conv_layer1=tf.nn.max_pool(conv_layer1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')

weight2=tf.Variable(tf.truncated_normal([5,5,64,64]))
bias2=tf.Variable(tf.zeros(64))
conv_layer2=tf.nn.conv2d(conv_layer1,weight2,strides=[1,1,1,1],padding='SAME')
conv_layer2=tf.nn.bias_add(conv_layer2,bias2)
conv_layer2=tf.nn.relu(conv_layer2)
conv_layer2=tf.nn.max_pool(conv_layer2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME')
conv_layer2 = tf.nn.lrn(conv_layer2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75)

reshape=tf.reshape(conv_layer2,[batch_size,-1])
dim=reshape.get_shape()[1].value
weight3=tf.Variable(tf.truncated_normal([dim,384]))
bias3=tf.Variable(tf.zeros(384))
fullconnect1=tf.add(tf.matmul(reshape,weight3),bias3)
fullconnect1=tf.nn.relu(fullconnect1)

weight4=tf.Variable(tf.truncated_normal([384,192]))
bias4=tf.Variable(tf.zeros(192))
fullconnect2=tf.add(tf.matmul(fullconnect1,weight4),bias4)
fullconnect2=tf.nn.relu(fullconnect2)

weight5=tf.Variable(tf.truncated_normal([192,43]))
bias5=tf.Variable(tf.zeros(43))
fullconnect3=tf.add(tf.matmul(fullconnect2,weight5),bias5)
fullconnect3=tf.nn.relu(fullconnect3)

logits=cifar10.inference(features)
softmax_layer=tf.nn.softmax(logits)

cross_entropy=-tf.reduce_sum(labels*tf.log(softmax_layer), reduction_indices=1)

loss=tf.reduce_mean(cross_entropy)


#labels = tf.cast(labels, tf.int64)
#cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
#      logits, labels)
#loss = tf.reduce_mean(cross_entropy)

train_feed_dict = {features: train_features, labels: train_labels}
valid_feed_dict = {features: valid_features, labels: valid_labels}
test_feed_dict = {features: test_features, labels: test_labels}

epochs = 5
learning_rate = 0.01



### DON'T MODIFY ANYTHING BELOW ###
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)    

# The accuracy measured against the validation set
validation_accuracy = 0.0

init=tf.initialize_all_variables()
with tf.Session() as session:
  session.run(init)
  #batch_count = int(math.ceil(len(train_features)/batch_size))
  batch_count = int(len(train_features)/batch_size)
  for step in xrange(1000000):
    feed_dict=fill_feed_dict(features,labels)
    start_time = time.time()
    _, loss_value = session.run([optimizer, loss],feed_dict=feed_dict)
    duration = time.time() - start_time

    #assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

    if step % 10 == 0:
      num_examples_per_step = batch_size
      examples_per_sec = num_examples_per_step / duration
      sec_per_batch = float(duration)
      format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                    'sec/batch)')
      print (format_str % (datetime.now(), step, loss_value,
                           examples_per_sec, sec_per_batch))
