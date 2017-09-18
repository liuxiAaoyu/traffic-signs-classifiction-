from datetime import datetime
import os.path
import time

import pickle
import math

import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt

from six.moves import xrange 
import random
import _labFunc

batch_size=128
train_dir='/tmp/Lab2_train'
MAXSTEP=200000
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
  start=random.randint(0,len(train_features)-batch_size-1)
  images_seed=train_features[start:start+batch_size]
  labels_seed=train_labels[start:start+batch_size]
  feed_dict={images_p1:images_seed,
             labels_p1:labels_seed,
             }

  return feed_dict



if tf.gfile.Exists(train_dir):
    tf.gfile.DeleteRecursively(train_dir)
tf.gfile.MakeDirs(train_dir)
with tf.Graph().as_default():

    image_width=32
    image_height=32
    color_channels=3

    features=tf.placeholder(tf.float32,shape=[batch_size,image_width,image_height,color_channels])
    labels=tf.placeholder(tf.float32,shape=[batch_size,43])

    logits1 = _labFunc.inference(features,batch_size)

    # Calculate loss.
    loss1 = _labFunc.loss(logits1, labels)

    global_step = tf.Variable(0, trainable=False)
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    opt = _labFunc.train(loss1, global_step)
    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    # Build the summary operation based on the TF collection of Summaries.
    #summary_op = tf.merge_all_summaries()

    init=tf.initialize_all_variables()

    sess=tf.Session()    
    #summary_writer = tf.train.SummaryWriter(train_dir, sess.graph)

    sess.run(init)
    for step in xrange(MAXSTEP):
      feed_dict=fill_feed_dict(features,labels)
      start_time = time.time()
      _, loss_value = sess.run([opt, loss1],feed_dict=feed_dict)
      duration = time.time() - start_time

      assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

      if step % 10 == 0:
        num_examples_per_step = batch_size
        examples_per_sec = num_examples_per_step / duration
        sec_per_batch = float(duration)

        format_str = ('%s: step %d, loss = %.2f (%.1f examples/sec; %.3f '
                      'sec/batch)')
        print (format_str % (datetime.now(), step, loss_value,
                             examples_per_sec, sec_per_batch))

#      if step % 100 == 0:
#        summary_str = sess.run(summary_op,feed_dict=feed_dict)
#        summary_writer.add_summary(summary_str, step)
#
      # Save the model checkpoint periodically.
      if step % 1000 == 0 or (step + 1) == MAXSTEP:
        checkpoint_path = os.path.join(train_dir, 'model.ckpt')
        saver.save(sess, checkpoint_path, global_step=step)

