from datetime import datetime
import pickle
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import matplotlib.pyplot as plt
import _labFunc

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

k_output1=64
image_width=32
image_height=32
color_channels=3
filter_size_width=5
filter_size_height=5
batch_size = 128

features=tf.placeholder(tf.float32,shape=[batch_size,image_width,image_height,color_channels])
labels=tf.placeholder(tf.float32,shape=[batch_size,43])



train_feed_dict = {features: train_features, labels: train_labels}
valid_feed_dict = {features: valid_features, labels: valid_labels}
test_feed_dict = {features: test_features, labels: test_labels}



init=tf.initialize_all_variables()

epochs = 5
learning_rate = 0.5

# Measurements use for graphing loss and accuracy
log_batch_step = 50
batches = []
loss_batch = []
train_acc_batch = []
valid_acc_batch = []

logits = _labFunc.inference(features,batch_size)

top_k_op = tf.nn.in_top_k(logits, tf.argmax(labels,1), 1)

saver = tf.train.Saver()

with tf.Session() as session:
    ckpt = tf.train.get_checkpoint_state('/tmp/Lab2_train')
    if ckpt and ckpt.model_checkpoint_path:
      # Restores from checkpoint
      saver.restore(session, ckpt.model_checkpoint_path)
      # Assuming model_checkpoint_path looks something like:
      #   /my-favorite-path/cifar10_train/model.ckpt-0,
      # extract global_step from it.
      global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
    else:
      print('No checkpoint file found')
    
    session.run(init)
    #batch_count = int(math.ceil(len(train_features)/batch_size))
    batch_count = int(len(valid_features)/batch_size)

    # Progress bar
    batches_pbar = tqdm(range(batch_count), desc='Epoch {:>2}/{}'.format(1, 1), unit='batches')
    
    true_count = 0  # Counts the number of correct predictions.
    total_sample_count = batch_count * batch_size
    step = 0
    # The training cycle
    for batch_i in batches_pbar:
        # Get a batch of training features and labels
        batch_start = batch_i*batch_size
        batch_features = test_features[batch_start:batch_start + batch_size]
        batch_labels = test_labels[batch_start:batch_start + batch_size]

        predictions = session.run(
            [top_k_op],
            feed_dict={features: batch_features, labels: batch_labels})
        true_count += np.sum(predictions)
        step += 1

      # Compute precision @ 1.
    precision = true_count / total_sample_count
    print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))

        
