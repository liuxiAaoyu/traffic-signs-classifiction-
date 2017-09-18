
import tensorflow as tf




def inference(images,batch_size):
  with tf.variable_scope('conv1') as scope:
    weight1=tf.Variable(tf.truncated_normal([5,5,3,64],stddev=0.05,dtype=tf.float32,name='weights'))
    bias1=tf.Variable(tf.zeros(64),dtype=tf.float32,name='bias')
    #keep_prob=tf.placeholder(tf.float32)

    conv_layer1=tf.nn.conv2d(images,weight1,strides=[1,1,1,1],padding='SAME')
    conv_layer1=tf.nn.bias_add(conv_layer1,bias1)
    conv_layer1=tf.nn.relu(conv_layer1)
    #conv_layer1=tf.nn.dropout(conv_layer1,keep_prob)
    
  pool1=tf.nn.max_pool(conv_layer1,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool1')
  
  with tf.variable_scope('conv2') as scope:
    weight2=tf.Variable(tf.truncated_normal([5,5,64,64],stddev=0.05,dtype=tf.float32,name='weights'))
    bias2=tf.Variable(tf.zeros(64),dtype=tf.float32,name='bias')
    conv_layer2=tf.nn.conv2d(pool1,weight2,strides=[1,1,1,1],padding='SAME')
    conv_layer2=tf.nn.bias_add(conv_layer2,bias2)
    conv_layer2=tf.nn.relu(conv_layer2)

  pool2=tf.nn.max_pool(conv_layer2,ksize=[1,3,3,1],strides=[1,2,2,1],padding='SAME',name='pool2')
  
  lrn2 = tf.nn.lrn(pool2, 4, bias=1.0, alpha=0.001 / 9.0, beta=0.75,name='lrn2')

  with tf.variable_scope('fullconnect1') as scope:
    reshape=tf.reshape(lrn2,[batch_size,-1])
    dim=reshape.get_shape()[1].value
    weight3=tf.Variable(tf.truncated_normal([dim,384],stddev=0.04,dtype=tf.float32,name='weights'))
    bias3=tf.Variable(tf.zeros(384),dtype=tf.float32,name='bias')
    fullconnect1=tf.add(tf.matmul(reshape,weight3),bias3)
    fullconnect1=tf.nn.relu(fullconnect1)

  with tf.variable_scope('fullconnect2') as scope:
    weight4=tf.Variable(tf.truncated_normal([384,192],stddev=0.04,dtype=tf.float32,name='weights'))
    bias4=tf.Variable(tf.zeros(192),dtype=tf.float32,name='bias')
    fullconnect2=tf.add(tf.matmul(fullconnect1,weight4),bias4)
    fullconnect2=tf.nn.relu(fullconnect2)
  
  with tf.variable_scope('logits') as scope:
    weight5=tf.Variable(tf.truncated_normal([192,43],stddev=1/192,dtype=tf.float32,name='weights'))
    bias5=tf.Variable(tf.zeros(43),dtype=tf.float32,name='bias')
    logits=tf.add(tf.matmul(fullconnect2,weight5),bias5)

  return logits

def loss(logits,labels):
  labels = tf.cast(labels, tf.int32)
  cross_entropy=tf.nn.softmax_cross_entropy_with_logits(logits,labels, name='cross_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  return cross_entropy_mean

def train(loss,global_step):
  lr = tf.train.exponential_decay(0.1,
                                  global_step,
                                  10000,
                                  0.1,
                                  staircase=True)
  opt = tf.train.GradientDescentOptimizer(lr)
  opt=opt.minimize(loss)
  return opt