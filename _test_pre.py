import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import _labFunc
import csv
import cv2

def normalize(image_data):
    a = 0.1
    b = 0.9
    greyscale_min = 0
    greyscale_max = 1
    return a + ( ( (image_data - greyscale_min)*(b - a) )/( greyscale_max - greyscale_min ) )


batch_size=10

d=dict()
for line in open("signnames.csv"):
     num,name = line.split(",")
     d[num]=name

images=np.zeros((10,32,32,3),dtype=np.float32)
for i in range(10):
    img=plt.imread("./lab 2 data/_"+str(i)+".png")
    img.resize(32,32,3)
    images[i]=img

images=normalize(images)
#from PIL import Image
#im=Image.open('0.png')
#a=im.resize((32,32))
#a.save('_0.png','PNG')



image_width=32
image_height=32
color_channels=3


features=tf.placeholder(tf.float32,shape=[batch_size,image_width,image_height,color_channels])
labels=tf.placeholder(tf.float32,shape=[batch_size,43])

init=tf.initialize_all_variables()


logits = _labFunc.inference(features,batch_size)

top_k_op = tf.nn.top_k(logits, 5)

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
    
    res = session.run(
        [top_k_op],
        feed_dict={features: images})

for i in range(9):
    plt.subplot(331+i)
    plt.bar(left=res[0][1][i],height=res[0][0][i])
    for j in range(len(res[0][1][i])):
        ss=d[str(res[0][1][i][j])][:-1]
        p1=(res[0][1][i][j],res[0][0][i][j])
        p2=(res[0][1][i][j]+1,res[0][0][i][j]+1)
        plt.annotate(ss,xy=p1,xytext=p2,
            #textcoords='offset points',
            bbox=dict(boxstyle="round",fc="0.8"),
            arrowprops=dict(arrowstyle="->",connectionstyle="arc3,rad=.2"),
            )
plt.show()

        
