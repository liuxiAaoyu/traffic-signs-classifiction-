#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 23 13:18:09 2016

@author: xiaoyu
"""
import pickle
training_file = './lab 2 data/train.p'
testing_file = './lab 2 data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)
    
X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# STOP: Do not change the tests below. Your implementation should pass these tests. 
assert(X_train.shape[0] == y_train.shape[0]), "The number of images is not equal to the number of labels."
assert(X_train.shape[1:] == (32,32,3)), "The dimensions of the images are not 32 x 32 x 3."

import numpy as np

X_train=np.array(X_train,dtype=np.float32)
X_train=(X_train/255)-0.5
assert(round(np.mean(X_train)) == 0), "The mean of the input data is: %f" % np.mean(X_train)
assert(np.min(X_train) == -0.5 and np.max(X_train) == 0.5), "The range of the input data is: %.1f to %.1f" % (np.min(X_train), np.max(X_train))

from keras.models import Sequential
from keras.layers import Activation,Dense

model=Sequential([
    Dense(128,input_shape=(3072,),name="hidden1"),
    Activation('relu'),
    Dense(128),
    Activation('relu'),
    Dense(43),
    Activation('softmax',name='output')
])

# STOP: Do not change the tests below. Your implementation should pass these tests.
assert(model.get_layer(name="hidden1").input_shape == (None, 32*32*3)), "The input shape is: %s" % model.get_layer(name="hidden1").input_shape
assert(model.get_layer(name="output").output_shape == (None, 43)), "The output shape is: %s" % model.get_layer(name="output").output_shape 

from sklearn.preprocessing import LabelBinarizer
encoder = LabelBinarizer()
encoder.fit(y_train)
y_train = encoder.transform(y_train)
y_test = encoder.transform(y_test)

X_train=X_train.ravel()
X_train.shape=(-1,32*32*3)

model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

#history=model.fit(X_train,y_train,nb_epoch=2,batch_size=128)
# STOP: Do not change the tests below. Your implementation should pass these tests.
#assert(history.history['acc'][0] > 0.5), "The training accuracy was: %.3f" % history.history['acc']

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X_train,
    y_train,
    test_size=0.228,
    random_state=832289)

valdata=(X_val,y_val)
history=model.fit(X_train,y_train,nb_epoch=2,batch_size=128,validation_data=valdata)

# STOP: Do not change the tests below. Your implementation should pass these tests.
assert(round((X_train.shape[0]) / float(X_val.shape[0])) == 3), "The training set is %.3f times larger than the validation set." % float(X_train.shape[0]) / float(X_val.shape[0])
assert(history.history['val_acc'][0] > 0.6), "The validation accuracy is: %.3f" % history.history['val_acc'][0]























